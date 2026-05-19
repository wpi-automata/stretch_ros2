#!/usr/bin/env python3
"""Node 4: Scene Graph Builder and GNN Ranker.

Accumulates ALL Detic detections (not just drawers) during exploration,
builds a scene graph via VoxelGraphBuilder, and runs the ContextGNN to
rank containers by how likely they are to hold the target object.

After scoring, pushes ranked container positions to Node 2 via
``/detection/set_rankings`` so the drawer selection can use GNN scores.

Camera data comes from the same ROS2 topics (DDS or rosbridge) as Node 2.
Detection uses the full Detic 21K vocabulary to capture scene context
(landmarks, furniture, appliances) that the GNN needs.

Services provided:
  - /scene_graph/build_and_rank (Trigger)  run GNN, push rankings to Node 2
  - /scene_graph/get_rankings   (Trigger)  return current rankings as JSON
  - /scene_graph/score_now      (Trigger)  force re-scoring mid-exploration

Requires:
  - GPU for CLIP + Detic (runs on automata-3 alongside Node 2)
  - GNN checkpoint from semantic-object-container-room/checkpoints/
"""

import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
import roslibpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image as RosImage
from std_msgs.msg import ColorRGBA, String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, TransformStamped, Vector3, Quaternion
from builtin_interfaces.msg import Duration as RosDuration
from tf2_msgs.msg import TFMessage
import tf2_ros

_SEMANTIC_ROOT = Path(__file__).resolve().parent.parent.parent / "semantic-object-container-room"
if str(_SEMANTIC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SEMANTIC_ROOT))

MAX_PROJECTION_DEPTH = 5.0
MIN_PROJECTION_DEPTH = 0.1


def _bbox_iou(a, b):
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _cross_class_nms(detections, iou_threshold=0.3):
    """Suppress overlapping bboxes across all classes, keeping higher confidence."""
    if not detections:
        return detections
    sorted_dets = sorted(detections, key=lambda d: d.score, reverse=True)
    keep = []
    for det in sorted_dets:
        suppressed = False
        for kept in keep:
            if _bbox_iou(det.bbox, kept.bbox) > iou_threshold:
                suppressed = True
                break
        if not suppressed:
            keep.append(det)
    return keep


class SceneGraphNode(Node):
    """Builds a scene graph from Detic detections and runs GNN ranking."""

    def __init__(self):
        super().__init__("scene_graph_node")

        self.declare_parameter("room_type", "kitchen")
        self.declare_parameter("query", "fork")
        self.declare_parameter("checkpoint", "")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("edge_cutoff", 0.9)
        self.declare_parameter("det_min_score", 0.7)
        self.declare_parameter("use_sim", False)
        self.declare_parameter("robot_ip", "")
        self.declare_parameter("robot_port", 9090)
        self.declare_parameter("remote_rgb_topic", "/camera/color/image_raw/compressed")
        self.declare_parameter("remote_depth_topic",
                               "/camera/aligned_depth_to_color/image_raw/compressedDepth")
        self.declare_parameter("remote_camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("throttle_rate_ms", 2000)

        self.room_type = self.get_parameter("room_type").value
        self.query = self.get_parameter("query").value
        self.checkpoint = self.get_parameter("checkpoint").value
        self.device = self.get_parameter("device").value
        self.edge_cutoff = self.get_parameter("edge_cutoff").value
        self.det_min_score = self.get_parameter("det_min_score").value
        self.use_sim = self.get_parameter("use_sim").value

        self.cb_group = ReentrantCallbackGroup()
        self.bridge = CvBridge()

        # Camera state
        self.latest_rgb = None
        self.latest_depth = None
        self.camera_K = None
        self._rgb_stamp = 0.0
        self._depth_stamp = 0.0
        self._rgb_ros_stamp = None
        self._depth_ros_stamp = None
        self._detecting = False

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=30))
        self._tf_ready = False
        self._has_tf = False
        self._has_tf_static = False

        # Lazy-loaded models
        self._models_loaded = False
        self._detector = None
        self._clip_model = None
        self._clip_preprocess = None
        self._builder = None
        self._gnn_model = None
        self._proto_matrix = None
        self._gnn_cfg = None
        self._query_embeddings = None
        self._text_embeddings = None

        # Rankings
        self._rankings = []
        self._rankings_lock = threading.Lock()
        self._n_observations = 0

        # Detection tracking for visualization
        self._detected_objects = []
        self._detected_objects_lock = threading.Lock()

        # Marker publishers
        self._marker_pub = self.create_publisher(
            MarkerArray, "/scene_graph/markers", 10
        )

        # Image transport: rosbridge WebSocket (real) or DDS (sim)
        self._robot_ip = self.get_parameter("robot_ip").value
        if self._robot_ip:
            self._setup_rosbridge_images()
        else:
            self._setup_dds_images()

        # Subscribe to exploration status
        # DDS always (works in sim); rosbridge added in _setup_rosbridge_images
        self.create_subscription(
            String, "/exploration_status",
            self._exploration_status_callback, 10,
        )

        # Service client to push rankings to Node 2
        from stretch_drawer_pipeline.srv import SetRankings
        self._set_rankings_srv_type = SetRankings
        self.set_rankings_client = self.create_client(
            SetRankings, "/detection/set_rankings", callback_group=self.cb_group
        )

        # Services
        # Services that don't need models
        self.create_service(
            Trigger, "/scene_graph/get_rankings",
            self._get_rankings_callback,
            callback_group=self.cb_group,
        )
        self.create_service(
            Trigger, "/scene_graph/score_now",
            self._score_now_callback,
            callback_group=self.cb_group,
        )

        # Defer model loading + service registration so the executor is
        # already spinning — TF callbacks can fill the buffer while
        # Detic/CLIP/GNN load on another executor thread.
        self._deferred_init_timer = self.create_timer(
            0.1, self._deferred_model_load, callback_group=self.cb_group,
        )

        self.get_logger().info(
            f"Scene graph node starting (model loading deferred): "
            f"room={self.room_type}, query={self.query}, device={self.device}"
        )

    def _deferred_model_load(self):
        """One-shot: load Detic/CLIP/GNN, then advertise services."""
        self._deferred_init_timer.cancel()
        self.get_logger().info("Loading models (Detic, CLIP, GNN)...")
        self._ensure_models_loaded()
        self.get_logger().info(f"All models loaded on {self.device}")

        self.create_service(
            Trigger, "/scene_graph/process_frame",
            self._process_frame_callback,
            callback_group=self.cb_group,
        )
        self.create_service(
            Trigger, "/scene_graph/build_and_rank",
            self._build_and_rank_callback,
            callback_group=self.cb_group,
        )
        if hasattr(self, "_ros_client"):
            self._ws_process_frame_service = roslibpy.Service(
                self._ros_client, "/scene_graph/process_frame", "std_srvs/srv/Trigger"
            )
            self._ws_process_frame_service.advertise(self._rosbridge_process_frame_handler)
            self.get_logger().info("Advertised /scene_graph/process_frame via rosbridge")

        self.get_logger().info("Scene graph node ready")

    # ── TF readiness ────────────────────────────────────────────────

    def _on_tf_msg(self, msg):
        if not self._has_tf:
            self._has_tf = True
            self._check_tf_ready()

    def _on_tf_static_msg(self, msg):
        if not self._has_tf_static:
            self._has_tf_static = True
            self._check_tf_ready()

    def _check_tf_ready(self):
        if self._has_tf and self._has_tf_static and not self._tf_ready:
            self._tf_ready = True
            self.get_logger().info("TF ready")

    # ── Image transport ──────────────────────────────────────────────

    def _setup_dds_images(self):
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        sensor_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        tf_static_qos = QoSProfile(
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(
            TFMessage, "/tf", self._on_tf_msg, 10
        )
        self.create_subscription(
            TFMessage, "/tf_static", self._on_tf_static_msg, tf_static_qos
        )

        self.create_subscription(
            RosImage, "/camera/color/image_raw",
            self._rgb_dds_callback, sensor_qos,
        )
        self.create_subscription(
            RosImage, "/camera/depth/image_rect_raw",
            self._depth_dds_callback, sensor_qos,
        )
        self.create_subscription(
            CameraInfo, "/camera/color/camera_info",
            self._camera_info_dds_callback, sensor_qos,
        )
        self.get_logger().info("Image transport: DDS")

    def _setup_rosbridge_images(self):
        import roslibpy
        robot_port = self.get_parameter("robot_port").value
        throttle_ms = self.get_parameter("throttle_rate_ms").value
        remote_rgb = self.get_parameter("remote_rgb_topic").value
        remote_depth = self.get_parameter("remote_depth_topic").value
        remote_cam_info = self.get_parameter("remote_camera_info_topic").value

        self.get_logger().info(
            f"Image transport: rosbridge at {self._robot_ip}:{robot_port}"
        )

        self._ros_client = roslibpy.Ros(host=self._robot_ip, port=robot_port)

        self._rgb_topic = roslibpy.Topic(
            self._ros_client, remote_rgb,
            "sensor_msgs/msg/CompressedImage",
            throttle_rate=throttle_ms,
        )
        self._depth_topic = roslibpy.Topic(
            self._ros_client, remote_depth,
            "sensor_msgs/msg/CompressedImage",
            throttle_rate=throttle_ms,
        )
        self._camera_info_topic = roslibpy.Topic(
            self._ros_client, remote_cam_info,
            "sensor_msgs/msg/CameraInfo",
        )

        self._exploration_status_topic = roslibpy.Topic(
            self._ros_client, "/exploration_status",
            "std_msgs/msg/String",
        )

        self._ws_tf_topic = roslibpy.Topic(
            self._ros_client, "/tf", "tf2_msgs/msg/TFMessage",
        )
        self._ws_tf_static_topic = roslibpy.Topic(
            self._ros_client, "/tf_static_volatile", "tf2_msgs/msg/TFMessage",
        )

        self._tf_pub = self.create_publisher(TFMessage, "/tf", 100)
        self._tf_static_pub = self.create_publisher(
            TFMessage, "/tf_static",
            rclpy.qos.QoSProfile(
                depth=100,
                durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )

        self._rgb_topic.subscribe(self._rgb_ws_callback)
        self._depth_topic.subscribe(self._depth_ws_callback)
        self._camera_info_topic.subscribe(self._camera_info_ws_callback)
        self._exploration_status_topic.subscribe(self._exploration_status_ws_callback)
        self._ws_tf_topic.subscribe(self._rosbridge_tf_callback)
        self._ws_tf_static_topic.subscribe(self._rosbridge_tf_static_callback)

        self._ros_client_thread = threading.Thread(
            target=self._ros_client.run, daemon=True
        )
        self._ros_client_thread.start()

        # /scene_graph/process_frame rosbridge advertisement is deferred
        # to after model loading — see end of __init__

    # ── DDS image callbacks ──────────────────────────────────────────

    def _rgb_dds_callback(self, msg: RosImage):
        import cv2
        try:
            arr = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            if not self.use_sim:
                arr = cv2.rotate(arr, cv2.ROTATE_90_CLOCKWISE)
            self.latest_rgb = arr
            self._rgb_stamp = time.monotonic()
            self._rgb_ros_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().warn(f"RGB decode error: {e}", throttle_duration_sec=5.0)

    def _depth_dds_callback(self, msg: RosImage):
        import cv2
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            if not self.use_sim:
                depth = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)
            self.latest_depth = depth.astype(np.float32)
            if self.latest_depth.max() > 100:
                self.latest_depth /= 1000.0
            self._depth_ros_stamp = msg.header.stamp
            self._depth_stamp = time.monotonic()
        except Exception as e:
            self.get_logger().warn(f"Depth decode error: {e}", throttle_duration_sec=5.0)

    def _camera_info_dds_callback(self, msg: CameraInfo):
        if self.camera_K is not None:
            return
        K = np.array(msg.k).reshape(3, 3)
        if K[0, 0] > 0:
            if not self.use_sim:
                H = 720
                fx, fy = K[1, 1], K[0, 0]
                cx = (H - 1) - K[1, 2]
                cy = K[0, 2]
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            self.camera_K = K
            self.get_logger().info(f"Camera K set from camera_info: fx={K[0,0]:.1f}")

    # ── Rosbridge image callbacks ────────────────────────────────────

    def _rgb_ws_callback(self, msg):
        import base64
        import cv2
        try:
            raw = base64.b64decode(msg["data"])
            arr = np.frombuffer(raw, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is not None:
                arr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if not self.use_sim:
                    arr = cv2.rotate(arr, cv2.ROTATE_90_CLOCKWISE)
                self.latest_rgb = arr
                self._rgb_stamp = time.monotonic()
                stamp = msg.get("header", {}).get("stamp", {})
                self._rgb_ros_stamp = rclpy.time.Time(
                    seconds=stamp.get("sec", 0),
                    nanoseconds=stamp.get("nanosec", 0),
                ).to_msg()
        except Exception as e:
            self.get_logger().warn(f"WS RGB error: {e}", throttle_duration_sec=5.0)

    def _depth_ws_callback(self, msg):
        import base64
        import cv2
        try:
            raw = base64.b64decode(msg["data"])
            arr = np.frombuffer(raw, dtype=np.uint8)
            # compressedDepth has a 12-byte header before the PNG data
            depth = cv2.imdecode(arr[12:], cv2.IMREAD_UNCHANGED)
            if depth is None:
                depth = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if depth is None:
                return
            if not self.use_sim:
                depth = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)
            self.latest_depth = depth.astype(np.float32)
            if self.latest_depth.max() > 100:
                self.latest_depth /= 1000.0
            stamp = msg.get("header", {}).get("stamp", {})
            self._depth_ros_stamp = rclpy.time.Time(
                seconds=stamp.get("sec", 0),
                nanoseconds=stamp.get("nanosec", 0),
            ).to_msg()
            self._depth_stamp = time.monotonic()
        except Exception as e:
            self.get_logger().warn(f"WS depth error: {e}", throttle_duration_sec=5.0)

    def _camera_info_ws_callback(self, msg):
        if self.camera_K is not None:
            return
        try:
            K = np.array(msg["k"]).reshape(3, 3)
            if K[0, 0] > 0:
                if not self.use_sim:
                    fx, fy = K[1, 1], K[0, 0]
                    cx = 719 - K[1, 2]
                    cy = K[0, 2]
                    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                self.camera_K = K
                self.get_logger().info(f"Camera K (WS): fx={K[0,0]:.1f}")
        except Exception:
            pass

    # ── Rosbridge TF callbacks ──────────────────────────────────────

    def _rosbridge_tf_callback(self, msg_dict):
        self._republish_tf(msg_dict, self._tf_pub)
        if not self._has_tf:
            self._has_tf = True
            self._check_tf_ready()

    def _rosbridge_tf_static_callback(self, msg_dict):
        self._republish_tf(msg_dict, self._tf_static_pub, static=True)
        if not self._has_tf_static:
            self._has_tf_static = True
            self._check_tf_ready()
        self._ws_tf_static_topic.unsubscribe()
        self.get_logger().info("Received and republished static TFs via rosbridge")

    def _republish_tf(self, msg_dict, publisher, static=False):
        try:
            tf_msg = TFMessage()
            for t in msg_dict.get("transforms", []):
                ts = TransformStamped()
                h = t.get("header", {})
                stamp = h.get("stamp", {})
                ts.header.stamp.sec = stamp.get("sec", 0)
                ts.header.stamp.nanosec = stamp.get("nanosec", 0)
                ts.header.frame_id = h.get("frame_id", "")
                ts.child_frame_id = t.get("child_frame_id", "")
                tr = t.get("transform", {})
                tl = tr.get("translation", {})
                rot = tr.get("rotation", {})
                ts.transform.translation = Vector3(
                    x=tl.get("x", 0.0),
                    y=tl.get("y", 0.0),
                    z=tl.get("z", 0.0),
                )
                ts.transform.rotation = Quaternion(
                    x=rot.get("x", 0.0),
                    y=rot.get("y", 0.0),
                    z=rot.get("z", 0.0),
                    w=rot.get("w", 1.0),
                )
                tf_msg.transforms.append(ts)
                if static:
                    self.tf_buffer.set_transform_static(ts, "rosbridge")
                else:
                    self.tf_buffer.set_transform(ts, "rosbridge")
        except Exception as e:
            self.get_logger().warn(f"TF decode failed: {e}", throttle_duration_sec=5.0)
        try:
            publisher.publish(tf_msg)
        except Exception as e:
            self.get_logger().warn(f"TF republish failed: {e}", throttle_duration_sec=10.0)

    # ── Exploration status ───────────────────────────────────────────

    def _exploration_status_callback(self, msg: String):
        self._handle_exploration_status(msg.data)

    def _exploration_status_ws_callback(self, msg):
        self._handle_exploration_status(msg.get("data", ""))

    def _handle_exploration_status(self, status: str):
        if status == "complete":
            self._detecting = False
            if not getattr(self, "_exploration_complete_handled", False):
                self._exploration_complete_handled = True
                self.get_logger().info("Exploration complete — auto-triggering GNN scoring")
                threading.Thread(target=self._auto_build_and_rank, daemon=True).start()
        elif status != "paused_for_detection":
            self._detecting = False

    def _auto_build_and_rank(self):
        """Auto-triggered when exploration completes. Runs GNN and pushes rankings."""
        ranking = self._run_scoring()
        if ranking:
            self._push_rankings_to_node2(ranking)

    # ── Lazy model loading ───────────────────────────────────────────

    def _ensure_models_loaded(self):
        if self._models_loaded:
            return True

        try:
            import torch
            import clip as clip_module
            from realrobot.detector import VisualDetector, nms_by_type
            from realrobot.voxel_graph_builder import VoxelGraphBuilder
            from realrobot.inference import load_locked_model

            device = self.device
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
                self.get_logger().warn("CUDA not available — falling back to CPU")

            self.get_logger().info("Loading perception models…")

            checkpoint = self.checkpoint or None
            self._gnn_model, self._proto_matrix, self._gnn_cfg = load_locked_model(
                checkpoint, "cpu"
            )
            ec = self._gnn_cfg.get("edge_cutoff", self.edge_cutoff)
            self.get_logger().info(
                f"GNN loaded: K={self._gnn_cfg.get('K')}, "
                f"alpha={self._gnn_cfg.get('alpha')}, ec={ec}"
            )

            self._detector = VisualDetector(
                device=device, score_threshold=self.det_min_score
            )
            self._detector._load_model()
            self._clip_model, self._clip_preprocess = clip_module.load(
                "ViT-B/32", device=device
            )

            from gnn.config import CLIP_DIM, QUERIES
            self._clip_dim = CLIP_DIM

            query_tokens = clip_module.tokenize(
                [f"a photo of a {q}" for q in QUERIES]
            ).to(device)
            with torch.no_grad():
                qe = self._clip_model.encode_text(query_tokens)
                qe = qe / qe.norm(dim=-1, keepdim=True)
            self._query_embeddings = {
                q: qe[i].float() for i, q in enumerate(QUERIES)
            }

            room_tokens = clip_module.tokenize(
                [f"a {self.room_type.replace('_', ' ')}"]
            ).to(device)
            with torch.no_grad():
                feat = self._clip_model.encode_text(room_tokens).squeeze(0)
                feat = feat / feat.norm()
            self._text_embeddings = {
                f"room_type:{self.room_type}": feat.cpu()
            }

            self._builder = VoxelGraphBuilder(
                scene_id=f"stretch_{self.room_type}",
                room_type=self.room_type,
                text_embeddings=self._text_embeddings,
                device=device,
                edge_cutoff=ec,
            )

            self._device = device
            self._models_loaded = True
            self.get_logger().info("All models loaded")
            return True

        except Exception as e:
            self.get_logger().error(f"Failed to load models: {e}")
            return False

    # ── Per-frame detection (runs in background thread) ──────────────

    def _process_current_frame(self):
        """Process the current camera frame: Detic → CLIP → 3D project → builder."""
        try:
            self._process_current_frame_inner()
        finally:
            self._detecting = False

    def _process_current_frame_inner(self):
        import torch
        from PIL import Image

        if not self._ensure_models_loaded():
            return

        # Wait for fresh images received AFTER the pause signal
        pause_t = getattr(self, '_pause_stamp', 0.0)
        stamp_slop = 0.05
        for _ in range(50):
            rgb_fresh = self.latest_rgb is not None and self._rgb_stamp > pause_t
            depth_fresh = self.latest_depth is not None and self._depth_stamp > pause_t
            if rgb_fresh and depth_fresh:
                rs = self._rgb_ros_stamp
                ds = self._depth_ros_stamp
                if rs is not None and ds is not None:
                    rgb_t = rs.sec + rs.nanosec / 1e9
                    depth_t = ds.sec + ds.nanosec / 1e9
                    if abs(rgb_t - depth_t) <= stamp_slop:
                        break
                    self.get_logger().debug(
                        f"RGB/depth stamp gap {abs(rgb_t - depth_t):.3f}s — waiting for matching pair",
                        throttle_duration_sec=2.0,
                    )
                else:
                    break
            time.sleep(0.1)

        if self.latest_rgb is None or self.latest_depth is None:
            self.get_logger().warn(
                f"No fresh camera data (rgb={'yes' if self.latest_rgb is not None else 'NO'}, "
                f"depth={'yes' if self.latest_depth is not None else 'NO'}) — skipping"
            )
            return

        rgb_stamp = self._rgb_ros_stamp
        camera_frame = "camera_color_optical_frame"
        try:
            stamp = rclpy.time.Time.from_msg(rgb_stamp) if rgb_stamp else rclpy.time.Time()
            transform = self.tf_buffer.lookup_transform(
                "odom", camera_frame,
                stamp,
                timeout=rclpy.duration.Duration(seconds=2.0),
            )
        except Exception as e:
            self.get_logger().error(
                f"TF lookup FAILED for stamp={stamp} — skipping frame: {e}"
            )
            return

        camera_pose = self._transform_to_matrix(transform)
        t = transform.transform.translation
        tf_stamp = transform.header.stamp
        now = self.get_clock().now()
        tf_age = now.nanoseconds / 1e9 - (tf_stamp.sec + tf_stamp.nanosec / 1e9)
        self.get_logger().info(
            f"Camera pose in odom: ({t.x:.3f}, {t.y:.3f}, {t.z:.3f})"
            f"  tf_stamp={tf_stamp.sec}.{tf_stamp.nanosec:09d}  age={tf_age:.1f}s"
        )

        if tf_age > 3.0:
            self.get_logger().warn(
                f"TF too stale ({tf_age:.1f}s) — skipping frame"
            )
            return

        if not self._tf_ready:
            self.get_logger().warn("TF not yet warmed up — skipping frame")
            return

        if self.camera_K is None:
            self.get_logger().debug("No camera_K yet")
            return

        rgb = self.latest_rgb.copy()
        depth = self.latest_depth.copy()
        camera_K = self.camera_K.copy()

        from realrobot.detector import nms_by_type
        from realrobot.stretch.projection import project_bbox_to_world_se3

        dets = self._detector.detect(rgb, return_crops=True)
        dets = [d for d in dets if d.score >= self.det_min_score]
        n_raw = len(dets)
        dets = nms_by_type(dets)
        n_nms1 = len(dets)
        dets = _cross_class_nms(dets, iou_threshold=0.3)
        self.get_logger().info(
            f"Detic: {n_raw} raw → {n_nms1} after per-type NMS → {len(dets)} after cross-class NMS")
        for d in dets:
            self.get_logger().info(
                f"  kept: {d.object_type} bbox={d.bbox} score={d.score:.2f}")

        if dets:
            self._ensure_type_text_embeddings(
                [d.object_type for d in dets]
            )

        n_added = 0
        scene_frame_emb = None

        for det in dets:
            world_pos = project_bbox_to_world_se3(
                det.bbox, depth, camera_pose, camera_K,
                max_depth=MAX_PROJECTION_DEPTH,
                image_rotated_cw90=not self.use_sim,
            )
            if world_pos is None:
                continue

            self.get_logger().info(
                f"  {det.object_type} @ ({world_pos[0]:.3f}, {world_pos[1]:.3f}, {world_pos[2]:.3f})"
                f"  bbox={det.bbox}  score={det.score:.2f}"
            )

            clip_emb = None
            if det.crop is not None:
                img_t = self._clip_preprocess(det.crop).unsqueeze(0).to(self._device)
                with torch.no_grad():
                    clip_emb = self._clip_model.encode_image(img_t).squeeze(0)
                    clip_emb = clip_emb / clip_emb.norm()
                clip_emb = clip_emb.cpu()

            if scene_frame_emb is None:
                ft = self._clip_preprocess(
                    Image.fromarray(rgb)
                ).unsqueeze(0).to(self._device)
                with torch.no_grad():
                    scene_frame_emb = self._clip_model.encode_image(ft).squeeze(0)
                    scene_frame_emb = scene_frame_emb / scene_frame_emb.norm()
                scene_frame_emb = scene_frame_emb.cpu()

            x0, y0, x1, y1 = det.bbox
            area = (x1 - x0) * (y1 - y0)

            n_before = self._builder.n_nodes
            self._builder.add_observation(
                obj_type=det.object_type,
                position_3d=world_pos,
                clip_embedding=clip_emb if clip_emb is not None else torch.zeros(self._clip_dim),
                scene_frame_clip=scene_frame_emb,
                crop_area=area,
                score=det.score,
            )
            n_after = self._builder.n_nodes
            if n_after > n_before:
                self.get_logger().info(
                    f"    NEW node for {det.object_type} (total {n_after})")
            else:
                self.get_logger().info(
                    f"    MERGED {det.object_type} into existing node")
            n_added += 1

            with self._detected_objects_lock:
                self._detected_objects.append({
                    "type": det.object_type,
                    "pos": world_pos.tolist() if hasattr(world_pos, "tolist") else list(world_pos),
                    "score": det.score,
                })

        self._n_observations += 1
        self.get_logger().info(
            f"Frame {self._n_observations}: {len(dets)} dets, {n_added} added, "
            f"{self._builder.n_raw_detections} total raw, "
            f"{self._builder.n_nodes} nodes"
        )
        self._publish_markers()

    def _ensure_type_text_embeddings(self, obj_types):
        import torch
        import clip as clip_module

        new_types = [
            t for t in obj_types
            if f"container_type:{t}" not in self._text_embeddings
        ]
        if not new_types:
            return
        tokens = clip_module.tokenize(
            [f"a photo of a {t}" for t in new_types]
        ).to(self._device)
        with torch.no_grad():
            embs = self._clip_model.encode_text(tokens)
            embs = embs / embs.norm(dim=-1, keepdim=True)
        for i, t in enumerate(new_types):
            self._text_embeddings[f"container_type:{t}"] = embs[i].float().cpu()

    # ── GNN scoring ──────────────────────────────────────────────────

    def _run_scoring(self) -> list:
        """Build scene graph and run GNN. Returns ranking list."""
        import torch
        from realrobot.inference import score_containers, compute_soft_ldist

        if not self._ensure_models_loaded():
            return []

        if self._builder.n_nodes == 0:
            self.get_logger().warn("No nodes in scene graph — cannot score")
            return []

        self.get_logger().info(
            f"Running GNN scoring: {self._builder.n_nodes} nodes, "
            f"{self._builder.n_raw_detections} raw detections"
        )

        self._gnn_model.to(self._device)
        self._proto_matrix = self._proto_matrix.to(self._device)

        graph = self._builder.get_graph()
        ranking = score_containers(
            self._gnn_model, graph, self._query_embeddings,
            self.query, self._proto_matrix,
        )

        self._gnn_model.to("cpu")
        self._proto_matrix = self._proto_matrix.to("cpu")

        with self._rankings_lock:
            self._rankings = ranking

        if ranking:
            top = ranking[0]
            self.get_logger().info(
                f"Top ranked: {top.get('container_type')} "
                f"(score={top.get('score', 0):.3f})"
            )

        self._publish_markers()
        return ranking

    def _push_rankings_to_node2(self, ranking: list):
        """Call /detection/set_rankings on Node 2."""
        if not self.set_rankings_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn(
                "/detection/set_rankings not available — rankings not pushed"
            )
            return

        request = self._set_rankings_srv_type.Request()
        request.rankings_json = json.dumps(ranking)
        future = self.set_rankings_client.call_async(request)

        timeout = time.time() + 10.0
        while not future.done() and time.time() < timeout:
            time.sleep(0.1)

        if future.done():
            try:
                result = future.result()
                self.get_logger().info(
                    f"Rankings pushed to Node 2: {result.message}"
                )
            except Exception as e:
                self.get_logger().error(f"set_rankings call failed: {e}")
        else:
            self.get_logger().warn("set_rankings call timed out")

    # ── Service callbacks ────────────────────────────────────────────

    def _process_frame_callback(self, request, response):
        """Process one frame: Detic + CLIP + 3D projection into scene graph."""
        if not self._tf_ready:
            response.success = False
            response.message = "TF not ready"
            return response
        self._pause_stamp = time.monotonic()
        self.latest_rgb = None
        self.latest_depth = None
        self._rgb_ros_stamp = None
        self._depth_ros_stamp = None
        self._detecting = True
        try:
            self._process_current_frame_inner()
            response.success = True
            response.message = (
                f"{self._builder.n_nodes} nodes, "
                f"{self._n_observations} observations"
            )
        except Exception as e:
            self.get_logger().error(f"process_frame failed: {e}")
            response.success = False
            response.message = str(e)
        finally:
            self._detecting = False
        return response

    def _build_and_rank_callback(self, request, response):
        """Build scene graph, run GNN, push rankings to Node 2."""
        ranking = self._run_scoring()
        if ranking:
            self._push_rankings_to_node2(ranking)
            response.success = True
            response.message = (
                f"Scored {len(ranking)} containers, "
                f"top={ranking[0].get('container_type')} "
                f"({ranking[0].get('score', 0):.3f})"
            )
        else:
            response.success = False
            response.message = "No containers to score"
        return response

    def _get_rankings_callback(self, request, response):
        with self._rankings_lock:
            response.success = len(self._rankings) > 0
            response.message = json.dumps(self._rankings)
        return response

    def _score_now_callback(self, request, response):
        """Force re-scoring (same as build_and_rank)."""
        return self._build_and_rank_callback(request, response)

    # ── Rosbridge service handlers ──────────────────────────────────

    def _rosbridge_process_frame_handler(self, request, response):
        """Handle /scene_graph/process_frame called via rosbridge."""
        def _do_process():
            if not self._tf_ready:
                response(roslibpy.ServiceResponse({
                    "success": False,
                    "message": "TF not ready",
                }))
                return
            self._pause_stamp = time.monotonic()
            self.latest_rgb = None
            self.latest_depth = None
            self._rgb_ros_stamp = None
            self._depth_ros_stamp = None
            self._detecting = True
            try:
                self._process_current_frame_inner()
                response(roslibpy.ServiceResponse({
                    "success": True,
                    "message": (
                        f"{self._builder.n_nodes} nodes, "
                        f"{self._n_observations} observations"
                    ),
                }))
            except Exception as e:
                self.get_logger().error(f"Rosbridge process_frame failed: {e}")
                response(roslibpy.ServiceResponse({
                    "success": False,
                    "message": str(e),
                }))
            finally:
                self._detecting = False
        threading.Thread(target=_do_process, daemon=True).start()

    # ── Helpers ───────────────────────────────────────────────────────

    # ── RViz markers ──────────────────────────────────────────────────

    def _publish_markers(self):
        """Publish the GNN scene graph as RViz markers.

        Visualises the actual graph that feeds the GNN:
          - Container nodes (merged ClusteredNodes from the builder)
          - Room node at the centroid of all containers
          - room↔container edges as lines
          - Ranked containers (after GNN scoring) as a second layer
        """
        now_stamp = self.get_clock().now().to_msg()
        ma = MarkerArray()

        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        delete_marker.header.frame_id = "odom"
        delete_marker.header.stamp = now_stamp
        ma.markers.append(delete_marker)

        marker_id = 0
        nodes = list(self._builder.nodes.values())

        # ── Container nodes: spheres sized by n_obs, coloured by confidence ──
        container_positions = []
        for node in nodes:
            p = node.position_3d
            px, py, pz = p["x"], p["y"], p["z"]
            container_positions.append((px, py, pz))

            sphere = Marker()
            sphere.header.frame_id = "odom"
            sphere.header.stamp = now_stamp
            sphere.ns = "graph_containers"
            sphere.id = marker_id
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = px
            sphere.pose.position.y = py
            sphere.pose.position.z = pz
            sphere.pose.orientation.w = 1.0
            sz = 0.06 + 0.02 * min(node.n_obs, 10)
            sphere.scale.x = sz
            sphere.scale.y = sz
            sphere.scale.z = sz
            s = min(node.max_score, 1.0)
            sphere.color = ColorRGBA(r=0.2, g=0.4 + 0.6 * s, b=0.9, a=0.85)
            sphere.lifetime = RosDuration(sec=0, nanosec=0)
            ma.markers.append(sphere)
            marker_id += 1

            label = Marker()
            label.header.frame_id = "odom"
            label.header.stamp = now_stamp
            label.ns = "graph_labels"
            label.id = marker_id
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = px
            label.pose.position.y = py
            label.pose.position.z = pz + 0.12
            label.pose.orientation.w = 1.0
            label.scale.z = 0.06
            label.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.9)
            label.text = f"{node.node_type} ({node.max_score:.2f})"
            label.lifetime = RosDuration(sec=0, nanosec=0)
            ma.markers.append(label)
            marker_id += 1

        # ── Room node: larger sphere at centroid of all containers ──
        if container_positions:
            cx = sum(p[0] for p in container_positions) / len(container_positions)
            cy = sum(p[1] for p in container_positions) / len(container_positions)
            cz = sum(p[2] for p in container_positions) / len(container_positions)

            room_sphere = Marker()
            room_sphere.header.frame_id = "odom"
            room_sphere.header.stamp = now_stamp
            room_sphere.ns = "graph_room"
            room_sphere.id = marker_id
            room_sphere.type = Marker.SPHERE
            room_sphere.action = Marker.ADD
            room_sphere.pose.position.x = cx
            room_sphere.pose.position.y = cy
            room_sphere.pose.position.z = cz
            room_sphere.pose.orientation.w = 1.0
            room_sphere.scale.x = 0.18
            room_sphere.scale.y = 0.18
            room_sphere.scale.z = 0.18
            room_sphere.color = ColorRGBA(r=1.0, g=0.6, b=0.0, a=0.8)
            room_sphere.lifetime = RosDuration(sec=0, nanosec=0)
            ma.markers.append(room_sphere)
            marker_id += 1

            room_label = Marker()
            room_label.header.frame_id = "odom"
            room_label.header.stamp = now_stamp
            room_label.ns = "graph_room_label"
            room_label.id = marker_id
            room_label.type = Marker.TEXT_VIEW_FACING
            room_label.action = Marker.ADD
            room_label.pose.position.x = cx
            room_label.pose.position.y = cy
            room_label.pose.position.z = cz + 0.15
            room_label.pose.orientation.w = 1.0
            room_label.scale.z = 0.08
            room_label.color = ColorRGBA(r=1.0, g=0.8, b=0.2, a=1.0)
            room_label.text = f"room:{self.room_type}"
            room_label.lifetime = RosDuration(sec=0, nanosec=0)
            ma.markers.append(room_label)
            marker_id += 1

            # ── Edges: room ↔ container as lines ──
            edges = Marker()
            edges.header.frame_id = "odom"
            edges.header.stamp = now_stamp
            edges.ns = "graph_edges"
            edges.id = marker_id
            edges.type = Marker.LINE_LIST
            edges.action = Marker.ADD
            edges.pose.orientation.w = 1.0
            edges.scale.x = 0.01
            edges.color = ColorRGBA(r=1.0, g=0.6, b=0.0, a=0.4)
            edges.lifetime = RosDuration(sec=0, nanosec=0)
            room_pt = Point(x=cx, y=cy, z=cz)
            for px, py, pz in container_positions:
                edges.points.append(room_pt)
                edges.points.append(Point(x=px, y=py, z=pz))
            ma.markers.append(edges)
            marker_id += 1

        # ── Ranked containers (after GNN scoring): green=high, red=low ──
        with self._rankings_lock:
            for rank in self._rankings:
                pos = rank.get("position_3d")
                if pos is None:
                    continue
                score = rank.get("score", 0.0)

                sphere = Marker()
                sphere.header.frame_id = "odom"
                sphere.header.stamp = now_stamp
                sphere.ns = "ranked_containers"
                sphere.id = marker_id
                sphere.type = Marker.SPHERE
                sphere.action = Marker.ADD
                sphere.pose.position.x = pos[0]
                sphere.pose.position.y = pos[1]
                sphere.pose.position.z = pos[2] if len(pos) > 2 else 0.5
                sphere.pose.orientation.w = 1.0
                sphere.scale.x = 0.15
                sphere.scale.y = 0.15
                sphere.scale.z = 0.15
                sphere.color = ColorRGBA(
                    r=1.0 - score, g=score, b=0.0, a=0.9
                )
                sphere.lifetime = RosDuration(sec=0, nanosec=0)
                ma.markers.append(sphere)
                marker_id += 1

                label = Marker()
                label.header.frame_id = "odom"
                label.header.stamp = now_stamp
                label.ns = "ranked_labels"
                label.id = marker_id
                label.type = Marker.TEXT_VIEW_FACING
                label.action = Marker.ADD
                label.pose.position.x = pos[0]
                label.pose.position.y = pos[1]
                label.pose.position.z = (pos[2] if len(pos) > 2 else 0.5) + 0.15
                label.pose.orientation.w = 1.0
                label.scale.z = 0.08
                label.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
                label.text = f"{rank.get('container_type', '?')} ({score:.2f})"
                label.lifetime = RosDuration(sec=0, nanosec=0)
                ma.markers.append(label)
                marker_id += 1

        self._marker_pub.publish(ma)

    @staticmethod
    def _transform_to_matrix(transform) -> np.ndarray:
        from tf_transformations import quaternion_matrix
        t = transform.transform.translation
        q = transform.transform.rotation
        mat = quaternion_matrix([q.x, q.y, q.z, q.w])
        mat[0, 3] = t.x
        mat[1, 3] = t.y
        mat[2, 3] = t.z
        return mat


def main(args=None):
    rclpy.init(args=args)
    node = SceneGraphNode()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
