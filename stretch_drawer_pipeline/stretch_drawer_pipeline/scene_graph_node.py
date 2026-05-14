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
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image as RosImage
from std_msgs.msg import ColorRGBA, String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from builtin_interfaces.msg import Duration as RosDuration
import tf2_ros

_SEMANTIC_ROOT = Path(__file__).resolve().parent.parent.parent / "semantic-object-container-room"
if str(_SEMANTIC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SEMANTIC_ROOT))

MAX_PROJECTION_DEPTH = 6.0


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
        self._detecting = False

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=30))

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
        self.create_service(
            Trigger, "/scene_graph/build_and_rank",
            self._build_and_rank_callback,
            callback_group=self.cb_group,
        )
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

        self.get_logger().info(
            f"Scene graph node initialized: room={self.room_type}, "
            f"query={self.query}, device={self.device}"
        )

    # ── Image transport ──────────────────────────────────────────────

    def _setup_dds_images(self):
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        sensor_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

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

        self._rgb_topic.subscribe(self._rgb_ws_callback)
        self._depth_topic.subscribe(self._depth_ws_callback)
        self._camera_info_topic.subscribe(self._camera_info_ws_callback)
        self._exploration_status_topic.subscribe(self._exploration_status_ws_callback)

        # TF: rely on Node 2 republishing TF locally from rosbridge.
        # We just need a TF listener on the local ROS2 network.
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._ros_client_thread = threading.Thread(
            target=self._ros_client.run, daemon=True
        )
        self._ros_client_thread.start()

    # ── DDS image callbacks ──────────────────────────────────────────

    def _rgb_dds_callback(self, msg: RosImage):
        import cv2
        try:
            arr = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            if not self.use_sim:
                arr = cv2.rotate(arr, cv2.ROTATE_90_CLOCKWISE)
            self.latest_rgb = arr
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
                    cx = (K.shape[0] - 1) - K[1, 2] if K.shape[0] > 3 else 719 - K[1, 2]
                    cy = K[0, 2]
                    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                self.camera_K = K
                self.get_logger().info(f"Camera K (WS): fx={K[0,0]:.1f}")
        except Exception:
            pass

    # ── Exploration status ───────────────────────────────────────────

    def _exploration_status_callback(self, msg: String):
        self._handle_exploration_status(msg.data)

    def _exploration_status_ws_callback(self, msg):
        self._handle_exploration_status(msg.get("data", ""))

    def _handle_exploration_status(self, status: str):
        if status == "paused_for_detection":
            if not self._detecting:
                self._detecting = True
                threading.Thread(
                    target=self._process_current_frame, daemon=True
                ).start()
        elif status == "complete":
            self._detecting = False
            self.get_logger().info("Exploration complete — auto-triggering GNN scoring")
            threading.Thread(target=self._auto_build_and_rank, daemon=True).start()
        else:
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
                device="cpu", score_threshold=self.det_min_score
            )
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
        import torch
        from PIL import Image

        if not self._ensure_models_loaded():
            return

        # Wait up to 3s for images to arrive
        import time as _time
        for _ in range(30):
            if self.latest_rgb is not None and self.latest_depth is not None:
                break
            _time.sleep(0.1)

        if self.latest_rgb is None or self.latest_depth is None:
            self.get_logger().warn(
                f"No camera data (rgb={'yes' if self.latest_rgb is not None else 'NO'}, "
                f"depth={'yes' if self.latest_depth is not None else 'NO'}) — skipping"
            )
            return

        camera_frame = "camera_color_optical_frame"
        try:
            transform = self.tf_buffer.lookup_transform(
                "odom", camera_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5),
            )
        except Exception as e:
            self.get_logger().warn(
                f"TF lookup failed: {e}", throttle_duration_sec=5.0
            )
            return

        camera_pose = self._transform_to_matrix(transform)

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
        dets = nms_by_type(dets)

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
            )
            if world_pos is None:
                continue

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

            self._builder.add_observation(
                obj_type=det.object_type,
                position_3d=world_pos,
                clip_embedding=clip_emb if clip_emb is not None else torch.zeros(self._clip_dim),
                scene_frame_clip=scene_frame_emb,
                crop_area=area,
            )
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

    # ── Helpers ───────────────────────────────────────────────────────

    # ── RViz markers ──────────────────────────────────────────────────

    def _publish_markers(self):
        """Publish detected objects and ranked containers as RViz markers."""
        ma = MarkerArray()

        # Delete all previous markers
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        delete_marker.header.frame_id = "odom"
        delete_marker.header.stamp = self.get_clock().now().to_msg()
        ma.markers.append(delete_marker)

        marker_id = 0

        # Raw detections: cyan spheres with text labels
        with self._detected_objects_lock:
            for obj in self._detected_objects:
                pos = obj["pos"]

                sphere = Marker()
                sphere.header.frame_id = "odom"
                sphere.header.stamp = self.get_clock().now().to_msg()
                sphere.ns = "scene_detections"
                sphere.id = marker_id
                sphere.type = Marker.SPHERE
                sphere.action = Marker.ADD
                sphere.pose.position.x = pos[0]
                sphere.pose.position.y = pos[1]
                sphere.pose.position.z = pos[2] if len(pos) > 2 else 0.5
                sphere.pose.orientation.w = 1.0
                sphere.scale.x = 0.08
                sphere.scale.y = 0.08
                sphere.scale.z = 0.08
                sphere.color = ColorRGBA(r=0.2, g=0.8, b=0.8, a=0.6)
                sphere.lifetime = RosDuration(sec=0, nanosec=0)
                ma.markers.append(sphere)
                marker_id += 1

                label = Marker()
                label.header.frame_id = "odom"
                label.header.stamp = self.get_clock().now().to_msg()
                label.ns = "scene_labels"
                label.id = marker_id
                label.type = Marker.TEXT_VIEW_FACING
                label.action = Marker.ADD
                label.pose.position.x = pos[0]
                label.pose.position.y = pos[1]
                label.pose.position.z = (pos[2] if len(pos) > 2 else 0.5) + 0.1
                label.pose.orientation.w = 1.0
                label.scale.z = 0.06
                label.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.9)
                label.text = obj["type"]
                label.lifetime = RosDuration(sec=0, nanosec=0)
                ma.markers.append(label)
                marker_id += 1

        # Ranked containers: colored by score (green=high, red=low)
        with self._rankings_lock:
            for rank in self._rankings:
                pos = rank.get("position_3d")
                if pos is None:
                    continue
                score = rank.get("score", 0.0)

                sphere = Marker()
                sphere.header.frame_id = "odom"
                sphere.header.stamp = self.get_clock().now().to_msg()
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
                label.header.stamp = self.get_clock().now().to_msg()
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
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
