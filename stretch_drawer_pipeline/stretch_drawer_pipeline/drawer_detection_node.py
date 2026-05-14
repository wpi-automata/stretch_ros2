#!/usr/bin/env python3
"""Node 2: Drawer Detection.

Detects drawers and handles in camera frames using a single Detic pass on
the full image. Handles are associated with drawers by containment: a handle
bbox whose center falls within a drawer bbox belongs to that drawer. The
grasp point is the center of the matched handle's bounding box.

For each drawer:
  1. Detic detects drawer and handle bounding boxes in one pass
  2. Handles are matched to drawers by bbox containment
  3. The handle bbox center is the grasp point, projected to world coordinates
  4. Drawer corners are also projected to world for RViz visualization
  5. Handle orientation (horizontal/vertical) is determined from bbox aspect ratio
  6. Reachability is computed based on the Stretch3's kinematic workspace

Uses VisualDetector from semantic-object-container-room for Detic inference,
and its nms_by_type for per-class non-maximum suppression.

Publishes:
  - /drawer_detections_json (std_msgs/String): all detected drawers as JSON
  - /drawer_markers (visualization_msgs/MarkerArray): RViz visualization
      - Green/red cubes sized to drawer bounding box (reachable/unreachable)
      - Blue spheres at handle grasp points
      - White text labels with ID, reachability, orientation, distance

Subscribes:
  - /camera/color/image_raw (sensor_msgs/Image): RGB frames
  - /camera/depth/image_rect_raw (sensor_msgs/Image): depth frames
  - /camera/color/camera_info (sensor_msgs/CameraInfo): camera intrinsics
  - /exploration_status (std_msgs/String): to know when exploration is active

Services:
  - /detection/trigger (std_srvs/Trigger): force a detection pass
  - /detection/get_drawers (std_srvs/Trigger): return current drawer list as JSON

Parameters:
  - detection_confidence: min Detic score for drawer class (default 0.5)
  - dedup_distance_m: distance threshold to consider two detections the same (default 0.3)
  - max_reach_height: max z the gripper can reach (default 1.4m)
  - min_reach_height: min z the gripper can reach (default 0.1m)
  - test_mode: if true, process all frames without waiting for exploration (default false)
  - rank_via_LOCUS: if true, calls the GNN node ranking stub (default false)
"""

import base64
import json
import math
import sys
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

from cv_bridge import CvBridge
from geometry_msgs.msg import Point, TransformStamped, Vector3, Quaternion
from sensor_msgs.msg import CameraInfo, Image as RosImage
from std_msgs.msg import String, Header, ColorRGBA
from std_srvs.srv import Trigger
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros
import roslibpy

# Path to semantic-object-container-room for imports
_SEMANTIC_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "semantic-object-container-room"
if str(_SEMANTIC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SEMANTIC_ROOT))


class DetectedDrawer:
    """Internal representation of a drawer detection."""

    def __init__(self):
        self.drawer_id = str(uuid.uuid4())[:8]
        self.drawer_bbox = None
        self.handle_bbox = None
        self.handle_center_world = None
        self.handle_grasp_world = None
        self.drawer_corners_world = None
        self.reachable = False
        self.handle_orientation = "horizontal"
        self.ranking = 0.0
        self.distance_to_robot = float("inf")
        self.annotated_image = None
        self.confidence = 0.0
        self.observations = 1


class DrawerDetectionNode(Node):
    """Detects drawers and their handles, publishing world-frame locations."""

    def __init__(self):
        super().__init__("drawer_detection_node")

        # Parameters
        self.declare_parameter("detection_confidence", 0.5)
        self.declare_parameter("enable_dedup", True)
        self.declare_parameter("dedup_distance_m", 0.1)
        self.declare_parameter("max_reach_height", 1.4)
        self.declare_parameter("min_reach_height", 0.1)
        self.declare_parameter("max_reach_distance", 0.6)
        self.declare_parameter("test_mode", False)
        self.declare_parameter("rank_via_LOCUS", False)
        self.declare_parameter("use_sim", False)
        self.declare_parameter("detection_rate_hz", 2.0)
        self.declare_parameter("robot_ip", "")
        self.declare_parameter("robot_port", 9090)
        self.declare_parameter("remote_rgb_topic", "/camera/color/image_raw/compressed")
        self.declare_parameter("remote_depth_topic", "/camera/aligned_depth_to_color/image_raw/compressedDepth")
        self.declare_parameter("remote_camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("throttle_rate_ms", 2000)
        self.declare_parameter("keep_drawers_open", True)

        self.detection_confidence = self.get_parameter("detection_confidence").value
        self.enable_dedup = self.get_parameter("enable_dedup").value
        self.dedup_distance = self.get_parameter("dedup_distance_m").value
        self.max_reach_height = self.get_parameter("max_reach_height").value
        self.min_reach_height = self.get_parameter("min_reach_height").value
        self.max_reach_distance = self.get_parameter("max_reach_distance").value
        self.test_mode = self.get_parameter("test_mode").value
        self.rank_via_locus = self.get_parameter("rank_via_LOCUS").value
        self.use_sim = self.get_parameter("use_sim").value
        self.detection_rate = self.get_parameter("detection_rate_hz").value
        self.keep_drawers_open = self.get_parameter("keep_drawers_open").value

        params = {p.name: p.value for p in self.get_parameters(
            [d.name for d in self._parameters.values()]
        )}
        self.get_logger().info(f"Parameters: {params}")

        # State
        self.drawers: list[DetectedDrawer] = []
        self.interacted_drawers: dict[str, dict] = {}
        self.drawer_items: dict[str, list[dict]] = {}
        self.drawers_lock = threading.Lock()
        self.chosen_drawer_id = None
        self._detection_mode = "detecting"
        self._pending_items_scan = None
        self.gripper_handle_locations = []

        self._drawer_classes = {
            "Drawer", "Cabinet", "Chest", "FilingCabinet",
            "Dresser", "NightStand", "SideTable",
            "Armoire", "Buffet", "CedarChest", "ChestOfDrawers",
            "ChinaCabinet", "Credenza", "Cupboard", "AiringCupboard",
            "HopeChest", "Hutch", "Locker", "Footlocker",
            "MedicineChest", "Pantry", "Sideboard", "Wardrobe",
            "Cabinetwork",
        }
        self._handle_classes = {
            "Handle", "Knob", "Doorknob",
            "Pull", "Bellpull", "PullChain",
        }
        self.bridge = CvBridge()
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_rgb_stamp = None
        # D435i intrinsics — fallback if camera_info topic is unavailable.
        # Original at 1280x720: fx=911.968, fy=911.456, cx=639.360, cy=375.114
        if self.use_sim:
            self.camera_K = np.array([
                [911.968, 0.0,     639.360],
                [0.0,     911.456, 375.114],
                [0.0,     0.0,     1.0],
            ])
        else:
            # After ROTATE_90_CLOCKWISE to 720x1280:
            # new_fx=old_fy, new_fy=old_fx,
            # new_cx=H-1-old_cy=719-375.114, new_cy=old_cx
            self.camera_K = np.array([
                [911.456, 0.0,     343.886],
                [0.0,     911.968, 639.360],
                [0.0,     0.0,     1.0],
            ])
        self.detector = None
        self.exploring = False

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=30))
        self._tf_ready = False
        self._tf_consecutive_ok = 0
        self._TF_READY_THRESHOLD = 5

        self.cb_group = ReentrantCallbackGroup()

        # Publishers
        self.drawer_pub = self.create_publisher(
            String, "/drawer_detections_json", 10
        )
        self.marker_pub = self.create_publisher(
            MarkerArray, "/drawer_markers", 10
        )
        self.chosen_marker_pub = self.create_publisher(
            Marker, "/drawer_chosen_marker", 10
        )
        self.chosen_drawer_json_pub = self.create_publisher(
            String, "/detection/chosen_drawer_json", 10
        )
        self.close_drawer_pub = self.create_publisher(
            String, "/detection/close_drawer_json", 10
        )

        self.create_subscription(
            String, "/exploration_status",
            self.exploration_status_callback, 10
        )
        self.create_subscription(
            String, "/navigate_open/opened_drawers_json",
            self._opened_drawers_json_callback, 10
        )

        # Image transport: rosbridge WebSocket (real) or DDS (sim)
        self._robot_ip = self.get_parameter("robot_ip").value
        if self._robot_ip:
            self._setup_rosbridge_images()
        else:
            self._setup_dds_images()

        # Services
        self.create_service(
            Trigger, "/detection/trigger",
            self.trigger_detection_callback,
            callback_group=self.cb_group,
        )
        self.create_service(
            Trigger, "/detection/get_drawers",
            self.get_drawers_callback,
            callback_group=self.cb_group,
        )
        self.create_service(
            Trigger, "/detection/choose_drawer",
            self.choose_drawer_callback,
            callback_group=self.cb_group,
        )

        # Detection timer
        period = 1.0 / self.detection_rate
        self.create_timer(period, self.detection_tick)

        # Visualization timer
        self.create_timer(1.0, self.publish_markers)

        self.get_logger().info(
            f"Drawer detection node initialized: "
            f"confidence={self.detection_confidence}, "
            f"test_mode={self.test_mode}, "
            f"rank_via_LOCUS={self.rank_via_locus}"
        )

    # ─── Image transport setup ───────────────────────────────────────

    def _setup_dds_images(self):
        """Subscribe to camera topics via DDS (sim or same-machine)."""
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        sensor_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(
            RosImage, "/camera/color/image_raw",
            self._rgb_dds_callback, sensor_qos
        )
        self.create_subscription(
            RosImage, "/camera/depth/image_rect_raw",
            self._depth_dds_callback, sensor_qos
        )
        self.create_subscription(
            CameraInfo, "/camera/color/camera_info",
            self._camera_info_dds_callback, sensor_qos
        )
        self.get_logger().info("Image transport: DDS subscriptions")

    def _setup_rosbridge_images(self):
        """Receive camera images via rosbridge WebSocket (real hardware)."""
        robot_port = self.get_parameter("robot_port").value
        throttle_ms = self.get_parameter("throttle_rate_ms").value
        remote_rgb = self.get_parameter("remote_rgb_topic").value
        remote_depth = self.get_parameter("remote_depth_topic").value

        self.get_logger().info(
            f"Image transport: rosbridge at {self._robot_ip}:{robot_port}"
        )
        remote_camera_info = self.get_parameter("remote_camera_info_topic").value

        self.get_logger().info(
            f"Remote topics: rgb={remote_rgb}, depth={remote_depth}, "
            f"camera_info={remote_camera_info}"
        )

        self._ws_rgb_count = 0
        self._ws_depth_count = 0

        self._ros_client = roslibpy.Ros(
            host=self._robot_ip, port=robot_port
        )

        self._rgb_topic = roslibpy.Topic(
            self._ros_client, remote_rgb, "sensor_msgs/msg/CompressedImage",
            throttle_rate=throttle_ms,
        )
        self._depth_topic = roslibpy.Topic(
            self._ros_client, remote_depth, "sensor_msgs/msg/CompressedImage",
            throttle_rate=throttle_ms,
        )
        self._camera_info_topic = roslibpy.Topic(
            self._ros_client, remote_camera_info, "sensor_msgs/msg/CameraInfo",
        )

        self._tf_topic = roslibpy.Topic(
            self._ros_client, "/tf", "tf2_msgs/msg/TFMessage",
        )
        self._tf_static_topic = roslibpy.Topic(
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

        self._robot_desc_topic = roslibpy.Topic(
            self._ros_client, "/robot_description", "std_msgs/msg/String",
        )
        self._robot_desc_pub = self.create_publisher(
            String, "/robot_description",
            rclpy.qos.QoSProfile(
                depth=1,
                durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )

        self._ws_drawer_pub = roslibpy.Topic(
            self._ros_client, "/drawer_detections_json", "std_msgs/msg/String",
        )
        self._ws_chosen_pub = roslibpy.Topic(
            self._ros_client, "/detection/chosen_drawer_json", "std_msgs/msg/String",
        )
        self._ws_close_drawer_pub = roslibpy.Topic(
            self._ros_client, "/detection/close_drawer_json", "std_msgs/msg/String",
        )

        # Subscribe to robot-side topics via rosbridge
        self._ws_opened_json_topic = roslibpy.Topic(
            self._ros_client, "/navigate_open/opened_drawers_json", "std_msgs/msg/String",
        )
        self._ws_exploration_status_topic = roslibpy.Topic(
            self._ros_client, "/exploration_status", "std_msgs/msg/String",
        )


        self._rgb_topic.subscribe(self._rosbridge_rgb_callback)
        self._depth_topic.subscribe(self._rosbridge_depth_callback)
        self._camera_info_topic.subscribe(self._rosbridge_camera_info_callback)
        self._tf_topic.subscribe(self._rosbridge_tf_callback)
        self._tf_static_topic.subscribe(self._rosbridge_tf_static_callback)
        self._robot_desc_topic.subscribe(self._rosbridge_robot_desc_callback)
        self._ws_opened_json_topic.subscribe(self._rosbridge_opened_json_callback)
        self._ws_exploration_status_topic.subscribe(self._rosbridge_exploration_status_callback)

        self._ros_client_thread = threading.Thread(
            target=self._ros_client.run, daemon=True
        )
        self._ros_client_thread.start()

        self.create_timer(10.0, self._ws_log_stats)

    def _rosbridge_rgb_callback(self, msg_dict):
        try:
            data = msg_dict["data"]
            if isinstance(data, str):
                data = base64.b64decode(data)
            arr = cv2.imdecode(
                np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if arr is None:
                return
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            if not self.use_sim:
                arr = cv2.rotate(arr, cv2.ROTATE_90_CLOCKWISE)
            self.latest_rgb = arr
            stamp = msg_dict.get("header", {}).get("stamp", {})
            image_sec = stamp.get("sec", 0)
            image_nsec = stamp.get("nanosec", 0)
            self.latest_rgb_stamp = rclpy.time.Time(
                seconds=image_sec, nanoseconds=image_nsec,
            ).to_msg()
            now_ns = time.time_ns()
            image_ns = image_sec * 1_000_000_000 + image_nsec
            lag = (now_ns - image_ns) / 1e9
            self.get_logger().debug(
                f"[rosbridge] RGB lag: {lag:.2f}s",
                throttle_duration_sec=5.0,
            )
            self._ws_rgb_count += 1
        except Exception as e:
            print(f"[rosbridge] RGB decode FAILED: {e}", flush=True)

    def _rosbridge_depth_callback(self, msg_dict):
        try:
            data = msg_dict["data"]
            if isinstance(data, str):
                data = base64.b64decode(data)
            raw = np.frombuffer(data, dtype=np.uint8)
            # compressedDepth has a 12-byte header before the PNG data
            arr = cv2.imdecode(raw[12:], cv2.IMREAD_UNCHANGED)
            if arr is None:
                arr = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
            if arr is None:
                self.get_logger().warn("Failed to decode compressed depth")
                return
            if not self.use_sim:
                arr = cv2.rotate(arr, cv2.ROTATE_90_CLOCKWISE)
            self.latest_depth = arr
            self._ws_depth_count += 1
        except Exception as e:
            self.get_logger().warn(
                f"rosbridge depth decode failed: {e}",
                throttle_duration_sec=5.0,
            )

    def _rosbridge_camera_info_callback(self, msg_dict):
        try:
            k = msg_dict.get("k", msg_dict.get("K", []))
            if len(k) == 9 and k[0] > 0:
                raw_K = np.array([
                    [k[0], k[1], k[2]],
                    [k[3], k[4], k[5]],
                    [k[6], k[7], k[8]],
                ])
                if not self.use_sim:
                    h = msg_dict.get("height", 720)
                    self.camera_K = np.array([
                        [raw_K[1, 1], 0.0,         h - 1 - raw_K[1, 2]],
                        [0.0,         raw_K[0, 0],  raw_K[0, 2]],
                        [0.0,         0.0,          1.0],
                    ])
                else:
                    self.camera_K = raw_K
                self.get_logger().info(
                    f"Camera intrinsics from rosbridge: "
                    f"fx={self.camera_K[0,0]:.1f}, fy={self.camera_K[1,1]:.1f}, "
                    f"cx={self.camera_K[0,2]:.1f}, cy={self.camera_K[1,2]:.1f}"
                )
                self._camera_info_topic.unsubscribe()
        except Exception as e:
            self.get_logger().warn(f"camera_info decode failed: {e}")

    def _rosbridge_tf_callback(self, msg_dict):
        self._republish_tf(msg_dict, self._tf_pub)

    def _rosbridge_tf_static_callback(self, msg_dict):
        self._republish_tf(msg_dict, self._tf_static_pub, static=True)
        self._tf_static_topic.unsubscribe()
        self.get_logger().info("Received and republished static TFs via rosbridge")

    def _rosbridge_robot_desc_callback(self, msg_dict):
        try:
            data = msg_dict.get("data", "")
            if data:
                msg = String()
                msg.data = data
                self._robot_desc_pub.publish(msg)
                self.get_logger().info(
                    f"Republished /robot_description ({len(data)} bytes)"
                )
        except Exception as e:
            self.get_logger().warn(f"robot_description relay failed: {e}")


    def _rosbridge_exploration_status_callback(self, msg_dict):
        try:
            data = msg_dict.get("data", "")
            if data:
                msg = String()
                msg.data = data
                self.exploration_status_callback(msg)
        except Exception as e:
            self.get_logger().warn(f"rosbridge exploration_status relay failed: {e}")

    def _rosbridge_opened_json_callback(self, msg_dict):
        try:
            data = msg_dict.get("data", "")
            if data:
                msg = String()
                msg.data = data
                self._opened_drawers_json_callback(msg)
        except Exception as e:
            self.get_logger().warn(f"rosbridge opened_drawers_json relay failed: {e}")


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

    def _ws_log_stats(self):
        connected = self._ros_client.is_connected
        self.get_logger().info(
            f"rosbridge: rgb={self._ws_rgb_count}, "
            f"depth={self._ws_depth_count}, connected={connected}"
        )

    def _publish_drawers_via_rosbridge(self):
        if not hasattr(self, "_ws_drawer_pub") or not self._ros_client.is_connected:
            return
        try:
            response = Trigger.Response()
            self.get_drawers_callback(None, response)
            self._ws_drawer_pub.publish(roslibpy.Message({"data": response.message}))
        except Exception as e:
            self.get_logger().warn(f"rosbridge drawer publish failed: {e}", throttle_duration_sec=10.0)

    # ─── Callbacks ────────────────────────────────────────────────────

    def _rgb_dds_callback(self, msg: RosImage):
        arr = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        if not self.use_sim:
            arr = cv2.rotate(arr, cv2.ROTATE_90_CLOCKWISE)
        self.latest_rgb = arr
        self.latest_rgb_stamp = msg.header.stamp

    def _depth_dds_callback(self, msg: RosImage):
        arr = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        if not self.use_sim:
            arr = cv2.rotate(arr, cv2.ROTATE_90_CLOCKWISE)
        self.latest_depth = arr

    def _camera_info_dds_callback(self, msg: CameraInfo):
        k = msg.k
        if k[0] == 0:
            return
        K = np.array([[k[0], k[1], k[2]],
                      [k[3], k[4], k[5]],
                      [k[6], k[7], k[8]]])
        if not self.use_sim:
            h = msg.height
            K = np.array([[K[1, 1], 0.0,     h - 1 - K[1, 2]],
                          [0.0,     K[0, 0], K[0, 2]],
                          [0.0,     0.0,     1.0]])
        if not hasattr(self, '_camera_info_logged'):
            self._camera_info_logged = True
            self.get_logger().info(
                f"Camera intrinsics from DDS: "
                f"fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, "
                f"cx={K[0,2]:.1f}, cy={K[1,2]:.1f}"
            )
        self.camera_K = K

    def exploration_status_callback(self, msg: String):
        self.exploring = msg.data in ("rotating", "planning", "navigating")

    def _opened_drawers_json_callback(self, msg: String):
        """Handle opened drawer notification from Node 3.

        Removes the drawer from the closed list, stores its gripper position
        in gripper_handle_locations for later matching, and resumes detection.
        """
        try:
            entry = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().warn(f"Bad opened drawers JSON: {e}")
            return

        drawer_id = entry.get("drawer_id")
        if not drawer_id:
            self.get_logger().warn("Opened drawer JSON missing drawer_id")
            return

        with self.drawers_lock:
            self.drawers = [d for d in self.drawers if d.drawer_id != drawer_id]

        self.gripper_handle_locations.append(entry)

        self.interacted_drawers[drawer_id] = {
            "drawer_id": drawer_id,
            "closed_handle": entry.get("closed_handle"),
            "closed_corners": entry.get("closed_corners"),
            "opened_handle": None,
            "opened_corners": None,
            "handle_orientation": entry.get("handle_orientation", "horizontal"),
            "pull_distance": entry.get("pull_distance", 0.0),
            "status": "opened",
        }

        self._pending_items_scan = drawer_id
        self._pending_items_scan_time = time.time()
        self._detection_mode = "detecting"

        gripper = entry.get("gripper_pos")
        gstr = (f"({gripper['x']:.3f}, {gripper['y']:.3f}, {gripper['z']:.3f})"
                if gripper else "None")
        self.get_logger().info(
            f"Drawer {drawer_id} opened — removed from closed "
            f"({len(self.drawers)} remaining), gripper at {gstr}, "
            f"items scan pending, detection resumed"
        )

    # UNUSED!!!
    # def _get_gripper_world_pos(self):
    #     """Get gripper tip position in odom frame via TF."""
    #     try:
    #         transform = self.tf_buffer.lookup_transform(
    #             "odom", "link_gripper_finger_left",
    #             rclpy.time.Time(),
    #             timeout=rclpy.duration.Duration(seconds=0.2),
    #         )
    #         t = transform.transform.translation
    #         return np.array([t.x, t.y, t.z])
    #     except Exception:
            # return None

    def trigger_detection_callback(self, request, response):
        """Manually trigger a detection pass."""
        count = self._run_detection()
        response.success = True
        response.message = f"Detected {count} total drawers"
        return response

    def get_drawers_callback(self, request, response):
        """Return current drawer list as JSON."""
        with self.drawers_lock:
            drawer_data = []
            for d in self.drawers:
                corners_list = None
                if d.drawer_corners_world is not None:
                    corners_list = [
                        {"x": float(c[0]), "y": float(c[1]), "z": float(c[2])}
                        for c in d.drawer_corners_world
                    ]
                drawer_data.append({
                    "drawer_id": d.drawer_id,
                    "handle_center_world": {
                        "x": float(d.handle_center_world[0]) if d.handle_center_world is not None else 0,
                        "y": float(d.handle_center_world[1]) if d.handle_center_world is not None else 0,
                        "z": float(d.handle_center_world[2]) if d.handle_center_world is not None else 0,
                    },
                    "drawer_corners_world": corners_list,
                    "reachable": d.reachable,
                    "handle_orientation": d.handle_orientation,
                    "ranking": d.ranking,
                    "distance_to_robot": d.distance_to_robot,
                })
        response.success = True
        response.message = json.dumps(drawer_data)
        return response

    def choose_drawer_callback(self, request, response):
        """Select the best drawer and publish its marker locally for RViz.

        Priority: ranking score (descending), then distance (ascending).
        Only considers reachable drawers unless none are reachable.
        Returns the chosen drawer as JSON. Also sends the chosen drawer
        to the robot via rosbridge so navigate_open_node can act on it.
        """
        with self.drawers_lock:
            if not self.drawers:
                response.success = False
                response.message = "No drawers detected"
                return response

            reachable = [d for d in self.drawers if d.reachable]
            candidates = reachable if reachable else self.drawers

            candidates.sort(
                key=lambda d: (-d.ranking, -d.confidence, d.distance_to_robot)
            )
            chosen = candidates[0]
            self.chosen_drawer_id = chosen.drawer_id

        self._publish_chosen_marker(chosen)

        corners_list = None
        if chosen.drawer_corners_world is not None:
            corners_list = [
                {"x": float(c[0]), "y": float(c[1]), "z": float(c[2])}
                for c in chosen.drawer_corners_world
            ]
        chosen_data = {
            "drawer_id": chosen.drawer_id,
            "handle_center_world": {
                "x": float(chosen.handle_center_world[0]),
                "y": float(chosen.handle_center_world[1]),
                "z": float(chosen.handle_center_world[2]),
            },
            "drawer_corners_world": corners_list,
            "reachable": chosen.reachable,
            "handle_orientation": chosen.handle_orientation,
            "ranking": chosen.ranking,
            "distance_to_robot": chosen.distance_to_robot,
        }
        chosen_json = json.dumps(chosen_data)

        chosen_msg = String()
        chosen_msg.data = chosen_json
        self.chosen_drawer_json_pub.publish(chosen_msg)

        if hasattr(self, "_ws_chosen_pub") and self._ros_client.is_connected:
            self._ws_chosen_pub.publish(
                roslibpy.Message({"data": chosen_json})
            )

        self._detection_mode = "stopped"
        self.get_logger().info(
            f"Chosen drawer: {chosen.drawer_id}, "
            f"distance={chosen.distance_to_robot:.2f}m, "
            f"reachable={chosen.reachable} — detection stopped"
        )
        response.success = True
        response.message = chosen_json
        return response

    def _publish_chosen_marker(self, drawer):
        """Publish a pink cube marker for the chosen drawer, visible in local RViz."""
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "chosen_drawer"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0

        if drawer.drawer_corners_world is not None and len(drawer.drawer_corners_world) == 4:
            xs = [float(c[0]) for c in drawer.drawer_corners_world]
            ys = [float(c[1]) for c in drawer.drawer_corners_world]
            zs = [float(c[2]) for c in drawer.drawer_corners_world]
            marker.pose.position.x = (min(xs) + max(xs)) / 2
            marker.pose.position.y = (min(ys) + max(ys)) / 2
            marker.pose.position.z = (min(zs) + max(zs)) / 2
            marker.scale.x = max(max(xs) - min(xs), 0.02)
            marker.scale.y = max(max(ys) - min(ys), 0.02)
            marker.scale.z = max(max(zs) - min(zs), 0.02)
        else:
            p = drawer.handle_center_world
            marker.pose.position.x = float(p[0])
            marker.pose.position.y = float(p[1])
            marker.pose.position.z = float(p[2])
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.15

        marker.color = ColorRGBA(r=1.0, g=0.4, b=0.7, a=0.9)
        marker.lifetime.sec = 120

        self.chosen_marker_pub.publish(marker)

    # ─── Detection logic ──────────────────────────────────────────────

    def detection_tick(self):
        """Periodic detection pass gated by detection mode."""
        if self._detection_mode == "stopped":
            return
        if not self.exploring and not self.test_mode and self._detection_mode != "detecting":
            return
        if self.latest_rgb is None or self.latest_depth is None:
            self.get_logger().info(
                f"Waiting: rgb={'ok' if self.latest_rgb is not None else 'NONE'}, "
                f"depth={'ok' if self.latest_depth is not None else 'NONE'}, "
                f"camera_K={'ok' if self.camera_K is not None else 'NONE'}",
                throttle_duration_sec=5.0,
            )
            return
        self._run_detection()

    def _run_detection(self) -> int:
        """Run Detic detection on current frame, find drawers and handles."""
        if self.latest_rgb is None or self.latest_depth is None:
            return 0
        if self.camera_K is None:
            self.get_logger().debug("Waiting for camera_info...")
            return 0

        camera_frame = "camera_color_optical_frame"
        try:
            transform = self.tf_buffer.lookup_transform(
                "odom", camera_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5),
            )
        except Exception as e:
            self._tf_consecutive_ok = 0
            self.get_logger().warn(
                f"TF lookup failed (skipping frame): {e}",
                throttle_duration_sec=5.0,
            )
            return len(self.drawers)

        tf_stamp = transform.header.stamp
        now = self.get_clock().now()
        tf_age = now.nanoseconds / 1e9 - (tf_stamp.sec + tf_stamp.nanosec / 1e9)
        self.get_logger().info(
            f"TF lookup OK — stamp={tf_stamp.sec}.{tf_stamp.nanosec:09d}, "
            f"age={tf_age:.1f}s, "
            f"t=({transform.transform.translation.x:.3f}, "
            f"{transform.transform.translation.y:.3f}, "
            f"{transform.transform.translation.z:.3f})",
            throttle_duration_sec=5.0,
        )

        # Gate: require N consecutive successful lookups before allowing detections
        if not self._tf_ready:
            self._tf_consecutive_ok += 1
            if self._tf_consecutive_ok < self._TF_READY_THRESHOLD:
                self.get_logger().info(
                    f"TF warming up: {self._tf_consecutive_ok}/{self._TF_READY_THRESHOLD}",
                    throttle_duration_sec=2.0,
                )
                return 0
            self._tf_ready = True
            self.get_logger().info("TF chain confirmed — detections enabled")

        rgb = self.latest_rgb.copy()
        depth = self.latest_depth.copy()
        camera_K = self.camera_K

        # Build camera pose matrix from TF
        camera_pose = self._transform_to_matrix(transform)

        # Detect drawers in the frame
        drawer_bboxes, all_detections, unpaired_handles = self._detect_drawers_detic(rgb)

        new_detections = 0
        projected_handles = []
        for drawer_bbox, handle_bbox, confidence in drawer_bboxes:

            drawer = self._create_drawer(projected_handles, camera_K, camera_pose, rgb, handle_bbox, drawer_bbox, confidence, depth)

            if drawer is not None:
                with self.drawers_lock:
                    self.drawers.append(drawer)
                new_detections += 1
        
        self._pair_handles_to_gripper_pos_after_opening_drawer(projected_handles, unpaired_handles, depth, camera_pose, camera_K)

        if self._pending_items_scan:
            elapsed = time.time() - self._pending_items_scan_time
            if elapsed >= 5.0:
                scan_id = self._pending_items_scan
                self._pending_items_scan = None
                self._scan_drawer_items(scan_id, all_detections, rgb, depth, camera_pose, camera_K)

        # Update distances to robot
        self._update_distances()

        # Optionally rank via LOCUS GNN
        if self.rank_via_locus:
            self._rank_via_locus_stub()

        self._save_debug_image(rgb, all_detections, drawer_bboxes)

        self.get_logger().info(
            f"Detection pass: {new_detections} new | "
            f"{len(self.drawers)} closed | "
            f"{len(self.interacted_drawers)} opened"
        )

        if new_detections > 0:
            self._save_debug_image(rgb, all_detections, drawer_bboxes)
            self._publish_drawers_via_rosbridge()

        return len(self.drawers)
    
    def _create_drawer(self, projected_handles, camera_K, camera_pose, rgb, handle_bbox, drawer_bbox, confidence, depth):
        # Project handle center to world
            u_center = int((handle_bbox[0] + handle_bbox[2]) / 2)
            v_center = int((handle_bbox[1] + handle_bbox[3]) / 2)
            depth_at_center = depth[
                max(0, v_center-3):v_center+3,
                max(0, u_center-3):u_center+3
            ]
            depth_m_center = self._depth_to_meters(depth_at_center)
            valid_m = depth_m_center[depth_m_center > 0.1]
            median_m = float(np.median(valid_m)) if len(valid_m) > 0 else 0.0
            self.get_logger().info(
                f"Projecting handle {handle_bbox}, "
                f"depth shape={depth.shape}, dtype={depth.dtype}, "
                f"rgb shape={rgb.shape}, "
                f"depth@center raw min={depth_at_center.min()} max={depth_at_center.max()} "
                f"median_m={median_m:.3f}m, "
                f"nonzero={np.count_nonzero(depth_at_center)}/{depth_at_center.size}, "
                f"whole_depth nonzero={np.count_nonzero(depth)}/{depth.size}"
            )
            cam_xyz = camera_pose[:3, 3]
            self.get_logger().info(
                f"Camera pose in odom: ({cam_xyz[0]:.3f}, {cam_xyz[1]:.3f}, {cam_xyz[2]:.3f})"
            )
            world_pos = self._project_to_world(
                handle_bbox, depth, camera_pose, camera_K
            )
            if world_pos is not None:
                dist = np.linalg.norm(world_pos[:2] - cam_xyz[:2])
                self.get_logger().info(
                    f"Handle world: ({world_pos[0]:.3f}, {world_pos[1]:.3f}, {world_pos[2]:.3f}), "
                    f"distance from camera={dist:.3f}m"
                )
            if world_pos is None:
                self.get_logger().warn(
                    f"Projection failed for handle {handle_bbox} — "
                    f"no valid depth at center"
                )
                self._save_projection_debug(rgb, depth, drawer_bbox, handle_bbox)
                return None
            self.get_logger().info(f"Handle projected to world: {world_pos}")
            projected_handles.append((world_pos, drawer_bbox, handle_bbox))

            handle_w = handle_bbox[2] - handle_bbox[0]
            handle_h = handle_bbox[3] - handle_bbox[1]
            orientation = "horizontal" if handle_w > handle_h else "vertical"

            # Check reachability
            reachable = self._check_reachability(world_pos)

            # Project drawer corners to world for bounding box
            drawer_corners = self._project_drawer_corners(
                drawer_bbox, depth, camera_pose, camera_K
            )
            if drawer_corners is None:
                self.get_logger().warn(
                    f"REJECTED drawer bbox {drawer_bbox} — "
                    f"bad depth at corners, cannot project 3D bounding box"
                )
                self._save_projection_debug(rgb, depth, drawer_bbox, handle_bbox)
                return None

            # De-duplicate against existing drawers
            if self.enable_dedup:
                merged = self._try_merge_detection(world_pos, confidence, drawer_corners)
                if merged:
                    return None

            # Create new drawer entry
            drawer = DetectedDrawer()
            drawer.handle_center_world = world_pos
            drawer.handle_grasp_world = world_pos.copy()
            drawer.drawer_corners_world = drawer_corners
            drawer.reachable = reachable
            drawer.handle_orientation = orientation
            drawer.confidence = confidence
            drawer.annotated_image = self._annotate_image(
                rgb, drawer_bbox, handle_bbox
            )
            return drawer

    def _pair_handles_to_gripper_pos_after_opening_drawer(self, projected_handles, unpaired_handles, depth, camera_pose, camera_K):
         # Project unpaired handles and add to projected list
        for handle_bbox in unpaired_handles:
            world_pos = self._project_to_world(
                handle_bbox, depth, camera_pose, camera_K
            )
            if world_pos is not None:
                projected_handles.append((world_pos, None, handle_bbox))

        # Match projected handles against gripper_handle_locations
        if self.gripper_handle_locations and projected_handles:
            NEAR_THRESHOLD = 0.15
            matched_indices = []
            for world_pos, drawer_bbox_or_none, handle_bbox in projected_handles:
                for idx, entry in enumerate(self.gripper_handle_locations):
                    if idx in matched_indices:
                        continue
                    gripper = entry.get("gripper_pos")
                    if gripper is None:
                        continue
                    ref = np.array([gripper["x"], gripper["y"], gripper["z"]])
                    dist = np.linalg.norm(world_pos - ref)
                    if dist < NEAR_THRESHOLD:
                        if drawer_bbox_or_none is not None:
                            corners = self._project_drawer_corners(
                                drawer_bbox_or_none, depth, camera_pose, camera_K
                            )
                            if corners is None:
                                corners = self._translate_closed_corners(world_pos, entry)
                        else:
                            corners = self._translate_closed_corners(world_pos, entry)
                        self._store_opened_drawer(world_pos, corners, entry)
                        matched_indices.append(idx)
                        self.get_logger().info(
                            f"Matched handle at ({world_pos[0]:.3f}, {world_pos[1]:.3f}, {world_pos[2]:.3f}) "
                            f"to gripper location for {entry['drawer_id']} (dist={dist:.3f}m)"
                        )
                        break
            for idx in sorted(matched_indices, reverse=True):
                self.gripper_handle_locations.pop(idx)

    def _translate_closed_corners(self, new_handle_pos: np.ndarray, entry: dict):
        """Translate the closed drawer bbox to the new handle location."""
        closed_handle_d = entry["closed_handle"]
        closed_handle = np.array([closed_handle_d["x"], closed_handle_d["y"], closed_handle_d["z"]])
        closed_corners_dicts = entry["closed_corners"]
        closed_corners = [np.array([c["x"], c["y"], c["z"]]) for c in closed_corners_dicts]
        offset = new_handle_pos - closed_handle
        return [c + offset for c in closed_corners]

    def _store_opened_drawer(self, opened_handle: np.ndarray,
                              opened_corners: list, entry: dict):
        """Record an opened drawer detection."""
        drawer_id = entry["drawer_id"]

        def _pos_dict(pos):
            return {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])}

        opened_corners_dicts = [_pos_dict(c) for c in opened_corners]
        opened_handle_dict = _pos_dict(opened_handle)

        if drawer_id in self.interacted_drawers:
            self.interacted_drawers[drawer_id]["opened_handle"] = opened_handle_dict
            self.interacted_drawers[drawer_id]["opened_corners"] = opened_corners_dicts

        self.get_logger().info(
            f"Opened drawer {drawer_id} recorded — "
            f"handle at ({opened_handle[0]:.3f}, {opened_handle[1]:.3f}, {opened_handle[2]:.3f}). "
            f"{len(self.drawers)} closed | {len(self.interacted_drawers)} opened."
        )

    def _scan_drawer_items(self, drawer_id, all_detections, rgb, depth, camera_pose, camera_K):
        """Detect objects inside an opened drawer using 2D projection.

        Projects the 3D drawer volume corners into the image, builds a
        convex hull, and checks which detection bboxes are fully contained
        within it. No depth needed for the containment check.
        """
        self.get_logger().info(
            f"Items scan starting for {drawer_id}, image={rgb.shape[1]}x{rgb.shape[0]}"
        )
        interacted = self.interacted_drawers.get(drawer_id)
        if not interacted:
            self.drawer_items[drawer_id] = []
            if not self.keep_drawers_open:
                self._send_close_drawer_command(drawer_id)
            return

        closed_corners = interacted.get("closed_corners")
        pull_distance = interacted.get("pull_distance", 0.0)
        if not closed_corners or len(closed_corners) != 4 or pull_distance <= 0:
            self.get_logger().info(
                f"No closed corners or pull_distance for drawer {drawer_id}, skipping items scan"
            )
            self.drawer_items[drawer_id] = []
            if not self.keep_drawers_open:
                self._send_close_drawer_command(drawer_id)
            return

        pts = np.array([[c["x"], c["y"], c["z"]] for c in closed_corners])

        v1 = pts[1] - pts[0]
        v2 = pts[3] - pts[0]
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-6:
            self.drawer_items[drawer_id] = []
            if not self.keep_drawers_open:
                self._send_close_drawer_command(drawer_id)
            return
        normal = normal / norm_len

        ch = interacted.get("closed_handle")
        if ch:
            handle_pos = np.array([ch["x"], ch["y"], ch["z"]])
            face_center = pts.mean(axis=0)
            if np.dot(normal, handle_pos - face_center) < 0:
                normal = -normal

        extruded_pts = pts + normal * pull_distance
        all_corners_3d = np.vstack([pts, extruded_pts])

        self.get_logger().info(
            f"Items scan 3D corners (closed): "
            + ", ".join(f"({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})" for p in pts)
        )
        self.get_logger().info(
            f"Items scan 3D corners (extruded): "
            + ", ".join(f"({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})" for p in extruded_pts)
        )
        self.get_logger().info(
            f"Items scan normal=({normal[0]:.3f},{normal[1]:.3f},{normal[2]:.3f}), "
            f"pull_dist={pull_distance:.3f}"
        )
        self.get_logger().info(
            f"Items scan camera_pos=({camera_pose[0,3]:.3f},{camera_pose[1,3]:.3f},{camera_pose[2,3]:.3f})"
        )
        if ch:
            handle_px = self._world_to_pixels(handle_pos.reshape(1, 3), camera_pose, camera_K)
            if handle_px is not None:
                self.get_logger().info(
                    f"Items scan handle projects to pixel ({handle_px[0,0]:.0f},{handle_px[0,1]:.0f})"
                )
            else:
                self.get_logger().info("Items scan handle behind camera")

        volume_pixels = self._world_to_pixels(all_corners_3d, camera_pose, camera_K)
        if volume_pixels is None:
            self.get_logger().info("Items scan: _world_to_pixels returned None (points behind camera)")
            self.drawer_items[drawer_id] = []
            if not self.keep_drawers_open:
                self._send_close_drawer_command(drawer_id)
            return

        self.get_logger().info(
            f"Items scan projected pixels (closed): "
            + ", ".join(f"({volume_pixels[i,0]:.0f},{volume_pixels[i,1]:.0f})" for i in range(4))
        )
        self.get_logger().info(
            f"Items scan projected pixels (extruded): "
            + ", ".join(f"({volume_pixels[i,0]:.0f},{volume_pixels[i,1]:.0f})" for i in range(4, 8))
        )

        hull = cv2.convexHull(volume_pixels.astype(np.float32))

        self.get_logger().info(
            f"Items scan 2D: pull_dist={pull_distance:.3f}, "
            f"hull has {len(hull)} pts, "
            f"normal=({normal[0]:.3f},{normal[1]:.3f},{normal[2]:.3f})"
        )

        for det in all_detections:
            if det.object_type in self._drawer_classes:
                world_pos = self._project_to_world(det.bbox, depth, camera_pose, camera_K)
                if world_pos is not None:
                    rt_px = self._world_to_pixels(world_pos.reshape(1, 3), camera_pose, camera_K)
                    if rt_px is not None:
                        cx_det = (det.bbox[0] + det.bbox[2]) / 2
                        cy_det = (det.bbox[1] + det.bbox[3]) / 2
                        self.get_logger().info(
                            f"Round-trip test '{det.object_type}': "
                            f"original=({cx_det:.0f},{cy_det:.0f}) → "
                            f"world=({world_pos[0]:.3f},{world_pos[1]:.3f},{world_pos[2]:.3f}) → "
                            f"pixel=({rt_px[0,0]:.0f},{rt_px[0,1]:.0f})"
                        )
                break

        items = []
        for det in all_detections:
            if det.object_type in self._drawer_classes or det.object_type in self._handle_classes:
                continue
            x0, y0, x1, y1 = det.bbox

            bbox_inside = all(
                cv2.pointPolygonTest(hull, (float(px), float(py)), False) >= 0
                for px, py in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            )
            if not bbox_inside:
                continue

            self.get_logger().info(
                f"  item: {det.object_type} ({det.score:.2f}) bbox={det.bbox}"
            )
            items.append({
                "label": det.object_type,
                "confidence": float(det.score),
                "bbox": list(det.bbox),
            })

        self.drawer_items[drawer_id] = items
        self.get_logger().info(
            f"Drawer items: id: {drawer_id} items ({len(items)}): "
            f"{[i['label'] for i in items]}"
        )
        self._save_items_scan_debug_2d(rgb, all_detections, hull, items, drawer_id)

        if not self.keep_drawers_open:
            self._send_close_drawer_command(drawer_id)

    def _send_close_drawer_command(self, drawer_id):
        """Publish a close-drawer command to the robot."""
        interacted = self.interacted_drawers.get(drawer_id)
        if not interacted:
            return

        close_data = {
            "drawer_id": drawer_id,
            "opened_handle": interacted.get("opened_handle"),
            "opened_corners": interacted.get("opened_corners"),
            "handle_orientation": interacted.get("handle_orientation", "horizontal"),
            "pull_distance": interacted.get("pull_distance", 0.0),
            "items": self.drawer_items.get(drawer_id, []),
        }

        if close_data["opened_handle"] is None:
            for g in self.gripper_handle_locations:
                if g["drawer_id"] == drawer_id:
                    close_data["opened_handle"] = g.get("gripper_pos")
                    break

        close_json = json.dumps(close_data)
        close_msg = String()
        close_msg.data = close_json
        self.close_drawer_pub.publish(close_msg)

        if hasattr(self, "_ws_close_drawer_pub") and self._ros_client.is_connected:
            self._ws_close_drawer_pub.publish(
                roslibpy.Message({"data": close_json})
            )

        interacted["status"] = "closing"
        self.get_logger().info(f"Sent close command for drawer {drawer_id}")

    def _detect_drawers_detic(self, rgb: np.ndarray):
        """Use Detic to find drawer and handle bboxes in a single pass.

        Runs Detic once on the full image. Drawers and handles are detected
        together, then handles are associated with drawers by checking which
        handle bbox falls within a drawer bbox. The grasp point is the center
        of the handle's bounding box.

        Returns list of (drawer_bbox, handle_bbox, confidence) tuples.
        Each bbox is [x0, y0, x1, y1].
        """
        results = []

        try:
            from realrobot.detector import VisualDetector, nms_by_type
        except ImportError:
            self.get_logger().warn(
                "Could not import from semantic-object-container-room. "
                "Using fallback detection."
            )
            return results

        if self.detector is None:
            self.detector = VisualDetector(
                device="cpu", score_threshold=self.detection_confidence
            )

        detections = self.detector.detect(rgb, return_crops=False)
        detections = nms_by_type(detections)

        # Log all Detic detections for debugging
        # for d in detections:
        #     self.get_logger().info(
        #         f"Detic: {d.object_type} ({d.detic_class}) "
        #         f"score={d.score:.2f} bbox={d.bbox}"
        #     )

        drawer_dets = [d for d in detections
                       if d.object_type in self._drawer_classes and d.score >= self.detection_confidence]
        drawer_dets = self._cross_class_nms(drawer_dets, iou_threshold=0.3)

        handle_dets = [d for d in detections if d.object_type in self._handle_classes]
        handle_dets = self._cross_class_nms(handle_dets, iou_threshold=0.3)

        self.get_logger().info(
            f"Detic pass: {len(detections)} total, "
            f"{len(drawer_dets)} drawers, {len(handle_dets)} handles"
        )

        paired_handle_indices = set()
        for det in drawer_dets:
            drawer_bbox = det.bbox

            # Associate: find the best handle whose bbox center falls inside this drawer bbox
            handle_bbox, h_idx = self._match_handle_to_drawer(drawer_bbox, handle_dets)
            if handle_bbox is None:
                self.get_logger().info(
                    f"Skipping drawer bbox {drawer_bbox} — no handle detected"
                )
                continue

            paired_handle_indices.add(h_idx)
            self.get_logger().info(
                f"Handle matched to drawer {drawer_bbox}: handle bbox={handle_bbox}"
            )
            results.append((drawer_bbox, handle_bbox, det.score))

        unpaired_handles = [
            handle_dets[i].bbox for i in range(len(handle_dets))
            if i not in paired_handle_indices
        ]

        return results, detections, unpaired_handles

    @staticmethod
    def _cross_class_nms(detections, iou_threshold=0.3):
        """Suppress overlapping bboxes across all classes, keeping higher confidence."""
        if not detections:
            return detections
        sorted_dets = sorted(detections, key=lambda d: d.score, reverse=True)
        keep = []
        for det in sorted_dets:
            suppressed = False
            for kept in keep:
                iou = DrawerDetectionNode._bbox_iou(det.bbox, kept.bbox)
                if iou > iou_threshold:
                    suppressed = True
                    break
            if not suppressed:
                keep.append(det)
        return keep

    @staticmethod
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

    @staticmethod
    def _match_handle_to_drawer(drawer_bbox, handle_dets):
        """Find the highest-confidence handle whose bbox center falls inside the drawer bbox.

        Returns (handle_bbox, index) or (None, None).
        """
        dx0, dy0, dx1, dy1 = drawer_bbox
        best_handle = None
        best_score = 0.0
        best_idx = None

        for i, det in enumerate(handle_dets):
            hx0, hy0, hx1, hy1 = det.bbox
            hcx = (hx0 + hx1) / 2
            hcy = (hy0 + hy1) / 2
            if dx0 <= hcx <= dx1 and dy0 <= hcy <= dy1 and det.score > best_score:
                best_handle = det.bbox
                best_score = det.score
                best_idx = i

        return best_handle, best_idx

    # ─── Projection and geometry ──────────────────────────────────────

    @staticmethod
    def _depth_to_meters(depth: np.ndarray) -> np.ndarray:
        """Convert depth to float32 meters regardless of source format."""
        if depth.dtype == np.uint16:
            return depth.astype(np.float32) / 1000.0
        return depth.astype(np.float32)

    def _project_to_world(
        self, bbox, depth, camera_pose, camera_K, max_depth=5.0
    ):
        """Project bbox center to 3D world coordinates using depth."""
        depth_m = self._depth_to_meters(depth)

        if self.use_sim:
            try:
                from realrobot.stretch.projection import project_bbox_to_world_se3
                result = project_bbox_to_world_se3(
                    bbox, depth_m,
                    camera_pose, camera_K, max_depth=max_depth
                )
                return result
            except ImportError:
                pass

        x0, y0, x1, y1 = bbox
        h, w = depth_m.shape[:2]

        u = int((x0 + x1) / 2)
        v = int((y0 + y1) / 2)
        u = max(0, min(u, w - 1))
        v = max(0, min(v, h - 1))

        region = depth_m[max(0, v-3):v+3, max(0, u-3):u+3]
        valid = region[region > 0.1]
        if len(valid) == 0:
            return None

        d = float(np.median(valid))
        if d > max_depth:
            return None

        fx, fy = camera_K[0, 0], camera_K[1, 1]
        cx, cy = camera_K[0, 2], camera_K[1, 2]

        x_cam = (u - cx) / fx * d
        y_cam = (v - cy) / fy * d
        z_cam = d

        if not self.use_sim:
            p_cam = np.array([y_cam, -x_cam, z_cam, 1.0])
        else:
            p_cam = np.array([x_cam, y_cam, z_cam, 1.0])
        p_world = camera_pose @ p_cam
        return p_world[:3]

    def _project_to_world_wide(
        self, bbox, depth_m, camera_pose, camera_K, max_depth=5.0, radius=15
    ):
        """Like _project_to_world but with wider depth sampling for small objects.

        Accepts depth already in meters. Uses a larger sampling radius to
        borrow depth from surrounding pixels (e.g. the drawer bottom around
        a small object).
        """
        if self.use_sim:
            try:
                from realrobot.stretch.projection import project_bbox_to_world_se3
                return project_bbox_to_world_se3(
                    bbox, depth_m, camera_pose, camera_K, max_depth=max_depth
                )
            except ImportError:
                pass

        x0, y0, x1, y1 = bbox
        h, w = depth_m.shape[:2]

        u = int((x0 + x1) / 2)
        v = int((y0 + y1) / 2)
        u = max(0, min(u, w - 1))
        v = max(0, min(v, h - 1))

        region = depth_m[max(0, v - radius):v + radius, max(0, u - radius):u + radius]
        valid = region[region > 0.1]
        if len(valid) == 0:
            return None

        d = float(np.median(valid))
        if d > max_depth:
            return None

        fx, fy = camera_K[0, 0], camera_K[1, 1]
        cx, cy = camera_K[0, 2], camera_K[1, 2]

        x_cam = (u - cx) / fx * d
        y_cam = (v - cy) / fy * d
        z_cam = d

        if not self.use_sim:
            p_cam = np.array([y_cam, -x_cam, z_cam, 1.0])
        else:
            p_cam = np.array([x_cam, y_cam, z_cam, 1.0])
        p_world = camera_pose @ p_cam
        return p_world[:3]

    def _world_to_pixels(
        self, pts_world: np.ndarray, camera_pose: np.ndarray, camera_K: np.ndarray
    ) -> np.ndarray | None:
        """Project Nx3 world points to Nx2 pixel coordinates."""
        cam_from_world = np.linalg.inv(camera_pose)
        ones = np.ones((pts_world.shape[0], 1))
        pts_cam = (cam_from_world @ np.hstack([pts_world, ones]).T).T[:, :3]

        X, Y, Z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
        if np.any(Z <= 0):
            return None

        fx, fy = camera_K[0, 0], camera_K[1, 1]
        cx, cy = camera_K[0, 2], camera_K[1, 2]

        if not self.use_sim:
            u = fx * (-Y) / Z + cx
            v = fy * X / Z + cy
        else:
            u = fx * X / Z + cx
            v = fy * Y / Z + cy

        return np.stack([u, v], axis=1)

    def _project_drawer_corners(
        self, drawer_bbox: list, depth: np.ndarray,
        camera_pose: np.ndarray, camera_K: np.ndarray,
    ):
        """Project the 4 corners of a drawer bbox to world coordinates.

        Each corner is projected at its own depth sampled from a small
        region around that corner pixel. This produces an accurate
        rectangle even when the camera views the drawer at an angle.
        """
        depth_m = self._depth_to_meters(depth)
        x0, y0, x1, y1 = [int(c) for c in drawer_bbox]
        h, w = depth_m.shape[:2]

        # Shrink bbox by 15% on each side to avoid sampling wall/background
        margin_x = int((x1 - x0) * 0.15)
        margin_y = int((y1 - y0) * 0.15)
        x0 += margin_x
        y0 += margin_y
        x1 -= margin_x
        y1 -= margin_y

        fx, fy = camera_K[0, 0], camera_K[1, 1]
        cx, cy = camera_K[0, 2], camera_K[1, 2]

        corners_px = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        corners_world = []

        for u, v in corners_px:
            uc = max(0, min(u, w - 1))
            vc = max(0, min(v, h - 1))

            d = self._sample_depth_at(depth_m, uc, vc)
            if d is None:
                return None

            x_cam = (uc - cx) / fx * d
            y_cam = (vc - cy) / fy * d
            z_cam = d
            if not self.use_sim:
                p_cam = np.array([y_cam, -x_cam, z_cam, 1.0])
            else:
                p_cam = np.array([x_cam, y_cam, z_cam, 1.0])
            p_world = camera_pose @ p_cam
            corners_world.append(p_world[:3])

        return corners_world

    @staticmethod
    def _sample_depth_at(depth: np.ndarray, u: int, v: int, radius: int = 5):
        """Sample median depth in a small region around (u, v). Expects meters."""
        h, w = depth.shape[:2]
        r = radius
        region = depth[max(0, v - r):min(h, v + r), max(0, u - r):min(w, u + r)]
        valid = region[region > 0.1]
        if len(valid) == 0:
            return None
        return float(np.median(valid))

    def _transform_to_matrix(self, transform: TransformStamped) -> np.ndarray:
        """Convert a TF TransformStamped to a 4x4 SE(3) matrix."""
        from tf_transformations import quaternion_matrix

        t = transform.transform.translation
        q = transform.transform.rotation
        mat = quaternion_matrix([q.x, q.y, q.z, q.w])
        mat[0, 3] = t.x
        mat[1, 3] = t.y
        mat[2, 3] = t.z
        return mat

    def _check_reachability(self, world_pos: np.ndarray) -> bool:
        """Determine if the Stretch3 gripper can reach this position."""
        z = world_pos[2]
        if z < self.min_reach_height or z > self.max_reach_height:
            return False
        return True

    # ─── De-duplication ───────────────────────────────────────────────

    def _try_merge_detection(self, world_pos: np.ndarray, confidence: float,
                             drawer_corners=None) -> bool:
        """Check if this detection matches an existing or interacted drawer.

        If the new detection has higher confidence, its position and
        bounding box replace the existing one entirely.
        Returns True if merged (i.e., it's a duplicate or already interacted).
        """
        with self.drawers_lock:
            for existing in self.drawers:
                if existing.handle_center_world is None:
                    continue
                dist = np.linalg.norm(
                    world_pos - np.array(existing.handle_center_world)
                )
                if dist < self.dedup_distance:
                    existing.observations += 1
                    if confidence > existing.confidence:
                        existing.handle_center_world = world_pos
                        existing.handle_grasp_world = world_pos.copy()
                        existing.confidence = confidence
                        if drawer_corners is not None:
                            existing.drawer_corners_world = drawer_corners
                    return True

        for info in self.interacted_drawers.values():
            ch = info.get("closed_handle")
            if ch:
                closed_pos = np.array([ch["x"], ch["y"], ch["z"]])
                if np.linalg.norm(world_pos - closed_pos) < self.dedup_distance:
                    return True
            oh = info.get("opened_handle")
            if oh:
                opened_pos = np.array([oh["x"], oh["y"], oh["z"]])
                if np.linalg.norm(world_pos - opened_pos) < self.dedup_distance:
                    return True

        return False

    def _update_distances(self):
        """Update distance_to_robot for all drawers."""
        try:
            transform = self.tf_buffer.lookup_transform(
                "odom", "base_link",
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
            robot_x = transform.transform.translation.x
            robot_y = transform.transform.translation.y
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
            return

        with self.drawers_lock:
            for d in self.drawers:
                if d.handle_center_world is not None:
                    dx = d.handle_center_world[0] - robot_x
                    dy = d.handle_center_world[1] - robot_y
                    d.distance_to_robot = math.sqrt(dx * dx + dy * dy)

    # ─── LOCUS GNN ranking stub ───────────────────────────────────────

    def _rank_via_locus_stub(self):
        """Stub for calling the GNN node ranking logic.

        When implemented, this should call into
        semantic-object-container-room/gnn/ with the following data:
          - All drawer world positions (handle_center_world)
          - CLIP embeddings of each drawer crop
          - Scene graph context (room type, nearby objects)
          - Spatial relationships between containers
          - Room layout / voxel occupancy context

        The GNN would return a ranking score for each drawer based on
        how likely it is to contain the target object.
        """
        # TODO: Implement GNN ranking via semantic-object-container-room
        # Required data for container node ranking:
        #   1. drawer CLIP embedding (from annotated_image crop)
        #   2. drawer world position (handle_center_world)
        #   3. nearby object types and positions (scene graph nodes)
        #   4. room type string
        #   5. spatial edges (distance-based) between all containers
        #   6. text embedding of target query object
        #
        # Call: from realrobot.inference import score_containers
        #       scores = score_containers(graph, query_embedding, model)
        #       for drawer, score in zip(self.drawers, scores):
        #           drawer.ranking = score
        pass

    def _save_projection_debug(self, rgb, depth, drawer_bbox, handle_bbox):
        debug_dir = Path("/tmp/detic_debug")
        debug_dir.mkdir(exist_ok=True)
        stamp = int(time.time() * 1000) % 1000000

        # RGB with bounding boxes
        vis = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
        cv2.rectangle(vis, (drawer_bbox[0], drawer_bbox[1]),
                       (drawer_bbox[2], drawer_bbox[3]), (255, 0, 0), 2)
        cv2.rectangle(vis, (handle_bbox[0], handle_bbox[1]),
                       (handle_bbox[2], handle_bbox[3]), (0, 255, 0), 2)
        hcx = (handle_bbox[0] + handle_bbox[2]) // 2
        hcy = (handle_bbox[1] + handle_bbox[3]) // 2
        cv2.circle(vis, (hcx, hcy), 5, (0, 0, 255), -1)
        depth_val = depth[min(hcy, depth.shape[0]-1), min(hcx, depth.shape[1]-1)]
        cv2.putText(vis, f"d={depth_val}", (hcx+10, hcy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imwrite(str(debug_dir / f"proj_rgb_{stamp}.jpg"), vis)

        # Depth as normalized grayscale
        depth_vis = depth.copy().astype(np.float32)
        depth_vis[depth_vis == 0] = np.nan
        dmin = np.nanmin(depth_vis) if np.any(~np.isnan(depth_vis)) else 0
        dmax = np.nanmax(depth_vis) if np.any(~np.isnan(depth_vis)) else 1
        depth_norm = ((depth_vis - dmin) / max(dmax - dmin, 1) * 255)
        depth_norm = np.nan_to_num(depth_norm, 0).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        cv2.rectangle(depth_color, (handle_bbox[0], handle_bbox[1]),
                       (handle_bbox[2], handle_bbox[3]), (0, 255, 0), 2)
        cv2.circle(depth_color, (hcx, hcy), 5, (255, 255, 255), -1)
        cv2.imwrite(str(debug_dir / f"proj_depth_{stamp}.jpg"), depth_color)

        self.get_logger().info(f"Projection debug saved to {debug_dir}/proj_*_{stamp}.jpg")

    # ─── Annotation and visualization ─────────────────────────────────

    def _annotate_image(
        self, rgb: np.ndarray, drawer_bbox: list, handle_bbox: list
    ) -> np.ndarray:
        """Draw drawer and handle bounding boxes on the image."""
        annotated = rgb.copy()
        # Drawer bbox in blue
        cv2.rectangle(
            annotated,
            (drawer_bbox[0], drawer_bbox[1]),
            (drawer_bbox[2], drawer_bbox[3]),
            (0, 0, 255), 2,
        )
        # Handle bbox in green
        cv2.rectangle(
            annotated,
            (handle_bbox[0], handle_bbox[1]),
            (handle_bbox[2], handle_bbox[3]),
            (0, 255, 0), 2,
        )
        # Handle center dot
        hcx = (handle_bbox[0] + handle_bbox[2]) // 2
        hcy = (handle_bbox[1] + handle_bbox[3]) // 2
        cv2.circle(annotated, (hcx, hcy), 5, (255, 0, 0), -1)
        return annotated

    def _save_debug_image(self, rgb, all_detections, matched_results):
        """Save a debug image showing all Detic detections to /tmp/detic_debug/."""
        debug_dir = Path("/tmp/detic_debug")
        debug_dir.mkdir(exist_ok=True)

        debug_img = rgb.copy()

        # Draw all Detic detections in gray with class labels
        for det in all_detections:
            x0, y0, x1, y1 = det.bbox
            cv2.rectangle(debug_img, (x0, y0), (x1, y1), (180, 180, 180), 1)
            label = f"{det.object_type} {det.score:.2f}"
            cv2.putText(debug_img, label, (x0, max(y0 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        # Draw matched drawer (blue) and handle (green) bboxes on top
        for drawer_bbox, handle_bbox, _conf in matched_results:
            cv2.rectangle(debug_img, (drawer_bbox[0], drawer_bbox[1]),
                          (drawer_bbox[2], drawer_bbox[3]), (255, 0, 0), 2)
            cv2.rectangle(debug_img, (handle_bbox[0], handle_bbox[1]),
                          (handle_bbox[2], handle_bbox[3]), (0, 255, 0), 2)
            hcx = (handle_bbox[0] + handle_bbox[2]) // 2
            hcy = (handle_bbox[1] + handle_bbox[3]) // 2
            cv2.circle(debug_img, (hcx, hcy), 5, (0, 0, 255), -1)

        stamp = int(time.time() * 1000) % 1000000
        path = debug_dir / f"detic_{stamp}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
        self.get_logger().info(f"Debug image saved: {path}")

    def _save_items_scan_debug_2d(self, rgb, all_detections, hull, items, drawer_id):
        """Save debug image showing 2D convex hull and detection containment."""
        debug_dir = Path("/tmp/detic_debug")
        debug_dir.mkdir(exist_ok=True)
        stamp = int(time.time() * 1000) % 1000000

        debug_img = rgb.copy()
        item_bboxes = {tuple(i["bbox"]) for i in items}

        cv2.polylines(debug_img, [hull.astype(np.int32)], True, (0, 255, 255), 2)

        for det in all_detections:
            x0, y0, x1, y1 = det.bbox
            if det.object_type in self._drawer_classes or det.object_type in self._handle_classes:
                color = (100, 100, 100)
            elif tuple(det.bbox) in item_bboxes:
                color = (0, 255, 0)
            else:
                color = (200, 200, 0)
            cv2.rectangle(debug_img, (x0, y0), (x1, y1), color, 2)
            label = f"{det.object_type} {det.score:.2f}"
            cv2.putText(debug_img, label, (x0, max(y0 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        path = debug_dir / f"items_2d_{drawer_id}_{stamp}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
        self.get_logger().info(f"Items scan 2D debug saved: {path}")

    def publish_markers(self):
        """Publish drawer markers in RViz.

        - Green/red rectangle outline around drawer bbox (green=reachable, red=not)
        - Blue sphere at handle grasp point
        - White text label with ID, reachability, orientation, distance
        """
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()

        with self.drawers_lock:
            for i, drawer in enumerate(self.drawers):
                if drawer.handle_center_world is None:
                    continue

                header = Header(frame_id="odom", stamp=now)

                # Drawer bounding box as 2D rectangle outline
                box_marker = Marker()
                box_marker.header = header
                box_marker.ns = "drawers"
                box_marker.id = i
                box_marker.type = Marker.LINE_STRIP
                box_marker.action = Marker.ADD
                box_marker.scale.x = 0.01

                if drawer.reachable:
                    box_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.9)
                else:
                    box_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.9)

                if drawer.drawer_corners_world is None or len(drawer.drawer_corners_world) != 4:
                    continue
                for c in drawer.drawer_corners_world:
                    box_marker.points.append(
                        Point(x=float(c[0]), y=float(c[1]), z=float(c[2]))
                    )
                box_marker.points.append(
                    Point(
                        x=float(drawer.drawer_corners_world[0][0]),
                        y=float(drawer.drawer_corners_world[0][1]),
                        z=float(drawer.drawer_corners_world[0][2]),
                    )
                )

                box_marker.pose.orientation.w = 1.0
                marker_array.markers.append(box_marker)

                # Handle grasp point sphere
                handle_marker = Marker()
                handle_marker.header = header
                handle_marker.ns = "handle_grasp"
                handle_marker.id = i
                handle_marker.type = Marker.SPHERE
                handle_marker.action = Marker.ADD
                handle_marker.pose.position.x = float(drawer.handle_center_world[0])
                handle_marker.pose.position.y = float(drawer.handle_center_world[1])
                handle_marker.pose.position.z = float(drawer.handle_center_world[2])
                handle_marker.pose.orientation.w = 1.0
                handle_marker.scale.x = 0.04
                handle_marker.scale.y = 0.04
                handle_marker.scale.z = 0.04
                handle_marker.color = ColorRGBA(r=0.0, g=0.3, b=1.0, a=1.0)
                marker_array.markers.append(handle_marker)

                # Text label
                text_marker = Marker()
                text_marker.header = header
                text_marker.ns = "drawer_labels"
                text_marker.id = i
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.action = Marker.ADD
                text_marker.pose.position.x = float(drawer.handle_center_world[0])
                text_marker.pose.position.y = float(drawer.handle_center_world[1])
                text_marker.pose.position.z = float(drawer.handle_center_world[2]) + 0.2
                text_marker.scale.z = 0.08
                text_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
                text_marker.text = (
                    f"{drawer.drawer_id} "
                    f"({'R' if drawer.reachable else 'X'}) "
                    f"{drawer.handle_orientation[0].upper()} "
                    f"{drawer.distance_to_robot:.1f}m"
                )
                marker_array.markers.append(text_marker)

        # Interacted drawers in yellow (both closed and opened positions)
        for j, (did, info) in enumerate(self.interacted_drawers.items()):
            od_header = Header(frame_id="odom", stamp=now)
            marker_id_base = 1000 + j * 10
            status = info.get("status", "opened")
            label_suffix = status.upper()

            # Closed position bbox
            closed_corners = info.get("closed_corners")
            if closed_corners and len(closed_corners) == 4:
                cc_marker = Marker()
                cc_marker.header = od_header
                cc_marker.ns = "interacted_closed"
                cc_marker.id = marker_id_base
                cc_marker.type = Marker.LINE_STRIP
                cc_marker.action = Marker.ADD
                cc_marker.scale.x = 0.01
                cc_marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.5)
                cc_marker.pose.orientation.w = 1.0
                for c in closed_corners:
                    cc_marker.points.append(
                        Point(x=float(c["x"]), y=float(c["y"]), z=float(c["z"]))
                    )
                cc_marker.points.append(
                    Point(x=float(closed_corners[0]["x"]),
                          y=float(closed_corners[0]["y"]),
                          z=float(closed_corners[0]["z"]))
                )
                marker_array.markers.append(cc_marker)

            # Opened position bbox (if resolved)
            opened_corners = info.get("opened_corners")
            if opened_corners and len(opened_corners) == 4:
                ob_marker = Marker()
                ob_marker.header = od_header
                ob_marker.ns = "interacted_opened"
                ob_marker.id = marker_id_base + 1
                ob_marker.type = Marker.LINE_STRIP
                ob_marker.action = Marker.ADD
                ob_marker.scale.x = 0.01
                ob_marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.9)
                ob_marker.pose.orientation.w = 1.0
                for c in opened_corners:
                    ob_marker.points.append(
                        Point(x=float(c["x"]), y=float(c["y"]), z=float(c["z"]))
                    )
                ob_marker.points.append(
                    Point(x=float(opened_corners[0]["x"]),
                          y=float(opened_corners[0]["y"]),
                          z=float(opened_corners[0]["z"]))
                )
                marker_array.markers.append(ob_marker)

            # Handle sphere at opened or closed position
            handle_pos = info.get("opened_handle") or info.get("closed_handle")
            if handle_pos:
                oh_marker = Marker()
                oh_marker.header = od_header
                oh_marker.ns = "interacted_handles"
                oh_marker.id = marker_id_base + 2
                oh_marker.type = Marker.SPHERE
                oh_marker.action = Marker.ADD
                oh_marker.pose.position.x = float(handle_pos["x"])
                oh_marker.pose.position.y = float(handle_pos["y"])
                oh_marker.pose.position.z = float(handle_pos["z"])
                oh_marker.pose.orientation.w = 1.0
                oh_marker.scale.x = 0.04
                oh_marker.scale.y = 0.04
                oh_marker.scale.z = 0.04
                oh_marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
                marker_array.markers.append(oh_marker)

            # Label
            label_pos = info.get("opened_handle") or info.get("closed_handle") or {}
            od_label = Marker()
            od_label.header = od_header
            od_label.ns = "interacted_labels"
            od_label.id = marker_id_base + 3
            od_label.type = Marker.TEXT_VIEW_FACING
            od_label.action = Marker.ADD
            od_label.pose.position.x = float(label_pos.get("x", 0))
            od_label.pose.position.y = float(label_pos.get("y", 0))
            od_label.pose.position.z = float(label_pos.get("z", 0)) + 0.2
            od_label.scale.z = 0.08
            od_label.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
            od_label.text = f"{did} {label_suffix}"
            od_label.pose.orientation.w = 1.0
            marker_array.markers.append(od_label)

        self.marker_pub.publish(marker_array)

        if self.chosen_drawer_id is not None:
            with self.drawers_lock:
                chosen = next(
                    (d for d in self.drawers if d.drawer_id == self.chosen_drawer_id),
                    None,
                )
            if chosen is not None:
                self._publish_chosen_marker(chosen)


def main(args=None):
    rclpy.init(args=args)
    node = DrawerDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
