#!/usr/bin/env python3
"""
Search for Drawers: ROS2 node that explores the environment, runs Detic
(via semantic-object-container-room submodule's VisualDetector) on both cameras
to detect drawers and handles, back-projects to world (odom) coordinates,
deduplicates using InstanceTracker, checks reachability, and publishes a list
of discovered drawers.

Detection pipeline (from semantic-object-container-room):
  1. DETIC detects drawers in RGB frames (VisualDetector)
  2. Detections projected to 3D via depth + TF2 camera pose
  3. InstanceTracker deduplicates across frames with drawer-aware radii
  4. Per-type NMS removes redundant detections per frame

Two exploration modes (set via 'exploration_mode' ROS2 parameter):
  - 'wall_following' (default): lidar-based wall following around the room perimeter
  - 'frontier': funmap's built-in frontier exploration (drive-to-scan + head-scan)

Subscribes to:
  /camera/color/image_raw (head camera RGB)
  /camera/depth/image_rect_raw (head camera depth)
  /camera/color/camera_info (head camera intrinsics)
  /gripper_camera/image_raw (wrist camera RGB)
  /gripper_camera/depth/image_rect_raw (wrist camera depth)
  /gripper_camera/camera_info (wrist camera intrinsics)
  /scan (lidar — wall_following mode only)

Publishes:
  /search_drawers/discovered_drawers (visualization_msgs/MarkerArray)
  /search_drawers/detection_image (sensor_msgs/Image)

Services provided:
  /search_drawers/get_drawers (std_srvs/srv/Trigger) — returns JSON of all drawers

Services called (frontier mode only):
  /funmap/trigger_head_scan (std_srvs/srv/Trigger)
  /funmap/trigger_drive_to_scan (std_srvs/srv/Trigger)
"""

import json
import math
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import rclpy.logging
from cv_bridge import CvBridge
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, CameraInfo, JointState, LaserScan
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from std_srvs.srv import Trigger

import hello_helpers.hello_misc as hm

# --- Import from semantic-object-container-room (installed as editable package) ---
from realrobot.detector import VisualDetector, nms_by_type, draw_detections, Detection
from realrobot.perception import InstanceTracker

# --- Configuration ---

DETIC_ROOT = Path(os.environ.get('DETIC_ROOT', os.path.expanduser('~/Detic')))

# Camera topics
HEAD_COLOR_TOPIC = '/camera/color/image_raw'
HEAD_DEPTH_TOPIC = '/camera/depth/image_rect_raw'
HEAD_INFO_TOPIC = '/camera/color/camera_info'
HEAD_OPTICAL_FRAME = 'camera_color_optical_frame'

WRIST_COLOR_TOPIC = '/gripper_camera/image_raw'
WRIST_DEPTH_TOPIC = '/gripper_camera/depth/image_rect_raw'
WRIST_INFO_TOPIC = '/gripper_camera/camera_info'
WRIST_OPTICAL_FRAME = 'gripper_camera_color_optical_frame'

# Head scan positions
HEAD_PAN_POSITIONS = [-1.2, -0.8, -0.4, 0.0, 0.4, 0.8]
HEAD_TILT_SEARCH = -0.5

# Reachability workspace limits (Stretch3 lateral arm)
EEF_HEIGHT_MIN = 0.2
EEF_HEIGHT_MAX = 1.3
MAX_LATERAL_REACH = 0.93
MIN_LATERAL_REACH = 0.45

# Wall-following parameters
WALL_FOLLOW_DISTANCE_M = 1.0
FORWARD_SPEED_M = 0.8
MIN_STEPS_BEFORE_LOOP_CHECK = 20

# Frontier parameters
MAX_FRONTIER_FAILURES = 3

# Detic class names of interest (matched against Detection.detic_class)
DRAWER_CLASS_NAMES = {'drawer', 'cabinet', 'chest_of_drawers', 'filing_cabinet'}
HANDLE_CLASS_NAMES = {'handle', 'knob', 'doorknob', 'door_handle', 'pull'}

# InstanceTracker dedup radii (meters) — drawer-specific
DRAWER_DEDUP_RADII = {
    'Drawer': {'horizontal': 0.3, 'vertical': 0.08},
    'Cabinet': {'horizontal': 0.4, 'vertical': 0.15},
    'default': {'horizontal': 0.5, 'vertical': 0.2},
}


class DiscoveredDrawer:
    """A drawer registered in the odom (world) frame."""

    def __init__(self, handle_world_xyz, rgb_image, drawer_bbox_px, handle_bbox_px,
                 handle_orientation, reachable):
        self.handle_world_xyz = handle_world_xyz  # np.array(3,) in odom frame
        self.rgb_image = rgb_image
        self.drawer_bbox_px = drawer_bbox_px
        self.handle_bbox_px = handle_bbox_px
        self.handle_orientation = handle_orientation  # 'vertical' or 'horizontal'
        self.reachable = reachable


def _is_drawer_class(detic_class: str) -> bool:
    """Check if a Detic class name is a drawer/cabinet type."""
    return detic_class.lower().replace('-', '_') in DRAWER_CLASS_NAMES


def _is_handle_class(detic_class: str) -> bool:
    """Check if a Detic class name is a handle type."""
    return detic_class.lower().replace('-', '_') in HANDLE_CLASS_NAMES


class SearchForDrawersNode(hm.HelloNode):

    def __init__(self):
        hm.HelloNode.__init__(self)
        self.rate = 10.0
        self.bridge = CvBridge()
        self.callback_group = None

        # Camera data
        self.head_color = None
        self.head_depth = None
        self.head_info = None
        self.wrist_color = None
        self.wrist_depth = None
        self.wrist_info = None
        self.image_lock = threading.Lock()

        # Lidar (wall_following mode)
        self.latest_scan = None
        self.scan_lock = threading.Lock()

        # Joint state
        self.joint_states = None
        self.joint_states_lock = threading.Lock()

        # Discovered drawers (all in odom frame for global deduplication)
        self.discovered_drawers = []
        self.drawers_lock = threading.Lock()

        # Detection via submodule (VisualDetector + InstanceTracker)
        self.detector = None
        self.instance_tracker = InstanceTracker(dedup_radii=DRAWER_DEDUP_RADII)

        # Funmap service clients (frontier mode)
        self.head_scan_client = None
        self.drive_to_scan_client = None

        # Exploration state
        self.exploration_complete = False
        self.exploration_mode = 'wall_following'

    # --- Callbacks ---

    def joint_states_callback(self, msg):
        with self.joint_states_lock:
            self.joint_states = msg

    def head_color_cb(self, msg):
        with self.image_lock:
            self.head_color = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def head_depth_cb(self, msg):
        with self.image_lock:
            self.head_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')

    def head_info_cb(self, msg):
        self.head_info = msg

    def wrist_color_cb(self, msg):
        with self.image_lock:
            self.wrist_color = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def wrist_depth_cb(self, msg):
        with self.image_lock:
            self.wrist_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')

    def wrist_info_cb(self, msg):
        self.wrist_info = msg

    def scan_cb(self, msg):
        with self.scan_lock:
            self.latest_scan = msg

    # --- Detic Detection (via submodule VisualDetector) ---

    def run_detic(self, rgb_image):
        """
        Run Detic on an RGB image via VisualDetector from
        semantic-object-container-room. Filters for drawer + handle classes,
        applies per-type NMS, and matches handles to their parent drawers.

        Returns list of (drawer_Detection, handle_center_px_or_None) tuples.
        """
        # VisualDetector expects RGB; cv_bridge delivers BGR
        if rgb_image.shape[2] == 3:
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        all_detections = self.detector.detect(rgb_image, return_crops=False)
        all_detections = nms_by_type(all_detections)

        drawer_dets = [d for d in all_detections if _is_drawer_class(d.detic_class)]
        handle_dets = [d for d in all_detections if _is_handle_class(d.detic_class)]

        results = []
        for drawer in drawer_dets:
            db = drawer.bbox
            matched_handle_center = None
            best_handle_score = 0.0

            matched_handle_bbox = None
            for handle in handle_dets:
                hx = (handle.bbox[0] + handle.bbox[2]) / 2.0
                hy = (handle.bbox[1] + handle.bbox[3]) / 2.0
                if db[0] <= hx <= db[2] and db[1] <= hy <= db[3]:
                    if handle.score > best_handle_score:
                        best_handle_score = handle.score
                        matched_handle_center = (hx, hy)
                        matched_handle_bbox = handle.bbox

            results.append((drawer, matched_handle_center, matched_handle_bbox))

        results.sort(key=lambda pair: pair[0].score, reverse=True)
        return results

    # --- 3D Geometry (all transforms go to odom frame) ---

    def pixel_to_odom(self, px, py, depth_image, camera_info, optical_frame):
        """
        Back-project a single pixel to the odom (world) frame.
        All drawer positions are stored in odom for consistent global deduplication.
        Returns np.array(3,) or None.
        """
        if camera_info is None or depth_image is None:
            return None

        fx = camera_info.k[0]
        fy = camera_info.k[4]
        cx = camera_info.k[2]
        cy = camera_info.k[5]
        if fx == 0 or fy == 0:
            return None

        px_int, py_int = int(round(px)), int(round(py))
        h, w = depth_image.shape[:2]
        if px_int < 0 or px_int >= w or py_int < 0 or py_int >= h:
            return None

        radius = 5
        y_min = max(0, py_int - radius)
        y_max = min(h, py_int + radius + 1)
        x_min = max(0, px_int - radius)
        x_max = min(w, px_int + radius + 1)
        depth_patch = depth_image[y_min:y_max, x_min:x_max]
        valid_depths = depth_patch[
            (depth_patch > 0.1) & (depth_patch < 5.0) & np.isfinite(depth_patch)]

        if len(valid_depths) == 0:
            return None

        depth = float(np.median(valid_depths))

        x_cam = (px - cx) * depth / fx
        y_cam = (py - cy) * depth / fy
        z_cam = depth

        cam_to_odom, _ = hm.get_p1_to_p2_matrix(
            optical_frame, 'odom', self.tf2_buffer, timeout_s=1.0)
        if cam_to_odom is None:
            return None

        pt_cam = np.array([x_cam, y_cam, z_cam, 1.0])
        pt_odom = (cam_to_odom @ pt_cam)[:3]

        if pt_odom[2] < 0.0 or pt_odom[2] > 3.0:
            return None

        return pt_odom

    # --- Global Deduplication (via submodule InstanceTracker) ---

    def track_detection(self, obj_type, world_xyz):
        """
        Register a detection with InstanceTracker. Returns (instance_id, is_new).
        The tracker uses drawer-aware dedup radii (separate horizontal/vertical
        thresholds) to preserve stacked drawers while merging duplicate sightings.
        """
        n_before = len(self.instance_tracker.instances)
        instance_id = self.instance_tracker.update(
            obj_type=obj_type,
            position=world_xyz,
        )
        is_new = len(self.instance_tracker.instances) > n_before
        return instance_id, is_new

    # --- Reachability ---

    def is_reachable(self, world_xyz):
        """
        Check if a point in odom frame is within the Stretch3 end-effector workspace.
        """
        z = world_xyz[2]
        if z < EEF_HEIGHT_MIN or z > EEF_HEIGHT_MAX:
            return False

        base_to_odom, _ = hm.get_p1_to_p2_matrix(
            'base_link', 'odom', self.tf2_buffer, timeout_s=1.0)
        if base_to_odom is None:
            return False

        robot_pos = base_to_odom[:3, 3]
        dx = world_xyz[0] - robot_pos[0]
        dy = world_xyz[1] - robot_pos[1]
        lateral_dist = math.sqrt(dx * dx + dy * dy)

        return MIN_LATERAL_REACH <= lateral_dist <= MAX_LATERAL_REACH

    # --- Detection + Registration Pipeline ---

    def process_camera(self, color, depth, info, optical_frame, camera_name):
        """
        Run Detic on a camera frame via VisualDetector. For each detected drawer,
        back-project the handle (or drawer center) to the odom frame, deduplicate
        via InstanceTracker, and register new drawers.
        """
        if color is None or depth is None or info is None:
            return

        detections = self.run_detic(color)
        if not detections:
            return

        for drawer_det, handle_center_px, handle_bbox in detections:
            if handle_center_px is not None:
                px, py = handle_center_px
            else:
                px = (drawer_det.bbox[0] + drawer_det.bbox[2]) / 2.0
                py = (drawer_det.bbox[1] + drawer_det.bbox[3]) / 2.0

            world_xyz = self.pixel_to_odom(px, py, depth, info, optical_frame)
            if world_xyz is None:
                self.logger.info(f'[{camera_name}] Could not back-project detection to odom frame')
                continue

            instance_id, is_new = self.track_detection(drawer_det.object_type, world_xyz)
            if not is_new:
                self.logger.info(
                    f'[{camera_name}] Drawer at odom ({world_xyz[0]:.2f}, '
                    f'{world_xyz[1]:.2f}, {world_xyz[2]:.2f}) merged into {instance_id}')
                continue

            reachable = self.is_reachable(world_xyz)

            # Determine handle orientation from bbox aspect ratio
            if handle_bbox is not None:
                handle_w = handle_bbox[2] - handle_bbox[0]
                handle_h = handle_bbox[3] - handle_bbox[1]
                handle_orientation = 'vertical' if handle_h > handle_w else 'horizontal'
            else:
                handle_orientation = 'horizontal'

            annotated = draw_detections(color, [drawer_det])
            annotated_np = np.array(annotated)
            if handle_center_px is not None:
                hx, hy = int(handle_center_px[0]), int(handle_center_px[1])
                cv2.circle(annotated_np, (hx, hy), 8, (255, 0, 255), 2)
                cv2.putText(annotated_np, f'handle ({handle_orientation})', (hx + 10, hy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

            drawer = DiscoveredDrawer(
                handle_world_xyz=world_xyz,
                rgb_image=annotated_np,
                drawer_bbox_px=tuple(drawer_det.bbox),
                handle_bbox_px=handle_center_px,
                handle_orientation=handle_orientation,
                reachable=reachable,
            )

            with self.drawers_lock:
                self.discovered_drawers.append(drawer)
                idx = len(self.discovered_drawers)

            self.logger.info(
                f'[{camera_name}] NEW drawer #{idx} ({instance_id}) at odom '
                f'({world_xyz[0]:.2f}, {world_xyz[1]:.2f}, {world_xyz[2]:.2f}) '
                f'reachable={reachable}')

            try:
                det_msg = self.bridge.cv2_to_imgmsg(annotated_np, encoding='bgr8')
                self.detection_pub.publish(det_msg)
            except Exception:
                pass

        self.publish_drawer_markers()

    # --- Camera Scanning (shared by both modes) ---

    def scan_with_cameras(self):
        """Pan head camera across positions and check both cameras for drawers."""
        self.move_to_pose({'joint_head_tilt': HEAD_TILT_SEARCH})
        time.sleep(0.3)

        for pan_angle in HEAD_PAN_POSITIONS:
            self.move_to_pose({'joint_head_pan': pan_angle})
            time.sleep(0.8)

            with self.image_lock:
                h_color = self.head_color.copy() if self.head_color is not None else None
                h_depth = self.head_depth.copy() if self.head_depth is not None else None

            self.process_camera(h_color, h_depth, self.head_info,
                                HEAD_OPTICAL_FRAME, 'head')

        self.move_to_pose({'joint_head_pan': 0.0, 'joint_wrist_yaw': 1.57})
        time.sleep(0.8)

        with self.image_lock:
            w_color = self.wrist_color.copy() if self.wrist_color is not None else None
            w_depth = self.wrist_depth.copy() if self.wrist_depth is not None else None

        self.process_camera(w_color, w_depth, self.wrist_info,
                            WRIST_OPTICAL_FRAME, 'wrist')

    # ===================================================================
    # Wall-Following Exploration
    # ===================================================================

    def get_wall_distance_right(self):
        """Get average distance to the wall on the right side from lidar scan."""
        with self.scan_lock:
            if self.latest_scan is None:
                return None
            scan = self.latest_scan

        angle_min = scan.angle_min
        angle_increment = scan.angle_increment
        ranges = np.array(scan.ranges)

        target_min_angle = -2.094  # -120 deg
        target_max_angle = -1.047  # -60 deg

        idx_min = max(0, int((target_min_angle - angle_min) / angle_increment))
        idx_max = min(len(ranges) - 1, int((target_max_angle - angle_min) / angle_increment))

        if idx_min >= idx_max:
            return None

        right_ranges = ranges[idx_min:idx_max]
        valid = right_ranges[np.isfinite(right_ranges) & (right_ranges > 0.1) & (right_ranges < 10.0)]

        if len(valid) == 0:
            return None

        return float(np.median(valid))

    def get_front_clearance(self):
        """Get minimum distance in front of the robot from lidar."""
        with self.scan_lock:
            if self.latest_scan is None:
                return None
            scan = self.latest_scan

        angle_min = scan.angle_min
        angle_increment = scan.angle_increment
        ranges = np.array(scan.ranges)

        target_min_angle = -0.35  # -20 deg
        target_max_angle = 0.35   # +20 deg

        idx_min = max(0, int((target_min_angle - angle_min) / angle_increment))
        idx_max = min(len(ranges) - 1, int((target_max_angle - angle_min) / angle_increment))

        if idx_min >= idx_max:
            return None

        front_ranges = ranges[idx_min:idx_max]
        valid = front_ranges[np.isfinite(front_ranges) & (front_ranges > 0.1) & (front_ranges < 10.0)]

        if len(valid) == 0:
            return None

        return float(np.min(valid))

    def follow_perimeter_step(self):
        """
        One step of wall-following to circle the room perimeter.
        Keeps the wall on the right at ~WALL_FOLLOW_DISTANCE_M.
        """
        right_dist = self.get_wall_distance_right()
        front_dist = self.get_front_clearance()

        if front_dist is not None and front_dist < 0.5:
            self.logger.info(f'Front obstacle at {front_dist:.2f}m, turning left')
            self.move_to_pose({'rotate_mobile_base': 0.6})
            time.sleep(0.8)
            return

        if right_dist is None:
            self.move_to_pose({'rotate_mobile_base': -0.3})
            time.sleep(0.5)
            return

        error = right_dist - WALL_FOLLOW_DISTANCE_M
        if abs(error) > 0.2:
            correction = -0.3 if error > 0 else 0.3
            self.move_to_pose({'rotate_mobile_base': correction})
            time.sleep(0.4)

        self.move_to_pose({'translate_mobile_base': FORWARD_SPEED_M})
        time.sleep(0.8)

    def wall_following_loop(self):
        """
        Explore by following walls around the room perimeter.
        Scans cameras at each stop, detects when a full loop is completed.
        """
        self.logger.info('Starting wall-following exploration')

        start_to_odom, _ = hm.get_p1_to_p2_matrix(
            'base_link', 'odom', self.tf2_buffer, timeout_s=2.0)
        start_position = start_to_odom[:3, 3] if start_to_odom is not None else None
        step_count = 0

        while rclpy.ok() and not self.exploration_complete:
            step_count += 1
            self.logger.info(f'=== Wall-following step {step_count} ===')

            # Scan cameras with Detic
            self.scan_with_cameras()

            # Stow arm after scanning
            self.move_to_pose({
                'wrist_extension': 0.01,
                'joint_head_pan': 0.0,
                'joint_head_tilt': 0.0,
            })
            time.sleep(0.3)

            # Move along perimeter
            self.follow_perimeter_step()

            # Check if we've completed the loop
            if start_position is not None and step_count > MIN_STEPS_BEFORE_LOOP_CHECK:
                current_to_odom, _ = hm.get_p1_to_p2_matrix(
                    'base_link', 'odom', self.tf2_buffer, timeout_s=1.0)
                if current_to_odom is not None:
                    current_pos = current_to_odom[:3, 3]
                    dist_to_start = np.linalg.norm(current_pos[:2] - start_position[:2])
                    if dist_to_start < 0.5:
                        self.logger.info('Completed perimeter loop!')
                        self.exploration_complete = True

    # ===================================================================
    # Frontier Exploration (via funmap)
    # ===================================================================

    def call_funmap_head_scan(self):
        """Call funmap's head scan service to map from current position."""
        if not self.head_scan_client.wait_for_service(timeout_sec=5.0):
            self.logger.warn('/funmap/trigger_head_scan service not available')
            return False

        req = Trigger.Request()
        future = self.head_scan_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=60.0)

        if future.result() is not None:
            result = future.result()
            self.logger.info(f'Head scan: success={result.success}, msg="{result.message}"')
            return result.success
        self.logger.warn('Head scan service call timed out')
        return False

    def call_funmap_drive_to_scan(self):
        """
        Call funmap's drive-to-scan service. Finds the next unexplored frontier
        and navigates there. Returns False if no more frontiers exist.
        """
        if not self.drive_to_scan_client.wait_for_service(timeout_sec=5.0):
            self.logger.warn('/funmap/trigger_drive_to_scan service not available')
            return False

        req = Trigger.Request()
        future = self.drive_to_scan_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=120.0)

        if future.result() is not None:
            result = future.result()
            self.logger.info(f'Drive to scan: success={result.success}, msg="{result.message}"')
            return result.success
        self.logger.warn('Drive to scan service call timed out')
        return False

    def frontier_loop(self):
        """
        Explore using funmap frontier exploration:
        1. Head scan to build/update lidar map
        2. Scan cameras with Detic
        3. Drive to next frontier
        4. Repeat until no more frontiers
        """
        self.logger.info('Starting frontier exploration with funmap')

        consecutive_failures = 0
        scan_count = 0

        while rclpy.ok() and not self.exploration_complete:
            scan_count += 1
            self.logger.info(f'=== Frontier exploration step {scan_count} ===')

            # Build/update lidar map
            self.logger.info('Calling funmap head scan...')
            self.call_funmap_head_scan()
            time.sleep(0.5)

            # Scan cameras with Detic
            self.logger.info('Scanning cameras with Detic...')
            self.scan_with_cameras()

            # Stow arm after scanning
            self.move_to_pose({
                'wrist_extension': 0.01,
                'joint_head_pan': 0.0,
                'joint_head_tilt': 0.0,
            })
            time.sleep(0.3)

            # Drive to next frontier
            self.logger.info('Calling funmap drive-to-scan (frontier exploration)...')
            success = self.call_funmap_drive_to_scan()

            if success:
                consecutive_failures = 0
                time.sleep(0.5)
            else:
                consecutive_failures += 1
                self.logger.info(
                    f'No frontier found ({consecutive_failures}/{MAX_FRONTIER_FAILURES})')

                if consecutive_failures >= MAX_FRONTIER_FAILURES:
                    self.logger.info('No more frontiers — room fully explored!')
                    self.exploration_complete = True

    # ===================================================================
    # Visualization
    # ===================================================================

    def publish_drawer_markers(self):
        """Publish visualization markers for all discovered drawers in odom frame."""
        marker_array = MarkerArray()
        with self.drawers_lock:
            for i, d in enumerate(self.discovered_drawers):
                m = Marker()
                m.header.frame_id = 'odom'
                m.header.stamp = self.get_clock().now().to_msg()
                m.ns = 'search_drawers'
                m.id = i
                m.type = Marker.SPHERE
                m.action = Marker.ADD
                m.pose.position.x = float(d.handle_world_xyz[0])
                m.pose.position.y = float(d.handle_world_xyz[1])
                m.pose.position.z = float(d.handle_world_xyz[2])
                m.pose.orientation.w = 1.0
                m.scale.x = 0.1
                m.scale.y = 0.1
                m.scale.z = 0.1
                if d.reachable:
                    m.color.r = 0.0
                    m.color.g = 1.0
                    m.color.b = 0.0
                else:
                    m.color.r = 1.0
                    m.color.g = 0.0
                    m.color.b = 0.0
                m.color.a = 0.8
                m.lifetime.sec = 0
                marker_array.markers.append(m)
        self.marker_pub.publish(marker_array)

    # ===================================================================
    # Service: Return Drawers
    # ===================================================================

    def get_drawers_callback(self, request, response):
        """Service callback that returns all discovered drawers as JSON."""
        with self.drawers_lock:
            drawers_list = []
            for i, d in enumerate(self.discovered_drawers):
                drawers_list.append({
                    'id': i,
                    'handle_center_world_coordinates': {
                        'x': float(d.handle_world_xyz[0]),
                        'y': float(d.handle_world_xyz[1]),
                        'z': float(d.handle_world_xyz[2]),
                    },
                    'handle_orientation': d.handle_orientation,
                    'reachable': d.reachable,
                })
        response.success = True
        response.message = json.dumps(drawers_list)
        return response

    # ===================================================================
    # Main Exploration Loop (dispatches to wall_following or frontier)
    # ===================================================================

    def exploration_loop(self):
        """Wait for cameras, stow arm, then run the selected exploration mode."""
        self.logger.info('Waiting for camera images...')
        for _ in range(200):
            with self.image_lock:
                if self.head_color is not None:
                    break
            time.sleep(0.1)

        if self.head_color is None:
            self.logger.error('No camera images received. Aborting.')
            return

        # Stow arm for safe travel
        self.move_to_pose({
            'wrist_extension': 0.01,
            'joint_lift': 0.5,
            'joint_wrist_yaw': 0.0,
            'gripper_aperture': -0.05,
        })
        time.sleep(0.5)

        self.logger.info(f'Exploration mode: {self.exploration_mode}')

        if self.exploration_mode == 'detect_only':
            self.detect_only_loop()
        elif self.exploration_mode == 'frontier':
            self.frontier_loop()
        else:
            self.wall_following_loop()

        # Final report
        with self.drawers_lock:
            total = len(self.discovered_drawers)
            reachable_count = sum(1 for d in self.discovered_drawers if d.reachable)

        self.logger.info(
            f'Exploration complete. Found {total} drawers ({reachable_count} reachable)')
        self.publish_drawer_markers()

    def detect_only_loop(self):
        """Scan cameras from current position without driving. Repeats every 5s."""
        self.logger.info('detect_only mode: scanning from current position')
        while rclpy.ok():
            self.scan_with_cameras()
            with self.drawers_lock:
                total = len(self.discovered_drawers)
                reachable = sum(1 for d in self.discovered_drawers if d.reachable)
            self.logger.info(f'detect_only: {total} drawers found ({reachable} reachable)')
            self.publish_drawer_markers()
            time.sleep(5.0)

    def get_all_drawers(self):
        """Return the list of discovered drawers."""
        with self.drawers_lock:
            return list(self.discovered_drawers)

    # ===================================================================
    # Node Setup
    # ===================================================================

    def main(self):
        hm.HelloNode.main(self, 'search_for_drawers', 'search_for_drawers',
                          wait_for_first_pointcloud=False)

        self.logger = self.get_logger()
        self.callback_group = ReentrantCallbackGroup()
        sensor_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

        # Read exploration_mode parameter: 'wall_following' (default) or 'frontier'
        if not self.has_parameter('exploration_mode'):
            self.declare_parameter('exploration_mode', 'wall_following')
        self.exploration_mode = self.get_parameter(
            'exploration_mode').get_parameter_value().string_value
        if self.exploration_mode not in ('wall_following', 'frontier', 'detect_only'):
            self.logger.warn(
                f'Unknown exploration_mode "{self.exploration_mode}", '
                f'defaulting to wall_following')
            self.exploration_mode = 'wall_following'

        # Initialize VisualDetector from semantic-object-container-room submodule
        self.logger.info(f'Loading VisualDetector (DETIC 21K) from {DETIC_ROOT}...')
        self.detector = VisualDetector(
            score_threshold=0.3,
            detic_root=DETIC_ROOT,
        )
        self.logger.info('VisualDetector ready')

        # Joint states
        self.create_subscription(
            JointState, '/stretch/joint_states',
            self.joint_states_callback, qos_profile=0,
            callback_group=self.callback_group)

        # Head camera
        self.create_subscription(
            Image, HEAD_COLOR_TOPIC, self.head_color_cb,
            qos_profile=sensor_qos, callback_group=self.callback_group)
        self.create_subscription(
            Image, HEAD_DEPTH_TOPIC, self.head_depth_cb,
            qos_profile=sensor_qos, callback_group=self.callback_group)
        self.create_subscription(
            CameraInfo, HEAD_INFO_TOPIC, self.head_info_cb,
            qos_profile=sensor_qos, callback_group=self.callback_group)

        # Wrist camera
        self.create_subscription(
            Image, WRIST_COLOR_TOPIC, self.wrist_color_cb,
            qos_profile=sensor_qos, callback_group=self.callback_group)
        self.create_subscription(
            Image, WRIST_DEPTH_TOPIC, self.wrist_depth_cb,
            qos_profile=sensor_qos, callback_group=self.callback_group)
        self.create_subscription(
            CameraInfo, WRIST_INFO_TOPIC, self.wrist_info_cb,
            qos_profile=sensor_qos, callback_group=self.callback_group)

        # Lidar (used by wall_following mode)
        self.create_subscription(
            LaserScan, '/scan', self.scan_cb,
            qos_profile=sensor_qos, callback_group=self.callback_group)

        # Publishers
        self.marker_pub = self.create_publisher(
            MarkerArray, '/search_drawers/discovered_drawers', 10,
            callback_group=self.callback_group)
        self.detection_pub = self.create_publisher(
            Image, '/search_drawers/detection_image', 10,
            callback_group=self.callback_group)

        # Funmap frontier exploration service clients (used by frontier mode)
        self.head_scan_client = self.create_client(
            Trigger, '/funmap/trigger_head_scan',
            callback_group=self.callback_group)
        self.drive_to_scan_client = self.create_client(
            Trigger, '/funmap/trigger_drive_to_scan',
            callback_group=self.callback_group)

        # Service to return discovered drawers
        self.create_service(
            Trigger, '/search_drawers/get_drawers',
            self.get_drawers_callback,
            callback_group=self.callback_group)

        self.logger.info(
            f'SearchForDrawersNode ready (mode={self.exploration_mode}). '
            f'Starting exploration...')

        self.explore_thread = threading.Thread(
            target=self.exploration_loop, daemon=True)
        self.explore_thread.start()


def main():
    try:
        node = SearchForDrawersNode()
        node.main()
        node.new_thread.join()
    except KeyboardInterrupt:
        rclpy.logging.get_logger('search_for_drawers').info('Shutting down')


if __name__ == '__main__':
    main()
