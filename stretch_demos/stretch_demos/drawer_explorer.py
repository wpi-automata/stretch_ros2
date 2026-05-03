#!/usr/bin/env python3
"""
Drawer Explorer: random-walk the Stretch3 through the environment, using
both head (D435i) and wrist (D405) cameras to find drawers via SAM
segmentation. Maintains a global list of discovered drawers and calls
the DrawerGrasp service when one is found.

Published topics:
  /drawer_explorer/discovered_drawers (visualization_msgs/MarkerArray)

Service clients:
  /drawer_grasp (stretch_drawer_interfaces/srv/DrawerGrasp)
"""

import math
import os
import random
import threading
import time

import cv2
import numpy as np
import rclpy
import rclpy.logging
from rclpy.qos import QoSProfile, ReliabilityPolicy
from cv_bridge import CvBridge
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image, CameraInfo, JointState
from geometry_msgs.msg import Point, Vector3, Transform, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from stretch_drawer_interfaces.srv import DrawerGrasp

import hello_helpers.hello_misc as hm

# SAM imports (optional — falls back to edge-based detection)
try:
    from segment_anything import sam_model_registry, SamPredictor
    HAS_SAM = True
except ImportError:
    try:
        from mobile_sam import sam_model_registry, SamPredictor
        HAS_SAM = True
    except ImportError:
        HAS_SAM = False

SAM_CHECKPOINT = os.environ.get(
    'SAM_CHECKPOINT', os.path.expanduser('~/sam_vit_h_4b8939.pth'))
SAM_MODEL_TYPE = os.environ.get('SAM_MODEL_TYPE', 'vit_h')

# Head camera
HEAD_COLOR_TOPIC = '/camera/color/image_raw'
HEAD_DEPTH_TOPIC = '/camera/depth/image_rect_raw'
HEAD_INFO_TOPIC = '/camera/color/camera_info'
HEAD_OPTICAL_FRAME = 'camera_color_optical_frame'

# Wrist camera
WRIST_COLOR_TOPIC = '/gripper_camera/image_raw'
WRIST_DEPTH_TOPIC = '/gripper_camera/depth/image_rect_raw'
WRIST_INFO_TOPIC = '/gripper_camera/camera_info'
WRIST_OPTICAL_FRAME = 'gripper_camera_color_optical_frame'

# Exploration parameters
HEAD_PAN_POSITIONS = [-1.2, -0.8, -0.4, 0.0, 0.4, 0.8]
HEAD_TILT_SEARCH = -0.5
WRIST_TILT_SEARCH = -0.3
RANDOM_WALK_DISTANCE_RANGE = (0.3, 1.0)
RANDOM_TURN_RANGE = (-1.0, 1.0)
DUPLICATE_DISTANCE_THRESHOLD_M = 0.5


class DrawerDetector:
    """SAM-based drawer segmentation with edge-detection fallback."""

    def __init__(self, logger):
        self.logger = logger
        self.predictor = None
        if HAS_SAM and os.path.exists(SAM_CHECKPOINT):
            logger.info(f'Loading SAM model: {SAM_MODEL_TYPE}')
            sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
            try:
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                device = 'cpu'
            sam.to(device)
            self.predictor = SamPredictor(sam)
            logger.info('SAM loaded')
        else:
            logger.info('SAM not available — using fallback edge detector')

    def detect(self, rgb_image):
        """Returns list of dicts: [{"mask": ndarray, "bbox": (x1,y1,x2,y2), "score": float}]"""
        if self.predictor is not None:
            return self._detect_sam(rgb_image)
        return self._detect_fallback(rgb_image)

    def _detect_sam(self, rgb_image):
        self.predictor.set_image(rgb_image)
        h, w = rgb_image.shape[:2]
        grid_points = []
        for yi in range(3, h - 3, h // 5):
            for xi in range(3, w - 3, w // 5):
                grid_points.append([xi, yi])
        grid_points = np.array(grid_points)
        grid_labels = np.ones(len(grid_points), dtype=int)

        masks, scores, _ = self.predictor.predict(
            point_coords=grid_points,
            point_labels=grid_labels,
            multimask_output=True,
        )

        detections = []
        for mask, score in zip(masks, scores):
            if score < 0.5:
                continue
            bbox = self._mask_to_bbox(mask)
            if bbox is None:
                continue
            if self._is_drawer_like(mask, bbox):
                detections.append({'mask': mask, 'bbox': bbox, 'score': float(score)})

        detections.sort(key=lambda d: d['score'], reverse=True)
        return detections[:5]

    def _detect_fallback(self, rgb_image):
        """Edge/contour-based drawer detection."""
        h, w = rgb_image.shape[:2]
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1500:
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = cw / max(ch, 1)
            if aspect < 1.2 or aspect > 8.0:
                continue
            cy_box = y + ch / 2
            if cy_box < h * 0.2:
                continue
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area < 1:
                continue
            solidity = area / hull_area
            if solidity < 0.6:
                continue

            mask = np.zeros((h, w), dtype=bool)
            cv2.drawContours(mask.view(np.uint8), [cnt], -1, 1, -1)
            score = area * solidity / (h * w)
            detections.append({
                'mask': mask,
                'bbox': (x, y, x + cw, y + ch),
                'score': float(score),
            })

        detections.sort(key=lambda d: d['score'], reverse=True)
        # Simple NMS
        kept = []
        for det in detections:
            overlap = False
            for k in kept:
                if self._iou(det['bbox'], k['bbox']) > 0.3:
                    overlap = True
                    break
            if not overlap:
                kept.append(det)
        return kept[:5]

    @staticmethod
    def _mask_to_bbox(mask):
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return None
        return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    @staticmethod
    def _is_drawer_like(mask, bbox):
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        if bw < 20 or bh < 10:
            return False
        aspect = bw / max(bh, 1)
        if aspect < 0.8 or aspect > 6.0:
            return False
        fill = mask.sum() / max(bw * bh, 1)
        return fill > 0.5

    @staticmethod
    def _iou(b1, b2):
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter / max(a1 + a2 - inter, 1)


class DiscoveredDrawer:
    """Record of a drawer found during exploration."""

    def __init__(self, xyz_global, rgb_image, bbox_corners, pull_direction):
        self.xyz_global = xyz_global        # np.array(3,) in map/odom frame
        self.rgb_image = rgb_image          # np.ndarray BGR
        self.bbox_corners = bbox_corners    # np.array(8,3) AABB corners
        self.pull_direction = pull_direction # np.array(3,)
        self.timestamp = time.time()


class DrawerExplorerNode(hm.HelloNode):

    def __init__(self):
        hm.HelloNode.__init__(self)
        self.rate = 10.0
        self.bridge = CvBridge()
        self.callback_group = None

        # Camera data (protected by locks)
        self.head_color = None
        self.head_depth = None
        self.head_info = None
        self.wrist_color = None
        self.wrist_depth = None
        self.wrist_info = None
        self.image_lock = threading.Lock()

        self.joint_states = None
        self.joint_states_lock = threading.Lock()

        self.discovered_drawers = []
        self.drawers_lock = threading.Lock()

        self.detector = None
        self.grasp_client = None

    # -- Callbacks --

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

    # -- 3D geometry --

    def back_project_mask_to_3d(self, mask, depth_image, camera_info, optical_frame):
        """
        Back-project masked depth pixels to the odom frame.
        Returns (points_3d, centroid, bbox_corners, pull_direction) or None.
        """
        if camera_info is None or depth_image is None:
            return None

        fx = camera_info.k[0]
        fy = camera_info.k[4]
        cx = camera_info.k[2]
        cy = camera_info.k[5]
        if fx == 0 or fy == 0:
            return None

        ys, xs = np.where(mask)
        if len(ys) < 20:
            return None

        if len(ys) > 2000:
            idx = np.random.choice(len(ys), 2000, replace=False)
            ys, xs = ys[idx], xs[idx]

        depths = depth_image[ys, xs]
        valid = np.isfinite(depths) & (depths > 0.1) & (depths < 5.0)
        ys, xs, depths = ys[valid], xs[valid], depths[valid]
        if len(depths) < 10:
            return None

        x_cam = (xs - cx) * depths / fx
        y_cam = (ys - cy) * depths / fy
        z_cam = depths

        cam_to_odom, _ = hm.get_p1_to_p2_matrix(
            optical_frame, 'odom', self.tf2_buffer, timeout_s=1.0)
        if cam_to_odom is None:
            return None

        pts_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(z_cam)], axis=-1)
        pts_odom = (cam_to_odom @ pts_cam.T).T[:, :3]

        valid_z = (pts_odom[:, 2] > 0.0) & (pts_odom[:, 2] < 3.0)
        pts_odom = pts_odom[valid_z]
        if len(pts_odom) < 10:
            return None

        mins = pts_odom.min(axis=0)
        maxs = pts_odom.max(axis=0)
        centroid = (mins + maxs) / 2.0

        corners = np.array([
            [mins[0], mins[1], mins[2]],
            [maxs[0], mins[1], mins[2]],
            [maxs[0], maxs[1], mins[2]],
            [mins[0], maxs[1], mins[2]],
            [mins[0], mins[1], maxs[2]],
            [maxs[0], mins[1], maxs[2]],
            [maxs[0], maxs[1], maxs[2]],
            [mins[0], maxs[1], maxs[2]],
        ])

        # Pull direction: from centroid toward camera (outward normal)
        cam_pos = cam_to_odom[:3, 3]
        pull_dir = cam_pos - centroid
        pull_dir[2] = 0.0
        norm = np.linalg.norm(pull_dir)
        if norm > 1e-6:
            pull_dir = pull_dir / norm
        else:
            pull_dir = np.array([1.0, 0.0, 0.0])

        return pts_odom, centroid, corners, pull_dir

    def is_duplicate(self, centroid):
        """Check if a drawer at this location was already discovered."""
        with self.drawers_lock:
            for d in self.discovered_drawers:
                if np.linalg.norm(d.xyz_global - centroid) < DUPLICATE_DISTANCE_THRESHOLD_M:
                    return True
        return False

    # -- Exploration loop --

    def process_camera(self, color, depth, info, optical_frame, cam_name):
        """Run detection on one camera's images. Returns DiscoveredDrawer or None."""
        if color is None or depth is None:
            return None

        detections = self.detector.detect(color)
        if not detections:
            return None

        best = detections[0]
        self.logger.info(
            f'[{cam_name}] Drawer candidate: bbox={best["bbox"]}, score={best["score"]:.2f}')

        result = self.back_project_mask_to_3d(
            best['mask'], depth, info, optical_frame)
        if result is None:
            self.logger.info(f'[{cam_name}] Could not back-project to 3D')
            return None

        _, centroid, corners, pull_dir = result

        if self.is_duplicate(centroid):
            self.logger.info(
                f'[{cam_name}] Drawer at ({centroid[0]:.2f}, {centroid[1]:.2f}) already known')
            return None

        self.logger.info(
            f'[{cam_name}] NEW drawer at odom ({centroid[0]:.2f}, '
            f'{centroid[1]:.2f}, {centroid[2]:.2f})')

        drawer = DiscoveredDrawer(centroid, color.copy(), corners, pull_dir)
        with self.drawers_lock:
            self.discovered_drawers.append(drawer)
            idx = len(self.discovered_drawers)

        self.logger.info(f'Drawer #{idx} added to list. Total: {idx}')
        self.publish_drawer_markers()
        return drawer

    def call_grasp_service(self, drawer):
        """Send the discovered drawer to the DrawerGrasp service."""
        if self.grasp_client is None:
            return

        if not self.grasp_client.wait_for_service(timeout_sec=2.0):
            self.logger.warn('DrawerGrasp service not available')
            return

        req = DrawerGrasp.Request()

        # Pack RGB image
        try:
            req.rgb_image = self.bridge.cv2_to_imgmsg(drawer.rgb_image, encoding='bgr8')
        except Exception as e:
            self.logger.error(f'Failed to encode image: {e}')
            return

        # Pack depth as empty (grasp server uses its own live depth)
        empty_depth = np.zeros((1, 1), dtype=np.float32)
        req.depth_image = self.bridge.cv2_to_imgmsg(empty_depth, encoding='32FC1')

        for i in range(8):
            req.bbox_corners[i] = Point(
                x=float(drawer.bbox_corners[i][0]),
                y=float(drawer.bbox_corners[i][1]),
                z=float(drawer.bbox_corners[i][2]),
            )

        req.drawer_centroid = Point(
            x=float(drawer.xyz_global[0]),
            y=float(drawer.xyz_global[1]),
            z=float(drawer.xyz_global[2]),
        )
        req.pull_direction = Vector3(
            x=float(drawer.pull_direction[0]),
            y=float(drawer.pull_direction[1]),
            z=float(drawer.pull_direction[2]),
        )
        req.camera_to_world = Transform()
        req.camera_to_world.rotation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        self.logger.info('Calling /drawer_grasp service...')
        future = self.grasp_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=60.0)

        if future.result() is not None:
            resp = future.result()
            self.logger.info(f'Grasp result: success={resp.success}, msg={resp.message}')
        else:
            self.logger.warn('Grasp service call timed out or failed')

    def publish_drawer_markers(self):
        """Publish visualization markers for all discovered drawers."""
        marker_array = MarkerArray()
        with self.drawers_lock:
            for i, d in enumerate(self.discovered_drawers):
                m = Marker()
                m.header.frame_id = 'odom'
                m.header.stamp = self.get_clock().now().to_msg()
                m.ns = 'discovered_drawers'
                m.id = i
                m.type = Marker.CUBE
                m.action = Marker.ADD
                m.pose.position.x = float(d.xyz_global[0])
                m.pose.position.y = float(d.xyz_global[1])
                m.pose.position.z = float(d.xyz_global[2])
                m.pose.orientation.w = 1.0
                mins = d.bbox_corners.min(axis=0)
                maxs = d.bbox_corners.max(axis=0)
                m.scale.x = float(max(maxs[0] - mins[0], 0.05))
                m.scale.y = float(max(maxs[1] - mins[1], 0.05))
                m.scale.z = float(max(maxs[2] - mins[2], 0.05))
                m.color.r = 0.2
                m.color.g = 0.8
                m.color.b = 0.2
                m.color.a = 0.6
                m.lifetime.sec = 0
                marker_array.markers.append(m)
        self.marker_pub.publish(marker_array)

    def exploration_loop(self):
        """Main behavior: random walk + scan cameras + detect drawers."""
        self.logger.info('Starting exploration loop')

        # Wait for images to arrive
        self.logger.info('Waiting for camera images...')
        for _ in range(100):
            with self.image_lock:
                if self.head_color is not None:
                    break
            time.sleep(0.1)

        scan_index = 0

        while rclpy.ok():
            # 1. Stow arm for safe travel
            self.move_to_pose({
                'wrist_extension': 0.01,
                'joint_lift': 0.5,
                'joint_wrist_yaw': 0.0,
            })
            time.sleep(0.3)

            # 2. Scan with head camera across pan positions
            self.logger.info(f'Scan #{scan_index + 1}: panning head camera')
            self.move_to_pose({'joint_head_tilt': HEAD_TILT_SEARCH})
            time.sleep(0.3)

            found_drawer = None
            for pan_angle in HEAD_PAN_POSITIONS:
                self.move_to_pose({'joint_head_pan': pan_angle})
                time.sleep(0.8)

                with self.image_lock:
                    h_color = self.head_color.copy() if self.head_color is not None else None
                    h_depth = self.head_depth.copy() if self.head_depth is not None else None

                drawer = self.process_camera(
                    h_color, h_depth, self.head_info,
                    HEAD_OPTICAL_FRAME, 'head')
                if drawer is not None:
                    found_drawer = drawer

            # 3. Also check wrist camera (tilt wrist down slightly)
            self.move_to_pose({
                'joint_head_pan': 0.0,
                'joint_head_tilt': WRIST_TILT_SEARCH,
                'joint_wrist_yaw': 1.57,
            })
            time.sleep(0.8)

            with self.image_lock:
                w_color = self.wrist_color.copy() if self.wrist_color is not None else None
                w_depth = self.wrist_depth.copy() if self.wrist_depth is not None else None

            drawer = self.process_camera(
                w_color, w_depth, self.wrist_info,
                WRIST_OPTICAL_FRAME, 'wrist')
            if drawer is not None and found_drawer is None:
                found_drawer = drawer

            # 4. If a new drawer was found, call the grasp service
            if found_drawer is not None:
                self.call_grasp_service(found_drawer)

            # 5. Random walk: turn, then drive forward
            turn = random.uniform(*RANDOM_TURN_RANGE)
            drive = random.uniform(*RANDOM_WALK_DISTANCE_RANGE)

            self.logger.info(
                f'Random walk: turn {math.degrees(turn):.0f} deg, '
                f'drive {drive:.2f} m')
            self.move_to_pose({'rotate_mobile_base': turn})
            time.sleep(0.5)
            self.move_to_pose({'translate_mobile_base': drive})
            time.sleep(1.0)

            scan_index += 1

    def main(self):
        hm.HelloNode.main(self, 'drawer_explorer', 'drawer_explorer',
                          wait_for_first_pointcloud=False)

        self.logger = self.get_logger()
        self.callback_group = ReentrantCallbackGroup()
        sensor_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

        # Joint states
        self.create_subscription(
            JointState, '/stretch/joint_states',
            self.joint_states_callback, qos_profile=0,
            callback_group=self.callback_group)

        # Head camera (D435i)
        self.create_subscription(
            Image, HEAD_COLOR_TOPIC, self.head_color_cb,
            qos_profile=sensor_qos, callback_group=self.callback_group)
        self.create_subscription(
            Image, HEAD_DEPTH_TOPIC, self.head_depth_cb,
            qos_profile=sensor_qos, callback_group=self.callback_group)
        self.create_subscription(
            CameraInfo, HEAD_INFO_TOPIC, self.head_info_cb,
            qos_profile=sensor_qos, callback_group=self.callback_group)

        # Wrist camera (D405)
        self.create_subscription(
            Image, WRIST_COLOR_TOPIC, self.wrist_color_cb,
            qos_profile=sensor_qos, callback_group=self.callback_group)
        self.create_subscription(
            Image, WRIST_DEPTH_TOPIC, self.wrist_depth_cb,
            qos_profile=sensor_qos, callback_group=self.callback_group)
        self.create_subscription(
            CameraInfo, WRIST_INFO_TOPIC, self.wrist_info_cb,
            qos_profile=sensor_qos, callback_group=self.callback_group)

        # Drawer visualization
        self.marker_pub = self.create_publisher(
            MarkerArray, '/drawer_explorer/discovered_drawers', 10,
            callback_group=self.callback_group)

        # Grasp service client
        self.grasp_client = self.create_client(
            DrawerGrasp, '/drawer_grasp',
            callback_group=self.callback_group)

        # Detector
        self.detector = DrawerDetector(self.logger)

        self.logger.info('DrawerExplorerNode ready. Starting exploration...')

        # Run exploration in a separate thread so ROS callbacks keep spinning
        self.explore_thread = threading.Thread(
            target=self.exploration_loop, daemon=True)
        self.explore_thread.start()


def main():
    try:
        node = DrawerExplorerNode()
        node.main()
        node.new_thread.join()
    except KeyboardInterrupt:
        rclpy.logging.get_logger('drawer_explorer').info('Shutting down')


if __name__ == '__main__':
    main()
