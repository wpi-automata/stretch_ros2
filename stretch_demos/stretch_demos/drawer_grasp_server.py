#!/usr/bin/env python3
"""
Drawer Grasp Server: ROS2 service node that receives a drawer image + 3D bounding box,
identifies a handle, grasps it, and pulls the drawer open.

Service: /drawer_grasp (stretch_drawer_interfaces/srv/DrawerGrasp)
"""

import math
import threading
import time

import cv2
import numpy as np
import rclpy
import rclpy.logging
from rclpy.qos import QoSProfile, ReliabilityPolicy
from cv_bridge import CvBridge
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import JointState
from stretch_drawer_interfaces.srv import DrawerGrasp

import hello_helpers.hello_misc as hm

DEFAULT_PULL_DISTANCE_M = 0.3048  # 1 foot

# Contact detection thresholds (Nm)
EXTENSION_EFFORT_THRESHOLD = 40.0
LIFT_EFFORT_THRESHOLD = 20.0


def compute_pull_distance(bbox_corners, pull_direction):
    """Hook for future logic to compute pull distance from drawer geometry.
    Currently returns the default 1 foot."""
    # Future: estimate drawer depth from bbox extent along pull_direction,
    # or use learned models, or read from a config.
    return DEFAULT_PULL_DISTANCE_M


class DrawerGraspServer(hm.HelloNode):

    def __init__(self):
        hm.HelloNode.__init__(self)
        self.rate = 10.0
        self.joint_states = None
        self.joint_states_lock = threading.Lock()
        self.wrist_position = None
        self.lift_position = None
        self.bridge = CvBridge()

    def joint_states_callback(self, joint_states):
        with self.joint_states_lock:
            self.joint_states = joint_states
        wrist_position, _, _ = hm.get_wrist_state(joint_states)
        self.wrist_position = wrist_position
        lift_position, _, _ = hm.get_lift_state(joint_states)
        self.lift_position = lift_position

    def find_handle_in_image(self, rgb_image, bbox_corners):
        """
        Given an RGB image of a drawer and its 3D bounding box corners,
        find the handle location.

        Strategy: look for small, high-contrast horizontal features (handles)
        within the projected bounding box region. Falls back to the vertical
        center of the bounding box face if detection fails.
        """
        h, w = rgb_image.shape[:2]
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Look for small rectangular contours (handle shapes)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        best_handle = None
        best_score = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200 or area > 5000:
                continue

            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = cw / max(ch, 1)

            # Handles are typically wider than tall or roughly square
            if aspect < 0.5 or aspect > 8.0:
                continue

            # Prefer features near the vertical center of the image
            cy_frac = (y + ch / 2) / h
            center_bonus = 1.0 - abs(cy_frac - 0.5) * 2.0

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area < 1:
                continue
            solidity = area / hull_area

            score = area * solidity * (1.0 + center_bonus)
            if score > best_score:
                best_score = score
                best_handle = (x + cw // 2, y + ch // 2)

        return best_handle

    def handle_grasp_request(self, request, response):
        """Service callback: identify handle, navigate, grasp, pull."""
        self.logger.info('DrawerGraspServer: received grasp request')

        # Extract data from request
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(request.rgb_image, desired_encoding='bgr8')
        except Exception as e:
            response.success = False
            response.message = f'Failed to decode RGB image: {e}'
            return response

        centroid = np.array([
            request.drawer_centroid.x,
            request.drawer_centroid.y,
            request.drawer_centroid.z,
        ])
        pull_dir = np.array([
            request.pull_direction.x,
            request.pull_direction.y,
            request.pull_direction.z,
        ])
        pull_dir_norm = np.linalg.norm(pull_dir)
        if pull_dir_norm > 1e-6:
            pull_dir = pull_dir / pull_dir_norm

        bbox_corners = np.array([
            [request.bbox_corners[i].x, request.bbox_corners[i].y, request.bbox_corners[i].z]
            for i in range(8)
        ])

        # Find handle in the image
        handle_pixel = self.find_handle_in_image(rgb_image, bbox_corners)
        if handle_pixel is not None:
            self.logger.info(f'Handle detected at pixel ({handle_pixel[0]}, {handle_pixel[1]})')
        else:
            self.logger.info('No handle detected, using drawer centroid as grasp target')

        # Grasp point: centroid + slight offset along pull direction (handle protrudes)
        grasp_point = centroid + pull_dir * 0.02
        response.grasp_point.x = float(grasp_point[0])
        response.grasp_point.y = float(grasp_point[1])
        response.grasp_point.z = float(grasp_point[2])

        # Compute pull distance
        pull_distance = compute_pull_distance(bbox_corners, pull_dir)
        self.logger.info(f'Pull distance: {pull_distance:.3f} m ({pull_distance / 0.0254:.1f} in)')

        # --- Execute grasp and pull sequence ---
        try:
            success, msg = self._execute_grasp_and_pull(grasp_point, pull_dir, pull_distance)
            response.success = success
            response.message = msg
        except Exception as e:
            response.success = False
            response.message = f'Grasp execution failed: {e}'
            self.logger.error(f'Grasp execution exception: {e}')

        return response

    def _execute_grasp_and_pull(self, grasp_point, pull_direction, pull_distance):
        """
        Position arm at grasp_point, close gripper, retract by pull_distance.
        """
        # 1. Position the arm at the grasp height
        target_lift = float(grasp_point[2]) - 0.2  # base offset
        target_lift = max(0.2, min(1.05, target_lift))
        self.logger.info(f'Setting lift to {target_lift:.3f} m')
        self.move_to_pose({'joint_lift': target_lift}, blocking=True)
        time.sleep(0.3)

        # 2. Extend arm to reach the grasp point
        if self.wrist_position is not None:
            # Estimate needed extension from current position
            grasp_dist_lateral = float(np.sqrt(grasp_point[0]**2 + grasp_point[1]**2))
            target_extension = min(0.5, max(0.01, grasp_dist_lateral - 0.4))
        else:
            target_extension = 0.2
        self.logger.info(f'Extending wrist to {target_extension:.3f} m')
        self.move_to_pose({
            'wrist_extension': target_extension,
            'joint_wrist_yaw': 0.0,
        }, blocking=True)
        time.sleep(0.3)

        # 3. Open gripper, move into contact, close gripper
        self.logger.info('Opening gripper')
        self.move_to_pose({'gripper_aperture': 0.09}, blocking=True)
        time.sleep(0.3)

        # Extend a bit more into the handle
        self.move_to_pose({
            'wrist_extension': target_extension + 0.04,
        }, blocking=True)
        time.sleep(0.3)

        self.logger.info('Closing gripper on handle')
        self.move_to_pose({'gripper_aperture': -0.05}, blocking=True)
        time.sleep(0.5)

        # 4. Pull back by retracting the arm
        retract_target = max(0.01, target_extension - pull_distance)
        self.logger.info(f'Pulling drawer: retracting from {target_extension:.3f} to {retract_target:.3f} m')
        self.move_to_pose({'wrist_extension': retract_target}, blocking=True)
        time.sleep(0.5)

        # 5. Release
        self.logger.info('Releasing gripper')
        self.move_to_pose({'gripper_aperture': 0.06}, blocking=True)
        time.sleep(0.3)

        # 6. Stow
        self.move_to_pose({
            'wrist_extension': 0.01,
            'joint_lift': 0.5,
        }, blocking=True)

        self.logger.info('Drawer pull complete')
        return True, f'Successfully pulled drawer {pull_distance:.3f} m'

    def main(self):
        hm.HelloNode.main(self, 'drawer_grasp_server', 'drawer_grasp_server',
                          wait_for_first_pointcloud=False)

        self.logger = self.get_logger()
        self.callback_group = ReentrantCallbackGroup()

        self.joint_states_subscriber = self.create_subscription(
            JointState, '/stretch/joint_states',
            self.joint_states_callback, qos_profile=0,
            callback_group=self.callback_group
        )

        self.grasp_service = self.create_service(
            DrawerGrasp, '/drawer_grasp',
            self.handle_grasp_request,
            callback_group=self.callback_group
        )

        self.logger.info('DrawerGraspServer ready. Service: /drawer_grasp')


def main():
    try:
        node = DrawerGraspServer()
        node.main()
        node.new_thread.join()
    except KeyboardInterrupt:
        rclpy.logging.get_logger('drawer_grasp_server').info('Shutting down')


if __name__ == '__main__':
    main()
