#!/usr/bin/env python3
"""
Open Drawers: ROS2 node that receives a list of drawers (from search_for_drawers)
and a drawer ID, orients the robot to face the handle, then grasps and pulls
the drawer open by 1 foot.

Input:
  - List of drawers with structure:
      drawer: {image, handle_center_world_coordinates: {x, y, z}, reachable: bool}
  - Drawer ID to open

Services provided:
  /open_drawers/open (stretch_drawer_interfaces/srv/DrawerGrasp)

Services called:
  /search_drawers/get_drawers (std_srvs/srv/Trigger) — to fetch drawer list
"""

import json
import math
import threading
import time

import numpy as np
import rclpy
import rclpy.logging
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

import hello_helpers.hello_misc as hm

PULL_DISTANCE_M = 0.3048  # 1 foot

# Approach: how far from the drawer face the robot should stop
APPROACH_DISTANCE_M = 0.55

# Force feedback thresholds (work in both sim and real via /stretch/joint_states effort)
EXTENSION_EFFORT_THRESHOLD = 40.0   # Nm — stop extending if arm hits something
PULL_EFFORT_THRESHOLD = 50.0        # Nm — stop pulling if drawer is stuck/locked
GRIPPER_EFFORT_THRESHOLD = 20.0     # Nm — detect when gripper has gripped something

# Lift height offset from handle (meters below handle for gripper clearance)
LIFT_OFFSET_M = 0.0

# Timing for effort monitoring
EFFORT_POLL_RATE_S = 0.05  # 20Hz effort check during extension/retraction
EFFORT_SETTLE_S = 0.3      # wait after motion command before reading effort


class OpenDrawersNode(hm.HelloNode):

    def __init__(self):
        hm.HelloNode.__init__(self)
        self.rate = 10.0
        self.callback_group = None

        self.joint_states = None
        self.joint_states_lock = threading.Lock()
        self.wrist_position = None
        self.wrist_effort = None
        self.lift_position = None
        self.gripper_effort = None

        self.drawers = []

    # --- Callbacks ---

    def joint_states_callback(self, msg):
        with self.joint_states_lock:
            self.joint_states = msg
        wrist_position, wrist_velocity, wrist_effort = hm.get_wrist_state(msg)
        self.wrist_position = wrist_position
        self.wrist_effort = wrist_effort
        lift_position, _, _ = hm.get_lift_state(msg)
        self.lift_position = lift_position
        if 'gripper_aperture' in msg.name:
            idx = list(msg.name).index('gripper_aperture')
            self.gripper_effort = msg.effort[idx] if idx < len(msg.effort) else None

    # --- Orientation ---

    def get_robot_pose_xya(self):
        """Get robot (x, y, yaw) in odom frame."""
        base_to_odom, _ = hm.get_p1_to_p2_matrix(
            'base_link', 'odom', self.tf2_buffer, timeout_s=1.0)
        if base_to_odom is None:
            return None
        x = base_to_odom[0, 3]
        y = base_to_odom[1, 3]
        yaw = math.atan2(base_to_odom[1, 0], base_to_odom[0, 0])
        return x, y, yaw

    def orient_to_handle(self, handle_xyz):
        """
        Rotate the robot so that its arm (extending to the left) can reach the handle.
        The Stretch3 arm extends perpendicular to the left of the drive direction.
        We need the robot's left side facing the drawer, meaning the robot should be
        positioned so the handle is to its left at arm-reach distance.

        Strategy:
        1. Compute angle from robot to handle
        2. The arm extends at +90 degrees from the robot's forward heading
        3. So we want: robot_yaw + 90° = angle_to_handle
           i.e., robot_yaw = angle_to_handle - 90°
        """
        pose = self.get_robot_pose_xya()
        if pose is None:
            self.logger.error('Cannot get robot pose')
            return False

        rx, ry, current_yaw = pose
        dx = handle_xyz[0] - rx
        dy = handle_xyz[1] - ry

        angle_to_handle = math.atan2(dy, dx)

        # Desired yaw: arm points left (+90° from forward), so forward is -90° from handle direction
        desired_yaw = angle_to_handle - math.pi / 2.0

        # Compute the rotation needed
        rotation = desired_yaw - current_yaw
        # Normalize to [-pi, pi]
        rotation = math.atan2(math.sin(rotation), math.cos(rotation))

        self.logger.info(
            f'Orienting to handle: current_yaw={math.degrees(current_yaw):.1f}°, '
            f'target_yaw={math.degrees(desired_yaw):.1f}°, '
            f'rotation={math.degrees(rotation):.1f}°')

        self.move_to_pose({'rotate_mobile_base': rotation})
        time.sleep(1.0)
        return True

    def drive_to_approach_distance(self, handle_xyz):
        """
        Drive forward/backward so the robot is at the correct lateral distance
        from the handle (APPROACH_DISTANCE_M).
        """
        pose = self.get_robot_pose_xya()
        if pose is None:
            return False

        rx, ry, _ = pose
        dx = handle_xyz[0] - rx
        dy = handle_xyz[1] - ry
        current_dist = math.sqrt(dx * dx + dy * dy)

        # We want to be APPROACH_DISTANCE_M away laterally
        drive_distance = current_dist - APPROACH_DISTANCE_M

        if abs(drive_distance) > 0.05:
            self.logger.info(f'Adjusting distance: drive {drive_distance:.3f}m')
            self.move_to_pose({'translate_mobile_base': drive_distance})
            time.sleep(1.0)

        return True

    # --- Force Feedback (works identically in sim and real) ---

    def get_wrist_effort(self):
        """Read current wrist extension effort from joint_states."""
        return self.wrist_effort or 0.0

    def extend_until_contact(self, target_extension, effort_threshold=EXTENSION_EFFORT_THRESHOLD):
        """
        Extend arm incrementally, stopping early if effort exceeds threshold.
        Returns (final_extension, contacted) — works in both sim and real
        because both publish effort via /stretch/joint_states.
        """
        step_size = 0.02  # 2cm increments
        current = self.wrist_position or 0.01

        while current < target_extension:
            next_pos = min(current + step_size, target_extension)
            self.move_to_pose({'wrist_extension': next_pos})
            time.sleep(EFFORT_POLL_RATE_S)

            effort = abs(self.get_wrist_effort())
            if effort > effort_threshold:
                self.logger.info(
                    f'Contact detected: effort={effort:.1f}Nm > threshold={effort_threshold}Nm '
                    f'at extension={next_pos:.3f}m')
                return next_pos, True

            current = next_pos

        time.sleep(EFFORT_SETTLE_S)
        return target_extension, False

    def retract_with_force_check(self, from_extension, pull_distance,
                                 effort_threshold=PULL_EFFORT_THRESHOLD):
        """
        Retract arm incrementally, stopping if effort exceeds threshold
        (drawer stuck/locked). Returns (retracted_distance, stalled).
        """
        step_size = 0.02
        current = from_extension
        target = max(0.01, from_extension - pull_distance)
        total_retracted = 0.0

        while current > target:
            next_pos = max(current - step_size, target)
            self.move_to_pose({'wrist_extension': next_pos})
            time.sleep(EFFORT_POLL_RATE_S)

            effort = abs(self.get_wrist_effort())
            if effort > effort_threshold:
                self.logger.info(
                    f'Pull stalled: effort={effort:.1f}Nm > threshold={effort_threshold}Nm '
                    f'after pulling {total_retracted:.3f}m')
                return total_retracted, True

            total_retracted += (current - next_pos)
            current = next_pos

        time.sleep(EFFORT_SETTLE_S)
        return total_retracted, False

    # --- Grasp and Pull ---

    def grasp_and_pull(self, handle_xyz, handle_orientation='horizontal'):
        """
        Extend arm to grasp the handle, close gripper, retract by 1 foot.
        Uses force feedback (effort from /stretch/joint_states) to detect
        contact and handle stuck drawers. Works identically in sim and real.
        """
        # 1. Set lift height to handle height
        target_lift = float(handle_xyz[2]) + LIFT_OFFSET_M
        target_lift = max(0.2, min(1.05, target_lift))
        self.logger.info(f'Setting lift to {target_lift:.3f}m (handle at z={handle_xyz[2]:.3f}m)')
        self.move_to_pose({'joint_lift': target_lift})
        time.sleep(0.5)

        # 2. Open gripper
        self.logger.info('Opening gripper')
        self.move_to_pose({'gripper_aperture': 0.09})
        time.sleep(0.3)

        # 3. Set wrist yaw based on handle orientation
        # Vertical handle → gripper horizontal (wrist_yaw = 0)
        # Horizontal handle → gripper vertical (wrist_yaw = pi/2)
        if handle_orientation == 'vertical':
            wrist_yaw = 0.0
        else:
            wrist_yaw = math.pi / 2.0
        self.logger.info(
            f'Setting wrist yaw to {math.degrees(wrist_yaw):.0f}° for {handle_orientation} handle')
        self.move_to_pose({'joint_wrist_yaw': wrist_yaw})
        time.sleep(0.3)

        # 4. Estimate extension needed from lateral distance
        pose = self.get_robot_pose_xya()
        if pose is not None:
            rx, ry, _ = pose
            dx = handle_xyz[0] - rx
            dy = handle_xyz[1] - ry
            lateral_dist = math.sqrt(dx * dx + dy * dy)
            target_extension = max(0.01, min(0.5, lateral_dist - 0.25))
        else:
            target_extension = 0.3

        # 5. Extend arm with force feedback — stops on contact
        self.logger.info(f'Extending arm toward handle (target={target_extension:.3f}m)')
        reached_ext, contacted = self.extend_until_contact(
            target_extension + 0.05,  # overshoot slightly to ensure contact
            effort_threshold=EXTENSION_EFFORT_THRESHOLD,
        )

        if contacted:
            self.logger.info(f'Handle contact at {reached_ext:.3f}m')
        else:
            self.logger.info(f'Reached target extension {reached_ext:.3f}m (no early contact)')

        # 6. Close gripper on handle
        self.logger.info('Closing gripper on handle')
        self.move_to_pose({'gripper_aperture': -0.05})
        time.sleep(0.5)

        # 7. Pull back with force monitoring — stops if drawer is stuck
        self.logger.info(
            f'Pulling drawer open (target pull={PULL_DISTANCE_M:.3f}m / 1 foot)')
        pulled_distance, stalled = self.retract_with_force_check(
            from_extension=reached_ext,
            pull_distance=PULL_DISTANCE_M,
            effort_threshold=PULL_EFFORT_THRESHOLD,
        )

        if stalled:
            self.logger.warn(
                f'Drawer appears stuck after pulling {pulled_distance:.3f}m — releasing')
        else:
            self.logger.info(f'Pulled {pulled_distance:.3f}m successfully')

        # 8. Release gripper
        self.logger.info('Releasing handle')
        self.move_to_pose({'gripper_aperture': 0.06})
        time.sleep(0.3)

        # 9. Stow arm
        self.logger.info('Stowing arm')
        self.move_to_pose({
            'wrist_extension': 0.01,
            'joint_lift': 0.5,
        })
        time.sleep(0.5)

        self.logger.info(f'Drawer pull complete (pulled {pulled_distance:.3f}m)')
        return not stalled

    # --- Open Drawer Sequence ---

    def open_drawer(self, drawer_id, drawers):
        """
        Full sequence to open a specific drawer:
        1. Orient robot so arm faces the handle
        2. Drive to correct lateral distance
        3. Extend arm, grasp handle, pull back 1 foot
        """
        if drawer_id < 0 or drawer_id >= len(drawers):
            self.logger.error(f'Invalid drawer_id={drawer_id}, have {len(drawers)} drawers')
            return False

        drawer = drawers[drawer_id]
        handle_xyz = np.array([
            drawer['handle_center_world_coordinates']['x'],
            drawer['handle_center_world_coordinates']['y'],
            drawer['handle_center_world_coordinates']['z'],
        ])
        handle_orientation = drawer.get('handle_orientation', 'horizontal')

        self.logger.info(
            f'Opening drawer #{drawer_id} at '
            f'({handle_xyz[0]:.2f}, {handle_xyz[1]:.2f}, {handle_xyz[2]:.2f}) '
            f'handle={handle_orientation}')

        # Stow arm first
        self.move_to_pose({
            'wrist_extension': 0.01,
            'joint_lift': 0.5,
            'joint_wrist_yaw': 0.0,
        })
        time.sleep(0.5)

        # Orient to face the handle (arm side)
        if not self.orient_to_handle(handle_xyz):
            return False

        # Drive to approach distance
        if not self.drive_to_approach_distance(handle_xyz):
            return False

        # Re-orient after driving (fine adjustment)
        if not self.orient_to_handle(handle_xyz):
            return False

        # Grasp and pull
        return self.grasp_and_pull(handle_xyz, handle_orientation)

    # --- Service Callbacks ---

    def trigger_open_callback(self, request, response):
        """
        Service callback: fetch drawers from search node, open the first reachable one.
        """
        self.logger.info('Open drawer triggered')

        # Fetch drawer list from search node
        if not self.get_drawers_client.wait_for_service(timeout_sec=5.0):
            response.success = False
            response.message = 'search_drawers service not available'
            return response

        req = Trigger.Request()
        future = self.get_drawers_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

        if future.result() is None:
            response.success = False
            response.message = 'Failed to get drawer list'
            return response

        result = future.result()
        if not result.success:
            response.success = False
            response.message = 'Search node returned failure'
            return response

        drawers = json.loads(result.message)
        if not drawers:
            response.success = False
            response.message = 'No drawers found'
            return response

        # Select first reachable drawer
        target_id = None
        for d in drawers:
            if d['reachable']:
                target_id = d['id']
                break

        if target_id is None:
            response.success = False
            response.message = 'No reachable drawers found'
            return response

        self.logger.info(f'Selected drawer #{target_id} to open')
        success = self.open_drawer(target_id, drawers)

        response.success = success
        response.message = f'Opened drawer #{target_id}' if success else f'Failed to open drawer #{target_id}'
        return response

    # --- Node Setup ---

    def main(self):
        hm.HelloNode.main(self, 'open_drawers', 'open_drawers',
                          wait_for_first_pointcloud=False)

        self.logger = self.get_logger()
        self.callback_group = ReentrantCallbackGroup()

        # Joint states
        self.create_subscription(
            JointState, '/stretch/joint_states',
            self.joint_states_callback, qos_profile=0,
            callback_group=self.callback_group)

        # Service client to get drawers from search node
        self.get_drawers_client = self.create_client(
            Trigger, '/search_drawers/get_drawers',
            callback_group=self.callback_group)

        # Service to trigger opening
        self.create_service(
            Trigger, '/open_drawers/trigger',
            self.trigger_open_callback,
            callback_group=self.callback_group)

        self.logger.info('OpenDrawersNode ready. Call /open_drawers/trigger to open first reachable drawer.')


def main():
    try:
        node = OpenDrawersNode()
        node.main()
        node.new_thread.join()
    except KeyboardInterrupt:
        rclpy.logging.get_logger('open_drawers').info('Shutting down')


if __name__ == '__main__':
    main()
