#!/usr/bin/env python3

import math
import threading
import time

import cv2
import numpy as np
import rclpy
import rclpy.logging
from rclpy.qos import QoSProfile
from cv_bridge import CvBridge
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image, JointState
from std_srvs.srv import Trigger

import hello_helpers.hello_misc as hm
import stretch_funmap.navigate as nv

# D435i intrinsics (from stretch_mujoco_driver camera settings)
D435I_FX = 304.24
D435I_FY = 304.07
D435I_CX = 212.0
D435I_CY = 120.0
D435I_CAMERA_FRAME = 'camera_color_optical_frame'

# Head scan config: tilt looks at counter/drawer height, pan sweeps left→right
HEAD_TILT_DRAWER_SEARCH = -0.6
HEAD_PAN_SCAN_POSITIONS = [-1.2, -0.8, -0.4, 0.0, 0.4]

# Stop this far in front of the drawer face before opening
DRAWER_APPROACH_DISTANCE_M = 0.55


class OpenDrawerNode(hm.HelloNode):

    def __init__(self):
        hm.HelloNode.__init__(self)
        self.rate = 10.0
        self.joint_states = None
        self.joint_states_lock = threading.Lock()
        self.move_base = None
        self.wrist_position = None
        self.lift_position = None
        self.bridge = CvBridge()
        self.color_image = None
        self.depth_image = None
        self.image_lock = threading.Lock()

    def joint_states_callback(self, joint_states):
        with self.joint_states_lock:
            self.joint_states = joint_states
        wrist_position, wrist_velocity, wrist_effort = hm.get_wrist_state(joint_states)
        self.wrist_position = wrist_position
        lift_position, lift_velocity, lift_effort = hm.get_lift_state(joint_states)
        self.lift_position = lift_position

    def color_image_callback(self, msg):
        print('[OpenDrawer]   got image', flush=True)
        print(f'[OpenDrawer]   image lock: {self.image_lock}', flush=True)
        with self.image_lock:
            self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def depth_image_callback(self, msg):
        with self.image_lock:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')

    def detect_drawer_in_image(self, color_image, depth_image):
        """
        Detect a drawer-like rectangle in the RGB image using edge/contour detection.
        Returns (center_u, center_v, depth_m) or None if not found.
        """
        h, w = color_image.shape[:2]
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_score = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1500:
                continue

            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = cw / max(ch, 1)

            # Drawers are wide horizontal rectangles
            if aspect < 1.5 or aspect > 10.0:
                continue

            # Ignore things near the very top of the frame (ceiling/upper cabinets)
            cy_box = y + ch / 2
            if cy_box < h * 0.25:
                continue

            # Prefer solid/rectangular contours
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area < 1:
                continue
            solidity = area / hull_area
            if solidity < 0.65:
                continue

            score = area * solidity
            if score > best_score:
                cx_pix = x + cw // 2
                cy_pix = y + ch // 2

                if depth_image is not None:
                    roi = depth_image[
                        max(0, cy_pix - 8): min(h, cy_pix + 8),
                        max(0, cx_pix - 8): min(w, cx_pix + 8)
                    ]
                    valid = roi[np.isfinite(roi) & (roi > 0.1) & (roi < 5.0)]
                    if len(valid) > 0:
                        best_score = score
                        best = (cx_pix, cy_pix, float(np.median(valid)))

        return best

    def pixel_to_base_link(self, u, v, depth_m):
        """
        Back-project a pixel (u, v) at depth_m into base_link frame.
        Returns np.array([x, y, z]) or None on TF failure.
        """
        X_cam = (u - D435I_CX) * depth_m / D435I_FX
        Y_cam = (v - D435I_CY) * depth_m / D435I_FY
        Z_cam = depth_m
        p_cam = np.array([X_cam, Y_cam, Z_cam, 1.0])

        cam_to_base, _ = hm.get_p1_to_p2_matrix(
            D435I_CAMERA_FRAME, 'base_link', self.tf2_buffer, timeout_s=1.0
        )
        if cam_to_base is None:
            self.logger.error('pixel_to_base_link: TF lookup failed')
            return None

        return (cam_to_base @ p_cam)[:3]

    def scan_for_drawer(self):
        """
        Sweep the head pan across HEAD_PAN_SCAN_POSITIONS at a downward tilt and
        look for a drawer in each frame.  Returns (x, y) in base_link or None.
        """
        total = len(HEAD_PAN_SCAN_POSITIONS)
        print(f'[OpenDrawer]   Tilting head down (tilt={HEAD_TILT_DRAWER_SEARCH:.2f} rad)...', flush=True)
        self.move_to_pose({'joint_head_tilt': HEAD_TILT_DRAWER_SEARCH})
        time.sleep(0.5)

        for i, pan_angle in enumerate(HEAD_PAN_SCAN_POSITIONS, 1):
            print(f'[OpenDrawer]   Scan position {i}/{total}: pan={pan_angle:.2f} rad', flush=True)
            self.move_to_pose({'joint_head_pan': pan_angle})
            time.sleep(1.0)

            with self.image_lock:
                color = self.color_image.copy() if self.color_image is not None else None
                depth = self.depth_image.copy() if self.depth_image is not None else None

            if color is None:
                print('[OpenDrawer]   No image received yet, skipping', flush=True)
                continue

            result = self.detect_drawer_in_image(color, depth)
            if result is not None:
                u, v, depth_m = result
                pos = self.pixel_to_base_link(u, v, depth_m)
                if pos is not None:
                    print(
                        f'[OpenDrawer]   Drawer found at pixel ({u}, {v}), '
                        f'depth={depth_m:.2f} m -> base_link x={pos[0]:.2f} y={pos[1]:.2f}',
                        flush=True
                    )
                    return pos[:2]
            else:
                print('[OpenDrawer]   No drawer detected at this position', flush=True)

        print('[OpenDrawer]   No drawer found during head scan', flush=True)
        self.logger.warn('No drawer found during head scan')
        return None

    def navigate_to_drawer(self, drawer_xy):
        """
        Rotate in place to face (x, y), then drive to DRAWER_APPROACH_DISTANCE_M away.
        """
        x, y = float(drawer_xy[0]), float(drawer_xy[1])
        angle = math.atan2(y, x)
        distance = math.sqrt(x ** 2 + y ** 2)
        drive_dist = distance - DRAWER_APPROACH_DISTANCE_M

        print(f'[OpenDrawer]   Rotating {math.degrees(angle):.1f} deg to face drawer...', flush=True)
        self.move_to_pose({'rotate_mobile_base': angle})
        time.sleep(0.5)

        if drive_dist > 0.05:
            print(f'[OpenDrawer]   Driving {drive_dist:.2f} m toward drawer...', flush=True)
            self.move_to_pose({'translate_mobile_base': drive_dist})
            time.sleep(0.5)
        else:
            print(f'[OpenDrawer]   Already within approach distance, no driving needed', flush=True)

    def _log_step(self, step, total, msg):
        print(f'\n[OpenDrawer] Step {step}/{total}: {msg}', flush=True)
        self.logger.info(f'Step {step}/{total}: {msg}')

    def extend_hook_until_contact(self):
        print('[OpenDrawer]   -> Extending arm until contact with drawer surface...', flush=True)
        max_extension_m = 0.5
        max_reach_m = 0.4
        extension_m = self.wrist_position + max_reach_m
        extension_m = min(extension_m, max_extension_m)
        extension_contact_effort = 42.0 #18.5 #effort_pct #42.0 #40.0 from funmap
        pose = {'wrist_extension': (extension_m, extension_contact_effort)}
        self.move_to_pose(pose, custom_contact_thresholds=True)
        print('[OpenDrawer]   -> Contact detected', flush=True)

    def lower_hook_until_contact(self):
        print('[OpenDrawer]   -> Lowering hook until contact with drawer handle...', flush=True)
        max_drop_m = 0.15
        lift_m = self.lift_position - max_drop_m
        lift_contact_effort = 42.0 #32.5 #effort_pct #18.0 #20.0 #20.0 from funmap
        pose = {'joint_lift': (lift_m, lift_contact_effort)}
        self.move_to_pose(pose, custom_contact_thresholds=True)

        use_correction = True
        if use_correction:
            # raise due to drop down after contact detection
            time.sleep(0.2) # wait for new lift position
            lift_m = self.lift_position + 0.015
            pose = {'joint_lift': lift_m}
            self.move_to_pose(pose)
            time.sleep(0.2) # wait for new lift position
        print('[OpenDrawer]   -> Hook seated on handle', flush=True)

    def raise_hook_until_contact(self):
        print('[OpenDrawer]   -> Raising hook until contact with drawer handle...', flush=True)
        max_raise_m = 0.15
        lift_m = self.lift_position + max_raise_m
        lift_contact_effort = 42.0 #effort_pct
        pose = {'joint_lift': (lift_m, lift_contact_effort)}
        self.move_to_pose(pose, custom_contact_thresholds=True)

        use_correction = True
        if use_correction:
            # raise due to drop down after contact detection
            time.sleep(0.5) # wait for new lift position
            lift_m = self.lift_position + 0.01 #0.015
            pose = {'joint_lift': lift_m}
            self.move_to_pose(pose)
            time.sleep(0.5) # wait for new lift position
        print('[OpenDrawer]   -> Hook seated on handle', flush=True)

    def backoff_from_surface(self):
        print('[OpenDrawer]   -> Backing off slightly from surface...', flush=True)
        if self.wrist_position is not None:
            wrist_target_m = self.wrist_position - 0.005
            pose = {'wrist_extension': wrist_target_m}
            self.move_to_pose(pose)
            return True
        else:
            self.logger.error('backoff_from_surface: self.wrist_position is None!')
            return False

    def pull_open(self):
        print('[OpenDrawer]   -> Pulling drawer open...', flush=True)
        if self.wrist_position is not None:
            max_extension_m = 0.5
            extension_m = self.wrist_position - 0.2
            extension_m = min(extension_m, max_extension_m)
            extension_m = max(0.01, extension_m)
            extension_contact_effort = 64.4 #effort_pct #100.0 #40.0 from funmap
            pose = {'wrist_extension': (extension_m, extension_contact_effort)}
            self.move_to_pose(pose, custom_contact_thresholds=True)
            print('[OpenDrawer]   -> Drawer pulled open', flush=True)
            return True
        else:
            self.logger.error('pull_open: self.wrist_position is None!')
            return False

    def push_closed(self):
        print('[OpenDrawer]   -> Pushing drawer closed...', flush=True)
        if self.wrist_position is not None:
            wrist_target_m = self.wrist_position + 0.22
            pose = {'wrist_extension': wrist_target_m}
            self.move_to_pose(pose)
            return True
        else:
            self.logger.error('pull_open: self.wrist_position is None!')
            return False

    def move_to_initial_configuration(self):
        print('[OpenDrawer]   -> Moving to initial hook configuration...', flush=True)
        initial_pose = {'wrist_extension': 0.01,
                        'joint_wrist_yaw': 1.570796327,
                        'gripper_aperture': 0.0}
        self.move_to_pose(initial_pose)

    def trigger_open_drawer_down_callback(self, request, response):
        return self.open_drawer('down')

    def trigger_open_drawer_up_callback(self, request, response):
        return self.open_drawer('up')

    MAX_BACKUP_ATTEMPTS = 5
    BACKUP_DISTANCE_M = 1.0

    def find_and_open_drawer(self):
        """
        Full autonomous sequence:
          1. Stow the arm for travel
          2. Scan the head to find a drawer; back up 1 m and retry if not found
          3. Navigate to face the drawer
          4. Re-scan for a more accurate position after navigation
          5. Open the drawer
        """
        print('\n[OpenDrawer] ========================================', flush=True)
        print('[OpenDrawer] Starting find-and-open-drawer sequence', flush=True)
        print('[OpenDrawer] ========================================', flush=True)

        self._log_step(1, 5, 'Stowing arm for travel')
        self.move_to_pose({
            'wrist_extension': 0.01,
            'joint_lift': 0.5,
            'joint_wrist_yaw': 0.0,
        })
        time.sleep(0.5)

        self._log_step(2, 5, 'Scanning for drawer')
        drawer_pos = None
        for attempt in range(1, self.MAX_BACKUP_ATTEMPTS + 1):
            print(f'[OpenDrawer]   Scan attempt {attempt}/{self.MAX_BACKUP_ATTEMPTS}', flush=True)
            drawer_pos = self.scan_for_drawer()
            if drawer_pos is not None:
                break
            if attempt < self.MAX_BACKUP_ATTEMPTS:
                print(
                    f'[OpenDrawer]   Drawer not found — backing up {self.BACKUP_DISTANCE_M:.1f} m '
                    f'and retrying ({attempt}/{self.MAX_BACKUP_ATTEMPTS - 1} backups done)',
                    flush=True
                )
                self.move_to_pose({'translate_mobile_base': -self.BACKUP_DISTANCE_M})
                time.sleep(1.0)

        if drawer_pos is None:
            msg = (
                f'Could not find a drawer after {self.MAX_BACKUP_ATTEMPTS} scan attempts '
                f'({(self.MAX_BACKUP_ATTEMPTS - 1) * self.BACKUP_DISTANCE_M:.1f} m backed up)'
            )
            print(f'[OpenDrawer] FAILED: {msg}\n', flush=True)
            return Trigger.Response(success=False, message=msg)

        self._log_step(3, 5, 'Navigating to drawer')
        self.navigate_to_drawer(drawer_pos)

        self._log_step(4, 5, 'Re-scanning for precise alignment')
        time.sleep(1.0)
        refined_pos = self.scan_for_drawer()
        if refined_pos is not None:
            self.navigate_to_drawer(refined_pos)
        else:
            print('[OpenDrawer]   Could not refine position, proceeding with initial estimate', flush=True)

        self._log_step(5, 5, 'Opening drawer')
        result = self.open_drawer('down')
        if result.success:
            print('\n[OpenDrawer] ========================================', flush=True)
            print('[OpenDrawer] SUCCESS: Drawer opened!', flush=True)
            print('[OpenDrawer] ========================================\n', flush=True)
        else:
            print(f'\n[OpenDrawer] FAILED: {result.message}\n', flush=True)
        return result

    def trigger_find_and_open_drawer_callback(self, request, response):
        return self.find_and_open_drawer()


    def open_drawer(self, direction):
        self.move_to_initial_configuration()

        self.extend_hook_until_contact()
        success = self.backoff_from_surface()
        if not success:
            return Trigger.Response(
                success=False,
                message='Failed to backoff from the surface.'
            )

        if direction == 'down':
            self.lower_hook_until_contact()
        elif direction == 'up':
            self.raise_hook_until_contact()

        success = self.pull_open()
        if not success:
            return Trigger.Response(
                success=False,
                message='Failed to pull open the drawer.'
            )

        push_drawer_closed = False
        if push_drawer_closed:
            time.sleep(3.0)
            self.push_closed()

        return Trigger.Response(success=True, message='Completed opening the drawer!')


    def main(self):
        hm.HelloNode.main(self, 'open_drawer', 'open_drawer', wait_for_first_pointcloud=False)

        self.logger = self.get_logger()
        self.move_base = nv.MoveBase(self)
        self.callback_group = ReentrantCallbackGroup()

        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.joint_states_subscriber = self.create_subscription(
            JointState, '/stretch/joint_states',
            self.joint_states_callback, qos_profile=reliable_qos,
            callback_group=self.callback_group
        )

        self.create_subscription(
            Image, '/camera/color/image_raw',
            self.color_image_callback, qos_profile=10,
            callback_group=self.callback_group
        )
        self.create_subscription(
            Image, '/camera/depth/image_rect_raw',
            self.depth_image_callback, qos_profile=1,
            callback_group=self.callback_group
        )

        self.trigger_open_drawer_service = self.create_service(
            Trigger, '/open_drawer/trigger_open_drawer_down',
            self.trigger_open_drawer_down_callback,
            callback_group=self.callback_group
        )
        self.trigger_open_drawer_service = self.create_service(
            Trigger, '/open_drawer/trigger_open_drawer_up',
            self.trigger_open_drawer_up_callback,
            callback_group=self.callback_group
        )
        self.create_service(
            Trigger, '/open_drawer/trigger_find_and_open_drawer',
            self.trigger_find_and_open_drawer_callback,
            callback_group=self.callback_group
        )

        self.trigger_reach_until_contact_service = self.create_client(Trigger, '/funmap/trigger_reach_until_contact', callback_group=self.callback_group)
        self.trigger_reach_until_contact_service.wait_for_service()
        self.logger.info('Node ' + self.get_name() + ' connected to /funmap/trigger_reach_until_contact.')

        self.trigger_lower_until_contact_service = self.create_client(Trigger, '/funmap/trigger_lower_until_contact', callback_group=self.callback_group)
        self.trigger_lower_until_contact_service.wait_for_service()
        self.logger.info('Node ' + self.get_name() + ' connected to /funmap/trigger_lower_until_contact.')

        self.logger.info('Open drawer node ready. Call /open_drawer/trigger_find_and_open_drawer to start autonomous sequence.')


def main():
    try:
        node = OpenDrawerNode()
        node.main()

        node.new_thread.join()
    except KeyboardInterrupt:
        rclpy.logging.get_logger('open_drawer').info('interrupt received, so shutting down')

if __name__ == '__main__':
    main()
