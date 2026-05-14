#!/usr/bin/env python3
"""Node 3: Navigate to Drawer and Open It.

Given a list of detected drawers from Node 2, this node:
  1. Sorts drawers by ranking (or distance if ranking is TBD)
  2. Navigates the robot base to 1m away from the target drawer
  3. Rotates 90 degrees to face the drawer with the arm
  4. Extends the arm toward the handle grasp location
  5. Uses pressure/force feedback to detect contact, then closes gripper
  6. Pulls back (toward robot) until force threshold is met
  7. Releases the gripper

Works in both MuJoCo simulation and on the real Stretch3 robot.

Publishes:
  - /navigate_open/path (nav_msgs/Path): planned path to drawer (for RViz)
  - /navigate_open/status (std_msgs/String): current state

Subscribes:
  - /drawer_detections_json (via service call to Node 2)

Services:
  - /navigate_open/execute (NavigateToDrawer): trigger navigation + open
  - /navigate_open/stop (std_srvs/Trigger): abort current operation

Parameters:
  - approach_distance: distance to stop from drawer (default 1.0m)
  - grasp_force_threshold: force (N) to detect contact (default 5.0)
  - pull_force_threshold: force (N) to stop pulling (default 15.0)
  - gripper_close_effort: effort for gripper close (default -50.0)
  - pull_speed: arm retraction speed m/s (default 0.02)
  - use_sim: whether running in simulation (default false)
"""

import json
import math
import threading
import time
from enum import Enum

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from rclpy.action import ActionClient
import tf2_ros


class OpenState(Enum):
    IDLE = "idle"
    SELECTING = "selecting_drawer"
    NAVIGATING = "navigating_to_drawer"
    ALIGNING = "aligning_to_drawer"
    APPROACHING = "approaching_handle"
    GRASPING = "grasping_handle"
    PULLING = "pulling_drawer"
    RELEASING = "releasing"
    COMPLETE = "complete"
    FAILED = "failed"


class NavigateOpenNode(Node):
    """Navigate to a drawer and open it using force-feedback grasping."""

    def __init__(self):
        super().__init__("navigate_open_node")

        # Parameters
        self.declare_parameter("approach_distance", 0.45)
        self.declare_parameter("grasp_force_threshold", 5.0)
        self.declare_parameter("pull_force_threshold", 15.0)
        self.declare_parameter("gripper_close_effort", -50.0)
        self.declare_parameter("pull_speed", 0.02)
        self.declare_parameter("use_sim", False)
        self.declare_parameter("arm_extension_speed", 0.01)
        self.declare_parameter("max_pull_distance", 0.4)

        self.approach_distance = self.get_parameter("approach_distance").value
        self.grasp_force_threshold = self.get_parameter("grasp_force_threshold").value
        self.pull_force_threshold = self.get_parameter("pull_force_threshold").value
        self.gripper_close_effort = self.get_parameter("gripper_close_effort").value
        self.pull_speed = self.get_parameter("pull_speed").value
        self.use_sim = self.get_parameter("use_sim").value
        self.arm_extension_speed = self.get_parameter("arm_extension_speed").value
        self.max_pull_distance = self.get_parameter("max_pull_distance").value

        # State
        self.state = OpenState.IDLE
        self.stop_requested = False
        self.current_joint_state = None
        self.current_effort = {}
        self.operation_thread = None
        self.opened_drawers = {}

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.cb_group = ReentrantCallbackGroup()

        # Publishers
        self.path_pub = self.create_publisher(Path, "/navigate_open/path", 10)
        self.status_pub = self.create_publisher(
            String, "/navigate_open/status", 10
        )
        self.opened_drawers_pub = self.create_publisher(
            String, "/navigate_open/opened_drawers_json", 10
        )
        self.trajectory_client = ActionClient(
            self, FollowJointTrajectory,
            "/stretch_controller/follow_joint_trajectory",
        )

        # Subscribers
        self.create_subscription(
            JointState, "/joint_states", self.joint_state_callback, 10
        )

        # Services
        self.create_service(
            Trigger, "/navigate_open/execute",
            self.execute_callback,
            callback_group=self.cb_group,
        )
        self.create_service(
            Trigger, "/navigate_open/stop",
            self.stop_callback,
            callback_group=self.cb_group,
        )

        # Receive chosen drawer from detection node on automata-3
        self._chosen_drawer_json = None
        self.create_subscription(
            String, "/detection/chosen_drawer_json",
            self._chosen_drawer_callback, 10
        )

        # Fallback: receive all detections via topic (works across rosbridge)
        self.create_subscription(
            String, "/drawer_detections_json",
            self._drawer_detections_callback, 10
        )

        # Mode switching (sim needs position mode for base translate/rotate)
        self.position_mode_client = self.create_client(
            Trigger, "/switch_to_position_mode"
        )
        self.navigation_mode_client = self.create_client(
            Trigger, "/switch_to_navigation_mode"
        )

        # Monitor driver mode
        self._driver_mode = None
        self.create_subscription(String, "/mode", self._mode_callback, 10)

        # Close-drawer command from detection node
        self.create_subscription(
            String, "/detection/close_drawer_json",
            self._close_drawer_callback, 10
        )

        # Status timer
        self.create_timer(0.5, self.publish_status)

        self.get_logger().info("Navigate and open node initialized")

    def _chosen_drawer_callback(self, msg: String):
        self._chosen_drawer_json = msg.data
        self.get_logger().info(f"Received chosen drawer from detection node")

    def _drawer_detections_callback(self, msg: String):
        pass

    def _mode_callback(self, msg: String):
        if self._driver_mode != msg.data:
            self.get_logger().info(f"Driver mode: {msg.data}")
            self._driver_mode = msg.data

    # ─── Callbacks ────────────────────────────────────────────────────

    def joint_state_callback(self, msg: JointState):
        """Track joint positions and efforts for force feedback."""
        self.current_joint_state = msg
        if not hasattr(self, '_js_logged'):
            self._js_logged = True
            self.get_logger().debug(
                f"joint_states source: names={list(msg.name)}, "
                f"effort_len={len(msg.effort)}, "
                f"has_nonzero_effort={any(abs(e) > 0.01 for e in msg.effort)}"
            )
        if msg.effort:
            for name, effort in zip(msg.name, msg.effort):
                self.current_effort[name] = effort
            arm_efforts = {n: e for n, e in zip(msg.name, msg.effort)
                          if n.startswith("joint_arm_l")}
            if any(abs(e) > 1.0 for e in arm_efforts.values()):
                self.get_logger().debug(
                    f"Arm effort spike: {arm_efforts}",
                    throttle_duration_sec=2.0,
                )
        else:
            self.get_logger().debug(
                "joint_states has EMPTY effort array!",
                throttle_duration_sec=10.0,
            )

    def execute_callback(self, request, response):
        """Execute the full navigate-and-open sequence."""
        if self.state not in (OpenState.IDLE, OpenState.COMPLETE, OpenState.FAILED):
            response.success = False
            response.message = f"Already busy (state={self.state.value})"
            return response

        self.stop_requested = False
        self.operation_thread = threading.Thread(
            target=self._execute_pipeline, daemon=True
        )
        self.operation_thread.start()
        response.success = True
        response.message = "Pipeline started"
        return response

    def stop_callback(self, request, response):
        """Stop current operation."""
        self.stop_requested = True
        self._stop_robot()
        response.success = True
        response.message = "Stop requested"
        return response

    # ─── Main pipeline ────────────────────────────────────────────────

    def _execute_pipeline(self):
        """Full sequence: select → navigate → align → approach → grasp → pull → release."""
        try:
            # Step 1: Get drawer list from Node 2
            self._set_state(OpenState.SELECTING)
            drawer = self._select_target_drawer()
            if drawer is None:
                self._set_state(OpenState.FAILED)
                self.get_logger().error("No reachable drawers available")
                return

            handle_pos = np.array([
                drawer["handle_center_world"]["x"],
                drawer["handle_center_world"]["y"],
                drawer["handle_center_world"]["z"],
            ])
            orientation = drawer["handle_orientation"]
            self.get_logger().info(
                f"Target drawer: {drawer['drawer_id']} at "
                f"({handle_pos[0]:.2f}, {handle_pos[1]:.2f}, {handle_pos[2]:.2f}), "
                f"orientation={orientation}"
            )

            # Step 2: Switch to position mode, retract arm, and navigate to approach pose
            if not self._switch_to_position_mode():
                self.get_logger().error("Cannot proceed without position mode")
                self._set_state(OpenState.FAILED)
                return
            time.sleep(0.5)

            self._set_state(OpenState.NAVIGATING)
            corners = drawer.get("drawer_corners_world")
            self._retract_arm()
            nav_success = self._navigate_to_approach_pose(handle_pos, corners)
            if not nav_success or self.stop_requested:
                self._set_state(OpenState.FAILED)
                return

            # Step 3: Open gripper, orient toward handle
            self._open_gripper()
            time.sleep(0.5)
            self._orient_gripper_toward(handle_pos, orientation)

            # Step 4: Extend arm to handle location
            self._set_state(OpenState.GRASPING)
            self._extend_to_point(handle_pos)

            # Step 9: Close gripper
            self._close_gripper()
            time.sleep(1.0)

            # Step 10: Pull drawer open
            self._set_state(OpenState.PULLING)
            pull_success, pull_distance = self._pull_drawer()

            # Record opened handle position (gripper is at the handle)
            opened_handle_pos = self._get_gripper_world_pos()

            # Step 11: Release
            self._set_state(OpenState.RELEASING)
            self._open_gripper()
            time.sleep(0.5)

            # Retract arm
            self._retract_arm()

            # Step 12: Look at the opened drawer
            self._look_at_drawer(handle_pos, drawer.get("drawer_corners_world"))

            if pull_success:
                self._record_opened_drawer(drawer, handle_pos, opened_handle_pos, pull_distance)
                self._set_state(OpenState.COMPLETE)
                self.get_logger().info("Drawer opened successfully!")
            else:
                self._set_state(OpenState.FAILED)
                self.get_logger().warn("Pull did not reach force threshold")

        except Exception as e:
            self.get_logger().error(f"Pipeline failed: {e}")
            self._set_state(OpenState.FAILED)
            self._stop_robot()
        finally:
            self._switch_to_navigation_mode()

    # ─── Drawer selection ─────────────────────────────────────────────

    def _select_target_drawer(self):
        """Return the drawer chosen by the detection node on automata-3."""
        if self._chosen_drawer_json is None:
            self.get_logger().error(
                "No chosen drawer received from detection node. "
                "Call /detection/choose_drawer on automata-3 first."
            )
            return None

        return json.loads(self._chosen_drawer_json)

    def _record_opened_drawer(self, drawer: dict, closed_handle_pos: np.ndarray,
                               opened_handle_pos, pull_distance: float = 0.0):
        """Record a successfully opened drawer and notify the detection node."""
        drawer_id = drawer["drawer_id"]

        gripper_pos = None
        if opened_handle_pos is not None:
            gripper_pos = {
                "x": float(opened_handle_pos[0]),
                "y": float(opened_handle_pos[1]),
                "z": float(opened_handle_pos[2]),
            }

        entry = {
            "drawer_id": drawer_id,
            "closed_handle": {
                "x": float(closed_handle_pos[0]),
                "y": float(closed_handle_pos[1]),
                "z": float(closed_handle_pos[2]),
            },
            "closed_corners": drawer.get("drawer_corners_world"),
            "handle_orientation": drawer.get("handle_orientation", "horizontal"),
            "gripper_pos": gripper_pos,
            "pull_distance": float(pull_distance),
        }

        self.opened_drawers[drawer_id] = entry

        self.get_logger().info(
            f"Recorded opened drawer {drawer_id}: "
            f"closed handle=({closed_handle_pos[0]:.3f}, {closed_handle_pos[1]:.3f}, {closed_handle_pos[2]:.3f}), "
            f"gripper_pos={gripper_pos}, pull_distance={pull_distance:.3f}m"
        )

        msg = String()
        msg.data = json.dumps(entry)
        self.opened_drawers_pub.publish(msg)

    def _close_drawer_callback(self, msg: String):
        """Handle close-drawer command from detection node."""
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().warn(f"Bad close_drawer JSON: {e}")
            return
        self.get_logger().info(f"Close drawer command received for {data.get('drawer_id')}")
        thread = threading.Thread(target=self._close_drawer_pipeline, args=(data,), daemon=True)
        thread.start()

    def _speak(self, text: str):
        """Speak text using espeak (non-blocking)."""
        self.get_logger().info(f"Speaking: {text}")
        try:
            import subprocess
            result = subprocess.run(
                ["espeak", "-s", "140", text],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                self.get_logger().warn(f"espeak failed: {result.stderr}")
        except FileNotFoundError:
            self.get_logger().warn("espeak not installed — install with: sudo apt install espeak")
        except Exception as e:
            self.get_logger().warn(f"Speech failed: {e}")

    def _close_drawer_pipeline(self, data: dict):
        """Close a drawer: extend to opened handle, grab, push back, release."""
        drawer_id = data.get("drawer_id", "?")
        pull_distance = data.get("pull_distance", 0.0)
        handle_pos_d = data.get("opened_handle")
        items = data.get("items", [])

        if items:
            labels = [i["label"] for i in items]
            unique = list(dict.fromkeys(labels))
            self._speak(f"I found {', '.join(unique)} in the drawer")
        else:
            self._speak("The drawer is empty")

        if not handle_pos_d or pull_distance <= 0:
            self.get_logger().warn(
                f"Cannot close drawer {drawer_id}: "
                f"handle={handle_pos_d}, pull_distance={pull_distance}"
            )
            return

        handle_pos = np.array([handle_pos_d["x"], handle_pos_d["y"], handle_pos_d["z"]])

        try:
            if not self._switch_to_position_mode():
                self.get_logger().error("Cannot switch to position mode for close")
                return
            time.sleep(0.5)

            self._extend_to_point(handle_pos)

            self._close_gripper()
            time.sleep(1.0)

            current_ext = self._get_current_extension()
            push_target = min(current_ext + pull_distance, 0.52)
            self.get_logger().info(
                f"Pushing drawer {drawer_id}: ext {current_ext:.3f} → {push_target:.3f}m"
            )
            self._send_joint_command("wrist_extension", push_target, duration_sec=4)
            time.sleep(2.0)

            self._open_gripper()
            time.sleep(0.5)
            self._retract_arm()

            self.get_logger().info(f"Drawer {drawer_id} closed successfully")
        except Exception as e:
            self.get_logger().error(f"Close drawer failed: {e}")
        finally:
            self._switch_to_navigation_mode()

    # ─── Navigation ───────────────────────────────────────────────────

    def _navigate_to_approach_pose(self, handle_pos: np.ndarray,
                                   corners_world=None) -> bool:
        """Navigate robot so the mast is directly in front of the handle,
        perpendicular to the drawer face, with the arm facing the handle.

        Computes where the mast should end up (approach_distance from
        handle along face normal), then back-computes the base_link
        target position accounting for the mast offset at the final
        heading. Executes:
          1. Rotate to face the base_link target
          2. Drive straight to the base_link target (with stall detection)
          3. Rotate so the arm (left side) faces the drawer

        Works regardless of the robot's starting position/orientation.
        """
        robot_pose = self._get_robot_pose()
        if robot_pose is None:
            return False

        face_normal = self._compute_drawer_face_normal(corners_world, handle_pos)
        normal_angle = math.atan2(face_normal[1], face_normal[0])

        # Where the mast should end up
        mast_target_x = handle_pos[0] + face_normal[0] * self.approach_distance
        mast_target_y = handle_pos[1] + face_normal[1] * self.approach_distance

        # Final heading: arm faces drawer (left side toward drawer)
        desired_heading = normal_angle - math.pi / 2

        # Back-compute base_link position from mast target at final heading
        cos_h = math.cos(desired_heading)
        sin_h = math.sin(desired_heading)
        base_target_x = mast_target_x - (cos_h * self.MAST_OFFSET_X - sin_h * self.MAST_OFFSET_Y)
        base_target_y = mast_target_y - (sin_h * self.MAST_OFFSET_X + cos_h * self.MAST_OFFSET_Y)

        self._publish_path(robot_pose, (base_target_x, base_target_y))
        self.get_logger().info(
            f"Mast target: ({mast_target_x:.2f}, {mast_target_y:.2f}), "
            f"base target: ({base_target_x:.2f}, {base_target_y:.2f}), "
            f"face normal {math.degrees(normal_angle):.1f} deg"
        )

        # Phase 1: Rotate to face the base_link target
        dx = base_target_x - robot_pose[0]
        dy = base_target_y - robot_pose[1]
        travel_dist = math.sqrt(dx * dx + dy * dy)

        if travel_dist > 0.1:
            angle_to_target = math.atan2(dy, dx)
            current_yaw = self._get_robot_yaw()
            if current_yaw is not None:
                angle_diff = (angle_to_target - current_yaw + math.pi) % (2 * math.pi) - math.pi
                if abs(angle_diff) > 0.05:
                    self.get_logger().info(
                        f"Phase 1: Rotating {math.degrees(angle_diff):.1f} deg to face target"
                    )
                    self._rotate_in_place(angle_diff)
                    if self.stop_requested:
                        return False

            # Phase 2: Drive to base_link target with stall detection
            stall_timeout = 10.0
            last_remaining = float("inf")
            last_progress_time = time.time()

            while not self.stop_requested:
                robot_pose = self._get_robot_pose()
                if robot_pose is None:
                    time.sleep(0.2)
                    continue

                dx = base_target_x - robot_pose[0]
                dy = base_target_y - robot_pose[1]
                remaining = math.sqrt(dx * dx + dy * dy)

                if remaining < 0.1:
                    self.get_logger().info("Reached base target")
                    break

                if last_remaining - remaining > 0.02:
                    last_progress_time = time.time()
                    last_remaining = remaining
                elif time.time() - last_progress_time > stall_timeout:
                    self.get_logger().warn(
                        f"Navigation stalled at {remaining:.2f}m from target"
                    )
                    return False

                step = min(remaining, 0.2)
                self.get_logger().info(f"Phase 2: Driving {step:.2f}m (remaining {remaining:.2f}m)")
                if not self._send_joint_command("translate_mobile_base", step, duration_sec=3):
                    self.get_logger().error("Translate command failed — check driver mode")
                    return False
                time.sleep(1.0)

            if self.stop_requested:
                return False

        # Phase 3: Rotate so the arm faces the drawer
        current_yaw = self._get_robot_yaw()
        if current_yaw is not None:
            angle_diff = (desired_heading - current_yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(angle_diff) > 0.05:
                self.get_logger().info(
                    f"Phase 3: Rotating {math.degrees(angle_diff):.1f} deg to align arm toward drawer"
                )
                self._rotate_in_place(angle_diff)

        self.get_logger().info("Mast aligned in front of handle")
        return True

    def _compute_drawer_face_normal(self, corners_world, handle_pos):
        """Compute the outward-facing normal of the drawer face.

        corners_world: list of 4 dicts with x,y,z (top-left, top-right,
                       bottom-right, bottom-left of the drawer bbox).
        Falls back to robot-to-handle direction if corners are unavailable.
        """
        if corners_world is not None and len(corners_world) >= 3:
            pts = np.array([[c["x"], c["y"], c["z"]] for c in corners_world])
            # Two edge vectors of the drawer face
            v1 = pts[1] - pts[0]  # top edge
            v2 = pts[3] - pts[0]  # left edge
            normal = np.cross(v1, v2)
            normal_2d = normal[:2]
            norm = np.linalg.norm(normal_2d)
            if norm > 1e-6:
                normal_2d = normal_2d / norm
                # Ensure normal points toward the robot (outward from drawer)
                mast_pose = self._get_mast_pose()
                if mast_pose is not None:
                    center = pts.mean(axis=0)[:2]
                    to_robot = np.array(mast_pose) - center
                    if np.dot(normal_2d, to_robot) < 0:
                        normal_2d = -normal_2d
                return normal_2d

        # Fallback: use mast-to-handle direction
        self.get_logger().warn("No valid drawer corners — falling back to robot-to-handle direction for approach angle")
        mast_pose = self._get_mast_pose()
        if mast_pose is not None:
            dx = mast_pose[0] - handle_pos[0]
            dy = mast_pose[1] - handle_pos[1]
            norm = math.sqrt(dx * dx + dy * dy)
            if norm > 1e-6:
                return np.array([dx / norm, dy / norm])
        return np.array([1.0, 0.0])

    # ─── Head control ─────────────────────────────────────────────────

    def _look_at_drawer(self, handle_pos: np.ndarray, corners_world=None):
        """Pan and tilt the head camera to look at the drawer center.

        Computes direction from the mast/head to the target in the
        robot's local frame, then derives pan and tilt joint angles.
        Pan = 0 is the robot's forward direction.
        Tilt = 0 is horizontal, negative looks down.
        """
        if corners_world and len(corners_world) >= 4:
            pts = np.array([[c["x"], c["y"], c["z"]] for c in corners_world])
            target = pts.mean(axis=0)
        else:
            target = handle_pos.copy()

        try:
            mast_pose = self._get_mast_pose()
            robot_yaw = self._get_robot_yaw()
            if mast_pose is None or robot_yaw is None:
                self.get_logger().warn("Cannot look at drawer: no mast pose")
                return

            # Target direction in world frame, from mast
            dx_world = target[0] - mast_pose[0]
            dy_world = target[1] - mast_pose[1]

            # Rotate into base_link frame (base_link X = forward)
            cos_yaw = math.cos(robot_yaw)
            sin_yaw = math.sin(robot_yaw)
            dx_base = cos_yaw * dx_world + sin_yaw * dy_world
            dy_base = -sin_yaw * dx_world + cos_yaw * dy_world

            dz = target[2] - self.MAST_HEIGHT

            pan = math.atan2(dy_base, dx_base)
            horiz_dist = math.sqrt(dx_base * dx_base + dy_base * dy_base)
            tilt = math.atan2(dz, horiz_dist)

            self.get_logger().info(
                f"Looking at drawer: base=({dx_base:.2f}, {dy_base:.2f}), "
                f"dz={dz:.2f}, pan={math.degrees(pan):.1f} deg, "
                f"tilt={math.degrees(tilt):.1f} deg"
            )
            self._send_joint_command("joint_head_pan", pan, duration_sec=2)
            self._send_joint_command("joint_head_tilt", tilt, duration_sec=2)
            time.sleep(2.0)

        except Exception as e:
            self.get_logger().warn(f"Cannot look at drawer: {e}")

    # ─── Arm control ──────────────────────────────────────────────────

    def _close_gripper(self):
        """Close the gripper to grasp the handle."""
        self._send_joint_command(
            "joint_gripper_finger_left", self.gripper_close_effort / 100.0
        )
        self.get_logger().info("Gripper closed")

    def _open_gripper(self):
        """Open the gripper to release."""
        self._send_joint_command("joint_gripper_finger_left", 0.3)
        self.get_logger().info("Gripper opened")

    def _pull_drawer(self) -> bool:
        """Retract the arm to pull the drawer open.

        In simulation: pulls back max_pull_distance in one command
        (MuJoCo publishes zero effort so force sensing is unavailable).

        On real robot: retracts incrementally, stopping when wrist effort
        exceeds pull_force_threshold.
        """
        if self.use_sim:
            return self._pull_drawer_sim()
        else:
            return self._pull_drawer_real()

    def _pull_drawer_sim(self) -> tuple[bool, float]:
        """Sim: retract arm by max_pull_distance (no force feedback)."""
        start_extension = self._get_current_extension()
        target = max(0.0, start_extension - self.max_pull_distance)
        self.get_logger().info(
            f"Sim pull: retracting from {start_extension:.3f}m to {target:.3f}m"
        )
        self._send_joint_command("wrist_extension", target, duration_sec=4)
        time.sleep(1.0)
        wrist_pulled = start_extension - target
        fingertip_length = 0.08
        pulled = wrist_pulled + fingertip_length
        self.get_logger().info(f"Pulled {pulled:.3f}m (wrist {wrist_pulled:.3f} + fingertip {fingertip_length})")
        return wrist_pulled > 0.05, pulled

    def _pull_drawer_real(self) -> tuple[bool, float]:
        """Real robot: retract incrementally with force threshold check."""
        start_extension = self._get_current_extension()
        target_extension = max(0.0, start_extension - self.max_pull_distance)
        current = start_extension
        fingertip_length = 0.08

        while current > target_extension and not self.stop_requested:
            current -= self.pull_speed
            current = max(current, target_extension)
            self._send_joint_command("wrist_extension", current)
            time.sleep(0.1)

            effort = abs(self.current_effort.get("wrist_extension", 0.0))
            if effort > self.pull_force_threshold:
                wrist_pulled = start_extension - current
                pulled_distance = wrist_pulled + fingertip_length
                self.get_logger().info(
                    f"Pull force threshold reached: {effort:.1f}N > {self.pull_force_threshold}N, "
                    f"pulled {pulled_distance:.3f}m (wrist {wrist_pulled:.3f} + fingertip {fingertip_length})"
                )
                return True, pulled_distance

        wrist_pulled = start_extension - current
        pulled_distance = wrist_pulled + fingertip_length
        self.get_logger().info(f"Pulled {pulled_distance:.3f}m (wrist {wrist_pulled:.3f} + fingertip {fingertip_length})")
        return wrist_pulled > 0.05, pulled_distance

    def _get_current_extension(self) -> float:
        """Read current arm extension from joint_states.

        The sim publishes joint_arm_l0..l3 (each is 1/4 of total extension),
        real robot publishes wrist_extension directly.
        """
        if self.current_joint_state is None:
            return 0.3

        names = list(self.current_joint_state.name)
        positions = list(self.current_joint_state.position)

        # Try wrist_extension first (real robot)
        if "wrist_extension" in names:
            return positions[names.index("wrist_extension")]

        # Sum joint_arm_l0..l3 (sim)
        total = 0.0
        for seg in ("joint_arm_l0", "joint_arm_l1", "joint_arm_l2", "joint_arm_l3"):
            if seg in names:
                total += positions[names.index(seg)]
        if total > 0:
            return total

        return 0.3

    def _retract_arm(self):
        """Fully retract the arm after releasing."""
        self._send_joint_command("wrist_extension", 0.0)
        time.sleep(2.0)

    def _get_gripper_world_pos(self):
        """Get the grasp center position in odom frame via TF."""
        try:
            transform = self.tf_buffer.lookup_transform(
                "odom", "link_grasp_center",
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5),
            )
            t = transform.transform.translation
            return np.array([t.x, t.y, t.z])
        except Exception as e:
            self.get_logger().error(f"Cannot get gripper world pos: {e}")
            return None

    def _orient_gripper_toward(self, target_world: np.ndarray, handle_orientation: str):
        """Align the gripper for grasping the handle.

        Yaw = 0: gripper in line with arm axis, pointing toward drawer.
        Roll = pi/2 for horizontal handles, 0 for vertical.
        """
        roll = math.pi / 2 if handle_orientation == "horizontal" else 0.0
        self.get_logger().info(
            f"Orienting gripper: yaw=0, roll={'pi/2' if handle_orientation == 'horizontal' else '0'}"
        )
        self._send_joint_command("joint_wrist_pitch", 0.0)
        self._send_joint_command("joint_wrist_yaw", 0.0)
        self._send_joint_command("joint_wrist_roll", roll)
        time.sleep(1.0)

    def _extend_to_point(self, target_world: np.ndarray):
        """Extend arm so the gripper tip reaches target_world.

        Computes the required extension from the robot base to the
        target point and also sets the lift to match the target Z.
        """
        # Set lift to target Z with gripper offset compensation
        gripper_z = self._get_gripper_z()
        if gripper_z is not None:
            current_lift = self._get_joint_position("joint_lift")
            if current_lift is not None:
                offset = gripper_z - current_lift
                target_lift = float(target_world[2]) - offset
                self._send_joint_command("joint_lift", target_lift)
                time.sleep(2.0)

        # Compute extension: measure how far the gripper tip extends
        # past the wrist, then subtract so the fingers land on the target
        mast_pose = self._get_mast_pose()
        if mast_pose is None:
            self.get_logger().error("Cannot get mast pose for extension")
            return

        gripper_pos = self._get_gripper_world_pos()
        current_ext = self._get_current_extension()
        if gripper_pos is not None and current_ext is not None:
            gx = gripper_pos[0] - mast_pose[0]
            gy = gripper_pos[1] - mast_pose[1]
            gripper_reach = math.sqrt(gx * gx + gy * gy)
            gripper_offset = gripper_reach - current_ext
        else:
            gripper_offset = 0.0

        dx = target_world[0] - mast_pose[0]
        dy = target_world[1] - mast_pose[1]
        dist = math.sqrt(dx * dx + dy * dy)
        grasp_pullback = 0.05
        calc_ext = dist - gripper_offset - grasp_pullback
        self.get_logger().info(
            f"Extend calc: mast=({mast_pose[0]:.3f},{mast_pose[1]:.3f}), "
            f"gripper=({gripper_pos[0]:.3f},{gripper_pos[1]:.3f}), "
            f"target=({target_world[0]:.3f},{target_world[1]:.3f}), "
            f"cur_ext={current_ext:.3f}, grip_reach={gripper_reach:.3f}, "
            f"grip_offset={gripper_offset:.3f}, dist={dist:.3f}, "
            f"calc_ext={calc_ext:.3f}"
        ) if gripper_pos is not None else None
        target_extension = max(0.0, min(calc_ext, 0.52))
        self.get_logger().info(
            f"Extending to handle: dist={dist:.3f}m, "
            f"gripper_offset={gripper_offset:.3f}m, "
            f"target_extension={target_extension:.3f}m"
        )
        self._send_joint_command("wrist_extension", target_extension, duration_sec=4)
        time.sleep(1.0)

    # ─── Low-level control helpers ────────────────────────────────────

    def _send_joint_command(self, joint_name: str, position: float, duration_sec: int = 1):
        """Send a single joint position command via FollowJointTrajectory action."""
        if not self.trajectory_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("FollowJointTrajectory action server not available")
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [joint_name]

        point = JointTrajectoryPoint()
        point.positions = [position]
        point.time_from_start = Duration(sec=duration_sec, nanosec=0)
        goal.trajectory.points = [point]

        t0 = time.time()
        self.get_logger().info(
            f"Sending joint command: {joint_name}={position:.3f} "
            f"(driver_mode={self._driver_mode})"
        )
        def _feedback_cb(fb_msg):
            fb = fb_msg.feedback
            if fb.error and fb.error.positions:
                errors = {n: f"{e:.4f}" for n, e in
                          zip(fb.joint_names, fb.error.positions)}
                self.get_logger().info(
                    f"  feedback: errors={errors}",
                    throttle_duration_sec=0.5,
                )

        future = self.trajectory_client.send_goal_async(
            goal, feedback_callback=_feedback_cb
        )

        timeout = time.time() + 5.0
        while not future.done() and time.time() < timeout:
            time.sleep(0.05)
        if not future.done():
            self.get_logger().error(f"Joint goal send timed out: {joint_name}")
            return False
        if future.result() is None:
            self.get_logger().error(f"Joint goal result is None: {joint_name}")
            return False

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error(f"Joint command REJECTED: {joint_name}={position:.3f}")
            return False

        self.get_logger().info(f"Joint command accepted: {joint_name}={position:.3f}")
        result_future = goal_handle.get_result_async()
        timeout = time.time() + duration_sec + 10
        while not result_future.done() and time.time() < timeout:
            time.sleep(0.05)

        elapsed = time.time() - t0
        if not result_future.done():
            self.get_logger().warn(f"Joint command execution timed out: {joint_name}")
            return False

        result = result_future.result()
        status = goal_handle.status
        # status: 2=ACTIVE, 4=SUCCEEDED, 5=CANCELED, 6=ABORTED
        if status != 4:  # GoalStatus.STATUS_SUCCEEDED
            error_code = result.result.error_code if result and result.result else "N/A"
            error_string = result.result.error_string if result and result.result else "N/A"
            self.get_logger().error(
                f"Joint command FAILED: {joint_name}={position:.3f}, "
                f"status={status}, error_code={error_code}, "
                f"error_string={error_string}, elapsed={elapsed:.3f}s"
            )
            return False

        self.get_logger().info(
            f"Joint command complete: {joint_name}={position:.3f} ({elapsed:.3f}s)"
        )
        return True

    def _switch_to_position_mode(self) -> bool:
        """Switch the driver to position mode (needed for base translate/rotate)."""
        if not self.position_mode_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Position mode service not available!")
            return False
        future = self.position_mode_client.call_async(Trigger.Request())
        timeout = time.time() + 5.0
        while not future.done() and time.time() < timeout:
            time.sleep(0.05)
        if future.done() and future.result() is not None:
            result = future.result()
            if result.success:
                self.get_logger().info(f"Switched to position mode: {result.message}")
            else:
                self.get_logger().error(f"Position mode switch FAILED: {result.message}")
            return result.success
        self.get_logger().error("Position mode switch timed out")
        return False

    def _switch_to_navigation_mode(self) -> bool:
        """Switch the driver back to navigation mode (needed for cmd_vel / frontier exploration)."""
        if not self.navigation_mode_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn("Navigation mode service not available")
            return True
        future = self.navigation_mode_client.call_async(Trigger.Request())
        timeout = time.time() + 5.0
        while not future.done() and time.time() < timeout:
            time.sleep(0.05)
        if future.done() and future.result() is not None:
            self.get_logger().info(f"Switched to navigation mode: {future.result().message}")
            return future.result().success
        self.get_logger().warn("Navigation mode switch timed out")
        return False

    def _rotate_in_place(self, target_angle_rad: float):
        """Rotate the base in place via FollowJointTrajectory.

        Uses stall detection: keeps sending rotation commands until the
        target heading is reached, only fails if yaw stops changing.
        """
        start_yaw = self._get_robot_yaw()
        if start_yaw is None:
            return
        desired_yaw = start_yaw + target_angle_rad

        stall_timeout = 5.0
        last_yaw = start_yaw
        last_progress_time = time.time()

        while not self.stop_requested:
            current_yaw = self._get_robot_yaw()
            if current_yaw is None:
                time.sleep(0.2)
                continue

            remaining = (desired_yaw - current_yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(remaining) < 0.05:
                self.get_logger().info("Rotation complete")
                return

            yaw_change = abs((current_yaw - last_yaw + math.pi) % (2 * math.pi) - math.pi)
            if yaw_change > 0.01:
                last_progress_time = time.time()
                last_yaw = current_yaw
            elif time.time() - last_progress_time > stall_timeout:
                self.get_logger().warn(
                    f"Rotation stalled with {math.degrees(remaining):.1f} deg remaining"
                )
                return

            duration_sec = max(2, int(abs(remaining) / 0.3))
            self.get_logger().info(f"Rotating {math.degrees(remaining):.1f} deg remaining")
            success = self._send_joint_command("rotate_mobile_base", remaining, duration_sec=duration_sec)
            if not success:
                self.get_logger().error("Rotation command failed — aborting rotation")
                return
            time.sleep(1.0)

    def _get_gripper_z(self):
        """Get the grasp center Z position in odom frame via TF."""
        try:
            transform = self.tf_buffer.lookup_transform(
                "odom", "link_grasp_center",
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5),
            )
            return transform.transform.translation.z
        except Exception as e:
            self.get_logger().warn(f"_get_gripper_z TF failed: {e}")
            return None

    def _get_joint_position(self, joint_name: str):
        """Get current position of a joint from joint_states."""
        if self.current_joint_state is None:
            return None
        names = list(self.current_joint_state.name)
        if joint_name in names:
            return self.current_joint_state.position[names.index(joint_name)]
        return None

    def _stop_robot(self):
        """Stop all motion."""
        pass

    def _get_robot_pose(self):
        """Get (x, y) of robot base in map frame."""
        try:
            transform = self.tf_buffer.lookup_transform(
                "odom", "base_link",
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5),
            )
            return (
                transform.transform.translation.x,
                transform.transform.translation.y,
            )
        except Exception as e:
            self.get_logger().warn(f"_get_robot_pose TF failed: {e}")
            return None

    # Mast offset from base_link in base_link frame (from URDF joint_mast)
    MAST_OFFSET_X = -0.067  # slightly behind base center
    MAST_OFFSET_Y = 0.135   # slightly to the left
    MAST_HEIGHT = 1.36       # head height above ground (0.0284 + 1.33)

    def _get_mast_pose(self):
        """Get (x, y) of the mast in the odom frame.

        The arm and head both extend from the mast, so distance
        calculations for reaching/viewing should use this, not base_link.
        """
        robot_pose = self._get_robot_pose()
        yaw = self._get_robot_yaw()
        if robot_pose is None or yaw is None:
            return None
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        mast_x = robot_pose[0] + cos_yaw * self.MAST_OFFSET_X - sin_yaw * self.MAST_OFFSET_Y
        mast_y = robot_pose[1] + sin_yaw * self.MAST_OFFSET_X + cos_yaw * self.MAST_OFFSET_Y
        return (mast_x, mast_y)

    def _get_robot_yaw(self):
        """Get the robot's current heading (yaw) in the odom frame."""
        try:
            transform = self.tf_buffer.lookup_transform(
                "odom", "base_link",
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5),
            )
            from tf_transformations import euler_from_quaternion
            q = transform.transform.rotation
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            return yaw
        except Exception as e:
            self.get_logger().warn(f"_get_robot_yaw TF failed: {e}")
            return None

    # ─── Visualization ────────────────────────────────────────────────

    def _publish_path(self, start_xy, end_xy):
        """Publish the planned path from start to end for RViz display."""
        path_msg = Path()
        path_msg.header.frame_id = "odom"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for t in np.linspace(0, 1, 20):
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = start_xy[0] + t * (end_xy[0] - start_xy[0])
            pose.pose.position.y = start_xy[1] + t * (end_xy[1] - start_xy[1])
            pose.pose.position.z = 0.1
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)

    def publish_status(self):
        """Publish current state."""
        msg = String()
        msg.data = self.state.value
        self.status_pub.publish(msg)

    def _set_state(self, new_state: OpenState):
        self.state = new_state
        self.get_logger().info(f"State → {new_state.value}")


def main(args=None):
    rclpy.init(args=args)
    node = NavigateOpenNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
