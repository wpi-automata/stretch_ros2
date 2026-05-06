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
  - /navigate_open/target_marker (visualization_msgs/Marker): pink marker on target
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

from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path
from sensor_msgs.msg import JointState
from std_msgs.msg import String, Header, ColorRGBA
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
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

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.cb_group = ReentrantCallbackGroup()

        # Publishers
        self.path_pub = self.create_publisher(Path, "/navigate_open/path", 10)
        self.target_pub = self.create_publisher(
            Marker, "/navigate_open/target_marker", 10
        )
        self.status_pub = self.create_publisher(
            String, "/navigate_open/status", 10
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

        # Client to get drawers from Node 2
        self.get_drawers_client = self.create_client(
            Trigger, "/detection/get_drawers"
        )

        # Mode switching (sim needs position mode for base translate/rotate)
        self.position_mode_client = self.create_client(
            Trigger, "/switch_to_position_mode"
        )
        self.navigation_mode_client = self.create_client(
            Trigger, "/switch_to_navigation_mode"
        )

        # Status timer
        self.create_timer(0.5, self.publish_status)

        self.get_logger().info("Navigate and open node initialized")

    # ─── Callbacks ────────────────────────────────────────────────────

    def joint_state_callback(self, msg: JointState):
        """Track joint positions and efforts for force feedback."""
        self.current_joint_state = msg
        for name, effort in zip(msg.name, msg.effort):
            self.current_effort[name] = effort

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

            # Publish target marker (pink) sized to drawer bounding box
            self._publish_target_marker(drawer)

            # Switch to position mode so base translate/rotate commands work
            self._switch_to_position_mode()

            # Step 2: Navigate base to approach_distance from drawer
            self._set_state(OpenState.NAVIGATING)
            nav_success = self._navigate_to_approach_pose(handle_pos)
            if not nav_success or self.stop_requested:
                self._set_state(OpenState.FAILED)
                return

            # Step 3: Align robot perpendicular to drawer face
            self._set_state(OpenState.ALIGNING)
            self._align_to_drawer(handle_pos, orientation, drawer.get("drawer_corners_world"))
            if self.stop_requested:
                self._set_state(OpenState.FAILED)
                return

            # Step 4: Set wrist orientation for handle type
            self._orient_wrist(orientation)

            # Step 5: Extend arm toward handle
            self._set_state(OpenState.APPROACHING)
            contact = self._approach_handle(handle_pos)
            if not contact or self.stop_requested:
                self._set_state(OpenState.FAILED)
                return

            # Step 6: Close gripper
            self._set_state(OpenState.GRASPING)
            self._close_gripper()
            time.sleep(1.0)

            # Step 7: Pull drawer open
            self._set_state(OpenState.PULLING)
            pull_success = self._pull_drawer()

            # Step 8: Release
            self._set_state(OpenState.RELEASING)
            self._open_gripper()
            time.sleep(0.5)

            # Retract arm
            self._retract_arm()

            if pull_success:
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
        """Get drawers from Node 2 and select the best one.

        Priority: ranking score > closest distance.
        Only considers reachable drawers.
        """
        if not self.get_drawers_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Drawer detection service not available")
            return None

        future = self.get_drawers_client.call_async(Trigger.Request())
        timeout = time.time() + 10.0
        while not future.done() and time.time() < timeout:
            time.sleep(0.05)
        if not future.done() or future.result() is None:
            return None

        response = future.result()
        if not response.success:
            return None

        drawers = json.loads(response.message)
        if not drawers:
            return None

        # Filter reachable
        reachable = [d for d in drawers if d["reachable"]]
        if not reachable:
            self.get_logger().warn("No reachable drawers. Using all drawers.")
            reachable = drawers

        # Sort by ranking (descending), then by distance (ascending)
        reachable.sort(
            key=lambda d: (-d["ranking"], d["distance_to_robot"])
        )

        return reachable[0]

    # ─── Navigation ───────────────────────────────────────────────────

    def _navigate_to_approach_pose(self, handle_pos: np.ndarray) -> bool:
        """Navigate robot base to approach_distance from the drawer.

        Uses rotate_mobile_base and translate_mobile_base via
        FollowJointTrajectory to rotate toward, then drive to, the
        approach point.
        """
        robot_pose = self._get_robot_pose()
        if robot_pose is None:
            return False

        dx = robot_pose[0] - handle_pos[0]
        dy = robot_pose[1] - handle_pos[1]
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 0.01:
            dx, dy = 1.0, 0.0
            dist = 1.0

        ux = dx / dist
        uy = dy / dist

        approach_x = handle_pos[0] + ux * self.approach_distance
        approach_y = handle_pos[1] + uy * self.approach_distance

        self._publish_path(robot_pose, (approach_x, approach_y))
        self.get_logger().info(
            f"Navigating to approach pose: ({approach_x:.2f}, {approach_y:.2f})"
        )

        # Rotate to face the approach point
        angle_to_target = math.atan2(
            approach_y - robot_pose[1], approach_x - robot_pose[0]
        )
        current_yaw = self._get_robot_yaw()
        if current_yaw is not None:
            angle_diff = (angle_to_target - current_yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(angle_diff) > 0.05:
                self.get_logger().info(f"Rotating {math.degrees(angle_diff):.1f} deg to face approach point")
                self._rotate_in_place(angle_diff)

        # Drive forward to approach point in increments.
        # Instead of a fixed timeout, track progress — only fail if the
        # robot stops making progress (stall detection).
        stall_timeout = 10.0
        last_remaining = float("inf")
        last_progress_time = time.time()

        while not self.stop_requested:
            robot_pose = self._get_robot_pose()
            if robot_pose is None:
                time.sleep(0.2)
                continue

            dx = approach_x - robot_pose[0]
            dy = approach_y - robot_pose[1]
            remaining = math.sqrt(dx * dx + dy * dy)

            if remaining < 0.1:
                self.get_logger().info("Reached approach point")
                return True

            # Check if we're still making progress
            if last_remaining - remaining > 0.02:
                last_progress_time = time.time()
                last_remaining = remaining
            elif time.time() - last_progress_time > stall_timeout:
                self.get_logger().warn(
                    f"Navigation stalled at {remaining:.2f}m from target"
                )
                return False

            step = min(remaining, 0.2)
            self.get_logger().info(f"Driving forward {step:.2f}m (remaining {remaining:.2f}m)")
            self._send_joint_command("translate_mobile_base", step, duration_sec=3)
            time.sleep(1.0)

        self.get_logger().warn("Navigation stopped by user")
        return False

    def _align_to_drawer(self, handle_pos: np.ndarray, orientation: str,
                         corners_world=None):
        """Rotate robot so the arm faces the drawer, perpendicular to its face.

        If drawer_corners_world is available (4 corner points from the
        detection node), the drawer face normal is computed from the
        corners. Otherwise falls back to using the robot-to-handle vector.

        The Stretch arm extends to the robot's left (+Y in base_link),
        so the robot's forward direction should be perpendicular to the
        drawer face, pointing along the face (with the arm toward it).
        """
        face_normal = self._compute_drawer_face_normal(corners_world, handle_pos)
        # face_normal points outward from the drawer face. The robot
        # should approach from the direction the normal points, so the
        # arm (extending left) faces into the drawer. The robot's
        # forward axis should be perpendicular to the normal, rotated
        # so the left side points opposite to the normal.
        # desired_heading = atan2(normal_y, normal_x) - pi/2
        normal_angle = math.atan2(face_normal[1], face_normal[0])
        desired_heading = normal_angle - math.pi / 2

        current_yaw = self._get_robot_yaw()
        if current_yaw is None:
            return

        angle_diff = (desired_heading - current_yaw + math.pi) % (2 * math.pi) - math.pi
        self.get_logger().info(
            f"Aligning: rotate {math.degrees(angle_diff):.1f} deg "
            f"(face normal {math.degrees(normal_angle):.1f} deg, "
            f"desired heading {math.degrees(desired_heading):.1f} deg)"
        )
        if abs(angle_diff) > 0.05:
            self._rotate_in_place(angle_diff)

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
                robot_pose = self._get_robot_pose()
                if robot_pose is not None:
                    center = pts.mean(axis=0)[:2]
                    to_robot = np.array(robot_pose) - center
                    if np.dot(normal_2d, to_robot) < 0:
                        normal_2d = -normal_2d
                return normal_2d

        # Fallback: use robot-to-handle direction
        robot_pose = self._get_robot_pose()
        if robot_pose is not None:
            dx = robot_pose[0] - handle_pos[0]
            dy = robot_pose[1] - handle_pos[1]
            norm = math.sqrt(dx * dx + dy * dy)
            if norm > 1e-6:
                return np.array([dx / norm, dy / norm])
        return np.array([1.0, 0.0])

    def _orient_wrist(self, handle_orientation: str):
        """Set wrist yaw for the handle orientation."""
        if handle_orientation == "vertical":
            wrist_yaw = 0.0
        else:
            wrist_yaw = math.pi / 2

        self._send_joint_command("joint_wrist_yaw", wrist_yaw)
        time.sleep(1.0)

    # ─── Arm control ──────────────────────────────────────────────────

    def _approach_handle(self, handle_pos: np.ndarray) -> bool:
        """Extend arm toward the handle.

        In simulation: computes the required extension from the robot's
        position to the handle and extends directly to that distance
        (MuJoCo publishes zero effort so force sensing is unavailable).

        On real robot: incrementally extends until wrist effort exceeds
        grasp_force_threshold.

        Returns True if contact/arrival was achieved.
        """
        # Set lift height to match handle z
        self._send_joint_command("joint_lift", float(handle_pos[2]))
        time.sleep(2.0)

        if self.use_sim:
            return self._approach_handle_sim(handle_pos)
        else:
            return self._approach_handle_real()

    def _approach_handle_sim(self, handle_pos: np.ndarray) -> bool:
        """Sim: extend arm to computed distance (no force feedback available)."""
        robot_pose = self._get_robot_pose()
        if robot_pose is None:
            self.get_logger().error("Cannot get robot pose for extension calc")
            return False

        dx = handle_pos[0] - robot_pose[0]
        dy = handle_pos[1] - robot_pose[1]
        dist_to_handle = math.sqrt(dx * dx + dy * dy)

        # Small overshoot to ensure contact
        target_extension = min(dist_to_handle + 0.02, 0.52)
        self.get_logger().info(
            f"Sim approach: dist_to_handle={dist_to_handle:.3f}m, "
            f"extending to {target_extension:.3f}m"
        )
        self._send_joint_command("wrist_extension", target_extension, duration_sec=4)
        time.sleep(1.0)
        return True

    def _approach_handle_real(self) -> bool:
        """Real robot: incrementally extend until force contact is detected."""
        max_extension = 0.52
        current_extension = 0.0
        step = self.arm_extension_speed

        while current_extension < max_extension and not self.stop_requested:
            current_extension += step
            self._send_joint_command("wrist_extension", current_extension)
            time.sleep(0.2)

            if self._detect_contact():
                self.get_logger().info(
                    f"Contact detected at extension={current_extension:.3f}m"
                )
                return True

        self.get_logger().warn("Max extension reached without contact")
        return False

    def _detect_contact(self) -> bool:
        """Check if the gripper is in contact based on effort readings."""
        effort = self.current_effort.get("wrist_extension", 0.0)
        return abs(effort) > self.grasp_force_threshold

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

    def _pull_drawer_sim(self) -> bool:
        """Sim: retract arm by max_pull_distance (no force feedback)."""
        start_extension = self._get_current_extension()
        target = max(0.0, start_extension - self.max_pull_distance)
        self.get_logger().info(
            f"Sim pull: retracting from {start_extension:.3f}m to {target:.3f}m"
        )
        self._send_joint_command("wrist_extension", target, duration_sec=4)
        time.sleep(1.0)
        pulled = start_extension - target
        self.get_logger().info(f"Pulled {pulled:.3f}m")
        return pulled > 0.05

    def _pull_drawer_real(self) -> bool:
        """Real robot: retract incrementally with force threshold check."""
        start_extension = self._get_current_extension()
        target_extension = max(0.0, start_extension - self.max_pull_distance)
        current = start_extension

        while current > target_extension and not self.stop_requested:
            current -= self.pull_speed
            current = max(current, target_extension)
            self._send_joint_command("wrist_extension", current)
            time.sleep(0.1)

            effort = abs(self.current_effort.get("wrist_extension", 0.0))
            if effort > self.pull_force_threshold:
                self.get_logger().info(
                    f"Pull force threshold reached: {effort:.1f}N > {self.pull_force_threshold}N"
                )
                return True

        pulled_distance = start_extension - current
        self.get_logger().info(f"Pulled {pulled_distance:.3f}m")
        return pulled_distance > 0.05

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

        self.get_logger().info(f"Sending joint command: {joint_name}={position:.3f}")
        future = self.trajectory_client.send_goal_async(goal)

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

        if not result_future.done():
            self.get_logger().warn(f"Joint command execution timed out: {joint_name}")
            return False

        self.get_logger().info(f"Joint command complete: {joint_name}={position:.3f}")
        return True

    def _switch_to_position_mode(self) -> bool:
        """Switch the driver to position mode (needed for base translate/rotate)."""
        if not self.position_mode_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn("Position mode service not available — may already be in position mode")
            return True
        future = self.position_mode_client.call_async(Trigger.Request())
        timeout = time.time() + 5.0
        while not future.done() and time.time() < timeout:
            time.sleep(0.05)
        if future.done() and future.result() is not None:
            self.get_logger().info(f"Switched to position mode: {future.result().message}")
            return future.result().success
        self.get_logger().warn("Position mode switch timed out")
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

    def _rotate_in_place(self, angle_rad: float):
        """Rotate the base in place via FollowJointTrajectory."""
        duration_sec = max(2, int(abs(angle_rad) / 0.3))
        self._send_joint_command("rotate_mobile_base", angle_rad, duration_sec=duration_sec)
        time.sleep(1.0)

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
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
            return None

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
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
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

    def _publish_target_marker(self, drawer: dict):
        """Publish a pink cube matching the drawer's bounding box in RViz."""
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "target_drawer"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0

        corners = drawer.get("drawer_corners_world")
        if corners is not None and len(corners) == 4:
            xs = [c["x"] for c in corners]
            ys = [c["y"] for c in corners]
            zs = [c["z"] for c in corners]
            marker.pose.position.x = (min(xs) + max(xs)) / 2
            marker.pose.position.y = (min(ys) + max(ys)) / 2
            marker.pose.position.z = (min(zs) + max(zs)) / 2
            marker.scale.x = max(max(xs) - min(xs), 0.02)
            marker.scale.y = max(max(ys) - min(ys), 0.02)
            marker.scale.z = max(max(zs) - min(zs), 0.02)
        else:
            h = drawer["handle_center_world"]
            marker.pose.position.x = h["x"]
            marker.pose.position.y = h["y"]
            marker.pose.position.z = h["z"]
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.15

        marker.color = ColorRGBA(r=1.0, g=0.4, b=0.7, a=0.9)
        marker.lifetime.sec = 60

        self.target_pub.publish(marker)

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
