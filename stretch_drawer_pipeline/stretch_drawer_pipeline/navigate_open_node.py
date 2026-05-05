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

from geometry_msgs.msg import PoseStamped, Point, Twist
from nav_msgs.msg import Path
from sensor_msgs.msg import JointState
from std_msgs.msg import String, Header, ColorRGBA
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
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
        self.declare_parameter("approach_distance", 1.0)
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
        self.cmd_vel_pub = self.create_publisher(Twist, "/stretch/cmd_vel", 10)
        self.joint_cmd_pub = self.create_publisher(
            JointTrajectory, "/stretch_controller/command", 10
        )
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)

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

            # Publish target marker (pink)
            self._publish_target_marker(handle_pos)

            # Step 2: Navigate base to approach_distance from drawer
            self._set_state(OpenState.NAVIGATING)
            nav_success = self._navigate_to_approach_pose(handle_pos)
            if not nav_success or self.stop_requested:
                self._set_state(OpenState.FAILED)
                return

            # Step 3: Align robot perpendicular to drawer face
            self._set_state(OpenState.ALIGNING)
            self._align_to_drawer(handle_pos, orientation)
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
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if future.result() is None:
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

        Positions the robot so the arm can reach the handle.
        The robot stands 1m away, facing sideways (arm faces drawer).
        """
        robot_pose = self._get_robot_pose()
        if robot_pose is None:
            return False

        # Compute approach point: 1m away from handle along the line
        # from handle to robot (so robot is behind its approach point)
        dx = robot_pose[0] - handle_pos[0]
        dy = robot_pose[1] - handle_pos[1]
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 0.01:
            dx, dy = 1.0, 0.0
            dist = 1.0

        # Unit vector from handle toward robot
        ux = dx / dist
        uy = dy / dist

        # Approach position: approach_distance from handle
        approach_x = handle_pos[0] + ux * self.approach_distance
        approach_y = handle_pos[1] + uy * self.approach_distance

        # Heading: robot faces perpendicular to the handle direction
        # (arm points toward handle, which is to the robot's left side)
        heading_to_handle = math.atan2(-uy, -ux)
        # Robot's arm is on its left side, so rotate 90 degrees
        approach_theta = heading_to_handle + math.pi / 2

        # Publish planned path for RViz
        self._publish_path(robot_pose, (approach_x, approach_y))

        # Send goal
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = "map"
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.position.x = approach_x
        goal_msg.pose.position.y = approach_y
        goal_msg.pose.position.z = 0.0

        from tf_transformations import quaternion_from_euler
        q = quaternion_from_euler(0, 0, approach_theta)
        goal_msg.pose.orientation.x = q[0]
        goal_msg.pose.orientation.y = q[1]
        goal_msg.pose.orientation.z = q[2]
        goal_msg.pose.orientation.w = q[3]

        self.goal_pub.publish(goal_msg)
        self.get_logger().info(
            f"Navigating to approach pose: ({approach_x:.2f}, {approach_y:.2f})"
        )

        # Wait for arrival
        timeout = 60.0
        start = time.time()
        while time.time() - start < timeout and not self.stop_requested:
            pose = self._get_robot_pose()
            if pose is not None:
                dx = approach_x - pose[0]
                dy = approach_y - pose[1]
                if math.sqrt(dx * dx + dy * dy) < 0.3:
                    return True
            time.sleep(0.5)

        return False

    def _align_to_drawer(self, handle_pos: np.ndarray, orientation: str):
        """Fine-tune robot rotation to face the drawer with the arm."""
        robot_pose = self._get_robot_pose()
        if robot_pose is None:
            return

        # Compute angle from robot to handle
        dx = handle_pos[0] - robot_pose[0]
        dy = handle_pos[1] - robot_pose[1]
        angle_to_handle = math.atan2(dy, dx)

        # Robot arm points left, so desired heading is handle_angle + pi/2
        desired_heading = angle_to_handle + math.pi / 2

        # Get current heading
        try:
            transform = self.tf_buffer.lookup_transform(
                "map", "base_link",
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0),
            )
            from tf_transformations import euler_from_quaternion
            q = transform.transform.rotation
            _, _, current_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        except Exception:
            return

        # Turn the difference
        angle_diff = desired_heading - current_yaw
        # Normalize to [-pi, pi]
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

        self._rotate_in_place(angle_diff)

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
        """Extend arm toward the handle until pressure is detected.

        Returns True if contact was made.
        """
        # First set lift height to match handle z
        self._send_joint_command("joint_lift", float(handle_pos[2]))
        time.sleep(2.0)

        # Incrementally extend wrist until contact
        max_extension = 0.52
        current_extension = 0.0
        step = self.arm_extension_speed

        while current_extension < max_extension and not self.stop_requested:
            current_extension += step
            self._send_joint_command("wrist_extension", current_extension)
            time.sleep(0.2)

            # Check force on wrist
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

        Monitors force and stops when pull_force_threshold is exceeded.
        Returns True if the force threshold was reached (drawer opened).
        """
        if self.current_joint_state is None:
            return False

        # Get current extension
        try:
            ext_idx = list(self.current_joint_state.name).index("wrist_extension")
            start_extension = self.current_joint_state.position[ext_idx]
        except (ValueError, IndexError):
            start_extension = 0.3

        target_extension = max(0.0, start_extension - self.max_pull_distance)
        current = start_extension

        while current > target_extension and not self.stop_requested:
            current -= self.pull_speed
            current = max(current, target_extension)
            self._send_joint_command("wrist_extension", current)
            time.sleep(0.1)

            # Check if force threshold exceeded (drawer is stuck or fully open)
            effort = abs(self.current_effort.get("wrist_extension", 0.0))
            if effort > self.pull_force_threshold:
                self.get_logger().info(
                    f"Pull force threshold reached: {effort:.1f}N > {self.pull_force_threshold}N"
                )
                return True

        pulled_distance = start_extension - current
        self.get_logger().info(f"Pulled {pulled_distance:.3f}m")
        return pulled_distance > 0.05

    def _retract_arm(self):
        """Fully retract the arm after releasing."""
        self._send_joint_command("wrist_extension", 0.0)
        time.sleep(2.0)

    # ─── Low-level control helpers ────────────────────────────────────

    def _send_joint_command(self, joint_name: str, position: float):
        """Send a single joint position command."""
        msg = JointTrajectory()
        msg.joint_names = [joint_name]

        point = JointTrajectoryPoint()
        point.positions = [position]
        point.time_from_start = Duration(sec=1, nanosec=0)
        msg.points = [point]

        self.joint_cmd_pub.publish(msg)

    def _rotate_in_place(self, angle_rad: float):
        """Rotate the base in place."""
        angular_speed = 0.4
        duration = abs(angle_rad) / angular_speed

        twist = Twist()
        twist.angular.z = angular_speed if angle_rad > 0 else -angular_speed

        start = time.time()
        while time.time() - start < duration and not self.stop_requested:
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.1)

        self.cmd_vel_pub.publish(Twist())
        time.sleep(0.3)

    def _stop_robot(self):
        """Emergency stop all motion."""
        self.cmd_vel_pub.publish(Twist())

    def _get_robot_pose(self):
        """Get (x, y) of robot base in map frame."""
        try:
            transform = self.tf_buffer.lookup_transform(
                "map", "base_link",
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5),
            )
            return (
                transform.transform.translation.x,
                transform.transform.translation.y,
            )
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
            return None

    # ─── Visualization ────────────────────────────────────────────────

    def _publish_path(self, start_xy, end_xy):
        """Publish the planned path from start to end for RViz display."""
        path_msg = Path()
        path_msg.header.frame_id = "map"
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

    def _publish_target_marker(self, handle_pos: np.ndarray):
        """Publish a pink marker at the target drawer for RViz."""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "target_drawer"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        marker.pose.position.x = float(handle_pos[0])
        marker.pose.position.y = float(handle_pos[1])
        marker.pose.position.z = float(handle_pos[2])
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.35
        marker.scale.y = 0.15
        marker.scale.z = 0.2

        # Pink color for target
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
