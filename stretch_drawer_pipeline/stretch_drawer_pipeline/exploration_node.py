#!/usr/bin/env python3
"""Node 1: Exploration with multiple backends.

Three exploration modes controlled by the ``exploration_mode`` parameter:

  1. **funmap** — scan-drive loop via stretch_funmap services (default).
     After each head scan the robot pauses so Node 2 can detect drawers.
  2. **occupancy_grid** — stub for future occupancy-grid frontier planner.
  3. **simple** — lightweight structured coverage using ROS2 joint commands.
     Rotates 360 at each position with head sweeps, pausing for detection
     at every head angle. Moves to new positions via base translate/rotate.

All modes publish ``/exploration_status`` and call ``/detection/trigger``
at each pause point so Node 2 runs exactly one detection pass then stops.

Services (same API as the old mapping_node):
  - /mapping/start   (Trigger)  begin exploration
  - /mapping/stop    (Trigger)  stop early
  - /mapping/is_complete (Trigger) query status
"""

import math
import threading
import time
from enum import Enum

import numpy as np
import rclpy
import roslibpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from std_msgs.msg import String
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectoryPoint
import tf2_ros

PAN_RANGE = (-1.2, 0.6)
# Tilt limited to -0.5 min to avoid seeing the gripper in frame
TILT_ANGLES = [-0.5, -0.3, 0.0, 0.3]

MOVE_DISTANCE = 0.8
HEAD_SETTLE = 0.4


class ExplorationState(Enum):
    IDLE = "idle"
    SCANNING = "scanning"
    DRIVING = "driving"
    PAUSED_FOR_DETECTION = "paused_for_detection"
    COMPLETE = "complete"


class ExplorationNode(Node):
    """Multi-mode exploration orchestrator for Stretch3."""

    def __init__(self):
        super().__init__("exploration_node")

        self.declare_parameter("exploration_mode", "funmap")
        self.declare_parameter("exploration_timeout_s", 300.0)
        self.declare_parameter("max_scan_drive_cycles", 20)
        self.declare_parameter("use_sim", False)
        self.declare_parameter("n_positions", 4)
        self.declare_parameter("move_distance", MOVE_DISTANCE)
        self.declare_parameter("num_pan_angles", 8)
        self.declare_parameter("head_settle_s", 2.0)
        self.declare_parameter("room_type", "kitchen")
        self.declare_parameter("query", "")

        self.declare_parameter("rosbridge_port", 9090)

        self.exploration_mode = self.get_parameter("exploration_mode").value
        self.exploration_timeout = self.get_parameter("exploration_timeout_s").value
        self.max_cycles = self.get_parameter("max_scan_drive_cycles").value
        self.use_sim = self.get_parameter("use_sim").value
        self.n_positions = self.get_parameter("n_positions").value
        self.move_distance = self.get_parameter("move_distance").value
        self.num_pan_angles = self.get_parameter("num_pan_angles").value
        self.head_sweep_angles = [
            (pan, tilt)
            for pan in np.linspace(PAN_RANGE[0], PAN_RANGE[1], self.num_pan_angles).tolist()
            for tilt in TILT_ANGLES
        ]
        self.head_settle_s = self.get_parameter("head_settle_s").value
        self.room_type = self.get_parameter("room_type").value
        self.query = self.get_parameter("query").value

        self._rosbridge_port = self.get_parameter("rosbridge_port").value

        self.state = ExplorationState.IDLE
        self.mapping_complete = False
        self.stop_requested = False
        self.exploration_thread = None


        self.cb_group = ReentrantCallbackGroup()

        # TF (for simple mode base pose)
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=30))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # FollowJointTrajectory action (for simple mode)
        self.trajectory_client = ActionClient(
            self, FollowJointTrajectory,
            "/stretch_controller/follow_joint_trajectory",
        )

        # Mode switching (simple mode needs position mode)
        self.position_mode_client = self.create_client(
            Trigger, "/switch_to_position_mode", callback_group=self.cb_group
        )
        self.navigation_mode_client = self.create_client(
            Trigger, "/switch_to_navigation_mode", callback_group=self.cb_group
        )

        # Funmap service clients (funmap mode only)
        self.head_scan_client = self.create_client(
            Trigger, "/funmap/trigger_head_scan", callback_group=self.cb_group
        )
        self.drive_to_scan_client = self.create_client(
            Trigger, "/funmap/trigger_drive_to_scan", callback_group=self.cb_group
        )

        # Detection & scene graph services — DDS for sim, rosbridge for real
        self._ros_client = None
        if self.use_sim:
            self.detection_trigger_client = self.create_client(
                Trigger, "/detection/trigger", callback_group=self.cb_group
            )
            self.scene_graph_frame_client = self.create_client(
                Trigger, "/scene_graph/process_frame", callback_group=self.cb_group
            )
        else:
            self._ros_client = roslibpy.Ros(
                host="localhost", port=self._rosbridge_port
            )
            self._ros_client_thread = threading.Thread(
                target=self._ros_client.run, daemon=True
            )
            self._ros_client_thread.start()
            self._ws_detection_service = roslibpy.Service(
                self._ros_client, "/detection/trigger", "std_srvs/srv/Trigger"
            )
            self._ws_scene_graph_frame_service = roslibpy.Service(
                self._ros_client, "/scene_graph/process_frame", "std_srvs/srv/Trigger"
            )
            self.get_logger().info(
                f"Rosbridge service clients on localhost:{self._rosbridge_port}"
            )

        # Status publisher
        self.status_pub = self.create_publisher(String, "/exploration_status", 10)

        # Services for pipeline control
        self.create_service(
            Trigger, "/mapping/start", self.start_callback,
            callback_group=self.cb_group
        )
        self.create_service(
            Trigger, "/mapping/stop", self.stop_callback,
            callback_group=self.cb_group
        )
        self.create_service(
            Trigger, "/mapping/is_complete", self.is_complete_callback,
            callback_group=self.cb_group
        )

        self.create_timer(1.0, self.publish_status)

        self.get_logger().info(
            f"Exploration node initialized (mode={self.exploration_mode})"
        )

    # ── Pipeline control services ────────────────────────────────────

    def start_callback(self, request, response):
        if self.state not in (ExplorationState.IDLE, ExplorationState.COMPLETE):
            response.success = False
            response.message = f"Already exploring (state={self.state.value})"
            return response

        self.mapping_complete = False
        self.stop_requested = False
        self.exploration_thread = threading.Thread(
            target=self._exploration_loop, daemon=True
        )
        self.exploration_thread.start()
        response.success = True
        response.message = f"Exploration started (mode={self.exploration_mode})"
        return response

    def stop_callback(self, request, response):
        self.stop_requested = True
        response.success = True
        response.message = "Stop requested"
        return response

    def is_complete_callback(self, request, response):
        response.success = self.mapping_complete
        response.message = f"state={self.state.value}, complete={self.mapping_complete}"
        return response

    # ── Status publishing ────────────────────────────────────────────

    def publish_status(self):
        msg = String()
        msg.data = self.state.value
        self.status_pub.publish(msg)

    def _set_state(self, new_state: ExplorationState):
        self.state = new_state
        self.get_logger().info(f"State → {new_state.value}")
        self.publish_status()

    # ── Exploration dispatch ─────────────────────────────────────────

    def _exploration_loop(self):
        try:
            self._exploration_loop_inner()
        except Exception as e:
            self.get_logger().error(f"Exploration thread crashed: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            self._set_state(ExplorationState.IDLE)

    def _retract_arm(self):
        """Fully retract the arm and point gripper down before exploring."""
        self.get_logger().info("Retracting arm before exploration")
        self._send_joint_command("wrist_extension", 0.0, duration_sec=4)
        self._send_joint_command("joint_wrist_pitch", -1.57, duration_sec=2)
        time.sleep(1.0)

    def _exploration_loop_inner(self):
        mode = self.exploration_mode
        self.get_logger().info(f"Exploration loop started (mode={mode})")

        self._retract_arm()

        if self.use_sim:
            if self.detection_trigger_client.wait_for_service(timeout_sec=5.0):
                self.get_logger().info("/detection/trigger available (DDS)")
            else:
                self.get_logger().warn("/detection/trigger not available (DDS)")
            if self.scene_graph_frame_client.wait_for_service(timeout_sec=5.0):
                self.get_logger().info("/scene_graph/process_frame available (DDS)")
            else:
                self.get_logger().warn("/scene_graph/process_frame not available (DDS)")
        else:
            self.get_logger().info("Using rosbridge for detection and scene graph services")

        if mode == "funmap":
            self._run_funmap_loop()
        elif mode == "occupancy_grid":
            self._run_occupancy_grid_loop()
        elif mode == "simple":
            self._run_simple_loop()
        else:
            self.get_logger().error(f"Unknown exploration_mode: {mode}")
            self._set_state(ExplorationState.IDLE)
            return

        self._on_exploration_complete()

    def _on_exploration_complete(self):
        """Called when any exploration mode finishes.

        Sets state to COMPLETE, which publishes on /exploration_status.
        The scene graph node picks up the "complete" status via that topic
        and auto-triggers GNN scoring + ranking push.
        """
        self.get_logger().info("Exploration finished")
        self._set_state(ExplorationState.COMPLETE)
        self.mapping_complete = True

    # ── Pause-and-detect helper ──────────────────────────────────────

    def _pause_for_detection(self):
        """Pause, trigger detection and scene graph frame processing, then return."""
        self._set_state(ExplorationState.PAUSED_FOR_DETECTION)

        det_result = self._call_service("/detection/trigger", timeout_sec=30.0)
        if det_result:
            self.get_logger().info(f"Detection: {det_result.get('message', '')}")

        sg_result = self._call_service("/scene_graph/process_frame", timeout_sec=30.0)
        if sg_result:
            self.get_logger().info(f"Scene graph: {sg_result.get('message', '')}")

    # ── Mode 1: funmap ───────────────────────────────────────────────

    def _run_funmap_loop(self):
        self.get_logger().info("Waiting for funmap services…")

        if not self.head_scan_client.wait_for_service(timeout_sec=30.0):
            self.get_logger().error("/funmap/trigger_head_scan not available")
            return
        if not self.drive_to_scan_client.wait_for_service(timeout_sec=30.0):
            self.get_logger().error("/funmap/trigger_drive_to_scan not available")
            return

        self.get_logger().info("Funmap services ready — beginning scan-drive loop")
        start_time = time.time()
        cycle = 0

        while not self.stop_requested:
            elapsed = time.time() - start_time
            if elapsed > self.exploration_timeout:
                self.get_logger().info("Exploration timeout reached")
                break
            if cycle >= self.max_cycles:
                self.get_logger().info("Max scan-drive cycles reached")
                break

            cycle += 1
            self.get_logger().info(f"Cycle {cycle}: triggering head scan")

            self._set_state(ExplorationState.SCANNING)
            scan_result = self._call_trigger_service(self.head_scan_client)
            if scan_result is None:
                self.get_logger().warn("Head scan service call failed")
                break

            if self.stop_requested:
                break

            self.get_logger().info("Detection sweep (looking for drawers/cabinets)")
            self._head_sweep()

            if self.stop_requested:
                break

            self._set_state(ExplorationState.DRIVING)
            drive_result = self._call_trigger_service(self.drive_to_scan_client)
            if drive_result is None:
                self.get_logger().warn("Drive to scan service call failed")
                break

            if not drive_result.success:
                self.get_logger().info(
                    f"No good scan location: {drive_result.message} — done"
                )
                break

        self.get_logger().info(
            f"Funmap loop done: {cycle} cycles, "
            f"{time.time() - start_time:.1f}s elapsed"
        )

    # ── Mode 2: occupancy grid (stub) ────────────────────────────────

    def _run_occupancy_grid_loop(self):
        self.get_logger().warn(
            "occupancy_grid exploration mode is TBD — "
            "will use area-coverage threshold when implemented"
        )

    # ── Mode 3: simple ───────────────────────────────────────────────

    def _run_simple_loop(self):
        """Structured room coverage using ROS2 joint commands.

        Same pattern as explore_simple.py but using FollowJointTrajectory
        and TF directly instead of ZMQ.

        1. Switch to position mode
        2. Rotate 360 at start (4 x 90), head sweep + detect at each
        3. Move to new positions, rotate 360 + sweep at each
        """
        self.get_logger().info(
            f"Simple exploration: {self.n_positions} positions, "
            f"{self.move_distance}m per step"
        )

        if not self._switch_to_position_mode():
            self.get_logger().error("Cannot switch to position mode — aborting")
            return

        time.sleep(1.0)
        positions_visited = []

        pose = self._get_base_pose()
        if pose:
            positions_visited.append(pose[:2])
            self.get_logger().info(f"Start: ({pose[0]:.2f}, {pose[1]:.2f})")

        # Phase 1: 360 rotation at start with head sweep + detect
        self.get_logger().info("Phase 1: Initial 360 sweep")
        self._rotate_and_sweep()

        # Phase 2: Move in a square pattern — turn 90° then move forward each step
        for pos_i in range(self.n_positions):
            if self.stop_requested:
                break

            self.get_logger().info(f"Position {pos_i + 1}/{self.n_positions}")
            self._set_state(ExplorationState.DRIVING)

            self._rotate_base(math.pi / 2)
            time.sleep(0.5)

            moved = self._try_move(self.move_distance)
            if not moved:
                moved = self._try_move(self.move_distance * 0.5)
            if not moved:
                self.get_logger().warn("Stuck — skipping this position")
                continue

            pose = self._get_base_pose()
            if pose:
                positions_visited.append(pose[:2])
                self.get_logger().info(f"At ({pose[0]:.2f}, {pose[1]:.2f})")

            self._rotate_and_sweep()

        self._switch_to_navigation_mode()
        self.get_logger().info(
            f"Simple exploration done. Visited {len(positions_visited)} positions."
        )

    def _rotate_and_sweep(self):
        """Rotate 360 in 4 steps, running head sweep + detect at each."""
        for i in range(4):
            if self.stop_requested:
                return
            if i > 0:
                self._set_state(ExplorationState.DRIVING)
                self._rotate_base(math.pi / 2)
                time.sleep(0.5)
            self._set_state(ExplorationState.SCANNING)
            self._head_sweep()
        # Complete the 360 so heading is restored
        if not self.stop_requested:
            self._rotate_base(math.pi / 2)
            time.sleep(0.5)

    def _head_sweep(self):
        """Pan-tilt sweep at current position, pausing for detection at each angle."""
        for pan, tilt in self.head_sweep_angles:
            if self.stop_requested:
                return
            self._send_joint_command("joint_head_pan", pan, duration_sec=1)
            self._send_joint_command("joint_head_tilt", tilt, duration_sec=1)
            time.sleep(self.head_settle_s)
            self._pause_for_detection()

    def _try_move(self, distance: float, angle_offset: float = 0.0) -> bool:
        """Try to move forward by distance at current heading + offset.

        Returns True if the robot actually moved.
        """
        pose_before = self._get_base_pose()
        if pose_before is None:
            return False

        if abs(angle_offset) > 0.05:
            self._rotate_base(angle_offset)
            time.sleep(0.5)

        self.get_logger().info(f"Translating {distance:.2f}m")
        self._send_joint_command(
            "translate_mobile_base", distance, duration_sec=max(3, int(distance / 0.2))
        )
        time.sleep(1.0)

        pose_after = self._get_base_pose()
        if pose_after is None:
            return False

        dist_moved = math.sqrt(
            (pose_after[0] - pose_before[0]) ** 2 +
            (pose_after[1] - pose_before[1]) ** 2
        )
        if dist_moved < 0.1:
            self.get_logger().info(f"Barely moved ({dist_moved:.2f}m) — likely blocked")
            return False

        self.get_logger().info(f"Moved {dist_moved:.2f}m")
        return True

    def _rotate_base(self, angle_rad: float):
        """Rotate the base by angle_rad using stall detection."""
        start_yaw = self._get_robot_yaw()
        if start_yaw is None:
            return
        desired_yaw = start_yaw + angle_rad

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
            self._send_joint_command(
                "rotate_mobile_base", remaining, duration_sec=duration_sec
            )
            time.sleep(1.0)

    # ── ROS2 low-level helpers ───────────────────────────────────────

    def _get_base_pose(self):
        """Get (x, y, theta) from TF odom → base_link."""
        try:
            transform = self.tf_buffer.lookup_transform(
                "odom", "base_link",
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0),
            )
            t = transform.transform.translation
            from tf_transformations import euler_from_quaternion
            q = transform.transform.rotation
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            return (t.x, t.y, yaw)
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return None

    def _get_robot_yaw(self):
        pose = self._get_base_pose()
        return pose[2] if pose else None

    def _send_joint_command(self, joint_name: str, position: float, duration_sec: int = 1):
        """Send a joint command via FollowJointTrajectory action."""
        if not self.trajectory_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("FollowJointTrajectory action server not available")
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [joint_name]
        point = JointTrajectoryPoint()
        point.positions = [position]
        point.time_from_start = Duration(sec=duration_sec, nanosec=0)
        goal.trajectory.points = [point]

        future = self.trajectory_client.send_goal_async(goal)
        timeout = time.time() + 5.0
        while not future.done() and time.time() < timeout:
            time.sleep(0.05)
        if not future.done() or future.result() is None:
            self.get_logger().warn(f"Joint goal send failed: {joint_name}")
            return False

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn(f"Joint command rejected: {joint_name}={position:.3f}")
            return False

        result_future = goal_handle.get_result_async()
        timeout = time.time() + duration_sec + 10
        while not result_future.done() and time.time() < timeout:
            time.sleep(0.05)

        if not result_future.done():
            self.get_logger().warn(f"Joint command timed out: {joint_name}")
            return False

        return goal_handle.status == 4  # SUCCEEDED

    def _switch_to_position_mode(self) -> bool:
        if not self.position_mode_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Position mode service not available")
            return False
        result = self._call_trigger_service(self.position_mode_client)
        if result and result.success:
            self.get_logger().info("Switched to position mode")
            return True
        self.get_logger().error("Position mode switch failed")
        return False

    def _switch_to_navigation_mode(self):
        if not self.navigation_mode_client.wait_for_service(timeout_sec=5.0):
            return
        result = self._call_trigger_service(self.navigation_mode_client)
        if result:
            self.get_logger().info("Switched to navigation mode")

    # ── Dual-transport service call ────────────────────────────────────

    _DDS_SERVICE_MAP = {
        "/detection/trigger": "detection_trigger_client",
        "/scene_graph/process_frame": "scene_graph_frame_client",
    }

    _WS_SERVICE_MAP = {
        "/detection/trigger": "_ws_detection_service",
        "/scene_graph/process_frame": "_ws_scene_graph_frame_service",
    }

    def _call_service(self, service_name, timeout_sec=30.0):
        """Call a Trigger service via DDS (sim) or rosbridge (real).

        Returns dict with 'success' and 'message' keys, or None on failure.
        """
        if self.use_sim:
            return self._call_service_dds(service_name, timeout_sec)
        return self._call_service_rosbridge(service_name, timeout_sec)

    def _call_service_dds(self, service_name, timeout_sec):
        attr = self._DDS_SERVICE_MAP.get(service_name)
        if not attr:
            self.get_logger().error(f"Unknown DDS service: {service_name}")
            return None

        client = getattr(self, attr, None)
        if client is None:
            return None

        request = Trigger.Request()
        future = client.call_async(request)
        deadline = time.time() + timeout_sec

        while not future.done():
            if self.stop_requested:
                return None
            if time.time() > deadline:
                self.get_logger().warn(
                    f"{service_name} timed out after {timeout_sec:.0f}s"
                )
                future.cancel()
                return None
            time.sleep(0.1)

        try:
            result = future.result()
            return {"success": result.success, "message": result.message}
        except Exception as e:
            self.get_logger().error(f"{service_name} exception: {e}")
            return None

    def _call_service_rosbridge(self, service_name, timeout_sec):
        attr = self._WS_SERVICE_MAP.get(service_name)
        if not attr:
            self.get_logger().error(f"Unknown rosbridge service: {service_name}")
            return None

        ws_service = getattr(self, attr, None)
        if ws_service is None:
            return None

        result = [None]
        done = threading.Event()

        def _on_response(resp):
            result[0] = resp
            done.set()

        def _on_error(exc):
            self.get_logger().error(f"{service_name} rosbridge error: {exc}")
            done.set()

        try:
            ws_service.call(
                roslibpy.ServiceRequest(),
                callback=_on_response,
                errback=_on_error,
            )
        except Exception as e:
            self.get_logger().error(f"{service_name} call failed: {e}")
            return None

        if not done.wait(timeout=timeout_sec):
            self.get_logger().warn(
                f"{service_name} rosbridge timed out after {timeout_sec:.0f}s"
            )
            return None

        if result[0] is None:
            return None

        return {
            "success": result[0].get("success", False),
            "message": result[0].get("message", ""),
        }

    def _call_trigger_service(self, client, timeout_sec=30.0):
        """Synchronously call a Trigger service via DDS. Used for local services."""
        request = Trigger.Request()
        future = client.call_async(request)
        deadline = time.time() + timeout_sec

        while not future.done():
            if self.stop_requested:
                return None
            if time.time() > deadline:
                self.get_logger().warn(
                    f"Service call timed out after {timeout_sec:.0f}s"
                )
                future.cancel()
                return None
            time.sleep(0.1)

        try:
            return future.result()
        except Exception as e:
            self.get_logger().error(f"Service call exception: {e}")
            return None


def main(args=None):
    rclpy.init(args=args)
    node = ExplorationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
