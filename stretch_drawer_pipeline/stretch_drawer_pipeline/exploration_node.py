#!/usr/bin/env python3
"""Node 1: Exploration with multiple backends.

Three exploration modes controlled by the ``exploration_mode`` parameter:

  1. **funmap** — scan-drive loop via stretch_funmap services (default).
     After each head scan the robot pauses so Node 2 can detect drawers.
  2. **occupancy_grid** — stub for future occupancy-grid frontier planner.
  3. **simple_explorer** — lightweight structured coverage imported from
     ``semantic-object-container-room/realrobot/stretch/explore_simple.py``.
     At every head-sweep angle the robot pauses for detection.

All modes publish ``/exploration_status`` and call ``/detection/trigger``
at each pause point so Node 2 runs exactly one detection pass then stops.

Services (same API as the old mapping_node):
  - /mapping/start   (Trigger)  begin exploration
  - /mapping/stop    (Trigger)  stop early
  - /mapping/is_complete (Trigger) query status
"""

import sys
import threading
import time
from enum import Enum
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

from std_msgs.msg import String
from std_srvs.srv import Trigger

_SEMANTIC_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "semantic-object-container-room"
if str(_SEMANTIC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SEMANTIC_ROOT))


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
        self.declare_parameter("robot_ip", "")
        self.declare_parameter("n_positions", 4)
        self.declare_parameter("room_type", "kitchen")
        self.declare_parameter("query", "")

        self.exploration_mode = self.get_parameter("exploration_mode").value
        self.exploration_timeout = self.get_parameter("exploration_timeout_s").value
        self.max_cycles = self.get_parameter("max_scan_drive_cycles").value
        self.use_sim = self.get_parameter("use_sim").value
        self.robot_ip = self.get_parameter("robot_ip").value
        self.n_positions = self.get_parameter("n_positions").value
        self.room_type = self.get_parameter("room_type").value
        self.query = self.get_parameter("query").value

        self.state = ExplorationState.IDLE
        self.mapping_complete = False
        self.stop_requested = False
        self.exploration_thread = None

        self.cb_group = ReentrantCallbackGroup()

        # Funmap service clients (used by funmap mode)
        self.head_scan_client = self.create_client(
            Trigger, "/funmap/trigger_head_scan", callback_group=self.cb_group
        )
        self.drive_to_scan_client = self.create_client(
            Trigger, "/funmap/trigger_drive_to_scan", callback_group=self.cb_group
        )

        # Detection trigger (all modes call this at pause points)
        self.detection_trigger_client = self.create_client(
            Trigger, "/detection/trigger", callback_group=self.cb_group
        )

        # Scene graph: tell Node 4 to build graph and run GNN
        self.scene_graph_client = self.create_client(
            Trigger, "/scene_graph/build_and_rank", callback_group=self.cb_group
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
        mode = self.exploration_mode
        self.get_logger().info(f"Exploration loop started (mode={mode})")

        if mode == "funmap":
            self._run_funmap_loop()
        elif mode == "occupancy_grid":
            self._run_occupancy_grid_loop()
        elif mode == "simple_explorer":
            self._run_simple_explorer_loop()
        else:
            self.get_logger().error(f"Unknown exploration_mode: {mode}")
            self._set_state(ExplorationState.IDLE)
            return

        self._on_exploration_complete()

    def _on_exploration_complete(self):
        """Called when any exploration mode finishes."""
        self.get_logger().info("Exploration finished — requesting scene graph build")
        if self.scene_graph_client.wait_for_service(timeout_sec=5.0):
            result = self._call_trigger(self.scene_graph_client)
            if result and result.success:
                self.get_logger().info(f"Scene graph built: {result.message}")
            else:
                msg = result.message if result else "service call failed"
                self.get_logger().warn(f"Scene graph build returned: {msg}")
        else:
            self.get_logger().warn(
                "/scene_graph/build_and_rank not available — skipping GNN ranking"
            )

        self._set_state(ExplorationState.COMPLETE)
        self.mapping_complete = True

    # ── Pause-and-detect helper ──────────────────────────────────────

    def _pause_for_detection(self):
        """Pause, trigger one detection pass on Node 2, then return."""
        self._set_state(ExplorationState.PAUSED_FOR_DETECTION)

        if not self.detection_trigger_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn("/detection/trigger not available — skipping")
            return

        result = self._call_trigger(self.detection_trigger_client)
        if result:
            self.get_logger().debug(f"Detection trigger: {result.message}")

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

            # Head scan
            self._set_state(ExplorationState.SCANNING)
            scan_result = self._call_trigger(self.head_scan_client)
            if scan_result is None:
                self.get_logger().warn("Head scan service call failed")
                break

            if self.stop_requested:
                break

            # Pause for detection after scan
            self._pause_for_detection()

            if self.stop_requested:
                break

            # Drive to next scan location
            self._set_state(ExplorationState.DRIVING)
            drive_result = self._call_trigger(self.drive_to_scan_client)
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

    # ── Mode 3: simple explorer ──────────────────────────────────────

    def _run_simple_explorer_loop(self):
        if not self.robot_ip:
            self.get_logger().error(
                "simple_explorer mode requires 'robot_ip' parameter"
            )
            return

        self.get_logger().info(
            f"Connecting to robot at {self.robot_ip} for simple_explorer…"
        )

        try:
            from stretch.agent.zmq_client import HomeRobotZmqClient
        except ImportError:
            self.get_logger().error(
                "stretch_ai not installed — cannot use simple_explorer mode"
            )
            return

        from realrobot.stretch.explore_simple import SimpleExplorer

        robot = HomeRobotZmqClient(
            robot_ip=self.robot_ip,
            recv_port=4401,
            send_port=4402,
            recv_state_port=4403,
            recv_servo_port=4404,
        )
        robot.start()
        self.get_logger().info("ZMQ client connected")

        def on_position():
            if self.stop_requested:
                return
            self._pause_for_detection()

        explorer = SimpleExplorer(
            robot,
            on_position_callback=on_position,
            verbose=True,
        )

        self.get_logger().info(
            f"Starting simple exploration ({self.n_positions} positions)"
        )
        self._set_state(ExplorationState.SCANNING)

        try:
            explorer.explore(n_positions=self.n_positions)
        except Exception as e:
            self.get_logger().error(f"SimpleExplorer error: {e}")
        finally:
            try:
                robot.stop()
            except Exception:
                pass

        self.get_logger().info("Simple exploration done")

    # ── Helpers ───────────────────────────────────────────────────────

    def _call_trigger(self, client):
        """Synchronously call a Trigger service. Returns response or None."""
        request = Trigger.Request()
        future = client.call_async(request)

        while not future.done():
            if self.stop_requested:
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
