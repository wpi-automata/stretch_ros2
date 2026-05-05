#!/usr/bin/env python3
"""Node 1: Mapping and Exploration (funmap-backed).

Orchestrates stretch_funmap's head-scan-based mapping to explore a room.
The exploration loop repeatedly calls funmap services:
  1. /funmap/trigger_head_scan — pan-tilt sweep to build the max-height-image map
  2. /funmap/trigger_drive_to_scan — navigate to the next best scan location

Exploration terminates when funmap reports no good scan location remains,
when a timeout is hit, or when manually stopped.

The node publishes:
  - /exploration_status (std_msgs/String): current exploration state

The node provides:
  - /mapping/start (std_srvs/Trigger): begin exploration
  - /mapping/stop (std_srvs/Trigger): stop exploration early
  - /mapping/is_complete (std_srvs/Trigger): query if mapping finished
"""

import threading
import time
from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

from std_msgs.msg import String
from std_srvs.srv import Trigger


class ExplorationState(Enum):
    IDLE = "idle"
    SCANNING = "scanning"
    DRIVING = "driving"
    COMPLETE = "complete"


class MappingNode(Node):
    """Funmap-backed exploration orchestrator for Stretch3."""

    def __init__(self):
        super().__init__("mapping_node")

        self.declare_parameter("exploration_timeout_s", 300.0)
        self.declare_parameter("max_scan_drive_cycles", 20)
        self.declare_parameter("use_sim", False)

        self.exploration_timeout = self.get_parameter("exploration_timeout_s").value
        self.max_cycles = self.get_parameter("max_scan_drive_cycles").value
        self.use_sim = self.get_parameter("use_sim").value

        self.state = ExplorationState.IDLE
        self.mapping_complete = False
        self.stop_requested = False
        self.exploration_thread = None

        self.cb_group = ReentrantCallbackGroup()

        # Funmap service clients
        self.head_scan_client = self.create_client(
            Trigger, "/funmap/trigger_head_scan", callback_group=self.cb_group
        )
        self.drive_to_scan_client = self.create_client(
            Trigger, "/funmap/trigger_drive_to_scan", callback_group=self.cb_group
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

        self.get_logger().info("Mapping node initialized (funmap-backed)")

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
        response.message = "Exploration started"
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

    def _exploration_loop(self):
        self.get_logger().info("Exploration loop started — waiting for funmap services")

        if not self.head_scan_client.wait_for_service(timeout_sec=30.0):
            self.get_logger().error("/funmap/trigger_head_scan service not available")
            self._set_state(ExplorationState.IDLE)
            return
        if not self.drive_to_scan_client.wait_for_service(timeout_sec=30.0):
            self.get_logger().error("/funmap/trigger_drive_to_scan service not available")
            self._set_state(ExplorationState.IDLE)
            return

        self.get_logger().info("Funmap services available — beginning scan-drive loop")
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

            # Step 1: Head scan
            self._set_state(ExplorationState.SCANNING)
            scan_result = self._call_service(self.head_scan_client)
            if scan_result is None:
                self.get_logger().warn("Head scan service call failed")
                break

            if self.stop_requested:
                break

            # Step 2: Drive to next scan location
            self._set_state(ExplorationState.DRIVING)
            drive_result = self._call_service(self.drive_to_scan_client)
            if drive_result is None:
                self.get_logger().warn("Drive to scan service call failed")
                break

            if not drive_result.success:
                self.get_logger().info(
                    f"No good scan location found: {drive_result.message} — exploration complete"
                )
                break

        self._set_state(ExplorationState.COMPLETE)
        self.mapping_complete = True
        self.get_logger().info(
            f"Exploration complete: {cycle} cycles, "
            f"{time.time() - start_time:.1f}s elapsed"
        )

    def _call_service(self, client):
        """Synchronously call a Trigger service. Returns response or None on failure."""
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

    def publish_status(self):
        msg = String()
        msg.data = self.state.value
        self.status_pub.publish(msg)

    def _set_state(self, new_state: ExplorationState):
        self.state = new_state
        self.get_logger().info(f"State → {new_state.value}")


def main(args=None):
    rclpy.init(args=args)
    node = MappingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
