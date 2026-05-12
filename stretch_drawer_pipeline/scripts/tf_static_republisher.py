#!/usr/bin/env python3
"""Re-publish /tf_static every few seconds with volatile QoS.

Run on the robot so rosbridge can relay static TFs to remote subscribers
that connect after robot_state_publisher's one-time latched publish.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from tf2_msgs.msg import TFMessage


class TfStaticRepublisher(Node):
    def __init__(self):
        super().__init__("tf_static_republisher")
        self._cache = {}
        self._sub = self.create_subscription(
            TFMessage, "/tf_static",
            self._cb,
            QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self._pub = self.create_publisher(
            TFMessage, "/tf_static_volatile", 100
        )
        self.create_timer(5.0, self._republish)

    def _cb(self, msg):
        for t in msg.transforms:
            self._cache[t.child_frame_id] = t
        self.get_logger().info(
            f"Cached {len(self._cache)} static TFs",
            throttle_duration_sec=30.0,
        )

    def _republish(self):
        if not self._cache:
            return
        msg = TFMessage()
        msg.transforms = list(self._cache.values())
        self._pub.publish(msg)


def main():
    rclpy.init()
    node = TfStaticRepublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()