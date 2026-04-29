#! /usr/bin/env python3

from functools import cache
import time
import copy
import math
import pickle
from pathlib import Path
from hello_helpers.hello_misc import *
from hello_helpers.simple_command_group import SimpleCommandGroup
from rclpy.action.server import ServerGoalHandle

from control_msgs.action import FollowJointTrajectory

import threading

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.duration import Duration

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import (
    JointTrajectoryPoint,
    MultiDOFJointTrajectory,
    JointTrajectory,
)

import hello_helpers.hello_misc as hm

from nav_msgs.msg import Path as NavPath
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stretch_mujoco_driver.stretch_mujoco_driver import StretchMujocoDriver

from stretch_mujoco.enums.actuators import Actuators

import rclpy
import rclpy.action
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor


class JointTrajectoryAction:

    def __init__(self, node: "StretchMujocoDriver", action_server_rate_hz: int):
        self.node = node
        self._goal_handle = None
        self._action_server = ActionServer(
            self.node,
            FollowJointTrajectory,
            "/stretch_controller/follow_joint_trajectory",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            handle_accepted_callback=self.handle_accepted_callback,
            callback_group=node.main_group,
        )
        self.timeout = 0.2  # seconds
        self.last_goal_time = self.node.get_clock().now().to_msg()

        self.latest_goal_id = 0

        self.nav_plan_pub = self.node.create_publisher(NavPath, '/nav_plan', 1)
        self.nav_marker_pub = self.node.create_publisher(Marker, '/nav_plan_marker', 1)

    def handle_accepted_callback(self, goal_handle: ServerGoalHandle):
        # Increment goal ID — signals any running execute_callback to preempt
        self.latest_goal_id += 1

        # Stop the base immediately so the previous goal's motion doesn't continue
        self.node.sim.set_base_velocity(0.0, 0.0)

        self._goal_handle = goal_handle
        goal_handle.execute()

    def goal_callback(self, goal_request):
        self.node.get_logger().info(f"Received goal request, {goal_request}")
        new_goal_time = self.node.get_clock().now().to_msg()
        time_duration = (new_goal_time.sec + new_goal_time.nanosec * pow(10, -9)) - (
            self.last_goal_time.sec + self.last_goal_time.nanosec * pow(10, -9)
        )

        if (
            self._goal_handle is not None
            and self._goal_handle.is_active
            and (time_duration < self.timeout)
        ):
            return (
                GoalResponse.REJECT
            )  # Reject goal if another goal is currently active

        self.last_goal_time = self.node.get_clock().now().to_msg()
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.node.get_logger().info("Received cancel request")
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        self.node.get_logger().info("Executing trajectory...")
        my_goal_id = self.latest_goal_id

        trajectory = goal_handle.request.trajectory
        joint_names = trajectory.joint_names
        last_positions = {name: 0.0 for name in joint_names}

        for point in trajectory.points:
            if self.latest_goal_id != my_goal_id:
                result = FollowJointTrajectory.Result()
                goal_handle.abort()
                return result
            positions: list[float] = point.positions
            velocities: list[float | None] = (
                point.velocities if point.velocities else [None] * len(joint_names)
            )

            actuators_in_use = []

            for i, joint in enumerate(joint_names):
                try:
                    actuator = get_actuator_by_joint_names_in_command_groups(joint)
                except:
                    self.node.get_logger().error(f"No command group for joint '{joint}'")
                    continue

                target_position = positions[i]
                delta = target_position - last_positions[joint]
                last_positions[joint] = target_position
                velocity = velocities[i]

                if (actuator == Actuators.left_wheel_vel or actuator == Actuators.right_wheel_vel) and velocity is not None:
                    self.node.sim.set_base_velocity(velocity, 0)
                    continue

                if actuator in (Actuators.base_rotate, Actuators.base_translate):
                    self._publish_nav_plan(actuator, delta)
                    self.node.sim.move_by(actuator, delta)
                else:
                    self.node.sim.move_to(actuator, target_position)

                actuators_in_use.append(actuator)

            base_commanded = any(
                a in (Actuators.base_rotate, Actuators.base_translate)
                for a in actuators_in_use
            )

            for actuator in actuators_in_use:
                if actuator not in (Actuators.base_rotate, Actuators.base_translate):
                    self.node.sim.wait_until_at_setpoint(actuator)

            if base_commanded:
                self._wait_for_base_stopped(my_goal_id)

            for actuator in [Actuators.left_wheel_vel, Actuators.right_wheel_vel]:
                self.node.sim.wait_while_is_moving(actuator, position_tolerance=0.01)

            # Simulate wait until point.time_from_start
            # self._wait_until(
            #     point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
            # )

        goal_handle.succeed()
        result = FollowJointTrajectory.Result()
        self.node.get_logger().info("Trajectory execution complete")
        return result

    def _publish_nav_plan(self, actuator: Actuators, delta: float):
        """Publish a planned path/marker for a base command so it is visible in RViz."""
        status = self.node.sim.pull_status()
        x, y, theta = status.base.x, status.base.y, status.base.theta
        stamp = self.node.get_clock().now().to_msg()

        if actuator == Actuators.base_translate:
            # Straight-line path from current pose to target pose
            steps = 10
            path = NavPath()
            path.header.stamp = stamp
            path.header.frame_id = 'odom'
            for i in range(steps + 1):
                t = i / steps
                ps = PoseStamped()
                ps.header = path.header
                ps.pose.position.x = x + t * delta * math.cos(theta)
                ps.pose.position.y = y + t * delta * math.sin(theta)
                ps.pose.orientation.w = 1.0
                path.poses.append(ps)
            self.nav_plan_pub.publish(path)

        elif actuator == Actuators.base_rotate:
            # Arrow marker showing the new heading after rotation
            target_theta = theta + delta
            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = 'odom'
            marker.ns = 'nav_plan'
            marker.id = 0
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.1
            # Quaternion for target heading (rotation around Z)
            marker.pose.orientation.z = math.sin(target_theta / 2.0)
            marker.pose.orientation.w = math.cos(target_theta / 2.0)
            marker.scale.x = 0.4   # arrow length
            marker.scale.y = 0.05  # arrow width
            marker.scale.z = 0.05  # arrow height
            marker.color.r = 1.0
            marker.color.g = 0.5
            marker.color.a = 1.0
            self.nav_marker_pub.publish(marker)

    def _wait_for_base_stopped(self, my_goal_id: int, timeout: float = 10.0, vel_tolerance: float = 0.01):
        """Wait until both linear and angular base velocities drop to near zero, or goal is preempted."""
        # Give the mujoco server time to process the command and start moving before
        # we begin polling velocity — without this the base reads 0 vel and returns immediately.
        time.sleep(0.3)
        t_start = time.time()
        while time.time() - t_start < timeout:
            if self.latest_goal_id != my_goal_id:
                return
            status = self.node.sim.pull_status()
            x_vel = abs(status.base.x_vel)
            theta_vel = abs(status.base.theta_vel)
            if x_vel < vel_tolerance and theta_vel < vel_tolerance:
                return
            time.sleep(0.05)
        self.node.get_logger().warn('_wait_for_base_stopped: timeout waiting for base to stop')

    def _wait_until(self, seconds):
        loop_rate = self.node.create_rate(10)
        t_start = self.node.get_clock().now().seconds_nanoseconds()[0]
        while (
            self.node.get_clock().now().seconds_nanoseconds()[0] - t_start
        ) < seconds:
            loop_rate.sleep()


@cache
def get_actuator_by_joint_names_in_command_groups(joint_name: str) -> Actuators:
    """
    Joint names defined by stretch_core command groups, return their Actuator here.
    """
    if joint_name == "joint_left_wheel":
        return Actuators.left_wheel_vel
    if joint_name == "joint_right_wheel":
        return Actuators.right_wheel_vel
    if joint_name == 'translate_mobile_base' or joint_name == 'position':
        return Actuators.base_translate
    if joint_name == 'rotate_mobile_base':
        return Actuators.base_rotate
    
    if joint_name == "joint_lift":
        return Actuators.lift
    if joint_name == "joint_arm" or joint_name == "wrist_extension":
        return Actuators.arm
    if joint_name == "joint_wrist_yaw":
        return Actuators.wrist_yaw
    if joint_name == "joint_wrist_pitch":
        return Actuators.wrist_pitch
    if joint_name == "joint_wrist_roll":
        return Actuators.wrist_roll
    if joint_name == "joint_gripper_slide" or joint_name == "joint_gripper_finger_left" or joint_name == "joint_gripper_finger_right" or joint_name == "gripper_aperture":
        return Actuators.gripper
    if joint_name == "joint_head_pan":
        return Actuators.head_pan
    if joint_name == "joint_head_tilt":
        return Actuators.head_tilt

    raise NotImplementedError(f"Actuator for {joint_name} is not defined.")
