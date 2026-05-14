"""Desktop-side pipeline for real-hardware drawer detection.

Runs on the dev machine (automata-3). Receives camera images from the robot
via rosbridge WebSocket, runs Detic detection locally, and shows results in
RViz. The navigate_open_node runs on the robot, not here.

Prerequisites — on the robot (stretch-se3-3096):
  1. stretch_driver in navigation mode
  2. RealSense camera driver
  3. rosbridge_websocket:
       ros2 launch rosbridge_server rosbridge_websocket_launch.xml

Then on the desktop:
  ros2 launch stretch_drawer_pipeline desktop_detection.launch.py \
      robot_ip:=<ROBOT_IP>

For sim, use test_navigate.launch.py instead — no relay needed.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_dir = get_package_share_directory("stretch_drawer_pipeline")
    config_dir = os.path.join(pkg_dir, "config")
    rviz_config = os.path.join(pkg_dir, "rviz", "mapping.rviz")

    robot_ip = LaunchConfiguration("robot_ip")
    robot_port = LaunchConfiguration("robot_port")
    approach_distance = LaunchConfiguration("approach_distance")
    max_pull_distance = LaunchConfiguration("max_pull_distance")
    launch_rviz = LaunchConfiguration("rviz")

    # If the robot uses bare realsense launch, topics are /camera/camera/...
    remote_rgb = LaunchConfiguration("remote_rgb_topic")
    remote_depth = LaunchConfiguration("remote_depth_topic")
    remote_info = LaunchConfiguration("remote_camera_info_topic")

    detection_node = Node(
        package="stretch_drawer_pipeline",
        executable="drawer_detection_node.py",
        name="drawer_detection_node",
        output="screen",
        parameters=[
            os.path.join(config_dir, "detection_params.yaml"),
            {"use_sim": False, "use_sim_time": False},
            {"test_mode": True},
            {"robot_ip": robot_ip},
            {"robot_port": robot_port},
            {"remote_rgb_topic": remote_rgb},
            {"remote_depth_topic": remote_depth},
        ],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
        condition=IfCondition(launch_rviz),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "robot_ip", description="IP address of the Stretch robot"
        ),
        DeclareLaunchArgument(
            "robot_port", default_value="9090",
            description="rosbridge WebSocket port on the robot"
        ),
        DeclareLaunchArgument(
            "remote_rgb_topic",
            default_value="/camera/color/image_raw",
            description="RGB topic name on the robot"
        ),
        DeclareLaunchArgument(
            "remote_depth_topic",
            default_value="/camera/depth/image_rect_raw",
            description="Depth topic name on the robot"
        ),
        DeclareLaunchArgument(
            "remote_camera_info_topic",
            default_value="/camera/color/camera_info",
            description="CameraInfo topic name on the robot"
        ),
        DeclareLaunchArgument(
            "approach_distance", default_value="0.45",
            description="Distance (m) from handle to position robot base"
        ),
        DeclareLaunchArgument(
            "max_pull_distance", default_value="0.4",
            description="Max distance (m) to retract arm when pulling"
        ),
        DeclareLaunchArgument(
            "rviz", default_value="true",
            description="Launch RViz"
        ),
        detection_node,
        rviz_node,
    ])