"""GPU workstation (automata-3) pipeline: detection + GNN ranking.

Runs on automata-3. Receives camera images from the robot via rosbridge.
The exploration and navigate nodes run on the robot via pipeline_robot.launch.py.

The scene graph ranker (GNN) is embedded inside drawer_detection_node —
no separate scene_graph_node process is needed.

Prerequisites — on the robot:
  ros2 launch stretch_core stretch_driver.launch.py
  ros2 launch rosbridge_server rosbridge_websocket_launch.xml

Usage:
  ros2 launch stretch_drawer_pipeline pipeline_automata3.launch.py robot_ip:=<ROBOT_IP>
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
    launch_rviz = LaunchConfiguration("rviz")

    detection_node = Node(
        package="stretch_drawer_pipeline",
        executable="drawer_detection_node.py",
        name="drawer_detection_node",
        output="screen",
        parameters=[
            os.path.join(config_dir, "detection_params.yaml"),
            os.path.join(config_dir, "scene_graph_params.yaml"),
            {"use_sim": False, "test_mode": False},
            {"robot_ip": robot_ip, "robot_port": robot_port},
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
            "robot_ip", description="IP address of the Stretch robot",
        ),
        DeclareLaunchArgument(
            "robot_port", default_value="9090",
            description="rosbridge WebSocket port on the robot",
        ),
        DeclareLaunchArgument(
            "rviz", default_value="true",
            description="Launch RViz",
        ),
        detection_node,
        rviz_node,
    ])
