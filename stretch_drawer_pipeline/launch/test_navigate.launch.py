"""Test: Detection + Navigation (quick-test mode) + RViz.

User drives robot via RViz goal poses. Detection runs continuously.
Trigger /navigate_open/execute to open closest detected drawer.

Prerequisites: Launch the MuJoCo sim driver FIRST in a separate terminal:
  ros2 launch stretch_simulation stretch_mujoco_driver.launch.py use_cameras:=true use_rviz:=false mode:=navigation

Then launch this file to start detection + navigation nodes.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_dir = get_package_share_directory("stretch_drawer_pipeline")
    config_dir = os.path.join(pkg_dir, "config")
    rviz_config = os.path.join(pkg_dir, "rviz", "mapping.rviz")

    detection_node = Node(
        package="stretch_drawer_pipeline",
        executable="drawer_detection_node.py",
        name="drawer_detection_node",
        output="screen",
        parameters=[
            os.path.join(config_dir, "detection_params.yaml"),
            {"use_sim": True},
            {"test_mode": True},
        ],
    )

    navigate_node = Node(
        package="stretch_drawer_pipeline",
        executable="navigate_open_node.py",
        name="navigate_open_node",
        output="screen",
        parameters=[
            os.path.join(config_dir, "navigate_params.yaml"),
            {"use_sim": True},
        ],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
        parameters=[{"use_sim_time": True}],
    )

    return LaunchDescription([
        detection_node,
        navigate_node,
        rviz_node,
    ])
