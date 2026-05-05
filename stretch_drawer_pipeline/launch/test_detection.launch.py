"""Test: Detection node in test mode + RViz.

Prerequisites (simulation): Launch the MuJoCo sim driver FIRST in a separate terminal:
  ros2 launch stretch_simulation stretch_mujoco_driver.launch.py use_cameras:=true use_rviz:=false mode:=navigation

Then launch this file:
  ros2 launch stretch_drawer_pipeline test_detection.launch.py use_sim:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_dir = get_package_share_directory("stretch_drawer_pipeline")
    config_dir = os.path.join(pkg_dir, "config")
    rviz_config = os.path.join(pkg_dir, "rviz", "mapping.rviz")

    use_sim = LaunchConfiguration("use_sim")

    detection_node = Node(
        package="stretch_drawer_pipeline",
        executable="drawer_detection_node.py",
        name="drawer_detection_node",
        output="screen",
        parameters=[
            os.path.join(config_dir, "detection_params.yaml"),
            {"use_sim": use_sim, "use_sim_time": use_sim},
            {"test_mode": True},
        ],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
        parameters=[{"use_sim_time": use_sim}],
    )

    return LaunchDescription([
        DeclareLaunchArgument("use_sim", default_value="false",
                              description="Set true for simulation (enables use_sim_time)"),
        detection_node,
        rviz_node,
    ])
