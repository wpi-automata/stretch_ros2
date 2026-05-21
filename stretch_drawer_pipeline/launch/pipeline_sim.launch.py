"""Launch the drawer pipeline nodes + RViz for simulation.

Prerequisites: Launch the MuJoCo sim driver FIRST in a separate terminal:
  ros2 launch stretch_simulation stretch_mujoco_driver.launch.py use_cameras:=true use_rviz:=false mode:=navigation

Then launch this file to start the pipeline.
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

    rank_type = LaunchConfiguration("rank_type")

    funmap_node = Node(
        package="stretch_funmap",
        executable="funmap",
        name="funmap",
        output="screen",
        parameters=[{"map_yaml": "", "debug_directory": "", "use_sim": True}],
    )

    mapping_node = Node(
        package="stretch_drawer_pipeline",
        executable="mapping_node.py",
        name="mapping_node",
        output="screen",
        parameters=[
            os.path.join(config_dir, "mapping_params.yaml"),
            {"use_sim": True, "use_sim_time": True},
        ],
    )

    detection_node = Node(
        package="stretch_drawer_pipeline",
        executable="drawer_detection_node.py",
        name="drawer_detection_node",
        output="screen",
        parameters=[
            os.path.join(config_dir, "detection_params.yaml"),
            {"use_sim": True, "use_sim_time": True},
            {"test_mode": False},
            {"rank_type": rank_type},
        ],
    )

    navigate_node = Node(
        package="stretch_drawer_pipeline",
        executable="navigate_open_node.py",
        name="navigate_open_node",
        output="screen",
        parameters=[
            os.path.join(config_dir, "navigate_params.yaml"),
            {"use_sim": True, "use_sim_time": True},
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
        DeclareLaunchArgument(
            "rank_type", default_value="scene_graph",
            description="Ranking mode: scene_graph, clip_text_text, or clip_image_text",
        ),
        funmap_node,
        mapping_node,
        detection_node,
        navigate_node,
        rviz_node,
    ])
