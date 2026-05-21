"""Launch the full drawer pipeline on real Stretch3 hardware.

Includes RViz with visualization for:
  - 2D occupancy grid (/map)
  - 3D voxel point cloud (/voxel_map_cloud)
  - Frontier markers (/frontier_markers)
  - Drawer detection markers (/drawer_markers)
  - Navigation path (/navigate_open/path)
  - Target drawer marker (/navigate_open/target_marker)
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

    frontier_method = LaunchConfiguration("frontier_method", default="occupancy_grid")
    launch_rviz = LaunchConfiguration("rviz", default="true")
    rank_type = LaunchConfiguration("rank_type")

    mapping_node = Node(
        package="stretch_drawer_pipeline",
        executable="mapping_node.py",
        name="mapping_node",
        output="screen",
        parameters=[
            os.path.join(config_dir, "mapping_params.yaml"),
            {"use_sim": False},
            {"frontier_method": frontier_method},
        ],
    )

    detection_node = Node(
        package="stretch_drawer_pipeline",
        executable="drawer_detection_node.py",
        name="drawer_detection_node",
        output="screen",
        parameters=[
            os.path.join(config_dir, "detection_params.yaml"),
            {"use_sim": False},
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
            {"use_sim": False},
        ],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config],
        condition=IfCondition(launch_rviz),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "frontier_method", default_value="occupancy_grid",
            description="Frontier method: occupancy_grid or voxel"
        ),
        DeclareLaunchArgument(
            "rviz", default_value="true",
            description="Launch RViz with full pipeline visualization"
        ),
        DeclareLaunchArgument(
            "rank_type", default_value="scene_graph",
            description="Ranking mode: scene_graph, clip_text_text, or clip_image_text",
        ),
        mapping_node,
        detection_node,
        navigate_node,
        rviz_node,
    ])
