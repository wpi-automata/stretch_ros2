"""Robot-side pipeline: exploration + navigate/open.

Runs on the Stretch3 robot. Nodes 2 and 4 (detection + scene graph) run on
automata-3 via pipeline_automata3.launch.py.

Prerequisites:
  ros2 launch stretch_core stretch_driver.launch.py
  ros2 launch rosbridge_server rosbridge_websocket_launch.xml
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_dir = get_package_share_directory("stretch_drawer_pipeline")
    config_dir = os.path.join(pkg_dir, "config")

    exploration_mode = LaunchConfiguration("exploration_mode", default="simple")

    funmap_node = Node(
        package="stretch_funmap",
        executable="funmap",
        name="funmap",
        output="screen",
        parameters=[{"map_yaml": "", "debug_directory": "", "use_sim": False}],
        condition=IfCondition(PythonExpression([
            "'", exploration_mode, "' == 'funmap'"
        ])),
    )

    exploration_node = Node(
        package="stretch_drawer_pipeline",
        executable="exploration_node.py",
        name="exploration_node",
        output="screen",
        parameters=[
            os.path.join(config_dir, "mapping_params.yaml"),
            {"exploration_mode": exploration_mode, "use_sim": False},
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

    return LaunchDescription([
        DeclareLaunchArgument(
            "exploration_mode", default_value="simple",
            description="Exploration mode: funmap, occupancy_grid, or simple",
        ),
        funmap_node,
        exploration_node,
        navigate_node,
    ])
