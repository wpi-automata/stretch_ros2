"""Launch the full drawer pipeline with GNN scene graph ranking.

Supports both simulation (use_sim:=true) and real robot (default).

The scene graph ranker (GNN) is embedded inside drawer_detection_node —
no separate process is needed.

Nodes launched:
  1. exploration_node — multi-mode exploration (funmap / occupancy_grid / simple)
  2. drawer_detection_node — Detic drawer+handle detection + GNN ranking
  3. navigate_open_node — navigate to and open chosen drawer
  4. funmap (sim only, when exploration_mode=funmap)
  5. rviz2 (optional)

Simulation prerequisites:
  ros2 launch stretch_simulation stretch_mujoco_driver.launch.py use_cameras:=true use_rviz:=false mode:=navigation

Real robot prerequisites:
  ros2 launch stretch_core stretch_driver.launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, LaunchConfigurationEquals
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_dir = get_package_share_directory("stretch_drawer_pipeline")
    config_dir = os.path.join(pkg_dir, "config")
    rviz_config = os.path.join(pkg_dir, "rviz", "mapping.rviz")

    use_sim = LaunchConfiguration("use_sim", default="false")
    exploration_mode = LaunchConfiguration("exploration_mode", default="funmap")
    launch_rviz = LaunchConfiguration("rviz", default="true")

    # Funmap is needed for funmap exploration mode in simulation
    funmap_node = Node(
        package="stretch_funmap",
        executable="funmap",
        name="funmap",
        output="screen",
        parameters=[{"map_yaml": "", "debug_directory": "", "use_sim": True}],
        condition=IfCondition(PythonExpression([
            "'", use_sim, "' == 'true' and '", exploration_mode, "' == 'funmap'"
        ])),
    )

    exploration_node = Node(
        package="stretch_drawer_pipeline",
        executable="exploration_node.py",
        name="exploration_node",
        output="screen",
        parameters=[
            os.path.join(config_dir, "mapping_params.yaml"),
            {"exploration_mode": exploration_mode, "use_sim": use_sim},
        ],
    )

    detection_node = Node(
        package="stretch_drawer_pipeline",
        executable="drawer_detection_node.py",
        name="drawer_detection_node",
        output="screen",
        parameters=[
            os.path.join(config_dir, "detection_params.yaml"),
            os.path.join(config_dir, "scene_graph_params.yaml"),
            {"use_sim": use_sim, "test_mode": False},
        ],
    )

    navigate_node = Node(
        package="stretch_drawer_pipeline",
        executable="navigate_open_node.py",
        name="navigate_open_node",
        output="screen",
        parameters=[
            os.path.join(config_dir, "navigate_params.yaml"),
            {"use_sim": use_sim},
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
            "use_sim", default_value="false",
            description="Simulation mode (enables sim-specific behavior)",
        ),
        DeclareLaunchArgument(
            "exploration_mode", default_value="funmap",
            description="Exploration mode: funmap, occupancy_grid, or simple",
        ),
        DeclareLaunchArgument(
            "rviz", default_value="true",
            description="Launch RViz",
        ),
        funmap_node,
        exploration_node,
        detection_node,
        navigate_node,
        rviz_node,
    ])
