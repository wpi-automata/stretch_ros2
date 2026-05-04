"""
Launch file for the Detic-based segmented drawer pipeline.

Launches:
  1. stretch_mujoco_driver (sim with robocasa kitchens)
  2. stretch_funmap (mapping via lidar scans)
  3. SLAM (online_async for /map)
  4. search_for_drawers (Detic-based detection, perimeter exploration)
  5. open_drawers (orient + grasp + pull)

Usage:
  ros2 launch stretch_demos open_drawer_segmented.launch.py
  ros2 launch stretch_demos open_drawer_segmented.launch.py robocasa_layout:=Random robocasa_style:=Random
"""

import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from stretch_mujoco.robocasa_gen import choose_layout, choose_style, get_styles, layouts


def _resolve_robocasa_and_launch(context):
    """Resolve layout/style once here so the inner launch never prompts."""
    layout = context.launch_configurations.get('robocasa_layout', 'Random')
    style = context.launch_configurations.get('robocasa_style', 'Random')

    # Interactive prompt only if user didn't specify on command line
    if layout == 'Random' and 'robocasa_layout' not in ' '.join(sys.argv):
        print("\n\nChoose a robocasa kitchen layout:\n")
        layout = layouts[choose_layout()]
        print(f"  Selected layout: {layout}")

    if style == 'Random' and 'robocasa_style' not in ' '.join(sys.argv):
        print("\n\nChoose a robocasa kitchen style:\n")
        style = get_styles()[choose_style()]
        print(f"  Selected style: {style}")

    use_rviz = context.launch_configurations.get('use_rviz', 'true')
    exploration_mode = context.launch_configurations.get('exploration_mode', 'wall_following')

    # Inject into sys.argv so the inner launch's sys.argv check sees them
    # (stretch_mujoco_driver.launch.py checks sys.argv directly, not launch args)
    if f'robocasa_layout:={layout}' not in ' '.join(sys.argv):
        sys.argv.append(f'robocasa_layout:={layout}')
    if f'robocasa_style:={style}' not in ' '.join(sys.argv):
        sys.argv.append(f'robocasa_style:={style}')

    # Pass resolved strings to inner launch
    stretch_simulation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('stretch_simulation'), 'launch'),
            '/stretch_mujoco_driver.launch.py'
        ]),
        launch_arguments={
            'mode': 'navigation',
            'use_cameras': 'true',
            'use_robocasa': 'true',
            'use_rviz': use_rviz,
            'robocasa_layout': layout,
            'robocasa_style': style,
            'use_slam': 'true',
        }.items(),
    )

    # --- 2. Funmap (mapping with lidar scans) ---

    funmap_node = Node(
        package='stretch_funmap',
        executable='funmap',
        output='screen',
        parameters=[{
            'map_yaml': '',
            'debug_directory': '',
        }],
    )

    # --- 3. SLAM (online async for /map and lidar-based mapping) ---

    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('stretch_nav2'), 'launch'),
            '/online_async_launch.py'
        ]),
        launch_arguments={
            'use_sim_time': 'false',
        }.items(),
    )

    # --- 4. Search for Drawers (Detic-based exploration) ---

    search_for_drawers = Node(
        package='stretch_demos',
        executable='search_for_drawers',
        output='screen',
        parameters=[{
            'exploration_mode': exploration_mode,
        }],
    )

    # --- 5. Open Drawers (grasp and pull) ---

    open_drawers = Node(
        package='stretch_demos',
        executable='open_drawers',
        output='screen',
        parameters=[{}],
    )

    return [
        stretch_simulation_launch,
        TimerAction(period=5.0, actions=[funmap_node]),
        TimerAction(period=5.0, actions=[slam_launch]),
        TimerAction(period=10.0, actions=[search_for_drawers]),
        TimerAction(period=12.0, actions=[open_drawers]),
    ]


def generate_launch_description():
    ld = LaunchDescription()

    # --- Arguments ---

    ld.add_action(DeclareLaunchArgument(
        'use_rviz', default_value='true', choices=['true', 'false'],
        description='Launch RViz for visualization'))

    ld.add_action(DeclareLaunchArgument(
        'robocasa_layout', default_value='Random',
        description='Robocasa kitchen layout'))

    ld.add_action(DeclareLaunchArgument(
        'robocasa_style', default_value='Random',
        description='Robocasa kitchen style'))

    ld.add_action(DeclareLaunchArgument(
        'exploration_mode', default_value='wall_following',
        choices=['wall_following', 'frontier'],
        description='Exploration strategy: wall_following (default) or frontier (funmap)'))

    # Resolve layout/style interactively once, then launch everything
    ld.add_action(OpaqueFunction(function=_resolve_robocasa_and_launch))

    return ld
