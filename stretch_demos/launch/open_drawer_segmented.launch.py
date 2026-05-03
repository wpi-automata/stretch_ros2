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
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


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

    # --- 1. Simulation Driver (MuJoCo + robocasa kitchens) ---

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
            'use_rviz': LaunchConfiguration('use_rviz'),
            'robocasa_layout': LaunchConfiguration('robocasa_layout'),
            'robocasa_style': LaunchConfiguration('robocasa_style'),
        }.items(),
    )
    ld.add_action(stretch_simulation_launch)

    # --- 2. Funmap (mapping with lidar scans) ---

    funmap_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('stretch_funmap'), 'launch'),
            '/funmap.launch.py'
        ]),
    )
    # Delay funmap to let sim start publishing
    ld.add_action(TimerAction(period=5.0, actions=[funmap_launch]))

    # --- 3. SLAM (online async for /map and lidar-based mapping) ---

    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('stretch_nav2'), 'launch'),
            '/online_async_launch.py'
        ]),
        launch_arguments={
            'use_sim_time': 'true',
        }.items(),
    )
    ld.add_action(TimerAction(period=5.0, actions=[slam_launch]))

    # --- 4. Search for Drawers (Detic-based exploration) ---

    search_for_drawers = Node(
        package='stretch_demos',
        executable='search_for_drawers',
        output='screen',
        parameters=[{'exploration_mode': LaunchConfiguration('exploration_mode')}],
    )
    # Delay to let sim + SLAM initialize
    ld.add_action(TimerAction(period=10.0, actions=[search_for_drawers]))

    # --- 5. Open Drawers (grasp and pull) ---

    open_drawers = Node(
        package='stretch_demos',
        executable='open_drawers',
        output='screen',
    )
    # Delay to let search node start
    ld.add_action(TimerAction(period=12.0, actions=[open_drawers]))

    return ld
