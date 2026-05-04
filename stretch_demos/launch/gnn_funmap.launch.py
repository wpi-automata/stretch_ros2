"""Launch funmap for fresh exploration (no pre-built map).

Funmap requires /camera/depth/color/points (PointCloud2) to build its map.

Simulation:
  # Terminal 1 — MuJoCo driver WITH cameras enabled, use_slam so funmap owns map→odom TF
  ros2 launch stretch_simulation stretch_mujoco_driver.launch.py use_cameras:=true use_slam:=true

  # Terminal 2 — This launch file
  ros2 launch stretch_demos gnn_funmap.launch.py

Real robot:
  The real stretch_driver + d435i publish the point cloud automatically.
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='stretch_funmap',
            executable='funmap',
            output='screen',
            parameters=[{
                'map_yaml': '',
                'debug_directory': '/tmp/funmap_debug/',
            }],
            remappings=[
                ('/move_base_simple/goal', '/goal_pose'),
            ],
        ),
    ])
