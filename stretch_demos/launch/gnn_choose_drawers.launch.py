from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('room_type', default_value='kitchen'),
        DeclareLaunchArgument('query', default_value='fork'),
        DeclareLaunchArgument('det_min_score', default_value='0.3'),
        DeclareLaunchArgument('rerun_enabled', default_value='true'),
        DeclareLaunchArgument('gnn_checkpoint', default_value=''),

        Node(
            package='stretch_demos',
            executable='gnn_choose_drawers',
            output='screen',
            parameters=[{
                'room_type': LaunchConfiguration('room_type'),
                'query': LaunchConfiguration('query'),
                'det_min_score': LaunchConfiguration('det_min_score'),
                'rerun_enabled': LaunchConfiguration('rerun_enabled'),
                'gnn_checkpoint': LaunchConfiguration('gnn_checkpoint'),
            }],
        ),
    ])
