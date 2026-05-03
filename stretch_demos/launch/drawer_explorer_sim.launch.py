import os
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    drawer_grasp_server = Node(
            package='stretch_demos',
            executable='drawer_grasp_server',
            output='screen',
    )

    drawer_explorer = Node(
            package='stretch_demos',
            executable='drawer_explorer',
            output='screen',
    )

    return LaunchDescription([
        drawer_grasp_server,
        drawer_explorer,
    ])
