import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

debug_directory_path = os.path.join(os.getenv('HELLO_FLEET_PATH'), 'debug') + '/' if os.getenv('HELLO_FLEET_PATH') else ''

configurable_parameters = [{'name': 'debug_directory', 'default': debug_directory_path, 'description': 'directory where debug imagery is saved'}]

def declare_configurable_parameters(parameters):
    return [DeclareLaunchArgument(param['name'], default_value=param['default'], description=param['description']) for param in parameters]

def generate_launch_description():
    debug_directory = LaunchConfiguration('debug_directory')
    open_drawer_params = [
        {'debug_directory': debug_directory,}
    ]

    open_drawer = Node(
            package='stretch_demos',
            executable='open_drawer',
            output='screen',
            parameters=open_drawer_params,
    )

    stretch_funmap = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('stretch_funmap'), 'launch'),
            '/funmap.launch.py']),
        )

    return LaunchDescription(declare_configurable_parameters(configurable_parameters) + [
        stretch_funmap,
        open_drawer,
        ])
