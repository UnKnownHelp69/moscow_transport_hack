"""Запуск узла резервной одометрии (и, по желанию, монитора точности).

ros2 launch tram_odometry tram_odometry.launch.py vehicle_id:=30618 [gnss_correction:=false] [eval:=true]
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    params = os.path.join(get_package_share_directory('tram_odometry'), 'config', 'params.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('vehicle_id', default_value=''),
        DeclareLaunchArgument('gnss_correction', default_value='true'),
        DeclareLaunchArgument('eval', default_value='false'),
        DeclareLaunchArgument('csv_path', default_value=''),
        Node(package='tram_odometry', executable='tram_odometry_node', name='tram_odometry', output='screen',
             parameters=[params, {'vehicle_id': ParameterValue(LaunchConfiguration('vehicle_id'), value_type=str),
                                  'gnss_correction': ParameterValue(LaunchConfiguration('gnss_correction'), value_type=bool)}]),
        Node(package='tram_odometry', executable='tram_odometry_eval', name='tram_odometry_eval', output='screen',
             parameters=[{'csv_path': LaunchConfiguration('csv_path')}],
             condition=IfCondition(LaunchConfiguration('eval'))),
    ])
