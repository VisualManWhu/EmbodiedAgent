"""MVP all-up launch.

Usage:
  ros2 launch embodied_mvp mvp.launch.py target_class:=chair

Override params file:
  ros2 launch embodied_mvp mvp.launch.py params_file:=/path/to/params.yaml
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('embodied_mvp')
    default_params = os.path.join(pkg_share, 'config', 'params.yaml')

    target_class = LaunchConfiguration('target_class')
    params_file = LaunchConfiguration('params_file')
    enable_yolo = LaunchConfiguration('enable_yolo')
    enable_search = LaunchConfiguration('enable_search')

    return LaunchDescription([
        DeclareLaunchArgument('target_class', default_value='chair'),
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument('enable_yolo', default_value='true'),
        DeclareLaunchArgument('enable_search', default_value='true'),

        # CSI camera: custom node spawns system rpicam-vid, publishes /camera/image_raw.
        Node(
            package='embodied_mvp', executable='csi_camera_node', name='camera',
            parameters=[params_file],
        ),

        Node(package='embodied_mvp', executable='motor_node', name='motor_node',
             parameters=[params_file]),
        Node(package='embodied_mvp', executable='ir_node', name='ir_node',
             parameters=[params_file]),
        Node(package='embodied_mvp', executable='side_ir_node', name='side_ir_node',
             parameters=[params_file]),
        Node(package='embodied_mvp', executable='pantilt_node', name='pantilt_node',
             parameters=[params_file]),

        GroupAction([
            Node(package='embodied_mvp', executable='yolo_node', name='yolo_node',
                 parameters=[params_file],
                 condition=__import__('launch.conditions', fromlist=['IfCondition']).IfCondition(enable_yolo)),
        ]),

        GroupAction([
            Node(package='embodied_mvp', executable='search_node', name='search_node',
                 parameters=[params_file, {'target_class': target_class}],
                 condition=__import__('launch.conditions', fromlist=['IfCondition']).IfCondition(enable_search)),
        ]),
    ])
