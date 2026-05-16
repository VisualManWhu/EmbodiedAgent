"""MVP all-up launch.

Default = offloaded detection: YOLO runs on the laptop, det_bridge_node
receives results. Keeps Pi5 CPU/current low (avoids motor-battery BMS trip).

Usage:
  ros2 launch embodied_mvp mvp.launch.py target_class:=bottle

Run YOLO on the Pi instead (no laptop):
  ros2 launch embodied_mvp mvp.launch.py enable_yolo:=true enable_bridge:=false

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
    enable_stream = LaunchConfiguration('enable_stream')
    enable_bridge = LaunchConfiguration('enable_bridge')

    IfCondition = __import__('launch.conditions', fromlist=['IfCondition']).IfCondition

    return LaunchDescription([
        DeclareLaunchArgument('target_class', default_value='chair'),
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument('enable_yolo', default_value='false'),    # YOLO offloaded to laptop
        DeclareLaunchArgument('enable_search', default_value='true'),
        DeclareLaunchArgument('enable_stream', default_value='true'),
        DeclareLaunchArgument('enable_bridge', default_value='true'),   # receive laptop detections

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

        # On-Pi YOLO (off by default — detection is offloaded to the laptop).
        Node(package='embodied_mvp', executable='yolo_node', name='yolo_node',
             parameters=[params_file],
             condition=IfCondition(enable_yolo)),

        # Receives detections POSTed by the laptop GPU detector -> /detections.
        Node(package='embodied_mvp', executable='det_bridge_node', name='det_bridge_node',
             parameters=[params_file],
             condition=IfCondition(enable_bridge)),

        Node(package='embodied_mvp', executable='search_node', name='search_node',
             parameters=[params_file, {'target_class': target_class}],
             condition=IfCondition(enable_search)),

        # Browser-viewable MJPEG stream at http://<pi-ip>:8080/
        Node(package='embodied_mvp', executable='mjpeg_node', name='mjpeg_node',
             parameters=[params_file],
             condition=IfCondition(enable_stream)),
    ])
