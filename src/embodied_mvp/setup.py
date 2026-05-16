from setuptools import find_packages, setup
from glob import glob

package_name = 'embodied_mvp'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yanfeng',
    maintainer_email='spatialintelligence2026@gmail.com',
    description='MVP autonomous object-search for Pi5 mecanum car',
    license='MIT',
    entry_points={
        'console_scripts': [
            'csi_camera_node = embodied_mvp.csi_camera_node:main',
            'motor_node = embodied_mvp.motor_node:main',
            'ir_node = embodied_mvp.ir_node:main',
            'side_ir_node = embodied_mvp.side_ir_node:main',
            'pantilt_node = embodied_mvp.pantilt_node:main',
            'yolo_node = embodied_mvp.yolo_node:main',
            'search_node = embodied_mvp.search_node:main',
            'mjpeg_node = embodied_mvp.mjpeg_node:main',
            'det_bridge_node = embodied_mvp.det_bridge_node:main',
        ],
    },
)
