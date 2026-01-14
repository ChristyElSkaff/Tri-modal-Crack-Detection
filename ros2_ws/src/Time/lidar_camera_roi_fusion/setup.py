from setuptools import find_packages, setup

package_name = 'lidar_camera_roi_fusion'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='semesterproject',
    maintainer_email='semesterproject@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
	    'roi_fusion_node = lidar_camera_roi_fusion.roi_fusion_node:main',
	    'collect_pairs = lidar_camera_roi_fusion.collext_pairs:main',
	    'aruco_plane_calibrate = lidar_camera_roi_fusion.aruco_cal:main',

        ],
    },
)