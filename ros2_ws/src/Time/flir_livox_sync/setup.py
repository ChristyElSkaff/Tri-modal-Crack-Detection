from setuptools import setup

package_name = 'flir_livox_sync'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='semesterproject',
    maintainer_email='semesterproject@todo.todo',
    description='Sync FLIR Blackfly S images with Livox MID-360 LiDAR using message_filters.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'flir_livox_sync_node = flir_livox_sync.flir_livox_sync_node:main',
            'livox_accumulator = flir_livox_sync.livox_accumulator_node:main',
	    'flir_seek_livox_recorder = flir_livox_sync.flir_seek_livox_recorder:main',
	    'livox_scan_saver = flir_livox_sync.livox_scan_saver:main',
	    'rgb_livox_sync_logger = flir_livox_sync.rgb_livox_sync_logger:main',
        ],
    },
)
