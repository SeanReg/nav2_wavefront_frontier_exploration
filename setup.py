from setuptools import setup

package_name = 'nav2_wfd'

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
    maintainer='Sean Regan',
    maintainer_email='',
    description='Wavefront Frontier Detector for ROS2 Navigation2',
    license='MIT License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        	'explore = nav2_wfd.wavefront_frontier:main',
        ],
    },
)
