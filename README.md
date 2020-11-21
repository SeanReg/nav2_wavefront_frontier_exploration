# Wavefront Frontier Detection

 ### Implementation of Frontier Exploration based on this research paper: https://arxiv.org/ftp/arxiv/papers/1806/1806.03581.pdf


## Overview

- Intended to work with ROS2's Nav2 stack
  
- Computes a list of Frontier centroids from the currently available Occupancy Grid
  
- Invoke's Nav2's waypoint follower to move the robot to the Frontiers
  
- Upon reaching the waypoint destinations, the latest Occupancy Grid will be evaluated for new Frontiers and continues to plot new waypoints until all Frontiers have been discovered


## Instructions

## Building

For basic/general build instructions follow this tutorial: https://index.ros.org/doc/ros2/Tutorials/Writing-A-Simple-Py-Publisher-And-Subscriber/

- git clone the project into your colcon workspace's "src" directory
- In your colcon workspace root directory run:
  
        rosdep install -i --from-path src --rosdistro foxy -y
        colcon build --packages-select nav2_wfd

- Setup development path:

        . install/setup.bash


## Running
Works with Nav2's Turtlebot Simulation: https://navigation.ros.org/getting_started/index.html#running-the-example   be sure to use "slam:=True" when launching "tb3_simulation_launch.py" such as:

    ros2 launch nav2_bringup tb3_simulation_launch.py slam:=True

Once the Turtlebot Simulation has launch, in a separate window run:
    
    ros2 run nav2_wfd explore


