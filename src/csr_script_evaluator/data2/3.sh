#!/bin/bash

# Training
# No training required for this SLAM system

# Inference / Demonstration
source devel/setup.bash
roslaunch lego_loam run.launch &
sleep 5
rosbag play ~/Downloads/jackal_dataset_loam.bag --clock --topic /velodyne_points /imu/data

# Testing / Evaluation
# No specific testing commands provided in README
```