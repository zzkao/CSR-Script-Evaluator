#!/bin/bash
# Data / Checkpoint / Weight Download (URL)

# Download Jackal dataset (sample bag files)
mkdir -p ~/datasets/jackal_dataset
cd ~/datasets/jackal_dataset
# Note: Manual download required from Google Drive
# https://drive.google.com/drive/folders/1_t5fX5yIqY-y6sAifY8pVWX4O9LCK5R2?usp=sharing

# Download Stevens VLP16 dataset (larger dataset with 20K+ scans)
mkdir -p ~/datasets/stevens_dataset
cd ~/datasets/stevens_dataset
# Note: Manual download required from Google Drive
# https://drive.google.com/drive/folders/16p5UPUCZ1uK0U4XE-hJKjazTsRghEMJa?usp=sharing

# Training
# Note: LeGO-LOAM is not a learning-based method, no training required

# Inference / Demonstration

# Clone and compile LeGO-LOAM
cd ~/catkin_ws/src
git clone https://github.com/RobustFieldAutonomyLab/LeGO-LOAM.git
cd ..
catkin_make -j1

# Source the workspace
source ~/catkin_ws/devel/setup.bash

# Launch LeGO-LOAM system
roslaunch lego_loam run.launch

# In a separate terminal, play bag files with sample data
rosbag play *.bag --clock --topic /velodyne_points /imu/data

# For real-time usage (modify launch file parameter)
# Set /use_sim_time to false in run.launch for real robot usage

# Testing / Evaluation

# Test with sample bag files (assuming downloaded to ~/datasets/)
cd ~/datasets/jackal_dataset
rosbag play *.bag --clock --topic /velodyne_points /imu/data

# Test with Stevens dataset
cd ~/datasets/stevens_dataset
rosbag play *.bag --clock --topic /velodyne_points /imu/data

# Visualize results in RViz (automatically launched with run.launch)
# Results will be displayed in RViz showing:
# - Point cloud segmentation
# - Lidar odometry trajectory
# - Mapping results
# - Ground plane detection

# Performance evaluation (check terminal output for timing information)
# The system outputs real-time pose estimation and mapping results
