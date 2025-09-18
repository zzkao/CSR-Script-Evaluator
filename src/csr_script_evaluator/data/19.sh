#!/bin/bash

# Environment Setup / Requirement / Installation
# Install dependencies
pip3 install mtcnn
conda install keras-gpu=2.1.0 tensorflow-gpu=1.10.0 cudnn=7.1.3 cuda80=1.0
conda install numpy=1.15.2 opencv

# Data / Checkpoint / Weight Download (URL)
# Option 1: Download pre-processed data (recommended)
mkdir -p data
cd data
wget https://drive.google.com/file/d/1j6GMx33DCcbUOS8J3NHZ-BMHgk7H-oC_/view?usp=sharing -O data.zip
unzip data.zip
cd ..

# Option 2: Process data from scratch
# Create directories and process type1 datasets
mkdir -p data/type1
cd data/type1
sh run_created_db_type1.sh
cd ../..

# Process BIWI dataset
cd data
python TYY_create_db_biwi.py
python TYY_create_db_biwi_70_30.py
cd ..

# Training
# Train FSANet model
sh run_fsanet_train.sh

# Inference / Demonstration
# Run demo with different face detectors
# LBP face detector (fast but less accurate)
cd demo
sh run_demo_FSANET.sh

# MTCNN face detector (slow but accurate)
cd demo
sh run_demo_FSANET_mtcnn.sh

# SSD face detector (fast and accurate)
cd demo
sh run_demo_FSANET_ssd.sh
cd ..

# Testing / Evaluation
# Test FSANet model
sh run_fsanet_test.sh

# Convert model to TensorFlow frozen graph
cd training_and_testing
python keras_to_tf.py \
    --trained-model-dir-path ../pre-trained/300W_LP_models/fsanet_var_capsule_3_16_2_21_5 \
    --output-dir-path ./converted_models
cd ..