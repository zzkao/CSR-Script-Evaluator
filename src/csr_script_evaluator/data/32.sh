#!/bin/bash

# Environment Setup / Requirement / Installation
# Create Python 3.6 environment (using conda)
conda create -n dsin python=3.6
conda activate dsin

# Install dependencies
pip install -r requirements.txt

# Data / Checkpoint / Weight Download (URL)
# Create data directory
mkdir -p raw_data

# Note: Download dataset manually from:
# https://tianchi.aliyun.com/dataset/dataDetail?dataId=56
# Extract files to raw_data directory

# Data preprocessing
# Sample data by user
python 0_gen_sampled_data.py

# Generate historical session sequence
python 1_gen_sessions.py

# Generate input data for different models
python 2_gen_din_input.py   # For DIN model
python 2_gen_dien_input.py  # For DIEN model
python 2_gen_dsin_input.py  # For DSIN model

# Training
# Train DIN model
python train_din.py

# Train DIEN model
python train_dien.py

# Train DSIN model
python train_dsin.py

# Inference / Demonstration
# No explicit inference commands in README

# Testing / Evaluation
# Evaluation is included in training scripts