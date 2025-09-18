#!/bin/bash

# Data / Checkpoint / Weight Download (URL)
# Create data directory
mkdir -p data

# Preprocess data (convert to .xyz or .npy format)
python data_preprocess.py

# Training
# Train auto-encoder model
python train_ae.py \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --categories airplane \
    --test_size 400

# Train generator model
python train_gen.py \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --categories airplane \
    --test_size 400

# Inference / Demonstration
# No explicit inference commands - inference is part of testing

# Testing / Evaluation
# Test auto-encoder model
python test_ae.py \
    --ckpt ./pretrained/AE_all.pt \
    --categories all

# Test generator model
python test_gen.py \
    --ckpt ./pretrained/GEN_airplane.pt \
    --categories airplane