#!/bin/bash

# Data / Checkpoint / Weight Download (URL)
# Create data directories
mkdir -p data/cityscapes
cd data/cityscapes

# Download Cityscapes dataset (requires login)
# Manual download required from:
# https://www.cityscapes-dataset.com/file-handling/?packageID=1 (gtFine_trainvaltest.zip)
# https://www.cityscapes-dataset.com/file-handling/?packageID=3 (leftImg8bit_trainvaltest.zip)

# Preprocess Cityscapes dataset
cd cityscapes
python3 preprocessing.py
cd ..

# Download pretrained weights
wget https://zenodo.org/record/1419051/files/pretrained_weights.tar.gz
tar -xvzf pretrained_weights.tar.gz -C /model

# Training
# Train probabilistic U-Net
cd training
python3 train_prob_unet.py --config prob_unet_config.py
cd ..

# Inference / Demonstration
# Generate samples from validation set
cd evaluation
python3 eval_cityscapes.py --write_samples

# Testing / Evaluation
# Evaluate generated samples
python3 eval_cityscapes.py --eval_samples

# View results in Jupyter notebook
jupyter notebook evaluation_plots.ipynb

# Run tests
cd ../tests/evaluation
python3 -m pytest eval_tests.py