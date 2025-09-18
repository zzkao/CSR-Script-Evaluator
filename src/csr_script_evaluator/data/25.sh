#!/bin/bash

# Environment Setup / Requirement / Installation
# Create and activate conda environment
conda create -n bcnet python=3.7 -y
source activate bcnet

# Install PyTorch and dependencies
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
pip install ninja yacs cython matplotlib tqdm
pip install opencv-python==4.4.0.40
pip install scikit-image

# Set installation directory
export INSTALL_DIR=$PWD

# Install COCO API
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# Install BCNet
cd $INSTALL_DIR
git clone https://github.com/lkeab/BCNet.git
cd BCNet/
python3 setup.py build develop

unset INSTALL_DIR

# Data / Checkpoint / Weight Download (URL)
# Create and setup dataset directories
mkdir -p datasets/coco
ln -s /path_to_coco_dataset/annotations datasets/coco/annotations
ln -s /path_to_coco_dataset/train2017 datasets/coco/train2017
ln -s /path_to_coco_dataset/test2017 datasets/coco/test2017
ln -s /path_to_coco_dataset/val2017 datasets/coco/val2017

# Create directory for pretrained models
mkdir -p pretrained_models
# Download pretrained model from: https://hkustconnect-my.sharepoint.com/:u:/g/personal/lkeab_connect_ust_hk/EfiDFLLEawFJpruwuOl3h3ABBjAKysTf0qJQU80iaKbqYg

# Training
# Train with multiple GPUs (using script)
bash all.sh

# Alternative: Train with specific GPUs
CUDA_VISIBLE_DEVICES=0,1 python3 tools/train_net.py --num-gpus 2 --config-file configs/fcos/fcos_imprv_R_50_FPN.yaml 2>&1 | tee log/train_log.txt

# Inference / Demonstration
# Run visualization
bash visualize.sh

# Process bilayer mask annotation
bash process.sh

# Testing / Evaluation
# Set Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Test on test-dev with pretrained model
CUDA_VISIBLE_DEVICES=0,1 python3 tools/train_net.py --num-gpus 2 \
    --config-file configs/fcos/fcos_imprv_R_101_FPN.yaml \
    --eval-only MODEL.WEIGHTS ./pretrained_models/model.pth 2>&1 | tee log/test_log.txt