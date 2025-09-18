#!/bin/bash


# Data / Checkpoint / Weight Download (URL)
# Create and setup dataset directories
mkdir -p datasets/coco
ln -s /path_to_coco_dataset/annotations datasets/coco/annotations
ln -s /path_to_coco_dataset/train2017 datasets/coco/train2017
ln -s /path_to_coco_dataset/test2017 datasets/coco/test2017
ln -s /path_to_coco_dataset/val2017 datasets/coco/val2017

# Create directory for pretrained models
mkdir -p pretrained_model
# Download pretrained models from:
# R50-FPN 1x: https://drive.google.com/file/d/1IHNEs7PLGaw2gftHzMIOAxFzlYVPMc26/view
# R50-FPN 3x: https://drive.google.com/file/d/1EA9pMdUK6Ad9QsjaZz0g5jqbo_JkqtME/view
# R50-FPN-DCN 3x: https://drive.google.com/file/d/1N0C_ZhES7iu8qEPG2mrdxf8rWteemxQD/view

# Training
# Train with ResNet-101 backbone (3x schedule)
bash scripts/train_transfiner_3x_101.sh

# Train with ResNet-50 backbone (1x schedule)
bash scripts/train_transfiner_1x_50.sh

# Inference / Demonstration
# Run visualization
bash scripts/visual.sh

# Run visualization for Swin-based model
bash scripts/visual_swinb.sh

# Testing / Evaluation
# Test with ResNet-101 backbone (3x schedule)
bash scripts/test_3x_transfiner_101.sh