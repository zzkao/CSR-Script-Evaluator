#!/bin/bash

# Training
python tools/train_val.py --config config/DEVIANT_kitti_train_small.yaml --gpu 0 --max_epochs 1 --batch_size 1

# Inference / Demonstration
python tools/train_val.py --config config/DEVIANT_kitti.yaml --checkpoint_path weights/DEVIANT_KITTI_final.pth --evaluate

# Testing / Evaluation
python tools/train_val.py --config config/DEVIANT_kitti.yaml --checkpoint_path weights/DEVIANT_KITTI_final.pth --evaluate
```