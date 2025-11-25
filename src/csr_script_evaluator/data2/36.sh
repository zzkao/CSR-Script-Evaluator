#!/bin/bash

# Training
python train.py --config configs/transmomo.yaml --dataset_path data/mixamo --checkpoint_dir checkpoints --num_epochs 1 --batch_size 1

# Inference / Demonstration
python demo.py --config configs/transmomo.yaml --checkpoint checkpoints/transmomo.pth --source_motion data/mixamo/source.npy --target_skeleton data/mixamo/target.bvh --output output.bvh

# Testing / Evaluation
python test.py --config configs/transmomo.yaml --checkpoint checkpoints/transmomo.pth --dataset_path data/mixamo
```