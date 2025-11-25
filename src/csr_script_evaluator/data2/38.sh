#!/bin/bash

# Training
python train.py --dataset_file data/train_data.npz --batch_size 8 --num_epochs 1 --checkpoint_dir checkpoints

# Inference / Demonstration
python demo.py --img_file data/sample_images/image.png --checkpoint checkpoints/model.pt --output_dir results

# Testing / Evaluation
python evaluate.py --dataset_file data/test_data.npz --checkpoint checkpoints/model.pt --batch_size 8
```