#!/bin/bash

# Training
python train.py --config configs/dad_3dheads.yaml --batch_size 1 --num_epochs 1 --data_root data/NPHM

# Inference / Demonstration
python demo.py --config configs/dad_3dheads.yaml --checkpoint checkpoints/dad_3dheads_model.pth --input_image examples/input.jpg --output_dir outputs/

# Testing / Evaluation
python evaluate.py --config configs/dad_3dheads.yaml --checkpoint checkpoints/dad_3dheads_model.pth --data_root data/NPHM
```