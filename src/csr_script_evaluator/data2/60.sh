#!/bin/bash

# Training
python train.py --data_dir ./data --batch_size 1 --epochs 1 --output_dir ./output

# Inference / Demonstration
python inference.py --checkpoint checkpoints/ddm2_pretrained.pt --input_path ./data/sample_input.nii.gz --output_path ./output/sample_output.nii.gz

# Testing / Evaluation
python evaluate.py --checkpoint checkpoints/ddm2_pretrained.pt --data_dir ./data --output_dir ./results
```