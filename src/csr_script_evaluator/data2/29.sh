#!/bin/bash

# Training
python main.py --batch_size 2 --epochs 1 --lr_drop 200 --coco_path data/coco --output_dir output --resume checkpoints/detr-r50-e632da11.pth --num_workers 2

# Inference / Demonstration
python inference.py --coco_path data/coco --resume checkpoints/detr-r50-e632da11.pth

# Testing / Evaluation
python main.py --batch_size 2 --no_aux_loss --eval --resume checkpoints/detr-r50-e632da11.pth --coco_path data/coco
```