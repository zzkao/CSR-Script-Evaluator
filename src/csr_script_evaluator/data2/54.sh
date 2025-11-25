#!/bin/bash

# Training
python train.py --dataset coco --data_path data/coco --arch resnet50 --batch_size 32 --epochs 1 --lr 0.001 --output_dir outputs/

# Inference / Demonstration
python demo.py --image_path demo/demo.jpg --checkpoint checkpoints/ccpl_coco_resnet50.pth --arch resnet50

# Testing / Evaluation
python test.py --dataset coco --data_path data/coco --checkpoint checkpoints/ccpl_coco_resnet50.pth --arch resnet50 --batch_size 32
```