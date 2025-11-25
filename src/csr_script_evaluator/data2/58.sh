#!/bin/bash

# Training
python train.py --epoch 1 --batch_size 1 --train_path data/sequences/00/ --eval_path data/sequences/00/

# Inference / Demonstration
python demo_superpoint.py --input assets/icl_snippet/ --output_dir dump_demo_sequence/
python match_pairs.py --input_pairs assets/scannet_sample_pairs_with_gt.txt --input_dir assets/scannet_sample_images/ --output_dir dump_match_pairs/ --resize 640 480

# Testing / Evaluation
python evaluation.py --data_path data/sequences/00/ --weights weights/superpoint_v1.pth --resize 640 480
```