#!/bin/bash
# Training
python tools/train_net.py --config-file configs/TotalText/R_50.yaml --num-gpus 1 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 SOLVER.MAX_ITER 1000

# Inference / Demonstration
python demo/demo.py --config-file configs/TotalText/R_50.yaml --input datasets/totaltext/test_images/*.jpg --output output_totaltext --opts MODEL.WEIGHTS tt_s1_poly.pth

# Testing / Evaluation
python tools/train_net.py --config-file configs/TotalText/R_50.yaml --eval-only MODEL.WEIGHTS tt_s1_poly.pth
```