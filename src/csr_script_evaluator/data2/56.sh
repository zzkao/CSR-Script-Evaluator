#!/bin/bash

# Training
python tools/train_net.py --config-file configs/fear_rcnn_R_101_FPN.yaml --num-gpus 1 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0001 SOLVER.MAX_ITER 1000 OUTPUT_DIR output/fear_training

# Inference / Demonstration
python demo/demo.py --config-file configs/fear_rcnn_R_101_FPN.yaml --video-input demo/sample_video.mp4 --output demo/output.mp4 --opts MODEL.WEIGHTS models/fear_ckpt.pth

# Testing / Evaluation
python tools/train_net.py --config-file configs/fear_rcnn_R_101_FPN.yaml --eval-only MODEL.WEIGHTS models/fear_ckpt.pth OUTPUT_DIR output/fear_eval
```