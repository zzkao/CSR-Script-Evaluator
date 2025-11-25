#!/bin/bash

# Training
python tools/train_net.py --config-file configs/transfiner/mask_rcnn_R_50_FPN_1x.yaml --num-gpus 1 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 SOLVER.MAX_ITER 1000 MODEL.WEIGHTS models/model_final_f10217.pkl OUTPUT_DIR ./output/transfiner_r50_1epoch

# Inference / Demonstration
python demo/demo.py --config-file configs/transfiner/mask_rcnn_R_50_FPN_3x.yaml --input demo/input/*.jpg --output demo/output --opts MODEL.WEIGHTS models/model_final_f10217.pkl

# Testing / Evaluation
python tools/train_net.py --config-file configs/transfiner/mask_rcnn_R_50_FPN_1x.yaml --eval-only MODEL.WEIGHTS models/model_final_f10217.pkl
```