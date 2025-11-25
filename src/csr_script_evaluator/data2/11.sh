#!/bin/bash

# Training
python tools/train_net.py --num-gpus 1 --config-file ./configs/OWOD/t1/t1_train.yaml SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.005 SOLVER.MAX_ITER 1000 SOLVER.STEPS "(500,)" SOLVER.CHECKPOINT_PERIOD 1000 OUTPUT_DIR ./output/t1_train

# Inference / Demonstration
python demo/demo.py --config-file ./configs/OWOD/t1/t1_test.yaml --input datasets/OWOD/JPEGImages/*.jpg --output ./output/demo --opts MODEL.WEIGHTS ./output/t1_train/model_final.pth

# Testing / Evaluation
python tools/train_net.py --eval-only --num-gpus 1 --config-file ./configs/OWOD/t1/t1_test.yaml OUTPUT_DIR ./output/t1_test MODEL.WEIGHTS ./output/t1_train/model_final.pth
```