#!/bin/bash

# Training
python run.py --config experiments/train_cat.yml --gpu 0 --num_workers 4 --batch_size 2 --max_epoch 1

# Inference / Demonstration
python -m demo.demo --input demo/images/human_face.png --result demo/results/human_face --checkpoint pretrained/pretrained_celeba/checkpoint030.pth
python -m demo.demo --input demo/images/cat_face.png --result demo/results/cat_face --checkpoint pretrained/pretrained_cat/checkpoint030.pth

# Testing / Evaluation
python run.py --config experiments/eval_cat.yml --gpu 0 --num_workers 4 --batch_size 2
```