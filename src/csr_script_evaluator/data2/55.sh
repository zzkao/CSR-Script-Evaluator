#!/bin/bash

# Training
python train.py --dataset coco --max_epoch 1 --lr 1e-4 --batch_size 1 --imdb_name coco_minus_refer --net res101 --optimizer adam

# Inference / Demonstration
python test.py --dataset coco --net res101 --checkepoch 28 --checkpoint 14680 --imdb_name coco_minus_refer --load_dir models

# Testing / Evaluation
python eval.py --dataset coco --net res101 --checkepoch 28 --checkpoint 14680 --imdb_name coco_minus_refer --load_dir models
```