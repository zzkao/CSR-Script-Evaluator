#!/bin/bash

# Training
python scripts/train.py --use_multiview --use_normal --use_topdown --use_relation --num_graph_steps 2 --num_locals 10 --batch_size 8 --epoch 50

# Inference / Demonstration
python scripts/visualize.py --folder output --scene_id scene0000_00 --use_multiview --use_normal --use_topdown --use_relation --num_graph_steps 2 --num_locals 10

# Testing / Evaluation
python scripts/eval.py --folder output --reference --use_multiview --use_normal --use_topdown --use_relation --num_graph_steps 2 --num_locals 10 --no_nms --force --repeat 5
```