#!/bin/bash
# Training
python trainer.py --batch_size 4 --dataset shapenet --num_input 3 --max_steps 1

# Inference / Demonstration
python evaler.py --batch_size 4 --dataset shapenet --num_input 3 --train_dir train_dir/shapenet-num_input_3

# Testing / Evaluation
python evaler.py --batch_size 4 --dataset shapenet --num_input 3 --train_dir train_dir/shapenet-num_input_3
```