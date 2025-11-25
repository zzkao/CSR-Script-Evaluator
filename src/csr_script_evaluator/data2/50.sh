#!/bin/bash

# Training
python train.py --data_path data --vocab_path vocab --data_name f30k_precomp --num_epochs 1 --batch_size 32 --logger_name runs/sgraf_f30k

# Inference / Demonstration
python demo.py

# Testing / Evaluation
python test.py --data_path data --vocab_path vocab --data_name f30k_precomp
```