#!/bin/bash

# Training
python main_molecules_graph_regression.py --dataset ZINC --gpu_id 0 --config 'configs/molecules_graph_regression_ZINC_GraphTransformer_LapPE_500k.json' --batch_size 32 --epochs 1 --init_lr 0.001 --lr_reduce_factor 0.5 --lr_schedule_patience 10 --min_lr 1e-6 --weight_decay 0

# Inference / Demonstration
python main_molecules_graph_regression.py --dataset ZINC --gpu_id 0 --config 'configs/molecules_graph_regression_ZINC_GraphTransformer_LapPE_500k.json' --batch_size 32

# Testing / Evaluation
python main_molecules_graph_regression.py --dataset ZINC --gpu_id 0 --config 'configs/molecules_graph_regression_ZINC_GraphTransformer_LapPE_500k.json' --batch_size 32
```