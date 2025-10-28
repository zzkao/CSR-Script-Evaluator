#!/bin/bash

# Training
mkdir -p logs/LongForecasting
python -u run_longExp.py --is_training 1 --root_path ./dataset/ --data_path ./scripts/EXP-LongForecasting/Linear/exchange_rate.csv --model_id Exchange_336_96 --model DLinear --data custom --features M --seq_len 336 --pred_len 96 --enc_in 8 --des 'Exp' --itr 1 --batch_size 8 --learning_rate 0.0005 > logs/LongForecasting/DLinear_Exchange_336_96.log

# Inference / Demonstration
mkdir -p weights_plot
python weight_plot.py

# Testing / Evaluation
# Note: Evaluation is integrated into the training process above
# Results are logged in logs/LongForecasting/ directory
# Model checkpoints are saved in checkpoints/ directory for further analysis
