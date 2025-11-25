#!/bin/bash

# Training
python ./train.py --dataset ./data/train.txt --model_checkpoint gpt2 --gradient_accumulation_steps 4 --lm_coef 2.0 --max_history 2 --n_epochs 1 --num_candidates 2 --personality_permutations 1 --train_batch_size 2 --valid_batch_size 2

# Inference / Demonstration
python ./interact.py --model_checkpoint ./runs/

# Testing / Evaluation
python ./evaluate.py --dataset ./data/valid.txt --model_checkpoint ./runs/
```