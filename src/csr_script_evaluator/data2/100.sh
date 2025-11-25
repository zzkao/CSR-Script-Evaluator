#!/bin/bash

# Training
tape-train transformer masked_language_modeling --batch_size 1 --num_train_epochs 1 --gradient_accumulation_steps 1 --learning_rate 1e-4 --fp16 false --warmup_steps 10 --num_log_iter 10 --save_freq 1

# Inference / Demonstration
tape-embed transformer pfam --load_from pretrained_models/transformer.h5 --batch_size 1024

# Testing / Evaluation
tape-eval transformer stability --from_pretrained bert-base --batch_size 1
```