#!/bin/bash

# Training
# No training commands provided in README

# Inference / Demonstration
torchrun --nproc_per_node 1 example_chat_completion.py --ckpt_dir Meta-Llama3.1-8B-Instruct/ --tokenizer_path Meta-Llama3.1-8B-Instruct/tokenizer.model --max_seq_len 512 --max_batch_size 4

torchrun --nproc_per_node 1 example_text_completion.py --ckpt_dir Meta-Llama3.1-8B/ --tokenizer_path Meta-Llama3.1-8B/tokenizer.model --max_seq_len 128 --max_batch_size 4

# Testing / Evaluation
# No testing/evaluation commands provided in README
```