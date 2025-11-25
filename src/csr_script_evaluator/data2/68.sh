#!/bin/bash

# Training
mistral-finetune --data data/ultrachat_100.jsonl --model-id ministral-8b-instruct-2410 --out-dir out --max-steps 10 --seq-len 512 --batch-size 1 --checkpoint-interval 100

# Inference / Demonstration
mistral-finetune-eval --model-id out --instruct true --max-tokens 256 --prompt "What is the best French cheese?"

# Testing / Evaluation

```