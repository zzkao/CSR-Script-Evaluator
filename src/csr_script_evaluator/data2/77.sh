#!/bin/bash

# Training

# Inference / Demonstration
./build/bin/main -m ./ReluLLaMA-7B/llama-7b-relu.powerinfer.gguf -n 128 -t 8 -p "Once upon a time"

# Testing / Evaluation

```