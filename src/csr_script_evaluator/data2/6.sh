#!/bin/bash

# Training
torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_port=12345 pretrain.py --data_path=/path/to/imagenet --model=resnet50 --batch_size=64 --epochs=1

# Inference / Demonstration
python demo.py --data_path=/path/to/imagenet --model=resnet50 --ckpt_path=/path/to/checkpoint.pth

# Testing / Evaluation
torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_port=12345 downstream_imagenet/linear_probe.py --data_path=/path/to/imagenet --model=resnet50 --init_weight=/path/to/checkpoint.pth --batch_size=64 --epochs=1
torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_port=12345 downstream_imagenet/finetune.py --data_path=/path/to/imagenet --model=resnet50 --init_weight=/path/to/checkpoint.pth --batch_size=64 --epochs=1
```