#!/bin/bash

# Environment Setup / Requirement / Installation
# Clone repository and install requirements
git clone https://github.com/huggingface/naacl_transfer_learning_tutorial
cd naacl_transfer_learning_tutorial
pip install -r requirements.txt

# Data / Checkpoint / Weight Download (URL)
# Data will be automatically downloaded during training
# wikitext-103 is the default dataset

# Training
# Pre-training on single GPU
python ./pretraining_train.py

# Pre-training on multiple GPUs (example with 8 GPUs)
python -m torch.distributed.launch --nproc_per_node 8 ./pretraining_train.py

# Fine-tuning on single GPU (using pretrained model)
python ./finetuning_train.py --model_checkpoint ./runs/pretrained_model

# Fine-tuning on multiple GPUs (example with 8 GPUs)
python -m torch.distributed.launch --nproc_per_node 8 ./finetuning_train.py --model_checkpoint ./runs/pretrained_model

# Inference / Demonstration
# No explicit inference commands in README
# Inference would be part of the fine-tuning evaluation

# Testing / Evaluation
# View available pre-training options
python ./pretraining_train.py --help

# View available fine-tuning options
python ./finetuning_train.py --help