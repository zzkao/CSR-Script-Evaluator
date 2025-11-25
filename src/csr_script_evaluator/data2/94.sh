#!/bin/bash

# Training
python -m train wandb=null experiment=hg38/hg38_hyena trainer.devices=1 dataset.max_length=1024 train.global_batch_size=8 trainer.max_epochs=1 dataset.batch_size=8

# Inference / Demonstration
python standalone_hyenadna.py --model_name hyenadna-tiny-1k-seqlen

# Testing / Evaluation
python -m evals.nucleotide_transformer callback.downstream_task_dir=data dataset.max_length=1024 dataset.batch_size=8 model.d_model=128 model.n_layer=2
```