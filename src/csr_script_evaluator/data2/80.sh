#!/bin/bash

# Training
python -m mamba_ssm.models.mixer_seq_simple --dataset cifar10 --model mamba --n_layer 4 --d_model 64 --batch_size 32 --epochs 1

# Inference / Demonstration
python -c "
from mamba_ssm import Mamba
import torch

batch, length, dim = 2, 64, 16
x = torch.randn(batch, length, dim).to('cuda')
model = Mamba(
    d_model=dim,
    d_state=16,
    d_conv=4,
    expand=2,
).to('cuda')
y = model(x)
assert y.shape == x.shape
print('Inference test passed')
"

# Testing / Evaluation
pytest tests/
```