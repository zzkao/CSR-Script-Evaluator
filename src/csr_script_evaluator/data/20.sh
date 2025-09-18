#!/bin/bash

# Environment Setup / Requirement / Installation
# Install mimicry package
pip install git+https://github.com/kwotsin/mimicry.git

# Create directories for data and logs
mkdir -p ./datasets
mkdir -p ./log/example

# Data / Checkpoint / Weight Download (URL)
# Data will be downloaded automatically when loading dataset
# Example using CIFAR-10 dataset (default)

# Training
# Example training SNGAN on CIFAR-10
python -c "
import torch
import torch.optim as optim
import torch_mimicry as mmc
from torch_mimicry.nets import sngan

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load dataset
dataset = mmc.datasets.load_dataset(root='./datasets', name='cifar10')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

# Initialize models and optimizers
netG = sngan.SNGANGenerator32().to(device)
netD = sngan.SNGANDiscriminator32().to(device)
optD = optim.Adam(netD.parameters(), 2e-4, betas=(0.0, 0.9))
optG = optim.Adam(netG.parameters(), 2e-4, betas=(0.0, 0.9))

# Train model
trainer = mmc.training.Trainer(
    netD=netD,
    netG=netG,
    optD=optD,
    optG=optG,
    n_dis=5,
    num_steps=100000,
    lr_decay='linear',
    dataloader=dataloader,
    log_dir='./log/example',
    device=device)
trainer.train()
"

# Inference / Demonstration
# Example generating samples and computing metrics
python -c "
import torch
import torch_mimicry as mmc
from torch_mimicry.nets import sngan

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
netG = sngan.SNGANGenerator32().to(device)

# Evaluate FID score
mmc.metrics.evaluate(
    metric='fid',
    log_dir='./log/example',
    netG=netG,
    dataset='cifar10',
    num_real_samples=50000,
    num_fake_samples=50000,
    evaluate_step=100000,
    device=device)
"

# Testing / Evaluation
# View training progress and results
tensorboard --logdir=./log/example