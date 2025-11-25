#!/bin/bash

# Training
python -m pytorch_lightning.utilities.cli fit --config configs/solar_nips.yaml --trainer.max_epochs 1 --trainer.limit_train_batches 10

# Inference / Demonstration
python -m pytorch_lightning.utilities.cli predict --config configs/solar_nips.yaml --ckpt_path trained_models/solar_nips.ckpt

# Testing / Evaluation
python -m pytorch_lightning.utilities.cli test --config configs/solar_nips.yaml --ckpt_path trained_models/solar_nips.ckpt
```