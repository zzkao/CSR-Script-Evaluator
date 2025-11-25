#!/bin/bash

# Training
python applications/train.py --outdir=training-runs/bicycle --data=pretrained/bicycle.pkl --num_heads=1 --flow_size=256 --num_epochs=1 --batch_size=1

# Inference / Demonstration
python applications/stn.py --network=pretrained/bicycle.pkl --flow_size=256 --outdir=out

# Testing / Evaluation
python applications/pck.py --network=pretrained/bicycle.pkl --dataset=cat --flow_size=256
```