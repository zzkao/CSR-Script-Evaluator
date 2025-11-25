#!/bin/bash

# Training
python train.py +experiment=small-parseq data.batch_size=8 trainer.max_epochs=1 data.root_dir=data/

# Inference / Demonstration
python read.py demo.png

# Testing / Evaluation
python test.py pretrained=parseq
```