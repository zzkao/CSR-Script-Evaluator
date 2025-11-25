#!/bin/bash

# Training
python train.py -e 1 -N snarf_386_1ep subject=386 training.epochs=1 training.batch_size=1

# Inference / Demonstration
python render.py mode=pose_novel_view subject=386 render_views=4

# Testing / Evaluation
python test.py subject=386

```