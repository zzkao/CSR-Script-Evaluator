#!/bin/bash

# Data / Checkpoint / Weight Download (URL)
# Datasets are included in the repository under /data/corpus
# twitter_asian_prejudice, reuters r8, and AG's news datasets are provided
# For new datasets, prepare [dataset_name]_labels.txt and [dataset_name]_sentences.txt in /data/corpus

# Training
python prep_data.py
python main.py

# Inference / Demonstration
python main.py --dataset twitter_asian_prejudice_small --num_epochs 50 --lr 0.02 --device cpu
python main.py --dataset r8_presplit --num_epochs 100 --lr 0.02 --device cpu
python main.py --dataset ag_presplit --num_epochs 100 --lr 0.02 --device cpu

# Testing / Evaluation
python eval.py
python main.py --dataset twitter_asian_prejudice_small --num_epochs 2 --validation_metric accuracy --device cpu
python main.py --dataset twitter_asian_prejudice_small --num_epochs 2 --validation_metric f1_weighted --device cpu
