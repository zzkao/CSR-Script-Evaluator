#!/bin/bash

# Data / Checkpoint / Weight Download (URL)
wget -O data/checkpoints/aist_vibe_3D_checkpoint.pth.tar "https://drive.google.com/uc?export=download&id=101TH_Z8uiXD58d_xkuFTh5bI4NtRm_cK"
wget -O data/checkpoints/h36m_fcn_3D_checkpoint.pth.tar "https://drive.google.com/uc?export=download&id=1ZketGlY4qA3kFp044T1-PaykV2llNUjB"
wget -O data/checkpoints/pw3d_spin_3D_checkpoint.pth.tar "https://drive.google.com/uc?export=download&id=106MnTXFLfMlJ2W7Fvw2vAlsUuQFdUe6k"
mkdir -p data/checkpoints/h36m_fcn_3D data/checkpoints/pw3d_spin_3D data/checkpoints/aist_vibe_3D
mv data/checkpoints/h36m_fcn_3D_checkpoint.pth.tar data/checkpoints/h36m_fcn_3D/checkpoint_8.pth.tar
mv data/checkpoints/pw3d_spin_3D_checkpoint.pth.tar data/checkpoints/pw3d_spin_3D/checkpoint_8.pth.tar
mv data/checkpoints/aist_vibe_3D_checkpoint.pth.tar data/checkpoints/aist_vibe_3D/checkpoint_8.pth.tar

# Training
python train_smoothnet.py --cfg configs/h36m_fcn_3D.yaml --dataset_name h36m --estimator fcn --body_representation 3D --slide_window_size 8
python train_smoothnet.py --cfg configs/h36m_fcn_3D.yaml --dataset_name aist,pw3d --estimator vibe,spin --body_representation 3D,3D --slide_window_size 8
python train_smoothnet.py --cfg configs/pw3d_spin_3D.yaml --dataset_name pw3d --estimator spin --body_representation 3D --slide_window_size 16
python train_smoothnet.py --cfg configs/aist_vibe_3D.yaml --dataset_name aist --estimator vibe --body_representation 3D --slide_window_size 32

# Inference / Demonstration
python visualize_smoothnet.py --cfg configs/pw3d_spin_3D.yaml --checkpoint data/checkpoints/pw3d_spin_3D/checkpoint_8.pth.tar --dataset_name pw3d --estimator spin --body_representation 3D --slide_window_size 32 --visualize_video_id 2 --output_video_path ./visualize
python visualize_smoothnet.py --cfg configs/h36m_fcn_3D.yaml --checkpoint data/checkpoints/h36m_fcn_3D/checkpoint_8.pth.tar --dataset_name h36m --estimator fcn --body_representation 3D --slide_window_size 8 --visualize_video_id 1 --output_video_path ./visualize
python visualize_smoothnet.py --cfg configs/aist_vibe_3D.yaml --checkpoint data/checkpoints/aist_vibe_3D/checkpoint_8.pth.tar --dataset_name aist --estimator vibe --body_representation 3D --slide_window_size 16 --visualize_video_id 0 --output_video_path ./visualize

# Testing / Evaluation
python eval_smoothnet.py --cfg configs/pw3d_spin_3D.yaml --checkpoint data/checkpoints/pw3d_spin_3D/checkpoint_8.pth.tar --dataset_name mpiinf3dhp,mpiinf3dhp --estimator tcmr,vibe --body_representation 3D,3D --slide_window_size 8 --tradition oneeuro
python eval_smoothnet.py --cfg configs/h36m_fcn_3D.yaml --checkpoint data/checkpoints/h36m_fcn_3D/checkpoint_8.pth.tar --dataset_name h36m --estimator fcn --body_representation 3D --slide_window_size 8 --tradition savgol
python eval_smoothnet.py --cfg configs/pw3d_spin_3D.yaml --checkpoint data/checkpoints/pw3d_spin_3D/checkpoint_8.pth.tar --dataset_name pw3d --estimator spin --body_representation 3D --slide_window_size 16 --tradition gaus1d
python eval_smoothnet.py --cfg configs/aist_vibe_3D.yaml --checkpoint data/checkpoints/aist_vibe_3D/checkpoint_8.pth.tar --dataset_name aist --estimator vibe --body_representation 3D --slide_window_size 8
python eval_smoothnet.py --cfg configs/h36m_fcn_3D.yaml --checkpoint data/checkpoints/h36m_fcn_3D/checkpoint_8.pth.tar --dataset_name h36m --estimator rle --body_representation 2D --slide_window_size 8