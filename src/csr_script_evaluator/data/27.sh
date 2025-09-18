#!/bin/bash

# Environment Setup / Requirement / Installation
# Clone repository
git clone git@github.com:bertjiazheng/Structured3D.git
cd Structured3D

# Install dependencies
conda install -y open3d -c open3d-admin
conda install -y opencv -c conda-forge
conda install -y descartes matplotlib numpy shapely
pip install panda3d

# Data / Checkpoint / Weight Download (URL)
# Note: Dataset requires filling agreement form at https://forms.gle/LXg4bcjC2aEjrL9o8
# After approval, download and extract dataset to /path/to/dataset

# Training
# No training commands in README

# Inference / Demonstration
# Visualize 3D annotations (wireframe/plane/floorplan)
python visualize_3d.py --path /path/to/dataset --scene scene_00000 --type wireframe
python visualize_3d.py --path /path/to/dataset --scene scene_00000 --type plane
python visualize_3d.py --path /path/to/dataset --scene scene_00000 --type floorplan

# Visualize 3D textured mesh
python visualize_mesh.py --path /path/to/dataset --scene scene_00000 --room room_0

# Visualize 2D layout (perspective/panorama)
python visualize_layout.py --path /path/to/dataset --scene scene_00000 --type perspective
python visualize_layout.py --path /path/to/dataset --scene scene_00000 --type panorama

# Visualize 3D bounding box
python visualize_bbox.py --path /path/to/dataset --scene scene_00000

# Visualize floorplan
python visualize_floorplan.py --path /path/to/dataset --scene scene_00000

# Testing / Evaluation
# No testing commands in README