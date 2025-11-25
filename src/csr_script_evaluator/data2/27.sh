#!/bin/bash
# Training

# Inference / Demonstration
python visualize_scene_layout.py --scene_path Structured3D/scene_00000 --use_textured_mesh
python visualize_bbox.py --scene_path Structured3D/scene_00000 --use_textured_mesh
python project_bbox_to_2d.py --scene_path Structured3D/scene_00000

# Testing / Evaluation
python calc_statistics_layout.py --dataset_path Structured3D/
python calc_statistics_3dobject.py --dataset_path Structured3D/
```