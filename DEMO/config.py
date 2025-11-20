# config.py
import os
import torch

project_root = os.path.dirname(__file__)

config = {
    # Image geometry
    "imgwidth": 128,
    "imgheight": 128,
    "img_size": 128,  

    # Normalization
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),

    # Model selection 
    "model_name": "resnet",

    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # Paths
    "weights_dir": os.path.join(project_root, "weights"),
    # CRITICAL: Point directly to the ResNet weights file
    "reload_checkpoint": os.path.join(project_root, "age_cnn.pt"), 

    # CLI test defaults (used by cnn_infer.py)
    "image_path_test": os.path.join(project_root, "sample.jpg"),
    "output_path_test": os.path.join(project_root, "sample_out.jpg"),

    # Optional logging
    "logdir": os.path.join(project_root, "logs"),
}
