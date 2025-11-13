# config.py
import os
import torch

project_root = os.path.dirname(__file__)

config = {
    "imgwidth": 128,
    "imgheight": 128,
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
    "modelname": "resnet",  # unused by AgeCNN, keep for future
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "weights_dir": os.path.join(project_root, "weights"),
    "logdir": os.path.join(project_root, "logs"),
    "reload_checkpoint": None,  # or set to a specific .pt
}
