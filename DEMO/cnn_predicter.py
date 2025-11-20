# cnn_predicter.py
import os
from typing import Optional

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np

from config import config
from model import AgeEstimationModel 

IMG_SIZE = int(config.get("img_size", config.get("imgwidth", 128)))
MEAN = tuple(config.get("mean", (0.485, 0.456, 0.406)))
STD = tuple(config.get("std", (0.229, 0.224, 0.225)))
DEVICE = torch.device(config.get("device", "cpu"))
WEIGHTS_DIR = config.get("weights_dir")
RELOAD_CKPT = config.get("reload_checkpoint")


_model: Optional[AgeEstimationModel] = None
_transform = T.Compose([T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])

def _load_latest_weights(weights_dir: str) -> Optional[str]:
    if not weights_dir or not os.path.isdir(weights_dir):
        return None
    cand = [os.path.join(weights_dir, f) for f in os.listdir(weights_dir) if f.endswith(".pt")]
    if not cand:
        return None
    cand.sort(key=os.path.getmtime, reverse=True)
    return cand[0]

def _ensure_model(weights_path: Optional[str] = None):
    global _model
    if _model is not None:
        return
        
    model_name = config.get("model_name", "resnet")
    _model = AgeEstimationModel(
        input_dim=IMG_SIZE, 
        output_nodes=1, 
        model_name=model_name, 
        pretrain_weights=None # loading weights from a file not a pretrained source 
    ).to(DEVICE).eval()
    
    path = weights_path or RELOAD_CKPT or _load_latest_weights(WEIGHTS_DIR)
    
    if path and os.path.isfile(path):
        print(f"Loading weights from: {path}") 
        state = torch.load(path, map_location=DEVICE)
        
        if isinstance(state, dict) and "model_state_dict" in state:
            _model.load_state_dict(state["model_state_dict"])
        elif isinstance(state, dict):
            _model.load_state_dict(state)
        else:
             _model.load_state_dict(state)
    else:
        print(f"WARNING: No weights found at {path}. Model will use untrained initial weights.")


def predict_face_age(img_bgr: np.ndarray, weights_path: Optional[str] = None) -> float:
    """
    Accepts BGR uint8 image (from cv2.imdecode) and returns age (float).
    """
    _ensure_model(weights_path)
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Empty image")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    tensor = _transform(img_resized).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = _model(tensor).squeeze(0).squeeze(-1).float()
        # The ResNet model outputs a tensor, we still need to extract the float value
        age = float(out.item())

    return max(0.0, min(110.0, age))
