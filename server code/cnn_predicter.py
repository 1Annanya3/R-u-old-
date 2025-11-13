# cnn_predictor.py
import os
import torch
import numpy as np
import cv2
import torchvision.transforms as T
from typing import Optional

# Minimal config for inference; you can import from config if preferred
IMG_SIZE = 128
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Your model definition should match training.
# If cnn_train.py used AgeCNN, re-declare it here identically.
import torch.nn as nn

class AgeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 16 * 16, 512),  # 128x128 input => after 3 pools: 16x16
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# Lazy singleton loader
_model: Optional[AgeCNN] = None
_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])

def _load_latest_weights(weights_dir: str) -> Optional[str]:
    if not os.path.isdir(weights_dir):
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
    _model = AgeCNN().to(DEVICE).eval()
    if weights_path is None:
        weights_path = _load_latest_weights(os.path.join(os.path.dirname(__file__), "weights"))
    if weights_path and os.path.isfile(weights_path):
        state = torch.load(weights_path, map_location=DEVICE)
        # Accept either plain state_dict or checkpoint dict
        if isinstance(state, dict) and "state_dict" in state:
            _model.load_state_dict(state["state_dict"])
        else:
            _model.load_state_dict(state)
    # else run with random init (useful for plumbing tests)

def predict_face_age(img_bgr: np.ndarray) -> float:
    """
    Accepts BGR uint8 image (from cv2.imdecode) and returns age (float).
    """
    _ensure_model()
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Empty image")
    # Convert BGR -> RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # Resize to training size
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    # To tensor and normalize
    tensor = _transform(img_resized).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = _model(tensor).squeeze(0).squeeze(-1).float()
    age = float(out.item())
    # Clamp to a reasonable range if desired
    age = max(0.0, min(110.0, age))
    return age
