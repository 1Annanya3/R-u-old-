# use_balance_model.py
import os
import json
import numpy as np
import pandas as pd
import joblib

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "balance_age_model.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "balance_model_metrics.json")

# Lazy-loaded globals
_loaded = False
_meta = None
_model_obj = None
_min_dur = None
_max_dur = None

def _load():
    global _loaded, _meta, _model_obj, _min_dur, _max_dur
    if _loaded:
        return
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model at {MODEL_PATH}")
    if not os.path.exists(METRICS_PATH):
        raise FileNotFoundError(f"Missing metrics at {METRICS_PATH}")

    with open(METRICS_PATH, "r") as f:
        _meta = json.load(f)
    _model_obj = joblib.load(MODEL_PATH)

    # Optional: load training range if stored; otherwise infer at runtime if needed.
    # For simplicity, leave unset; caller can clamp with observed session range.
    _min_dur = None
    _max_dur = None
    _loaded = True

def _normalize_gender(g: str) -> str:
    g = str(g).strip().upper()
    if g in ("M","F"):
        return g
    return "O"

def estimate_age_from_balance(gender: str, balance_duration_s: float, clamp_to_range: tuple[float,float] | None = None) -> tuple[float, dict]:
    """
    Returns (predicted_age, info_dict).
    clamp_to_range=(min,max) optionally clamps balance_duration_s before prediction.
    """
    _load()
    g = _normalize_gender(gender)
    x = float(balance_duration_s)
    if clamp_to_range is not None:
        lo, hi = clamp_to_range
        x = max(lo, min(hi, x))

    schema = _meta.get("feature_schema", {})
    if schema.get("type") == "raw":
        # polynomial pipeline path
        inp = pd.DataFrame([{"participant_gender": g, "balance_duration_s": x}])
        pred = float(_model_obj.predict(inp)[0])
        return pred, {"model": "poly", "clamped": clamp_to_range is not None}
    else:
        # svr path with fixed columns
        cols = schema["columns"]
        base = pd.DataFrame([{"participant_gender": g, "balance_duration_s": x}])
        enc = pd.get_dummies(base, columns=["participant_gender"], drop_first=True)
        for c in cols:
            if c not in enc.columns:
                enc[c] = 0
        enc = enc[cols]
        model = _model_obj["model"] if isinstance(_model_obj, dict) else _model_obj
        pred = float(model.predict(enc)[0])
        return pred, {"model": "svr", "clamped": clamp_to_range is not None}

# Simple CLI for manual test
if __name__ == "__main__":
    import sys
    g = sys.argv[1] if len(sys.argv) > 1 else "F"
    d = float(sys.argv[2]) if len(sys.argv) > 2 else 12.3
    y, info = estimate_age_from_balance(g, d)
    print(f"Predicted age: {y:.2f} ({info})")
