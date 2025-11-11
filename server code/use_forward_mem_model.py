# use_forward_mem_model.py
import numpy as np, pandas as pd, joblib, sklearn, os, glob

print("sklearn runtime:", sklearn.__version__)

def _resolve_model_path():
    stable = "forward_mem_poly3_ridge.joblib"
    if os.path.exists(stable):
        return stable
    matches = sorted(glob.glob("forward_mem_poly3_ridge*.joblib"))
    if not matches:
        raise FileNotFoundError("No forward memory model .joblib found in CWD.")
    return matches[-1]

try:
    MODEL_PATH = _resolve_model_path()
    pipe = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"WARNING: Forward memory model not available: {e}")
    pipe = None

def predict_score(age: float, gender: str) -> float:
    if pipe is None: raise RuntimeError("Forward memory model not loaded")
    X = pd.DataFrame({'age':[age], 'gender':[gender]})
    return float(pipe.predict(X)[0])

def estimate_age_from_score(gender: str, score: float, lo: float=18.0, hi: float=90.0, step: float=0.05):
    if pipe is None: raise RuntimeError("Forward memory model not loaded")
    ages = np.arange(lo, hi + step, step)
    X = pd.DataFrame({'age': ages, 'gender': [gender]*len(ages)})
    preds = pipe.predict(X).astype(float)
    idx = int(np.argmin(np.abs(preds - score)))
    return float(ages[idx]), float(preds[idx])
