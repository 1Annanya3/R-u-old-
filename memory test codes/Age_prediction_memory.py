import numpy as np, pandas as pd, joblib

MODEL_PATH = "forward_mem_poly3_ridge_scaled_545.5594781168514.joblib"

pipe = joblib.load(MODEL_PATH)

def predict_score(age: float, gender: str) -> float:
    X = pd.DataFrame({'age':[age], 'gender':[gender]})
    return float(pipe.predict(X)[0])

def estimate_age_from_score(gender: str, score: float, lo: float=18.0, hi: float=90.0, step: float=0.05):
    ages = np.arange(lo, hi + step, step)
    X = pd.DataFrame({'age': ages, 'gender': [gender]*len(ages)})
    preds = pipe.predict(X).astype(float)
    idx = int(np.argmin(np.abs(preds - score)))
    return float(ages[idx]), float(preds[idx])

# Examples:
# print(predict_score(25.0, "Female"))
# print(estimate_age_from_score("Male", 10.5))
