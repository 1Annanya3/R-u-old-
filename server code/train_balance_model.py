# train_balance_model.py
import os
import json
import numpy as np
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ============== CONFIG ==============
SERVICE_ACCOUNT_FILE = "service_account.json"
SPREADSHEET_NAME = "CS3237_Health_Assessment_Data"
WORKSHEET_NAME = "Balance Test (Synthetic)"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "balance_age_model.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "balance_model_metrics.json")
CHOSEN_MODEL = "poly"  # "poly" or "svr"
# ====================================

def load_balance_sheet():
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(SERVICE_ACCOUNT_FILE, scope)
    gc = gspread.authorize(creds)
    sh = gc.open(SPREADSHEET_NAME)
    ws = sh.worksheet(WORKSHEET_NAME)
    data = ws.get_all_records()
    return pd.DataFrame(data)

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    req = ["participant_gender", "balance_duration_s", "actual_age"]
    missing = [c for c in req if c not in df.columns]
    assert not missing, f"Missing required columns: {missing}"
    df = df.copy()
    df["balance_duration_s"] = (
        df["balance_duration_s"].astype(str).str.replace(",", "", regex=False).astype(float)
    )
    df["actual_age"] = df["actual_age"].astype(float).astype(int)
    # Normalize gender tokens to "M"/"F"/"O"
    df["participant_gender"] = (
        df["participant_gender"].astype(str).str.strip().str.upper().map({"M": "M", "F": "F"}).fillna("O")
    )
    return df

def train_poly(X_train, y_train):
    pre = ColumnTransformer([
        ("gender", OneHotEncoder(categories=[["M","F","O"]], drop="first", handle_unknown="ignore"), ["participant_gender"]),
        ("dur", "passthrough", ["balance_duration_s"]),
    ])
    poly = Pipeline(steps=[
        ("pre", pre),
        ("poly", PolynomialFeatures(degree=3, include_bias=False)),
        ("reg", LinearRegression())
    ])
    poly.fit(X_train, y_train)
    return poly

def train_svr(X_enc_train, y_train):
    svr_pipe = make_pipeline(
        StandardScaler(),
        SVR(kernel="rbf", C=100, epsilon=0.1)
    )
    svr_pipe.fit(X_enc_train, y_train)
    return svr_pipe

def evaluate(y_true, y_pred):
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_balance_sheet()
    df = clean_df(df)

    X = df[["participant_gender", "balance_duration_s"]]
    y = df["actual_age"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train polynomial
    poly_model = train_poly(X_train, y_train)
    y_pred_poly = poly_model.predict(X_test)
    metrics_poly = evaluate(y_test, y_pred_poly)
    print("Polynomial Regression:", metrics_poly)

    # Train SVR (with one-hot outside the pipeline for clarity)
    X_enc = pd.get_dummies(X, columns=["participant_gender"], drop_first=True)
    X_enc_train, X_enc_test, y_enc_train, y_enc_test = train_test_split(
        X_enc, y, test_size=0.2, random_state=42
    )
    svr_model = train_svr(X_enc_train, y_enc_train)
    y_pred_svr = svr_model.predict(X_enc_test)
    metrics_svr = evaluate(y_enc_test, y_pred_svr)
    print("SVR:", metrics_svr)

    # Persist chosen model + metadata
    meta = {
        "chosen": CHOSEN_MODEL,
        "poly_metrics": metrics_poly,
        "svr_metrics": metrics_svr,
        "feature_schema": {
            "type": "raw",
            "features": ["participant_gender", "balance_duration_s"],
            "gender_vocab": ["M","F","O"]
        }
    }

    if CHOSEN_MODEL == "poly":
        joblib.dump(poly_model, MODEL_PATH)
    else:
        # For SVR we will also store the columns used
        meta["feature_schema"] = {
            "type": "onehot",
            "columns": list(X_enc.columns)  # order matters
        }
        joblib.dump({"model": svr_model, "columns": list(X_enc.columns)}, MODEL_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")

if __name__ == "__main__":
    main()
