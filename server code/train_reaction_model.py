# file: train_reaction_model_export_equation.py
# Trains a degree-2 polynomial LinearRegression on Reaction Time Test sheet
# and exports a JSON "equation" with intercept and named coefficients.

import json
import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

SERVICE_ACCOUNT_FILE = "service_account.json"
SPREADSHEET_NAME = "CS3237_Health_Assessment_Data"
WORKSHEET_NAME = "Reaction Time Test"
OUTPUT_JSON = "reaction_equation.json"

def main():
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(SERVICE_ACCOUNT_FILE, scope)
    client = gspread.authorize(creds)

    sheet = client.open(SPREADSHEET_NAME).worksheet(WORKSHEET_NAME)
    df = pd.DataFrame(sheet.get_all_records())
    req = ["participant_gender", "reaction_time_ms", "actual_age"]
    assert all(c in df.columns for c in req), f"Missing columns; need {req}"

    df["reaction_time_ms"] = df["reaction_time_ms"].astype(str).str.replace(",", "", regex=False).astype(float)
    df["actual_age"] = df["actual_age"].astype(float)

    X = df[["participant_gender", "reaction_time_ms"]]
    y = df["actual_age"].values

    pre = ColumnTransformer(
        transformers=[
            ("gender", OneHotEncoder(categories=[["M","F"]], drop="first"), ["participant_gender"]),
            ("num", "passthrough", ["reaction_time_ms"])
        ]
    )
    pipe = Pipeline(steps=[
        ("pre", pre),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("lr", LinearRegression())
    ])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(Xtr, ytr)
    r2 = float(pipe.score(Xte, yte))
    print(f"Test R^2 = {r2:.3f}")

    # Extract feature names after poly
    pre_out = []
    ohe = pipe.named_steps["pre"].named_transformers_["gender"]
    # With drop='first' on ["M","F"], output has 1 column for "F"
    gender_names = [f"gender={cat}" for cat in ohe.categories_[0][1:]]  # ["F"]
    pre_out.extend(gender_names)
    pre_out.append("reaction_time_ms")

    poly = pipe.named_steps["poly"]
    names = poly.get_feature_names_out(pre_out).tolist()

    coef = pipe.named_steps["lr"].coef_.ravel().tolist()
    intercept = float(pipe.named_steps["lr"].intercept_)

    equation = {
        "intercept": intercept,
        "terms": dict(zip(names, coef)),
        "r2_test": r2,
        "encoding": {
            "gender_baseline": "M",
            "gender_onehot": "F",
            "feature_order_prepoly": pre_out,
            "poly_degree": 2,
            "include_bias": False
        }
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(equation, f, indent=2)
    print(f"Wrote {OUTPUT_JSON} with {len(names)} terms")

if __name__ == "__main__":
    main()
