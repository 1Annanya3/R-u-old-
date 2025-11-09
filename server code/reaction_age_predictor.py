{
  "intercept": -12.03531186172301,
  "terms": {
    "gender=F": -4.3659461981361165,
    "reaction_time_ms": 0.1487539858135746,
    "gender=F^2": -4.365946198136601,
    "gender=F reaction_time_ms": 0.0026552972557565063,
    "reaction_time_ms^2": -5.2398297322814313e-05
  },
  "r2_test": 0.34373468271304697,
  "encoding": {
    "gender_baseline": "M",
    "gender_onehot": "F",
    "feature_order_prepoly": [
      "gender=F",
      "reaction_time_ms"
    ],
    "poly_degree": 2,
    "include_bias": false
  }
}(venv) annanya13@DESKTOP-GPRDFD9:~/CS3237_Lab5/project$ cat reaction_age_predictor.py
# file: reaction_age_equation.py
# Loads the exported equation JSON and provides a pure-Python predictor.

import json
from typing import Literal

EQUATION_JSON = "reaction_equation.json"

class ReactionAgeEquation:
    def __init__(self, path: str = EQUATION_JSON):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        self.intercept = float(obj["intercept"])
        self.terms = obj["terms"]  # mapping from feature string to coeff
        enc = obj.get("encoding", {})
        self.gender_onehot = enc.get("gender_onehot", "F")
        self.gender_baseline = enc.get("gender_baseline", "M")

    def predict(self, gender: Literal["M","F"], reaction_time_ms: float) -> float:
        # Build feature values consistent with training naming (degree=2 interactions)
        gF = 1.0 if str(gender).upper() == self.gender_onehot else 0.0
        rt = float(reaction_time_ms)

        # Primitive features present before PolynomialFeatures
        feats = {
            f"gender={self.gender_onehot}": gF,
            "reaction_time_ms": rt
        }

        # Now compute polynomial feature values used by exported terms
        def feat_value(name: str) -> float:
            # names look like: 'reaction_time_ms', 'gender=F', 'reaction_time_ms^2', 'gender=F reaction_time_ms'
            if " " in name:  # interaction, e.g., 'gender=F reaction_time_ms'
                a, b = name.split(" ")
                return feats[a] * feats[b]
            elif "^2" in name:  # squared term
                base = name.replace("^2", "")
                return feats[base] ** 2
            else:
                return feats[name]

        y = self.intercept
        for name, coeff in self.terms.items():
            y += float(coeff) * feat_value(name)
        return float(y)

# Convenience singleton
try:
    _EQ = ReactionAgeEquation()
    def predict_age_from_reaction(gender: Literal["M","F"], reaction_time_ms: float) -> float:
        return _EQ.predict(gender, reaction_time_ms)
except Exception as _e:
    # Fallback that raises helpful error at call site if JSON not present
    def predict_age_from_reaction(gender: Literal["M","F"], reaction_time_ms: float) -> float:
        raise RuntimeError(f"reaction_equation.json not loaded: {_e}")
