# train_forward_mem_model_poly3_ridge_fixed.py
import re, numpy as np, pandas as pd, joblib
from sklearn.model_selection import KFold, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error

INPUT_CSV = "forward_summary.csv.csv"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def age_band_midpoint(b):
    b = str(b).strip()
    m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", b)
    if m: return (int(m.group(1)) + int(m.group(2))) / 2.0
    m2 = re.match(r"^\s*(\d+)\s*\+\s*$", b)
    if m2: return float(m2.group(1)) + 5.0
    try: return float(b)
    except: return np.nan

raw = pd.read_csv(INPUT_CSV)
raw['specific_subtest_id'] = raw['specific_subtest_id'].astype(str).str.strip()
raw = raw[raw['specific_subtest_id']=='43'].copy()
raw['age_mid'] = raw['age'].apply(age_band_midpoint)

df = raw[['age_mid','gender','N','mean']].dropna().copy()
df.rename(columns={'age_mid':'age','mean':'memory_score'}, inplace=True)
df['N'] = df['N'].astype(int)

X = df[['age','gender']]
y = df['memory_score'].values
w = df['N'].values

# Preprocess:
# - numeric: Standardize age, then polynomial expansion (degree=3)
# - categorical: One-hot gender
num_pipe = Pipeline([
    ('sc', StandardScaler(with_mean=True, with_std=True)),
    ('poly', PolynomialFeatures(degree=3, include_bias=False))
])

ct = ColumnTransformer(
    transformers=[
        ('num', num_pipe, ['age']),
        ('cat', OneHotEncoder(drop='if_binary', handle_unknown='ignore'), ['gender'])
    ],
    remainder='drop'
)

# Slightly stronger alpha grid to avoid ill-conditioning warnings
alphas = np.logspace(-2, 3, 20)  # 0.01 ... 1000
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

best = {'alpha': None, 'cv_r2': -np.inf, 'model': None}
for a in alphas:
    pipe = Pipeline([
        ('ct', ct),
        ('ridge', Ridge(alpha=a, fit_intercept=True, random_state=RANDOM_SEED))
    ])
    r2s = []
    for tr_idx, te_idx in cv.split(X):
        Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]
        wtr, wte = w[tr_idx], w[te_idx]
        pipe.fit(Xtr, ytr, ridge__sample_weight=wtr)
        yp = pipe.predict(Xte)
        r2s.append(r2_score(yte, yp, sample_weight=wte))
    m = float(np.mean(r2s))
    if m > best['cv_r2']:
        best.update({'alpha': a, 'cv_r2': m, 'model': pipe})

# Final hold-out check
Xtr, Xte, ytr, yte, wtr, wte = train_test_split(X, y, w, test_size=0.25, random_state=RANDOM_SEED)
best['model'].fit(Xtr, ytr, ridge__sample_weight=wtr)
yp = best['model'].predict(Xte)
print(f"Best alpha={best['alpha']:.4g} | CV R^2={best['cv_r2']:.3f}")
print(f"Hold-out R^2={r2_score(yte, yp, sample_weight=wte):.3f}  MAE={mean_absolute_error(yte, yp):.3f}")

joblib.dump(best['model'], f"forward_mem_poly3_ridge_scaled_{best['alpha']}.joblib")
