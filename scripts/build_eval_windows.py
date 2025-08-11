# scripts/build_eval_windows.py
import argparse, pandas as pd, joblib
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--csv", required=True)
a = p.parse_args()

df = pd.read_csv(a.csv, parse_dates=["date"])

# Force deterministic categories
df["weekday"] = pd.Categorical(
    df["weekday"].astype(int),
    categories=list(range(7)),    # 0..6
    ordered=False
)
df["month"] = pd.Categorical(
    df["month"].astype(int),
    categories=list(range(1, 13)),  # 1..12
    ordered=False
)

# Split windows
ref = df[df["date"] < df["date"].min() + pd.Timedelta(days=90)].copy()
cur = df[df["date"] >= df["date"].min() + pd.Timedelta(days=90)].copy()

def make_features(d):
    X = d[["price","promo_flag","stock","weekday","month"]].copy()
    # get_dummies with fixed categories -> identical columns in ref/cur
    X = pd.get_dummies(X, columns=["weekday","month"], drop_first=True)
    # Ensure numeric and fill any accidental NaNs
    return X.apply(pd.to_numeric, errors="coerce").fillna(0)

# Load features list from the trained bundle (for eval predictions)
bundle = joblib.load("artifacts/models/demand-latest.joblib")
model = bundle["model"]
feats = bundle["features"]

def predict_eval(d):
    X = make_features(d).reindex(columns=feats, fill_value=0)
    y_true = d["units"].values
    y_pred = model.predict(X)
    return pd.DataFrame({"y_true": y_true, "y_pred": y_pred})

# Build matrices for Evidently (same columns for ref/cur)
ref_X = make_features(ref)
cur_X = make_features(cur)

# Optional: enforce identical column set (paranoia)
all_cols = sorted(set(ref_X.columns) | set(cur_X.columns))
ref_X = ref_X.reindex(columns=all_cols, fill_value=0)
cur_X = cur_X.reindex(columns=all_cols, fill_value=0)

Path("artifacts/data").mkdir(parents=True, exist_ok=True)
ref_X.to_csv("artifacts/data/ref_features.csv", index=False)
cur_X.to_csv("artifacts/data/cur_features.csv", index=False)
predict_eval(ref).to_csv("artifacts/data/ref_eval.csv", index=False)
predict_eval(cur).to_csv("artifacts/data/cur_eval.csv", index=False)
print("Built CSV windows for Evidently with fixed one-hot columns.")
