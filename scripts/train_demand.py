\
import pandas as pd, joblib, argparse, numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from datetime import datetime
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument("--csv", required=True)
p.add_argument("--out", default="artifacts/models/demand-latest.joblib")
a=p.parse_args()

df = pd.read_csv(a.csv, parse_dates=["date"])
df = df.dropna(subset=["price","units"])

# Feature set
df["weekday"] = df["weekday"].astype(int)
df["month"] = df["month"].astype(int)
X = df[["price","promo_flag","stock","weekday","month"]].copy()
X = pd.get_dummies(X, columns=["weekday","month"], drop_first=True)
y = df["units"]

Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
model = GradientBoostingRegressor(random_state=42).fit(Xtr,ytr)
yhat = model.predict(Xte)
mae = mean_absolute_error(yte, yhat)
mape = mean_absolute_percentage_error(yte.clip(lower=1), np.maximum(yhat, 0.001))

bundle = {
    "model": model,
    "features": X.columns.tolist(),
    "meta": {"timestamp": datetime.utcnow().isoformat(), "mae": float(mae), "mape": float(mape), "n_obs": int(len(df))}
}
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
joblib.dump(bundle, a.out)
print("Saved:", a.out, "| MAE:", mae, "| MAPE:", mape)
