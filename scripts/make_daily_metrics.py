\
import argparse, pandas as pd
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument("--csv", required=True)
a=p.parse_args()

df = pd.read_csv(a.csv, parse_dates=["date"])
df["revenue"] = df["price"] * df["units"]
daily = df.groupby("date", as_index=False)["revenue"].sum()
Path("artifacts/metrics").mkdir(parents=True, exist_ok=True)
daily.to_csv("artifacts/metrics/daily_revenue.csv", index=False)
print("Saved artifacts/metrics/daily_revenue.csv")
