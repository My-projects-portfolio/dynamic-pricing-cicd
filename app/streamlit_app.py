\
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import streamlit as st, pandas as pd, joblib, json, os, time
from pathlib import Path
from loguru import logger
from scripts.optimize_price import optimal_price

Path("logs").mkdir(exist_ok=True)
logger.add("logs/pricing.log", rotation="10 MB", retention="14 days", enqueue=True)

st.set_page_config(page_title="Dynamic Pricing — UCI Online Retail", layout="wide")
st.title("Dynamic Pricing — UCI Online Retail")

bundle_path = "artifacts/models/demand-latest.joblib"
if not os.path.exists(bundle_path):
    st.warning("Model bundle not found. Train it first (CI will also handle retraining).")
else:
    bundle = joblib.load(bundle_path)
    meta = bundle["meta"]
    c1,c2,c3 = st.columns(3)
    c1.metric("MAPE", f"{meta.get('mape',0):.2%}")
    c2.metric("#Obs", meta.get("n_obs", 0))
    c3.write(f"Deployed (UTC): {meta['timestamp']}")

st.subheader("Revenue trend")
if os.path.exists("artifacts/metrics/daily_revenue.csv"):
    rev = pd.read_csv("artifacts/metrics/daily_revenue.csv", parse_dates=["date"])
    st.line_chart(rev.set_index("date")["revenue"])
else:
    st.info("metrics missing. Run: python scripts/make_daily_metrics.py --csv data/processed/sample_sales.csv")

st.subheader("Price Optimizer (demo)")
weekday = st.selectbox("Weekday", list(range(7)), index=4)
month = st.selectbox("Month", list(range(1,13)), index=11)
promo = st.checkbox("Promo?", value=False)
stock = st.number_input("Stock", min_value=0, value=1000, step=10)
pmin = st.number_input("Price min", min_value=0.1, value=1.0, step=0.1)
pmax = st.number_input("Price max", min_value=0.1, value=20.0, step=0.1)
step = st.number_input("Grid step", min_value=0.05, value=0.5, step=0.05)

context = {"promo_flag": int(promo), "stock": stock, "weekday": weekday, "month": month}
if st.button("Recommend price"):
    if not os.path.exists(bundle_path):
        st.error("No model bundle found.")
    else:
        t0 = time.time()
        best_p, best_rev = optimal_price(bundle, context, pmin, pmax, step=step)
        ms = int((time.time()-t0)*1000)
        st.success(f"Recommended price: {best_p:.2f} | Expected revenue: {best_rev:.2f}")
        logger.info("rec | context={} pmin={} pmax={} best_p={} best_rev={} model={} latency_ms={}",
                    context, pmin, pmax, best_p, best_rev, bundle['meta'].get('timestamp','n/a'), ms)

st.subheader("Drift summaries")
cols = st.columns(2)
for path, col, title in [
    ("artifacts/drift/data_drift.json", cols[0], "Data Drift (Evidently)"),
    ("artifacts/drift/perf_drift.json", cols[1], "Performance Drift (Evidently)"),
]:
    with col:
        st.markdown(f"**{title}**")
        if os.path.exists(path):
            with open(path) as f: d = json.load(f)
            try:
                if "dataset_drift" in d["metrics"][0]["result"]:
                    st.write("dataset_drift:", d["metrics"][0]["result"]["dataset_drift"])
                elif "current" in d["metrics"][0]["result"]:
                    st.write("MAPE (current):", d["metrics"][0]["result"]["current"]["mean_abs_perc_error"])
            except Exception as e:
                st.write("Could not parse summary:", e)
        else:
            st.caption("No drift files found yet.")
