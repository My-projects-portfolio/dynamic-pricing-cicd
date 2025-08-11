# Dynamic Pricing (Retail) — UCI Online Retail (Cloud-First)

This repo implements a **dynamic pricing system** for retail using the **UCI Online Retail** dataset. It trains a demand model, recommends prices that maximize expected revenue, and exposes a **Streamlit Cloud** dashboard. **EvidentlyAI** is used in CI to detect **data/performance drift** and trigger **automated retraining**.

## Quickstart (Local)
```bash
python -m venv .venv
# Windows: .\.venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# 1) Drop the UCI file at: data/raw/Online Retail.xlsx
#    (UCI Machine Learning Repository: "Online Retail" dataset)
# 2) Prepare training data from UCI invoices
python scripts/prepare_uci_online_retail.py --in_xlsx "data/raw/Online Retail.xlsx" --out_csv data/processed/sample_sales.csv

# 3) Train demand model
python scripts/train_demand.py --csv data/processed/sample_sales.csv --out artifacts/models/demand-latest.joblib

# 4) Build metrics + drift windows
python scripts/make_daily_metrics.py --csv data/processed/sample_sales.csv
python scripts/build_eval_windows.py --csv data/processed/sample_sales.csv
python scripts/check_drift_pricing.py

# 5) Run dashboard
streamlit run app/streamlit_app.py
```

## Streamlit Cloud (Deploy)
1. Push this repo to GitHub.
2. On Streamlit Cloud, create an app from the repo.
3. Main file: `app/streamlit_app.py`. Python version: 3.11. Add `requirements.txt`.
4. Deploy. The app reads artifacts from the repo and shows metrics.
5. The included **GitHub Actions** workflow runs nightly to check drift and retrain as needed.

## Repo Layout
```
.
├─ app/streamlit_app.py              # Monitoring dashboard (MAPE/MAE, revenue, drift, price tool)
├─ scripts/
│  ├─ prepare_uci_online_retail.py   # Convert UCI invoices -> product/day features for pricing
│  ├─ train_demand.py                # Train demand regression model
│  ├─ optimize_price.py              # Grid-search revenue-maximizing price under bounds
│  ├─ make_daily_metrics.py          # Build daily revenue series for dashboard
│  ├─ build_eval_windows.py          # Make reference/current eval windows for drift
│  └─ check_drift_pricing.py         # Evidently drift checks; writes JSON & retrain flag
├─ data/raw/                          # Put "Online Retail.xlsx" here
├─ data/processed/                    # Generated product/day CSV
├─ artifacts/                         # model/metrics/drift artifacts (committed by CI)
├─ .github/workflows/pricing_cron.yml # Nightly drift + retrain + commit
└─ requirements.txt
```

## Notes
- We exclude refunds/credit notes and negative quantities when preparing data.
- `promo_flag` is inferred per product/day if daily median price < 90% of product median.
- If you lack stock data, a constant stock placeholder is used.
- No PII is logged; keep it that way.
