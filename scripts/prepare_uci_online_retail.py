\
import argparse, pandas as pd, numpy as np
from pathlib import Path

"""
Prepare UCI Online Retail (Excel) into a product/day dataset for dynamic pricing.
Input file should be the original "Online Retail.xlsx".
Output CSV columns: date, product_id, price, units, promo_flag, stock, weekday, month
"""

def main(in_xlsx, out_csv):
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    # Load
    df = pd.read_excel(in_xlsx, sheet_name="Online Retail")
    # Basic clean
    df = df.dropna(subset=["InvoiceNo","StockCode","InvoiceDate","Quantity","UnitPrice"])
    # Exclude credit notes (InvoiceNo starting with 'C') and negative quantities/prices
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    # Date only
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])
    df["date"] = df["InvoiceDate"].dt.date

    # Aggregate per product/day
    grp = df.groupby(["date","StockCode"], as_index=False).agg(
        units=("Quantity","sum"),
        price=("UnitPrice","median")
    )
    grp = grp.rename(columns={"StockCode":"product_id"})
    # Derive promo_flag: daily price < 90% of product median (rough proxy for discount)
    med = grp.groupby("product_id")["price"].transform("median")
    grp["promo_flag"] = (grp["price"] < 0.9 * med).astype(int)
    # Add stock placeholder + calendar
    grp["stock"] = 1000
    dts = pd.to_datetime(grp["date"])
    grp["weekday"] = dts.dt.weekday
    grp["month"] = dts.dt.month

    grp = grp[["date","product_id","price","units","promo_flag","stock","weekday","month"]].copy()
    grp.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(grp)} rows.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in_xlsx", required=True)
    p.add_argument("--out_csv", default="data/processed/sample_sales.csv")
    a = p.parse_args()
    main(a.in_xlsx, a.out_csv)
