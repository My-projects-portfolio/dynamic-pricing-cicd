\
import numpy as np, joblib, pandas as pd

def optimal_price(bundle, context_row, price_min, price_max, step=0.5):
    model, feats = bundle["model"], bundle["features"]
    grid = np.arange(price_min, price_max+1e-9, step)
    best_p, best_rev = None, -1
    for p in grid:
        x = context_row.copy()
        x["price"] = p
        x = pd.Series(x).reindex(feats, fill_value=0).to_frame().T
        demand_hat = max(0.0, float(model.predict(x)[0]))
        rev = p * demand_hat
        if rev > best_rev:
            best_rev, best_p = rev, p
    return best_p, best_rev
