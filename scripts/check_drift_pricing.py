\
import pandas as pd, json, os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from pathlib import Path

Path("artifacts/drift").mkdir(parents=True, exist_ok=True)

def data_drift():
    ref = pd.read_csv("artifacts/data/ref_features.csv")
    cur = pd.read_csv("artifacts/data/cur_features.csv")
    rep = Report(metrics=[DataDriftPreset()])
    rep.run(reference_data=ref, current_data=cur)
    dd = rep.as_dict()
    with open("artifacts/drift/data_drift.json","w") as f: json.dump(dd,f)
    return dd["metrics"][0]["result"]["dataset_drift"]

def perf_drift():
    ref = pd.read_csv("artifacts/data/ref_eval.csv")
    cur = pd.read_csv("artifacts/data/cur_eval.csv")
    rep = Report(metrics=[RegressionPreset()])
    rep.run(reference_data=ref, current_data=cur)
    rr = rep.as_dict()
    with open("artifacts/drift/perf_drift.json","w") as f: json.dump(rr,f)
    cur_mape = rr["metrics"][0]["result"]["current"]["mean_abs_perc_error"]
    return cur_mape > 0.20

flag = False
try:
    if data_drift() or perf_drift():
        Path("artifacts/drift/RETRAIN.flag").write_text("1")
        flag = True
except Exception as e:
    print("Drift check error:", e)

print("RETRAIN:", flag)
