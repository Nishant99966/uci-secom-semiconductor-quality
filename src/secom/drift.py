"""Exploratory sensor drift audit using original UCI SECOM observations.

Drift is descriptive, not proof of root cause or predictive quality. It must
never use outcome labels to pick features for a claimed untouched holdout score.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .data import load_secom


def run(data_dir="data/raw", output_dir="results_drift", periods=5):
    x, y, timestamp = load_secom(data_dir)
    order = np.argsort(timestamp.to_numpy(), kind="stable")
    x = x.iloc[order].reset_index(drop=True)
    y = y.iloc[order].reset_index(drop=True)
    timestamp = timestamp.iloc[order].reset_index(drop=True)
    group = np.minimum(np.arange(len(x)) * periods // len(x), periods - 1)
    summary = []
    for period in range(periods):
        ids = group == period
        summary.append({"period": period + 1, "rows": int(ids.sum()),
                        "failures": int(y[ids].sum()), "failure_rate": float(y[ids].mean()),
                        "mean_missing_fraction": float(x[ids].isna().to_numpy().mean()),
                        "time_min": timestamp[ids].min().isoformat(),
                        "time_max": timestamp[ids].max().isoformat()})
    # Compare the first two time segments against the last two. Keep this
    # explicitly exploratory: final holdout included in drift description.
    first, last = x[group < 2], x[group >= periods - 2]
    rows = []
    for col in x.columns:
        a, b = first[col], last[col]
        ma, mb = float(a.isna().mean()), float(b.isna().mean())
        avg_a, avg_b = float(a.mean()), float(b.mean())
        var_a, var_b = float(a.var()), float(b.var())
        pooled = np.sqrt(max((np.nan_to_num(var_a) + np.nan_to_num(var_b))/2., 0.))
        if np.isfinite(avg_a) and np.isfinite(avg_b):
            standardized_shift = abs(avg_b-avg_a)/(pooled+1e-9)
        else:
            standardized_shift = None
        rows.append({"feature":col,"early_missing":ma,"late_missing":mb,
                     "absolute_missingness_shift":abs(mb-ma),
                     "early_mean":avg_a if np.isfinite(avg_a) else None,
                     "late_mean":avg_b if np.isfinite(avg_b) else None,
                     "standardized_mean_shift":standardized_shift})
    drift = pd.DataFrame(rows)
    drift["drift_score"] = drift["absolute_missingness_shift"] + drift["standardized_mean_shift"].fillna(0)
    drift = drift.sort_values("drift_score",ascending=False)
    out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    drift.to_csv(out/"sensor_drift.csv",index=False)
    pd.DataFrame(summary).to_csv(out/"period_summary.csv",index=False)
    report={"data_source":"original UCI SECOM", "rows":len(x),"sensors":x.shape[1],
            "failures":int(y.sum()),"periods":summary,
            "top_drifting_sensors":drift.head(12).where(pd.notna(drift.head(12)),None).to_dict("records"),
            "caution":"Exploratory full-dataset drift description; not a prospective holdout assessment."}
    (out/"drift_report.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(report,indent=2))
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir",default="data/raw")
    p.add_argument("--output-dir",default="results_drift")
    a=p.parse_args()
    run(a.data_dir,a.output_dir)
