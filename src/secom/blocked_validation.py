"""Chronologically blocked, training-only model evaluation.

Each fold trains on earlier periods and validates on a later period. The final
20% remains a locked holdout and is NOT read for this experiment.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, matthews_corrcoef
from sklearn.model_selection import TimeSeriesSplit
from .data import load_secom
from .train import candidates

def run(data_dir="data/raw", output_dir="results_blocked", n_splits=4,
        target_recall=0.70, max_alert_rate=0.25):
    x, y, dates = load_secom(data_dir)
    order = np.argsort(dates.to_numpy(), kind="stable")
    cut = int(len(order) * .8)
    training = order[:cut]
    x, y = x.iloc[training].reset_index(drop=True), y.iloc[training].reset_index(drop=True)
    folds = TimeSeriesSplit(n_splits=n_splits)
    report = {"protocol": "expanding-window chronological validation on first 80%; no final holdout access",
              "training_rows": len(y), "training_failures": int(y.sum()),
              "folds": n_splits, "target_recall": target_recall,
              "max_alert_rate": max_alert_rate, "models": {}}
    all_predictions = []
    for name, model in candidates().items():
        measurements = []
        for fold, (train_idx, valid_idx) in enumerate(folds.split(x), 1):
            yt, yv = y.iloc[train_idx], y.iloc[valid_idx]
            if yt.nunique() < 2:
                raise ValueError(f"Fold {fold} training data has one class only")
            fitted = clone(model).fit(x.iloc[train_idx], yt)
            prob = fitted.predict_proba(x.iloc[valid_idx])[:, 1]
            for idx, actual, score in zip(valid_idx, yv, prob):
                all_predictions.append({"model": name, "fold": fold,
                                        "row_in_chronological_training": int(idx),
                                        "actual": int(actual), "probability": float(score)})
            measurements.append({"fold": fold, "train_rows": len(train_idx),
                                 "validation_rows": len(valid_idx),
                                 "train_failures": int(yt.sum()),
                                 "validation_failures": int(yv.sum()),
                                 "average_precision": float(average_precision_score(yv, prob)),
                                 "roc_auc": float(roc_auc_score(yv, prob)) if yv.nunique() == 2 else None})
        preds = pd.DataFrame(p for p in all_predictions if p["model"] == name)
        pos = preds[preds["actual"] == 1]["probability"].to_numpy()
        thresholds = np.unique(np.r_[0., preds["probability"].to_numpy(), 1.])
        operating = []
        for t in thresholds:
            pred = (preds["probability"].to_numpy() >= t).astype(int)
            operating.append({"threshold": float(t),
                              "recall": float(recall_score(preds["actual"], pred, zero_division=0)),
                              "precision": float(precision_score(preds["actual"], pred, zero_division=0)),
                              "alert_rate": float(pred.mean()),
                              "mcc": float(matthews_corrcoef(preds["actual"], pred)),
                              "false_positives": int(((pred == 1) & (preds["actual"].to_numpy() == 0)).sum())})
        feasible = [v for v in operating if v["recall"] >= target_recall and
                    v["alert_rate"] <= max_alert_rate]
        if feasible:
            selected = max(feasible, key=lambda v: (v["precision"], -v["alert_rate"]))
            status = "feasible_on_training_only_blocked_cv"
        else:
            selected = max(operating, key=lambda v: (v["recall"] >= target_recall,
                                                    v["precision"], -v["alert_rate"]))
            status = "constraints_not_simultaneously_met"
        report["models"][name] = {"fold_results": measurements,
                                  "pooled_blocked_cv_average_precision":
                                      float(average_precision_score(preds["actual"], preds["probability"])),
                                  "operating_point": selected, "constraint_status": status}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_predictions).to_csv(out / "blocked_cv_predictions.csv", index=False)
    (out / "blocked_cv_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--output-dir", default="results_blocked")
    p.add_argument("--n-splits", type=int, default=4)
    p.add_argument("--target-recall", type=float, default=.70)
    p.add_argument("--max-alert-rate", type=float, default=.25)
    a = p.parse_args()
    run(a.data_dir, a.output_dir, a.n_splits, a.target_recall, a.max_alert_rate)
