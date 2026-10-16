"""Pre-specified sensitivity analysis on the untouched chronological SECOM holdout.

Compare Logistic Regression and Random Forest. The decision thresholds are derived
exclusively from training out-of-fold predictions to target >=70% failure recall.
Both candidates are reported; no model is cherry-picked by held-out performance.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, roc_auc_score, confusion_matrix,
                             f1_score, precision_score, recall_score, matthews_corrcoef)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from .data import load_secom
from .train import candidates


def recall_threshold(y, probabilities, target=0.70):
    """Largest observed threshold obtaining at least target sensitivity in OOF predictions."""
    positives = np.asarray(probabilities)[np.asarray(y) == 1]
    if not len(positives):
        raise ValueError("At least one positive is required")
    candidates_ = np.unique(np.r_[0., np.asarray(probabilities), 1.])
    suitable = [float(t) for t in candidates_ if np.mean(positives >= t) >= target]
    return max(suitable)


def evaluate(data_dir="data/raw", output_dir="results_recall", target_recall=0.70):
    x, y, timestamps = load_secom(data_dir)
    order = np.argsort(timestamps.to_numpy(), kind="stable")
    cut = int(len(y) * .8)
    tr, te = order[:cut], order[cut:]
    xt, xv = x.iloc[tr], x.iloc[te]
    yt, yv = y.iloc[tr], y.iloc[te]
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    report = {
        "methodology": "pre-specified two-model comparison, thresholds set on train OOF",
        "target_training_oof_recall": target_recall,
        "train_rows": int(len(tr)), "test_rows": int(len(te)),
        "train_failures": int(yt.sum()), "test_failures": int(yv.sum()),
        "models": {}
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, model in candidates().items():
        oof = cross_val_predict(model, xt, yt, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
        threshold = recall_threshold(yt, oof, target_recall)
        oof_pred = (oof >= threshold).astype(int)
        model.fit(xt, yt)
        prob = model.predict_proba(xv)[:, 1]
        pred = (prob >= threshold).astype(int)
        report["models"][name] = {
            "train_oof_average_precision": float(average_precision_score(yt, oof)),
            "train_oof_recall": float(recall_score(yt, oof_pred, zero_division=0)),
            "train_oof_precision": float(precision_score(yt, oof_pred, zero_division=0)),
            "threshold": float(threshold),
            "heldout_average_precision": float(average_precision_score(yv, prob)),
            "heldout_roc_auc": float(roc_auc_score(yv, prob)),
            "heldout_recall": float(recall_score(yv, pred, zero_division=0)),
            "heldout_precision": float(precision_score(yv, pred, zero_division=0)),
            "heldout_f1": float(f1_score(yv, pred, zero_division=0)),
            "heldout_mcc": float(matthews_corrcoef(yv, pred)),
            "heldout_confusion_matrix": confusion_matrix(yv, pred, labels=[0, 1]).tolist(),
            "heldout_predicted_failures": int(pred.sum())
        }
        pd.DataFrame({"row": te, "truth": yv.to_numpy(), "probability": prob,
                      "prediction": pred}).to_csv(output / f"{name}_predictions.csv", index=False)
    (output / "recall_experiment_metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="results_recall")
    parser.add_argument("--target-recall", type=float, default=.70)
    args = parser.parse_args()
    evaluate(args.data_dir, args.output_dir, args.target_recall)


if __name__ == "__main__":
    main()
