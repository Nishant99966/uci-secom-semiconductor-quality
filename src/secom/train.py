import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, ConfusionMatrixDisplay)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

from .data import load_secom
from .metrics import best_f1_threshold


def candidates():
    return {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True, add_indicator=True)),
            ("variance", VarianceThreshold(threshold=0)),
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True, add_indicator=True)),
            ("variance", VarianceThreshold(threshold=0)),
            ("classifier", RandomForestClassifier(n_estimators=250, min_samples_leaf=3,
                                                  class_weight="balanced_subsample", random_state=42,
                                                  n_jobs=-1)),
        ]),
    }


def run(data_dir="data/raw", output_dir="results", split="chronological", test_size=.2, cv_folds=4):
    x, y, timestamps = load_secom(data_dir)
    if split == "chronological":
        order = np.argsort(timestamps.to_numpy(), kind="stable")
        cut = int(len(y) * (1 - test_size))
        tr_idx, te_idx = order[:cut], order[cut:]
    elif split == "stratified":
        tr_idx, te_idx = train_test_split(np.arange(len(y)), test_size=test_size,
                                         stratify=y, random_state=42)
    else:
        raise ValueError("split must be chronological or stratified")
    x_tr, x_te = x.iloc[tr_idx], x.iloc[te_idx]
    y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
    if int(y_tr.sum()) < cv_folds or int((1-y_tr).sum()) < cv_folds:
        raise ValueError("Not enough samples per class for requested CV folds")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    models = candidates()
    comparison = {name: float(np.mean(cross_val_score(model, x_tr, y_tr,
                     scoring="average_precision", cv=cv, n_jobs=1)))
                  for name, model in models.items()}
    selected = max(comparison, key=comparison.get)
    estimator = models[selected]
    oof = cross_val_predict(estimator, x_tr, y_tr, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    threshold, oof_f1 = best_f1_threshold(y_tr.to_numpy(), oof)
    estimator.fit(x_tr, y_tr)
    probabilities = estimator.predict_proba(x_te)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    metrics = {
        "model": selected, "cross_validation_average_precision": comparison,
        "train_oof_optimal_threshold": threshold, "train_oof_f1": oof_f1,
        "test_average_precision": float(average_precision_score(y_te, probabilities)),
        "test_roc_auc": float(roc_auc_score(y_te, probabilities)) if y_te.nunique() > 1 else None,
        "test_f1": float(f1_score(y_te, predictions, zero_division=0)),
        "test_recall": float(recall_score(y_te, predictions, zero_division=0)),
        "test_precision": float(precision_score(y_te, predictions, zero_division=0)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_te, predictions)),
        "test_mcc": float(matthews_corrcoef(y_te, predictions)),
        "test_confusion_matrix": confusion_matrix(y_te, predictions, labels=[0, 1]).tolist(),
        "train_rows": int(len(tr_idx)), "test_rows": int(len(te_idx)),
        "train_failures": int(y_tr.sum()), "test_failures": int(y_te.sum()),
        "split": split,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame({"original_row_index": te_idx, "timestamp": timestamps.iloc[te_idx].astype(str).values,
                  "actual_failure": y_te.to_numpy(), "predicted_failure": predictions,
                  "failure_probability": probabilities}).to_csv(output / "test_predictions.csv", index=False)
    joblib.dump({"pipeline": estimator, "threshold": threshold, "feature_names": list(x.columns)},
                output / "model.joblib")
    classifier = estimator.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        names = estimator.named_steps["imputer"].get_feature_names_out(x.columns)
        names = estimator.named_steps["variance"].get_feature_names_out(names)
        pd.DataFrame({"feature": names, "importance": classifier.feature_importances_}).sort_values(
            "importance", ascending=False).to_csv(output / "feature_importances.csv", index=False)
    ConfusionMatrixDisplay.from_predictions(y_te, predictions, labels=[0, 1])
    plt.tight_layout(); plt.savefig(output / "confusion_matrix.png", dpi=140); plt.close()
    precision, recall, _ = precision_recall_curve(y_te, probabilities)
    plt.plot(recall, precision)
    plt.xlabel("Recall (failure class)"); plt.ylabel("Precision (failure class)")
    plt.title("SECOM: held-out precision-recall curve"); plt.tight_layout()
    plt.savefig(output / "precision_recall_curve.png", dpi=140); plt.close()
    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--split", choices=["chronological", "stratified"], default="chronological")
    parser.add_argument("--test-size", type=float, default=.2)
    parser.add_argument("--cv-folds", type=int, default=4)
    args = parser.parse_args()
    run(**vars(args))

if __name__ == "__main__":
    main()
