# UCI SECOM — Semiconductor Manufacturing Failure Prediction

End-to-end, leakage-aware classification of pass/fail outcomes in the [UCI SECOM dataset](https://archive.ics.uci.edu/dataset/179/secom), also published on [Kaggle](https://www.kaggle.com/datasets/paresh2047/uci-semcom/data).

**Status:** Implemented training/evaluation pipeline with synthetic-data tests. No genuine SECOM model score is claimed until the actual dataset is downloaded and training is run.

## Data
1,567 manufacturing samples, ~590 sensor measurements, 104 failures (~6.6%). Accuracy alone is misleading for this imbalanced dataset, so the project reports average precision, recall, F1, balanced accuracy, ROC AUC, MCC, and confusion matrices.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Download
Use `python scripts/download_data.py` or place `secom.data` and `secom_labels.data` from UCI/Kaggle in `data/raw/`.

## Train and evaluate
```bash
python -m secom.train --data-dir data/raw --output-dir results --split chronological
```

Chronological validation is the default; use `--split stratified` for an alternative split. Model selection is by training-fold average precision; threshold tuning uses out-of-fold F1 rather than held-out test labels.

## Synthetic smoke tests
```bash
python scripts/make_synthetic_data.py
python -m secom.train --data-dir data/synthetic --output-dir results_synthetic --split stratified
pytest -q
```

## Models and outputs
Logistic regression and random forest pipelines with median imputation and appropriate imbalance handling. Outputs: `metrics.json`, `test_predictions.csv`, `model.joblib`, precision-recall curve, confusion matrix, and feature importance when available.

**Limitations:** Small minority class; uncertain generalization; synthetic results do not constitute a real-data benchmark. Features are anonymized and their associations are not proven root causes.

## Attribution
McCann, M. & Johnston, A. (2008). *SECOM*. UCI Machine Learning Repository, DOI: 10.24432/C54305. Dataset: CC BY 4.0. Code: MIT.
