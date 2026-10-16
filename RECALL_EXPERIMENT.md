# Recall-targeted SECOM real-data experiment

**Date:** 2026-10-16. **Source:** Official UCI SECOM, 1,567 samples and 104 failures.
**Reproducible workflow:** https://github.com/Nishant99966/uci-secom-semiconductor-quality/actions/runs/38956067492

## Fixed protocol

- 80% chronological training (1,253 rows, 87 failures); 20% chronological holdout (314 rows, 17 failures).
- Same preprocessing and classifiers as initial pipeline: median imputation, variance filtering, balanced Logistic Regression and Random Forest.
- Stratified four-fold training-only out-of-fold predictions for both candidate models.
- **Pre-specified threshold criterion:** largest threshold with >=70% failure recall on training OOF predictions.
- Apply each selected threshold unchanged on its corresponding fitted model's held-out predictions.
- Report both models without choosing a winner on test labels.

## Actual chronological holdout results

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Training OOF AP | 0.110636 | 0.199043 |
| Training OOF recall | 0.701149 | 0.701149 |
| Threshold | 0.0000525532 | 0.1052646695 |
| Held-out AP | 0.146486 | 0.096639 |
| Held-out ROC AUC | 0.773024 | 0.556942 |
| Held-out recall | 0.764706 | 0.235294 |
| Held-out precision | 0.110169 | 0.080000 |
| Held-out F1 | 0.192593 | 0.119403 |
| Held-out MCC | 0.192113 | 0.049732 |
| True negatives | 192 | 251 |
| False positives | 105 | 46 |
| False negatives | 4 | 13 |
| True positives | 13 | 4 |

Logistic Regression detects 13/17 actual failures but flags 105 passing records; Random Forest detects 4/17 and flags 46 passing records. Thresholds optimize *training* sensitivity only; they do not guarantee holdout sensitivity.

## Interpretation and cautions

The previous F1-optimized Random Forest missed all 17 holdout failures. The recall-targeted decision rule increases detection but creates false alerts. Accuracy is an inadequate success criterion in this roughly 6.6%-failure dataset. The unchanged held-out data has now been examined for multiple comparisons, so using it repeatedly to choose a final method would invite overfitting. This study is **exploratory** and **not production-ready**. A future, independent production-time sample or strictly pre-registered blocked validation is needed before deployment claims.

### Reproduce

```bash
python scripts/download_data.py
python -m secom.recall_experiment --data-dir data/raw --output-dir results_recall --target-recall 0.70
```

The GitHub Actions workflow downloads the official UCI archive and attaches prediction CSVs and the evaluation JSON. Raw manufacturing data are not redistributed in the repository.
