# Real-data evaluation — UCI SECOM

**Evaluated:** 2026-10-16 (GitHub Actions, Python 3.11).
**Source:** UCI SECOM official original downloadable archive: `https://archive.ics.uci.edu/static/public/179/secom.zip`.
**Verifiable run:** https://github.com/Nishant99966/uci-secom-semiconductor-quality/actions/runs/38954915322

## Protocol
- 1,567 original labeled manufacturing samples; 104 defective (1 = failure).
- Chronological 80/20 split by source timestamp: 1,253 training records (87 failures); 314 held-out records (17 failures).
- Models: median-imputed, variance-filtered, class-balanced Logistic Regression and Random Forest.
- Model chosen by four-fold stratified CV average precision within the training segment.
- Binary classification threshold chosen using out-of-fold training F1; test labels not used for selection.
- Final fit on the whole training segment; evaluate once on chronological holdout.

## Measured results

| Metric | Value |
| --- | ---: |
| Chosen model | Random Forest |
| Logistic Regression 4-fold CV average precision | 0.123118 |
| Random Forest 4-fold CV average precision | 0.243528 |
| Out-of-fold training F1 at selected threshold | 0.288136 |
| Selected failure-probability threshold | 0.161173 |
| Held-out average precision | 0.096639 |
| Held-out ROC AUC | 0.556942 |
| Held-out precision | 0.000000 |
| Held-out recall | 0.000000 |
| Held-out F1 | 0.000000 |
| Held-out MCC | -0.023498 |
| Held-out balanced accuracy | 0.494949 |

Confusion matrix (rows actual pass/failure; columns predicted pass/failure):

| | Predicted pass | Predicted failure |
| --- | ---: | ---: |
| Actual pass | 294 | 3 |
| Actual failure | 17 | 0 |

**Interpretation:** The selected model detected **zero** of the 17 genuine failures in the held-out chronological data. Despite higher training-fold average precision, it is **not suitable for deployment**. Strong apparent class-wise performance from a random split would not establish chronological generalization.

## Next steps
- Inspect dataset shift, timestamp grouping, and prior/posterior defect rates.
- Quantify run-to-run variability with repeated **blocked-time validation** rather than a single random split.
- Calibrate predicted probabilities and optimize costs/recall subject to tolerable false-positive rate, using only training folds.
- Benchmark alternate feature selection, anomaly detection, balanced ensembles and boosted trees.
- Preserve time-ordered held-out data until all modeling decisions are finalized.

Metrics above are from the real dataset, **not synthetic tests**. See the linked GitHub Actions run for provenance and downloadable evaluation artifacts.
