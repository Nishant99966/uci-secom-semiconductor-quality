# Real-data chronological blocked validation (SECOM)

**Verified GitHub Actions run:** https://github.com/Nishant99966/uci-secom-semiconductor-quality/actions/runs/38957145560

## Protocol

The original 1,567 UCI SECOM samples were sorted chronologically. The last 20% (314 samples) was excluded from this experiment. The first 80% (1,253 samples, 87 failures) was evaluated using four expanding-window time folds, each with 250 future-period validation observations. Preprocessing and models were refitted independently on each earlier period. Threshold candidates were evaluated on pooled out-of-fold predictions for exploratory operating-point analysis.

Target: **at least 70% failure recall with no more than 25% of samples flagged for inspection**.

## Measured results

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Pooled out-of-fold average precision | 0.063773 | 0.100414 |
| Fold 1 ROC AUC | 0.517609 | 0.520217 |
| Fold 2 ROC AUC | 0.507748 | 0.583161 |
| Fold 3 ROC AUC | 0.712771 | 0.662056 |
| Fold 4 ROC AUC | 0.381514 | 0.481552 |
| 70%-recall / 25%-alert constraint | **Not feasible** | **Not feasible** |

Exploratory fallback operating points (NOT recommended for production):

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Pooled recall | 79.17% | 95.83% |
| Pooled precision | 5.16% | 5.16% |
| Fraction of parts flagged | 73.6% | 89.1% |
| False-positive alerts in 1,000 pooled validation rows | 698 | 845 |
| MCC | 0.0284 | 0.0485 |

These fallback thresholds **violate** the 25% inspection-budget constraint. They illustrate how badly the recall-focused models trade off inspection volume against sensitivity, rather than representing useful selected operating points.

## Conclusion

A promising 76.5% recall on the previously inspected final holdout did **not** establish stable temporal generalization. Across strictly forward-looking folds, neither candidate achieves the desired quality-control operating point. The apparent quality varies sharply between production periods, with end-of-training fold ROC AUC below 0.5 for both models. A highly imbalanced process, possible manufacturing drift, noisy sensors, and low defect sample counts make this problem difficult.

**No production deployment or reliable defect-reduction claim is justified.**

## Follow-up research

1. Diagnose distribution shift: compare per-period target prevalence, missingness patterns, sensor summary statistics, and model score distributions.
2. Explore sensor feature selection within training folds, robust scaling, gradient-boosted models and calibrated anomaly detection.
3. Report recall **at a fixed false-positive/inspection budget** and per-fold uncertainty rather than selecting sensitivity on the same validation data used for reporting.
4. Once a procedure is pre-registered based exclusively on chronological training folds, obtain a new independent future-period holdout for a final confirmation. The original last 20% has already been viewed repeatedly, so it should not be treated as pristine.

## Reproduce

```bash
python -m secom.blocked_validation --data-dir data/raw --output-dir results_blocked --n-splits 4 --target-recall 0.70 --max-alert-rate 0.25
```

The corresponding GitHub Actions artifact includes detailed metrics and each blocked validation prediction. No proprietary manufacturing data are stored in this repository.
