# Fixed 25% Inspection Budget — Stability-Filtered SECOM Evaluation

**Run:** https://github.com/Nishant99966/uci-secom-semiconductor-quality/actions/runs/38959534122  
**Dataset:** Official UCI SECOM, 1,567 parts, 590 anonymized sensors, 104 failures.

## Protocol

The last 20% of the time-sorted data was excluded from this experiment. Four expanding-window folds covered 1,000 validation observations from the earlier 80%. Stable sensors were selected independently within each fold's **training-only** data, based on missingness and first-half vs second-half mean-shift criteria. The selected columns were then used to train balanced Logistic Regression and Random Forest pipelines.

For each 250-row validation segment, the model flagged the top `floor(0.25 * 250) = 62` highest-risk observations. This is a batch risk-ranking simulation, **not** a deployable prospective threshold, because it assumes the full batch's score distribution is available when deciding which parts to inspect. Selection did not use validation labels.

## Verified real-data results

| Metric | Stable Logistic Regression | Stable Random Forest |
|---|---:|---:|
| Total validation parts | 1,000 | 1,000 |
| Parts flagged | 248 | 248 |
| True defective parts | 48 | 48 |
| True defects detected | 13 | 18 |
| Missed defects | 35 | 30 |
| False-positive alarms | 235 | 230 |
| Pooled failure recall | 27.08% | 37.50% |
| Precision among flagged | 5.24% | 7.26% |
| Mean fold average precision | 7.26% | 11.79% |
| Mean fold recall (unweighted) | 25.61% | 38.67% |
| Mean number of retained stable features across folds | 420.75 | 420.75 |

### Per-period defects detected

| Fold | Actual failures | Logistic Regression | Random Forest |
|---|---:|---:|---:|
| 1 | 20 | 7 | 7 |
| 2 | 8 | 2 | 4 |
| 3 | 9 | 3 | 3 |
| 4 | 11 | 1 | 4 |

## Interpretation

At equal inspection effort, the stability-filtered Random Forest detected five more true defects. Both models still have substantial miss rates and low precision, so neither should be used as a production quality-release decision system. There is **no directly matched unfiltered-model experiment at the same 25% budget**, so these results do not establish whether sensor stability filtering helped. The four sequential validation folds also share overlapping training histories.

Recommended next experiment: compare stable-filtered vs identical unfiltered models under the same forward-time folds and 25% inspection capacity, then compute bootstrap uncertainty at the period level where possible. The previously examined last 20% should not be described as a pristine prospective final holdout.

## Reproduce

```bash
python -m secom.stability_budget --data-dir data/raw --output-dir results_stability --budget 0.25
```

Raw dataset and credentials are not committed. GitHub Actions uploads `stability_budget_metrics.json` and `stability_budget_predictions.csv` as artifacts.
