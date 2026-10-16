# SECOM paired sensor-stability ablation — verified real-data experiment

**Reproducible workflow:** https://github.com/Nishant99966/uci-secom-semiconductor-quality/actions/runs/38960916945

## Protocol

Original public UCI SECOM data. Chronological ordering. The last 20% is excluded from this experiment. Four identical forward-chaining folds each validate on 250 future observations within the earlier 80%. Logistic Regression and Random Forest are tested with all 590 sensors and with sensors filtered by training-only stability criteria. In each 250-row validation batch, only the 62 highest-risk samples are flagged, for a fixed 24.8% inspection workload. Validation labels are not used to determine which parts to inspect.

## Real-data results (1,000 forward-time validation observations; 48 real defects)

| Pipeline | Sensors | Flagged | Detected | Missed | False alarms | Recall | Mean fold AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression, all sensors | 590 | 248 | 11 | 37 | 237 | 22.92% | 7.80% |
| Logistic Regression, stable sensors | 412–426 | 248 | 13 | 35 | 235 | 27.08% | 7.26% |
| Random Forest, all sensors | 590 | 248 | 16 | 32 | 232 | 33.33% | 10.07% |
| Random Forest, stable sensors | 412–426 | 248 | 18 | 30 | 230 | 37.50% | 11.79% |

**Matched ablation:** stability filtering improved the count of detected failures by **2** and reduced false alerts by **2**, for each classifier at the same inspection capacity. Mean average precision improved only for Random Forest, while declining slightly for Logistic Regression. These small differences over just four chronological folds and 48 defects are **not evidence of statistically reliable generalization gains**.

The 25% inspection decision uses a **within-batch ranking**, requiring all 250 predictions for each batch before selecting 62 parts. A live streaming system requires a separately trained threshold or capacity-aware queue.

## Limits and next steps

- Few defect cases, low precision and substantial missed defects prevent practical deployment.
- Overlapping chronological training periods limit effective sample independence.
- The earlier final 20% holdout has been examined in previous experiments; no claim of a fresh untouched final test set is justified.
- A controlled future study should quantify fold-level uncertainty, calibrate a realistic dynamic inspection decision without seeing all future batch scores, and evaluate on genuinely new production data.
- The sensor drift ranking is descriptive, not evidence of causality.

## Reproduce

```bash
python -m secom.ablation --data-dir data/raw --output-dir results_ablation --budget 0.25
```

GitHub Actions archives `ablation_metrics.json` and `ablation_predictions.csv` for independent verification. Dataset files are not committed.
