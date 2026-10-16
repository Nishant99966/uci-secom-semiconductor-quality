# Real UCI SECOM — Manufacturing Drift Audit

**Audit run:** https://github.com/Nishant99966/uci-secom-semiconductor-quality/actions/runs/38959083198

**Data:** original UCI SECOM, 1,567 manufacturing parts, 590 sensor columns, 104 failures. Chronologically sorted and divided into five approximately equal observation-count periods. The analysis is **descriptive and exploratory**; it uses all observation periods including the previously evaluated last 20%, so it cannot serve as an unbiased model-selection or prospective validation result.

## Failure prevalence

| Chronological period | Samples | Failures | Failure rate | Overall missing-cell fraction |
|---|---:|---:|---:|---:|
| 1 | 314 | 44 | 14.01% | 4.99% |
| 2 | 313 | 21 | 6.71% | 5.54% |
| 3 | 314 | 11 | 3.50% | 4.90% |
| 4 | 313 | 11 | 3.51% | 3.41% |
| 5 | 313 | 17 | 5.43% | 3.85% |

The first period's failure prevalence exceeds the third period's by about a factor of four. This is meaningful label-distribution drift, but does not identify causal drivers.

## Selected sensor shifts: early (periods 1–2) versus late (periods 4–5)

| Sensor | Standardized mean shift | Early missing | Late missing |
|---|---:|---:|---:|
| sensor_079 | 1.911 | 0.00% | 3.83% |
| sensor_543 | 1.513 | 0.00% | 0.32% |
| sensor_029 | 1.476 | 0.16% | 0.00% |
| sensor_549 | 1.165 | 24.56% | 1.44% |
| sensor_248 | 0.733 | 19.14% | 77.32% |
| sensor_080 | 1.201 | 0.00% | 3.83% |

The shift indicator is the absolute mean difference divided by pooled standard deviation (where possible); the ranking additionally adds absolute missingness-rate change. The rank is not a measure of predictive relevance or statistical significance. Missingness changes may indicate evolving measurement coverage or process configuration.

## Interpretation

- Both sensor distributions and the defect base rate vary with time. This is consistent with the models' unstable performance in expanding chronological validation.
- The actual factory root causes are **unknown**; the sensor labels are anonymized.
- An apparent improvement in recall on a previously examined test period does not prove stable future performance.
- Do **not** use the full-data drift ranking to select model features and then report an unchanged holdout score: that would contaminate the supposedly untouched evaluation process.

## Suggested next controlled experiment

Within each forward-chaining training fold only, rank sensors by predictive signal and temporal stability, fit a regularized model, and assess recall at a fixed inspection-rate budget on that fold's later period. Aggregate fold-level metrics and uncertainty rather than optimizing thresholds on the same validation rows. A genuinely unseen future sample is ultimately required for independent confirmation.

## Reproduce

```bash
python -m secom.drift --data-dir data/raw --output-dir results_drift
```

The GitHub Actions workflow **SECOM Sensor Drift Audit** downloads the original UCI dataset and preserves `drift_report.json`, `period_summary.csv`, and `sensor_drift.csv` as artifacts.
