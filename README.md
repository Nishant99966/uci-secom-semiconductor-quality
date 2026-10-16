# Semiconductor Manufacturing Quality Analysis — SQL

**SQL portfolio project | SQLite | Real UCI SECOM manufacturing data**

Analyze the quality of 1,567 semiconductor manufacturing records and 590 anonymized sensor measurements using **SQL**, not a machine-learning model. Questions answered: How frequently does production fail? Does the failure rate change over time? Which sensors are frequently missing? Which sensor readings differ between failed and passed parts?

> The earlier machine-learning work is preserved in the [machine-learning branch](../../tree/machine-learning). The \`main\` branch is exclusively for SQL analysis.

## Dataset and questions

Source: [UCI SECOM](https://archive.ics.uci.edu/dataset/179/secom), a public semiconductor manufacturing dataset with sensor columns and pass/fail labels. The loader uses the official [UCI data archive](https://archive.ics.uci.edu/static/public/179/secom.zip); files are **not** stored in Git.

- **Data modeling:** \`parts\` (one manufactured part per row) and \`sensor_readings\` (one reading per part and sensor).
- **Cleaning:** standardize time stamps, convert \`NaN\` to database \`NULL\`, validate row and column counts, enforce valid outcome labels.
- **SQL skills:** joins, grouping, CTEs, conditional aggregation, window functions, analytic views, and NULL-aware statistics.
- **Business questions:** defect rate, missing sensor coverage, changes in failure rates over time, and differences in readings between passed and failed parts.

## Run locally (Python 3.10+, no third-party packages required)

\`\`\`bash
python scripts/load_secom.py --database data/secom.sqlite
python scripts/run_analysis.py --database data/secom.sqlite --out results
python -m unittest discover -s tests -v
\`\`\`

By default, the loader downloads the official original UCI archive. To use an existing local copy, point \`--data-dir\` at a folder containing \`secom.data\` and \`secom_labels.data\`.

\`\`\`bash
python scripts/load_secom.py --data-dir data/raw --database data/secom.sqlite
\`\`\`

Results are exported to CSV in \`results/\`; the SQLite database and raw datasets are ignored by Git. **GitHub Actions runs the same analysis on the original public data.**

## Project layout

\`\`\`text
sql/01_schema.sql                  Relational tables and constraints
sql/02_quality_kpis.sql            Overall production pass/fail metrics
sql/03_monthly_quality.sql         Time-based quality trends with a window function
sql/04_sensor_missingness.sql      Missing values by sensor
sql/05_sensor_failure_comparison.sql Passed-vs-failed sensor mean differences
scripts/load_secom.py              Reproducible data download and ETL into SQLite
scripts/run_analysis.py            Execute queries and export CSV reports
tests/test_sql_project.py          Self-contained ETL/SQL smoke tests
.github/workflows/sql-analysis.yml Runs full SQL analysis on real UCI data
\`\`\`

## How to interpret the analysis

Differences between sensor readings for passed and failed products are **descriptive associations**, not evidence of root causes. Anonymized sensor names prevent physical interpretation. Failure rates may vary over time; do not treat exploratory SQL comparisons as causal claims. The project does not predict whether an individual product will fail.

The published report is [SQL_RESULTS.md](SQL_RESULTS.md) after the verified real-data workflow runs.

## Branches

- **main (this branch):** SQL manufacturing data engineering and business analysis.
- **machine-learning:** preserved historical defect-prediction research, experiments and reports.

License: MIT.
