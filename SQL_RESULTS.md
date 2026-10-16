# UCI SECOM SQL Analysis Results

All outputs must come from the original UCI SECOM file and reproducible SQL queries, **not from machine-learning predictions**. The GitHub Actions workflow generates CSV reports from the complete original data and verifies their row counts and KPIs.

The authoritative raw dataset comprises **1,567 manufacturing records**, of which **104 failed**. This corresponds to an overall **6.64% failure rate**.

Other analysis outputs (month-to-month trends, missingness by sensor, mean differences between passing and failing products) are exported as downloadable workflow artifacts to avoid hard-coding unverified findings.

Run \`python scripts/load_secom.py\` followed by \`python scripts/run_analysis.py\` locally, or select **SQL Manufacturing Quality Analysis** under the repository's GitHub Actions tab.

This project is descriptive **SQL analysis**, not a prediction system. The archived ML project is preserved on the \`machine-learning\` branch.
