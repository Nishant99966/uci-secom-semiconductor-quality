import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/synthetic")
    parser.add_argument("--rows", type=int, default=350)
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    n = args.rows
    x = rng.normal(size=(n, 30))
    x[rng.random(x.shape) < .08] = np.nan
    score = np.nan_to_num(x[:, 0]) + .3 * np.nan_to_num(x[:, 1]) + rng.normal(size=n)
    y = np.where(score > np.quantile(score, .91), 1, -1)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(x).to_csv(root / "secom.data", sep=" ", header=False, index=False, na_rep="NaN")
    dates = pd.date_range("2008-07-19", periods=n, freq="h")
    with (root / "secom_labels.data").open("w") as f:
        for label, date in zip(y, dates):
            f.write(f"{label} {date:%d/%m/%Y %H:%M:%S}
")
    print(root)

if __name__ == "__main__":
    main()
