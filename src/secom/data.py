from pathlib import Path
import numpy as np
import pandas as pd


def load_secom(data_dir: str | Path):
    root = Path(data_dir)
    features_path = root / "secom.data"
    labels_path = root / "secom_labels.data"
    if not features_path.exists() or not labels_path.exists():
        raise FileNotFoundError(f"Expected secom.data and secom_labels.data under {root}")
    features = pd.read_csv(features_path, sep=r"\s+", header=None, na_values=["NaN", "nan"])
    labels = pd.read_csv(labels_path, sep=r"\s+", header=None)
    if len(features) != len(labels):
        raise ValueError("Sensor and label row counts do not match")
    if labels.shape[1] < 3:
        raise ValueError("Expected label and separate date/time tokens in secom_labels.data")
    y = pd.to_numeric(labels.iloc[:, 0], errors="raise")
    if not set(y.unique()).issubset({-1, 1}):
        raise ValueError("Expected labels encoded as -1 (pass) or 1 (fail)")
    timestamps = pd.to_datetime(labels.iloc[:, 1].astype(str) + " " + labels.iloc[:, 2].astype(str),
                                format="%d/%m/%Y %H:%M:%S", errors="coerce")
    if timestamps.isna().any():
        raise ValueError("Unexpected or invalid SECOM date/time value")
    x = features.apply(pd.to_numeric, errors="coerce").astype("float64")
    x.columns = [f"sensor_{i+1:03d}" for i in range(x.shape[1])]
    return x, (y == 1).astype(int), timestamps
