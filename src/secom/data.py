from pathlib import Path
import pandas as pd


def load_secom(data_dir: str | Path):
    root = Path(data_dir)
    features_path = root / "secom.data"
    labels_path = root / "secom_labels.data"
    if not features_path.exists() or not labels_path.exists():
        raise FileNotFoundError(f"Expected secom.data and secom_labels.data under {root}")

    features = pd.read_csv(features_path, sep=r"\s+", header=None,
                           na_values=["NaN", "nan"])

    # UCI's official timestamp is sometimes enclosed in quotes, so splitting
    # on arbitrary whitespace can produce either two or three columns.
    # Keep the complete timestamp after the first whitespace instead.
    labels = []
    stamps = []
    with labels_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Invalid SECOM labels line: {line!r}")
            labels.append(int(parts[0]))
            stamps.append(parts[1].strip().strip('"').strip("'"))

    if len(features) != len(labels):
        raise ValueError("Sensor and label row counts do not match")
    if not set(labels).issubset({-1, 1}):
        raise ValueError("Expected labels encoded as -1 (pass) or 1 (fail)")

    timestamps = pd.to_datetime(stamps, format="%d/%m/%Y %H:%M:%S",
                                errors="coerce")
    if timestamps.isna().any():
        raise ValueError("Unexpected or invalid SECOM date/time value")

    x = features.apply(pd.to_numeric, errors="coerce").astype("float64")
    x.columns = [f"sensor_{i+1:03d}" for i in range(x.shape[1])]
    y = pd.Series(labels, name="failure").eq(1).astype(int)
    return x, y, pd.Series(timestamps)
