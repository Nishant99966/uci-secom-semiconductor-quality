"""Load the *original* UCI SECOM raw files into a normalized SQLite database."""
from __future__ import annotations

import argparse
import io
import math
import sqlite3
import urllib.request
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile

UCI_URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"
ROWS_EXPECTED = 1567
SENSORS_EXPECTED = 590


def read_source(data_dir):
    if data_dir:
        root = Path(data_dir)
        return (root / "secom.data").read_bytes(), (root / "secom_labels.data").read_bytes()
    with urllib.request.urlopen(UCI_URL, timeout=90) as response:
        archive_bytes = response.read()
    with ZipFile(io.BytesIO(archive_bytes)) as archive:
        def load(name):
            matches = [p for p in archive.namelist() if Path(p).name == name]
            if len(matches) != 1:
                raise ValueError(f"Expected one {name} in UCI archive, found {matches}")
            return archive.read(matches[0])
        return load("secom.data"), load("secom_labels.data")


def parsed_records(features_bytes, labels_bytes):
    features = features_bytes.decode("utf-8").splitlines()
    labels = labels_bytes.decode("utf-8").splitlines()
    if len(features) != len(labels):
        raise ValueError(f"Rows do not match: {len(features)} features and {len(labels)} labels")
    for index, (feature_line, label_line) in enumerate(zip(features, labels), start=1):
        raw = feature_line.split()
        if len(raw) != SENSORS_EXPECTED:
            raise ValueError(f"Part {index} has {len(raw)} sensor values; expected {SENSORS_EXPECTED}")
        label_parts = label_line.split(maxsplit=1)
        if len(label_parts) != 2:
            raise ValueError(f"Bad label row {index}")
        label = int(label_parts[0])
        if label not in (-1, 1):
            raise ValueError(f"Unexpected class {label!r} in row {index}")
        timestamp = label_parts[1].strip().strip('"').strip("'")
        parsed_time = datetime.strptime(timestamp, "%d/%m/%Y %H:%M:%S")
        measurements = []
        for col, val in enumerate(raw, start=1):
            number = float(val)
            number = number if math.isfinite(number) else None
            measurements.append((index, col, number))
        yield (index, parsed_time.isoformat(sep=" "), int(label == 1)), measurements


def build_database(database, data_dir=None, enforce_original=True):
    features, labels = read_source(data_dir)
    database = Path(database)
    database.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(database)
    try:
        db.execute("PRAGMA foreign_keys = ON")
        schema = (Path(__file__).resolve().parents[1] / "sql" / "01_schema.sql").read_text()
        # Prevent accidental mixing with a previous load.
        db.executescript("DROP TABLE IF EXISTS sensor_readings; DROP TABLE IF EXISTS parts;")
        db.executescript(schema)
        n = 0
        with db:
            for row, measurements in parsed_records(features, labels):
                db.execute("INSERT INTO parts (part_id, recorded_at, is_failed) VALUES (?, ?, ?)", row)
                db.executemany("INSERT INTO sensor_readings (part_id, sensor_id, reading) VALUES (?, ?, ?)",
                               measurements)
                n += 1
        if enforce_original and n != ROWS_EXPECTED:
            raise ValueError(f"Original UCI SECOM should contain {ROWS_EXPECTED} records; got {n}")
        count = db.execute("SELECT COUNT(*) FROM sensor_readings").fetchone()[0]
        if count != n * SENSORS_EXPECTED:
            raise ValueError(f"Incorrect sensor reading count {count}")
        summary = db.execute("SELECT COUNT(*), SUM(is_failed) FROM parts").fetchone()
        print(f"Loaded {summary[0]} manufacturing parts, {summary[1]} failures, "
              f"{count} sensor cells into {database}")
        return summary[0], summary[1], count
    finally:
        db.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--database", default="data/secom.sqlite")
    p.add_argument("--data-dir", default=None)
    a = p.parse_args()
    build_database(a.database, a.data_dir)
