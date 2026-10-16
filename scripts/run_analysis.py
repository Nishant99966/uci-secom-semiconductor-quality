"""Run SQL analyses against SQLite and export machine-readable CSV result tables."""
from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path


def run(database="data/secom.sqlite", out="results"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    query_dir = Path(__file__).resolve().parents[1] / "sql"
    produced = {}
    with sqlite3.connect(database) as db:
        for sql_file in sorted(query_dir.glob("0[2-9]_*.sql")):
            cursor = db.execute(sql_file.read_text(encoding="utf-8"))
            columns = [x[0] for x in cursor.description]
            rows = cursor.fetchall()
            csv_path = out / (sql_file.stem + ".csv")
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(columns)
                writer.writerows(rows)
            produced[sql_file.stem] = {"rows": len(rows), "path": str(csv_path)}
            print(f"{sql_file.name}: {len(rows)} rows -> {csv_path}")
    return produced


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--database", default="data/secom.sqlite")
    p.add_argument("--out", default="results")
    a = p.parse_args()
    run(a.database, a.out)
