"""Validate real parser shape and SQL queries on a tiny synthetic fixture."""
import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path
from scripts.load_secom import build_database
from scripts.run_analysis import run


class SQLPortfolioTests(unittest.TestCase):
    def test_loading_and_reporting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixtures = root / "raw"
            fixtures.mkdir()
            records = [
                ["0.1"] * 590,
                ["0.2"] * 590,
                ["NaN"] + ["0.3"] * 589,
            ]
            (fixtures / "secom.data").write_text("\n".join(" ".join(row) for row in records) + "\n")
            (fixtures / "secom_labels.data").write_text(
                '-1 "01/01/2008 00:00:00"\n'
                '1 "02/01/2008 00:00:00"\n'
                '-1 "03/01/2008 00:00:00"\n'
            )
            db = root / "database.sqlite"
            self.assertEqual(build_database(db, fixtures, enforce_original=False),
                             (3, 1, 1770))
            with sqlite3.connect(db) as con:
                self.assertEqual(con.execute(
                    "SELECT COUNT(*) FROM sensor_readings WHERE reading IS NULL").fetchone()[0], 1)
            summary = run(db, root / "reports")
            self.assertEqual(summary["02_quality_kpis"]["rows"], 1)
            with (root / "reports" / "02_quality_kpis.csv").open() as f:
                metrics = next(csv.DictReader(f))
            self.assertEqual(int(metrics["failed_parts"]), 1)
            self.assertAlmostEqual(float(metrics["failure_rate_pct"]), 33.33)


if __name__ == "__main__":
    unittest.main()
