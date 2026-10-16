import subprocess
import sys
from pathlib import Path
from secom.train import run

def test_pipeline(tmp_path: Path):
    raw = tmp_path / "raw"
    subprocess.run([sys.executable, "scripts/make_synthetic_data.py", "--output-dir", str(raw), "--rows", "180"], check=True)
    output = tmp_path / "out"
    result = run(str(raw), str(output), split="stratified", cv_folds=3)
    assert (output / "metrics.json").exists()
    assert (output / "test_predictions.csv").exists()
    assert result["test_rows"] == 36
