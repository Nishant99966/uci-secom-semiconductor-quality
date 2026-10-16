from pathlib import Path
import subprocess
import sys
from secom.data import load_secom

def test_synthetic_parser(tmp_path: Path):
    subprocess.run([sys.executable, "scripts/make_synthetic_data.py", "--output-dir", str(tmp_path), "--rows", "120"], check=True)
    x, y, timestamps = load_secom(tmp_path)
    assert x.shape == (120, 30)
    assert len(y) == 120
    assert y.sum() > 0
    assert timestamps.notna().all()
