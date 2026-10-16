from pathlib import Path
from urllib.request import urlretrieve

BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/"
folder = Path("data/raw")
folder.mkdir(parents=True, exist_ok=True)
for name in ("secom.data", "secom_labels.data", "secom.names"):
    target = folder / name
    if not target.exists():
        print(f"Downloading {name} ...")
        urlretrieve(BASE + name, target)
print("Dataset ready at data/raw")
