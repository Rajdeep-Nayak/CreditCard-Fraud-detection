"""Fetch the dataset.

  python scripts/download_data.py              # real data via Kaggle CLI (needs ~/.kaggle/kaggle.json)
  python scripts/download_data.py --synthetic  # demo data with the same schema (never report results on it)

Manual alternative: download creditcard.csv from
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and place it in data/raw/.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

DEST = Path("data/raw")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    target = DEST / "creditcard.csv"

    if args.synthetic:
        from fraudguard.synthetic import make_synthetic_creditcard

        make_synthetic_creditcard().to_csv(target, index=False)
        print(f"Wrote synthetic demo data to {target}  (not real data, don't quote metrics from it)")
        return 0

    if shutil.which("kaggle") is None:
        print("Kaggle CLI not found. `pip install kaggle` and add credentials, or download manually (see docstring).")
        return 1
    subprocess.run(["kaggle", "datasets", "download", "-d", "mlg-ulb/creditcardfraud", "-p", str(DEST)], check=True)
    archive = DEST / "creditcardfraud.zip"
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(DEST)
    archive.unlink()
    print(f"Ready: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
