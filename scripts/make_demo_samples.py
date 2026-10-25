"""Create the demo data for the Streamlit app from the real dataset (held-out test window only).

    python scripts/make_demo_samples.py

Writes app/demo_samples.json (a few single transactions) and app/demo_batch.csv (a small labelled batch).
The rows are an excerpt of the public Kaggle dataset (ULB Machine Learning Group); none were used for training.
"""

from __future__ import annotations

import json
from pathlib import Path

from fraudguard.config import load_config
from fraudguard.data import TARGET, load_data, split_data
from fraudguard.ui_core import build_demo_batch, pick_demo_samples

OUT = Path("app")


def main() -> None:
    cfg = load_config("configs/train.yaml")
    df, _ = load_data(cfg["data"]["path"], cfg["data"]["drop_duplicates"])
    splits = split_data(df, **{k: cfg["split"][k] for k in ("strategy", "test_size", "val_size", "seed")})
    test = splits.test

    OUT.mkdir(exist_ok=True)
    samples = pick_demo_samples(test, n_each=3, seed=cfg["split"]["seed"])
    (OUT / "demo_samples.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")

    batch = build_demo_batch(test, n_legit=300, seed=cfg["split"]["seed"])
    batch.to_csv(OUT / "demo_batch.csv", index=False)

    print(f"Wrote {len(samples)} samples and a {len(batch)}-row weighted batch ({int(batch[TARGET].sum())} frauds, "
          f"weights sum to {batch['weight'].sum():,.0f} = test window size) to {OUT}/")


if __name__ == "__main__":
    main()
