"""Framework-free serving logic: load a model bundle and score records.

Kept separate from FastAPI so it is easy to test and reuse (batch scoring, notebooks, a CLI).

Model sources (environment variables):
    FRAUD_MODEL_SOURCE = local (default) | registry
    FRAUD_MODEL_DIR    = artifacts/champion   (for local; needs model.joblib + metadata.json)
    FRAUD_CONFIG       = configs/train.yaml   (for registry)

Security note: model.joblib is a pickle. Only load files that you produced yourself.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .data import FEATURE_COLS

VALIDATION_KEYS = ("pr_auc", "roc_auc", "precision", "recall", "f1", "cost_saving_pct")


@dataclass
class ModelBundle:
    model: Any
    threshold: float
    info: dict[str, Any]

    @property
    def version(self) -> str:
        return str(self.info.get("version", "unknown"))


def to_frame(records: Sequence[Mapping[str, float]]) -> pd.DataFrame:
    """Records -> DataFrame with exactly the model's raw feature columns, in order.

    Rejects missing, NaN and infinite values: some models (e.g. gradient boosting) would
    otherwise score them silently, which is worse than failing loudly.
    """
    if len(records) == 0:
        raise ValueError("No records to score.")
    df = pd.DataFrame(list(records)).reindex(columns=FEATURE_COLS).astype(float)
    bad = [c for c in FEATURE_COLS if not np.isfinite(df[c]).all()]
    if bad:
        raise ValueError(f"Missing, NaN or infinite values in: {bad}")
    return df


def predict_records(
    bundle: ModelBundle, records: Sequence[Mapping[str, float]], threshold: float | None = None
) -> list[dict[str, Any]]:
    """Score records. `threshold` overrides the model's tuned threshold for this call only."""
    thr = bundle.threshold if threshold is None else float(threshold)
    scores = bundle.model.predict_proba(to_frame(records))[:, 1]
    return [{"fraud_probability": float(s), "is_fraud": bool(s >= thr), "threshold": thr} for s in scores]


def _build_info(meta: Mapping[str, Any], source: str) -> dict[str, Any]:
    tags = meta.get("tags") or {}
    val_block = meta.get("val_metrics") or {}
    validation: dict[str, float] = {}
    for key in VALIDATION_KEYS:
        raw = val_block.get(key, tags.get(f"val_{key}"))
        try:
            validation[key] = float(raw)
        except (TypeError, ValueError):
            continue
    return {
        "version": str(meta.get("model_version") or meta.get("run_name") or "unknown"),
        "model_type": meta.get("model") or tags.get("model"),
        "source": source,
        "feature_columns": list(meta["feature_columns"]),
        "validation_metrics": validation,
        "trained_at_utc": meta.get("trained_at_utc") or tags.get("trained_at_utc"),
        "git_sha": meta.get("git_sha") or tags.get("git_sha"),
    }


def load_local_bundle(model_dir: str | Path) -> ModelBundle:
    """Load model.joblib + metadata.json from a directory.

    Works for an exported champion (`registry export`) and for a raw training run folder (models/<run>/).
    """
    d = Path(model_dir)
    model_path, meta_path = d / "model.joblib", d / "metadata.json"
    for path in (model_path, meta_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Export the champion first: "
                f"python -m fraudguard.registry export --out {d}"
            )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if list(meta.get("feature_columns", [])) != FEATURE_COLS:
        raise ValueError("Model feature columns do not match this version of the code; retrain and re-export.")
    return ModelBundle(joblib.load(model_path), float(meta["threshold"]), _build_info(meta, f"local:{d}"))


def load_registry_bundle(cfg: dict) -> ModelBundle:
    from .registry import load_champion

    model, threshold, info = load_champion(cfg)
    reg = cfg["registry"]
    meta = {"model_version": info["version"], "tags": info["tags"], "feature_columns": FEATURE_COLS}
    return ModelBundle(model, threshold, _build_info(meta, f"registry:{reg['model_name']}@{reg['champion_alias']}"))


def load_bundle(env: Mapping[str, str] | None = None) -> ModelBundle:
    env = os.environ if env is None else env
    source = env.get("FRAUD_MODEL_SOURCE", "local").lower()
    if source == "local":
        return load_local_bundle(env.get("FRAUD_MODEL_DIR", "artifacts/champion"))
    if source == "registry":
        from .config import load_config

        return load_registry_bundle(load_config(env.get("FRAUD_CONFIG", "configs/train.yaml")))
    raise ValueError(f"FRAUD_MODEL_SOURCE must be 'local' or 'registry', got {source!r}")
