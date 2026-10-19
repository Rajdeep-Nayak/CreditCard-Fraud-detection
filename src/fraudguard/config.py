"""Config loading: YAML file + defaults + CLI overrides, with basic validation."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "data": {"path": "data/raw/creditcard.csv", "drop_duplicates": True},
    "split": {"strategy": "time", "test_size": 0.2, "val_size": 0.2, "seed": 42},
    "features": {"use_time_features": True, "log_amount": True},
    "imbalance": "class_weight",
    "model": {"name": "hgb", "params": {}},
    "threshold": {
        "strategy": "cost",
        "fn_cost_mode": "amount",
        "cost_fn": 100.0,
        "cost_fp": 5.0,
        "min_precision": 0.8,
    },
    "tracking": {"enabled": True, "uri": "sqlite:///mlflow.db", "experiment": "credit-card-fraud"},
    "outputs": {"dir": "models"},
}

_CHOICES = {
    ("split", "strategy"): {"time", "stratified"},
    ("threshold", "strategy"): {"cost", "f1", "precision_floor"},
    ("threshold", "fn_cost_mode"): {"amount", "fixed"},
}
IMBALANCE_CHOICES = {"class_weight", "smote", "none"}


def deep_update(base: dict, new: dict) -> dict:
    """Recursively merge `new` into a copy of `base`."""
    out = copy.deepcopy(base)
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def validate(cfg: dict) -> None:
    for (section, key), allowed in _CHOICES.items():
        if cfg[section][key] not in allowed:
            raise ValueError(f"{section}.{key} must be one of {sorted(allowed)}, got {cfg[section][key]!r}")
    if cfg["imbalance"] not in IMBALANCE_CHOICES:
        raise ValueError(f"imbalance must be one of {sorted(IMBALANCE_CHOICES)}, got {cfg['imbalance']!r}")
    split = cfg["split"]
    if not (0 < split["test_size"] < 0.5 and 0 < split["val_size"] < 0.5):
        raise ValueError("split.test_size and split.val_size must be in (0, 0.5)")


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        with open(path, encoding="utf-8") as fh:
            cfg = deep_update(cfg, yaml.safe_load(fh) or {})
    if overrides:
        cfg = deep_update(cfg, overrides)
    validate(cfg)
    return cfg


def flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    """{'a': {'b': 1}} -> {'a.b': 1}; handy for logging params."""
    flat: dict[str, Any] = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            flat.update(flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat
