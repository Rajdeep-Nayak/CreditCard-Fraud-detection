"""Model factory. Every model is a full Pipeline: raw columns in, fraud probability out."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .features import build_preprocessor

MODEL_NAMES = ("logreg", "rf", "hgb", "lightgbm")
LINEAR = {"logreg"}


def _make_classifier(name: str, params: dict[str, Any], seed: int, class_weight: str | None):
    if name == "logreg":
        base = {"max_iter": 1000, "class_weight": class_weight, "random_state": seed}
        return LogisticRegression(**{**base, **params})
    if name == "rf":
        cw = "balanced_subsample" if class_weight else None
        base = {"n_estimators": 300, "min_samples_leaf": 2, "n_jobs": -1, "class_weight": cw, "random_state": seed}
        return RandomForestClassifier(**{**base, **params})
    if name == "hgb":
        base = {"max_iter": 300, "learning_rate": 0.08, "class_weight": class_weight, "random_state": seed}
        return HistGradientBoostingClassifier(**{**base, **params})
    if name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError("lightgbm is not installed: pip install lightgbm") from exc
        base = {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "class_weight": class_weight,
            "random_state": seed,
            "verbose": -1,
        }
        return LGBMClassifier(**{**base, **params})
    raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")


def build_model(cfg: dict) -> Pipeline:
    """Build the full pipeline from a config dict."""
    name = cfg["model"]["name"]
    params = dict(cfg["model"].get("params") or {})
    seed = cfg["split"]["seed"]
    imbalance = cfg["imbalance"]

    prep = build_preprocessor(
        use_time_features=cfg["features"]["use_time_features"],
        log_amount=cfg["features"]["log_amount"],
        scale_v=name in LINEAR,
    )
    clf = _make_classifier(name, params, seed, "balanced" if imbalance == "class_weight" else None)

    if imbalance == "smote":
        # SMOTE lives inside an imblearn Pipeline, so it only ever touches training data.
        try:
            from imblearn.over_sampling import SMOTE
            from imblearn.pipeline import Pipeline as ImbPipeline
        except ImportError as exc:
            raise ImportError("imbalance='smote' needs imbalanced-learn: pip install imbalanced-learn") from exc
        return ImbPipeline([("prep", prep), ("smote", SMOTE(random_state=seed)), ("clf", clf)])

    return Pipeline([("prep", prep), ("clf", clf)])
