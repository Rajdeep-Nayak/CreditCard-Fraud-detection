"""Training entry point:  python -m fraudguard.train --config configs/train.yaml"""

from __future__ import annotations

import argparse
import datetime as dt
import platform
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from .config import flatten, load_config
from .data import FEATURE_COLS, TARGET, load_data, split_data
from .metrics import compute_metrics, recall_at_precision, select_threshold
from .models import build_model
from .reporting import plot_confusion, plot_pr_curve, write_json
from .tracking import get_tracker


def _git_sha() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _fn_costs(df: pd.DataFrame, thr_cfg: dict) -> np.ndarray:
    if thr_cfg["fn_cost_mode"] == "amount":
        return df["Amount"].to_numpy(dtype=float)
    return np.full(len(df), float(thr_cfg["cost_fn"]))


def run_training(cfg: dict, df: pd.DataFrame | None = None, data_info: dict | None = None) -> dict:
    """Train one model, evaluate on the time-held-out test set, save artifacts. Returns a summary dict."""
    if df is None:
        df, data_info = load_data(cfg["data"]["path"], cfg["data"]["drop_duplicates"])
    data_info = data_info or {}

    splits = split_data(df, **{k: cfg["split"][k] for k in ("strategy", "test_size", "val_size", "seed")})
    model = build_model(cfg)
    thr_cfg = cfg["threshold"]
    name = cfg["model"]["name"]
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = f"{name}-{cfg['imbalance']}-{stamp}"

    out_dir = Path(cfg["outputs"]["dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    tracker = get_tracker(cfg["tracking"])

    with tracker.run(run_name, tags={"model": name, "git_sha": _git_sha() or "unknown"}):
        tracker.log_params({**flatten(cfg), **{f"data.{k}": v for k, v in data_info.items()}})

        model.fit(splits.train[FEATURE_COLS], splits.train[TARGET])

        # 1) choose the threshold on validation data only
        val_score = model.predict_proba(splits.val[FEATURE_COLS])[:, 1]
        threshold = select_threshold(
            splits.val[TARGET], val_score,
            strategy=thr_cfg["strategy"],
            fn_costs=_fn_costs(splits.val, thr_cfg),
            cost_fp=thr_cfg["cost_fp"],
            min_precision=thr_cfg["min_precision"],
        )

        # 2) report on the untouched test set
        test_score = model.predict_proba(splits.test[FEATURE_COLS])[:, 1]
        metrics = compute_metrics(
            splits.test[TARGET], test_score, threshold,
            fn_costs=_fn_costs(splits.test, thr_cfg), cost_fp=thr_cfg["cost_fp"],
        )
        metrics["recall_at_precision_0.9"] = recall_at_precision(splits.test[TARGET], test_score, 0.9)

        tracker.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        pr_png = plot_pr_curve(splits.test[TARGET], test_score, metrics, out_dir / "pr_curve.png")
        cm_png = plot_confusion(metrics, out_dir / "confusion_matrix.png")

        joblib.dump(model, out_dir / "model.joblib")
        metadata = {
            "run_name": run_name,
            "model": name,
            "threshold": threshold,
            "feature_columns": FEATURE_COLS,
            "metrics": metrics,
            "splits": splits.describe(),
            "data": data_info,
            "config": cfg,
            "git_sha": _git_sha(),
            "trained_at_utc": stamp,
            "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__},
        }
        meta_json = write_json(metadata, out_dir / "metadata.json")

        for artifact in (pr_png, cm_png, meta_json):
            tracker.log_artifact(artifact)
        tracker.log_model(model, splits.test[FEATURE_COLS].head(5))

    return {"run_name": run_name, "out_dir": str(out_dir), "threshold": threshold, "metrics": metrics}


def _print_summary(summary: dict) -> None:
    m = summary["metrics"]
    print(f"\n== {summary['run_name']} ==")
    print(f"  PR-AUC {m['pr_auc']:.4f} | ROC-AUC {m['roc_auc']:.4f} | threshold {m['threshold']:.4f}")
    print(f"  fraud precision {m['precision']:.3f} | recall {m['recall']:.3f} | F1 {m['f1']:.3f}")
    print(f"  TP {int(m['tp'])}  FP {int(m['fp'])}  FN {int(m['fn'])}  TN {int(m['tn'])}")
    if "cost_saving_pct" in m:
        print(f"  cost saving vs. no model: {m['cost_saving_pct']:.1f}%  (fraud amount caught {m['fraud_amount_caught_pct']:.1f}%)")
    print(f"  artifacts -> {summary['out_dir']}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train fraud detection model(s).")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--models", help="comma-separated list, e.g. logreg,rf,hgb (overrides model.name)")
    parser.add_argument("--no-tracking", action="store_true", help="disable MLflow logging")
    args = parser.parse_args(argv)

    overrides: dict = {"tracking": {"enabled": False}} if args.no_tracking else {}
    base_cfg = load_config(args.config, overrides)
    df, info = load_data(base_cfg["data"]["path"], base_cfg["data"]["drop_duplicates"])
    print(f"Loaded {info['rows']:,} rows ({info['duplicates_dropped']} duplicates dropped), fraud rate {info['fraud_rate']:.4%}")

    names = args.models.split(",") if args.models else [base_cfg["model"]["name"]]
    for model_name in names:
        cfg = load_config(args.config, {**overrides, "model": {"name": model_name.strip(), "params": {}}})
        _print_summary(run_training(cfg, df, info))


if __name__ == "__main__":
    main()
