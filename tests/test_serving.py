import json
import tempfile
from functools import lru_cache
from pathlib import Path

import pytest

from fraudguard.config import load_config
from fraudguard.data import FEATURE_COLS
from fraudguard.serving import ModelBundle, load_bundle, load_local_bundle, predict_records, to_frame
from fraudguard.synthetic import make_synthetic_creditcard
from fraudguard.train import run_training


@lru_cache(maxsize=1)
def trained_run_dir() -> str:
    """Train one tiny model once; the folder layout is the same one training always writes."""
    out = tempfile.mkdtemp()
    cfg = load_config(
        None,
        {
            "model": {"name": "hgb", "params": {"max_iter": 40}},
            "tracking": {"enabled": False},
            "outputs": {"dir": out},
        },
    )
    return run_training(cfg, make_synthetic_creditcard(6000, fraud_rate=0.03))["out_dir"]


def rows(n=4, seed=11):
    return make_synthetic_creditcard(n, seed=seed)[FEATURE_COLS].to_dict("records")


def test_load_local_bundle_from_a_training_run_folder():
    bundle = load_local_bundle(trained_run_dir())
    assert isinstance(bundle, ModelBundle)
    assert 0 < bundle.threshold <= 1.0
    assert bundle.info["feature_columns"] == FEATURE_COLS
    assert bundle.info["model_type"] == "hgb"
    assert 0 < bundle.info["validation_metrics"]["pr_auc"] <= 1


def test_predict_records_shapes_and_decision_rule():
    bundle = load_local_bundle(trained_run_dir())
    out = predict_records(bundle, rows(6))
    assert len(out) == 6
    for p in out:
        assert 0.0 <= p["fraud_probability"] <= 1.0
        assert p["is_fraud"] == (p["fraud_probability"] >= bundle.threshold)
        assert p["threshold"] == bundle.threshold


def test_threshold_override_changes_the_decision_only():
    bundle = load_local_bundle(trained_run_dir())
    base = predict_records(bundle, rows(20))
    strict = predict_records(bundle, rows(20), threshold=1.0)
    assert [p["fraud_probability"] for p in base] == [p["fraud_probability"] for p in strict]
    assert all(p["threshold"] == 1.0 for p in strict)
    assert sum(p["is_fraud"] for p in strict) <= sum(p["is_fraud"] for p in base)


def test_to_frame_orders_columns_and_rejects_bad_input():
    shuffled = [dict(reversed(list(r.items()))) for r in rows(3)]
    assert list(to_frame(shuffled).columns) == FEATURE_COLS
    broken = rows(2)
    del broken[0]["V7"]
    with pytest.raises(ValueError, match="V7"):
        to_frame(broken)
    with pytest.raises(ValueError):
        to_frame([])


def test_missing_model_dir_gives_a_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="registry export"):
        load_local_bundle(tmp_path / "nope")


def test_feature_mismatch_is_rejected(tmp_path):
    src = Path(trained_run_dir())
    (tmp_path / "model.joblib").write_bytes((src / "model.joblib").read_bytes())
    meta = json.loads((src / "metadata.json").read_text(encoding="utf-8"))
    meta["feature_columns"] = meta["feature_columns"][:-1]
    (tmp_path / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="feature columns"):
        load_local_bundle(tmp_path)


def test_exported_champion_metadata_shape_is_understood(tmp_path):
    """The shape written by `registry export` (tags instead of val_metrics)."""
    src = Path(trained_run_dir())
    (tmp_path / "model.joblib").write_bytes((src / "model.joblib").read_bytes())
    meta = {
        "model_version": 2,
        "model": "rf",
        "threshold": 0.3,
        "feature_columns": FEATURE_COLS,
        "tags": {"val_pr_auc": "0.76", "val_recall": "0.7", "git_sha": "abc123"},
    }
    (tmp_path / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    bundle = load_local_bundle(tmp_path)
    assert bundle.version == "2" and bundle.threshold == 0.3
    assert bundle.info["validation_metrics"] == {"pr_auc": 0.76, "recall": 0.7}
    assert bundle.info["git_sha"] == "abc123"


def test_load_bundle_reads_environment():
    bundle = load_bundle({"FRAUD_MODEL_SOURCE": "local", "FRAUD_MODEL_DIR": trained_run_dir()})
    assert bundle.info["model_type"] == "hgb"
    with pytest.raises(ValueError, match="FRAUD_MODEL_SOURCE"):
        load_bundle({"FRAUD_MODEL_SOURCE": "s3"})
