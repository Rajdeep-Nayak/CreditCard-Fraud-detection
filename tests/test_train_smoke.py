import json
from pathlib import Path

import joblib
import pytest

from fraudguard.config import load_config
from fraudguard.data import FEATURE_COLS
from fraudguard.synthetic import make_synthetic_creditcard
from fraudguard.train import run_training


@pytest.mark.parametrize(
    ("model", "params"),
    [("logreg", {}), ("rf", {"n_estimators": 40}), ("hgb", {"max_iter": 60})],
)
def test_end_to_end_training(tmp_path, model, params):
    cfg = load_config(
        None,
        {
            "model": {"name": model, "params": params},
            "tracking": {"enabled": False},
            "outputs": {"dir": str(tmp_path)},
            "threshold": {"strategy": "cost", "fn_cost_mode": "amount", "cost_fp": 5.0},
        },
    )
    summary = run_training(cfg, make_synthetic_creditcard(8000, fraud_rate=0.03))
    out = Path(summary["out_dir"])

    assert summary["metrics"]["pr_auc"] > 0.3  # synthetic data is learnable
    for name in ("model.joblib", "metadata.json", "pr_curve.png", "confusion_matrix.png"):
        assert (out / name).exists()

    meta = json.loads((out / "metadata.json").read_text())
    assert meta["feature_columns"] == FEATURE_COLS

    # the saved pipeline accepts raw columns, which is exactly what the API will send
    model_obj = joblib.load(out / "model.joblib")
    proba = model_obj.predict_proba(make_synthetic_creditcard(50, seed=7)[FEATURE_COLS])[:, 1]
    assert proba.shape == (50,) and ((proba >= 0) & (proba <= 1)).all()


def test_invalid_config_is_rejected():
    with pytest.raises(ValueError):
        load_config(None, {"split": {"strategy": "random-ish"}})
