import numpy as np
import pandas as pd
import pytest

from fraudguard.data import FEATURE_COLS, TARGET, split_data
from fraudguard.serving import load_local_bundle
from fraudguard.synthetic import make_synthetic_creditcard
from fraudguard.ui_core import (
    AMOUNT_REFERENCE,
    HttpBackend,
    LocalBackend,
    apply_threshold,
    evaluate_labelled,
    explain_by_occlusion,
    make_backend,
    pick_demo_samples,
    score_dataframe,
)
from test_serving import trained_run_dir


def local_backend():
    return LocalBackend(load_local_bundle(trained_run_dir()))


class StubBackend:
    """Probability depends only on V14, so the explanation must single it out."""

    def __init__(self):
        self.calls = []

    def predict(self, records, threshold=None):
        self.calls.append(len(records))
        return [{"fraud_probability": float(np.clip(0.1 + 0.2 * max(r["V14"], 0), 0, 1)), "is_fraud": False, "threshold": 0.5} for r in records]


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload, self.status_code, self.text = payload, status_code, str(payload)

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, fail=False):
        self.posts, self.fail = [], fail

    def get(self, url, timeout=None):
        return FakeResponse({"version": "2", "threshold": 0.3})

    def post(self, url, json=None, params=None, timeout=None):
        self.posts.append((url, len(json["transactions"]), params))
        if self.fail:
            return FakeResponse({"detail": "Model not loaded"}, 503)
        return FakeResponse({"predictions": [{"fraud_probability": 0.5, "is_fraud": True, "threshold": 0.3}] * len(json["transactions"])})


def test_local_backend_info_and_predict():
    backend = local_backend()
    assert backend.info()["threshold"] == backend.bundle.threshold
    rows = make_synthetic_creditcard(3, seed=5)[FEATURE_COLS].to_dict("records")
    assert len(backend.predict(rows)) == 3


def test_score_dataframe_keeps_order_and_extra_columns():
    df = make_synthetic_creditcard(25, seed=2)
    scored = score_dataframe(local_backend(), df)
    assert len(scored) == 25 and TARGET in scored.columns
    assert scored["fraud_probability"].between(0, 1).all()
    single = local_backend().predict([df.iloc[7][FEATURE_COLS].to_dict()])[0]["fraud_probability"]
    assert scored["fraud_probability"].iloc[7] == pytest.approx(single)


def test_score_dataframe_rejects_bad_csv():
    df = make_synthetic_creditcard(5).drop(columns=["V9"])
    with pytest.raises(ValueError, match="V9"):
        score_dataframe(local_backend(), df)
    df = make_synthetic_creditcard(5)
    df.loc[2, "Amount"] = np.nan
    with pytest.raises(ValueError, match="Amount"):
        score_dataframe(local_backend(), df)


def test_apply_threshold():
    scored = pd.DataFrame({"fraud_probability": [0.1, 0.5, 0.9]})
    assert apply_threshold(scored, 0.5)["is_fraud"].tolist() == [False, True, True]


def test_evaluate_labelled():
    scored = pd.DataFrame({"fraud_probability": [0.9, 0.8, 0.2, 0.7, 0.1, 0.05], "Class": [1, 1, 1, 0, 0, 0]})
    m = evaluate_labelled(scored, 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 2)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)
    assert evaluate_labelled(scored.drop(columns="Class"), 0.5) is None
    assert evaluate_labelled(scored.assign(Class=0), 0.5) is None


def test_explanation_singles_out_the_driving_feature_in_one_call():
    stub = StubBackend()
    record = {c: 0.0 for c in FEATURE_COLS}
    record["V14"] = 2.0
    record["Amount"] = 500.0
    rows = explain_by_occlusion(stub, record)
    assert stub.calls == [30]  # the original + 29 occluded variants, one batch call
    assert rows[0]["feature"] == "V14" and rows[0]["effect"] == pytest.approx(0.4)
    assert all(abs(r["effect"]) < 1e-9 for r in rows[1:])
    assert next(r for r in rows if r["feature"] == "Amount")["reference"] == AMOUNT_REFERENCE


def test_explanation_with_real_model_covers_all_features():
    rows = explain_by_occlusion(local_backend(), make_synthetic_creditcard(1, seed=9)[FEATURE_COLS].iloc[0].to_dict())
    assert len(rows) == 29
    effects = [abs(r["effect"]) for r in rows]
    assert effects == sorted(effects, reverse=True)


def test_http_backend_chunks_large_batches_and_forwards_threshold():
    session = FakeSession()
    backend = HttpBackend("http://api.test/", session=session)
    out = backend.predict([{"x": 1.0}] * 1500, threshold=0.4)
    assert len(out) == 1500
    assert [n for _, n, _ in session.posts] == [1000, 500]
    assert all(url == "http://api.test/predict/batch" and params == {"threshold": 0.4} for url, _, params in session.posts)
    assert backend.info()["version"] == "2"


def test_http_backend_surfaces_api_errors():
    backend = HttpBackend("http://api.test", session=FakeSession(fail=True))
    with pytest.raises(RuntimeError, match="503"):
        backend.predict([{"x": 1.0}])


def test_make_backend_follows_environment():
    assert isinstance(make_backend({"FRAUD_API_URL": "http://localhost:8000"}), HttpBackend)
    local = make_backend({"FRAUD_MODEL_SOURCE": "local", "FRAUD_MODEL_DIR": trained_run_dir()})
    assert isinstance(local, LocalBackend)


def test_pick_demo_samples_from_test_window():
    test = split_data(make_synthetic_creditcard(8000, fraud_rate=0.03)).test
    samples = pick_demo_samples(test, n_each=3, seed=1)
    assert [s["actual_class"] for s in samples] == [0, 0, 0, 1, 1, 1]
    assert all(set(s["record"]) == set(FEATURE_COLS) for s in samples)
    assert samples[3]["name"] == "Held-out fraud #1"


# ---------------------------------------------------------------- threshold analysis
from fraudguard.ui_core import best_cost_threshold, build_demo_batch, operating_point, threshold_sweep  # noqa: E402


def _toy_scored():
    # 2 frauds (amounts 100 and 50), 4 legit rows each standing for 10 transactions
    return pd.DataFrame(
        {
            "fraud_probability": [0.9, 0.4, 0.6, 0.2, 0.1, 0.05],
            "Class": [1, 1, 0, 0, 0, 0],
            "Amount": [100.0, 50.0, 10.0, 10.0, 10.0, 10.0],
            "weight": [1.0, 1.0, 10.0, 10.0, 10.0, 10.0],
        }
    )


def test_sweep_matches_hand_computation():
    row = operating_point(_toy_scored(), cost_fp=2.0, threshold=0.5)
    # flagged: the 0.9 fraud (caught) and the 0.6 legit row (weight 10 -> 10 false alarms)
    assert row["caught"] == 1 and row["missed"] == 1 and row["false_alarms"] == 10
    assert row["precision"] == pytest.approx(1 / 11) and row["recall"] == pytest.approx(0.5)
    assert row["total_cost"] == pytest.approx(50.0 + 2.0 * 10)  # missed 50 + 10 false alarms * 2
    assert row["baseline_cost"] == 150.0 and row["cost_saving_pct"] == pytest.approx((150 - 70) / 150 * 100)
    assert row["fp_per_10k_legit"] == pytest.approx(10 / 40 * 10_000)


def test_weights_default_to_one_and_best_threshold_is_lowest_cost():
    scored = _toy_scored().drop(columns="weight")
    sweep = threshold_sweep(scored, cost_fp=1.0)
    assert len(sweep) == 99 and sweep["threshold"].is_monotonic_increasing
    best = best_cost_threshold(sweep)
    assert sweep.loc[sweep["threshold"] == best, "total_cost"].iloc[0] == sweep["total_cost"].min()


def test_free_false_alarms_push_the_best_threshold_down():
    scored = _toy_scored()
    cheap = best_cost_threshold(threshold_sweep(scored, cost_fp=0.0))
    pricey = best_cost_threshold(threshold_sweep(scored, cost_fp=1000.0))
    assert cheap <= pricey


def test_demo_batch_weights_restore_the_test_window():
    test = split_data(make_synthetic_creditcard(8000, fraud_rate=0.03)).test
    batch = build_demo_batch(test, n_legit=50, seed=3)
    assert batch["weight"].sum() == pytest.approx(len(test), rel=1e-4)
    assert int(batch["Class"].sum()) == int(test["Class"].sum())  # every fraud is kept
    assert list(batch.columns) == [*FEATURE_COLS, "Class", "weight"]
    assert (batch.loc[batch["Class"] == 1, "weight"] == 1.0).all()
