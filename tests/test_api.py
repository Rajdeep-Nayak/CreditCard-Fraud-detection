import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from fraudguard.api.app import create_app  # noqa: E402
from fraudguard.api.example import EXAMPLE_TRANSACTION  # noqa: E402
from fraudguard.api.schemas import MAX_BATCH  # noqa: E402
from fraudguard.serving import load_local_bundle  # noqa: E402

from test_serving import trained_run_dir  # noqa: E402


def make_client():
    bundle = load_local_bundle(trained_run_dir())
    return TestClient(create_app(loader=lambda: bundle)), bundle


def test_health_and_root():
    client, bundle = make_client()
    with client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok", "model_loaded": True, "model_version": bundle.version}
        assert client.get("/").json()["docs"] == "/docs"


def test_model_info():
    client, bundle = make_client()
    with client:
        info = client.get("/model-info").json()
    assert info["threshold"] == bundle.threshold
    assert info["feature_columns"][0] == "Time" and info["feature_columns"][-1] == "Amount"
    assert "pr_auc" in info["validation_metrics"]


def test_predict_single_transaction():
    client, bundle = make_client()
    with client:
        r = client.post("/predict", json=EXAMPLE_TRANSACTION)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["fraud_probability"] <= 1
    assert body["is_fraud"] == (body["fraud_probability"] >= body["threshold"])
    assert body["threshold"] == bundle.threshold and body["model_version"] == bundle.version


def test_threshold_override_is_echoed_and_validated():
    client, _ = make_client()
    with client:
        ok = client.post("/predict?threshold=0.9", json=EXAMPLE_TRANSACTION).json()
        assert ok["threshold"] == 0.9 and ok["is_fraud"] == (ok["fraud_probability"] >= 0.9)
        assert client.post("/predict?threshold=0", json=EXAMPLE_TRANSACTION).status_code == 422
        assert client.post("/predict?threshold=1.5", json=EXAMPLE_TRANSACTION).status_code == 422


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.pop("V3"),                      # missing field
        lambda d: d.update(Extra=1.0),              # unknown field
        lambda d: d.update(Amount=-5.0),            # negative amount
        lambda d: d.update(Time=-1.0),              # negative time
        lambda d: d.update(V1="not-a-number"),      # wrong type
    ],
)
def test_invalid_transactions_are_rejected(mutate):
    client, _ = make_client()
    payload = dict(EXAMPLE_TRANSACTION)
    mutate(payload)
    with client:
        assert client.post("/predict", json=payload).status_code == 422


def test_batch_prediction_summary():
    client, _ = make_client()
    with client:
        r = client.post("/predict/batch", json={"transactions": [EXAMPLE_TRANSACTION] * 3})
    body = r.json()
    assert r.status_code == 200 and body["count"] == 3 and len(body["predictions"]) == 3
    assert body["flagged"] == sum(p["is_fraud"] for p in body["predictions"])


def test_batch_limits():
    client, _ = make_client()
    with client:
        assert client.post("/predict/batch", json={"transactions": []}).status_code == 422
        too_many = {"transactions": [EXAMPLE_TRANSACTION] * (MAX_BATCH + 1)}
        assert client.post("/predict/batch", json=too_many).status_code == 422


def test_service_reports_503_when_model_cannot_load():
    def broken():
        raise FileNotFoundError("artifacts/champion/model.joblib not found")

    with TestClient(create_app(loader=broken)) as client:
        health = client.get("/health")
        assert health.status_code == 503 and "not found" in health.json()["detail"]
        assert client.post("/predict", json=EXAMPLE_TRANSACTION).status_code == 503
