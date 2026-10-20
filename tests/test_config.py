import os

from fraudguard.config import load_config


def test_tracking_uri_can_be_overridden_by_environment():
    previous = os.environ.get("FRAUD_TRACKING_URI")
    try:
        os.environ["FRAUD_TRACKING_URI"] = "http://127.0.0.1:5000"
        assert load_config()["tracking"]["uri"] == "http://127.0.0.1:5000"
        del os.environ["FRAUD_TRACKING_URI"]
        assert load_config()["tracking"]["uri"] == "sqlite:///mlflow.db"
    finally:
        if previous is not None:
            os.environ["FRAUD_TRACKING_URI"] = previous
        else:
            os.environ.pop("FRAUD_TRACKING_URI", None)
