"""Framework-free logic behind the Streamlit app: model backends, batch scoring,
explanations and demo-sample selection. Kept out of the UI file so it can be unit-tested.

Backends (chosen by environment):
    FRAUD_API_URL=http://host:8000  -> HttpBackend: talks to the FastAPI service
    (unset)                         -> LocalBackend: loads artifacts/champion in-process
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .data import FEATURE_COLS, TARGET, V_COLS
from .metrics import compute_metrics
from .serving import ModelBundle, load_bundle, predict_records, to_frame

MAX_BATCH = 1000
AMOUNT_REFERENCE = 22.0  # roughly the median transaction amount in the Kaggle data


# --------------------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------------------
class LocalBackend:
    """Scores in-process with a loaded model bundle (no HTTP)."""

    def __init__(self, bundle: ModelBundle):
        self.bundle = bundle

    def info(self) -> dict[str, Any]:
        return {**self.bundle.info, "threshold": self.bundle.threshold}

    def predict(self, records: Sequence[Mapping[str, float]], threshold: float | None = None) -> list[dict]:
        return predict_records(self.bundle, records, threshold)


class HttpBackend:
    """Scores through the FastAPI service. `session` is injectable for tests."""

    def __init__(self, base_url: str, timeout: float = 90.0, session: Any = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout  # generous: free hosting can take a while to wake up
        self._session = session

    def _http(self):
        if self._session is None:
            import requests

            self._session = requests.Session()
        return self._session

    @staticmethod
    def _check(response) -> Any:
        if response.status_code >= 400:
            raise RuntimeError(f"API error {response.status_code}: {str(response.text)[:300]}")
        return response.json()

    def info(self) -> dict[str, Any]:
        return self._check(self._http().get(f"{self.base_url}/model-info", timeout=self.timeout))

    def predict(self, records: Sequence[Mapping[str, float]], threshold: float | None = None) -> list[dict]:
        out: list[dict] = []
        params = {"threshold": threshold} if threshold is not None else None
        for start in range(0, len(records), MAX_BATCH):
            chunk = [dict(r) for r in records[start : start + MAX_BATCH]]
            body = self._check(
                self._http().post(
                    f"{self.base_url}/predict/batch",
                    json={"transactions": chunk},
                    params=params,
                    timeout=self.timeout,
                )
            )
            out.extend(body["predictions"])
        return out


def make_backend(env: Mapping[str, str] | None = None):
    env = os.environ if env is None else env
    url = env.get("FRAUD_API_URL", "").strip()
    if url:
        return HttpBackend(url)
    return LocalBackend(load_bundle(env))


# --------------------------------------------------------------------------------------
# Batch scoring
# --------------------------------------------------------------------------------------
def score_dataframe(backend, df: pd.DataFrame) -> pd.DataFrame:
    """Add `fraud_probability` to a copy of df. Extra columns (e.g. Class) are kept."""
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    frame = to_frame(df.to_dict("records"))  # rejects NaN/inf, gives clean python floats
    records = [{c: float(v) for c, v in row.items()} for row in frame.to_dict("records")]
    predictions = backend.predict(records)
    out = df.copy()
    out["fraud_probability"] = [p["fraud_probability"] for p in predictions]
    return out


def apply_threshold(scored: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = scored.copy()
    out["is_fraud"] = out["fraud_probability"] >= threshold
    return out


def evaluate_labelled(scored: pd.DataFrame, threshold: float, label_col: str = TARGET) -> dict[str, float] | None:
    """Fraud-class metrics if the CSV has a 0/1 label column with both classes; otherwise None."""
    if label_col not in scored.columns:
        return None
    y = scored[label_col].astype(int)
    if y.nunique() < 2:
        return None
    m = compute_metrics(y, scored["fraud_probability"], threshold)
    keys = ("precision", "recall", "f1", "pr_auc", "tp", "fp", "fn", "tn")
    return {k: m[k] for k in keys}


# --------------------------------------------------------------------------------------
# Explanation
# --------------------------------------------------------------------------------------
def explain_by_occlusion(backend, record: Mapping[str, float], reference: Mapping[str, float] | None = None) -> list[dict]:
    """Approximate, model-agnostic explanation.

    For every feature, replace its value with a neutral reference (0 for the PCA components V1-V28,
    a typical amount for Amount) and measure how far the fraud probability drops. A positive effect
    means that feature pushes this transaction's score *up*. This is not SHAP: it ignores interactions.
    One batch call scores the original plus all variants. `Time` is excluded (no neutral value).
    """
    ref = {**{c: 0.0 for c in V_COLS}, "Amount": AMOUNT_REFERENCE, **(reference or {})}
    features = [*V_COLS, "Amount"]
    variants = [dict(record)]
    for feature in features:
        variant = dict(record)
        variant[feature] = ref[feature]
        variants.append(variant)

    predictions = backend.predict(variants)
    base = predictions[0]["fraud_probability"]
    rows = [
        {
            "feature": feature,
            "value": float(record[feature]),
            "reference": float(ref[feature]),
            "effect": base - predictions[i + 1]["fraud_probability"],
        }
        for i, feature in enumerate(features)
    ]
    rows.sort(key=lambda r: abs(r["effect"]), reverse=True)
    return rows


# --------------------------------------------------------------------------------------
# Demo samples
# --------------------------------------------------------------------------------------
def pick_demo_samples(test_df: pd.DataFrame, n_each: int = 3, seed: int = 42) -> list[dict]:
    """Random real transactions from the held-out test window: some legitimate, some fraud."""
    samples: list[dict] = []
    for label, cls in (("legitimate", 0), ("fraud", 1)):
        part = test_df[test_df[TARGET] == cls]
        chosen = part.sample(n=min(n_each, len(part)), random_state=seed)
        for i, (_, row) in enumerate(chosen.iterrows(), start=1):
            samples.append(
                {
                    "name": f"Held-out {label} #{i}",
                    "actual_class": cls,
                    "record": {c: float(row[c]) for c in FEATURE_COLS},
                }
            )
    return samples


def build_demo_batch(test_df: pd.DataFrame, n_legit: int = 300, seed: int = 42) -> pd.DataFrame:
    """All frauds of the test window plus a sample of legitimate rows, with a `weight` column.

    A 20 % fraud rate would exaggerate precision, so each sampled legitimate row carries the weight of the
    legitimate rows it stands for. Weighted metrics then describe the full test window, and the weights
    sum to len(test_df).
    """
    frauds = test_df[test_df[TARGET] == 1]
    legit_all = test_df[test_df[TARGET] == 0]
    legit = legit_all.sample(n=min(n_legit, len(legit_all)), random_state=seed)
    frauds = frauds.assign(weight=1.0)
    legit = legit.assign(weight=len(legit_all) / len(legit))
    batch = pd.concat([frauds, legit]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    batch["weight"] = batch["weight"].round(6)
    return batch[[*FEATURE_COLS, TARGET, "weight"]]


# --------------------------------------------------------------------------------------
# Threshold analysis (weighted, so a re-weighted demo batch behaves like the full test window)
# --------------------------------------------------------------------------------------
def _arrays(scored: pd.DataFrame, label_col: str = TARGET):
    y = scored[label_col].astype(int).to_numpy() == 1
    p = scored["fraud_probability"].to_numpy(dtype=float)
    w = scored["weight"].to_numpy(dtype=float) if "weight" in scored.columns else np.ones(len(scored))
    amount = scored["Amount"].to_numpy(dtype=float)
    return y, p, w, amount


def threshold_sweep(scored: pd.DataFrame, cost_fp: float, thresholds: Sequence[float] | None = None) -> pd.DataFrame:
    """Precision, recall and cost at many thresholds.

    Cost model (same as training): a missed fraud costs its Amount, a false alarm costs `cost_fp`.
    `cost_saving_pct` is relative to having no model at all.
    """
    y, p, w, amount = _arrays(scored)
    grid = np.round(np.linspace(0.01, 0.99, 99), 4) if thresholds is None else np.asarray(thresholds, dtype=float)
    baseline = float((w * amount)[y].sum())
    legit_total = float(w[~y].sum())
    rows = []
    for t in grid:
        pred = p >= t
        tp = float(w[y & pred].sum())
        fp = float(w[~y & pred].sum())
        fn = float(w[y & ~pred].sum())
        total = float((w * amount)[y & ~pred].sum()) + cost_fp * fp
        rows.append(
            {
                "threshold": float(t),
                "precision": tp / (tp + fp) if (tp + fp) else 1.0,
                "recall": tp / (tp + fn) if (tp + fn) else 0.0,
                "caught": tp,
                "missed": fn,
                "false_alarms": fp,
                "fp_per_10k_legit": fp / legit_total * 10_000 if legit_total else 0.0,
                "total_cost": total,
                "baseline_cost": baseline,
                "cost_saving_pct": (baseline - total) / baseline * 100.0 if baseline else 0.0,
            }
        )
    return pd.DataFrame(rows)


def best_cost_threshold(sweep: pd.DataFrame) -> float:
    """Threshold with the lowest total cost (ties go to the higher threshold = fewer alerts)."""
    lowest = sweep["total_cost"].min()
    return float(sweep.loc[sweep["total_cost"] == lowest, "threshold"].max())


def operating_point(scored: pd.DataFrame, cost_fp: float, threshold: float) -> dict[str, float]:
    """Weighted metrics at exactly one threshold."""
    return threshold_sweep(scored, cost_fp, [threshold]).iloc[0].to_dict()
