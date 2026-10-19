"""Metrics and threshold selection for highly imbalanced fraud data.

Why this module exists: accuracy / weighted-F1 are dominated by the 99.8 % legitimate class.
We report PR-AUC (threshold-free) and precision / recall / F1 of the *fraud* class, and
choose the decision threshold from business costs instead of defaulting to 0.5.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def confusion_counts(y_true, y_pred) -> dict[str, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
    }


def _div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def compute_metrics(
    y_true,
    y_score,
    threshold: float,
    fn_costs=None,
    cost_fp: float = 0.0,
) -> dict[str, float]:
    """All metrics at one operating point. Fraud (Class=1) is the positive class.

    fn_costs: per-transaction cost incurred if that transaction is fraud and we miss it.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    c = confusion_counts(y_true, y_pred)

    precision = _div(c["tp"], c["tp"] + c["fp"])
    recall = _div(c["tp"], c["tp"] + c["fn"])
    out: dict[str, float] = {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": _div(2 * precision * recall, precision + recall),
        "specificity": _div(c["tn"], c["tn"] + c["fp"]),
        "false_positive_rate": _div(c["fp"], c["fp"] + c["tn"]),
        "alert_rate": float(y_pred.mean()),
        **{k: float(v) for k, v in c.items()},
    }

    if fn_costs is not None:
        fn_costs = np.asarray(fn_costs, dtype=float)
        missed = float(fn_costs[(y_true == 1) & (y_pred == 0)].sum())
        total = missed + cost_fp * c["fp"]
        baseline = float(fn_costs[y_true == 1].sum())  # cost of having no model at all
        out.update(
            {
                "total_cost": total,
                "baseline_cost_no_model": baseline,
                "cost_saving_pct": _div(baseline - total, baseline) * 100.0,
                "fraud_amount_caught_pct": _div(baseline - missed, baseline) * 100.0,
            }
        )
    return out


def _sweep(y_true: np.ndarray, y_score: np.ndarray, fn_costs: np.ndarray):
    """Cumulative counts for every distinct threshold (predict fraud if score >= threshold)."""
    order = np.argsort(-y_score, kind="mergesort")
    s, y, w = y_score[order], y_true[order], fn_costs[order] * y_true[order]
    tps, fps, caught = np.cumsum(y), np.cumsum(1 - y), np.cumsum(w)
    last_of_each_score = np.r_[np.where(np.diff(s) != 0)[0], len(s) - 1]
    return s[last_of_each_score], tps[last_of_each_score], fps[last_of_each_score], caught[last_of_each_score]


def select_threshold(
    y_true,
    y_score,
    strategy: str = "cost",
    fn_costs=None,
    cost_fp: float = 1.0,
    min_precision: float = 0.8,
) -> float:
    """Pick the decision threshold on *validation* data.

    cost:            minimise  sum(cost of missed frauds) + cost_fp * (#false alarms)
    f1:              maximise fraud-class F1
    precision_floor: maximise recall subject to precision >= min_precision
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int(y_true.sum())
    if n_pos == 0 or n_pos == len(y_true):
        raise ValueError("Need both classes to select a threshold.")
    fn_costs = np.ones(len(y_true)) if fn_costs is None else np.asarray(fn_costs, dtype=float)

    thr, tps, fps, caught = _sweep(y_true, y_score, fn_costs)

    if strategy == "cost":
        total_fraud_cost = float((fn_costs * y_true).sum())
        cost = (total_fraud_cost - caught) + cost_fp * fps
        best = int(np.argmin(cost))
        if total_fraud_cost <= cost[best]:  # model can't beat "flag nothing"
            warnings.warn("No threshold beats flagging nothing under the given costs.", stacklevel=2)
            return float(np.nextafter(thr[0], np.inf))
        return float(thr[best])

    if strategy == "f1":
        f1 = 2 * tps / (2 * tps + fps + (n_pos - tps))
        return float(thr[int(np.argmax(f1))])

    if strategy == "precision_floor":
        precision = tps / (tps + fps)
        ok = np.where(precision >= min_precision)[0]
        if len(ok) == 0:
            warnings.warn(f"Precision {min_precision} is unreachable; using the most precise threshold.", stacklevel=2)
            return float(thr[int(np.argmax(precision))])
        return float(thr[ok[int(np.argmax(tps[ok]))]])

    raise ValueError(f"Unknown threshold strategy: {strategy!r}")


def recall_at_precision(y_true, y_score, min_precision: float) -> float:
    """Best achievable fraud recall while keeping precision >= min_precision."""
    y_true = np.asarray(y_true).astype(int)
    thr, tps, fps, _ = _sweep(y_true, np.asarray(y_score, dtype=float), np.ones(len(y_true)))
    precision = tps / (tps + fps)
    ok = precision >= min_precision
    return float(tps[ok].max() / y_true.sum()) if ok.any() else 0.0
