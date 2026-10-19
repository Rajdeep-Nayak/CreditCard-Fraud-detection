import numpy as np
import pytest

from fraudguard.metrics import compute_metrics, confusion_counts, recall_at_precision, select_threshold


def _brute_force_best_cost(y, s, fn_costs, cost_fp):
    best = (float(fn_costs[y == 1].sum()), None)  # flag nothing
    for t in np.unique(s):
        pred = s >= t
        cost = fn_costs[(y == 1) & ~pred].sum() + cost_fp * ((y == 0) & pred).sum()
        if cost < best[0]:
            best = (float(cost), float(t))
    return best


def test_confusion_counts():
    c = confusion_counts([1, 1, 0, 0, 0], [1, 0, 1, 0, 0])
    assert c == {"tn": 2, "fp": 1, "fn": 1, "tp": 1}


def test_fraud_class_metrics_are_correct():
    # 3 frauds, 7 legit. Model catches 2 frauds and raises 1 false alarm.
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.1, 0.7, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1])
    m = compute_metrics(y, s, threshold=0.5)
    assert m["tp"] == 2 and m["fp"] == 1 and m["fn"] == 1 and m["tn"] == 6
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    # the legacy notebook's "precision" was TN/(TN+FN) = 6/7 and its "recall" TN/(TN+FP) = 6/7
    assert m["precision"] != pytest.approx(6 / 7)
    assert m["specificity"] == pytest.approx(6 / 7)


def test_cost_threshold_matches_brute_force():
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.1).astype(int)
    s = np.clip(0.55 * y + rng.normal(0.25, 0.2, 400), 0, 1)
    amounts = rng.lognormal(3, 1, 400)
    thr = select_threshold(y, s, "cost", fn_costs=amounts, cost_fp=5.0)
    pred = s >= thr
    got = amounts[(y == 1) & ~pred].sum() + 5.0 * ((y == 0) & pred).sum()
    want, _ = _brute_force_best_cost(y, s, amounts, 5.0)
    assert got == pytest.approx(want)


def test_costlier_false_alarms_raise_the_threshold():
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.05).astype(int)
    s = np.clip(0.5 * y + rng.normal(0.25, 0.2, 2000), 0, 1)
    cheap = select_threshold(y, s, "cost", fn_costs=np.full(2000, 100.0), cost_fp=1.0)
    pricey = select_threshold(y, s, "cost", fn_costs=np.full(2000, 100.0), cost_fp=50.0)
    assert pricey >= cheap


def test_precision_floor_and_recall_at_precision():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    s = np.array([0.95, 0.9, 0.85, 0.4, 0.6, 0.3, 0.2, 0.1, 0.05, 0.01])
    thr = select_threshold(y, s, "precision_floor", min_precision=1.0)
    m = compute_metrics(y, s, thr)
    assert m["precision"] == 1.0 and m["recall"] == pytest.approx(0.75)
    assert recall_at_precision(y, s, 1.0) == pytest.approx(0.75)


def test_f1_strategy_beats_default_threshold():
    rng = np.random.default_rng(2)
    y = (rng.random(3000) < 0.03).astype(int)
    s = np.clip(0.4 * y + rng.normal(0.2, 0.15, 3000), 0, 1)
    best = compute_metrics(y, s, select_threshold(y, s, "f1"))["f1"]
    assert best >= compute_metrics(y, s, 0.5)["f1"]


def test_cost_saving_is_reported():
    y = np.array([1, 0, 0, 1])
    s = np.array([0.9, 0.1, 0.2, 0.8])
    m = compute_metrics(y, s, 0.5, fn_costs=np.array([100.0, 5, 5, 50]), cost_fp=10.0)
    assert m["total_cost"] == 0.0 and m["cost_saving_pct"] == pytest.approx(100.0)


def test_select_threshold_needs_both_classes():
    with pytest.raises(ValueError):
        select_threshold([0, 0, 0], [0.1, 0.2, 0.3])
