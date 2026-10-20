from fraudguard.registry import choose_champion


def v(version, score, metric="val_pr_auc"):
    return {"version": version, "tags": {metric: str(score)}}


def test_first_champion_is_the_best_version():
    d = choose_champion([v(1, 0.70), v(2, 0.82), v(3, 0.75)], None, "val_pr_auc")
    assert d["action"] == "promote" and d["best"]["version"] == 2


def test_challenger_must_beat_champion():
    versions = [v(1, 0.80), v(2, 0.78)]
    d = choose_champion(versions, versions[0], "val_pr_auc")
    assert d["action"] == "keep" and d["best"]["version"] == 1


def test_better_challenger_replaces_champion():
    versions = [v(1, 0.70), v(2, 0.74)]
    d = choose_champion(versions, versions[0], "val_pr_auc")
    assert d["action"] == "promote" and d["best"]["version"] == 2


def test_min_improvement_blocks_marginal_gains():
    versions = [v(1, 0.800), v(2, 0.805)]
    assert choose_champion(versions, versions[0], "val_pr_auc", min_improvement=0.01)["action"] == "keep"
    assert choose_champion(versions, versions[0], "val_pr_auc", min_improvement=0.001)["action"] == "promote"


def test_ties_prefer_the_newer_version():
    d = choose_champion([v(1, 0.80), v(2, 0.80)], None, "val_pr_auc")
    assert d["best"]["version"] == 2


def test_versions_without_the_metric_are_ignored():
    versions = [{"version": 1, "tags": {}}, {"version": 2, "tags": {"val_pr_auc": "not-a-number"}}, v(3, 0.6)]
    d = choose_champion(versions, None, "val_pr_auc")
    assert d["best"]["version"] == 3
    assert choose_champion(versions[:2], None, "val_pr_auc")["action"] == "none"


def test_champion_without_metric_gets_replaced():
    versions = [{"version": 1, "tags": {}}, v(2, 0.5)]
    assert choose_champion(versions, versions[0], "val_pr_auc")["action"] == "promote"
