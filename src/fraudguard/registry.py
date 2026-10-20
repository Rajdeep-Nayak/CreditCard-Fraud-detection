"""Model registry: every trained model is registered as a version; `promote` gives the best
one the `champion` alias, and only replaces an existing champion if the challenger is better.

    python -m fraudguard.registry list
    python -m fraudguard.registry promote [--metric val_pr_auc] [--dry-run]
    python -m fraudguard.registry show        # loads the champion and scores a few demo rows
"""

from __future__ import annotations

import argparse
import datetime as dt
from typing import Any

from .config import load_config


# --------------------------------------------------------------------------------------
# Pure decision logic (no MLflow import, unit-tested)
# --------------------------------------------------------------------------------------
def _score(candidate: dict | None, metric: str) -> float | None:
    if candidate is None:
        return None
    try:
        return float(candidate["tags"][metric])
    except (KeyError, TypeError, ValueError):
        return None


def choose_champion(
    candidates: list[dict], current: dict | None, metric: str, min_improvement: float = 0.0
) -> dict[str, Any]:
    """Decide whether to promote. Candidates look like {"version": 3, "tags": {"val_pr_auc": "0.81"}}.

    Higher metric is better. Returns {"action": "promote"|"keep"|"none", "best": candidate|None, "reason": str}.
    """
    scored = [(s, c["version"], c) for c in candidates if (s := _score(c, metric)) is not None]
    if not scored:
        return {"action": "none", "best": None, "reason": f"no registered version has a '{metric}' tag"}

    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)  # best score, then newest version
    best_score, _, best = scored[0]

    if current is None:
        return {"action": "promote", "best": best, "reason": "no champion yet"}
    if best["version"] == current["version"]:
        return {"action": "keep", "best": best, "reason": "current champion is already the best version"}

    current_score = _score(current, metric)
    if current_score is None:
        return {"action": "promote", "best": best, "reason": "current champion has no comparable metric"}
    if best_score > current_score + min_improvement:
        return {
            "action": "promote",
            "best": best,
            "reason": f"{metric} {best_score:.4f} beats champion {current_score:.4f} (+{best_score - current_score:.4f})",
        }
    return {
        "action": "keep",
        "best": best,
        "reason": f"best challenger {best_score:.4f} does not beat champion {current_score:.4f} by more than {min_improvement}",
    }


# --------------------------------------------------------------------------------------
# MLflow-facing helpers
# --------------------------------------------------------------------------------------
def _client(cfg: dict):
    from mlflow import MlflowClient

    uri = cfg["tracking"]["uri"]
    return MlflowClient(tracking_uri=uri, registry_uri=uri)


def list_versions(client, name: str) -> list[dict]:
    try:  # alias -> version map lives on the registered model
        by_alias = {alias: int(ver) for alias, ver in (client.get_registered_model(name).aliases or {}).items()}
    except Exception:
        by_alias = {}
    return sorted(
        (
            {
                "version": int(mv.version),
                "tags": dict(mv.tags or {}),
                "aliases": [a for a, ver in by_alias.items() if ver == int(mv.version)],
            }
            for mv in client.search_model_versions(f"name='{name}'")
        ),
        key=lambda v: v["version"],
    )

def current_champion(client, name: str, alias: str, versions: list[dict]) -> dict | None:
    try:
        version = int(client.get_model_version_by_alias(name, alias).version)
    except Exception:  # alias not set yet
        return None
    return next((v for v in versions if v["version"] == version), None)


def _print_table(versions: list[dict], metric: str, alias: str) -> None:
    print(f"{'ver':>4}  {'model':<10} {'imbalance':<13} {metric:<14} {'threshold':<12} aliases")
    for v in versions:
        t = v["tags"]
        score = _score(v, metric)
        print(
            f"{v['version']:>4}  {t.get('model', '?'):<10} {t.get('imbalance', '?'):<13} "
            f"{('%.4f' % score) if score is not None else '-':<14} "
            f"{t.get('threshold', '-')[:10]:<12} {','.join(v['aliases'])}"
        )


def promote(cfg: dict, metric: str | None = None, dry_run: bool = False) -> dict:
    reg = cfg["registry"]
    metric = metric or reg["promote_metric"]
    client = _client(cfg)
    versions = list_versions(client, reg["model_name"])
    if not versions:
        raise SystemExit(f"No versions of '{reg['model_name']}' found. Train first: python -m fraudguard.train")

    champion = current_champion(client, reg["model_name"], reg["champion_alias"], versions)
    decision = choose_champion(versions, champion, metric, reg["min_improvement"])

    _print_table(versions, metric, reg["champion_alias"])
    print(f"\nDecision: {decision['action'].upper()}: {decision['reason']}")

    if decision["action"] == "promote" and not dry_run:
        best = decision["best"]["version"]
        client.set_registered_model_alias(reg["model_name"], reg["champion_alias"], str(best))
        client.set_model_version_tag(
            reg["model_name"], str(best), "promoted_at_utc", dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        )
        print(f"Alias '{reg['champion_alias']}' -> version {best}")
    elif dry_run:
        print("(dry run, nothing changed)")
    return decision


def load_champion(cfg: dict):
    """Load the champion model and its decision threshold. This is what the API will call."""
    import mlflow
    import mlflow.sklearn

    reg = cfg["registry"]
    mlflow.set_tracking_uri(cfg["tracking"]["uri"])
    mlflow.set_registry_uri(cfg["tracking"]["uri"])
    client = _client(cfg)
    mv = client.get_model_version_by_alias(reg["model_name"], reg["champion_alias"])
    model = mlflow.sklearn.load_model(f"models:/{reg['model_name']}@{reg['champion_alias']}")
    info = {"version": int(mv.version), "tags": dict(mv.tags or {})}
    return model, float(info["tags"]["threshold"]), info


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Model registry helper.")
    parser.add_argument("command", choices=["list", "promote", "show"])
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--metric", help="metric tag to rank by (default: registry.promote_metric)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    reg = cfg["registry"]

    if args.command == "list":
        client = _client(cfg)
        _print_table(list_versions(client, reg["model_name"]), args.metric or reg["promote_metric"], reg["champion_alias"])
    elif args.command == "promote":
        promote(cfg, args.metric, args.dry_run)
    else:
        from .data import FEATURE_COLS
        from .synthetic import make_synthetic_creditcard

        model, threshold, info = load_champion(cfg)
        tags = info["tags"]
        print(f"Champion: version {info['version']} ({tags.get('model')}, {tags.get('imbalance')}), threshold {threshold:.6f}")
        demo = make_synthetic_creditcard(5)[FEATURE_COLS]
        scores = model.predict_proba(demo)[:, 1]
        print("Demo rows (synthetic) ->", [f"{s:.4f}" for s in scores], "| flagged:", [bool(s >= threshold) for s in scores])
        print("Champion loaded from the registry and scored successfully.")


if __name__ == "__main__":
    main()
