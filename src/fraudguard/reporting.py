"""Plots and JSON artifacts for a training run."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def write_json(obj, path: Path) -> Path:
    path.write_text(json.dumps(obj, indent=2, cls=_NumpyEncoder), encoding="utf-8")
    return path


def plot_pr_curve(y_true, y_score, metrics: dict, path: Path) -> Path:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(recall, precision, label=f"PR-AUC = {metrics['pr_auc']:.3f}")
    ax.scatter([metrics["recall"]], [metrics["precision"]], c="red", zorder=3, label="chosen threshold")
    ax.axhline(float(np.mean(y_true)), ls="--", c="grey", label="no-skill")
    ax.set(xlabel="Recall (fraud)", ylabel="Precision (fraud)", title="Precision-recall (test)", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_confusion(metrics: dict, path: Path) -> Path:
    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{int(v):,}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    ax.set(
        xticks=[0, 1], yticks=[0, 1], xticklabels=["legit", "fraud"], yticklabels=["legit", "fraud"],
        xlabel="Predicted", ylabel="Actual", title=f"Confusion matrix @ {metrics['threshold']:.3f}",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
