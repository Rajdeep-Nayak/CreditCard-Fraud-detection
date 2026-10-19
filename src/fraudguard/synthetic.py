"""Synthetic data with the same schema as the Kaggle file.

Only for demos, tests and CI. Never report model quality from this data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from .data import FEATURE_COLS, TARGET, V_COLS


def make_synthetic_creditcard(n_rows: int = 20_000, fraud_rate: float = 0.01, seed: int = 42) -> pd.DataFrame:
    X, y = make_classification(
        n_samples=n_rows,
        n_features=len(V_COLS),
        n_informative=10,
        n_redundant=4,
        n_clusters_per_class=2,
        weights=[1 - fraud_rate, fraud_rate],
        class_sep=1.2,
        flip_y=0.002,
        random_state=seed,
    )
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(X, columns=V_COLS)
    df["Time"] = np.sort(rng.uniform(0, 172_792, n_rows))
    df["Amount"] = np.round(rng.lognormal(3.2, 1.4, n_rows), 2)
    df[TARGET] = y
    return df[[*FEATURE_COLS, TARGET]]
