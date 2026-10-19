"""Preprocessing. The whole thing lives inside the sklearn Pipeline, so the API
later receives raw columns (Time, V1..V28, Amount) and never re-implements this."""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler

from .data import V_COLS


def _hour_of_day_cyclic(x) -> np.ndarray:
    """Seconds since first transaction -> (sin, cos) of hour of day.

    Raw elapsed seconds would extrapolate badly on a time-based split, hour-of-day does not.
    """
    hours = (np.asarray(x, dtype=float).ravel() / 3600.0) % 24
    angle = 2 * np.pi * hours / 24
    return np.column_stack([np.sin(angle), np.cos(angle)])


def _hour_names_out(transformer, input_features):  # noqa: ARG001
    return np.array(["hour_sin", "hour_cos"])


def build_preprocessor(
    use_time_features: bool = True, log_amount: bool = True, scale_v: bool = False
) -> ColumnTransformer:
    """scale_v=True for linear models; trees don't need it."""
    amount_steps = []
    if log_amount:
        amount_steps.append(("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))
    amount_steps.append(("robust", RobustScaler()))

    transformers = [("amount", Pipeline(amount_steps), ["Amount"])]
    if use_time_features:
        transformers.append(
            ("hour", FunctionTransformer(_hour_of_day_cyclic, feature_names_out=_hour_names_out), ["Time"])
        )
    transformers.append(("v", StandardScaler() if scale_v else "passthrough", V_COLS))

    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
