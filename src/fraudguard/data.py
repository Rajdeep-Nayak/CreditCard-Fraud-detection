"""Loading, validating and splitting the credit card fraud data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

V_COLS = [f"V{i}" for i in range(1, 29)]
FEATURE_COLS = ["Time", *V_COLS, "Amount"]
TARGET = "Class"


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def describe(self) -> dict[str, dict[str, float]]:
        return {
            name: {"rows": len(part), "frauds": int(part[TARGET].sum()), "fraud_rate": float(part[TARGET].mean())}
            for name, part in (("train", self.train), ("val", self.val), ("test", self.test))
        }


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in [*FEATURE_COLS, TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if df[[*FEATURE_COLS, TARGET]].isna().any().any():
        raise ValueError("Dataset contains missing values; the pipeline expects none.")
    if not set(df[TARGET].unique()) <= {0, 1}:
        raise ValueError("Target column 'Class' must contain only 0/1.")


def load_data(path: str | Path, drop_duplicates: bool = True) -> tuple[pd.DataFrame, dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` "
            "(Kaggle credentials needed) or `python scripts/download_data.py --synthetic` for demo data."
        )
    df = pd.read_csv(path)
    validate_schema(df)
    info = {"rows_raw": len(df), "duplicates_dropped": 0}
    if drop_duplicates:
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        info["duplicates_dropped"] = before - len(df)
    info["rows"] = len(df)
    info["frauds"] = int(df[TARGET].sum())
    info["fraud_rate"] = float(df[TARGET].mean())
    return df, info


def _check_both_classes(splits: Splits) -> None:
    for name, part in (("train", splits.train), ("val", splits.val), ("test", splits.test)):
        if part[TARGET].nunique() < 2:
            raise ValueError(
                f"The {name} split contains only one class. Use more data, a larger split, "
                "or split.strategy='stratified'."
            )


def split_data(
    df: pd.DataFrame,
    strategy: str = "time",
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 42,
) -> Splits:
    """Split into train / val / test.

    time:       sort by Time; the last `test_size` is test, the last `val_size` of the rest is val.
                This mimics production (the model only ever sees the past).
    stratified: random split preserving the fraud rate in every part.
    """
    if strategy == "time":
        ordered = df.sort_values("Time", kind="mergesort").reset_index(drop=True)
        n_test = int(round(len(ordered) * test_size))
        train_val, test = ordered.iloc[: len(ordered) - n_test], ordered.iloc[len(ordered) - n_test :]
        n_val = int(round(len(train_val) * val_size))
        train, val = train_val.iloc[: len(train_val) - n_val], train_val.iloc[len(train_val) - n_val :]
    elif strategy == "stratified":
        train_val, test = train_test_split(df, test_size=test_size, stratify=df[TARGET], random_state=seed)
        train, val = train_test_split(
            train_val, test_size=val_size, stratify=train_val[TARGET], random_state=seed
        )
    else:
        raise ValueError(f"Unknown split strategy: {strategy!r}")

    splits = Splits(*(part.reset_index(drop=True) for part in (train, val, test)))
    _check_both_classes(splits)
    return splits
