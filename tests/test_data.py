import pandas as pd
import pytest

from fraudguard.data import FEATURE_COLS, TARGET, load_data, split_data, validate_schema
from fraudguard.synthetic import make_synthetic_creditcard


def test_schema_validation_rejects_missing_columns():
    df = make_synthetic_creditcard(500).drop(columns=["V3"])
    with pytest.raises(ValueError, match="V3"):
        validate_schema(df)


def test_time_split_never_leaks_future_into_past():
    s = split_data(make_synthetic_creditcard(5000), strategy="time")
    assert s.train["Time"].max() <= s.val["Time"].min()
    assert s.val["Time"].max() <= s.test["Time"].min()
    assert len(s.train) + len(s.val) + len(s.test) == 5000


def test_stratified_split_preserves_fraud_rate():
    df = make_synthetic_creditcard(10_000, fraud_rate=0.02)
    s = split_data(df, strategy="stratified")
    for part in (s.train, s.val, s.test):
        assert part[TARGET].mean() == pytest.approx(df[TARGET].mean(), abs=0.004)


def test_single_class_split_is_rejected():
    df = make_synthetic_creditcard(2000)
    df[TARGET] = 0
    df.loc[0, TARGET] = 1  # a single fraud can't be in every split
    with pytest.raises(ValueError):
        split_data(df, strategy="time")


def test_load_data_drops_duplicates(tmp_path):
    df = make_synthetic_creditcard(300)
    pd.concat([df, df.iloc[:10]]).to_csv(tmp_path / "cc.csv", index=False)
    loaded, info = load_data(tmp_path / "cc.csv")
    assert info["duplicates_dropped"] == 10 and len(loaded) == 300
    assert list(loaded.columns) == [*FEATURE_COLS, TARGET]


def test_missing_file_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data"):
        load_data(tmp_path / "nope.csv")
