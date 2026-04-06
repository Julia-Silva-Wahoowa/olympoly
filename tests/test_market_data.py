# test_market_data.py

import pandas as pd
import pytest

from olympoly.olympics_betting.market_data import (
    validate_market_data,
    normalize_prices,
    merge_market_data,
    get_latest_prices
)


# -------------------------
# Fixture (replaces hardcoded data)
# -------------------------

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "market": ["kalshi", "kalshi", "kalshi", "kalshi"],
        "event": ["USA_gold", "USA_gold", "USA_gold", "USA_gold"],
        "outcome": ["YES", "NO", "YES", "NO"],
        "price": [0.6, 0.4, 0.65, 0.35],
        "timestamp": [
            "2024-07-01",
            "2024-07-01",
            "2024-07-02",
            "2024-07-02",
        ]
    })


# -------------------------
# Tests
# -------------------------

def test_validate_market_data(sample_df):
    validate_market_data(sample_df)


def test_validate_market_data_invalid_price(sample_df):
    bad_df = sample_df.copy()
    bad_df.loc[0, "price"] = 1.5

    with pytest.raises(ValueError):
        validate_market_data(bad_df)


def test_validate_market_data_missing_column(sample_df):
    bad_df = sample_df.drop(columns=["price"])

    with pytest.raises(ValueError):
        validate_market_data(bad_df)


def test_normalize_prices(sample_df):
    result = normalize_prices(sample_df)

    grouped = result.groupby(["event", "timestamp"])["normalized_price"].sum()

    for val in grouped:
        assert abs(val - 1.0) < 1e-6


def test_merge_market_data(sample_df):
    merged = merge_market_data(sample_df, sample_df)

    assert len(merged) == len(sample_df) * 2


def test_get_latest_prices(sample_df):
    df_copy = sample_df.copy()
    df_copy["timestamp"] = pd.to_datetime(df_copy["timestamp"])

    latest = get_latest_prices(df_copy)

    assert len(latest) == 2
    assert set(latest["outcome"]) == {"YES", "NO"}


def test_get_latest_prices_correct_date(sample_df):
    df_copy = sample_df.copy()
    df_copy["timestamp"] = pd.to_datetime(df_copy["timestamp"])

    latest = get_latest_prices(df_copy)

    assert all(latest["timestamp"] == pd.Timestamp("2024-07-02"))