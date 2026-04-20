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
# Fixture
# -------------------------

@pytest.fixture
def sample_df():
    """Sample dataset for testing market data functions"""
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
# Tests: validate_market_data
# -------------------------

def test_validate_market_data(sample_df):
    """Should pass for valid data"""
    validate_market_data(sample_df)


def test_validate_market_data_invalid_price(sample_df):
    """Should fail if price is not between 0 and 1"""
    bad_df = sample_df.copy()
    bad_df.loc[0, "price"] = 1.5

    with pytest.raises(ValueError):
        validate_market_data(bad_df)


def test_validate_market_data_missing_column(sample_df):
    """Should fail if required column missing"""
    bad_df = sample_df.drop(columns=["price"])

    with pytest.raises(ValueError):
        validate_market_data(bad_df)


# -------------------------
# Tests: normalize_prices
# -------------------------

def test_normalize_prices(sample_df):
    """Normalized prices should sum to 1"""
    result = normalize_prices(sample_df)

    grouped = result.groupby(["event", "timestamp"])["normalized_price"].sum()

    for val in grouped:
        assert abs(val - 1.0) < 1e-6


def test_normalize_prices_column_exists(sample_df):
    """Check normalized_price column is added"""
    result = normalize_prices(sample_df)

    assert "normalized_price" in result.columns


# -------------------------
# Tests: merge_market_data
# -------------------------

def test_merge_market_data(sample_df):
    """Merged dataset should double in size"""
    merged = merge_market_data(sample_df, sample_df)

    assert len(merged) == len(sample_df) * 2


# -------------------------
# Tests: get_latest_prices
# -------------------------

def test_get_latest_prices(sample_df):
    """Should return latest rows per event/outcome"""
    df_copy = sample_df.copy()
    df_copy["timestamp"] = pd.to_datetime(df_copy["timestamp"])

    latest = get_latest_prices(df_copy)

    assert len(latest) == 2
    assert set(latest["outcome"]) == {"YES", "NO"}


def test_get_latest_prices_correct_date(sample_df):
    """Ensure latest timestamp is selected"""
    df_copy = sample_df.copy()
    df_copy["timestamp"] = pd.to_datetime(df_copy["timestamp"])

    latest = get_latest_prices(df_copy)

    assert all(latest["timestamp"] == pd.Timestamp("2024-07-02"))