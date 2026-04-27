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
    """Create a sample market data frame for testing"""
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
    """Verify valid data passes validation"""
    validate_market_data(sample_df)


def test_validate_market_data_invalid_price(sample_df):
    """Verify error raised for invalid price"""
    bad_df = sample_df.copy()
    bad_df.loc[0, "price"] = 1.5

    with pytest.raises(ValueError):
        validate_market_data(bad_df)


def test_validate_market_data_missing_column(sample_df):
    """Verify error raised when required column is missing"""
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
    """Check normalized_price column exists"""
    result = normalize_prices(sample_df)

    assert "normalized_price" in result.columns


# -------------------------
# Tests: merge_market_data
# -------------------------

def test_merge_market_data(sample_df):
    """Verify merging doubles dataset size"""
    merged = merge_market_data(sample_df, sample_df)

    assert len(merged) == len(sample_df) * 2


# -------------------------
# Tests: get_latest_prices
# -------------------------

def test_get_latest_prices(sample_df):
    """Verify latest row per outcome is returned"""
    df_copy = sample_df.copy()
    df_copy["timestamp"] = pd.to_datetime(df_copy["timestamp"])

    latest = get_latest_prices(df_copy)

    assert len(latest) == 2
    assert set(latest["outcome"]) == {"YES", "NO"}


def test_get_latest_prices_correct_date(sample_df):
    """Verify latest timestamp is selected"""
    df_copy = sample_df.copy()
    df_copy["timestamp"] = pd.to_datetime(df_copy["timestamp"])

    latest = get_latest_prices(df_copy)

    assert all(latest["timestamp"] == pd.Timestamp("2024-07-02"))

def test_validate_market_data_price_boundaries():
    """Verify prices at the valid boundaries 0 and 1 pass validation"""

    df = pd.DataFrame({
        "market": ["kalshi", "kalshi"],
        "event": ["USA_gold", "USA_gold"],
        "outcome": ["YES", "NO"],
        "price": [1.0, 0.0],
        "timestamp": ["2024-07-01", "2024-07-01"]
    })

    validate_market_data(df)