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
    """Create a sample market data frame for testing validation, normalization, merging, and latest-price logic"""
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
    """Verify that valid market data passes validation without raising an error"""
    validate_market_data(sample_df)


def test_validate_market_data_invalid_price(sample_df):
    """verify that validation raises an ValueError when a price falls outside the valid probability range"""
    bad_df = sample_df.copy()
    bad_df.loc[0, "price"] = 1.5

    with pytest.raises(ValueError):
        validate_market_data(bad_df)


def test_validate_market_data_missing_column(sample_df):
    """Verify that validation raises ValueError when a required column is missing"""
    bad_df = sample_df.drop(columns=["price"])

    with pytest.raises(ValueError):
        validate_market_data(bad_df)


# -------------------------
# Tests: normalize_prices
# -------------------------

def test_normalize_prices(sample_df):
    """Verify that normalized prices sum to 1.0 within each event and timestamp group"""
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
<<<<<<< HEAD
    """Merged dataset should double in size"""
=======
    """verify that mergining two market data frames combines their rows as expected"""
>>>>>>> 55eb3f4 (created docstrings for test_data_input, test_historical_model, test_market data, and test_ market_vs_model)
    merged = merge_market_data(sample_df, sample_df)

    assert len(merged) == len(sample_df) * 2


# -------------------------
# Tests: get_latest_prices
# -------------------------

def test_get_latest_prices(sample_df):
<<<<<<< HEAD
    """Should return latest rows per event/outcome"""
=======
    """verify that the latest prices function returns only the most recent row for each outcome"""
>>>>>>> 55eb3f4 (created docstrings for test_data_input, test_historical_model, test_market data, and test_ market_vs_model)
    df_copy = sample_df.copy()
    df_copy["timestamp"] = pd.to_datetime(df_copy["timestamp"])

    latest = get_latest_prices(df_copy)

    assert len(latest) == 2
    assert set(latest["outcome"]) == {"YES", "NO"}


def test_get_latest_prices_correct_date(sample_df):
<<<<<<< HEAD
    """Ensure latest timestamp is selected"""
=======
    """verify that the latest prices all come from the most recent timestamp in the dataset"""
>>>>>>> 55eb3f4 (created docstrings for test_data_input, test_historical_model, test_market data, and test_ market_vs_model)
    df_copy = sample_df.copy()
    df_copy["timestamp"] = pd.to_datetime(df_copy["timestamp"])

    latest = get_latest_prices(df_copy)

    assert all(latest["timestamp"] == pd.Timestamp("2024-07-02"))