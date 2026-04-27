# test_historical_model.py

import pandas as pd
import numpy as np
import pytest

from olympoly.olympics_betting.historical_model import (
    country_medal_probability,
    country_gold_probability,
    sport_medal_probability,
    rolling_country_performance
)


# -------------------------
# Fixture
# -------------------------

@pytest.fixture
def sample_df():
    """Small dataset for testing"""
    return pd.DataFrame({
        "Team": ["USA", "USA", "USA", "China", "China", "China"],
        "Sport": ["Swimming", "Swimming", "Athletics", "Gymnastics", "Gymnastics", "Athletics"],
        "Year": [2000, 2000, 2004, 2000, 2004, 2004],
        "Medal": ["Gold", None, "Silver", "Gold", None, None]
    })


# -------------------------
# Tests: country probabilities
# -------------------------

def test_country_medal_probability(sample_df):
    """Verify medal probability for a country is computed correctly"""
    prob = country_medal_probability(sample_df, "USA")

    # USA: 3 entries, 2 medals → 2/3
    assert np.isclose(prob, 2/3)


def test_country_gold_probability(sample_df):
    """Verify gold medal probability for a counttry is computed correctly"""
    prob = country_gold_probability(sample_df, "USA")

    # USA: 3 entries, 1 gold → 1/3
    assert np.isclose(prob, 1/3)


def test_country_probability_no_data(sample_df):
    """Verify gold medal probability returns 0.0 when the country is absent"""
    prob = country_medal_probability(sample_df, "Nonexistent")

    assert prob == 0.0


# -------------------------
# Tests: sport probability
# -------------------------

def test_sport_medal_probability(sample_df):
    """verify medal probability for a sport is computed correctly"""
    prob = sport_medal_probability(sample_df, "Swimming")

    # Swimming: 2 entries, 1 medal → 1/2
    assert np.isclose(prob, 0.5)


def test_sport_probability_no_data(sample_df):
    """Verify sport medal proability returns 0.0 when the sport is absent"""
    prob = sport_medal_probability(sample_df, "Boxing")

    assert prob == 0.0


# -------------------------
# Tests: rolling performance
# -------------------------

def test_rolling_country_performance_structure(sample_df):
    """Verify rolling country performance returns a non-emptty data frame with expected columns"""
    result = rolling_country_performance(sample_df, "USA", window=2)

    assert not result.empty
    assert "Medal_Probability" in result.columns
    assert "Rolling_Probability" in result.columns


def test_rolling_country_performance_values(sample_df):
    """verify rolling country performance computes expected probibility values"""
    result = rolling_country_performance(sample_df, "USA", window=2)

    # Year 2000: 2 athletes, 1 medal → 0.5
    assert np.isclose(result.loc[2000, "Medal_Probability"], 0.5)


def test_rolling_country_performance_no_country(sample_df):
    """verify rolling country performance returns and empty dataframe for an unkown country"""
    result = rolling_country_performance(sample_df, "Nonexistent")

    assert result.empty

def test_country_medal_probability_missing_columns(sample_df):
    """Ensure country_medal_probability raises an error when required columns are missing from the input DataFrame"""

    bad_df = sample_df.drop(columns=["Medal"])

    with pytest.raises(KeyError):
        country_medal_probability(bad_df, "USA")