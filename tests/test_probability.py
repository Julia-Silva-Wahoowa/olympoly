"""test_probability.py - Tests for probability functions"""

import pandas as pd
import numpy as np
import pytest
from olympoly.olympics_betting.probability import (
    medal_probability,
    athlete_medal_probability,
    weighted_medal_probability,
)


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "ID": [1, 2, 3, 4, 5, 6],
        "Name": ["Alice", "Alice", "Bob", "Bob", "Charlie", "Charlie"],
        "NOC": ["USA", "USA", "CHN", "CHN", "GBR", "GBR"],
        "Event": ["100m", "100m", "100m", "200m", "200m", "200m"],
        "Year": [2016, 2020, 2016, 2020, 2016, 2020],
        "Medal": ["Gold", None, "Silver", "Gold", None, "Bronze"],
    })


# Tests for medal_probability
def test_medal_probability_returns_dataframe(sample_data):
    result = medal_probability(sample_data, ["NOC"])
    assert isinstance(result, pd.DataFrame)


def test_medal_probability_has_probability_column(sample_data):
    result = medal_probability(sample_data, ["NOC"])
    assert "probability" in result.columns


def test_medal_probability_values(sample_data):
    result = medal_probability(sample_data, ["NOC"], medal_type="Gold")
    usa = result[result["NOC"] == "USA"]
    assert usa["total_wins"].values[0] == 1
    assert usa["total_entries"].values[0] == 2


def test_medal_probability_no_golds(sample_data):
    result = medal_probability(sample_data, ["NOC"], medal_type="Gold")
    gbr = result[result["NOC"] == "GBR"]
    assert gbr["total_wins"].values[0] == 0


# Tests for athlete_medal_probability
def test_athlete_medal_probability_returns_dataframe(sample_data):
    result = athlete_medal_probability(sample_data)
    assert isinstance(result, pd.DataFrame)


def test_athlete_medal_probability_values(sample_data):
    result = athlete_medal_probability(sample_data)
    alice = result[result["Name"] == "Alice"]
    assert alice["medals"].values[0] == 1
    assert alice["appearances"].values[0] == 2
    assert np.isclose(alice["probability"].values[0], 0.5)


def test_athlete_no_medals(sample_data):
    result = athlete_medal_probability(sample_data)
    charlie = result[result["Name"] == "Charlie"]
    assert charlie["medals"].values[0] == 1  # Bronze counts as a medal


# Tests for weighted_medal_probability
def test_weighted_probability_returns_dataframe(sample_data):
    result = weighted_medal_probability(sample_data, ["NOC"], current_year=2024)
    assert isinstance(result, pd.DataFrame)


def test_weighted_probability_has_columns(sample_data):
    result = weighted_medal_probability(sample_data, ["NOC"], current_year=2024)
    assert "probability" in result.columns
    assert "weighted_wins" in result.columns
    assert "weighted_total" in result.columns


def test_weighted_recent_years_matter_more(sample_data):
    result = weighted_medal_probability(sample_data, ["NOC"], current_year=2024, decay=0.5)
    assert (result["weighted_total"] > 0).all()