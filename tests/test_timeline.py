# test_timeline.py

import pandas as pd
import pytest

from olympoly.timeline import participation_trends, medal_trends, sport_popularity


# -------------------------
# Fixture
# -------------------------

@pytest.fixture
def sample_df():
    """Small dataset for testing timeline functions"""
    return pd.DataFrame({
        "ID": [1, 2, 3, 4, 1, 2],
        "Gender": ["M", "F", "M", "F", "M", "F"],
        "Team": ["USA", "USA", "CAN", "CAN", "USA", "USA"],
        "Sport": ["Swimming", "Running", "Running", "Swimming", "Swimming", "Running"],
        "Medal": ["Gold", None, "Silver", None, "Gold", "Bronze"],
        "Year": [2000, 2000, 2004, 2004, 2008, 2008]
    })


# -------------------------
# Tests: participation_trends
# -------------------------

def test_participation_trends_by_gender(sample_df):
    result = participation_trends(sample_df, by_gender=True, plot=False)

    assert not result.empty
    assert result.index.is_monotonic_increasing
    assert ("M" in result.columns) or ("F" in result.columns)


def test_participation_trends_no_gender(sample_df):
    result = participation_trends(sample_df, by_gender=False, plot=False)

    assert not result.empty
    assert "Count" in result.columns


def test_participation_trends_missing_columns():
    df = pd.DataFrame({"Year": [2000, 2004]})  # Missing ID

    with pytest.raises(ValueError):
        participation_trends(df)


# -------------------------
# Tests: medal_trends
# -------------------------

def test_medal_trends_basic(sample_df):
    result = medal_trends(sample_df, entity="Team", top_n=2, plot=False)

    assert not result.empty
    assert result.shape[1] <= 2


def test_medal_trends_missing_columns(sample_df):
    bad_df = sample_df.drop(columns=["Medal"])

    with pytest.raises(ValueError):
        medal_trends(bad_df)


# -------------------------
# Tests: sport_popularity
# -------------------------

def test_sport_popularity_basic(sample_df):
    result = sport_popularity(sample_df, plot=False)

    assert not result.empty
    assert result.shape[1] > 0


def test_sport_popularity_missing_columns(sample_df):
    bad_df = sample_df.drop(columns=["Sport"])

    with pytest.raises(ValueError):
        sport_popularity(bad_df)