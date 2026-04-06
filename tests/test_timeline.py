# test_timeline.py

import pandas as pd
import pytest

from olympoly.timeline import (
    participation_trends,
    medal_trends,
    sport_popularity
)


# -------------------------
# Fixture (replaces CSV)
# -------------------------

@pytest.fixture
def sample_df():
    """Small sample Olympics dataset"""
    return pd.DataFrame({
        "ID": [1, 2, 3, 4, 1, 2],
        "Gender": ["M", "F", "M", "F", "M", "F"],
        "Team": ["USA", "USA", "CAN", "CAN", "USA", "USA"],
        "Sport": ["Swimming", "Running", "Running", "Swimming", "Swimming", "Running"],
        "Medal": ["Gold", None, "Silver", None, "Gold", "Bronze"],
        "Year": [2000, 2000, 2004, 2004, 2008, 2008]
    })


# -------------------------
# Tests
# -------------------------

def test_participation_trends(sample_df):
    result = participation_trends(sample_df, by_gender=True, plot=False)

    assert not result.empty
    assert result.index.is_monotonic_increasing
    assert any(col in result.columns for col in ["M", "F"])


def test_medal_trends(sample_df):
    result = medal_trends(sample_df, entity="Team", top_n=2, plot=False)

    assert not result.empty
    assert result.index.is_monotonic_increasing
    assert result.shape[1] <= 2


def test_sport_popularity(sample_df):
    result = sport_popularity(sample_df, plot=False)

    assert not result.empty
    assert result.index.is_monotonic_increasing
    assert result.shape[1] > 0