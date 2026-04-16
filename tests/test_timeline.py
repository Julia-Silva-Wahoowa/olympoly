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
    
    """
    Provides a small synthetic Olympics dataset for timeline analysis.

    Includes:
    - Multiple years
    - Multiple teams and sports
    - Mixed medal outcomes
    - Gender variation for grouping tests
    """
    
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
    
    """
    Test participation_trends with gender breakdown enabled.

    Ensures:
    - Output is not empty
    - Years are sorted in increasing order
    - Gender columns (M/F) are present
    """
    
    result = participation_trends(sample_df, by_gender=True, plot=False)

    assert not result.empty
    assert result.index.is_monotonic_increasing
    assert any(col in result.columns for col in ["M", "F"])


def test_medal_trends(sample_df):
    
    """
    Test medal_trends with team-level aggregation.

    Ensures:
    - Output is not empty
    - Years are sorted in increasing order
    - Number of columns is limited to top_n
    """
    
    result = medal_trends(sample_df, entity="Team", top_n=2, plot=False)

    assert not result.empty
    assert result.index.is_monotonic_increasing
    assert result.shape[1] <= 2


def test_sport_popularity(sample_df):
    
    """
    Test sport_popularity aggregation.

    Ensures:
    - Output is non-empty
    - Years are sorted
    - At least one sport column exists
    """
    
    result = sport_popularity(sample_df, plot=False)

    assert not result.empty
    assert result.index.is_monotonic_increasing
    assert result.shape[1] > 0
    
    
def test_participation_trends_missing_gender():
    """
    Ensure participation_trends raises ValueError when by_gender=True
    but no gender column exists.
    """
    df = pd.DataFrame({
        "ID": [1, 2],
        "Year": [2000, 2004]
    })

    with pytest.raises(ValueError):
        participation_trends(df, by_gender=True, plot=False)
        
def test_timeline_no_plot(sample_df):
   
    """
    Ensure timeline functions do not error when plot=False.
    """
    
    participation_trends(sample_df, plot=False)
    medal_trends(sample_df, plot=False)
    sport_popularity(sample_df, plot=False)
    
def test_participation_unique_ids():
    
    """
    Ensure participation counts unique athletes, not duplicate rows.
    """
    
    df = pd.DataFrame({
        "ID": [1, 1, 2],
        "Gender": ["M", "M", "F"],
        "Year": [2000, 2000, 2000]
    })

    result = participation_trends(df, by_gender=False, plot=False)

    assert result.loc[2000, 'Count'] == 2