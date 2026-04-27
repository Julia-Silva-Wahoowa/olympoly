import pandas as pd
import numpy as np
import pytest

from olympoly.performance import country_efficiency, efficiency_trends

# -------------------------
# Fixtures
# -------------------------


@pytest.fixture
def sample_data():
    """
    Mock dataset for testing pure functions.
    
    Since performance.py no longer handles data loading, we just pass 
    this DataFrame directly into the functions to ensure deterministic tests.
    """
    return pd.DataFrame({
        'ID': [1, 2, 3, 4, 5, 6],
        'Team': ['USA', 'USA', 'China', 'China', 'China', 'USA'],
        'Year': [2000, 2000, 2000, 2004, 2004, 2004],
        'Medal': ['Gold', None, 'Silver', 'Gold', None, 'Bronze'],
        'Age': [23, 25, 24, 26, 27, 28],
        'Height': [180, 175, 170, 165, 160, 185],
        'Weight': [75, 70, 65, 60, 55, 80],
        'Sport': ['Swimming', 'Swimming', 'Gymnastics', 'Gymnastics', 'Gymnastics', 'Athletics']
    })


# -------------------------
# Tests for country_efficiency
# -------------------------
def test_country_efficiency_basic(sample_data):
    """
    Test that country_efficiency returns a valid output structure.

    Ensures:
    - Output contains an 'Efficiency' column
    - Efficiency values are non-negative
    - Function executes without plotting errors when plot=False
    """
    result = country_efficiency(sample_data, min_athletes=1, plot=False)

    assert 'Efficiency' in result.columns
    assert (result['Efficiency'] >= 0).all()


def test_country_efficiency_calculation(sample_data):
    """
    Validate correctness of efficiency calculation.

    Confirms:
    - Efficiency = medals / unique athletes per country
    - Uses known dataset where both USA and China should equal 2/3
    """
    result = country_efficiency(sample_data, min_athletes=1, plot=False)

    usa_eff = result.loc['USA', 'Efficiency']
    china_eff = result.loc['China', 'Efficiency']

    # USA: 3 athletes, 2 medals -> 2/3
    # China: 3 athletes, 2 medals -> 2/3
    assert np.isclose(usa_eff, 2/3)
    assert np.isclose(china_eff, 2/3)


def test_country_efficiency_min_athletes_filter(sample_data):
    """
    Ensure min_athletes parameter correctly filters countries.

    Verifies:
    - Countries with fewer than the threshold are excluded
    - When threshold exceeds all countries, result is empty
    """
    result = country_efficiency(sample_data, min_athletes=10, plot=False)

    assert result.empty


def test_country_efficiency_zero_athletes():
    """
    Ensure efficiency calculation handles division-by-zero safely.
    """
    df = pd.DataFrame({'ID': [], 'Team': [], 'Year': [], 'Medal': []})

    result = country_efficiency(df, min_athletes=0, plot=False)

    assert result.empty or (result['Efficiency'] == 0).all()


def test_country_efficiency_sorted(sample_data):
    """
    Ensure countries are sorted in descending order of efficiency.
    """
    result = country_efficiency(sample_data, min_athletes=1, plot=False)

    efficiencies = result['Efficiency'].values
    assert all(efficiencies[i] >= efficiencies[i+1]
               for i in range(len(efficiencies)-1))


def test_unique_athlete_count():
    """
    Ensure athletes are counted uniquely by ID, not by row count.
    """
    df = pd.DataFrame({
        'ID': [1, 1, 2, 2],  # duplicates
        'Team': ['USA', 'USA', 'USA', 'USA'],
        'Year': [2000, 2000, 2000, 2000],
        'Medal': ['Gold', 'Gold', None, None]
    })

    result = country_efficiency(df, min_athletes=1, plot=False)

    # Unique athletes = 2, medals = 2 -> efficiency = 1.0
    assert np.isclose(result.loc['USA', 'Efficiency'], 1.0)


# -------------------------
# Tests for efficiency_trends
# -------------------------
def test_efficiency_trends_structure(sample_data):
    """
    Verify output structure of efficiency_trends.
    """
    trends = efficiency_trends(sample_data, country="USA", plot=False)

    assert 'Efficiency' in trends.columns
    assert 'Athletes' in trends.columns
    assert 'Medals' in trends.columns


def test_efficiency_trends_values(sample_data):
    """
    Validate correctness of yearly efficiency calculation.
    """
    trends = efficiency_trends(sample_data, country="USA", plot=False)

    # Year 2000: 2 athletes, 1 medal -> 0.5
    assert np.isclose(trends.loc[2000, 'Efficiency'], 0.5)


def test_efficiency_trends_no_country(sample_data):
    """
    Test behavior when requested country does not exist.
    """
    trends = efficiency_trends(sample_data, country="Nonexistent", plot=False)

    assert trends.empty or (trends['Efficiency'] == 0).all()


def test_efficiency_trends_empty_df():
    """
    Ensure efficiency_trends handles empty input without crashing.
    """
    df = pd.DataFrame(columns=['ID', 'Team', 'Year', 'Medal'])

    trends = efficiency_trends(df, country="USA", plot=False)

    assert trends.empty


# -------------------------
# General Output Tests
# -------------------------
def test_no_plot_flag(sample_data):
    """
    Ensure functions run without plotting when plot=False.
    """
    # If plotting was triggered, this would likely stall the test suite
    country_efficiency(sample_data, plot=False)
    efficiency_trends(sample_data, country="USA", plot=False)
