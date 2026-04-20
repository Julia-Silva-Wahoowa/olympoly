import pandas as pd
import numpy as np
import pytest

from olympoly.performance import prepare_data, country_efficiency, efficiency_trends


# -------------------------
# Fixtures
# -------------------------
@pytest.fixture
def sample_data(monkeypatch):
    """Mock dataset to avoid relying on real data.
    
    Replaces the real `load_data()` function using monkeypatch to ensure:
    - Tests do not depend on external files or I/O
    - Data is deterministic and small
    - Edge cases (missing medals, string numerics) are included
    """
    
    def mock_load_data():
        return pd.DataFrame({
            'ID': [1, 2, 3, 4, 5, 6],
            'Team': ['USA', 'USA', 'China', 'China', 'China', 'USA'],
            'Year': [2000, 2000, 2000, 2004, 2004, 2004],
            'Medal': ['Gold', None, 'Silver', 'Gold', None, 'Bronze'],
            'Age': ['23', '25', '24', '26', '27', '28'],
            'Height': ['180', '175', '170', '165', '160', '185'],
            'Weight': ['75', '70', '65', '60', '55', '80'],
            'Sport': ['Swimming', 'Swimming', 'Gymnastics', 'Gymnastics', 'Gymnastics', 'Athletics']
        })

    # ✅ FIXED monkeypatch path
    monkeypatch.setattr("olympoly.performance.load_data", mock_load_data)

    return mock_load_data()


# -------------------------
# Tests for prepare_data
# -------------------------
def test_prepare_data_types(sample_data):
    
    """
    Verify that prepare_data correctly converts numeric columns to numeric dtypes.

    Specifically checks that:
    - Year, Age, Height, and Weight are no longer strings
    - Invalid parsing (if any) is coerced rather than crashing
    """
    
    df = prepare_data()

    assert pd.api.types.is_numeric_dtype(df['Year'])
    assert pd.api.types.is_numeric_dtype(df['Age'])
    assert pd.api.types.is_numeric_dtype(df['Height'])
    assert pd.api.types.is_numeric_dtype(df['Weight'])


def test_prepare_data_no_missing_keys(sample_data):
    
    """
    Ensure that prepare_data removes rows with missing critical identifiers.

    Confirms that:
    - ID, Team, and Year contain no null values after cleaning
    - The dropna subset logic is working correctly
    """
    
    df = prepare_data()

    assert df['ID'].isnull().sum() == 0
    assert df['Team'].isnull().sum() == 0
    assert df['Year'].isnull().sum() == 0


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
    
    df = prepare_data()

    result = country_efficiency(df, min_athletes=1, plot=False)

    assert 'Efficiency' in result.columns
    assert (result['Efficiency'] >= 0).all()


def test_country_efficiency_calculation(sample_data):
    
    """
    Validate correctness of efficiency calculation.

    Confirms:
    - Efficiency = medals / unique athletes per country
    - Uses known dataset where both USA and China should equal 2/3
    - Floating-point comparison handled via np.isclose
    """
    
    df = prepare_data()

    result = country_efficiency(df, min_athletes=1, plot=False)

    usa_eff = result.loc['USA', 'Efficiency']
    china_eff = result.loc['China', 'Efficiency']

    # USA: 3 athletes, 2 medals → 2/3
    # China: 3 athletes, 2 medals → 2/3
    assert np.isclose(usa_eff, 2/3)
    assert np.isclose(china_eff, 2/3)


def test_country_efficiency_min_athletes_filter(sample_data):
    
    """
    Ensure min_athletes parameter correctly filters countries.

    Verifies:
    - Countries with fewer than the threshold are excluded
    - When threshold exceeds all countries, result is empty
    """
    
    df = prepare_data()

    result = country_efficiency(df, min_athletes=10, plot=False)

    assert result.empty


# -------------------------
# Tests for efficiency_trends
# -------------------------
def test_efficiency_trends_structure(sample_data):
    
    """
    Verify output structure of efficiency_trends.

    Ensures returned DataFrame includes:
    - 'Efficiency'
    - 'Athletes'
    - 'Medals'
    indexed by Year
    """
    
    df = prepare_data()

    trends = efficiency_trends(df, country="USA", plot=False)

    assert 'Efficiency' in trends.columns
    assert 'Athletes' in trends.columns
    assert 'Medals' in trends.columns


def test_efficiency_trends_values(sample_data):
    
    """
    Validate correctness of yearly efficiency calculation.

    Uses known case:
    - USA in 2000: 2 athletes, 1 medal → efficiency = 0.5
    """
    
    df = prepare_data()

    trends = efficiency_trends(df, country="USA", plot=False)

    # Year 2000: 2 athletes, 1 medal → 0.5
    assert np.isclose(trends.loc[2000, 'Efficiency'], 0.5)


def test_efficiency_trends_no_country(sample_data):
    
    """
    Test behavior when requested country does not exist.

    Ensures:
    - Function returns empty DataFrame OR
    - Efficiency values are all zero
    - No exceptions are raised
    """
    
    df = prepare_data()

    trends = efficiency_trends(df, country="Nonexistent", plot=False)

    assert trends.empty or (trends['Efficiency'] == 0).all()
    
    
def test_country_efficiency_zero_athletes():
    
    """
    Ensure efficiency calculation handles division-by-zero safely.

    Countries with zero athletes should not produce NaN or inf values.
    """
    
    df = pd.DataFrame({
        'ID': [],
        'Team': [],
        'Year': [],
        'Medal': []
    })

    result = country_efficiency(df, min_athletes=0, plot=False)

    assert result.empty or (result['Efficiency'] == 0).all()
    
    
def test_efficiency_trends_empty_df():
    
    """
    Ensure efficiency_trends handles empty input without crashing.
    """
    df = pd.DataFrame(columns=['ID', 'Team', 'Year', 'Medal'])

    trends = efficiency_trends(df, country="USA", plot=False)

    assert trends.empty
    
def test_country_efficiency_sorted(sample_data):
    
    """
    Ensure countries are sorted in descending order of efficiency.
    """
    df = prepare_data()

    result = country_efficiency(df, min_athletes=1, plot=False)

    efficiencies = result['Efficiency'].values

    assert all(efficiencies[i] >= efficiencies[i+1] for i in range(len(efficiencies)-1))
    
def test_no_plot_flag(sample_data):
    
    """
    Ensure functions run without plotting when plot=False.
    """
    
    df = prepare_data()

    country_efficiency(df, plot=False)
    efficiency_trends(df, country="USA", plot=False)
    
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

    # Unique athletes = 2, medals = 2 → efficiency = 1.0
    assert np.isclose(result.loc['USA', 'Efficiency'], 1.0)