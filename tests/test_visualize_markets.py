import pandas as pd
import pytest

from olympoly.olympics_betting.visualize_markets import (
    plot_price_over_time,
    plot_market_comparison,
    plot_latest_snapshot,
    plot_market_vs_model,
    plot_top_edges,
    plot_probability_distribution
)

# -----------------------------
# Fixtures
# -----------------------------

@pytest.fixture
def sample_market_data():
    return pd.DataFrame({
        
        """
    Provides mock market data for plotting functions.

    Includes:
    - Multiple markets (Kalshi, Polymarket)
    - Multiple outcomes
    - Time-series price data
    """
        
        "market": ["kalshi", "kalshi", "polymarket", "polymarket"],
        "event": ["100m_final", "100m_final", "100m_final", "100m_final"],
        "outcome": ["USA", "Jamaica", "USA", "Jamaica"],
        "price": [0.6, 0.4, 0.65, 0.35],
        "timestamp": pd.to_datetime([
            "2024-07-01", "2024-07-01",
            "2024-07-02", "2024-07-02"
        ])
    })


@pytest.fixture
def sample_compare_data():
    
    """
    Provides mock comparison data between market and model probabilities.

    Includes:
    - Probabilities for multiple events
    - Precomputed differences for edge analysis
    """
    
    return pd.DataFrame({
        "event": ["USA", "China", "UK"],
        "market_prob": [0.7, 0.4, 0.2],
        "model_prob": [0.6, 0.5, 0.25],
        "difference": [0.1, -0.1, -0.05]
    })


# -----------------------------
# Tests
# -----------------------------

def test_plot_price_over_time_runs(sample_market_data):
    
    """
    Ensure plot_price_over_time executes without errors for valid input.

    Does not validate visual output, only confirms no runtime exceptions.
    """
    
    # Should run without error
    plot_price_over_time(sample_market_data, "100m_final")


def test_plot_price_over_time_invalid_event(sample_market_data):
    
    """
    Ensure plot_price_over_time raises ValueError for missing event.

    Confirms proper input validation and error handling.
    """
    
    with pytest.raises(ValueError):
        plot_price_over_time(sample_market_data, "nonexistent_event")


def test_plot_market_comparison_runs(sample_market_data):
    
    """
    Ensure plot_market_comparison executes successfully.

    Confirms grouping and plotting logic does not raise errors.
    """
    
    plot_market_comparison(sample_market_data, "100m_final")


def test_plot_latest_snapshot_runs(sample_market_data):
    
    """
    Ensure plot_latest_snapshot executes without error.

    Validates latest timestamp selection logic.
    """
    
    plot_latest_snapshot(sample_market_data, "100m_final")


def test_plot_market_vs_model_runs(sample_compare_data):
    
    """
    Ensure plot_market_vs_model executes with valid input.

    Confirms side-by-side bar plotting works without errors.
    """
    
    plot_market_vs_model(sample_compare_data)


def test_plot_market_vs_model_missing_columns():
    
    """
    Ensure plot_market_vs_model raises ValueError when required columns are missing.

    Specifically tests absence of 'model_prob'.
    """
    
    df = pd.DataFrame({
        "event": ["USA"],
        "market_prob": [0.7]
        # missing model_prob
    })

    with pytest.raises(ValueError):
        plot_market_vs_model(df)


def test_plot_top_edges_runs(sample_compare_data):
    
    """
    Ensure plot_top_edges executes correctly.

    Confirms sorting by absolute difference and plotting top N entries.
    """
    
    plot_top_edges(sample_compare_data, n=2)


def test_plot_top_edges_missing_column():
    
    """
    Ensure plot_top_edges raises ValueError if 'difference' column is missing.
    """
    
    df = pd.DataFrame({
        "event": ["USA"],
        "market_prob": [0.7],
        "model_prob": [0.6]
        # missing difference
    })

    with pytest.raises(ValueError):
        plot_top_edges(df)


def test_plot_probability_distribution_runs(sample_compare_data):
    
    """
    Ensure scatter plot of market vs model probabilities runs without error.
    """
    
    plot_probability_distribution(sample_compare_data)


def test_plot_probability_distribution_missing_columns():
    
    """
    Ensure plot_probability_distribution raises ValueError if required columns are missing.
    """
    
    df = pd.DataFrame({
        "market_prob": [0.7]
        # missing model_prob
    })

    with pytest.raises(ValueError):
        plot_probability_distribution(df)