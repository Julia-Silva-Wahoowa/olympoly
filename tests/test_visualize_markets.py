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
    # Should run without error
    plot_price_over_time(sample_market_data, "100m_final")


def test_plot_price_over_time_invalid_event(sample_market_data):
    with pytest.raises(ValueError):
        plot_price_over_time(sample_market_data, "nonexistent_event")


def test_plot_market_comparison_runs(sample_market_data):
    plot_market_comparison(sample_market_data, "100m_final")


def test_plot_latest_snapshot_runs(sample_market_data):
    plot_latest_snapshot(sample_market_data, "100m_final")


def test_plot_market_vs_model_runs(sample_compare_data):
    plot_market_vs_model(sample_compare_data)


def test_plot_market_vs_model_missing_columns():
    df = pd.DataFrame({
        "event": ["USA"],
        "market_prob": [0.7]
        # missing model_prob
    })

    with pytest.raises(ValueError):
        plot_market_vs_model(df)


def test_plot_top_edges_runs(sample_compare_data):
    plot_top_edges(sample_compare_data, n=2)


def test_plot_top_edges_missing_column():
    df = pd.DataFrame({
        "event": ["USA"],
        "market_prob": [0.7],
        "model_prob": [0.6]
        # missing difference
    })

    with pytest.raises(ValueError):
        plot_top_edges(df)


def test_plot_probability_distribution_runs(sample_compare_data):
    plot_probability_distribution(sample_compare_data)


def test_plot_probability_distribution_missing_columns():
    df = pd.DataFrame({
        "market_prob": [0.7]
        # missing model_prob
    })

    with pytest.raises(ValueError):
        plot_probability_distribution(df)