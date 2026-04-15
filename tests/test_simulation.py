# test_simulation.py

import pandas as pd
import pytest

from olympoly.simulation import simulate_market_strategy, simulate_edge_strategy


# -------------------------
# Fixture
# -------------------------

@pytest.fixture
def sample_df():
    """Sample dataset for simulation tests"""
    return pd.DataFrame({
        "event": ["A", "B", "C", "D"],
        "price": [0.4, 0.6, 0.5, 0.7],
        "model_prob": [0.6, 0.5, 0.55, 0.6]
    })


# -------------------------
# Tests: simulate_market_strategy
# -------------------------

def test_simulate_market_strategy_runs(sample_df):
    results, summary = simulate_market_strategy(sample_df)

    assert isinstance(results, pd.DataFrame)
    assert isinstance(summary, dict)


def test_simulate_market_strategy_columns(sample_df):
    results, _ = simulate_market_strategy(sample_df)

    if not results.empty:
        expected_cols = {
            "event", "market_prob", "model_prob", "edge",
            "direction", "bet_amount", "win", "pnl", "bankroll"
        }
        assert expected_cols.issubset(results.columns)


def test_simulate_market_strategy_summary(sample_df):
    _, summary = simulate_market_strategy(sample_df)

    assert "final_bankroll" in summary
    assert "total_return" in summary
    assert "roi" in summary
    assert "num_bets" in summary
    assert "win_rate" in summary


def test_simulate_market_strategy_edge_filter(sample_df):
    # No bets should occur
    results, summary = simulate_market_strategy(sample_df, edge_threshold=1.0)

    assert results.empty
    assert summary["num_bets"] == 0


def test_simulate_market_strategy_missing_columns(sample_df):
    bad_df = sample_df.drop(columns=["price"])

    with pytest.raises(ValueError):
        simulate_market_strategy(bad_df)


# -------------------------
# Tests: simulate_edge_strategy
# -------------------------

def test_simulate_edge_strategy_runs(sample_df):
    result = simulate_edge_strategy(sample_df)

    assert isinstance(result, dict)


def test_simulate_edge_strategy_keys(sample_df):
    result = simulate_edge_strategy(sample_df)

    assert "accuracy" in result
    assert "total_bets" in result


def test_simulate_edge_strategy_threshold(sample_df):
    result = simulate_edge_strategy(sample_df, threshold=1.0)

    assert result["total_bets"] == 0
    assert result["accuracy"] == 0