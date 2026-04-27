# test_simulation.py

import pandas as pd
import pytest

from olympoly.performance import country_efficiency
from olympoly.olympics_betting.simulation import (
    simulate_market_strategy,
    simulate_edge_strategy
)


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
# simulate_market_strategy
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


def test_simulate_market_strategy_summary_keys(sample_df):
    _, summary = simulate_market_strategy(sample_df)

    expected_keys = {
        "final_bankroll",
        "total_return",
        "roi",
        "num_bets",
        "win_rate"
    }

    assert expected_keys.issubset(summary.keys())


def test_simulate_market_strategy_edge_filter(sample_df):
    results, summary = simulate_market_strategy(sample_df, edge_threshold=1.0)

    assert results.empty
    assert summary["num_bets"] == 0


def test_simulate_market_strategy_missing_columns(sample_df):
    bad_df = sample_df.drop(columns=["price"])

    with pytest.raises(ValueError):
        simulate_market_strategy(bad_df)


def test_bankroll_updates(sample_df):
    results, summary = simulate_market_strategy(sample_df, seed=42)

    if not results.empty:
        assert results.iloc[-1]["bankroll"] == summary["final_bankroll"]


def test_simulation_reproducibility(sample_df):
    r1, s1 = simulate_market_strategy(sample_df, seed=42)
    r2, s2 = simulate_market_strategy(sample_df, seed=42)

    pd.testing.assert_frame_equal(r1, r2)
    assert s1 == s2


def test_bet_size_effect(sample_df):
    _, summary_small = simulate_market_strategy(sample_df, bet_size=0.1, seed=42)
    _, summary_large = simulate_market_strategy(sample_df, bet_size=0.5, seed=42)

    assert abs(summary_large["total_return"]) >= abs(summary_small["total_return"])


def test_zero_bankroll():
    df = pd.DataFrame({
        "event": ["A"],
        "price": [0.4],
        "model_prob": [0.6]
    })

    _, summary = simulate_market_strategy(df, bankroll=0)

    assert summary["roi"] == 0


def test_bet_direction_logic():
    df = pd.DataFrame({
        "event": ["A", "B"],
        "price": [0.4, 0.7],
        "model_prob": [0.6, 0.5]
    })

    results, _ = simulate_market_strategy(df, edge_threshold=0.0, seed=42)

    directions = results["direction"].values

    assert "YES" in directions
    assert "NO" in directions


def test_pnl_values(sample_df):
    results, _ = simulate_market_strategy(sample_df, seed=42)

    if not results.empty:
        for _, row in results.iterrows():
            assert row["pnl"] in [row["bet_amount"], -row["bet_amount"]]


def test_empty_dataframe():
    df = pd.DataFrame(columns=["event", "price", "model_prob"])

    results, summary = simulate_market_strategy(df)

    assert results.empty
    assert summary["num_bets"] == 0
    assert summary["roi"] == 0


def test_market_strategy_summary_sanity(sample_df):
    """
    Validate internal consistency of summary output.
    """

    results, summary = simulate_market_strategy(
        sample_df,
        bankroll=100,
        bet_size=0.1,
        edge_threshold=0.0,
        seed=42
    )

    # num_bets should match results length
    assert summary["num_bets"] == len(results)

    # bankroll math consistency
    assert summary["final_bankroll"] == pytest.approx(
        100 + summary["total_return"]
    )

    # ROI formula
    assert summary["roi"] == pytest.approx(
        summary["total_return"] / 100
    )

    # bankroll should not be negative
    assert summary["final_bankroll"] >= 0


# -------------------------
# simulate_edge_strategy
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


def test_simulate_edge_strategy_missing_columns():
    df = pd.DataFrame({
        "event": ["A"],
        "price": [0.4]
    })

    with pytest.raises(KeyError):
        simulate_edge_strategy(df)


def test_edge_strategy_accuracy_bounds(sample_df):
    result = simulate_edge_strategy(sample_df)

    assert 0 <= result["accuracy"] <= 1


def test_edge_strategy_no_bets():
    df = pd.DataFrame({
        "event": ["A"],
        "price": [0.5],
        "model_prob": [0.51]
    })

    result = simulate_edge_strategy(df, threshold=1.0)

    assert result["total_bets"] == 0
    assert result["accuracy"] == 0


# -------------------------
# performance integration
# -------------------------

def test_large_dataset():
    df = pd.DataFrame({
        "ID": range(10000),
        "Team": ["USA"] * 10000,
        "Year": [2000] * 10000,
        "Medal": [None] * 10000
    })

    result = country_efficiency(df, min_athletes=1, plot=False)

    assert not result.empty