# test_simulation.py

from olympoly.performance import country_efficiency
import pandas as pd
import pytest

from olympoly.olympics_betting.simulation import simulate_market_strategy, simulate_edge_strategy


# -------------------------
# Fixture
# -------------------------

@pytest.fixture
def sample_df():
    
    """
    Sample dataset for simulation tests
    
    Includes:
    - Multiple events
    - Varying market vs model probabilities
    - Both positive and negative edges

    Designed to trigger betting decisions under default parameters.
    """
    
    return pd.DataFrame({
        "event": ["A", "B", "C", "D"],
        "price": [0.4, 0.6, 0.5, 0.7],
        "model_prob": [0.6, 0.5, 0.55, 0.6]
    })


# -------------------------
# Tests: simulate_market_strategy
# -------------------------

def test_simulate_market_strategy_runs(sample_df):
    
    """
    Ensure simulate_market_strategy executes without errors.

    Verifies:
    - Function runs end-to-end
    - Returns correct types (DataFrame, dict)
    """
    
    results, summary = simulate_market_strategy(sample_df)

    assert isinstance(results, pd.DataFrame)
    assert isinstance(summary, dict)


def test_simulate_market_strategy_columns(sample_df):
   
    """
    Validate structure of results DataFrame.

    Ensures:
    - All expected columns are present when bets occur
    - Output schema matches simulation design
    """
    
    results, _ = simulate_market_strategy(sample_df)

    if not results.empty:
        expected_cols = {
            "event", "market_prob", "model_prob", "edge",
            "direction", "bet_amount", "win", "pnl", "bankroll"
        }
        assert expected_cols.issubset(results.columns)


def test_simulate_market_strategy_summary(sample_df):
    
    """
    Validate contents of summary output.

    Ensures summary includes:
    - final_bankroll
    - total_return
    - roi
    - num_bets
    - win_rate
    """
    
    _, summary = simulate_market_strategy(sample_df)

    assert "final_bankroll" in summary
    assert "total_return" in summary
    assert "roi" in summary
    assert "num_bets" in summary
    assert "win_rate" in summary


def test_simulate_market_strategy_edge_filter(sample_df):
    
    """
    Ensure edge_threshold correctly filters out bets.

    With a very high threshold:
    - No bets should be placed
    - Results DataFrame should be empty
    - num_bets should be zero
    """
    
    # No bets should occur
    results, summary = simulate_market_strategy(sample_df, edge_threshold=1.0)

    assert results.empty
    assert summary["num_bets"] == 0


def test_simulate_market_strategy_missing_columns(sample_df):
    
    """
    Ensure function raises ValueError when required columns are missing.

    Specifically tests:
    - Missing market probability column ('price')
    """
    
    bad_df = sample_df.drop(columns=["price"])

    with pytest.raises(ValueError):
        simulate_market_strategy(bad_df)


# -------------------------
# Tests: simulate_edge_strategy
# -------------------------

def test_simulate_edge_strategy_runs(sample_df):
    
    """
    Ensure simulate_edge_strategy executes without errors.

    Verifies:
    - Function returns a dictionary
    - No runtime issues occur
    """
    
    result = simulate_edge_strategy(sample_df)

    assert isinstance(result, dict)


def test_simulate_edge_strategy_keys(sample_df):
    
    """
    Validate output structure of edge strategy.

    Ensures:
    - 'accuracy' and 'total_bets' keys are present
    """
    
    result = simulate_edge_strategy(sample_df)

    assert "accuracy" in result
    assert "total_bets" in result


def test_simulate_edge_strategy_threshold(sample_df):
    
    """
    Ensure threshold filtering works correctly.

    With a high threshold:
    - No bets should be evaluated
    - total_bets should be zero
    - accuracy should default to zero
    """
    
    result = simulate_edge_strategy(sample_df, threshold=1.0)

    assert result["total_bets"] == 0
    assert result["accuracy"] == 0
    
    
def test_simulate_edge_strategy_missing_columns():
    
    """
    Ensure simulate_edge_strategy fails when required columns are missing.
    """
    
    df = pd.DataFrame({
        "event": ["A"],
        "price": [0.4]
    })

    with pytest.raises(KeyError):
        simulate_edge_strategy(df)
    
    
def test_bankroll_updates(sample_df):
    
    """
    Ensure bankroll updates correctly after each bet.

    Verifies:
    - Bankroll changes from initial value when bets occur
    - Final bankroll matches last recorded bankroll in results
    """
    
    results, summary = simulate_market_strategy(sample_df, seed=42)

    if not results.empty:
        assert results.iloc[-1]["bankroll"] == summary["final_bankroll"]
        
def test_simulation_reproducibility(sample_df):
    
    """
    Ensure simulation is reproducible given the same random seed.
    """
    
    r1, s1 = simulate_market_strategy(sample_df, seed=42)
    r2, s2 = simulate_market_strategy(sample_df, seed=42)

    pd.testing.assert_frame_equal(r1, r2)
    assert s1 == s2
    
def test_bet_size_effect(sample_df):
    
    """
    Ensure larger bet sizes lead to larger bankroll fluctuations.
    """
    
    _, summary_small = simulate_market_strategy(sample_df, bet_size=0.1, seed=42)
    _, summary_large = simulate_market_strategy(sample_df, bet_size=0.5, seed=42)

    assert abs(summary_large["total_return"]) >= abs(summary_small["total_return"])

def test_zero_bankroll():
    
    """
    Ensure simulation handles zero starting bankroll safely.

    Expected:
    - No meaningful bets placed
    - ROI should be zero (not NaN or inf)
    """
    
    df = pd.DataFrame({
        "event": ["A"],
        "price": [0.4],
        "model_prob": [0.6]
    })

    _, summary = simulate_market_strategy(df, bankroll=0)

    assert summary["roi"] == 0
    
def test_bet_direction_logic():
    
    """
    Ensure betting direction is determined correctly.

    - model_prob > market_prob → YES
    - model_prob < market_prob → NO
    """
    
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
    
    """
    Ensure pnl values are correctly computed as +/- bet_amount.
    """
    
    results, _ = simulate_market_strategy(sample_df, seed=42)

    if not results.empty:
        for _, row in results.iterrows():
            assert row["pnl"] in [row["bet_amount"], -row["bet_amount"]]
            
def test_edge_strategy_accuracy_bounds(sample_df):
    
    """
    Ensure accuracy is always between 0 and 1.
    """
    
    result = simulate_edge_strategy(sample_df)

    assert 0 <= result["accuracy"] <= 1
    
def test_invalid_probabilities():
    
    """
    Ensure simulation handles invalid probability inputs.

    Probabilities should be within [0, 1].
    """
    
    df = pd.DataFrame({
        "event": ["A"],
        "price": [1.2],          # invalid
        "model_prob": [-0.1]     # invalid
    })

    # You may want to enforce ValueError in your code later
    results, summary = simulate_market_strategy(df)

    assert not results.empty  # or change to expect ValueError
    
def test_extreme_probabilities():
    
    """
    Ensure simulation behaves correctly for edge probabilities (0 and 1).
    """
    
    df = pd.DataFrame({
        "event": ["A", "B"],
        "price": [0.0, 1.0],
        "model_prob": [1.0, 0.0]
    })

    results, _ = simulate_market_strategy(df, edge_threshold=0.0)

    assert not results.empty

def test_edge_strategy_no_bets():
    
    """
    Ensure correct behavior when no edges exceed threshold.
    """
    
    df = pd.DataFrame({
        "event": ["A"],
        "price": [0.5],
        "model_prob": [0.51]
    })

    result = simulate_edge_strategy(df, threshold=1.0)

    assert result["total_bets"] == 0
    assert result["accuracy"] == 0
    
def test_large_dataset():
    
    """
    Ensure functions handle larger datasets without failure.
    """
    
    df = pd.DataFrame({
        "ID": range(10000),
        "Team": ["USA"] * 10000,
        "Year": [2000] * 10000,
        "Medal": [None] * 10000
    })

    result = country_efficiency(df, min_athletes=1, plot=False)

    assert not result.empty

def test_empty_dataframe():
    """test for empty input DataFrame which is important for edge case"""
    df = pd.DataFrame(columns=["event", "price", "model_prob"])

    results, summary = simulate_market_strategy(df)

    assert results.empty
    assert summary["num_bets"] == 0
    assert summary["roi"] == 0