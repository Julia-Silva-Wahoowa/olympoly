"""
simulation.py - Betting simulation engine for Olympics prediction markets

Simulates a trading strategy using:
- Market probabilities (prices)
- Model probabilities (historical_model or ML outputs)

NOTE:
Since we don't yet have true outcomes, we simulate outcomes using model probability.
"""

import numpy as np
import pandas as pd


def simulate_market_strategy(
    df,
    model_col="model_prob",
    market_col="price",
    edge_threshold=0.05,
    bet_size=1.0,
    bankroll=100.0,
    seed=42
):
    """
    Simulate betting strategy on prediction markets.

    Parameters:
    - df: DataFrame containing at least:
        ['event', market_col, model_col]
    - model_col: column with model probabilities
    - market_col: column with market probabilities
    - edge_threshold: minimum edge required to place a bet
    - bet_size: fraction of bankroll per bet
    - bankroll: starting capital
    - seed: randomness seed

    Returns:
    - results DataFrame
    - summary dict
    """

    np.random.seed(seed)

    df = df.copy()

    required = {"event", model_col, market_col}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required}")

    results = []
    current_bankroll = bankroll

    for _, row in df.iterrows():

        market_prob = row[market_col]
        model_prob = row[model_col]

        edge = model_prob - market_prob

        # Decide whether to bet
        if abs(edge) < edge_threshold:
            continue

        direction = "YES" if edge > 0 else "NO"

        bet_amount = current_bankroll * bet_size

        # Simulated true outcome (unknown reality)
        true_outcome = np.random.rand() < model_prob

        # Determine win
        if direction == "YES":
            win = true_outcome
        else:
            win = not true_outcome

        # Payout assumption (binary contract: win = +1, lose = -1)
        pnl = bet_amount * (1 if win else -1)

        current_bankroll += pnl

        results.append({
            "event": row["event"],
            "market_prob": market_prob,
            "model_prob": model_prob,
            "edge": edge,
            "direction": direction,
            "bet_amount": bet_amount,
            "win": win,
            "pnl": pnl,
            "bankroll": current_bankroll
        })

    results_df = pd.DataFrame(results)

    summary = {
        "final_bankroll": current_bankroll,
        "total_return": current_bankroll - bankroll,
        "roi": (current_bankroll - bankroll) / bankroll if bankroll > 0 else 0,
        "num_bets": len(results_df),
        "win_rate": results_df["win"].mean() if not results_df.empty else 0
    }

    return results_df, summary


def simulate_edge_strategy(df, threshold=0.1):
    """
    Simplified version: only counts correct edge direction (no bankroll compounding).

    Useful for debugging model quality.
    """

    df = df.copy()

    correct = 0
    total = 0

    for _, row in df.iterrows():

        edge = row["model_prob"] - row["price"]

        if abs(edge) < threshold:
            continue

        true_outcome = np.random.rand() < row["model_prob"]

        predicted = edge > 0
        actual = true_outcome

        if predicted == actual:
            correct += 1

        total += 1

    accuracy = correct / total if total > 0 else 0

    return {
        "accuracy": accuracy,
        "total_bets": total
    }