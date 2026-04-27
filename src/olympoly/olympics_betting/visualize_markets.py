"""
visualize_markets.py - Visualization tools for Olympics prediction markets

Supports:
- Price trends over time
- Market vs model comparisons
- Edge (mispricing) visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_price_over_time(df, event, use_normalized=False):
    """
    Plot price (or normalized price) over time for a given event.

    Args:
        df (pd.DataFrame): market data
        event (str): event name
        use_normalized (bool): whether to use normalized_price
    """
    subset = df[df["event"] == event].copy()

    if subset.empty:
        raise ValueError(f"No data found for event: {event}")

    price_col = "normalized_price" if use_normalized and "normalized_price" in subset.columns else "price"

    for outcome in subset["outcome"].unique():
        data = subset[subset["outcome"] == outcome].sort_values("timestamp")
        plt.plot(data["timestamp"], data[price_col], label=outcome)

    plt.xlabel("Time")
    plt.ylabel("Price" if price_col == "price" else "Normalized Price")
    plt.title(f"Price Over Time: {event}")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_market_comparison(df, event):
    """
    Compare prices across markets (e.g., Kalshi vs Polymarket) for a given event.

    Args:
        df (pd.DataFrame): market data
        event (str): event name
    """
    subset = df[df["event"] == event].copy()

    if subset.empty:
        raise ValueError(f"No data found for event: {event}")

    latest = subset.sort_values("timestamp").groupby(
        ["market", "outcome"]
    ).tail(1)

    for market in latest["market"].unique():
        data = latest[latest["market"] == market]
        plt.bar(
            data["outcome"].astype(str) + f" ({market})",
            data["price"]
        )

    plt.xlabel("Outcome")
    plt.ylabel("Price")
    plt.title(f"Market Comparison: {event}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_latest_snapshot(df, event):
    """
    Bar chart of latest market probabilities for an event.

    Args:
        df (pd.DataFrame): market data
        event (str): event name
    """
    subset = df[df["event"] == event].copy()

    if subset.empty:
        raise ValueError(f"No data found for event: {event}")

    idx = subset.groupby(["outcome"])["timestamp"].idxmax()
    latest = subset.loc[idx]

    plt.bar(latest["outcome"], latest["price"])

    plt.xlabel("Outcome")
    plt.ylabel("Price")
    plt.title(f"Latest Market Snapshot: {event}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_market_vs_model(df_compare):
    """
    Compare market probabilities vs model probabilities.

    Args:
        df_compare (pd.DataFrame):
            Output from compare_market_vs_model()
            Must contain ['event', 'market_prob', 'model_prob']
    """
    required_cols = {"event", "market_prob", "model_prob"}
    if not required_cols.issubset(df_compare.columns):
        raise ValueError(f"Missing required columns: {required_cols}")

    df_plot = df_compare.copy()

    x = np.arange(len(df_plot))
    width = 0.35

    plt.bar(x - width/2, df_plot["market_prob"], width, label="Market")
    plt.bar(x + width/2, df_plot["model_prob"], width, label="Model")

    plt.xticks(x, df_plot["event"], rotation=45)
    plt.xlabel("Event")
    plt.ylabel("Probability")
    plt.title("Market vs Model Probabilities")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_top_edges(df_compare, n=10):
    """
    Plot top N biggest mispricings (edges).

    Args:
        df_compare (pd.DataFrame):
            Output from compare_market_vs_model()
            Must contain ['event', 'difference']
        n (int): number of top edges to display
    """
    if "difference" not in df_compare.columns:
        raise ValueError("Column 'difference' not found")

    df_plot = df_compare.copy()
    df_plot["abs_diff"] = df_plot["difference"].abs()

    top = df_plot.sort_values("abs_diff", ascending=False).head(n)

    plt.barh(top["event"], top["difference"])

    plt.xlabel("Market - Model Probability")
    plt.ylabel("Event")
    plt.title(f"Top {n} Market Inefficiencies")
    plt.tight_layout()
    plt.show()


def plot_probability_distribution(df_compare):
    """
    Scatter plot of market vs model probabilities.

    Helps visualize calibration.

    Args:
        df_compare (pd.DataFrame):
            Must contain ['market_prob', 'model_prob']
    """
    if not {"market_prob", "model_prob"}.issubset(df_compare.columns):
        raise ValueError("Missing required columns")

    plt.scatter(df_compare["market_prob"], df_compare["model_prob"])

    plt.xlabel("Market Probability")
    plt.ylabel("Model Probability")
    plt.title("Market vs Model Scatter")

    # 45-degree reference line
    min_val = min(df_compare["market_prob"].min(), df_compare["model_prob"].min())
    max_val = max(df_compare["market_prob"].max(), df_compare["model_prob"].max())

    plt.plot([min_val, max_val], [min_val, max_val])

    plt.tight_layout()
    plt.show()