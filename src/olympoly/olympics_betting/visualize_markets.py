"""
visualize_markets.py - Visualization tools for Olympics prediction markets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_price_over_time(df, event, use_normalized=False):
    """Plot prediction market price trends over time for a given event.

    Parameters:
        df: DataFrame with 'event', 'outcome', 'timestamp', and 'price' columns.
        event: Event name to filter by.
        use_normalized: If True, use normalized prices instead of raw prices.
    """
    plt.figure()  # ✅ FIX

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
    """Compare latest prices across different prediction markets for an event.

    Parameters:
        df: DataFrame with 'event', 'market', 'outcome', 'timestamp', and 'price' columns.
        event: Event name to filter by.
    """
    plt.figure()  # ✅ FIX

    subset = df[df["event"] == event].copy()

    if subset.empty:
        raise ValueError(f"No data found for event: {event}")

    latest = subset.sort_values("timestamp").groupby(
        ["market", "outcome"]
    ).tail(1)

    for market in latest["market"].unique():
        data = latest[latest["market"] == market]

        labels = data["outcome"].astype(str) + f" ({market})"

        plt.bar(
            labels,
            data["price"]
        )

    plt.xlabel("Outcome")
    plt.ylabel("Price")
    plt.title(f"Market Comparison: {event}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_latest_snapshot(df, event):
    """Display a bar chart of the most recent prices for each outcome of an event.

    Parameters:
        df: DataFrame with 'event', 'outcome', 'timestamp', and 'price' columns.
        event: Event name to filter by.
    """
    plt.figure()  # ✅ FIX

    subset = df[df["event"] == event].copy()

    if subset.empty:
        raise ValueError(f"No data found for event: {event}")

    idx = subset.groupby(["outcome"])["timestamp"].idxmax()
    latest = subset.loc[idx]

    plt.bar(latest["outcome"].astype(str), latest["price"])  # ✅ FIX (force string)

    plt.xlabel("Outcome")
    plt.ylabel("Price")
    plt.title(f"Latest Market Snapshot: {event}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_market_vs_model(df_compare):
    """Plot side-by-side bar chart comparing market and model probabilities.

    Parameters:
        df_compare: DataFrame with 'event', 'market_prob', and 'model_prob' columns.
    """
    plt.figure()  # ✅ FIX

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
    """Plot the largest disagreements between market and model probabilities.

    Parameters:
        df_compare: DataFrame with 'event' and 'difference' columns.
        n: Number of top edges to display.
    """
    plt.figure()  # ✅ FIX

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
    """Scatter plot comparing market vs model probabilities with a diagonal reference line.

    Parameters:
        df_compare: DataFrame with 'market_prob' and 'model_prob' columns.
    """
    plt.figure()  # ✅ FIX

    if not {"market_prob", "model_prob"}.issubset(df_compare.columns):
        raise ValueError("Missing required columns")

    plt.scatter(df_compare["market_prob"], df_compare["model_prob"])

    plt.xlabel("Market Probability")
    plt.ylabel("Model Probability")
    plt.title("Market vs Model Scatter")

    min_val = min(df_compare["market_prob"].min(), df_compare["model_prob"].min())
    max_val = max(df_compare["market_prob"].max(), df_compare["model_prob"].max())

    plt.plot([min_val, max_val], [min_val, max_val])

    plt.tight_layout()
    plt.show()