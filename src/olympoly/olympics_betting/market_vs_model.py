import pandas as pd


def compare_market_vs_model(df_market, df_model, model_col="event"):
    """
    Merge market and model probabilities and compute edge (difference).
    """

    # -----------------------
    # Validate required columns
    # -----------------------
    if "event" not in df_market.columns:
        raise ValueError("df_market must contain 'event'")
    if "price" not in df_market.columns:
        raise ValueError("df_market must contain 'price'")
    if "model_prob" not in df_model.columns:
        raise ValueError("df_model must contain 'model_prob'")

    # -----------------------
    # Rename for consistency
    # -----------------------
    market = df_market.copy().rename(columns={"price": "market_prob"})
    model = df_model.copy()

    # If model uses different join column (e.g. NOC)
    if model_col != "event":
        model = model.rename(columns={model_col: "event"})

    # -----------------------
    # Merge
    # -----------------------
    merged = pd.merge(market, model, on="event", how="inner")

    # -----------------------
    # Compute difference
    # -----------------------
    merged["difference"] = merged["market_prob"] - merged["model_prob"]

    return merged