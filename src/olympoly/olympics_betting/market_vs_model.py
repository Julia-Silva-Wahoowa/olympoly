import pandas as pd


def compare_market_vs_model(df_market, df_model, market_col="event", model_col="event"):
    """
    Compare market probabilities with model predictions.

    Parameters:
        df_market (pd.DataFrame):
            Must contain columns ['event', 'price']

        df_model (pd.DataFrame):
            Must contain columns ['event' or other identifier, 'model_prob']

        market_col (str):
            Column name in df_market to merge on (default: "event")

        model_col (str):
            Column name in df_model to merge on (default: "event")

    Returns:
        pd.DataFrame with comparison and differences
    """

    # Rename market column for clarity
    market = df_market.copy().rename(columns={"price": "market_prob"})

    # Merge using flexible column names
    merged = pd.merge(
        market,
        df_model,
        left_on=market_col,
        right_on=model_col,
        how="inner"
    )

    # Compute disagreement
    merged["difference"] = merged["market_prob"] - merged["model_prob"]

    # Sort by biggest disagreement
    merged = merged.sort_values(by="difference", key=abs, ascending=False)

    return merged