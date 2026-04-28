import pandas as pd

from olympoly.olympics_betting.market_vs_model import compare_market_vs_model


def run_market_model_comparison(
    model_probs: pd.DataFrame,
    market_odds: pd.DataFrame,
    historical_probs: pd.DataFrame | None = None,
    *,
    valuation_threshold: float = 0.02,
) -> pd.DataFrame:
    """
    Run the full end-to-end workflow comparing model probabilities to market odds.

    This function ties together model output, market data, and comparison logic
    into a single, easy-to-use entry point.

    Parameters
    ----------
    model_probs : pd.DataFrame
        Model-generated probabilities.
    market_odds : pd.DataFrame
        Market odds or implied probabilities.
    historical_probs : pd.DataFrame, optional
        Optional historical probabilities for additional context.
    valuation_threshold : float, optional
        Threshold to flag over/undervalued outcomes.

    Returns
    -------
    pd.DataFrame
        Final comparison DataFrame with interpretation columns.
    """

    # Core comparison using existing logic
    comparison = compare_market_vs_model(
        model_probs=model_probs,
        market_odds=market_odds,
    )

    # Optional historical context
    if historical_probs is not None:
        comparison = comparison.merge(
            historical_probs,
            on=["Event", "NOC"],
            how="left",
            suffixes=("", "_historical"),
        )

    # Simple interpretation / valuation flags
    if "model_prob" in comparison.columns and "market_implied_prob" in comparison.columns:
        diff = comparison["model_prob"] - comparison["market_implied_prob"]

        comparison["valuation"] = "neutral"
        comparison.loc[diff > valuation_threshold, "valuation"] = "undervalued"
        comparison.loc[diff < -valuation_threshold, "valuation"] = "overvalued"

    return comparison