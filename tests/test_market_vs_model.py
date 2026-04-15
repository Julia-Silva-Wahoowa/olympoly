import pandas as pd
import pytest

from olympoly.olympics_betting.market_vs_model import compare_market_vs_model


def test_compare_market_vs_model_basic():
    """Verify that market and model data are merged correctly and that the probability difference is computed"""

    # Create fake market data
    df_market = pd.DataFrame({
        "event": ["USA", "China"],
        "price": [0.7, 0.4]
    })

    # Create fake model data
    df_model = pd.DataFrame({
        "event": ["USA", "China"],
        "model_prob": [0.6, 0.5]
    })

    result = compare_market_vs_model(df_market, df_model)

    # Check columns exist
    assert "market_prob" in result.columns
    assert "model_prob" in result.columns
    assert "difference" in result.columns

    # Check calculation is correct
    usa_row = result[result["event"] == "USA"].iloc[0]
    assert pytest.approx(usa_row["difference"]) == 0.7 - 0.6


def test_custom_column_names():
    """Verify that comparison works when the model DataFrame uses a different join column name"""
    # Test flexibility when model uses a different column name (e.g., NOC instead of event)
    df_market = pd.DataFrame({
        "event": ["USA"],
        "price": [0.7]
    })

    df_model = pd.DataFrame({
        "NOC": ["USA"],
        "model_prob": [0.6]
    })

    result = compare_market_vs_model(df_market, df_model, model_col="NOC")

    assert not result.empty