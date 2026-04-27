import pandas as pd
import pytest

from olympoly.olympics_betting.market_vs_model import compare_market_vs_model


def test_compare_market_vs_model_basic():
    """Verify merge + difference calculation"""

    df_market = pd.DataFrame({
        "event": ["USA", "China"],
        "price": [0.7, 0.4]
    })

    df_model = pd.DataFrame({
        "event": ["USA", "China"],
        "model_prob": [0.6, 0.5]
    })

    result = compare_market_vs_model(df_market, df_model)

    # Column checks
    assert "market_prob" in result.columns
    assert "model_prob" in result.columns
    assert "difference" in result.columns

    # Value check
    usa = result[result["event"] == "USA"].iloc[0]
    assert pytest.approx(usa["difference"]) == 0.7 - 0.6


def test_custom_column_names():
    """Ensure function works with different join column"""

    df_market = pd.DataFrame({
        "event": ["USA"],
        "price": [0.7]
    })

    df_model = pd.DataFrame({
        "NOC": ["USA"],
        "model_prob": [0.6]
    })

    result = compare_market_vs_model(
        df_market,
        df_model,
        model_col="NOC"
    )

    assert not result.empty


def test_missing_columns():
    """Ensure function raises error if required columns are missing"""

    df_market = pd.DataFrame({
        "event": ["USA"]
        # missing price
    })

    df_model = pd.DataFrame({
        "event": ["USA"],
        "model_prob": [0.6]
    })

    with pytest.raises(ValueError):
        compare_market_vs_model(df_market, df_model)


def test_difference_sign():
    """Ensure difference = market_prob - model_prob"""

    df_market = pd.DataFrame({
        "event": ["A"],
        "price": [0.3]
    })

    df_model = pd.DataFrame({
        "event": ["A"],
        "model_prob": [0.7]
    })

    result = compare_market_vs_model(df_market, df_model)

    diff = result.iloc[0]["difference"]
    assert diff == pytest.approx(0.3 - 0.7)