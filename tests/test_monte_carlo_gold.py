"""test_monte_carlo_gold.py - Test vectorized is_gold logic"""

import pandas as pd


def test_gold_from_medal_column():
    df = pd.DataFrame({
        "Medal": ["Gold", "Silver", "Bronze", "gold", " Gold ", "NA"]
    })
    result = df["Medal"].astype(str).str.strip().str.lower().isin(["gold", "g"])
    assert result.tolist() == [True, False, False, True, True, False]


def test_gold_from_rank_column():
    df = pd.DataFrame({
        "Rank": [1, 2, 3, 1, None]
    })
    result = pd.to_numeric(df["Rank"], errors="coerce") == 1
    assert result.tolist() == [True, False, False, True, False]