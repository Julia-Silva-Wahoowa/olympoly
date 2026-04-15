"""test_monte_carlo_gold.py - Test vectorized is_gold logic"""

import pandas as pd
import numpy as np
import pytest


# Fixtures
@pytest.fixture
def medal_data():
    return pd.DataFrame({
        "Medal": ["Gold", "Silver", "Bronze", "gold", " Gold ", "NA"]
    })


@pytest.fixture
def rank_data():
    return pd.DataFrame({
        "Rank": [1, 2, 3, 1, None]
    })


# Tests for medal-based gold detection
def test_gold_from_medal_column(medal_data):
    result = medal_data["Medal"].astype(str).str.strip().str.lower().isin(["gold", "g"])
    assert result.tolist() == [True, False, False, True, True, False]


def test_gold_medal_count(medal_data):
    result = medal_data["Medal"].astype(str).str.strip().str.lower().isin(["gold", "g"])
    assert result.sum() == 3


def test_non_gold_excluded(medal_data):
    result = medal_data["Medal"].astype(str).str.strip().str.lower().isin(["gold", "g"])
    non_gold = medal_data[~result]
    assert len(non_gold) == 3


# Tests for rank-based gold detection
def test_gold_from_rank_column(rank_data):
    result = pd.to_numeric(rank_data["Rank"], errors="coerce") == 1
    assert result.tolist() == [True, False, False, True, False]


def test_rank_null_handling(rank_data):
    result = pd.to_numeric(rank_data["Rank"], errors="coerce") == 1
    assert result.iloc[-1] == False