import pandas as pd
import pytest

from olympoly.load_data import load_data


def test_load_data_returns_dataframe():
    """Ensure load_data returns a non-empty DataFrame"""

    df = load_data()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_load_data_has_expected_columns():
    """Basic sanity check for Olympic dataset structure"""

    df = load_data()

    expected_cols = {"ID", "Name", "Team", "Year", "Sport"}
    assert expected_cols.issubset(set(df.columns))


def test_load_data_no_missing_key_columns():
    """Ensure critical columns are not entirely null"""

    df = load_data()

    assert df["ID"].notna().any()
    assert df["Year"].notna().any()
    assert df["Team"].notna().any()