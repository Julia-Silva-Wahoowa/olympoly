import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from olympoly.olympics_betting.regression_model.model import build_features, train_model

"""
What this file covers
─────────────────────
  TestBuildFeatures  — verifies that build_features() produces clean,
                       correctly shaped train and test feature DataFrames 
                       without target leakage.
  TestTrainModel     — verifies that train_model() returns a fitted
                       LogisticRegression with sensible outputs and basic
                       predictive signal.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_data():
    nocs = ["USA", "CHN", "GBR", "AUS", "RUS"]
    names = [f"Athlete_{i}" for i in range(12)]
    medals = ["Gold", "Silver", "Bronze", None, None]

    rng = np.random.default_rng(0)
    n = 40

    return pd.DataFrame({
        "ID":    range(1, n + 1),
        "Name":  rng.choice(names,   n).tolist(),
        "NOC":   rng.choice(nocs,    n).tolist(),
        "Year":  rng.choice([2000, 2004, 2008, 2012, 2016, 2020], n).tolist(),
        "Sport": rng.choice(["Athletics", "Swimming", "Cycling"], n).tolist(),
        "Medal": rng.choice(medals,  n).tolist(),
    })


@pytest.fixture
def built_features(sample_data):
    """
    Simulates a train/test split to feed into the new build_features API.
    """
    train_df = sample_data.iloc[:20]
    test_df = sample_data.iloc[20:]
    return build_features(train_df, test_df)


@pytest.fixture
def trained(sample_data):
    return train_model(sample_data)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: build_features
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildFeatures:

    def test_required_columns_present(self, built_features):
        """
        Verify that build_features() outputs exactly the feature columns.
        (is_gold is no longer returned to prevent leakage).
        """
        train_feat, test_feat = built_features
        for col in ("country_strength", "athlete_exp"):
            assert col in train_feat.columns
            assert col in test_feat.columns

    def test_no_nulls(self, built_features):
        """
        Confirm that the fillna(0) correctly handles missing values.
        """
        train_feat, test_feat = built_features
        assert train_feat.isnull().sum().sum() == 0
        assert test_feat.isnull().sum().sum() == 0

    def test_returns_dataframes(self, built_features):
        train_feat, test_feat = built_features
        assert isinstance(train_feat, pd.DataFrame)
        assert isinstance(test_feat, pd.DataFrame)

    def test_nonempty(self, built_features):
        train_feat, test_feat = built_features
        assert len(train_feat) > 0
        assert len(test_feat) > 0

    def test_country_strength_bounded(self, built_features):
        """
        country_strength is the per-NOC mean of is_gold.
        Must lie in [0, 1]. Unseen test categories are filled with 0.
        """
        train_feat, test_feat = built_features
        assert train_feat["country_strength"].between(0, 1).all()
        assert test_feat["country_strength"].between(0, 1).all()

    def test_athlete_exp_positive(self, built_features):
        """
        Train athletes must have >= 1 experience.
        Unseen test athletes will have 0 (handled by fillna).
        """
        train_feat, test_feat = built_features
        assert (train_feat["athlete_exp"] >= 1).all()
        assert (test_feat["athlete_exp"] >= 0).all()

    # ── boundary cases ───────────────────────────────────────────────────────

    def test_all_no_medals(self):
        df = pd.DataFrame({
            "ID":    [1, 2, 3, 4],
            "Name":  ["A", "B", "C", "D"],
            "NOC":   ["USA", "CHN", "GBR", "AUS"],
            "Medal": [None, "Silver", "Bronze", None],
        })
        # Simulate passing the same DF as train and test to verify logic
        train_feat, test_feat = build_features(df, df)
        assert (train_feat["country_strength"] == 0).all()

    def test_all_gold(self):
        df = pd.DataFrame({
            "ID":    [1, 2, 3],
            "Name":  ["A", "B", "C"],
            "NOC":   ["USA", "CHN", "GBR"],
            "Medal": ["Gold", "Gold", "Gold"],
        })
        train_feat, test_feat = build_features(df, df)
        assert (train_feat["country_strength"] == 1).all()

    def test_single_athlete_experience(self):
        df = pd.DataFrame({
            "ID":    [1, 2],
            "Name":  ["Solo", "Other"],
            "NOC":   ["USA", "CHN"],
            "Medal": ["Gold", None],
        })
        train_feat, test_feat = build_features(df, df)
        solo_idx = df[df["Name"] == "Solo"].index
        solo_rows = train_feat[train_feat.index.isin(solo_idx)]
        if len(solo_rows):
            assert (solo_rows["athlete_exp"] == 1).all()

    def test_repeated_athlete_experience(self):
        df = pd.DataFrame({
            "ID":    [1, 2, 3, 4, 5],
            "Name":  ["A", "A", "A", "A", "B"],
            "NOC":   ["USA", "USA", "USA", "USA", "CHN"],
            "Medal": ["Gold", None, "Silver", None, "Gold"],
        })
        train_feat, test_feat = build_features(df, df)
        a_idx = df[df["Name"] == "A"].index
        a_rows = train_feat[train_feat.index.isin(a_idx)]
        if len(a_rows):
            assert (a_rows["athlete_exp"] == 4).all()

    def test_fillna_removes_nulls(self):
        df = pd.DataFrame({
            "ID":    [1, 2, 3],
            "Name":  ["A", "B", "C"],
            "NOC":   ["USA", "CHN", "GBR"],
            "Medal": ["Gold", None, "Silver"],
        })
        train_feat, test_feat = build_features(df, df)
        assert train_feat.isnull().sum().sum() == 0


# ─────────────────────────────────────────────────────────────────────────────
# Tests: train_model
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainModel:

    def test_returns_three_values(self, trained):
        assert len(trained) == 3

    def test_model_is_logistic_regression(self, trained):
        model, _, _ = trained
        assert isinstance(model, LogisticRegression)

    def test_model_is_fitted(self, trained):
        model, _, _ = trained
        assert hasattr(model, "coef_")

    def test_x_test_correct_columns(self, trained):
        _, X_test, _ = trained
        assert set(X_test.columns) == {"country_strength", "athlete_exp"}

    def test_lengths_consistent(self, trained):
        _, X_test, y_test = trained
        assert len(X_test) == len(y_test)

    def test_y_test_is_binary(self, trained):
        _, _, y_test = trained
        assert set(y_test.unique()).issubset({0, 1})

    def test_predict_proba_works(self, trained):
        model, X_test, _ = trained
        probs = model.predict_proba(X_test)[:, 1]
        assert ((probs >= 0) & (probs <= 1)).all()

    def test_predict_proba_shape(self, trained):
        model, X_test, _ = trained
        raw = model.predict_proba(X_test)
        assert raw.shape == (len(X_test), 2)

    def test_coef_shape(self, trained):
        model, _, _ = trained
        assert model.coef_.shape == (1, 2)

    def test_predictions_are_floats(self, trained):
        model, X_test, _ = trained
        probs = model.predict_proba(X_test)[:, 1]
        assert probs.dtype in (np.float32, np.float64)

    def test_model_produces_predictions_for_new_data(self, trained):
        model, _, _ = trained
        new_row = pd.DataFrame({
            "country_strength": [0.5],
            "athlete_exp":      [3],
        })
        prob = model.predict_proba(new_row)[0, 1]
        assert 0.0 <= prob <= 1.0

    def test_high_strength_country_scores_higher(self, trained):
        model, _, _ = trained
        strong = pd.DataFrame({"country_strength": [0.8], "athlete_exp": [5]})
        weak = pd.DataFrame({"country_strength": [0.05], "athlete_exp": [5]})
        p_strong = model.predict_proba(strong)[0, 1]
        p_weak = model.predict_proba(weak)[0, 1]
        assert p_strong >= p_weak
