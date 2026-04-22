import pandas as pd
import pytest

# import functions from your file
from olympoly.olympics_betting.regression_model.random_forests import build_features, train_rf_model, get_results

# -------------------------
# Fixture: mock dataset
# -------------------------
@pytest.fixture
def sample_data():
    
    """
    Provides a mock Olympic results DataFrame for all tests in this file.
 
    Uses 8 rows across multiple NOCs, athletes, years, and medal outcomes to
    give build_features() and train_rf_model() realistic but controlled input
    without touching the real dataset.
    """

    return pd.DataFrame({
        'ID': [1, 2, 3, 4, 5, 6, 7, 8],
        'Name': [
            'Athlete A', 'Athlete A',
            'Athlete B',
            'Athlete C',
            'Athlete D',
            'Athlete E',
            'Athlete F',
            'Athlete G'
        ],
        'NOC': [
            'USA', 'USA',
            'CHN',
            'GBR',
            'USA',
            'CHN',
            'FIJ',
            'CMR'
        ],
        'Year': [2000, 2004, 2008, 2012, 2016, 2020, 2016, 2008],
        'Sport': [
            'Athletics', 'Athletics',
            'Swimming',
            'Cycling',
            'Basketball',
            'Table Tennis',
            'Rugby',
            'Boxing'
        ],
        'Medal': [
            'Gold', 'Silver',
            'Gold',
            None,
            'Bronze',
            'Gold',
            'Gold',
            None
        ]
    })


# -------------------------
# Test: build_features
# -------------------------
def test_build_features(sample_data):
    
    """
    Checks that build_features() returns a DataFrame containing the three
    required columns (country_strength, athlete_exp, is_gold) with no null
    values, country_strength bounded to [0, 1], and athlete_exp of at least 1.
    """
    
    df_feat = build_features(sample_data)

    assert "country_strength" in df_feat.columns
    assert "athlete_exp" in df_feat.columns
    assert "is_gold" in df_feat.columns

    assert df_feat.isnull().sum().sum() == 0
    assert df_feat["country_strength"].between(0, 1).all()
    assert (df_feat["athlete_exp"] >= 1).all()


# -------------------------
# Test: model runs
# -------------------------
def test_train_rf_model(sample_data):
    
    """
    Checks that train_rf_model() returns a 4-tuple of (model, X_test, y_test,
    probs) where X_test and y_test have equal length, and all predicted
    probabilities fall within [0, 1].
    """
    
    model, X_test, y_test, probs = train_rf_model(sample_data)

    assert model is not None
    assert len(X_test) == len(y_test)
    assert len(probs) == len(y_test)
    assert ((probs >= 0) & (probs <= 1)).all()


# -------------------------
# Test: predictions vary
# -------------------------
def test_prediction_variation(sample_data):
    
    """
    Checks that the model produces more than one unique predicted probability
    across the test set, confirming it is not outputting a constant score for
    every row.
    """
    
    model, X_test, y_test, probs = train_rf_model(sample_data)

    assert len(set(probs)) > 1


# -------------------------
# Test: model has signal
# -------------------------
def test_model_signal(sample_data):
    
    """
    Checks that the model's average predicted probability is higher for actual
    gold-medal winners than for non-winners, verifying the model has learned
    a meaningful directional signal from the training data.
    """
    
    model, X_test, y_test, probs = train_rf_model(sample_data)

    results = X_test.copy()
    results["pred_prob"] = probs
    results["actual"] = y_test.values

    winners = results[results["actual"] == 1]["pred_prob"].mean()
    losers = results[results["actual"] == 0]["pred_prob"].mean()

    assert winners > losers