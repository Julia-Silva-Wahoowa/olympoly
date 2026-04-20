import pandas as pd
import pytest

# import functions from your file
from olympoly.olympics_betting.regression_model.random_forests import build_features, train_rf_model

# -------------------------
# Fixture: mock dataset
# -------------------------
@pytest.fixture
def sample_data():
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
    model, X_test, y_test, probs = train_rf_model(sample_data)

    assert model is not None
    assert len(X_test) == len(y_test)
    assert len(probs) == len(y_test)
    assert ((probs >= 0) & (probs <= 1)).all()


# -------------------------
# Test: predictions vary
# -------------------------
def test_prediction_variation(sample_data):
    model, X_test, y_test, probs = train_rf_model(sample_data)

    assert len(set(probs)) > 1


# -------------------------
# Test: model has signal
# -------------------------
def test_model_signal(sample_data):
    model, X_test, y_test, probs = train_rf_model(sample_data)

    results = X_test.copy()
    results["pred_prob"] = probs
    results["actual"] = y_test.values

    winners = results[results["actual"] == 1]["pred_prob"].mean()
    losers = results[results["actual"] == 0]["pred_prob"].mean()

    assert winners > losers