import pandas as pd
import pytest

from olympoly.olympics_betting.regression_model.random_forests import train_rf_model, OlympicFeatureEngineer

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
# Test: Leakage Prevention
# -------------------------
def test_pipeline_no_leakage(sample_data):
    sample_data['is_gold'] = (sample_data['Medal'] == 'Gold').astype(int)

    train_df = sample_data.iloc[:5]  # Mock train set
    test_df = sample_data.iloc[5:]  # Mock test set

    engineer = OlympicFeatureEngineer()
    engineer.fit(train_df[['NOC', 'Name', 'Sport']], train_df['is_gold'])

    transformed_test = engineer.transform(test_df[['NOC', 'Name', 'Sport']])

    # Test 1: 'CHN' had 1 gold out of 1 appearance in the TRAIN set (1.0).
    # In test set, CHN gets another gold, but the aggregate shouldn't see it.
    assert engineer.country_map_['CHN'] == 1.0

    # Test 2: Unseen entities (FIJ, CMR) in test set should get the training set's global mean.
    train_mean = train_df['is_gold'].mean()  # 2 golds in 5 rows = 0.4
    assert transformed_test.iloc[1]['country_strength'] == train_mean  # FIJ
    assert transformed_test.iloc[2]['country_strength'] == train_mean  # CMR


# -------------------------
# Test: model runs
# -------------------------
def test_train_rf_model(sample_data):
    pipeline, X_test, y_test, probs = train_rf_model(sample_data)

    assert pipeline is not None
    assert len(X_test) == len(y_test)
    assert len(probs) == len(y_test)
    assert ((probs >= 0) & (probs <= 1)).all()


# -------------------------
# Test: predictions vary
# -------------------------
def test_prediction_variation(sample_data):
    pipeline, X_test, y_test, probs = train_rf_model(sample_data)

    assert len(set(probs)) > 1

# -------------------------
# Test: model has signal
# -------------------------
def test_model_signal():
    """
    To verify the model can learn a signal without leakage, 
    we must provide data that actually contains a strong pattern.
    """
    # 15 guaranteed winners and 15 guaranteed losers
    signal_data = pd.DataFrame({
        'NOC': ['WIN_NOC']*15 + ['LOSE_NOC']*15,
        'Name': ['Winner_Name']*15 + ['Loser_Name']*15,
        'Sport': ['Win_Sport']*15 + ['Lose_Sport']*15,
        'Medal': ['Gold']*15 + [None]*15
    })

    pipeline, X_test, y_test, probs = train_rf_model(signal_data)

    results = X_test.copy()
    results["pred_prob"] = probs
    results["actual"] = y_test.values

    # Check the means of actual winners vs actual losers in the test set
    winners = results[results["actual"] == 1]["pred_prob"].mean()
    losers = results[results["actual"] == 0]["pred_prob"].mean()

    # The model should easily pick up the 100% vs 0% pattern
    assert winners > losers
