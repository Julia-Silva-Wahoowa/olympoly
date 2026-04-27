import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.tree import plot_tree
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class OlympicFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to compute group aggregates on the training set ONLY.
    Prevents target leakage into the test set.
    """

    def __init__(self):
        self.country_map_ = {}
        self.athlete_map_ = {}
        self.sport_map_ = {}
        self.global_gold_rate_ = 0

    def fit(self, X, y):
        X_temp = X.copy()
        X_temp['is_gold'] = y

        self.global_gold_rate_ = y.mean()

        # Learn aggregates strictly from training data
        self.country_map_ = X_temp.groupby('NOC')['is_gold'].mean().to_dict()
        self.athlete_map_ = X_temp.groupby('Name')['is_gold'].count().to_dict()
        self.sport_map_ = X_temp.groupby('Sport')['is_gold'].mean().to_dict()

        return self

    def transform(self, X):
        X_transformed = pd.DataFrame(index=X.index)

        # Map learned aggregates; fill unseen entities with sensible defaults
        X_transformed['country_strength'] = X['NOC'].map(
            self.country_map_).fillna(self.global_gold_rate_)
        X_transformed['athlete_exp'] = X['Name'].map(
            self.athlete_map_).fillna(0)
        X_transformed['sport_strength'] = X['Sport'].map(
            self.sport_map_).fillna(self.global_gold_rate_)

        return X_transformed


def train_rf_model(df):
    df = df.copy()
    df['is_gold'] = (df['Medal'] == 'Gold').astype(int)

    # 1. Split FIRST
    X = df[['NOC', 'Name', 'Sport']]
    y = df['is_gold']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # 2. Build Pipeline
    pipeline = Pipeline([
        ('features', OlympicFeatureEngineer()),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
    ])

    # 3. Fit pipeline
    pipeline.fit(X_train, y_train)

    # 4. Predict
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    print("Accuracy:", accuracy_score(y_test, preds))
    print("AUC:", roc_auc_score(y_test, probs))

    return pipeline, X_test, y_test, probs


# Results table
def get_results(X_test, y_test, probs):
    results = X_test.copy()
    results['pred_prob'] = probs
    results['actual'] = y_test.values

    return results.sort_values('pred_prob', ascending=False)

# Feature Importance plot


def plot_feature_importance(pipeline):
    model = pipeline.named_steps['rf']
    importances = model.feature_importances_
    features = ['country_strength', 'athlete_exp', 'sport_strength']

    sorted_idx = np.argsort(importances)

    plt.figure(figsize=(6, 4))
    plt.barh(np.array(features)[sorted_idx], importances[sorted_idx])
    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Importance")
    plt.show()

# Visualizing a Tree


def plot_sample_tree(pipeline):
    model = pipeline.named_steps['rf']
    tree = model.estimators_[0]

    plt.figure(figsize=(12, 6))
    plot_tree(tree,
              feature_names=['country_strength',
                             'athlete_exp', 'sport_strength'],
              filled=True,
              max_depth=3
              )
    plt.title("Sample Tree from Random Forest")
    plt.show()


def predict_from_real_data(pipeline, noc, athlete_name, sport_name):
    # Pipeline makes inference trivial. Just pass raw features in.
    X_input = pd.DataFrame([{
        "NOC": noc,
        "Name": athlete_name,
        "Sport": sport_name
    }])

    return pipeline.predict_proba(X_input)[0, 1]


# =========================
# RUN EVERYTHING (FIXED)
# =========================
if __name__ == "__main__":
    # Assuming `df` is loaded somewhere up here in the real script
    # df = pd.read_csv('your_data.csv')

    pipeline, X_test, y_test, probs = train_rf_model(df)

    results = get_results(X_test, y_test, probs)

    print(results.head(15))

    print("Winners avg prob:",
          results[results['actual'] == 1]['pred_prob'].mean())

    print("Losers avg prob:",
          results[results['actual'] == 0]['pred_prob'].mean())

    plot_feature_importance(pipeline)
    plot_sample_tree(pipeline)
