import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.tree import plot_tree


def build_features(df):
    df = df.copy()

    # target: gold medal or not
    df['is_gold'] = (df['Medal'] == 'Gold').astype(int)

    # country strength
    country_strength = df.groupby('NOC')['is_gold'].mean()

    # athlete experience
    athlete_exp = df.groupby('Name')['ID'].count()

    # map features back
    df = df.join(country_strength, on='NOC', rsuffix='_country')
    df = df.join(athlete_exp, on='Name', rsuffix='_athlete')

    # clean columns
    df = df.rename(columns={
        'is_gold_country': 'country_strength',
        'ID_athlete': 'athlete_exp'
    })

    return df[['country_strength', 'athlete_exp', 'is_gold']].dropna()


def train_rf_model(df):
    df_feat = build_features(df)

    X = df_feat[['country_strength', 'athlete_exp']]
    y = df_feat['is_gold']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(
        n_estimators=100,   # number of trees
        max_depth=5,        # prevents overfitting
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("Accuracy:", accuracy_score(y_test, preds))
    print("AUC:", roc_auc_score(y_test, probs))

    return model, X_test, y_test, probs


# Results table
def get_results(X_test, y_test, probs):
    results = X_test.copy()
    results['pred_prob'] = probs
    results['actual'] = y_test.values

    return results.sort_values('pred_prob', ascending=False)

# Feature Importance plot
def plot_feature_importance(model):
    importances = model.feature_importances_
    features = ['country_strength', 'athlete_exp']

    sorted_idx = np.argsort(importances)

    plt.figure(figsize=(6,4))
    plt.barh(np.array(features)[sorted_idx], importances[sorted_idx])
    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Importance")
    plt.show()

# Visualizing a Tree
def plot_sample_tree(model):
    tree = model.estimators_[0]

    plt.figure(figsize=(12,6))
    plot_tree(tree,
        feature_names=['country_strength', 'athlete_exp'],
        filled=True,
        max_depth=3
    )
    plt.title("Sample Tree from Random Forest")
    plt.show()

# Run Everything
if __name__ == "__main__":
    model, X_test, y_test, probs = train_rf_model(df)

    results = get_results(X_test, y_test, probs)

    print(results.head(15))

    print("Winners avg prob:",
        results[results['actual'] == 1]['pred_prob'].mean())

    print("Losers avg prob:",
        results[results['actual'] == 0]['pred_prob'].mean())

    plot_feature_importance(model)

    plot_sample_tree(model)
