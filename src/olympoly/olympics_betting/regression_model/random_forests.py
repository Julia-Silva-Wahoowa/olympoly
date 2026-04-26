import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.tree import plot_tree

from olympoly.load_data import load_olympic_data

def compute_group_features(df):
    df = df.copy()
    df['is_gold'] = (df['Medal'] == 'Gold').astype(int)

    country_strength = df.groupby('NOC')['is_gold'].mean()
    sport_strength = df.groupby('Sport')['is_gold'].mean()

    athlete_exp = df.groupby('Name').cumcount() + 1

    return {
        "country_strength": country_strength,
        "athlete_exp": athlete_exp,
        "sport_strength": sport_strength
    }

def build_features(df):
    df = df.copy()
    df['is_gold'] = (df['Medal'] == 'Gold').astype(int)

    features = compute_group_features(df)

    df['country_strength'] = df['NOC'].map(features['country_strength'])
    df['athlete_exp'] = features['athlete_exp']
    df['sport_strength'] = df['Sport'].map(features['sport_strength'])

    return df[
        ['country_strength', 'athlete_exp', 'sport_strength', 'is_gold']
    ].dropna()


def train_rf_model(df):
    df_feat = build_features(df)

    X = df_feat[['country_strength', 'athlete_exp', 'sport_strength']]
    y = df_feat['is_gold']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("Accuracy:", accuracy_score(y_test, preds))
    print("AUC:", roc_auc_score(y_test, probs))

    return model, X_test, y_test, probs


def get_results(X_test, y_test, probs):
    results = X_test.copy()
    results['pred_prob'] = probs
    results['actual'] = y_test.values

    return results.sort_values('pred_prob', ascending=False)


def plot_feature_importance(model):
    importances = model.feature_importances_
    features = ['country_strength', 'athlete_exp', 'sport_strength']

    sorted_idx = np.argsort(importances)

    plt.figure(figsize=(6,4))
    plt.barh(np.array(features)[sorted_idx], importances[sorted_idx])
    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Importance")
    plt.show()


def plot_sample_tree(model):
    tree_model = model.estimators_[0]

    plt.figure(figsize=(12,6))
    plot_tree(
        tree_model,
        feature_names=['country_strength', 'athlete_exp', 'sport_strength'],
        filled=True,
        max_depth=3
    )
    plt.title("Sample Tree from Random Forest")
    plt.show()


def predict_from_real_data(model, df, noc, athlete_name, sport_name):
    features = compute_group_features(df)

    cs = features['country_strength'].get(noc, 0)
    ag = features['athlete_exp'].get(athlete_name, 0)
    es = features['sport_strength'].get(sport_name, 0)

    X_input = pd.DataFrame({
        "country_strength": [cs],
        "athlete_exp": [ag],
        "sport_strength": [es]
    })

    return model.predict_proba(X_input)[:, 1][0]


# =========================
# RUN EVERYTHING (FIXED)
# =========================
if __name__ == "__main__":

    df = load_olympic_data()

    model, X_test, y_test, probs = train_rf_model(df)

    results = get_results(X_test, y_test, probs)

    print(results.head(15))

    print("Winners avg prob:",
          results[results['actual'] == 1]['pred_prob'].mean())

    print("Losers avg prob:",
          results[results['actual'] == 0]['pred_prob'].mean())

    plot_feature_importance(model)
    plot_sample_tree(model)

    # Pick 5 real athletes from dataset
    examples = df.dropna(subset=["Name", "Sport", "NOC"]).sample(5)[
    ["NOC", "Name", "Sport"]
    ].values

    for noc, name, sport in examples:
        prob = predict_from_real_data(model, df, noc, name, sport)
    
        print(f"Athlete: {name}")
        print(f"Country: {noc}")
        print(f"Sport: {sport}")
        print(f"Predicted Gold Probability: {prob:.4f}")
        print("-" * 40)