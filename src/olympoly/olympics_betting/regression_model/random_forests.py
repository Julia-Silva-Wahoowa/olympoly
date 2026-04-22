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

    df['is_gold'] = (df['Medal'] == 'Gold').astype(int)

    country_strength = df.groupby('NOC')['is_gold'].mean()
    athlete_gold_rate = df.groupby('Name')['is_gold'].mean()
    event_strength = df.groupby('Event')['is_gold'].mean()

    df = df.join(country_strength, on='NOC', rsuffix='_country')
    df = df.join(athlete_gold_rate, on='Name', rsuffix='_athlete')
    df = df.join(event_strength, on='Event', rsuffix='_event')

    return df[[
        'is_gold_country',
        'is_gold_athlete',
        'is_gold_event',
        'is_gold'
    ]].rename(columns={
        'is_gold_country': 'country_strength',
        'is_gold_athlete': 'athlete_gold_rate',
        'is_gold_event': 'event_strength'
    }).dropna()


def train_rf_model(df):
    df_feat = build_features(df)

    X = df_feat[['country_strength', 'athlete_gold_rate', 'event_strength']]
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
    features = ['country_strength', 'athlete_gold_rate', 'event_strength']

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
        feature_names=['country_strength', 'athlete_gold_rate', 'event_strength'],
        filled=True,
        max_depth=3
    )
    plt.title("Sample Tree from Random Forest")
    plt.show()


def predict_from_real_data(model, df, noc, athlete_name, event_name):
    df = df.copy()

    df['is_gold'] = (df['Medal'] == 'Gold').astype(int)

    country_strength = df.groupby('NOC')['is_gold'].mean()
    athlete_gold_rate = df.groupby('Name')['is_gold'].mean()
    event_strength = df.groupby('Event')['is_gold'].mean()

    cs = country_strength.get(noc, 0)
    ag = athlete_gold_rate.get(athlete_name, 0)
    es = event_strength.get(event_name, 0)

    X_input = pd.DataFrame({
        "country_strength": [cs],
        "athlete_gold_rate": [ag],
        "event_strength": [es]
    })

    return model.predict_proba(X_input)[:, 1][0]


# =========================
# RUN EVERYTHING (FIXED)
# =========================
if __name__ == "__main__":

    # IMPORTANT: define df only once and use consistently
    df = df.copy()  # ensures no accidental mutation issues

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
    examples = df.dropna(subset=["Name", "Event", "NOC"]).sample(5)[
    ["NOC", "Name", "Event"]
    ].values

    for noc, name, event in examples:
        prob = predict_from_real_data(model, df, noc, name, event)
    
        print(f"Athlete: {name}")
        print(f"Country: {noc}")
        print(f"Event: {event}")
        print(f"Predicted Gold Probability: {prob:.4f}")
        print("-" * 40)