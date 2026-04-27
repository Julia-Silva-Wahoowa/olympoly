import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def build_features(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Target only used for feature engineering (train only)
    train_df['is_gold'] = (train_df['Medal'] == 'Gold').astype(int)

    # Aggregate features computed ONLY from training data
    country_strength = train_df.groupby('NOC')['is_gold'].mean()
    athlete_exp = train_df.groupby('Name')['ID'].count()

    # Map to train
    train_df['country_strength'] = train_df['NOC'].map(country_strength)
    train_df['athlete_exp'] = train_df['Name'].map(athlete_exp)

    # Map to test (no target leakage here)
    test_df['country_strength'] = test_df['NOC'].map(country_strength)
    test_df['athlete_exp'] = test_df['Name'].map(athlete_exp)

    # Fill missing values from unseen categories
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)

    # Return ONLY features (no labels inside)
    train_feat = train_df[['country_strength', 'athlete_exp']]
    test_feat = test_df[['country_strength', 'athlete_exp']]

    return train_feat, test_feat


def train_model(df):

    # =========================
    # ENTITY-LEVEL SPLIT (FIXED LEAKAGE)
    # =========================
    unique_names = df['Name'].unique()
    train_names, test_names = train_test_split(unique_names, random_state=42)

    train_df = df[df['Name'].isin(train_names)].copy()
    test_df = df[df['Name'].isin(test_names)].copy()

    # =========================
    # FEATURE ENGINEERING
    # =========================
    train_feat, test_feat = build_features(train_df, test_df)

    X_train = train_feat
    X_test = test_feat

    # =========================
    # TARGETS (NO FEATURE LEAKAGE)
    # =========================
    y_train = (train_df['Medal'] == 'Gold').astype(int)
    y_test = (test_df['Medal'] == 'Gold').astype(int)

    # =========================
    # MODEL
    # =========================
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model, X_test, y_test


# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":

    model, X_test, y_test = train_model(df)

    preds = model.predict_proba(X_test)[:, 1]

    results = X_test.copy()
    results['pred_prob'] = preds
    results['actual'] = y_test.values

    top = results.sort_values('pred_prob', ascending=False)

    print(top.head(15))

    print("Unique probabilities:", results['pred_prob'].nunique())

    print("Winners avg prob:", results[results['actual'] == 1]['pred_prob'].mean())
    print("Losers avg prob:", results[results['actual'] == 0]['pred_prob'].mean())

    auc = roc_auc_score(y_test, preds)
    print("AUC:", auc)