import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
<<<<<<< Updated upstream

def build_features(df):
    df = df.copy()
=======
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
>>>>>>> Stashed changes


def build_features(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df['is_gold'] = (train_df['Medal'] == 'Gold').astype(int)
    test_df['is_gold'] = (test_df['Medal'] == 'Gold').astype(int)

    country_strength = train_df.groupby('NOC')['is_gold'].mean()
    athlete_exp = train_df.groupby('Name')['ID'].count()

    train_df['country_strength'] = train_df['NOC'].map(country_strength)
    train_df['athlete_exp'] = train_df['Name'].map(athlete_exp)
    test_df['country_strength'] = test_df['NOC'].map(country_strength)
    test_df['athlete_exp'] = test_df['Name'].map(athlete_exp)

    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)

    train_feat = train_df[['country_strength', 'athlete_exp', 'is_gold']]
    test_feat = test_df[['country_strength', 'athlete_exp', 'is_gold']]

    return train_feat, test_feat


def train_model(df):
    train_df, test_df = train_test_split(df, random_state=42)

    train_feat, test_feat = build_features(train_df, test_df)

    X_train = train_feat[['country_strength', 'athlete_exp']]
    y_train = train_feat['is_gold']

    X_test = test_feat[['country_strength', 'athlete_exp']]
    y_test = test_feat['is_gold']

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model, X_test, y_test


# =========================
# Possible Applications of the Model:
# =========================



if __name__ == "__main__":
    # =========================
    # 1. TRAIN THE MODEL
    # =========================

    # Train your logistic regression model using your feature pipeline
    model, X_test, y_test = train_model(df)

    # =========================
    # 2. GET PREDICTIONS
    # =========================

    # Predict probability of winning gold for each test row
    preds = model.predict_proba(X_test)[:, 1]

    # =========================
    # 3. BUILD RESULTS TABLE
    # =========================

    # Copy test features so we can attach predictions
    results = X_test.copy()

    # Add predicted probability
    results['pred_prob'] = preds

    # Add actual outcome (0 = no gold, 1 = gold)
    results['actual'] = y_test.values

    # =========================
    # 4. SORT BEST PREDICTIONS
    # =========================

    # Sort rows from highest predicted probability → lowest
    top = results.sort_values('pred_prob', ascending=False)

    # Show top 15 predicted gold chances
    print(top.head(15))

    # =========================
    # 5. CHECK MODEL VARIATION
    # =========================

    # How many unique probability values exist?
    print("Unique probabilities:", results['pred_prob'].nunique())

    # =========================
    # 6. MODEL QUALITY CHECK
    # =========================

    # Average predicted probability for actual winners
    print("Winners avg prob:",results[results['actual'] == 1]['pred_prob'].mean())

    # Average predicted probability for non-winners
    print("Losers avg prob:",results[results['actual'] == 0]['pred_prob'].mean())

    # =========================
    # 7. MODEL PERFORMANCE SCORE
    # =========================

    from sklearn.metrics import roc_auc_score

    # AUC = how well model separates winners vs losers
    auc = roc_auc_score(y_test, preds)

    print("AUC:", auc)