import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def build_features(df):
    df = df.copy()

    df['is_gold'] = (df['Medal'] == 'Gold').astype(int)

    country_strength = df.groupby('NOC')['is_gold'].mean()
    athlete_exp = df.groupby('Name')['ID'].count()

    df = df.join(country_strength, on='NOC', rsuffix='_country')
    df = df.join(athlete_exp, on='Name', rsuffix='_athlete')

    return df[['is_gold_country', 'ID_athlete', 'is_gold']].rename(columns={
        'is_gold_country': 'country_strength',
        'ID_athlete': 'athlete_exp'
    }).dropna()


def train_model(df):
    df_feat = build_features(df)

    X = df_feat[['country_strength', 'athlete_exp']]
    y = df_feat['is_gold']

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    from sklearn.linear_model import LogisticRegression
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
    print("Winners avg prob:",
        results[results['actual'] == 1]['pred_prob'].mean())

    # Average predicted probability for non-winners
    print("Losers avg prob:",
        results[results['actual'] == 0]['pred_prob'].mean())

    # =========================
    # 7. MODEL PERFORMANCE SCORE
    # =========================

    from sklearn.metrics import roc_auc_score

    # AUC = how well model separates winners vs losers
    auc = roc_auc_score(y_test, preds)

    print("AUC:", auc)