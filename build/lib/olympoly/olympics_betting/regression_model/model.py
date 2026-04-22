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

    return df[['is_gold_country', 'ID_athlete', 'is_gold']].dropna()


def train_model(df):
    df_feat = build_features(df)

    X = df_feat[['is_gold_country', 'ID_athlete']]
    y = df_feat['is_gold']

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    return model