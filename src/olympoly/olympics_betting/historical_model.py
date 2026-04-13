# historical_model.py

"""
historical_model.py - Estimate probabilities from historical Olympic data.

Functions:
- country_medal_probability: Probability a country wins a medal
- country_gold_probability: Probability a country wins gold
- sport_medal_probability: Probability of medals in a sport
- rolling_country_performance: Trend-based probability estimate
"""

import pandas as pd


def country_medal_probability(df, country):
    """
    Estimate probability that a country wins any medal.

    Parameters:
    - df: DataFrame with ['Team', 'Medal']
    - country: str

    Returns:
    - float probability
    """
    country_df = df[df["Team"] == country]

    if country_df.empty:
        return 0.0

    total_entries = len(country_df)
    medal_entries = country_df["Medal"].notnull().sum()

    return medal_entries / total_entries


def country_gold_probability(df, country):
    """
    Estimate probability that a country wins gold.

    Parameters:
    - df: DataFrame with ['Team', 'Medal']
    - country: str

    Returns:
    - float probability
    """
    country_df = df[df["Team"] == country]

    if country_df.empty:
        return 0.0

    total_entries = len(country_df)
    gold_entries = (country_df["Medal"] == "Gold").sum()

    return gold_entries / total_entries


def sport_medal_probability(df, sport):
    """
    Probability that an entry in a sport results in a medal.

    Parameters:
    - df: DataFrame with ['Sport', 'Medal']
    - sport: str

    Returns:
    - float probability
    """
    sport_df = df[df["Sport"] == sport]

    if sport_df.empty:
        return 0.0

    total_entries = len(sport_df)
    medal_entries = sport_df["Medal"].notnull().sum()

    return medal_entries / total_entries


def rolling_country_performance(df, country, window=2):
    """
    Estimate country performance trend using rolling average.

    Parameters:
    - df: DataFrame with ['Team', 'Year', 'Medal']
    - country: str
    - window: int (number of Olympic cycles)

    Returns:
    - DataFrame with rolling medal probability
    """
    country_df = df[df["Team"] == country].copy()

    if country_df.empty:
        return pd.DataFrame()

    # Create binary medal column
    country_df["WonMedal"] = country_df["Medal"].notnull().astype(int)

    # Aggregate by year
    yearly = (
        country_df.groupby("Year")
        .agg(
            Athletes=("Team", "count"),
            Medals=("WonMedal", "sum")
        )
        .sort_index()
    )

    # Compute probability
    yearly["Medal_Probability"] = yearly["Medals"] / yearly["Athletes"]

    # Rolling average
    yearly["Rolling_Probability"] = yearly["Medal_Probability"].rolling(window=window).mean()

    return yearly