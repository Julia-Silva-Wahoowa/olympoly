# timeline.py

"""
timeline.py - Analyze Olympics trends over time.

Functions:
- participation_trends: Athlete participation over the years, overall or by gender.
- medal_trends: Medal counts by country or sport over time.
- sport_popularity: How many athletes per sport over the years.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def participation_trends(df, by_gender=True, plot=True):
    """
    Analyze athlete participation over time.
    """

    # ✅ Validate required columns
    if "Year" not in df.columns or "ID" not in df.columns:
        raise ValueError("DataFrame must contain 'Year' and 'ID' columns")

    # ✅ FIXED: inside function
    gender_col = (
        "Sex" if "Sex" in df.columns
        else "Gender" if "Gender" in df.columns
        else None
    )

    if by_gender:
        if gender_col is None:
            raise ValueError("DataFrame must contain 'Sex' or 'Gender' column")

        trends = (
            df.groupby(["Year", gender_col])["ID"]
            .nunique()
            .reset_index()
            .pivot(index="Year", columns=gender_col, values="ID")
            .fillna(0)
        )
    else:
        trends = (
            df.groupby("Year")["ID"]
            .nunique()
            .to_frame(name="Count")
        )

    if plot:
        trends.plot(figsize=(12, 6), marker="o")
        plt.title("Olympic Athlete Participation Over Time")
        plt.xlabel("Year")
        plt.ylabel("Number of Athletes")
        plt.show()

    return trends


def medal_trends(df, entity="Team", top_n=5, plot=True):
    """
    Analyze medal trends over time.
    """

    # ✅ Validate columns
    required_cols = {"Year", "Medal", entity}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_cols}")

    medal_df = df[df["Medal"].notnull()]

    medal_counts = (
        medal_df.groupby([entity, "Year"])["Medal"]
        .count()
        .reset_index()
    )

    top_entities = (
        medal_counts.groupby(entity)["Medal"]
        .sum()
        .nlargest(top_n)
        .index
    )

    trends = (
        medal_counts[medal_counts[entity].isin(top_entities)]
        .pivot(index="Year", columns=entity, values="Medal")
        .fillna(0)
    )

    if plot:
        trends.plot(figsize=(12, 6), marker="o")
        plt.title(f"Top {top_n} {entity}s Medal Trends Over Time")
        plt.xlabel("Year")
        plt.ylabel("Number of Medals")
        plt.show()

    return trends


def sport_popularity(df, plot=True):
    """
    Analyze popularity of sports over time.
    """

    # ✅ Validate columns
    required_cols = {"Year", "Sport", "ID"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_cols}")

    trends = (
        df.groupby(["Year", "Sport"])["ID"]
        .nunique()
        .reset_index()
        .pivot(index="Year", columns="Sport", values="ID")
        .fillna(0)
    )

    if plot:
        top_sports = trends.sum().nlargest(5).index
        trends[top_sports].plot(figsize=(12, 6), marker="o")
        plt.title("Top 5 Sports Popularity Over Time")
        plt.xlabel("Year")
        plt.ylabel("Number of Athletes")
        plt.show()

    return trends