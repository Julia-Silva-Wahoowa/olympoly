import numpy as np
import pandas as pd


def medal_probability(df, group_cols, medal_type="Gold"):
    """
    Estimate probability of winning a specific medal given grouping columns.

    Args:
        df: pandas DataFrame
        group_cols: list (e.g., ['NOC', 'Event'])
        medal_type: 'Gold', 'Silver', 'Bronze'

    Returns:
        DataFrame with probabilities
    """
    df = df.copy()

    df['is_medal'] = (df['Medal'] == medal_type).astype(int)

    grouped = df.groupby(group_cols).agg(
        total_entries=('ID', 'count'),
        total_wins=('is_medal', 'sum')
    ).reset_index()

    grouped['probability'] = grouped['total_wins'] / grouped['total_entries']

    return grouped

def athlete_medal_probability(df):
    """
    Estimate probability of any medal for each athlete based on appearances.

    Args:
        df: pandas DataFrame with 'Name', 'ID', and 'Medal' columns

    Returns:
        DataFrame with athlete name, appearances, medals, and probability
    """
    df = df.copy()
    df['is_medal'] = df['Medal'].notna().astype(int)

    grouped = df.groupby('Name').agg(
        appearances=('ID', 'count'),
        medals=('is_medal', 'sum')
    ).reset_index()

    grouped['probability'] = grouped['medals'] / grouped['appearances']

    return grouped

def weighted_medal_probability(df, group_cols, current_year, decay=0.9):
    df = df.copy()

    df['years_ago'] = current_year - df['Year']
    df['weight'] = decay ** df['years_ago']
    df['is_gold'] = (df['Medal'] == 'Gold').astype(int)

    grouped = df.groupby(group_cols).apply(
        lambda x: pd.Series({
            'weighted_wins': (x['is_gold'] * x['weight']).sum(),
            'weighted_total': x['weight'].sum()
        })
    ).reset_index()

    grouped['probability'] = grouped['weighted_wins'] / grouped['weighted_total']

    return grouped

def exponential_recency_weight(years: pd.Series, current_year: int, half_life: float = 12.0) -> pd.Series:
    """Exponential decay weights: halves every `half_life` years."""
    age = (current_year - years).clip(lower=0)
    return np.power(0.5, age / half_life)

def compute_event_priors(df: pd.DataFrame, win_col: str, weight_col: str, k: float = 8.0) -> pd.DataFrame:
    """
    Empirical Bayes priors per Event.
    Prior mean = weighted base win-rate in that event.
    Prior strength = k (pseudo-observations).
    """
    g = df.groupby("Event", as_index=False).apply(
        lambda x: pd.Series({
            "event_win_rate": (x[win_col] * x[weight_col]).sum() / max(x[weight_col].sum(), 1e-12),
            "event_weight_sum": x[weight_col].sum()
        })
    ).reset_index(drop=True)

    # Convert mean to Beta(alpha,beta) with total mass k:
    g["alpha0"] = (g["event_win_rate"] * k).clip(lower=1e-6)
    g["beta0"]  = ((1 - g["event_win_rate"]) * k).clip(lower=1e-6)
    return g[["Event", "alpha0", "beta0", "event_win_rate"]]

def beta_binomial_posterior_mean(w: float, n: float, alpha: float, beta: float) -> float:
    return (w + alpha) / (n + alpha + beta)

def compute_country_event_probabilities(
    df: pd.DataFrame,
    *,
    year_col: str = "Year",
    noc_col: str = "NOC",
    event_col: str = "Event",
    win_col: str = "Win",
    current_year: int | None = None,

    # filters
    min_effective_n: float = 8.0,
    min_year: int = 2000,
    require_recent_event: bool = True,
    recent_event_cutoff_year: int = 2000,
    drop_defunct_nocs: bool = True,
    defunct_nocs: set | None = None,

    # smoothing
    prior_strength_k: float = 8.0,   # pseudo-count mass per event
    fallback_laplace: bool = True,   # if event prior cannot be computed
    laplace_alpha: float = 1.0,
    laplace_beta: float = 1.0,

    # recency
    half_life_years: float = 12.0
) -> pd.DataFrame:
    """
    Returns posterior-smoothed probabilities for (NOC, Event) using:
      - recency weighting
      - event-level empirical Bayes Beta prior
      - minimum effective sample size filter
      - optional filters for obsolete events and defunct NOCs
    """

    df = df.copy()

    # Basic column checks
    required = {year_col, noc_col, event_col, win_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Current year default: max year in data
    if current_year is None:
        current_year = int(df[year_col].max())

    # Optional filters: remove very old data
    if min_year is not None:
        df = df[df[year_col] >= min_year]

    # Optional: drop defunct NOCs directly
    if drop_defunct_nocs:
        if defunct_nocs is None:
            defunct_nocs = {"URS", "EUN", "YUG", "TCH", "FRG", "GDR", "ANZ", "AHO", "BOH", "EUA"}  # adjust as needed
        df = df[~df[noc_col].isin(defunct_nocs)]

    # Optional: keep only events contested recently (helps kill Tug-of-War etc.)
    if require_recent_event:
        last_year_by_event = df.groupby(event_col)[year_col].max()
        active_events = last_year_by_event[last_year_by_event >= recent_event_cutoff_year].index
        df = df[df[event_col].isin(active_events)]

    # Recency weights
    df["weight"] = exponential_recency_weight(df[year_col], current_year=current_year, half_life=half_life_years)

    # Compute event priors (Empirical Bayes)
    event_priors = compute_event_priors(df.rename(columns={event_col: "Event"}), win_col=win_col, weight_col="weight", k=prior_strength_k)
    df = df.merge(event_priors, left_on=event_col, right_on="Event", how="left").drop(columns=["Event"])

    # Aggregate weighted wins and weighted trials by (NOC, Event)
    agg = df.groupby([noc_col, event_col], as_index=False).apply(
        lambda x: pd.Series({
            "wins_w": float((x[win_col] * x["weight"]).sum()),
            "n_w": float(x["weight"].sum()),
            "last_year": int(x[year_col].max()),
            "alpha0": float(x["alpha0"].iloc[0]) if pd.notna(x["alpha0"].iloc[0]) else np.nan,
            "beta0": float(x["beta0"].iloc[0]) if pd.notna(x["beta0"].iloc[0]) else np.nan,
        })
    ).reset_index(drop=True)

    # Minimum effective sample filter
    agg = agg[agg["n_w"] >= min_effective_n].copy()

    # Posterior mean with event prior; fallback to Laplace if missing
    def posterior(row):
        if pd.notna(row["alpha0"]) and pd.notna(row["beta0"]):
            return beta_binomial_posterior_mean(row["wins_w"], row["n_w"], row["alpha0"], row["beta0"])
        if fallback_laplace:
            return beta_binomial_posterior_mean(row["wins_w"], row["n_w"], laplace_alpha, laplace_beta)
        return row["wins_w"] / row["n_w"] if row["n_w"] > 0 else np.nan

    agg["prob"] = agg.apply(posterior, axis=1)

    # Never output exactly 0 or 1 (helps downstream sims/logits)
    agg["prob"] = agg["prob"].clip(1e-6, 1 - 1e-6)

    # Helpful diagnostics
    agg["effective_n"] = agg["n_w"]
    agg["wins_effective"] = agg["wins_w"]

    # Sort: strongest signals first, not rare artifacts
    agg = agg.sort_values(["prob", "effective_n", "last_year"], ascending=[False, False, False])

    return agg