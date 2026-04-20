# market_data.py

"""
market_data.py - Load and process prediction market data
for Olympics-related contracts (Kalshi / Polymarket style).
"""

from __future__ import annotations
import pandas as pd


REQUIRED_COLUMNS = {"market", "event", "outcome", "price", "timestamp"}


def load_market_data(filepath):
    """
    Load market data from CSV.

    Expected columns:
    - market (str): platform name (e.g., kalshi, polymarket)
    - event (str): event description
    - outcome (str): outcome label (e.g., YES/NO)
    - price (float): price between 0 and 1
    - timestamp (datetime or str)

    Returns:
    - pd.DataFrame
    """
    df = pd.read_csv(filepath)

    validate_market_data(df)

    # Ensure correct types
    df["price"] = df["price"].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df


def validate_market_data(df):
    """
    Validate required columns and value ranges.
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if not df["price"].between(0, 1).all():
        raise ValueError("All prices must be between 0 and 1")


def normalize_prices(df):
    """
    Normalize prices so YES/NO outcomes sum to 1 per event + timestamp.

    Returns:
    - normalized DataFrame
    """
    df = df.copy()

    grouped = df.groupby(["event", "timestamp"])["price"].transform("sum")
    df["normalized_price"] = df["price"] / grouped

    return df


def merge_market_data(df1, df2):
    """
    Merge two market datasets.

    Returns:
    - combined DataFrame
    """
    combined = pd.concat([df1, df2], ignore_index=True)
    validate_market_data(combined)
    return combined


def get_latest_prices(df):
    """
    Get latest price per event + outcome.

    Returns:
    - pd.DataFrame
    """
    idx = df.groupby(["event", "outcome"])["timestamp"].idxmax()
    return df.loc[idx].reset_index(drop=True)



from typing import Any
import requests
import pandas as pd

GAMMA_BASE_URL = "https://gamma-api.polymarket.com"


class PolymarketAPIError(RuntimeError):
    """Raised when a Polymarket API request fails."""


def _get_json(path: str, params: dict[str, Any] | None = None) -> Any:
    url = f"{GAMMA_BASE_URL}{path}"
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise PolymarketAPIError(f"Request to {url} failed: {exc}") from exc


def search_polymarket(query: str) -> pd.DataFrame:
    """
    Search Polymarket for matching events/markets and return a DataFrame.
    """
    data = _get_json("/public-search", params={"q": query})

    if isinstance(data, dict):
        for key in ("results", "data", "events", "markets"):
            if key in data and isinstance(data[key], list):
                return pd.DataFrame(data[key])

    if isinstance(data, list):
        return pd.DataFrame(data)

    return pd.DataFrame()


def fetch_active_events(limit: int = 200, offset: int = 0) -> pd.DataFrame:
    """
    Fetch active Polymarket events.
    """
    data = _get_json(
        "/events",
        params={
            "active": "true",
            "closed": "false",
            "limit": limit,
            "offset": offset,
        },
    )

    if isinstance(data, list):
        return pd.DataFrame(data)

    if isinstance(data, dict):
        for key in ("events", "data", "results"):
            if key in data and isinstance(data[key], list):
                return pd.DataFrame(data[key])

    return pd.DataFrame()


def _safe_float(value: Any) -> float | None:
    """
    Convert API values to float when possible.
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None

    return None


def extract_market_odds(events_df: pd.DataFrame, query: str) -> pd.DataFrame:
    """
    Filter active events to the user's query and return tidy odds data.
    """
    columns = [
        "event_title",
        "market_question",
        "yes_price",
        "no_price",
        "yes_percent",
        "no_percent",
        "slug",
    ]

    if events_df.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    query_lower = query.lower().strip()

    for _, event_row in events_df.iterrows():
        event_title = str(
            event_row.get("title")
            or event_row.get("name")
            or event_row.get("slug")
            or ""
        )

        markets = event_row.get("markets") or event_row.get("Markets") or []
        event_matches = query_lower in event_title.lower()

        if not isinstance(markets, list):
            continue

        for market in markets:
            if not isinstance(market, dict):
                continue

            question = str(
                market.get("question")
                or market.get("title")
                or market.get("name")
                or ""
            )
            slug = market.get("slug")

            if not event_matches and query_lower not in question.lower():
                continue

            # Try common Gamma fields
            yes_price = _safe_float(
                market.get("bestAsk")
                or market.get("price")
                or market.get("lastTradePrice")
            )
            no_price = _safe_float(market.get("bestBid"))

            # If outcomePrices exists, try to use it
            outcome_prices = market.get("outcomePrices")
            if yes_price is None and isinstance(outcome_prices, list) and len(outcome_prices) >= 1:
                yes_price = _safe_float(outcome_prices[0])
            if no_price is None and isinstance(outcome_prices, list) and len(outcome_prices) >= 2:
                no_price = _safe_float(outcome_prices[1])

            rows.append(
                {
                    "event_title": event_title,
                    "market_question": question,
                    "yes_price": yes_price,
                    "no_price": no_price,
                    "yes_percent": round(yes_price * 100, 2) if yes_price is not None else None,
                    "no_percent": round(no_price * 100, 2) if no_price is not None else None,
                    "slug": slug,
                }
            )

    if not rows:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(rows).sort_values(
        by=["event_title", "market_question"]
    ).reset_index(drop=True)


def get_current_odds_for_event(query: str, limit: int = 200) -> pd.DataFrame:
    """
    Main function:
    1. fetch active Polymarket events
    2. filter by user query
    3. return a tidy DataFrame
    """
    events_df = fetch_active_events(limit=limit)
    odds_df = extract_market_odds(events_df, query=query)

    if not odds_df.empty:
        return odds_df

    # Fallback: raw search results
    return search_polymarket(query)