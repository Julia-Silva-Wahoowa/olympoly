"""monte_carlo_simulation.py

Monte Carlo simulator for Olympic-style outcomes using smoothed, recency-weighted
historical results.

Why this exists
---------------
Raw win-rate estimates (e.g., 1/1 = 1.0) are unstable and misleading.
This script instead builds *event-level categorical probabilities* from
historical winners (or medalists) using:
  • Recency weighting (exponential decay)
  • Dirichlet smoothing (pseudo-counts)
  • Filters for obsolete events / defunct NOCs (optional)
  • Minimum evidence thresholds

It then runs Monte Carlo simulations to produce expected medal tables and
uncertainty intervals.

Usage (CLI)
-----------
python monte_carlo_simulation.py --n_sims 20000 --half_life 12 --min_year 2000 \
  --recent_event_cutoff 2000 --medals_per_event 3 --seed 42

Data loading
------------
By default, the script tries to import and call `load_data.load_data()`.
If that fails, you can provide a CSV path:
  python monte_carlo_simulation.py --csv_path path/to/olympics.csv

Notes
-----
The historical dataset schema varies. This script attempts to infer
common column names (Year, NOC/Country, Event/Sport/Discipline, Medal/Rank).
If inference fails, pass explicit overrides:
  --year_col Year --noc_col NOC --event_col Event --medal_col Medal

Outputs
-------
Creates two CSVs in the current directory by default:
  • mc_expected_medals.csv  (expected medals by NOC)
  • mc_medal_quantiles.csv  (5/50/95% quantiles by NOC)

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Utilities
# -----------------------------

def _pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def exponential_recency_weight(years: pd.Series, current_year: int, half_life: float) -> pd.Series:
    """Exponential decay: halves every `half_life` years."""
    age = (current_year - years).clip(lower=0)
    return np.power(0.5, age / float(half_life))


def sequential_sample_without_replacement(rng: np.random.Generator, items: np.ndarray, probs: np.ndarray, k: int) -> List[str]:
    """Sample k distinct items sequentially without replacement using re-normalized probs."""
    chosen = []
    available_items = items.copy()
    available_probs = probs.copy().astype(float)

    for _ in range(min(k, len(available_items))):
        total = available_probs.sum()
        if total <= 0:
            break
        p = available_probs / total
        idx = rng.choice(len(available_items), p=p)
        chosen.append(str(available_items[idx]))
        available_items = np.delete(available_items, idx)
        available_probs = np.delete(available_probs, idx)

    return chosen


# -----------------------------
# Configuration
# -----------------------------

@dataclass
class Schema:
    year_col: str
    noc_col: str
    event_col: str
    medal_col: Optional[str] = None
    rank_col: Optional[str] = None


@dataclass
class BuildParams:
    min_year: int = 2000
    recent_event_cutoff: int = 2000
    half_life: float = 12.0
    dirichlet_alpha: float = 1.0
    min_effective_event_weight: float = 3.0
    drop_defunct_nocs: bool = True
    defunct_nocs: Tuple[str, ...] = ("URS", "EUN", "YUG", "TCH", "FRG", "GDR", "ANZ", "AHO", "BOH", "EUA")


@dataclass
class SimParams:
    n_sims: int = 10000
    medals_per_event: int = 3  # 1 = gold only, 3 = gold/silver/bronze
    seed: int = 42


# -----------------------------
# Data loading
# -----------------------------

def load_dataframe(csv_path: Optional[str] = None) -> pd.DataFrame:
    """Load olympic dataset from load_data.load_data() or a CSV path."""
    if csv_path:
        return pd.read_csv(csv_path)

    # Try repository loader
    try:
        from olympoly.load_data import load_data  # type: ignore
        df = load_data()
        if not isinstance(df, pd.DataFrame):
            raise TypeError("load_data.load_data() did not return a pandas DataFrame")
        return df
    except Exception as e:
        raise RuntimeError(
            "Failed to load data via load_data.load_data(). "
            "Provide --csv_path to a local CSV. Underlying error: " + str(e)
        )


def infer_schema(df: pd.DataFrame,
                 year_col: Optional[str] = None,
                 noc_col: Optional[str] = None,
                 event_col: Optional[str] = None,
                 medal_col: Optional[str] = None,
                 rank_col: Optional[str] = None) -> Schema:
    """Infer likely schema from common column names."""
    year_col = year_col or _pick_first_existing(df, ["Year", "year", "Edition", "GamesYear"])
    noc_col = noc_col or _pick_first_existing(df, ["NOC", "noc", "Team", "Country", "country", "NOC_code"])

    # Event can be a direct column or composite
    event_col = event_col or _pick_first_existing(df, ["Event", "event", "Discipline", "Sport", "SportEvent"])

    medal_col = medal_col or _pick_first_existing(df, ["Medal", "medal", "medal_type"])
    rank_col = rank_col or _pick_first_existing(df, ["Rank", "rank", "Place", "place", "Position"])

    if year_col is None or noc_col is None:
        raise ValueError(
            "Could not infer required columns. "
            "Please pass --year_col and --noc_col (and optionally --event_col)."
        )

    if event_col is None:
        # Try to build a composite event key
        sport = _pick_first_existing(df, ["Sport", "sport"])
        discipline = _pick_first_existing(df, ["Discipline", "discipline"])
        event = _pick_first_existing(df, ["Event", "event"])
        parts = [c for c in [sport, discipline, event] if c]
        if not parts:
            raise ValueError("Could not infer event column or build a composite event key.")
        # We'll create one later
        event_col = "__EVENT_KEY__"

    return Schema(year_col=year_col, noc_col=noc_col, event_col=event_col, medal_col=medal_col, rank_col=rank_col)


def ensure_event_key(df: pd.DataFrame, schema: Schema) -> Tuple[pd.DataFrame, Schema]:
    """If schema.event_col points to synthetic key, build it from available columns."""
    if schema.event_col != "__EVENT_KEY__":
        return df, schema

    sport = _pick_first_existing(df, ["Sport", "sport"])
    discipline = _pick_first_existing(df, ["Discipline", "discipline"])
    event = _pick_first_existing(df, ["Event", "event"])
    parts = [c for c in [sport, discipline, event] if c]

    if not parts:
        raise ValueError("Cannot build event key; no Sport/Discipline/Event columns found.")

    df = df.copy()
    df["__EVENT_KEY__"] = df[parts].astype(str).agg(" | ".join, axis=1)
    return df, schema


# -----------------------------
# Build event probability tables
# -----------------------------

def is_gold_row(row: pd.Series, medal_col: Optional[str], rank_col: Optional[str]) -> bool:
    """Determine if a row indicates an event win (gold)."""
    if medal_col and medal_col in row.index:
        m = str(row[medal_col]).strip().lower()
        if m in ("gold", "g"):
            return True

    if rank_col and rank_col in row.index:
        try:
            return int(row[rank_col]) == 1
        except Exception:
            return False

    # Fallback: if no medal/rank, treat every row as a 'participation' (not recommended)
    return False


def build_event_country_probs(df: pd.DataFrame, schema: Schema, bp: BuildParams) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (event_probs, event_meta).

    event_probs columns: [Event, NOC, prob]
    event_meta columns: [Event, last_year, total_weight]
    """

    df = df.copy()

    # Normalize year
    df[schema.year_col] = pd.to_numeric(df[schema.year_col], errors="coerce")
    df = df.dropna(subset=[schema.year_col, schema.noc_col, schema.event_col])
    df[schema.year_col] = df[schema.year_col].astype(int)

    # Filter recent window
    df = df[df[schema.year_col] >= bp.min_year]

    # Drop defunct NOCs
    if bp.drop_defunct_nocs:
        df = df[~df[schema.noc_col].astype(str).isin(set(bp.defunct_nocs))]

    current_year = int(df[schema.year_col].max())
    df["weight"] = exponential_recency_weight(df[schema.year_col], current_year=current_year, half_life=bp.half_life)

    # Determine winners (gold) rows
    if schema.medal_col is None and schema.rank_col is None:
        raise ValueError(
            "This simulator needs either a Medal column or a Rank column to identify winners. "
            "Pass --medal_col or --rank_col."
        )

    # Create a boolean win indicator per row (vectorized for speed)
    if schema.medal_col and schema.medal_col in df.columns:
        df["is_gold"] = df[schema.medal_col].astype(str).str.strip().str.lower().isin(["gold", "g"])
    elif schema.rank_col and schema.rank_col in df.columns:
        df["is_gold"] = pd.to_numeric(df[schema.rank_col], errors="coerce") == 1
    else:
        df["is_gold"] = False
winners = df[df["is_gold"]].copy()
    if winners.empty:
        raise ValueError(
            "No gold/winner rows found. Check your --medal_col/--rank_col settings or dataset contents."
        )

    # Remove duplicates where athlete-level data could repeat team golds.
    # Keep one (Year, Event, NOC) record.
    winners = winners.drop_duplicates(subset=[schema.year_col, schema.event_col, schema.noc_col])

    # Event activity filter to remove obsolete events
    event_last_year = winners.groupby(schema.event_col)[schema.year_col].max()
    active_events = event_last_year[event_last_year >= bp.recent_event_cutoff].index
    winners = winners[winners[schema.event_col].isin(active_events)]

    # Weighted counts of golds per (Event, NOC)
    counts = winners.groupby([schema.event_col, schema.noc_col], as_index=False)["weight"].sum()
    counts = counts.rename(columns={schema.event_col: "Event", schema.noc_col: "NOC", "weight": "gold_weight"})

    # Event totals
    totals = counts.groupby("Event", as_index=False)["gold_weight"].sum().rename(columns={"gold_weight": "total_weight"})

    # Minimum evidence per event
    totals = totals[totals["total_weight"] >= bp.min_effective_event_weight]
    counts = counts[counts["Event"].isin(set(totals["Event"]))]

    # Dirichlet smoothing:
    # prob = (gold_weight + alpha) / (total_weight + alpha * K)
    # where K = number of countries observed in the event.
    k_by_event = counts.groupby("Event")["NOC"].nunique().rename("K").reset_index()
    tmp = counts.merge(totals, on="Event", how="left").merge(k_by_event, on="Event", how="left")

    alpha = float(bp.dirichlet_alpha)
    tmp["prob"] = (tmp["gold_weight"] + alpha) / (tmp["total_weight"] + alpha * tmp["K"].clip(lower=1))

    # Clip for numerical stability
    tmp["prob"] = tmp["prob"].clip(1e-9, 1 - 1e-9)

    # Normalize within each event (because we only include observed NOCs)
    tmp["prob"] = tmp["prob"] / tmp.groupby("Event")["prob"].transform("sum")

    event_meta = winners.groupby(schema.event_col, as_index=False).agg(
        last_year=(schema.year_col, "max"),
        n_editions=(schema.year_col, "nunique")
    ).rename(columns={schema.event_col: "Event"})

    event_meta = event_meta.merge(totals, on="Event", how="left")

    event_probs = tmp[["Event", "NOC", "prob", "gold_weight"]].sort_values(["Event", "prob"], ascending=[True, False])

    return event_probs, event_meta


# -----------------------------
# Monte Carlo simulation
# -----------------------------

def run_monte_carlo(event_probs: pd.DataFrame, sp: SimParams) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run Monte Carlo; return (expected_table, quantiles_table)."""
    rng = np.random.default_rng(sp.seed)

    # Build per-event arrays for faster sampling
    events = event_probs["Event"].unique()
    per_event = {}
    for ev in events:
        sub = event_probs[event_probs["Event"] == ev]
        items = sub["NOC"].to_numpy()
        probs = sub["prob"].to_numpy(dtype=float)
        probs = probs / probs.sum()
        per_event[ev] = (items, probs)

    # Collect medal counts per simulation
    all_nocs = np.unique(event_probs["NOC"].to_numpy())
    noc_index = {noc: i for i, noc in enumerate(all_nocs)}

    # medal arrays: sims x nocs
    gold = np.zeros((sp.n_sims, len(all_nocs)), dtype=np.int32)
    silver = np.zeros_like(gold)
    bronze = np.zeros_like(gold)

    for s in range(sp.n_sims):
        for ev in events:
            items, probs = per_event[ev]
            winners = sequential_sample_without_replacement(rng, items, probs, k=sp.medals_per_event)
            if not winners:
                continue

            # Assign medals
            if len(winners) >= 1:
                gold[s, noc_index[winners[0]]] += 1
            if sp.medals_per_event >= 3:
                if len(winners) >= 2:
                    silver[s, noc_index[winners[1]]] += 1
                if len(winners) >= 3:
                    bronze[s, noc_index[winners[2]]] += 1

    total = gold + silver + bronze

    # Expected values
    expected = pd.DataFrame({
        "NOC": all_nocs,
        "Gold_exp": gold.mean(axis=0),
        "Silver_exp": silver.mean(axis=0),
        "Bronze_exp": bronze.mean(axis=0),
        "Total_exp": total.mean(axis=0),
    }).sort_values(["Gold_exp", "Total_exp"], ascending=False)

    # Quantiles
    def q(arr, qv):
        return np.quantile(arr, qv, axis=0)

    qs = [0.05, 0.50, 0.95]
    quant = pd.DataFrame({
        "NOC": all_nocs,
        "Gold_p05": q(gold, qs[0]),
        "Gold_p50": q(gold, qs[1]),
        "Gold_p95": q(gold, qs[2]),
        "Total_p05": q(total, qs[0]),
        "Total_p50": q(total, qs[1]),
        "Total_p95": q(total, qs[2]),
    }).sort_values("Gold_p50", ascending=False)

    return expected, quant


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monte Carlo simulation with smoothed event probabilities")

    p.add_argument("--csv_path", type=str, default=None, help="Path to local Olympics CSV (optional)")

    # Schema overrides
    p.add_argument("--year_col", type=str, default=None, help="Year column name")
    p.add_argument("--noc_col", type=str, default=None, help="Country/NOC column name")
    p.add_argument("--event_col", type=str, default=None, help="Event column name")
    p.add_argument("--medal_col", type=str, default=None, help="Medal column name")
    p.add_argument("--rank_col", type=str, default=None, help="Rank/Place column name")

    # Probability build params
    p.add_argument("--min_year", type=int, default=2000)
    p.add_argument("--recent_event_cutoff", type=int, default=2000)
    p.add_argument("--half_life", type=float, default=12.0)
    p.add_argument("--dirichlet_alpha", type=float, default=1.0)
    p.add_argument("--min_effective_event_weight", type=float, default=3.0)
    p.add_argument("--keep_defunct_nocs", action="store_true", help="Do NOT drop defunct NOCs")

    # Simulation params
    p.add_argument("--n_sims", type=int, default=10000)
    p.add_argument("--medals_per_event", type=int, default=3, choices=[1, 3])
    p.add_argument("--seed", type=int, default=42)

    # Outputs
    p.add_argument("--out_expected", type=str, default="mc_expected_medals.csv")
    p.add_argument("--out_quantiles", type=str, default="mc_medal_quantiles.csv")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    df = load_dataframe(args.csv_path)

    schema = infer_schema(
        df,
        year_col=args.year_col,
        noc_col=args.noc_col,
        event_col=args.event_col,
        medal_col=args.medal_col,
        rank_col=args.rank_col,
    )
    df, schema = ensure_event_key(df, schema)

    bp = BuildParams(
        min_year=args.min_year,
        recent_event_cutoff=args.recent_event_cutoff,
        half_life=args.half_life,
        dirichlet_alpha=args.dirichlet_alpha,
        min_effective_event_weight=args.min_effective_event_weight,
        drop_defunct_nocs=not args.keep_defunct_nocs,
    )

    event_probs, event_meta = build_event_country_probs(df, schema, bp)

    sp = SimParams(
        n_sims=args.n_sims,
        medals_per_event=args.medals_per_event,
        seed=args.seed,
    )

    expected, quantiles = run_monte_carlo(event_probs, sp)

    expected.to_csv(args.out_expected, index=False)
    quantiles.to_csv(args.out_quantiles, index=False)

    # Small console preview
    print("\nTop 15 by expected gold:")
    print(expected.head(15).to_string(index=False))
    print("\nTop 15 by median gold (p50):")
    print(quantiles.head(15).to_string(index=False))

    # Save probability table for inspection
    event_probs.to_csv("event_country_probs.csv", index=False)
    event_meta.to_csv("event_meta.csv", index=False)

    print("\nWrote:")
    print(f"  {args.out_expected}")
    print(f"  {args.out_quantiles}")
    print("  event_country_probs.csv")
    print("  event_meta.csv")


if __name__ == "__main__":
    main()
