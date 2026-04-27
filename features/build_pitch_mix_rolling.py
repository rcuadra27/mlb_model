#!/usr/bin/env python3
"""
features/build_pitchmix_rolling.py

Builds/updates pitcher_pitchmix_rolling and batter_vs_pitchtype_rolling
from statcast_pitches. Safe to run repeatedly — uses upsert.

Run once to backfill, then nightly after statcast ingest:
    PG_DSN=... python features/build_pitchmix_rolling.py --start 2026-03-27
    PG_DSN=... python features/build_pitchmix_rolling.py --date 2026-04-17

These tables feed build_lineup_matchups.py which computes:
    lineup_skill_diff, matchup_diff, lineup_vs_sp_score_diff
— three of the top 5 most important features in the model.
"""

import os
import argparse
import warnings
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from tqdm import tqdm

warnings.filterwarnings("ignore")

WINDOW_DAYS = 365
MIN_PITCHES = 50  # minimum pitches to compute a rolling record

PITCH_TYPE_MAP = {
    "FF": "ff", "SI": "si", "FC": "fc", "SL": "sl",
    "CU": "cu", "CH": "ch", "SP": "sp", "FS": "fs",
    "KC": "cu", "ST": "sl", "SV": "sl", "CS": "cu",
}
PITCH_TYPES = ["ff", "si", "fc", "sl", "cu", "ch", "sp", "fs"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_statcast(engine, schema: str, d0: str, d1: str) -> pd.DataFrame:
    print(f"  Loading statcast {d0} → {d1}...")
    df = pd.read_sql(text(f"""
        SELECT game_date, pitcher, batter, pitch_type,
               estimated_woba_using_speedangle AS xwoba,
               woba_denom
        FROM {schema}.statcast_pitches
        WHERE game_date BETWEEN :d0 AND :d1
          AND game_date IS NOT NULL
          AND pitch_type IS NOT NULL
    """), engine, params={"d0": d0, "d1": d1})
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["pitch_mapped"] = df["pitch_type"].map(PITCH_TYPE_MAP)
    df = df[df["pitch_mapped"].notna()].copy()
    print(f"  Loaded {len(df):,} pitches")
    return df


def load_game_dates(engine, schema: str, start: str, end: str) -> list:
    """Get all unique game dates in range that need updating."""
    df = pd.read_sql(text(f"""
        SELECT DISTINCT game_date
        FROM {schema}.games
        WHERE game_date BETWEEN :start AND :end
          AND home_runs IS NOT NULL
        ORDER BY game_date
    """), engine, params={"start": start, "end": end})
    return pd.to_datetime(df["game_date"]).tolist()


# ---------------------------------------------------------------------------
# Pitcher pitch mix
# ---------------------------------------------------------------------------

def compute_pitcher_pitchmix(
    statcast: pd.DataFrame,
    pitcher_id: int,
    as_of_date: pd.Timestamp,
    window_days: int = WINDOW_DAYS,
) -> dict | None:
    date_min = as_of_date - pd.Timedelta(days=window_days)
    mask = (
        (statcast["pitcher"] == pitcher_id) &
        (statcast["game_date"] >= date_min) &
        (statcast["game_date"] < as_of_date)
    )
    df = statcast[mask]
    n = len(df)
    if n < MIN_PITCHES:
        return None

    counts = df["pitch_mapped"].value_counts()
    total  = counts.sum()

    row = {
        "pitcher_id":  int(pitcher_id),
        "as_of_date":  as_of_date.date(),
        "window_days": window_days,
        "n_pitches":   int(n),
    }
    for pt in PITCH_TYPES:
        row[f"pct_{pt}"] = float(counts.get(pt, 0) / total) if total > 0 else None

    return row


# ---------------------------------------------------------------------------
# Batter vs pitch type skill
# ---------------------------------------------------------------------------

def compute_batter_vs_pitchtype(
    statcast: pd.DataFrame,
    batter_id: int,
    as_of_date: pd.Timestamp,
    window_days: int = WINDOW_DAYS,
) -> dict | None:
    date_min = as_of_date - pd.Timedelta(days=window_days)
    mask = (
        (statcast["batter"] == batter_id) &
        (statcast["game_date"] >= date_min) &
        (statcast["game_date"] < as_of_date) &
        (statcast["woba_denom"] == 1)  # plate appearances only
    )
    df = statcast[mask]
    n = len(df)
    if n < MIN_PITCHES:
        return None

    row = {
        "batter_id":   int(batter_id),
        "as_of_date":  as_of_date.date(),
        "window_days": window_days,
        "n_pitches":   int(n),
    }

    for pt in PITCH_TYPES:
        pt_df = df[df["pitch_mapped"] == pt]
        if len(pt_df) >= 10 and pt_df["xwoba"].notna().any():
            row[f"skill_{pt}"] = float(pt_df["xwoba"].mean())
        else:
            row[f"skill_{pt}"] = None

    return row


# ---------------------------------------------------------------------------
# Upsert helpers
# ---------------------------------------------------------------------------

def upsert_pitchmix(engine, schema: str, rows: list) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    # Ensure pct columns are float not object/text
    for pt in PITCH_TYPES:
        col = f"pct_{pt}"
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    with engine.begin() as conn:
        df.to_sql("_pitchmix_tmp", conn, schema=schema,
                  if_exists="replace", index=False, method="multi")
        conn.execute(text(f"""
            INSERT INTO {schema}.pitcher_pitchmix_rolling
                (pitcher_id, as_of_date, window_days, n_pitches,
                 pct_ff, pct_si, pct_fc, pct_sl, pct_cu, pct_ch, pct_sp, pct_fs)
            SELECT pitcher_id, as_of_date, window_days, n_pitches,
                   pct_ff::real, pct_si::real, pct_fc::real, pct_sl::real,
                   pct_cu::real, pct_ch::real, pct_sp::real, pct_fs::real
            FROM {schema}._pitchmix_tmp
            ON CONFLICT (pitcher_id, as_of_date, window_days) DO UPDATE SET
                n_pitches = EXCLUDED.n_pitches,
                pct_ff = EXCLUDED.pct_ff, pct_si = EXCLUDED.pct_si,
                pct_fc = EXCLUDED.pct_fc, pct_sl = EXCLUDED.pct_sl,
                pct_cu = EXCLUDED.pct_cu, pct_ch = EXCLUDED.pct_ch,
                pct_sp = EXCLUDED.pct_sp, pct_fs = EXCLUDED.pct_fs
        """))
        conn.execute(text(f"DROP TABLE IF EXISTS {schema}._pitchmix_tmp"))


def upsert_batter_skill(engine, schema: str, rows: list) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    # Ensure skill columns are float not object/text
    for pt in PITCH_TYPES:
        col = f"skill_{pt}"
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    with engine.begin() as conn:
        df.to_sql("_batter_tmp", conn, schema=schema,
                  if_exists="replace", index=False, method="multi")
        conn.execute(text(f"""
            INSERT INTO {schema}.batter_vs_pitchtype_rolling
                (batter_id, as_of_date, window_days, n_pitches,
                 skill_ff, skill_si, skill_fc, skill_sl,
                 skill_cu, skill_ch, skill_sp, skill_fs)
            SELECT batter_id, as_of_date, window_days, n_pitches,
                   skill_ff::real, skill_si::real, skill_fc::real, skill_sl::real,
                   skill_cu::real, skill_ch::real, skill_sp::real, skill_fs::real
            FROM {schema}._batter_tmp
            ON CONFLICT (batter_id, as_of_date, window_days) DO UPDATE SET
                n_pitches  = EXCLUDED.n_pitches,
                skill_ff = EXCLUDED.skill_ff, skill_si = EXCLUDED.skill_si,
                skill_fc = EXCLUDED.skill_fc, skill_sl = EXCLUDED.skill_sl,
                skill_cu = EXCLUDED.skill_cu, skill_ch = EXCLUDED.skill_ch,
                skill_sp = EXCLUDED.skill_sp, skill_fs = EXCLUDED.skill_fs
        """))
        conn.execute(text(f"DROP TABLE IF EXISTS {schema}._batter_tmp"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema",      default="public")
    ap.add_argument("--start",       default=None)
    ap.add_argument("--end",         default=None)
    ap.add_argument("--date",        default=None, help="Single date mode")
    ap.add_argument("--window_days", type=int, default=WINDOW_DAYS)
    args = ap.parse_args()

    pg_dsn = os.getenv("PG_DSN")
    if not pg_dsn:
        raise RuntimeError("PG_DSN env var required.")

    engine = create_engine(pg_dsn, pool_pre_ping=True)

    if args.date:
        start_date = end_date = args.date
    else:
        start_date = args.start or "2026-03-27"
        end_date   = args.end   or pd.Timestamp.today().strftime("%Y-%m-%d")

    print(f"Range: {start_date} → {end_date}  window={args.window_days}d")

    # Load statcast with lookback for rolling window
    lookback = (pd.Timestamp(start_date) - pd.Timedelta(days=args.window_days)).strftime("%Y-%m-%d")
    statcast = load_statcast(engine, args.schema, lookback, end_date)

    if statcast.empty:
        print("No statcast data found.")
        return

    # Get game dates to process
    game_dates = load_game_dates(engine, args.schema, start_date, end_date)
    print(f"Processing {len(game_dates)} game dates...")

    for as_of_date in tqdm(game_dates, desc="Dates"):
        # Get active pitchers and batters for this date window
        date_min = as_of_date - pd.Timedelta(days=args.window_days)
        window_sc = statcast[
            (statcast["game_date"] >= date_min) &
            (statcast["game_date"] < as_of_date)
        ]

        if window_sc.empty:
            continue

        # Pitcher pitch mix
        pitcher_rows = []
        for pid in window_sc["pitcher"].unique():
            row = compute_pitcher_pitchmix(statcast, pid, as_of_date, args.window_days)
            if row:
                pitcher_rows.append(row)
        upsert_pitchmix(engine, args.schema, pitcher_rows)

        # Batter vs pitch type
        batter_rows = []
        pa_sc = window_sc[window_sc["woba_denom"] == 1]
        for bid in pa_sc["batter"].unique():
            row = compute_batter_vs_pitchtype(statcast, bid, as_of_date, args.window_days)
            if row:
                batter_rows.append(row)
        upsert_batter_skill(engine, args.schema, batter_rows)

    print("\nDone. Now run build_lineup_matchups.py to update features_game.")


if __name__ == "__main__":
    main()