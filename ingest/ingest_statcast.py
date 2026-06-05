#!/usr/bin/env python3
"""
ingest_statcast.py

Pulls yesterday's statcast pitch-level data from Baseball Savant
and inserts into the statcast_pitches table. Run daily at 4:45pm PT
— by then yesterday's games are fully processed by Baseball Savant.

This keeps lineup features current throughout the season. Without
daily statcast ingestion, lineup xwOBA/K-rate/barrel-rate features
are frozen at end-of-2025 values and miss:
  - New acquisitions from offseason/trades
  - Players having breakout or decline seasons
  - Spring training → regular season performance shifts

Usage:
    PG_DSN=... python ingest_statcast.py --date 2026-04-01  # ingest specific date
    PG_DSN=... python ingest_statcast.py                     # ingest yesterday
    PG_DSN=... python ingest_statcast.py --start 2026-04-01 --end 2026-04-30  # range
"""

import os
import argparse
import io
import time
import hashlib
import requests
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

SAVANT_URL = "https://baseballsavant.mlb.com/statcast_search/csv"

# Columns we need from Baseball Savant (aligned with statcast_pitches table)
SAVANT_COLS = [
    "game_date", "game_pk", "at_bat_number", "pitch_number",
    "pitcher", "batter", "stand", "p_throws",
    "pitch_type", "events", "bb_type",
    "estimated_woba_using_speedangle", "woba_denom",
    "launch_speed", "launch_angle",
    "home_team", "away_team", "inning", "inning_topbot",
]

# Pitch type normalization
PITCH_TYPE_MAP = {
    "FF": "FF", "SI": "SI", "FC": "FC", "SL": "SL",
    "CU": "CU", "CH": "CH", "SP": "SP", "FS": "FS",
    "KC": "CU", "ST": "SL", "SV": "SL", "CS": "CU",
}


def make_row_id(df: pd.DataFrame) -> pd.Series:
    """Deterministic row key — must match ingest/statcast_pitches.py."""
    keys = (
        df["game_date"].astype(str).fillna("") + "|" +
        df["game_pk"].astype("Int64").astype(str).fillna("") + "|" +
        df["at_bat_number"].astype("Int64").astype(str).fillna("") + "|" +
        df["pitch_number"].astype("Int64").astype(str).fillna("") + "|" +
        df["pitcher"].astype("Int64").astype(str).fillna("") + "|" +
        df["batter"].astype("Int64").astype(str).fillna("")
    )
    return keys.map(lambda s: hashlib.md5(s.encode("utf-8")).hexdigest())


def fetch_statcast_for_date(date_str: str) -> pd.DataFrame:
    """
    Pull statcast data for a single date from Baseball Savant CSV export.
    Returns empty DataFrame if no data available yet.
    """
    params = {
        "all":            "true",
        "hfGT":           "R|",          # regular season only
        "game_date_gt":   date_str,
        "game_date_lt":   date_str,
        "player_type":    "pitcher",
        "hfZ":            "",
        "stadium":        "",
        "hfBBL":          "",
        "hfNewZones":     "",
        "hfPull":         "",
        "hfC":            "",
        "hfSea":          f"{date_str[:4]}|",
        "hfSit":          "",
        "hfOuts":         "",
        "opponent":       "",
        "pitcher_throws": "",
        "batter_stands":  "",
        "hfSA":           "",
        "hfInfield":      "",
        "hfOutfield":     "",
        "hfRO":           "",
        "home_road":      "",
        "hfFlag":         "",
        "hfBBT":          "",
        "metric_1":       "",
        "hfInn":          "",
        "min_pitches":    "0",
        "min_results":    "0",
        "group_by":       "name",
        "sort_col":       "pitches",
        "player_event_sort": "api_p_release_speed",
        "sort_order":     "desc",
        "min_pas":        "0",
        "type":           "details",
        "is_shift":       "",
    }

    try:
        print(f"  Fetching statcast from Baseball Savant for {date_str}...")
        resp = requests.get(SAVANT_URL, params=params, timeout=60)
        resp.raise_for_status()

        if not resp.content or len(resp.content) < 100:
            print(f"  No data returned for {date_str} — game may not be processed yet")
            return pd.DataFrame()

        df = pd.read_csv(io.StringIO(resp.text), low_memory=False)

        if df.empty or "pitcher" not in df.columns:
            print(f"  Empty response for {date_str}")
            return pd.DataFrame()

        print(f"  Fetched {len(df):,} pitches for {date_str}")
        return df

    except Exception as e:
        print(f"  Savant fetch error for {date_str}: {e}")
        return pd.DataFrame()


def clean_statcast(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize statcast data for DB insertion."""
    if df.empty:
        return df

    # Keep only needed columns that exist
    keep = [c for c in SAVANT_COLS if c in df.columns]
    df = df[keep].copy()

    # Normalize types
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
    for col in ["game_pk", "at_bat_number", "pitch_number", "pitcher", "batter", "inning"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # Normalize pitch types
    if "pitch_type" in df.columns:
        df["pitch_type"] = df["pitch_type"].map(
            lambda x: PITCH_TYPE_MAP.get(str(x).upper(), x) if pd.notna(x) else None
        )

    # Numeric columns
    for col in ["estimated_woba_using_speedangle", "woba_denom", "launch_speed", "launch_angle"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing identifiers required for row_id + dedup
    required = ["pitcher", "batter", "game_date", "game_pk", "at_bat_number", "pitch_number"]
    df = df.dropna(subset=[c for c in required if c in df.columns])
    if df.empty:
        return df

    df["row_id"] = make_row_id(df)
    return df.dropna(subset=["row_id"])


def upsert_statcast(engine, schema: str, df: pd.DataFrame) -> None:
    """Insert new statcast rows, skipping duplicates."""
    if df.empty:
        return

    # Only keep columns that exist in DB
    with engine.connect() as conn:
        db_cols = pd.read_sql(
            text("SELECT column_name FROM information_schema.columns "
                 "WHERE table_schema = :s AND table_name = 'statcast_pitches'"),
            conn, params={"s": schema}
        )["column_name"].tolist()

    insert_cols = [c for c in df.columns if c in db_cols]
    if "row_id" not in insert_cols:
        print("  row_id missing — cannot insert into statcast_pitches")
        return

    df_insert = df[insert_cols].copy()

    # Insert in chunks to avoid memory issues
    chunk_size = 5000
    total_inserted = 0

    for i in range(0, len(df_insert), chunk_size):
        chunk = df_insert.iloc[i:i + chunk_size]
        with engine.begin() as conn:
            chunk.to_sql("_statcast_tmp", conn, schema=schema,
                         if_exists="replace", index=False, method="multi")

            # Check which columns the temp table has
            tmp_cols = pd.read_sql(
                text("SELECT column_name FROM information_schema.columns "
                     "WHERE table_schema = :s AND table_name = '_statcast_tmp'"),
                conn, params={"s": schema}
            )["column_name"].tolist()

            common_cols = [c for c in insert_cols if c in tmp_cols]
            col_str = ", ".join(common_cols)

            result = conn.execute(text(f"""
                INSERT INTO {schema}.statcast_pitches ({col_str})
                SELECT {col_str} FROM {schema}._statcast_tmp
                ON CONFLICT (row_id) DO NOTHING
            """))
            conn.execute(text(f"DROP TABLE IF EXISTS {schema}._statcast_tmp"))
            total_inserted += result.rowcount

    print(f"  Inserted {total_inserted:,} new statcast rows")


def ingest_date(engine, schema: str, date_str: str) -> None:
    """Ingest statcast data for a single date."""
    df_raw   = fetch_statcast_for_date(date_str)
    if df_raw.empty:
        return
    df_clean = clean_statcast(df_raw)
    if df_clean.empty:
        return
    upsert_statcast(engine, schema, df_clean)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date",   default=None,
                    help="Specific date to ingest (YYYY-MM-DD). "
                         "Defaults to yesterday.")
    ap.add_argument("--start",  default=None,
                    help="Start date for range ingestion")
    ap.add_argument("--end",    default=None,
                    help="End date for range ingestion")
    ap.add_argument("--schema", default="public")
    ap.add_argument("--sleep",  type=float, default=3.0,
                    help="Seconds between requests for range ingestion")
    args = ap.parse_args()

    pg_dsn = os.getenv("PG_DSN")
    if not pg_dsn:
        raise RuntimeError("PG_DSN env var required.")

    engine = create_engine(pg_dsn, pool_pre_ping=True)

    if args.start and args.end:
        # Range ingestion
        dates = pd.date_range(args.start, args.end, freq="D")
        print(f"Ingesting statcast for {len(dates)} dates: "
              f"{args.start} → {args.end}")
        for d in dates:
            date_str = d.strftime("%Y-%m-%d")
            print(f"\n── {date_str} ──")
            ingest_date(engine, args.schema, date_str)
            time.sleep(args.sleep)
    else:
        # Single date — default to yesterday
        if args.date:
            date_str = args.date
        else:
            date_str = (pd.Timestamp.today() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        print(f"Ingesting statcast for {date_str}")
        ingest_date(engine, args.schema, date_str)

    print("\nDone.")


if __name__ == "__main__":
    main()