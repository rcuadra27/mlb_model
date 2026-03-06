#!/usr/bin/env python3
"""
ingest/statcast_pitches.py

Season-by-season Statcast pitch-level ingestion using pybaseball.

Usage:
  export PG_DSN="postgresql+psycopg2://USER:PASS@localhost:5432/mlb_model"
  python ingest/statcast_pitches.py --season 2015
  python ingest/statcast_pitches.py --season 2015 --season_end 2015-11-15
  python ingest/statcast_pitches.py --season 2015 --months 3-11
"""

import os
import argparse
import hashlib
import datetime as dt

import pandas as pd
from tqdm import tqdm
from sqlalchemy import create_engine, text

from pybaseball import statcast


# Keep columns aligned with table definition (we'll add missing as null)
KEEP_COLS = [
    "game_date", "game_pk", "at_bat_number", "pitch_number",
    "pitcher", "batter", "stand", "p_throws",
    "pitch_type", "pitch_name",
    "release_speed", "release_pos_x", "release_pos_z",
    "plate_x", "plate_z", "zone",
    "balls", "strikes",
    "events", "description", "type", "bb_type",
    "launch_speed", "launch_angle", "hit_distance_sc",
    "woba_value", "woba_denom", "estimated_woba_using_speedangle",
    "home_team", "away_team", "inning", "inning_topbot",
]

def month_ranges(season: int, months: str | None, season_end: str | None):
    """
    Returns list of (start_date, end_date) monthly chunks.
    """
    # MLB regular season starts late March/early April; statcast endpoint can handle earlier too.
    start = dt.date(season, 3, 1)
    end = dt.date(season, 11, 30)
    if season_end:
        end = min(end, pd.to_datetime(season_end).date())

    if months:
        # e.g. "3-11" or "4-10"
        a, b = months.split("-")
        m_start, m_end = int(a), int(b)
    else:
        m_start, m_end = 3, 11

    ranges = []
    cur = dt.date(season, m_start, 1)
    while cur.year == season and cur.month <= m_end:
        # month end
        if cur.month == 12:
            nxt = dt.date(season + 1, 1, 1)
        else:
            nxt = dt.date(season, cur.month + 1, 1)
        chunk_start = max(cur, start)
        chunk_end = min(nxt - dt.timedelta(days=1), end)
        if chunk_start <= chunk_end:
            ranges.append((chunk_start, chunk_end))
        cur = nxt
        if cur > end:
            break
    return ranges

def make_row_id(df: pd.DataFrame) -> pd.Series:
    """
    Deterministic row key based on stable identifiers.
    """
    keys = (
        df["game_date"].astype(str).fillna("") + "|" +
        df["game_pk"].astype("Int64").astype(str).fillna("") + "|" +
        df["at_bat_number"].astype("Int64").astype(str).fillna("") + "|" +
        df["pitch_number"].astype("Int64").astype(str).fillna("") + "|" +
        df["pitcher"].astype("Int64").astype(str).fillna("") + "|" +
        df["batter"].astype("Int64").astype(str).fillna("")
    )
    return keys.map(lambda s: hashlib.md5(s.encode("utf-8")).hexdigest())

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure expected columns exist
    for c in KEEP_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    out = df[KEEP_COLS].copy()

    # types
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.date
    for c in ["game_pk", "at_bat_number", "pitch_number", "pitcher", "batter", "zone", "balls", "strikes", "inning"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")

    for c in [
        "release_speed","release_pos_x","release_pos_z","plate_x","plate_z",
        "launch_speed","launch_angle","hit_distance_sc","woba_value","woba_denom",
        "estimated_woba_using_speedangle"
    ]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # text cleanup
    for c in ["stand","p_throws","pitch_type","pitch_name","events","description","type","bb_type","home_team","away_team","inning_topbot"]:
        if c in out.columns:
            out[c] = out[c].astype("string")

    out["row_id"] = make_row_id(out)
    return out.dropna(subset=["row_id"])

def upsert(engine, df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    # Convert to python-native for psycopg2
    records = df.to_dict(orient="records")

    sql = text("""
        INSERT INTO public.statcast_pitches (
          game_date, game_pk, at_bat_number, pitch_number,
          pitcher, batter, stand, p_throws,
          pitch_type, pitch_name,
          release_speed, release_pos_x, release_pos_z,
          plate_x, plate_z, zone,
          balls, strikes,
          events, description, type, bb_type,
          launch_speed, launch_angle, hit_distance_sc,
          woba_value, woba_denom, estimated_woba_using_speedangle,
          home_team, away_team, inning, inning_topbot,
          row_id
        )
        VALUES (
          :game_date, :game_pk, :at_bat_number, :pitch_number,
          :pitcher, :batter, :stand, :p_throws,
          :pitch_type, :pitch_name,
          :release_speed, :release_pos_x, :release_pos_z,
          :plate_x, :plate_z, :zone,
          :balls, :strikes,
          :events, :description, :type, :bb_type,
          :launch_speed, :launch_angle, :hit_distance_sc,
          :woba_value, :woba_denom, :estimated_woba_using_speedangle,
          :home_team, :away_team, :inning, :inning_topbot,
          :row_id
        )
        ON CONFLICT (row_id) DO NOTHING
    """)

    with engine.begin() as conn:
        conn.execute(sql, records)
    return len(records)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--months", default=None, help='Optional like "3-11" or "4-10"')
    ap.add_argument("--season_end", default=None, help="Optional YYYY-MM-DD for season cutoff")
    ap.add_argument("--sleep", type=float, default=1.0, help="Sleep seconds between month chunks")
    args = ap.parse_args()

    pg_dsn = os.getenv("PG_DSN")
    if not pg_dsn:
        raise RuntimeError("PG_DSN env var required")

    engine = create_engine(pg_dsn, pool_pre_ping=True)

    ranges = month_ranges(args.season, args.months, args.season_end)
    print(f"Season {args.season}: {len(ranges)} chunk(s)")

    total = 0
    for (a, b) in tqdm(ranges, desc=f"statcast {args.season}"):
        start_s = a.strftime("%Y-%m-%d")
        end_s = b.strftime("%Y-%m-%d")
        try:
            raw = statcast(start_s, end_s)  # returns pandas df
        except Exception as e:
            print(f"[WARN] statcast({start_s},{end_s}) failed: {repr(e)}")
            continue

        if raw is None or len(raw) == 0:
            continue

        df = normalize(raw)
        n = upsert(engine, df)
        total += n
        print(f"{start_s} → {end_s}: fetched={len(raw):,} normalized={len(df):,} inserted_attempt={n:,} total_attempt={total:,}")

        if args.sleep > 0:
            import time
            time.sleep(args.sleep)

    print(f"Done. Total rows attempted insert: {total:,}")
    # Quick sanity
    with engine.connect() as conn:
        rowcount = conn.execute(text("SELECT COUNT(*) FROM public.statcast_pitches")).scalar()
    print(f"statcast_pitches total rows now: {rowcount:,}")

if __name__ == "__main__":
    main()
