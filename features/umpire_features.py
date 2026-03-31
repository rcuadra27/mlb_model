#!/usr/bin/env python3
"""
umpire_features.py

Fetches home plate umpire assignments from the MLB Stats API and computes
rolling umpire tendency features. Umpires vary significantly in their
strike zone size — a tight-zone umpire suppresses scoring, a wide-zone
umpire inflates it.

Columns written to features_game:
  umpire_id              — home plate umpire MLB ID
  umpire_k_rate_boost    — umpire's historical K rate vs league avg (rolling 3yr)
  umpire_bb_rate_boost   — umpire's historical BB rate vs league avg (rolling 3yr)
  umpire_runs_boost      — umpire's historical runs/game vs league avg (rolling 3yr)
  umpire_n_games         — number of games in umpire's rolling history

Positive boost = umpire tends to inflate strikeouts/walks/runs vs league avg.
Negative boost = umpire tends to suppress.

Usage:
    PG_DSN=... python umpire_features.py --date 2026-03-25
    PG_DSN=... python umpire_features.py --backfill --start 2015-04-01
"""

import os
import argparse
import time
import requests
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
MLB_GAME_URL     = "https://statsapi.mlb.com/api/v1/game/{game_id}/linescore"
MLB_BOXSCORE_URL = "https://statsapi.mlb.com/api/v1/game/{game_id}/boxscore"

UMPIRE_WINDOW_YEARS = 3  # rolling years for umpire stats


def _as_umpire_id(raw) -> int | None:
    """Coerce MLB API official id to int so pandas/to_sql use INTEGER, not TEXT."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def fetch_umpire_for_game(game_id: int) -> dict | None:
    """Fetch home plate umpire from MLB boxscore API."""
    try:
        resp = requests.get(
            MLB_BOXSCORE_URL.format(game_id=game_id),
            params={"hydrate": "officials"},
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  Boxscore API error game_id={game_id}: {e}")
        return None

    officials = data.get("officials", [])
    for official in officials:
        if official.get("officialType") == "Home Plate":
            uid = _as_umpire_id(official.get("official", {}).get("id"))
            if uid is None:
                return None
            return {
                "umpire_id":   uid,
                "umpire_name": official["official"]["fullName"],
            }
    return None


def fetch_schedule_with_umpires(date_str: str) -> list[dict]:
    """Fetch schedule for a date including official hydration."""
    try:
        resp = requests.get(MLB_SCHEDULE_URL, params={
            "sportId":    1,
            "startDate":  date_str,
            "endDate":    date_str,
            "gameTypes":  "R",
            "hydrate":    "officials",
        }, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  Schedule API error: {e}")
        return []

    results = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            game_id = int(g["gamePk"])
            umpire  = None
            for official in g.get("officials", []):
                if official.get("officialType") == "Home Plate":
                    uid = _as_umpire_id(official.get("official", {}).get("id"))
                    umpire = {
                        "game_id":     game_id,
                        "umpire_id":   uid,
                        "umpire_name": official["official"].get("fullName", ""),
                    }
                    break
            if umpire:
                results.append(umpire)
            else:
                results.append({"game_id": game_id, "umpire_id": None, "umpire_name": None})

    return results


def compute_umpire_boosts(engine, schema: str, umpire_id: int,
                          before_date: str, window_years: int = UMPIRE_WINDOW_YEARS) -> dict:
    """
    Compute umpire tendency boosts vs league average over the prior N years.
    Uses games table + statcast aggregates stored in DB.
    Returns dict with boost values.
    """
    null_result = {
        "umpire_k_rate_boost":   0.0,
        "umpire_bb_rate_boost":  0.0,
        "umpire_runs_boost":     0.0,
        "umpire_n_games":        0,
    }

    if umpire_id is None:
        return null_result

    # Get umpire game history from umpire_games table (if exists)
    # If the table doesn't exist yet, fall back to zero boosts
    try:
        df = pd.read_sql(text(f"""
            SELECT
                ug.game_id,
                g.home_runs + g.away_runs AS total_runs
            FROM {schema}.umpire_games ug
            JOIN {schema}.games g USING (game_id)
            WHERE ug.umpire_id = :uid
              AND g.game_date BETWEEN :start AND :end
              AND g.home_runs IS NOT NULL
        """), engine, params={
            "uid":   umpire_id,
            "start": pd.Timestamp(before_date) - pd.DateOffset(years=window_years),
            "end":   before_date,
        })
    except Exception:
        # umpire_games table doesn't exist yet — return neutral boosts
        return null_result

    if len(df) < 20:
        return null_result

    # Get league average for same period
    league = pd.read_sql(text(f"""
        SELECT AVG(home_runs + away_runs) AS league_avg_runs
        FROM {schema}.games
        WHERE game_date BETWEEN :start AND :end
          AND home_runs IS NOT NULL
    """), engine, params={
        "start": pd.Timestamp(before_date) - pd.DateOffset(years=window_years),
        "end":   before_date,
    })

    league_avg = float(league["league_avg_runs"].iloc[0]) if not league.empty else 9.0
    ump_avg    = float(df["total_runs"].mean())

    return {
        "umpire_k_rate_boost":  0.0,   # requires statcast K data per game per ump
        "umpire_bb_rate_boost": 0.0,   # requires statcast BB data per game per ump
        "umpire_runs_boost":    ump_avg - league_avg,
        "umpire_n_games":       len(df),
    }


def ensure_umpire_columns(engine, schema: str) -> None:
    """Add umpire columns to features_game if they don't exist."""
    cols = [
        "umpire_id",
        "umpire_k_rate_boost",
        "umpire_bb_rate_boost",
        "umpire_runs_boost",
        "umpire_n_games",
    ]
    with engine.begin() as conn:
        existing = pd.read_sql(
            text("SELECT column_name FROM information_schema.columns "
                 "WHERE table_schema = :s AND table_name = 'features_game'"),
            conn, params={"s": schema}
        )["column_name"].tolist()

        for col in cols:
            if col not in existing:
                dtype = "INTEGER" if col in ("umpire_id", "umpire_n_games") else "DOUBLE PRECISION"
                conn.execute(text(
                    f"ALTER TABLE {schema}.features_game ADD COLUMN {col} {dtype}"
                ))
                print(f"  Added column: {col}")


def ensure_umpire_games_table(engine, schema: str) -> None:
    """Create umpire_games lookup table if it doesn't exist."""
    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {schema}.umpire_games (
                game_id    BIGINT  NOT NULL,
                umpire_id  INTEGER NOT NULL,
                umpire_name TEXT,
                PRIMARY KEY (game_id)
            );
        """))


def upsert_umpire_features(engine, schema: str, date_str: str) -> None:
    """
    For every game on date_str:
    1. Fetch home plate umpire from MLB API
    2. Compute umpire tendency boosts vs league average
    3. Upsert into features_game
    """
    print(f"  Fetching umpire assignments for {date_str}...")
    assignments = fetch_schedule_with_umpires(date_str)

    if not assignments:
        print(f"  No umpire data found for {date_str}")
        return

    # Store umpire assignments in lookup table
    ensure_umpire_games_table(engine, schema)
    ump_rows = [a for a in assignments if a.get("umpire_id")]
    if ump_rows:
        df_ump = pd.DataFrame(ump_rows)
        df_ump["umpire_id"] = pd.to_numeric(df_ump["umpire_id"], errors="coerce").astype("Int64")
        with engine.begin() as conn:
            df_ump.to_sql("_ump_tmp", conn, schema=schema,
                          if_exists="replace", index=False, method="multi")
            conn.execute(text(f"""
                INSERT INTO {schema}.umpire_games (game_id, umpire_id, umpire_name)
                SELECT game_id, umpire_id, umpire_name FROM {schema}._ump_tmp
                ON CONFLICT (game_id) DO UPDATE
                SET umpire_id = EXCLUDED.umpire_id,
                    umpire_name = EXCLUDED.umpire_name
            """))
            conn.execute(text(f"DROP TABLE IF EXISTS {schema}._ump_tmp"))

    # Compute boosts and build feature rows
    rows = []
    for a in assignments:
        game_id   = a["game_id"]
        umpire_id = a.get("umpire_id")

        boosts = compute_umpire_boosts(engine, schema, umpire_id, date_str)

        row = {
            "game_id":             game_id,
            "umpire_id":           umpire_id,
            "umpire_k_rate_boost": boosts["umpire_k_rate_boost"],
            "umpire_bb_rate_boost":boosts["umpire_bb_rate_boost"],
            "umpire_runs_boost":   boosts["umpire_runs_boost"],
            "umpire_n_games":      boosts["umpire_n_games"],
        }
        rows.append(row)

        ump_name = a.get("umpire_name", "Unknown")
        runs_boost = boosts["umpire_runs_boost"]
        print(f"  game_id={game_id} ump={ump_name} "
              f"runs_boost={runs_boost:+.2f} n={boosts['umpire_n_games']}")

    if not rows:
        return

    df_out = pd.DataFrame(rows)
    df_out["umpire_id"] = pd.to_numeric(df_out["umpire_id"], errors="coerce").astype("Int64")
    df_out["umpire_n_games"] = pd.to_numeric(
        df_out["umpire_n_games"], errors="coerce"
    ).fillna(0).astype(np.int64)
    cols   = [c for c in df_out.columns if c != "game_id"]
    set_clause = ", ".join(f"{c} = s.{c}" for c in cols)

    with engine.begin() as conn:
        df_out.to_sql("_ump_feat_tmp", conn, schema=schema,
                      if_exists="replace", index=False, method="multi")
        conn.execute(text(f"""
            UPDATE {schema}.features_game AS t
            SET {set_clause}
            FROM {schema}._ump_feat_tmp AS s
            WHERE t.game_id = s.game_id
        """))
        conn.execute(text(f"DROP TABLE IF EXISTS {schema}._ump_feat_tmp"))

    print(f"  Umpire features upserted for {len(rows)} games.")


def backfill_umpire_assignments(engine, schema: str, start: str, end: str,
                                sleep: float = 0.2) -> None:
    """
    Backfill umpire_games table from historical MLB API data.
    Run once to populate historical assignments for computing boosts.
    """
    ensure_umpire_games_table(engine, schema)

    dates = pd.date_range(start, end, freq="D")
    print(f"Backfilling umpire assignments {start} → {end} ({len(dates)} days)...")

    for d in dates:
        date_str = d.strftime("%Y-%m-%d")
        assignments = fetch_schedule_with_umpires(date_str)
        ump_rows = [a for a in assignments if a.get("umpire_id")]

        if ump_rows:
            df_ump = pd.DataFrame(ump_rows)
            df_ump["umpire_id"] = pd.to_numeric(df_ump["umpire_id"], errors="coerce").astype("Int64")
            with engine.begin() as conn:
                df_ump.to_sql("_ump_bf_tmp", conn, schema=schema,
                              if_exists="replace", index=False, method="multi")
                conn.execute(text(f"""
                    INSERT INTO {schema}.umpire_games (game_id, umpire_id, umpire_name)
                    SELECT game_id, umpire_id, umpire_name FROM {schema}._ump_bf_tmp
                    ON CONFLICT (game_id) DO NOTHING
                """))
                conn.execute(text(f"DROP TABLE IF EXISTS {schema}._ump_bf_tmp"))

            print(f"  {date_str}: {len(ump_rows)} games")
        else:
            print(f"  {date_str}: no games")

        time.sleep(sleep)

    print("Backfill complete.")

def compute_all_umpire_boosts(engine, schema: str,
                               start: str, end: str,
                               window_years: int = 3,
                               min_games: int = 20) -> None:
    """
    Compute umpire_runs_boost and umpire_n_games for all games
    in features_game between start and end. Uses a rolling
    window of prior N years strictly before each game date.
    """
    print("Loading game + umpire data...")

    # Load all completed games with umpire assignments
    df = pd.read_sql(text(f"""
        SELECT
            g.game_id, g.game_date, g.season,
            g.home_runs + g.away_runs AS total_runs,
            ug.umpire_id
        FROM {schema}.games g
        JOIN {schema}.umpire_games ug USING (game_id)
        WHERE g.home_runs IS NOT NULL
        ORDER BY g.game_date
    """), engine)

    df["game_date"] = pd.to_datetime(df["game_date"])

    # Load target games to compute boosts for
    target = pd.read_sql(text(f"""
        SELECT g.game_id, g.game_date, ug.umpire_id
        FROM {schema}.games g
        JOIN {schema}.umpire_games ug USING (game_id)
        WHERE g.game_date BETWEEN :start AND :end
        ORDER BY g.game_date
    """), engine, params={"start": start, "end": end})

    target["game_date"] = pd.to_datetime(target["game_date"])

    # League average over full dataset for reference
    league_avg_overall = float(df["total_runs"].mean())

    print(f"  Computing boosts for {len(target)} games...")

    rows = []
    for _, row in target.iterrows():
        gdate     = row["game_date"]
        umpire_id = row["umpire_id"]
        game_id   = row["game_id"]

        if pd.isna(umpire_id):
            rows.append({
                "game_id":           int(game_id),
                "umpire_id":         None,
                "umpire_runs_boost": 0.0,
                "umpire_n_games":    0,
            })
            continue

        # Prior N years strictly before this game
        cutoff = gdate - pd.DateOffset(years=window_years)
        mask = (
            (df["umpire_id"] == umpire_id) &
            (df["game_date"] >= cutoff) &
            (df["game_date"] < gdate)
        )
        ump_games = df[mask]
        n = len(ump_games)

        if n < min_games:
            rows.append({
                "game_id":           int(game_id),
                "umpire_id":         int(umpire_id),
                "umpire_runs_boost": 0.0,
                "umpire_n_games":    n,
            })
            continue

        # League average over same window
        league_mask = (
            (df["game_date"] >= cutoff) &
            (df["game_date"] < gdate)
        )
        league_avg = float(df[league_mask]["total_runs"].mean())
        ump_avg    = float(ump_games["total_runs"].mean())

        rows.append({
            "game_id":           int(game_id),
            "umpire_id":         int(umpire_id),
            "umpire_runs_boost": ump_avg - league_avg,
            "umpire_n_games":    n,
        })

    print(f"  Upserting {len(rows)} rows...")
    df_out     = pd.DataFrame(rows)
    cols       = [c for c in df_out.columns if c != "game_id"]
    set_clause = ", ".join(f"{c} = s.{c}" for c in cols)

    with engine.begin() as conn:
        df_out.to_sql("_ump_boost_tmp", conn, schema=schema,
                      if_exists="replace", index=False, method="multi")
        conn.execute(text(f"""
            UPDATE {schema}.features_game AS t
            SET {set_clause}
            FROM {schema}._ump_boost_tmp AS s
            WHERE t.game_id = s.game_id
        """))
        conn.execute(text(f"DROP TABLE IF EXISTS {schema}._ump_boost_tmp"))

    print(f"  Umpire boosts computed for {len(rows)} games.")

    # Quick sanity check
    sample = df_out[df_out["umpire_n_games"] >= 20]["umpire_runs_boost"]
    print(f"\n  Boost distribution (n>={min_games} games):")
    print(f"    Mean:  {sample.mean():+.3f}")
    print(f"    Std:   {sample.std():.3f}")
    print(f"    Min:   {sample.min():+.3f}")
    print(f"    Max:   {sample.max():+.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date",     default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    ap.add_argument("--schema",   default="public")
    ap.add_argument("--backfill", action="store_true",
                    help="Backfill umpire_games table from historical data")
    ap.add_argument("--start",    default="2015-04-01")
    ap.add_argument("--end",      default=None)
    ap.add_argument("--compute_all", action="store_true",
                help="Compute umpire boosts for all historical games")
    args = ap.parse_args()

    pg_dsn = os.getenv("PG_DSN")
    if not pg_dsn:
        raise RuntimeError("PG_DSN env var required.")

    engine = create_engine(pg_dsn, pool_pre_ping=True)
    ensure_umpire_columns(engine, args.schema)

    if args.backfill:
        end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
        backfill_umpire_assignments(engine, args.schema, args.start, end)
    elif args.compute_all:
        end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
        compute_all_umpire_boosts(engine, args.schema, args.start, end)
    else:
        upsert_umpire_features(engine, args.schema, args.date)