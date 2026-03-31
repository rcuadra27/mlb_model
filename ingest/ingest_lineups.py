#!/usr/bin/env python3
"""
ingest_lineups.py

Polls the MLB Stats API continuously from 6am PT, checking for confirmed
lineups per game. As soon as BOTH home and away lineups are confirmed for
a game, immediately triggers:
    1. features/build_features1.py --date <date>
    2. inference/inference.py --date <date>
    3. export_to_bigquery.py --date <date>
    → game appears on dashboard

Each game is triggered independently as its lineup confirms, so early
East Coast games (1pm ET) appear on the dashboard hours before late
West Coast games (10pm ET).

Usage:
    PG_DSN=... ODDS_API_KEY=... python ingest/ingest_lineups.py --date 2026-04-01
    PG_DSN=... ODDS_API_KEY=... python ingest/ingest_lineups.py  # defaults to today

Cron: start at 6am PT daily (catches early ET games)
    0 6 * * * cd /path/to/mlb_model && PG_DSN=... python ingest/ingest_lineups.py >> logs/lineups_$(date +\%Y-\%m-\%d).log 2>&1
"""

import os
import sys
import time
import argparse
import subprocess
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from sqlalchemy import create_engine, text
import pytz

PT                   = pytz.timezone("America/Los_Angeles")
POLL_INTERVAL        = 300   # 5 minutes
MAX_RUNTIME_HOURS    = 16    # safety cutoff — covers all games including late PT
MLB_BOXSCORE_URL     = "https://statsapi.mlb.com/api/v1/game/{game_id}/boxscore"
MLB_SCHEDULE_URL     = "https://statsapi.mlb.com/api/v1/schedule"

MODEL_PATH    = "artifacts/runs_model_v8.txt"
FEATURES_PATH = "artifacts/runs_model_v8_features.txt"


# ---------------------------------------------------------------------------
# MLB API helpers
# ---------------------------------------------------------------------------

def fetch_game_ids_for_date(date_str: str) -> list:
    try:
        resp = requests.get(MLB_SCHEDULE_URL, params={
            "sportId":   1,
            "startDate": date_str,
            "endDate":   date_str,
            "gameTypes": "R",
        }, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  Schedule API error: {e}")
        return []

    game_ids = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            game_ids.append(int(g["gamePk"]))
    return game_ids


def fetch_lineup_for_game(game_id: int) -> list:
    """
    Returns list of player rows if BOTH lineups are confirmed.
    Returns empty list if either lineup is not yet posted.
    """
    try:
        resp = requests.get(
            MLB_BOXSCORE_URL.format(game_id=game_id),
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  Boxscore API error game_id={game_id}: {e}")
        return []

    rows  = []
    teams = data.get("teams", {})
    sides_confirmed = 0

    for side, is_home in [("home", 1), ("away", 0)]:
        team_data = teams.get(side, {})
        team_id   = team_data.get("team", {}).get("id")
        batters   = team_data.get("batters", [])
        players   = team_data.get("players", {})

        if not batters or not team_id:
            continue

        side_rows = []
        for order, player_id in enumerate(batters[:9], start=1):
            player_key  = f"ID{player_id}"
            player_info = players.get(player_key, {})
            person      = player_info.get("person", {})
            name        = person.get("fullName", "")
            position    = player_info.get("position", {}).get("abbreviation", "")
            if position == "P":
                continue
            side_rows.append({
                "game_id":       game_id,
                "team_id":       int(team_id),
                "is_home":       is_home,
                "batting_order": order,
                "player_id":     int(player_id),
                "player_name":   name,
            })

        if len(side_rows) >= 8:  # at least 8 batters = confirmed lineup
            sides_confirmed += 1
            rows.extend(side_rows)

    # Only return rows if BOTH sides are confirmed
    return rows if sides_confirmed == 2 else []


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def ensure_lineup_columns(engine, schema: str) -> None:
    with engine.begin() as conn:
        existing = pd.read_sql(
            text("SELECT column_name FROM information_schema.columns "
                 "WHERE table_schema = :s AND table_name = 'game_lineups'"),
            conn, params={"s": schema}
        )["column_name"].tolist()

        for col, typedef in [("player_name", "TEXT"), ("game_date", "DATE")]:
            if col not in existing:
                try:
                    conn.execute(text(
                        f"ALTER TABLE {schema}.game_lineups ADD COLUMN {col} {typedef}"
                    ))
                    print(f"  Added column: {col}")
                except Exception:
                    pass


def upsert_lineups(engine, schema: str, rows: list, date_str: str) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df["game_date"] = date_str
    with engine.begin() as conn:
        df.to_sql("_lineup_tmp", conn, schema=schema,
                  if_exists="replace", index=False, method="multi")
        conn.execute(text(f"""
            INSERT INTO {schema}.game_lineups
                (game_id, game_date, team_id, is_home, batting_order,
                 player_id, player_name)
            SELECT game_id, game_date::date, team_id, (is_home != 0),
                   batting_order, player_id, player_name
            FROM {schema}._lineup_tmp
            ON CONFLICT (game_id, team_id, batting_order)
            DO UPDATE SET
                player_id   = EXCLUDED.player_id,
                player_name = EXCLUDED.player_name,
                game_date   = EXCLUDED.game_date
        """))
        conn.execute(text(f"DROP TABLE IF EXISTS {schema}._lineup_tmp"))


# ---------------------------------------------------------------------------
# Inference chain — triggered per game as lineup confirms
# ---------------------------------------------------------------------------

def run_inference_chain(date_str: str, schema: str) -> None:
    """
    Run build_features → inference → export for the given date.
    Called immediately when a new game's lineup confirms.
    Safe to run multiple times — all scripts are idempotent.
    """
    print(f"  → Running inference chain for {date_str}...")

    steps = [
        {
            "name": "build_features",
            "cmd": [
                sys.executable, "features/build_features1.py",
                "--date", date_str,
                "--schema", schema,
            ]
        },
        {
            "name": "inference",
            "cmd": [
                sys.executable, "inference/inference.py",
                "--date", date_str,
                "--team_model",    MODEL_PATH,
                "--team_features", FEATURES_PATH,
                "--no_calibrate",
                "--fill_missing",
            ]
        },
        {
            "name": "export_to_bigquery",
            "cmd": [
                sys.executable, "export_to_bigquery.py",
                "--date", date_str,
                "--schema", schema,
            ]
        },
    ]

    env = os.environ.copy()

    for step in steps:
        print(f"    Running {step['name']}...")
        try:
            result = subprocess.run(
                step["cmd"],
                capture_output=True,
                text=True,
                timeout=300,
                env=env,
            )
            if result.returncode == 0:
                # Print key output lines only
                for line in result.stdout.split("\n"):
                    if any(k in line for k in [
                        "Saved", "games", "Exported", "predictions",
                        "Value bets", "No value", "Found"
                    ]):
                        print(f"      {line.strip()}")
                print(f"    ✓ {step['name']} complete")
            else:
                print(f"    ✗ {step['name']} failed:")
                print(f"      {result.stderr[:300]}")
        except subprocess.TimeoutExpired:
            print(f"    ✗ {step['name']} timed out after 300s")
        except Exception as e:
            print(f"    ✗ {step['name']} error: {e}")


# ---------------------------------------------------------------------------
# Main scheduler loop
# ---------------------------------------------------------------------------

def run_scheduler(engine, schema: str, date_str: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Lineup Scheduler — {date_str}")
    print(f"  Polling every {POLL_INTERVAL}s, max {MAX_RUNTIME_HOURS}h")
    print(f"{'='*60}\n")

    game_ids  = fetch_game_ids_for_date(date_str)
    if not game_ids:
        print(f"  No games found for {date_str} — exiting")
        return

    print(f"  Monitoring {len(game_ids)} games: {game_ids}\n")

    confirmed     = set()   # game_ids with confirmed lineups
    chain_ran     = False   # has inference chain run at least once?
    start_ts      = datetime.now(timezone.utc)
    max_end       = start_ts + timedelta(hours=MAX_RUNTIME_HOURS)

    while True:
        now_utc    = datetime.now(timezone.utc)
        now_pt_str = now_utc.astimezone(PT).strftime("%I:%M%p PT")

        if now_utc > max_end:
            print(f"  [{now_pt_str}] Max runtime reached — exiting")
            break

        remaining = [g for g in game_ids if g not in confirmed]
        if not remaining:
            print(f"  [{now_pt_str}] All {len(game_ids)} lineups confirmed — done")
            break

        print(f"  [{now_pt_str}] Checking {len(remaining)} remaining games...")

        newly_confirmed = []
        for game_id in remaining:
            rows = fetch_lineup_for_game(game_id)
            if rows:
                upsert_lineups(engine, schema, rows, date_str)
                confirmed.add(game_id)
                newly_confirmed.append(game_id)
                home = [r["player_name"] for r in rows if r["is_home"]]
                away = [r["player_name"] for r in rows if not r["is_home"]]
                print(f"    ✓ game_id={game_id}: lineup confirmed "
                      f"({len(away)} away, {len(home)} home batters)")
            else:
                print(f"    — game_id={game_id}: not yet posted")

        # Run inference chain whenever new lineups come in
        if newly_confirmed:
            print(f"\n  {len(newly_confirmed)} new lineup(s) confirmed → triggering inference chain")
            run_inference_chain(date_str, schema)
            chain_ran = True
            print()

        if len(confirmed) == len(game_ids):
            print(f"  [{now_pt_str}] All lineups confirmed ✓")
            break

        print(f"    Sleeping {POLL_INTERVAL}s...\n")
        time.sleep(POLL_INTERVAL)

    if not chain_ran:
        print(f"\n  No lineups confirmed today — inference chain never ran")

    print(f"\n  Lineup scheduler finished. {len(confirmed)}/{len(game_ids)} games confirmed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date",   default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    ap.add_argument("--schema", default="public")
    args = ap.parse_args()

    pg_dsn = os.getenv("PG_DSN")
    if not pg_dsn:
        raise RuntimeError("PG_DSN env var required.")

    engine = create_engine(pg_dsn, pool_pre_ping=True)
    ensure_lineup_columns(engine, args.schema)
    run_scheduler(engine, args.schema, args.date)