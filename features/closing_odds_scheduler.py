#!/usr/bin/env python3
"""
closing_odds_scheduler.py

Polls The Odds API continuously from 3pm PT until all games have had
their closing odds pulled 75 minutes before first pitch.

Each game gets its own closing pull timed to its specific first pitch.
Also re-runs inference after each closing pull so predictions update
with the latest market prices.

Usage:
    ODDS_API_KEY=... PG_DSN=... python closing_odds_scheduler.py --date 2026-04-01
    ODDS_API_KEY=... PG_DSN=... python closing_odds_scheduler.py  # defaults to today

Cron: start at 3pm PT daily
    0 15 * * * cd /path/to/mlb_model && ODDS_API_KEY=... PG_DSN=... python features/closing_odds_scheduler.py >> logs/closing_$(date +\%Y-\%m-\%d).log 2>&1
"""

import os
import sys
import time
import argparse
import subprocess
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from sqlalchemy import create_engine, text
import pytz

ODDS_API_BASE  = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/"
ODDS_API_SPORT = "baseball_mlb"
PT             = pytz.timezone("America/Los_Angeles")

CLOSING_WINDOW_MINUTES = 75
POLL_INTERVAL_SECONDS  = 300
MAX_RUNTIME_HOURS      = 8


# ---------------------------------------------------------------------------
# Odds fetching
# ---------------------------------------------------------------------------

def american_to_implied(price: float) -> float:
    if price < 0:
        return abs(price) / (abs(price) + 100.0)
    return 100.0 / (price + 100.0)


def vig_free_prob(home_price: float, away_price: float) -> tuple[float, float]:
    p_home = american_to_implied(home_price)
    p_away = american_to_implied(away_price)
    total  = p_home + p_away
    return p_home / total, p_away / total


def fetch_all_odds(api_key: str) -> pd.DataFrame:
    """Fetch current odds for all upcoming MLB games."""
    params = {
        "apiKey":    api_key,
        "regions":   "us",
        "markets":   "h2h,totals",
        "oddsFormat":"american",
        "dateFormat":"iso",
    }
    try:
        resp = requests.get(ODDS_API_BASE, params=params, timeout=15)
        resp.raise_for_status()
        games = resp.json()
    except Exception as e:
        print(f"  Odds API error: {e}")
        return pd.DataFrame()

    now_utc = pd.Timestamp.now(tz="UTC")
    records = []
    for g in games:
        home_team = g.get("home_team", "")
        away_team = g.get("away_team", "")
        commence  = g.get("commence_time", "")
        # Skip games that have already started — never use live odds
        if commence:
            commence_utc = pd.to_datetime(commence, utc=True)
            if commence_utc <= now_utc:
                continue

        ml_h, ml_a = [], []
        ou_lines   = []

        for bk in g.get("bookmakers", []):
            for mkt in bk.get("markets", []):
                if mkt["key"] == "h2h":
                    for o in mkt["outcomes"]:
                        if o["name"] == home_team:
                            ml_h.append(o["price"])
                        elif o["name"] == away_team:
                            ml_a.append(o["price"])
                elif mkt["key"] == "totals":
                    for o in mkt["outcomes"]:
                        if o["name"] == "Over":
                            ou_lines.append(o.get("point", np.nan))

        if ml_h and ml_a:
            home_med = float(np.median(ml_h))
            away_med = float(np.median(ml_a))
            p_home, p_away = vig_free_prob(home_med, away_med)
        else:
            home_med = away_med = np.nan
            p_home = p_away = np.nan

        records.append({
            "api_home_team":    home_team,
            "api_away_team":    away_team,
            "commence_time":    commence,
            "p_home_fair":      p_home,
            "p_away_fair":      p_away,
            # Raw consensus American prices (median across books)
            "home_price_raw":   int(round(home_med)) if not np.isnan(home_med) else None,
            "away_price_raw":   int(round(away_med)) if not np.isnan(away_med) else None,
            "ou_line":          float(np.median(ou_lines)) if ou_lines else np.nan,
            "n_books":          len(ml_h),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Game matching
# ---------------------------------------------------------------------------

def match_game_to_odds(home_team: str, away_team: str,
                       df_odds: pd.DataFrame) -> dict | None:
    ht = home_team.lower().strip()
    at = away_team.lower().strip()

    for _, row in df_odds.iterrows():
        kh = row["api_home_team"].lower().strip()
        ka = row["api_away_team"].lower().strip()
        if (set(ht.split()) & set(kh.split())) and \
           (set(at.split()) & set(ka.split())):
            return row.to_dict()

    return None


# ---------------------------------------------------------------------------
# Store morning odds (first poll of the day)
# ---------------------------------------------------------------------------

def store_morning_odds(engine, schema: str, game_id: int, odds: dict) -> None:
    """Store morning odds including raw American prices."""
    p_home      = odds.get("p_home_fair")
    ou          = odds.get("ou_line")
    home_price  = odds.get("home_price_raw")
    away_price  = odds.get("away_price_raw")

    if p_home is None:
        return

    with engine.begin() as conn:
        conn.execute(text(f"""
            UPDATE {schema}.features_game
            SET morning_p_home    = :ph,
                morning_ou_line   = :ou,
                morning_home_price = :hp,
                morning_away_price = :ap
            WHERE game_id = :gid
              AND morning_p_home IS NULL
        """), {
            "ph":  float(p_home),
            "ou":  float(ou) if ou and not np.isnan(ou) else None,
            "hp":  home_price,
            "ap":  away_price,
            "gid": game_id,
        })


# ---------------------------------------------------------------------------
# Store closing odds
# ---------------------------------------------------------------------------

def store_closing_odds(engine, schema: str, game_id: int, odds: dict) -> None:
    """Store closing odds including raw American prices and compute line movement."""
    p_home     = odds.get("p_home_fair")
    ou         = odds.get("ou_line")
    home_price = odds.get("home_price_raw")
    away_price = odds.get("away_price_raw")

    if p_home is None:
        return

    with engine.begin() as conn:
        conn.execute(text(f"""
            UPDATE {schema}.features_game
            SET closing_p_home    = :ph,
                closing_ou_line   = :ou,
                closing_home_price = :hp,
                closing_away_price = :ap
            WHERE game_id = :gid
              AND closing_p_home IS NULL
        """), {
            "ph":  float(p_home),
            "ou":  float(ou) if ou and not np.isnan(ou) else None,
            "hp":  home_price,
            "ap":  away_price,
            "gid": game_id,
        })

        # Compute and store line movement
        conn.execute(text(f"""
            UPDATE {schema}.features_game
            SET home_line_move      = closing_p_home - morning_p_home,
                away_line_move      = morning_p_home - closing_p_home,
                total_line_move     = COALESCE(closing_ou_line, 0)
                                      - COALESCE(morning_ou_line, 0),
                line_move_magnitude = ABS(closing_p_home - morning_p_home),
                sharp_action_home   = CASE
                    WHEN ABS(closing_p_home - morning_p_home) < 0.03 THEN 0
                    WHEN closing_p_home > morning_p_home THEN 1
                    ELSE -1
                END
            WHERE game_id = :gid
              AND morning_p_home IS NOT NULL
              AND closing_p_home IS NOT NULL
        """), {"gid": game_id})


# ---------------------------------------------------------------------------
# Re-run inference after closing pull
# ---------------------------------------------------------------------------

def rerun_inference(date_str: str, model_path: str, features_path: str) -> None:
    cmd = [
        sys.executable, "inference/inference.py",
        "--date",          date_str,
        "--team_model",    model_path,
        "--team_features", features_path,
        "--no_calibrate",
        "--fill_missing",
    ]
    print(f"  Re-running inference for {date_str}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if any(k in line for k in ["Saved", "games", "ALL GAMES", "VALUE BETS", "No value"]):
                    print(f"    {line}")
            # Export to BigQuery after successful inference
            export_cmd = [sys.executable, "export_to_bigquery.py", "--date", date_str]
            export_result = subprocess.run(export_cmd, capture_output=True, text=True, timeout=120)
            if export_result.returncode == 0:
                print(f"    ✓ Exported to BigQuery")
            else:
                print(f"  Export error: {export_result.stderr[:200]}")
        else:
            print(f"  Inference error: {result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        print("  Inference timed out after 120s")
    except Exception as e:
        print(f"  Inference subprocess error: {e}")


# ---------------------------------------------------------------------------
# Get first pitch times
# ---------------------------------------------------------------------------

def get_games_with_first_pitch(engine, schema: str, date_str: str) -> pd.DataFrame:
    try:
        df = pd.read_sql(text(f"""
            SELECT g.game_id, g.first_pitch_utc,
                   th.team_name AS home_team,
                   ta.team_name AS away_team
            FROM {schema}.games g
            LEFT JOIN {schema}.teams th ON th.mlb_team_id = g.home_team_id
            LEFT JOIN {schema}.teams ta ON ta.mlb_team_id = g.away_team_id
            WHERE g.game_date = :d
              AND g.first_pitch_utc IS NOT NULL
        """), engine, params={"d": date_str})

        if not df.empty:
            df["first_pitch_utc"] = pd.to_datetime(df["first_pitch_utc"], utc=True)
            return df
    except Exception:
        pass

    print("  Warning: first_pitch_utc not available — will pull odds for all games now")
    df = pd.read_sql(text(f"""
        SELECT g.game_id,
               NULL::timestamptz AS first_pitch_utc,
               th.team_name AS home_team,
               ta.team_name AS away_team
        FROM {schema}.games g
        LEFT JOIN {schema}.teams th ON th.mlb_team_id = g.home_team_id
        LEFT JOIN {schema}.teams ta ON ta.mlb_team_id = g.away_team_id
        WHERE g.game_date = :d
    """), engine, params={"d": date_str})

    return df


# ---------------------------------------------------------------------------
# Main scheduler loop
# ---------------------------------------------------------------------------

def run_scheduler(engine, schema: str, date_str: str, api_key: str,
                  model_path: str, features_path: str) -> None:

    print(f"\n{'='*60}")
    print(f"  Closing Odds Scheduler — {date_str}")
    print(f"  Pulling odds {CLOSING_WINDOW_MINUTES} min before first pitch")
    print(f"  Polling every {POLL_INTERVAL_SECONDS}s")
    print(f"{'='*60}\n")

    games         = get_games_with_first_pitch(engine, schema, date_str)
    pulled        = set()
    morning_done  = set()
    start_ts      = datetime.now(timezone.utc)
    max_end       = start_ts + timedelta(hours=MAX_RUNTIME_HOURS)

    if games.empty:
        print(f"  No games found for {date_str} — exiting")
        return

    print(f"  Monitoring {len(games)} games:")
    for _, g in games.iterrows():
        fp = g["first_pitch_utc"]
        if pd.notna(fp):
            fp_pt = fp.astimezone(PT).strftime("%I:%M%p PT")
            print(f"    {g['away_team']} @ {g['home_team']} — first pitch {fp_pt}")
        else:
            print(f"    {g['away_team']} @ {g['home_team']} — time unknown")
    print()

    first_poll = True

    while True:
        now_utc    = datetime.now(timezone.utc)
        now_pt_str = now_utc.astimezone(PT).strftime("%I:%M%p PT")

        if now_utc > max_end:
            print(f"  Max runtime ({MAX_RUNTIME_HOURS}h) reached — exiting")
            break

        remaining = [g for _, g in games.iterrows() if int(g["game_id"]) not in pulled]
        if not remaining:
            print(f"  [{now_pt_str}] All games processed — scheduler complete")
            break

        print(f"  [{now_pt_str}] Polling — {len(remaining)} games remaining...")
        df_odds = fetch_all_odds(api_key)

        if df_odds.empty:
            print(f"    No odds returned — will retry in {POLL_INTERVAL_SECONDS}s")
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        newly_pulled = []

        for _, game in games.iterrows():
            game_id = int(game["game_id"])
            if game_id in pulled:
                continue

            odds = match_game_to_odds(game["home_team"], game["away_team"], df_odds)
            if odds is None:
                continue

            # Store morning odds on first poll
            if first_poll and game_id not in morning_done:
                store_morning_odds(engine, schema, game_id, odds)
                morning_done.add(game_id)
                print(f"    Morning odds stored: {game['away_team']} @ {game['home_team']} "
                      f"away={odds.get('away_price_raw')} home={odds.get('home_price_raw')}")

            fp_utc = game["first_pitch_utc"]
            should_pull = False
            if pd.isna(fp_utc):
                should_pull = True
            else:
                minutes_to_game = (fp_utc - now_utc).total_seconds() / 60
                if minutes_to_game <= CLOSING_WINDOW_MINUTES:
                    should_pull = True
                    print(f"    {game['away_team']} @ {game['home_team']}: "
                          f"{minutes_to_game:.0f} min to first pitch — pulling closing odds")
                else:
                    print(f"    {game['away_team']} @ {game['home_team']}: "
                          f"waiting ({minutes_to_game:.0f} min to first pitch)")

            if not should_pull:
                continue

            store_closing_odds(engine, schema, game_id, odds)

            print(f"    ✓ {game['away_team']} @ {game['home_team']}: "
                  f"p_home={odds.get('p_home_fair', 0):.3f} "
                  f"away={odds.get('away_price_raw')} home={odds.get('home_price_raw')} "
                  f"ou={odds.get('ou_line', '—')} n_books={odds.get('n_books', 0)}")

            pulled.add(game_id)
            newly_pulled.append(game_id)

        first_poll = False

        if newly_pulled:
            rerun_inference(date_str, model_path, features_path)

        if len(pulled) == len(games):
            print(f"\n  [{now_pt_str}] All {len(games)} games processed ✓")
            break

        print(f"    Sleeping {POLL_INTERVAL_SECONDS}s...\n")
        time.sleep(POLL_INTERVAL_SECONDS)

    print("\n  Scheduler finished.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date",           default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    ap.add_argument("--schema",         default="public")
    ap.add_argument("--model",          default="artifacts/runs_model_v8.joblib")
    ap.add_argument("--features",       default="artifacts/runs_model_v8_features.txt")
    ap.add_argument("--minutes_before", type=int, default=CLOSING_WINDOW_MINUTES)
    ap.add_argument("--poll_interval",  type=int, default=POLL_INTERVAL_SECONDS)
    args = ap.parse_args()

    pg_dsn  = os.getenv("PG_DSN")
    api_key = os.getenv("ODDS_API_KEY")

    if not pg_dsn:
        raise RuntimeError("PG_DSN env var required.")
    if not api_key:
        raise RuntimeError("ODDS_API_KEY env var required.")

    engine = create_engine(pg_dsn, pool_pre_ping=True)

    run_scheduler(
        engine        = engine,
        schema        = args.schema,
        date_str      = args.date,
        api_key       = api_key,
        model_path    = args.model,
        features_path = args.features,
    )


if __name__ == "__main__":
    main()