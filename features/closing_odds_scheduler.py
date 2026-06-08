#!/usr/bin/env python3
"""
closing_odds_scheduler.py

Pulls closing odds once per game at T−75 minutes before first pitch.
Morning odds come from market_movement.py (morning_inference); this script
only hits The Odds API when at least one game enters its closing window.

Each GET /odds costs 2 credits (h2h + totals × us region). A 15-game slate
with staggered first pitches typically needs ~5–15 calls, not 20–30+ polls.

Usage:
    ODDS_API_KEY=... PG_DSN=... python closing_odds_scheduler.py --date 2026-04-01
    ODDS_API_KEY=... PG_DSN=... python closing_odds_scheduler.py  # defaults to today

Cron: start at 3pm PT daily (before earliest T−75 windows)
    0 15 * * * cd /path/to/mlb_model && ODDS_API_KEY=... PG_DSN=... python features/closing_odds_scheduler.py >> logs/closing_$(date +\%Y-%m-%d).log 2>&1
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

from features.odds_team_match import match_game_to_odds_row

ODDS_API_BASE  = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/"
ODDS_API_SPORT = "baseball_mlb"
PT             = pytz.timezone("America/Los_Angeles")

CLOSING_WINDOW_MINUTES = 75
POLL_INTERVAL_SECONDS  = 900   # fallback retry / unknown first-pitch backoff
MAX_RUNTIME_HOURS      = 8
MAX_API_CALLS_PER_RUN  = 40    # hard cap (~80 credits); prevents runaway loops
MAX_ZERO_PULL_STREAK   = 3     # consecutive fetches with no stored games → exit
MATCH_FAIL_BACKOFF_SEC = 600   # sleep when API returns but no games match


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


def _log_odds_api_usage(resp: requests.Response) -> int:
    """Log and return quota cost from The Odds API response headers."""
    try:
        cost = int(resp.headers.get("x-requests-last", 0))
    except (TypeError, ValueError):
        cost = 0
    remaining = resp.headers.get("x-requests-remaining")
    msg = f"  Odds API cost: {cost or '?'} credit(s)"
    if remaining is not None:
        msg += f", {remaining} remaining this month"
    print(msg)
    return cost or 2


def fetch_all_odds(api_key: str) -> tuple[pd.DataFrame, int]:
    """Fetch current odds for all upcoming MLB games (2 credits: h2h + totals)."""
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
        cost = _log_odds_api_usage(resp)
        games = resp.json()
    except Exception as e:
        print(f"  Odds API error: {e}")
        return pd.DataFrame(), 0

    now_utc = pd.Timestamp.now(tz="UTC")
    records = []
    for g in games:
        home_team = g.get("home_team", "")
        away_team = g.get("away_team", "")
        commence  = g.get("commence_time", "")
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
            "home_price_raw":   int(round(home_med)) if not np.isnan(home_med) else None,
            "away_price_raw":   int(round(away_med)) if not np.isnan(away_med) else None,
            "ou_line":          float(np.median(ou_lines)) if ou_lines else np.nan,
            "n_books":          len(ml_h),
        })

    return pd.DataFrame(records), cost


def closing_deadline(first_pitch_utc, closing_window_minutes: int):
    if pd.isna(first_pitch_utc):
        return None
    return first_pitch_utc - timedelta(minutes=closing_window_minutes)


def games_due_for_closing(
    games: pd.DataFrame,
    pulled: set,
    now_utc: datetime,
    closing_window_minutes: int,
) -> list[pd.Series]:
    due = []
    for _, game in games.iterrows():
        game_id = int(game["game_id"])
        if game_id in pulled:
            continue
        fp_utc = game["first_pitch_utc"]
        if pd.isna(fp_utc):
            due.append(game)
            continue
        minutes_to_game = (fp_utc - now_utc).total_seconds() / 60
        if minutes_to_game <= closing_window_minutes:
            due.append(game)
    return due


def seconds_until_next_closing_deadline(
    games: pd.DataFrame,
    pulled: set,
    now_utc: datetime,
    closing_window_minutes: int,
) -> float | None:
    """Seconds until the next unpulled game enters its closing window."""
    upcoming = []
    for _, game in games.iterrows():
        game_id = int(game["game_id"])
        if game_id in pulled:
            continue
        deadline = closing_deadline(game["first_pitch_utc"], closing_window_minutes)
        if deadline is None:
            return 0.0
        if deadline > now_utc:
            upcoming.append(deadline)
    if not upcoming:
        return None
    return max(0.0, (min(upcoming) - now_utc).total_seconds())


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
# Re-run inference after all closing pulls
# ---------------------------------------------------------------------------

def rerun_inference(date_str: str, ml_model_path: str, total_model_path: str) -> None:
    cmds = [
        [
            sys.executable, "inference/inference_v10.py",
            "--date", date_str,
            "--model", ml_model_path,
            "--fill_missing",
        ],
        [
            sys.executable, "inference/inference_v10_total.py",
            "--date", date_str,
            "--model", total_model_path,
        ],
        [
            sys.executable, "export_to_bigquery.py",
            "--date", date_str,
        ],
    ]
    print(f"  Re-running v10 inference + export for {date_str}...")
    for cmd in cmds:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if result.returncode != 0:
                print(f"  {' '.join(cmd[:2])} error: {result.stderr[:200]}")
                return
            for line in result.stdout.split("\n"):
                if any(k in line for k in ["Saved", "games", "Updated", "Exported", "Wrote"]):
                    print(f"    {line}")
        except subprocess.TimeoutExpired:
            print(f"  {' '.join(cmd[:2])} timed out")
            return
    print("    ✓ Closing inference + export complete")


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

def run_scheduler(
    engine,
    schema: str,
    date_str: str,
    api_key: str,
    ml_model_path: str,
    total_model_path: str,
    poll_interval: int = POLL_INTERVAL_SECONDS,
    closing_window_minutes: int = CLOSING_WINDOW_MINUTES,
) -> None:

    print(f"\n{'='*60}")
    print(f"  Closing Odds Scheduler — {date_str}")
    print(f"  One API call per closing window (T−{closing_window_minutes} min)")
    print(f"  Morning odds: market_movement.py (not duplicated here)")
    print(f"{'='*60}\n")

    games    = get_games_with_first_pitch(engine, schema, date_str)
    pulled   = set()
    skipped  = set()  # games abandoned after repeated match failures
    match_fails: dict[int, int] = {}
    api_calls = 0
    credits   = 0
    zero_pull_streak = 0
    start_ts  = datetime.now(timezone.utc)
    max_end   = start_ts + timedelta(hours=MAX_RUNTIME_HOURS)

    if games.empty:
        print(f"  No games found for {date_str} — exiting")
        return

    print(f"  Monitoring {len(games)} games:")
    for _, g in games.iterrows():
        fp = g["first_pitch_utc"]
        if pd.notna(fp):
            fp_pt = fp.astimezone(PT).strftime("%I:%M%p PT")
            close_pt = (fp - timedelta(minutes=closing_window_minutes)).astimezone(PT).strftime("%I:%M%p PT")
            print(f"    {g['away_team']} @ {g['home_team']} — pitch {fp_pt}, close pull ~{close_pt}")
        else:
            print(f"    {g['away_team']} @ {g['home_team']} — time unknown (pull on first wake)")
    print()

    while True:
        now_utc    = datetime.now(timezone.utc)
        now_pt_str = now_utc.astimezone(PT).strftime("%I:%M%p PT")

        if now_utc > max_end:
            print(f"  Max runtime ({MAX_RUNTIME_HOURS}h) reached — exiting")
            break

        if len(pulled) + len(skipped) >= len(games):
            print(f"  [{now_pt_str}] All games processed or skipped — scheduler complete")
            break

        if api_calls >= MAX_API_CALLS_PER_RUN:
            print(f"  PIPELINE_ALERT severity=critical job=closing_odds "
                  f"message=Hit MAX_API_CALLS_PER_RUN ({MAX_API_CALLS_PER_RUN}) — exiting to protect quota")
            break

        due = games_due_for_closing(games, pulled | skipped, now_utc, closing_window_minutes)
        if not due:
            wait = seconds_until_next_closing_deadline(
                games, pulled | skipped, now_utc, closing_window_minutes,
            )
            if wait is None:
                break
            nap = int(min(max(wait, 30), poll_interval * 4, (max_end - now_utc).total_seconds()))
            remaining = len(games) - len(pulled) - len(skipped)
            print(f"  [{now_pt_str}] Waiting for next closing window — "
                  f"{remaining} games left, sleep {nap}s")
            time.sleep(nap)
            continue

        print(f"  [{now_pt_str}] {len(due)} game(s) due — fetching odds...")
        df_odds, cost = fetch_all_odds(api_key)
        api_calls += 1
        credits += cost
        stored_this_round = 0

        if df_odds.empty:
            zero_pull_streak += 1
            err_sleep = min(max(poll_interval, 600), 3600)
            print(f"    No odds returned — will retry in {err_sleep}s "
                  f"(zero-pull streak {zero_pull_streak}/{MAX_ZERO_PULL_STREAK})")
            if zero_pull_streak >= MAX_ZERO_PULL_STREAK:
                print("  PIPELINE_ALERT severity=critical job=closing_odds "
                      "message=Odds API returned empty repeatedly — exiting")
                break
            time.sleep(err_sleep)
            continue

        for game in due:
            game_id = int(game["game_id"])
            odds = match_game_to_odds_row(game["home_team"], game["away_team"], df_odds)
            if odds is None:
                match_fails[game_id] = match_fails.get(game_id, 0) + 1
                nfail = match_fails[game_id]
                print(f"    ✗ No odds match: {game['away_team']} @ {game['home_team']} "
                      f"(attempt {nfail}/3)")
                if nfail >= 3:
                    skipped.add(game_id)
                continue

            store_closing_odds(engine, schema, game_id, odds)
            pulled.add(game_id)
            stored_this_round += 1
            print(f"    ✓ {game['away_team']} @ {game['home_team']}: "
                  f"p_home={odds.get('p_home_fair', 0):.3f} "
                  f"away={odds.get('away_price_raw')} home={odds.get('home_price_raw')} "
                  f"ou={odds.get('ou_line', '—')} n_books={odds.get('n_books', 0)}")

        if stored_this_round == 0:
            zero_pull_streak += 1
            print(f"    No games stored this fetch — backoff {MATCH_FAIL_BACKOFF_SEC}s "
                  f"(zero-pull streak {zero_pull_streak}/{MAX_ZERO_PULL_STREAK})")
            if zero_pull_streak >= MAX_ZERO_PULL_STREAK:
                print("  PIPELINE_ALERT severity=critical job=closing_odds "
                      "message=Team match failed for all due games — exiting to protect quota")
                break
            time.sleep(MATCH_FAIL_BACKOFF_SEC)
        else:
            zero_pull_streak = 0

    if pulled:
        rerun_inference(date_str, ml_model_path, total_model_path)

    print(f"\n  Scheduler finished — {api_calls} API call(s), ~{credits} credit(s) used for closing")
    if len(pulled) < len(games):
        print(f"  Warning: only {len(pulled)}/{len(games)} games got closing odds")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date",           default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    ap.add_argument("--schema",         default="public")
    ap.add_argument("--model",          default="artifacts/baseline_v10_production.joblib",
                    help="v10 moneyline model for post-closing inference")
    ap.add_argument("--total_model",    default="artifacts/totals_v10_umpire_runs_boost_sp_xwoba_total.joblib",
                    help="v10 totals model for post-closing inference")
    ap.add_argument("--minutes_before", type=int, default=CLOSING_WINDOW_MINUTES)
    ap.add_argument("--poll_interval",  type=int, default=POLL_INTERVAL_SECONDS,
                    help="Retry backoff when API fails or first_pitch unknown")
    args = ap.parse_args()

    pg_dsn  = os.getenv("PG_DSN")
    api_key = os.getenv("ODDS_API_KEY")

    if not pg_dsn:
        raise RuntimeError("PG_DSN env var required.")
    if not api_key:
        raise RuntimeError("ODDS_API_KEY env var required.")

    engine = create_engine(pg_dsn, pool_pre_ping=True)

    run_scheduler(
        engine                 = engine,
        schema                 = args.schema,
        date_str               = args.date,
        api_key                = api_key,
        ml_model_path          = args.model,
        total_model_path       = args.total_model,
        poll_interval          = args.poll_interval,
        closing_window_minutes = args.minutes_before,
    )


if __name__ == "__main__":
    main()
