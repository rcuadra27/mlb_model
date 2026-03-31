#!/usr/bin/env python3
"""
market_movement.py

Tracks betting line movement between morning and closing odds pulls.
Line movement is itself a powerful feature — sharp money moves lines,
and when lines move against your model's prediction, that's a signal
the market knows something the model doesn't.

Strategy:
  - Morning pull: ~9am (when inference first runs)
  - Closing pull:  ~6pm (1-2 hours before first pitch)
  - Movement = closing_prob - morning_prob

Columns written to features_game / odds_ml:
  home_line_move     — vig-free prob change home side (closing - opening)
  away_line_move     — vig-free prob change away side (closing - opening)
  total_line_move    — O/U line movement (closing - opening, positive = line went up)
  sharp_action_home  — 1 if significant sharp line move toward home, -1 away, 0 neutral
  line_move_magnitude— absolute size of the move (proxy for sharp action)

Usage:
    # Morning pull (store opening lines):
    ODDS_API_KEY=... PG_DSN=... python market_movement.py --date 2026-03-25 --pull morning

    # Closing pull (compute movement vs morning):
    ODDS_API_KEY=... PG_DSN=... python market_movement.py --date 2026-03-25 --pull closing
"""

import os
import argparse
import requests
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

ODDS_API_SPORT = "baseball_mlb"
ODDS_API_BASE  = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/"

# Sharp line move threshold — moves larger than this are considered sharp action
SHARP_MOVE_THRESHOLD = 0.03  # 3 percentage points


def american_to_implied(price: float) -> float:
    if price < 0:
        return abs(price) / (abs(price) + 100.0)
    else:
        return 100.0 / (price + 100.0)


def vig_free_prob(home_price: float, away_price: float) -> tuple[float, float]:
    """Remove vig and return fair home/away probabilities."""
    p_home = american_to_implied(home_price)
    p_away = american_to_implied(away_price)
    total  = p_home + p_away
    return p_home / total, p_away / total


def fetch_current_odds(api_key: str, date_str: str) -> pd.DataFrame:
    """Fetch current h2h + totals odds from The Odds API."""
    params = {
        "apiKey":    api_key,
        "regions":   "us",
        "markets":   "h2h,totals",
        "oddsFormat":"american",
        "dateFormat":"iso",
    }
    try:
        resp = requests.get(ODDS_API_BASE, params=params, timeout=10)
        resp.raise_for_status()
        games = resp.json()
    except Exception as e:
        print(f"  Odds API error: {e}")
        return pd.DataFrame()

    records = []
    for g in games:

        home_team = g.get("home_team", "")
        away_team = g.get("away_team", "")

        ml_h, ml_a = [], []
        ou_lines, ou_prices = [], []

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
            p_home = p_away = np.nan

        ou_line = float(np.median(ou_lines)) if ou_lines else np.nan

        records.append({
            "api_home_team": home_team,
            "api_away_team": away_team,
            "p_home_fair":   p_home,
            "p_away_fair":   p_away,
            "ou_line":       ou_line,
            "n_books":       len(ml_h),
        })

    return pd.DataFrame(records)


def ensure_movement_columns(engine, schema: str) -> None:
    """Add line movement columns to features_game."""
    cols = {
        "home_line_move":      "DOUBLE PRECISION",
        "away_line_move":      "DOUBLE PRECISION",
        "total_line_move":     "DOUBLE PRECISION",
        "sharp_action_home":   "INTEGER",
        "line_move_magnitude": "DOUBLE PRECISION",
        "morning_p_home":      "DOUBLE PRECISION",
        "closing_p_home":      "DOUBLE PRECISION",
        "morning_ou_line":     "DOUBLE PRECISION",
        "closing_ou_line":     "DOUBLE PRECISION",
    }
    with engine.begin() as conn:
        existing = pd.read_sql(
            text("SELECT column_name FROM information_schema.columns "
                 "WHERE table_schema = :s AND table_name = 'features_game'"),
            conn, params={"s": schema}
        )["column_name"].tolist()

        for col, dtype in cols.items():
            if col not in existing:
                conn.execute(text(
                    f"ALTER TABLE {schema}.features_game ADD COLUMN {col} {dtype}"
                ))
                print(f"  Added column: {col}")


def match_game(df_games: pd.DataFrame, api_home: str, api_away: str) -> int | None:
    """Match API team name to game_id using partial name matching."""
    api_home_l = api_home.lower().strip()
    api_away_l = api_away.lower().strip()

    for _, g in df_games.iterrows():
        ht = g["home_team"].lower().strip()
        at = g["away_team"].lower().strip()

        # Exact match
        if (api_home_l == ht or any(w in ht for w in api_home_l.split())) and \
           (api_away_l == at or any(w in at for w in api_away_l.split())):
            return int(g["game_id"])

    return None


def pull_odds(engine, schema: str, date_str: str, pull_type: str,
              api_key: str) -> None:
    """
    Pull current odds and store as morning or closing snapshot.
    pull_type: 'morning' or 'closing'
    """
    print(f"  Fetching {pull_type} odds for {date_str}...")
    df_odds = fetch_current_odds(api_key, date_str)

    if df_odds.empty:
        print(f"  No odds returned for {date_str}")
        return

    # Get games for this date
    df_games = pd.read_sql(text(f"""
        SELECT g.game_id, th.team_name AS home_team, ta.team_name AS away_team
        FROM {schema}.games g
        LEFT JOIN {schema}.teams th ON th.mlb_team_id = g.home_team_id
        LEFT JOIN {schema}.teams ta ON ta.mlb_team_id = g.away_team_id
        WHERE g.game_date = :d
    """), engine, params={"d": date_str})

    rows = []
    for _, odds in df_odds.iterrows():
        game_id = match_game(df_games, odds["api_home_team"], odds["api_away_team"])
        if game_id is None:
            continue

        if pull_type == "morning":
            row = {
                "game_id":         game_id,
                "morning_p_home":  odds["p_home_fair"],
                "morning_ou_line": odds["ou_line"],
            }
        else:  # closing
            row = {
                "game_id":         game_id,
                "closing_p_home":  odds["p_home_fair"],
                "closing_ou_line": odds["ou_line"],
            }
        rows.append(row)

    if not rows:
        print(f"  No games matched for {date_str}")
        return

    df_out = pd.DataFrame(rows)
    cols   = [c for c in df_out.columns if c != "game_id"]
    set_clause = ", ".join(f"{c} = s.{c}" for c in cols)

    with engine.begin() as conn:
        df_out.to_sql("_mv_tmp", conn, schema=schema,
                      if_exists="replace", index=False, method="multi")
        conn.execute(text(f"""
            UPDATE {schema}.features_game AS t
            SET {set_clause}
            FROM {schema}._mv_tmp AS s
            WHERE t.game_id = s.game_id
        """))
        conn.execute(text(f"DROP TABLE IF EXISTS {schema}._mv_tmp"))

    print(f"  {pull_type.capitalize()} odds stored for {len(rows)} games.")

    # If closing pull, compute movement vs morning
    if pull_type == "closing":
        compute_line_movement(engine, schema, date_str)


def compute_line_movement(engine, schema: str, date_str: str) -> None:
    """
    Compute line movement features after closing pull.
    Updates features_game with movement columns.
    """
    df = pd.read_sql(text(f"""
        SELECT f.game_id, f.morning_p_home, f.closing_p_home,
               f.morning_ou_line, f.closing_ou_line
        FROM {schema}.features_game f
        JOIN {schema}.games g USING (game_id)
        WHERE g.game_date = :d
          AND f.morning_p_home IS NOT NULL
          AND f.closing_p_home IS NOT NULL
    """), engine, params={"d": date_str})

    if df.empty:
        print("  No morning+closing pairs found — skipping movement computation")
        return

    rows = []
    for _, r in df.iterrows():
        home_move  = float(r["closing_p_home"]) - float(r["morning_p_home"])
        away_move  = -home_move
        ou_move    = (float(r["closing_ou_line"]) - float(r["morning_ou_line"])
                      if pd.notna(r["closing_ou_line"]) and pd.notna(r["morning_ou_line"])
                      else 0.0)
        magnitude  = abs(home_move)

        # Sharp action signal
        if home_move > SHARP_MOVE_THRESHOLD:
            sharp = 1   # sharp money on home
        elif home_move < -SHARP_MOVE_THRESHOLD:
            sharp = -1  # sharp money on away
        else:
            sharp = 0   # no significant move

        rows.append({
            "game_id":            int(r["game_id"]),
            "home_line_move":     home_move,
            "away_line_move":     away_move,
            "total_line_move":    ou_move,
            "sharp_action_home":  sharp,
            "line_move_magnitude":magnitude,
        })

    df_out     = pd.DataFrame(rows)
    cols       = [c for c in df_out.columns if c != "game_id"]
    set_clause = ", ".join(f"{c} = s.{c}" for c in cols)

    with engine.begin() as conn:
        df_out.to_sql("_mv2_tmp", conn, schema=schema,
                      if_exists="replace", index=False, method="multi")
        conn.execute(text(f"""
            UPDATE {schema}.features_game AS t
            SET {set_clause}
            FROM {schema}._mv2_tmp AS s
            WHERE t.game_id = s.game_id
        """))
        conn.execute(text(f"DROP TABLE IF EXISTS {schema}._mv2_tmp"))

    n_sharp = (df_out["sharp_action_home"] != 0).sum()
    print(f"  Line movement computed: {len(rows)} games, {n_sharp} with sharp action detected")
    for _, r in df_out.iterrows():
        print(f"    game_id={r['game_id']} move={r['home_line_move']:+.3f} "
              f"sharp={'HOME' if r['sharp_action_home']==1 else 'AWAY' if r['sharp_action_home']==-1 else 'none'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date",   default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    ap.add_argument("--schema", default="public")
    ap.add_argument("--pull",   choices=["morning", "closing"], required=True,
                    help="Which pull to run: 'morning' stores opening odds, "
                         "'closing' stores closing odds and computes movement")
    args = ap.parse_args()

    pg_dsn  = os.getenv("PG_DSN")
    api_key = os.getenv("ODDS_API_KEY")

    if not pg_dsn:
        raise RuntimeError("PG_DSN env var required.")
    if not api_key:
        raise RuntimeError("ODDS_API_KEY env var required.")

    engine = create_engine(pg_dsn, pool_pre_ping=True)
    ensure_movement_columns(engine, args.schema)
    pull_odds(engine, args.schema, args.date, args.pull, api_key)