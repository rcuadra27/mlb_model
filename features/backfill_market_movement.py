#!/usr/bin/env python3
"""
Backfill morning/closing ML snapshots and line-movement features into features_game.

Uses The Odds API historical endpoint (2 snapshots per calendar date):
  - morning: 09:00 PT  (default 16:00 UTC during PDT)
  - closing: 23:00 UTC (proxy for pre-first-pitch market, bulk backfill)

Historical coverage begins ~2020; earlier seasons remain NULL / zero-filled.

Usage:
  PG_DSN=... ODDS_API_KEY=... python features/backfill_market_movement.py \\
      --start 2020-03-01 --end 2025-12-31

  # Single date smoke test:
  PG_DSN=... ODDS_API_KEY=... python features/backfill_market_movement.py --date 2025-06-01
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from features.market_movement import SHARP_MOVE_THRESHOLD, ensure_movement_columns
from odds.the_odds_api import TheOddsAPI


def american_to_implied(price: float) -> float:
    if price < 0:
        return abs(price) / (abs(price) + 100.0)
    return 100.0 / (price + 100.0)


def vig_free_prob(home_price: float, away_price: float) -> tuple[float, float]:
    p_home = american_to_implied(home_price)
    p_away = american_to_implied(away_price)
    total = p_home + p_away
    return p_home / total, p_away / total


def snapshot_iso(game_date: date, hour_utc: int) -> str:
    return f"{game_date.isoformat()}T{hour_utc:02d}:00:00Z"


def fetch_snapshot_probs(client: TheOddsAPI, snapshot: str) -> pd.DataFrame:
    """Return median vig-free home prob per API event at snapshot."""
    try:
        events = client.get_moneylines_historical(snapshot_iso=snapshot)
    except Exception as exc:
        msg = str(exc)
        print(f"    API error @ {snapshot}: {exc}")
        if "401" in msg:
            raise RuntimeError("ODDS_API_UNAUTHORIZED") from exc
        return pd.DataFrame()

    rows = []
    for ev in events:
        home_team = ev.get("home_team", "")
        away_team = ev.get("away_team", "")
        ml_h, ml_a = [], []
        for bk in ev.get("bookmakers", []) or []:
            for mkt in bk.get("markets", []) or []:
                if mkt.get("key") != "h2h":
                    continue
                for o in mkt.get("outcomes", []) or []:
                    if o.get("name") == home_team:
                        ml_h.append(o.get("price"))
                    elif o.get("name") == away_team:
                        ml_a.append(o.get("price"))
        if not ml_h or not ml_a:
            continue
        home_med = float(np.median(ml_h))
        away_med = float(np.median(ml_a))
        p_home, _ = vig_free_prob(home_med, away_med)
        rows.append({
            "api_home_team": home_team,
            "api_away_team": away_team,
            "p_home_fair": p_home,
            "n_books": len(ml_h),
        })
    return pd.DataFrame(rows)


def load_games_for_date(engine, schema: str, game_date: str) -> pd.DataFrame:
    return pd.read_sql(
        text(f"""
            SELECT game_id, home_team_name AS home_team, away_team_name AS away_team
            FROM {schema}.games
            WHERE game_date = :d
        """),
        engine,
        params={"d": game_date},
    )


def match_snapshot_to_games(df_games: pd.DataFrame, df_odds: pd.DataFrame) -> pd.DataFrame:
    if df_games.empty or df_odds.empty:
        return pd.DataFrame()

    odds = df_odds.copy()
    odds["home_key"] = odds["api_home_team"].str.lower().str.strip()
    odds["away_key"] = odds["api_away_team"].str.lower().str.strip()
    games = df_games.copy()
    games["home_key"] = games["home_team"].str.lower().str.strip()
    games["away_key"] = games["away_team"].str.lower().str.strip()

    merged = games.merge(
        odds[["home_key", "away_key", "p_home_fair", "n_books"]],
        on=["home_key", "away_key"],
        how="inner",
    )
    return merged[["game_id", "p_home_fair", "n_books"]]


def upsert_snapshot(
    engine,
    schema: str,
    game_date: str,
    col_p_home: str,
    df_matched: pd.DataFrame,
) -> int:
    if df_matched.empty:
        return 0

    df_out = df_matched.rename(columns={"p_home_fair": col_p_home})[["game_id", col_p_home]]
    with engine.begin() as conn:
        df_out.to_sql("_bm_tmp", conn, schema=schema, if_exists="replace", index=False, method="multi")
        conn.execute(text(f"""
            UPDATE {schema}.features_game AS t
            SET {col_p_home} = s.{col_p_home}
            FROM {schema}._bm_tmp AS s
            WHERE t.game_id = s.game_id
        """))
        conn.execute(text(f"DROP TABLE IF EXISTS {schema}._bm_tmp"))
    return len(df_out)


def backfill_date(
    engine,
    schema: str,
    client: TheOddsAPI,
    game_date: date,
    morning_hour: int,
    closing_hours: list[int],
) -> dict:
    d_str = game_date.isoformat()
    df_games = load_games_for_date(engine, schema, d_str)
    if df_games.empty:
        return {"date": d_str, "games": 0, "morning": 0, "closing": 0, "movement": 0}

    morning_snap = snapshot_iso(game_date, morning_hour)
    df_morning = fetch_snapshot_probs(client, morning_snap)
    m_morning = match_snapshot_to_games(df_games, df_morning)

    # Latest closing snapshot per game (try hours in order, later overwrites earlier)
    closing_frames = []
    for hour in closing_hours:
        df_closing = fetch_snapshot_probs(client, snapshot_iso(game_date, hour))
        m = match_snapshot_to_games(df_games, df_closing)
        if not m.empty:
            closing_frames.append(m)

    if closing_frames:
        m_closing = pd.concat(closing_frames, ignore_index=True)
        m_closing = m_closing.drop_duplicates(subset=["game_id"], keep="last")
    else:
        m_closing = pd.DataFrame(columns=["game_id", "p_home_fair", "n_books"])

    n_morning = upsert_snapshot(engine, schema, d_str, "morning_p_home", m_morning)
    n_closing = upsert_snapshot(engine, schema, d_str, "closing_p_home", m_closing)
    n_move = compute_movement_for_date(engine, schema, d_str)

    return {
        "date": d_str,
        "games": len(df_games),
        "morning": n_morning,
        "closing": n_closing,
        "movement": n_move,
    }


def compute_movement_for_date(engine, schema: str, game_date: str) -> int:
    with engine.begin() as conn:
        result = conn.execute(
            text(f"""
                UPDATE {schema}.features_game AS f
                SET
                    home_line_move = f.closing_p_home - f.morning_p_home,
                    away_line_move = f.morning_p_home - f.closing_p_home,
                    total_line_move = COALESCE(f.closing_ou_line, 0) - COALESCE(f.morning_ou_line, 0),
                    line_move_magnitude = ABS(f.closing_p_home - f.morning_p_home),
                    sharp_action_home = CASE
                        WHEN ABS(f.closing_p_home - f.morning_p_home) < :thr THEN 0
                        WHEN f.closing_p_home > f.morning_p_home THEN 1
                        ELSE -1
                    END
                FROM {schema}.games g
                WHERE f.game_id = g.game_id
                  AND g.game_date = :d
                  AND f.morning_p_home IS NOT NULL
                  AND f.closing_p_home IS NOT NULL
            """),
            {"d": game_date, "thr": SHARP_MOVE_THRESHOLD},
        )
        return int(result.rowcount or 0)


def iter_dates(start: date, end: date, engine, schema: str, skip_existing: bool = False) -> list[date]:
    skip_clause = ""
    if skip_existing:
        skip_clause = f"""
            AND (
                SELECT COUNT(*)
                FROM {schema}.games g2
                WHERE g2.game_date = g.game_date
            ) <> (
                SELECT COUNT(*)
                FROM {schema}.games g2
                JOIN {schema}.features_game f2 ON f2.game_id = g2.game_id
                WHERE g2.game_date = g.game_date
                  AND f2.morning_p_home IS NOT NULL
                  AND f2.closing_p_home IS NOT NULL
            )
        """
    df = pd.read_sql(
        text(f"""
            SELECT DISTINCT g.game_date::date AS d
            FROM {schema}.games g
            WHERE g.game_date BETWEEN :start AND :end
            {skip_clause}
            ORDER BY g.game_date
        """),
        engine,
        params={"start": start.isoformat(), "end": end.isoformat()},
    )
    return [pd.Timestamp(x).date() for x in df["d"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", default="public")
    ap.add_argument("--start", default="2020-03-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--date", help="Single YYYY-MM-DD (overrides start/end)")
    ap.add_argument("--morning-hour-utc", type=int, default=16)
    ap.add_argument(
        "--closing-hours-utc",
        default="20,22",
        help="Comma-separated UTC hours; later hours overwrite earlier per game",
    )
    ap.add_argument("--sleep", type=float, default=0.35, help="Seconds between dates (2 API calls/date)")
    ap.add_argument("--limit", type=int, default=0, help="Max dates to process (0 = all)")
    ap.add_argument(
        "--stop-on-401",
        action="store_true",
        help="Stop immediately if API returns 401 (out of credits)",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip dates where all games already have morning+closing probs",
    )
    args = ap.parse_args()

    pg_dsn = os.getenv("PG_DSN")
    if not pg_dsn:
        raise RuntimeError("PG_DSN required")

    client = TheOddsAPI()
    engine = create_engine(pg_dsn, pool_pre_ping=True)
    ensure_movement_columns(engine, args.schema)

    if args.date:
        dates = [date.fromisoformat(args.date)]
    else:
        dates = iter_dates(
            date.fromisoformat(args.start),
            date.fromisoformat(args.end),
            engine,
            args.schema,
            skip_existing=args.skip_existing,
        )

    if args.limit > 0:
        dates = dates[: args.limit]

    closing_hours = [int(x.strip()) for x in args.closing_hours_utc.split(",") if x.strip()]

    print(f"Backfilling {len(dates)} dates ({dates[0] if dates else 'n/a'} → {dates[-1] if dates else 'n/a'})")
    print(f"  Snapshots: morning {args.morning_hour_utc:02d}:00 UTC, closing {closing_hours} UTC")
    print(f"  Sharp threshold: {SHARP_MOVE_THRESHOLD}")

    totals = {"games": 0, "morning": 0, "closing": 0, "movement": 0}
    t0 = time.time()

    for i, d in enumerate(dates, 1):
        try:
            stats = backfill_date(
                engine, args.schema, client, d,
                args.morning_hour_utc, closing_hours,
            )
        except RuntimeError as exc:
            if "ODDS_API_UNAUTHORIZED" in str(exc) and args.stop_on_401:
                print(f"\nStopping: Odds API unauthorized at {d} (credits exhausted?)")
                break
            raise
        for k in totals:
            totals[k] += stats[k]
        if i % 25 == 0 or i == len(dates) or stats["morning"] or stats["closing"]:
            print(
                f"  [{i:4d}/{len(dates)}] {stats['date']}  "
                f"games={stats['games']} morning={stats['morning']} "
                f"closing={stats['closing']} moved={stats['movement']}"
            )
        if args.sleep > 0 and i < len(dates):
            time.sleep(args.sleep)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"  Game-rows touched: morning={totals['morning']:,} closing={totals['closing']:,}")
    print(f"  Dates with movement>0: {totals['movement']:,} game-rows (last date only in counter; see SQL)")


if __name__ == "__main__":
    main()
