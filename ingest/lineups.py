#!/usr/bin/env python3
"""
ingest/lineups.py

Ingest MLB batting orders (lineups) into public.game_lineups.

Table assumed:
  public.game_lineups (
    game_id BIGINT NOT NULL,
    game_date DATE NOT NULL,
    team_id INT NOT NULL,
    is_home BOOLEAN NOT NULL,
    batting_order INT NOT NULL,
    player_id INT NOT NULL,
    bats TEXT,
    pos TEXT,
    PRIMARY KEY (game_id, team_id, batting_order)
  )

Usage:
  export PG_DSN="postgresql+psycopg2://USER:PASS@HOST:5432/DBNAME"
  python ingest/lineups.py --start 2024-03-01 --end 2024-10-01 --sleep 0.25
"""

import os
import time
import argparse
import datetime as dt
from typing import Dict, Any, List, Tuple

import requests
from sqlalchemy import create_engine, text


SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
BOXSCORE_URL = "https://statsapi.mlb.com/api/v1/game/{gamePk}/boxscore"


def daterange(start: dt.date, end: dt.date):
    cur = start
    while cur <= end:
        yield cur
        cur += dt.timedelta(days=1)


def get_json(url: str, params: Dict[str, Any] | None = None, timeout: int = 30) -> Dict[str, Any]:
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def fetch_game_pks(start_date: str, end_date: str) -> List[Tuple[int, str]]:
    """
    Returns list of (gamePk, game_date_str) for MLB regular + postseason games in date range.
    game_date_str is YYYY-MM-DD (the schedule date bucket).
    """
    params = {
        "sportId": 1,
        "startDate": start_date,
        "endDate": end_date,
    }
    data = get_json(SCHEDULE_URL, params=params)

    out: List[Tuple[int, str]] = []
    for d in data.get("dates", []):
        game_date = d.get("date")  # "YYYY-MM-DD"
        for g in d.get("games", []):
            game_pk = g.get("gamePk")
            if game_pk is not None:
                out.append((int(game_pk), game_date))
    return out


def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def parse_team_lineup(box: Dict[str, Any], side: str) -> Tuple[int, List[int], Dict[int, Dict[str, Any]]]:
    """
    side: "home" or "away"
    Returns:
      team_id, batting_order_player_ids, players_dict_for_side
    """
    teams = box.get("teams", {})
    team_obj = teams.get(side, {})
    team_id = int(safe_get(team_obj, ["team", "id"], 0) or 0)

    batting_order = team_obj.get("battingOrder") or []
    # battingOrder often comes as list of player id strings like ["ID12345", "ID67890"]
    player_ids: List[int] = []
    for x in batting_order:
        if isinstance(x, str) and x.startswith("ID"):
            x = x[2:]
        try:
            player_ids.append(int(x))
        except Exception:
            continue

    # players are keyed like "ID12345"
    players_dict = team_obj.get("players") or {}
    return team_id, player_ids, players_dict


def extract_player_meta(players_dict: Dict[str, Any], player_id: int) -> Tuple[str | None, str | None]:
    """
    Extract bats and pos from boxscore players dict if available.
    This API is a bit inconsistent across seasons/contexts, so we handle multiple paths.
    """
    key = f"ID{player_id}"
    p = players_dict.get(key, {}) if isinstance(players_dict, dict) else {}

    # bats: try a few likely locations
    bats = (
        safe_get(p, ["person", "batSide", "code"])
        or safe_get(p, ["batSide", "code"])
        or safe_get(p, ["batSide", "description"])
    )
    if isinstance(bats, str):
        bats = bats.strip()[:10]
    else:
        bats = None

    # pos: player position abbreviation
    pos = (
        safe_get(p, ["position", "abbreviation"])
        or safe_get(p, ["position", "name"])
    )
    if isinstance(pos, str):
        pos = pos.strip()[:10]
    else:
        pos = None

    return bats, pos


def build_rows_for_game(game_pk: int, game_date: str, sleep: float = 0.0) -> List[Dict[str, Any]]:
    """
    Fetch boxscore and build rows for both home and away.
    Returns list of dicts matching INSERT parameters.
    """
    url = BOXSCORE_URL.format(gamePk=game_pk)
    box = get_json(url)

    rows: List[Dict[str, Any]] = []

    for side, is_home in [("home", True), ("away", False)]:
        team_id, player_ids, players_dict = parse_team_lineup(box, side)

        # If lineup not available yet, skip
        if team_id == 0 or len(player_ids) == 0:
            continue

        for i, pid in enumerate(player_ids, start=1):
            bats, pos = extract_player_meta(players_dict, pid)
            rows.append({
                "game_id": int(game_pk),
                "game_date": game_date,
                "team_id": int(team_id),
                "is_home": bool(is_home),
                "batting_order": int(i),
                "player_id": int(pid),
                "bats": bats,
                "pos": pos,
            })

    if sleep > 0:
        time.sleep(sleep)

    return rows


def upsert_rows(engine, rows: List[Dict[str, Any]]) -> int:
    if not rows:
        return 0

    sql = text("""
        INSERT INTO public.game_lineups (
            game_id, game_date, team_id, is_home, batting_order, player_id, bats, pos
        )
        VALUES (
            :game_id, :game_date, :team_id, :is_home, :batting_order, :player_id, :bats, :pos
        )
        ON CONFLICT (game_id, team_id, batting_order)
        DO UPDATE SET
            player_id = EXCLUDED.player_id,
            bats = EXCLUDED.bats,
            pos = EXCLUDED.pos,
            is_home = EXCLUDED.is_home,
            game_date = EXCLUDED.game_date
    """)

    with engine.begin() as conn:
        conn.execute(sql, rows)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--sleep", type=float, default=0.2, help="Seconds to sleep per gamePk boxscore call")
    ap.add_argument("--limit_games", type=int, default=0, help="Debug: limit number of games processed (0 = no limit)")
    ap.add_argument("--pg_dsn", default=os.environ.get("PG_DSN"), help="PostgreSQL DSN (default: PG_DSN env var)")
    args = ap.parse_args()

    pg_dsn = args.pg_dsn
    if not pg_dsn:
        raise RuntimeError("PG_DSN env var not set. Set it to your real connection string, e.g. export PG_DSN='postgresql+psycopg2://YOUR_USER:YOUR_PASSWORD@localhost:5432/mlb_model'")
    # Catch placeholder DSNs so we don't get confusing "role USER does not exist" from Postgres
    if "/USER:" in pg_dsn.upper() or ":PASS@" in pg_dsn.upper():
        raise RuntimeError(
            "PG_DSN looks like the example placeholder. Replace USER and PASS with your real PostgreSQL username and password. "
            "Example: export PG_DSN='postgresql+psycopg2://myuser:mypassword@localhost:5432/mlb_model'"
        )

    engine = create_engine(pg_dsn, pool_pre_ping=True)

    # Fetch game IDs
    game_pks = fetch_game_pks(args.start, args.end)
    if args.limit_games and args.limit_games > 0:
        game_pks = game_pks[: args.limit_games]

    print(f"Found {len(game_pks):,} games from {args.start} to {args.end}")

    total_rows = 0
    total_games_with_lineups = 0

    for idx, (game_pk, game_date) in enumerate(game_pks, start=1):
        try:
            rows = build_rows_for_game(game_pk, game_date, sleep=args.sleep)
            n = upsert_rows(engine, rows)
            total_rows += n
            if n > 0:
                total_games_with_lineups += 1

            if idx % 50 == 0 or idx == len(game_pks):
                print(f"[{idx:,}/{len(game_pks):,}] gamePk={game_pk} rows_upserted={n} total_rows={total_rows:,}")

        except requests.HTTPError as e:
            # Some games can be missing boxscore or API hiccups
            print(f"[WARN] gamePk={game_pk} HTTPError: {e}")
            continue
        except Exception as e:
            print(f"[WARN] gamePk={game_pk} error: {repr(e)}")
            continue

    print("\nDone.")
    print(f"Games processed: {len(game_pks):,}")
    print(f"Games with lineups inserted: {total_games_with_lineups:,}")
    print(f"Total rows upserted: {total_rows:,}")


if __name__ == "__main__":
    main()
