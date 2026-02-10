import argparse
import os
import random
import time
from typing import Optional, Dict, Any, List

import requests
import psycopg2
from psycopg2.extras import execute_values
from concurrent.futures import ThreadPoolExecutor, as_completed

FEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{gamePk}/feed/live"


def ip_str_to_outs(ip: Optional[str]) -> Optional[int]:
    """
    MLB often encodes IP like '5.2' meaning 5 + 2/3 innings = 17 outs.
    Returns outs pitched as int.
    """
    if ip is None:
        return None
    s = str(ip).strip()
    if s == "":
        return None
    if "." not in s:
        try:
            return int(s) * 3
        except:
            return None
    whole, frac = s.split(".", 1)
    try:
        w = int(whole)
        f = int(frac)
        if f not in (0, 1, 2):  # sometimes weird, but usually 0/1/2
            return w * 3
        return w * 3 + f
    except:
        return None


def outs_to_ip(outs: Optional[int]) -> Optional[float]:
    if outs is None:
        return None
    return outs / 3.0


def fetch_feed(game_id: int, session: requests.Session, max_retries: int = 6) -> Dict[str, Any]:
    url = FEED_URL.format(gamePk=game_id)
    for attempt in range(1, max_retries + 1):
        try:
            r = session.get(url, timeout=25)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"status={r.status_code}", response=r)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == max_retries:
                raise RuntimeError(f"fetch failed game_id={game_id}: {e}")
            sleep_s = min(0.5 * (2 ** (attempt - 1)) + random.random() * 0.25, 10.0)
            time.sleep(sleep_s)
    raise RuntimeError(f"fetch failed game_id={game_id}")


def extract_starter_stats(feed: Dict[str, Any], pitcher_id: int, side: str) -> Dict[str, Any]:
    """
    side: 'home' or 'away' (the team the pitcher belongs to in this game)
    Returns outs_pitched, innings_pitched, runs_allowed
    """
    box = feed.get("liveData", {}).get("boxscore", {})
    team = (box.get("teams", {}) or {}).get(side, {}) or {}
    players = team.get("players", {}) or {}
    p = players.get(f"ID{pitcher_id}", {}) or {}
    stats = (p.get("stats", {}) or {}).get("pitching", {}) or {}

    ip = stats.get("inningsPitched")  # typically string like "5.2"
    outs = ip_str_to_outs(ip)
    runs = stats.get("runs")  # int

    return {
        "outs_pitched": outs,
        "innings_pitched": outs_to_ip(outs),
        "runs_allowed": int(runs) if runs is not None else None,
    }


def process_game_row(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    row contains: game_id, game_date, home_team_id, away_team_id, home_sp_id, away_sp_id
    Return 2 pitcher_start rows (home starter + away starter), if possible.
    """
    if not hasattr(process_game_row, "_session"):
        process_game_row._session = requests.Session()  # type: ignore[attr-defined]
    session = process_game_row._session  # type: ignore[attr-defined]

    gid = int(row["game_id"])
    feed = fetch_feed(gid, session=session)

    out: List[Dict[str, Any]] = []

    # Home starter row
    if row["home_sp_id"] is not None:
        pid = int(row["home_sp_id"])
        st = extract_starter_stats(feed, pid, "home")
        out.append({
            "game_id": gid,
            "game_date": row["game_date"],
            "pitcher_id": pid,
            "team_id": int(row["home_team_id"]),
            "opponent_team_id": int(row["away_team_id"]),
            "is_home": True,
            **st,
            "source": "mlb_feed_boxscore",
        })

    # Away starter row
    if row["away_sp_id"] is not None:
        pid = int(row["away_sp_id"])
        st = extract_starter_stats(feed, pid, "away")
        out.append({
            "game_id": gid,
            "game_date": row["game_date"],
            "pitcher_id": pid,
            "team_id": int(row["away_team_id"]),
            "opponent_team_id": int(row["home_team_id"]),
            "is_home": False,
            **st,
            "source": "mlb_feed_boxscore",
        })

    return out


def upsert_pitcher_starts(conn, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    sql = """
    INSERT INTO pitcher_starts (
      game_id, game_date, pitcher_id, team_id, opponent_team_id, is_home,
      outs_pitched, innings_pitched, runs_allowed, source
    )
    VALUES %s
    ON CONFLICT (game_id, pitcher_id) DO UPDATE SET
      game_date = EXCLUDED.game_date,
      team_id = EXCLUDED.team_id,
      opponent_team_id = EXCLUDED.opponent_team_id,
      is_home = EXCLUDED.is_home,
      outs_pitched = EXCLUDED.outs_pitched,
      innings_pitched = EXCLUDED.innings_pitched,
      runs_allowed = EXCLUDED.runs_allowed,
      source = EXCLUDED.source,
      updated_at = now();
    """

    values = [[
        r["game_id"], r["game_date"], r["pitcher_id"], r["team_id"], r["opponent_team_id"], r["is_home"],
        r["outs_pitched"], r["innings_pitched"], r["runs_allowed"], r["source"]
    ] for r in rows]

    with conn.cursor() as cur:
        execute_values(cur, sql, values, page_size=1000)
    conn.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pg_dsn", default=os.environ.get("PG_DSN"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not args.pg_dsn:
        raise SystemExit("Missing --pg_dsn and $PG_DSN is not set")

    conn = psycopg2.connect(args.pg_dsn)

    # Select games where starters exist, and we haven't ingested pitcher_starts for them yet
    q = """
    SELECT
      g.game_id,
      g.game_date,
      g.home_team_id,
      g.away_team_id,
      sp.home_sp_id,
      sp.away_sp_id
    FROM games g
    JOIN game_starting_pitchers sp ON sp.game_id = g.game_id
    LEFT JOIN pitcher_starts ps_home ON ps_home.game_id = g.game_id AND ps_home.pitcher_id = sp.home_sp_id
    LEFT JOIN pitcher_starts ps_away ON ps_away.game_id = g.game_id AND ps_away.pitcher_id = sp.away_sp_id
    WHERE
      g.home_runs IS NOT NULL AND g.away_runs IS NOT NULL
      AND (ps_home.game_id IS NULL OR ps_away.game_id IS NULL)
    ORDER BY g.game_date, g.game_id;
    """

    with conn.cursor() as cur:
        cur.execute(q)
        raw = cur.fetchall()

    rows = [{
        "game_id": r[0],
        "game_date": r[1],
        "home_team_id": r[2],
        "away_team_id": r[3],
        "home_sp_id": r[4],
        "away_sp_id": r[5],
    } for r in raw]

    if args.limit:
        rows = rows[: args.limit]

    total = len(rows)
    print(f"Need starter boxscore stats for {total} games")

    buffer: List[Dict[str, Any]] = []
    failures = 0
    completed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_game_row, r): r["game_id"] for r in rows}

        for fut in as_completed(futures):
            gid = futures[fut]
            try:
                out_rows = fut.result()
                buffer.extend(out_rows)
            except Exception as e:
                failures += 1
                print(f"FAILED game_id={gid}: {e}")

            completed += 1

            if len(buffer) >= args.batch_size:
                upsert_pitcher_starts(conn, buffer)
                buffer.clear()

            if completed % 500 == 0 or completed == total:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                eta_min = (total - completed) / rate / 60 if rate > 0 else float("inf")
                print(f"Progress: {completed}/{total} | failures={failures} | {rate:.1f} games/s | ETA {eta_min:.1f} min")

    if buffer:
        upsert_pitcher_starts(conn, buffer)

    conn.close()
    print(f"DONE. completed={completed}, failures={failures}")


if __name__ == "__main__":
    main()
