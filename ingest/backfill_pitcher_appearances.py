import argparse
import os
import random
import time
from typing import Optional, Dict, Any, List, Tuple

import requests
import psycopg2
from psycopg2.extras import execute_values
from concurrent.futures import ThreadPoolExecutor, as_completed

FEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{gamePk}/feed/live"


def ip_str_to_outs(ip: Optional[str]) -> Optional[int]:
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
        # MLB convention: .1=1 out, .2=2 outs
        if f not in (0, 1, 2):
            return w * 3
        return w * 3 + f
    except:
        return None


def outs_to_ip(outs: Optional[int]) -> Optional[float]:
    return None if outs is None else outs / 3.0


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


def extract_team_pitchers(feed: Dict[str, Any], side: str) -> Dict[int, Dict[str, Any]]:
    """
    Returns mapping pitcher_id -> pitching stats dict for a given side ('home'/'away').
    """
    box = feed.get("liveData", {}).get("boxscore", {}) or {}
    team = (box.get("teams", {}) or {}).get(side, {}) or {}
    players = team.get("players", {}) or {}
    pitcher_ids = team.get("pitchers") or []

    out = {}
    for pid in pitcher_ids:
        p = players.get(f"ID{pid}", {}) or {}
        st = (p.get("stats", {}) or {}).get("pitching", {}) or {}
        ip = st.get("inningsPitched")
        outs = ip_str_to_outs(ip)
        out[int(pid)] = {
            "outs_pitched": outs,
            "innings_pitched": outs_to_ip(outs),
            "runs_allowed": int(st["runs"]) if st.get("runs") is not None else None,
        }
    return out


def process_game(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not hasattr(process_game, "_session"):
        process_game._session = requests.Session()  # type: ignore[attr-defined]
    session = process_game._session  # type: ignore[attr-defined]

    gid = int(row["game_id"])
    feed = fetch_feed(gid, session=session)

    home_pitchers = extract_team_pitchers(feed, "home")
    away_pitchers = extract_team_pitchers(feed, "away")

    home_sp = row.get("home_sp_id")
    away_sp = row.get("away_sp_id")
    if home_sp is not None:
        home_sp = int(home_sp)
    if away_sp is not None:
        away_sp = int(away_sp)

    out: List[Dict[str, Any]] = []

    # Home team pitchers
    for pid, st in home_pitchers.items():
        out.append({
            "game_id": gid,
            "game_date": row["game_date"],
            "pitcher_id": pid,
            "team_id": int(row["home_team_id"]),
            "opponent_team_id": int(row["away_team_id"]),
            "is_home": True,
            "outs_pitched": st["outs_pitched"],
            "innings_pitched": st["innings_pitched"],
            "runs_allowed": st["runs_allowed"],
            "is_starter": (home_sp is not None and pid == home_sp),
            "source": "mlb_feed_boxscore",
        })

    # Away team pitchers
    for pid, st in away_pitchers.items():
        out.append({
            "game_id": gid,
            "game_date": row["game_date"],
            "pitcher_id": pid,
            "team_id": int(row["away_team_id"]),
            "opponent_team_id": int(row["home_team_id"]),
            "is_home": False,
            "outs_pitched": st["outs_pitched"],
            "innings_pitched": st["innings_pitched"],
            "runs_allowed": st["runs_allowed"],
            "is_starter": (away_sp is not None and pid == away_sp),
            "source": "mlb_feed_boxscore",
        })

    return out


def upsert(conn, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    sql = """
    INSERT INTO pitcher_appearances (
      game_id, game_date, pitcher_id, team_id, opponent_team_id, is_home,
      outs_pitched, innings_pitched, runs_allowed, is_starter, source
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
      is_starter = EXCLUDED.is_starter,
      source = EXCLUDED.source,
      updated_at = now();
    """
    values = [[
        r["game_id"], r["game_date"], r["pitcher_id"], r["team_id"], r["opponent_team_id"], r["is_home"],
        r["outs_pitched"], r["innings_pitched"], r["runs_allowed"], r["is_starter"], r["source"]
    ] for r in rows]

    with conn.cursor() as cur:
        execute_values(cur, sql, values, page_size=2000)
    conn.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pg_dsn", default=os.environ.get("PG_DSN"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not args.pg_dsn:
        raise SystemExit("Missing --pg_dsn and $PG_DSN is not set")

    conn = psycopg2.connect(args.pg_dsn)

    # Only completed games; join starters so we can mark is_starter
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
    LEFT JOIN pitcher_appearances pa ON pa.game_id = g.game_id
    WHERE g.home_runs IS NOT NULL AND g.away_runs IS NOT NULL
      AND pa.game_id IS NULL
    ORDER BY g.game_date, g.game_id;
    """
    with conn.cursor() as cur:
        cur.execute(q)
        raw = cur.fetchall()

    jobs = [{
        "game_id": r[0],
        "game_date": r[1],
        "home_team_id": r[2],
        "away_team_id": r[3],
        "home_sp_id": r[4],
        "away_sp_id": r[5],
    } for r in raw]

    if args.limit:
        jobs = jobs[: args.limit]

    total = len(jobs)
    print(f"Need pitcher appearances for {total} games")

    buf: List[Dict[str, Any]] = []
    failures = 0
    completed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_game, job): job["game_id"] for job in jobs}

        for fut in as_completed(futures):
            gid = futures[fut]
            try:
                rows = fut.result()
                buf.extend(rows)
            except Exception as e:
                failures += 1
                print(f"FAILED game_id={gid}: {e}")

            completed += 1

            if len(buf) >= args.batch_size:
                upsert(conn, buf)
                buf.clear()

            if completed % 500 == 0 or completed == total:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                eta_min = (total - completed) / rate / 60 if rate > 0 else float("inf")
                print(f"Progress: {completed}/{total} | failures={failures} | {rate:.1f} games/s | ETA {eta_min:.1f} min")

    if buf:
        upsert(conn, buf)

    conn.close()
    print(f"DONE. completed={completed}, failures={failures}")


if __name__ == "__main__":
    main()
