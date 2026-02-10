import argparse
import os
import random
import time
from typing import Optional, Dict, Any, Tuple, List

import requests
import psycopg2
from psycopg2.extras import execute_values
from concurrent.futures import ThreadPoolExecutor, as_completed


FEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{gamePk}/feed/live"


# -----------------------------
# HTTP helpers (retry/backoff)
# -----------------------------
def fetch_feed(game_id: int, session: requests.Session, max_retries: int = 6) -> Dict[str, Any]:
    """
    Fetch game feed with retry/backoff on timeouts, 429, and transient 5xx errors.
    """
    url = FEED_URL.format(gamePk=game_id)
    backoff = 0.5

    for attempt in range(1, max_retries + 1):
        try:
            r = session.get(url, timeout=25)
            # handle rate limits + transient server errors
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"status={r.status_code}", response=r)
            r.raise_for_status()
            return r.json()

        except Exception as e:
            # exponential backoff with jitter
            sleep_s = backoff * (2 ** (attempt - 1)) + random.random() * 0.25
            # cap backoff so we don't stall forever
            sleep_s = min(sleep_s, 10.0)
            if attempt == max_retries:
                raise RuntimeError(f"fetch failed for game_id={game_id} after {max_retries} attempts: {e}")
            time.sleep(sleep_s)

    # unreachable
    raise RuntimeError(f"fetch failed for game_id={game_id}")


# -----------------------------
# Extraction logic
# -----------------------------
def extract_from_probables(feed: Dict[str, Any]) -> Tuple[Optional[int], Optional[str], Optional[int], Optional[str]]:
    gp = feed.get("gameData", {}).get("probablePitchers", {})
    home = gp.get("home")
    away = gp.get("away")

    return (
        int(home.get("id")) if home and home.get("id") is not None else None,
        home.get("fullName") if home else None,
        int(away.get("id")) if away and away.get("id") is not None else None,
        away.get("fullName") if away else None,
    )


def extract_from_pbp(feed: Dict[str, Any]) -> Tuple[Optional[int], Optional[str], Optional[int], Optional[str]]:
    plays = feed.get("liveData", {}).get("plays", {}).get("allPlays", [])
    home_sp = away_sp = None
    home_name = away_name = None

    for play in plays:
        half = play.get("about", {}).get("halfInning")
        pitcher = play.get("matchup", {}).get("pitcher", {}) or {}
        pid = pitcher.get("id")
        pname = pitcher.get("fullName")

        if half == "top" and away_sp is None and pid is not None:
            away_sp = int(pid)
            away_name = pname

        if half == "bottom" and home_sp is None and pid is not None:
            home_sp = int(pid)
            home_name = pname

        if home_sp is not None and away_sp is not None:
            break

    return home_sp, home_name, away_sp, away_name


def get_starters(feed: Dict[str, Any]) -> Tuple[Optional[int], Optional[str], Optional[int], Optional[str], str]:
    home_id, home_name, away_id, away_name = extract_from_probables(feed)
    if home_id is not None and away_id is not None:
        return home_id, home_name, away_id, away_name, "probablePitchers"

    h2, hn2, a2, an2 = extract_from_pbp(feed)
    return h2, hn2, a2, an2, "playByPlay"


# -----------------------------
# DB upsert
# -----------------------------
def upsert_starters(conn, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    sql = """
    INSERT INTO game_starting_pitchers
      (game_id, home_sp_id, home_sp_name, away_sp_id, away_sp_name, source)
    VALUES %s
    ON CONFLICT (game_id) DO UPDATE SET
      home_sp_id = EXCLUDED.home_sp_id,
      home_sp_name = EXCLUDED.home_sp_name,
      away_sp_id = EXCLUDED.away_sp_id,
      away_sp_name = EXCLUDED.away_sp_name,
      source = EXCLUDED.source,
      updated_at = now();
    """

    values = [[
        r["game_id"],
        r["home_sp_id"],
        r["home_sp_name"],
        r["away_sp_id"],
        r["away_sp_name"],
        r["source"],
    ] for r in rows]

    with conn.cursor() as cur:
        execute_values(cur, sql, values, page_size=500)
    conn.commit()


# -----------------------------
# Worker function
# -----------------------------
def process_game(game_id: int) -> Dict[str, Any]:
    # Each thread gets its own Session for connection pooling
    if not hasattr(process_game, "_session"):
        process_game._session = requests.Session()  # type: ignore[attr-defined]
    session = process_game._session  # type: ignore[attr-defined]

    feed = fetch_feed(game_id, session=session)
    home_id, home_name, away_id, away_name, source = get_starters(feed)

    return dict(
        game_id=game_id,
        home_sp_id=home_id,
        home_sp_name=home_name,
        away_sp_id=away_id,
        away_sp_name=away_name,
        source=source,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pg_dsn", default=os.environ.get("PG_DSN"), help="postgres dsn; default from $PG_DSN")
    ap.add_argument("--workers", type=int, default=12, help="thread workers (recommended 8-16)")
    ap.add_argument("--batch_size", type=int, default=500, help="db upsert batch size")
    ap.add_argument("--limit", type=int, default=None, help="optional limit for testing")
    args = ap.parse_args()

    if not args.pg_dsn:
        raise SystemExit("Missing --pg_dsn and $PG_DSN is not set")

    conn = psycopg2.connect(args.pg_dsn)

    # Pull game_ids missing starters
    q = """
      SELECT g.game_id
      FROM games g
      LEFT JOIN game_starting_pitchers sp ON sp.game_id = g.game_id
      WHERE sp.game_id IS NULL
      ORDER BY g.game_date ASC;
    """
    with conn.cursor() as cur:
        cur.execute(q)
        game_ids = [int(r[0]) for r in cur.fetchall()]

    if args.limit:
        game_ids = game_ids[: args.limit]

    total = len(game_ids)
    print(f"Need starters for {total} games")
    if total == 0:
        conn.close()
        return

    rows_buffer: List[Dict[str, Any]] = []
    failures = 0
    completed = 0
    t0 = time.time()

    # Parallel fetch/parse, main thread does DB writes in batches
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_game, gid): gid for gid in game_ids}

        for fut in as_completed(futures):
            gid = futures[fut]
            try:
                row = fut.result()
                rows_buffer.append(row)
            except Exception as e:
                failures += 1
                print(f"FAILED game_id={gid}: {e}")

            completed += 1

            # periodic DB flush
            if len(rows_buffer) >= args.batch_size:
                upsert_starters(conn, rows_buffer)
                rows_buffer.clear()

            # progress
            if completed % 500 == 0 or completed == total:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                eta_min = (total - completed) / rate / 60 if rate > 0 else float("inf")
                print(f"Progress: {completed}/{total} | failures={failures} | {rate:.1f} games/s | ETA {eta_min:.1f} min")

    # final flush
    if rows_buffer:
        upsert_starters(conn, rows_buffer)

    conn.close()
    print(f"DONE. completed={completed}, failures={failures}")


if __name__ == "__main__":
    main()
