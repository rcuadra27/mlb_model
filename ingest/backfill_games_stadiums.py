import argparse
import os
import time
import random
import requests
import psycopg2
from concurrent.futures import ThreadPoolExecutor, as_completed

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"


def fetch_schedule(start, end, session):
    params = {
        "sportId": 1,
        "startDate": start,
        "endDate": end,
        "hydrate": "venue",
    }
    r = session.get(SCHEDULE_URL, params=params, timeout=40)
    r.raise_for_status()
    return r.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pg_dsn", default=os.environ.get("PG_DSN"))
    ap.add_argument("--start_season", type=int, required=True)
    ap.add_argument("--end_season", type=int, required=True)
    args = ap.parse_args()

    conn = psycopg2.connect(args.pg_dsn)
    session = requests.Session()

    updates = []

    for season in range(args.start_season, args.end_season + 1):
        print(f"Fetching season {season}")
        data = session.get(
            "https://statsapi.mlb.com/api/v1/schedule",
            params={
                "sportId": 1,
                "season": season,
                "hydrate": "venue",
            },
            timeout=60,
        ).json()

        for d in data.get("dates", []):
            for g in d.get("games", []):
                updates.append((
                    g.get("venue", {}).get("id"),
                    g.get("venue", {}).get("name"),
                    g.get("gameType"),
                    g.get("doubleHeader"),
                    g.get("seriesGameNumber"),
                    g.get("gamePk"),
                ))

        time.sleep(0.3)  # be polite

    print(f"Updating {len(updates)} games")

    sql = """
    UPDATE games
    SET
      venue_id = %s,
      venue_name = %s,
      game_type = %s,
      doubleheader = %s,
      series_game_number = %s,
      updated_at = now()
    WHERE game_id = %s;
    """

    with conn.cursor() as cur:
        cur.executemany(sql, updates)
    conn.commit()
    conn.close()
    print("DONE")


if __name__ == "__main__":
    main()
