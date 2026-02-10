import argparse
import datetime as dt
import time
from typing import Iterator, Dict, Any, Optional

import requests
import psycopg2
from psycopg2.extras import execute_values


SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"


def daterange(start: dt.date, end: dt.date) -> Iterator[dt.date]:
    cur = start
    while cur <= end:
        yield cur
        cur += dt.timedelta(days=1)


def fetch_schedule(date: dt.date) -> Dict[str, Any]:
    params = {
        "sportId": 1,
        "startDate": date.isoformat(),
        "endDate": date.isoformat(),
        "gameTypes": "R",                 # <-- Regular season only
        "hydrate": "linescore,team",
    }
    r = requests.get(SCHEDULE_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json()



def parse_games(schedule_json: Dict[str, Any]) -> list[dict]:
    out = []
    for d in schedule_json.get("dates", []):
        game_date = dt.date.fromisoformat(d["date"])
        for g in d.get("games", []):
            game_id = int(g["gamePk"])
            status = g["status"]["detailedState"]
            season = int(g.get("season", game_date.year))

            home = g["teams"]["home"]["team"]
            away = g["teams"]["away"]["team"]

            home_runs = g["teams"]["home"].get("score")
            away_runs = g["teams"]["away"].get("score")

            home_win: Optional[bool] = None
            if home_runs is not None and away_runs is not None:
                home_win = bool(home_runs > away_runs)

            out.append(
                dict(
                    game_id=game_id,
                    game_date=game_date,
                    season=season,
                    status=status,
                    home_team_id=int(home["id"]),
                    away_team_id=int(away["id"]),
                    home_team_name=str(home["name"]),
                    away_team_name=str(away["name"]),
                    home_runs=home_runs,
                    away_runs=away_runs,
                    home_win=home_win,
                )
            )
    return out


def upsert_games(conn, rows: list[dict]) -> None:
    if not rows:
        return

    cols = [
        "game_id", "game_date", "season", "status",
        "home_team_id", "away_team_id", "home_team_name", "away_team_name",
        "home_runs", "away_runs", "home_win",
    ]
    values = [[r[c] for c in cols] for r in rows]

    sql = f"""
    INSERT INTO games ({",".join(cols)})
    VALUES %s
    ON CONFLICT (game_id) DO UPDATE SET
      game_date=EXCLUDED.game_date,
      season=EXCLUDED.season,
      status=EXCLUDED.status,
      home_team_id=EXCLUDED.home_team_id,
      away_team_id=EXCLUDED.away_team_id,
      home_team_name=EXCLUDED.home_team_name,
      away_team_name=EXCLUDED.away_team_name,
      home_runs=EXCLUDED.home_runs,
      away_runs=EXCLUDED.away_runs,
      home_win=EXCLUDED.home_win,
      updated_at=now();
    """

    with conn.cursor() as cur:
        execute_values(cur, sql, values, page_size=500)
    conn.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--pg_dsn", required=True, help="Postgres DSN, e.g. postgresql://user:pass@host:5432/db")
    ap.add_argument("--sleep", type=float, default=0.15)
    args = ap.parse_args()

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)

    conn = psycopg2.connect(args.pg_dsn)

    for day in daterange(start, end):
        js = fetch_schedule(day)
        rows = parse_games(js)
        upsert_games(conn, rows)
        print(f"{day}: upserted {len(rows)} games")
        time.sleep(args.sleep)

    conn.close()


if __name__ == "__main__":
    main()
