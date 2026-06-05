#!/usr/bin/env python3
"""
Backfill historical game-time weather into features_game.

For past seasons we store Open-Meteo *archive* actuals in the same columns
used at inference time (forecast_*), so training and live paths stay aligned.

Two modes (run both in a full backfill):
  1. --sync-game-weather   Copy existing game_weather rows → features_game
  2. Default fetch         Open-Meteo archive API for games still missing weather

Uses venue lat/lon from public.venues (geocoded by ingest/backfill_weather.py),
with VENUE_COORDS fallback from weather_forecast.py.

Usage:
  PG_DSN=... python features/backfill_weather_historical.py \\
      --start 2021-01-01 --end 2024-12-31 --sync-game-weather

  PG_DSN=... python features/backfill_weather_historical.py \\
      --start 2015-04-01 --end 2019-12-31 --sleep 0.12

  PG_DSN=... python features/backfill_weather_historical.py \\
      --start 2015-04-01 --end 2024-12-31 --sync-game-weather --sleep 0.12
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests
from sqlalchemy import create_engine, text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from features.weather_forecast import (
    VENUE_COORDS,
    celsius_to_fahrenheit,
    ensure_forecast_columns,
)

MLB_SCHEDULE = "https://statsapi.mlb.com/api/v1/schedule"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

FORECAST_COLS = [
    "forecast_temp_f",
    "forecast_wind_mph",
    "forecast_wind_dir_deg",
    "forecast_precip_in",
    "forecast_humidity",
]


def parse_date(s: str) -> date:
    return date.fromisoformat(s)


def venue_coords(engine, schema: str, venue_id: int, home_team_id: int) -> tuple[float, float] | None:
    with engine.connect() as conn:
        row = conn.execute(
            text(f"""
                SELECT lat, lon FROM {schema}.venues
                WHERE venue_id = :vid AND lat IS NOT NULL AND lon IS NOT NULL
            """),
            {"vid": venue_id},
        ).fetchone()
    if row and row[0] is not None and row[1] is not None:
        return float(row[0]), float(row[1])
    coords = VENUE_COORDS.get(int(venue_id)) or VENUE_COORDS.get(int(home_team_id))
    if coords:
        return float(coords[0]), float(coords[1])
    return None


def fetch_commence_time(game_id: int, game_date: date, session: requests.Session) -> datetime | None:
    """MLB schedule fallback when first_pitch_utc / game_weather.commence_time missing."""
    try:
        r = session.get(
            MLB_SCHEDULE,
            params={"sportId": 1, "date": game_date.isoformat()},
            timeout=30,
        )
        r.raise_for_status()
        for d in r.json().get("dates") or []:
            for g in d.get("games") or []:
                if int(g.get("gamePk", -1)) != game_id:
                    continue
                iso = g.get("gameDate")
                if not iso:
                    return None
                return datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except Exception:
        return None
    return None


def pick_closest_hour_open_meteo(
    lat: float,
    lon: float,
    commence_utc: datetime,
    session: requests.Session,
) -> dict | None:
    """Hourly archive actuals ±3h around first pitch (UTC)."""
    if commence_utc.tzinfo is None:
        commence_utc = commence_utc.replace(tzinfo=timezone.utc)
    start = (commence_utc - timedelta(hours=3)).replace(minute=0, second=0, microsecond=0)
    end = (commence_utc + timedelta(hours=3)).replace(minute=0, second=0, microsecond=0)

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.date().isoformat(),
        "end_date": end.date().isoformat(),
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "wind_direction_10m",
        ]),
        "timezone": "UTC",
    }
    try:
        resp = session.get(OPEN_METEO_ARCHIVE, params=params, timeout=60)
        resp.raise_for_status()
        hourly = resp.json().get("hourly") or {}
    except Exception as exc:
        print(f"    Open-Meteo error: {exc}")
        return None

    times = hourly.get("time") or []
    if not times:
        return None

    target = commence_utc.replace(minute=0, second=0, microsecond=0)
    best_i, best_dt = None, None
    for i, t in enumerate(times):
        try:
            dt = datetime.fromisoformat(t).replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if dt < start or dt > end:
            continue
        if best_dt is None or abs((dt - target).total_seconds()) < abs((best_dt - target).total_seconds()):
            best_dt, best_i = dt, i

    if best_i is None:
        return None

    def get(key: str):
        arr = hourly.get(key)
        if arr is None or best_i >= len(arr):
            return None
        return arr[best_i]

    temp_c = get("temperature_2m")
    wind_kmh = get("wind_speed_10m")
    precip_mm = get("precipitation")
    return {
        "forecast_temp_f": float(celsius_to_fahrenheit(temp_c)) if temp_c is not None else None,
        "forecast_wind_mph": float(wind_kmh) * 0.621371 if wind_kmh is not None else None,
        "forecast_wind_dir_deg": float(get("wind_direction_10m")) if get("wind_direction_10m") is not None else None,
        "forecast_precip_in": float(precip_mm) / 25.4 if precip_mm is not None else None,
        "forecast_humidity": float(get("relative_humidity_2m")) if get("relative_humidity_2m") is not None else None,
    }


def sync_from_game_weather(engine, schema: str, start: date, end: date) -> int:
    """Copy game_weather actuals into features_game.forecast_* columns."""
    with engine.begin() as conn:
        result = conn.execute(
            text(f"""
                UPDATE {schema}.features_game AS fg
                SET
                    forecast_temp_f       = gw.temp_f,
                    forecast_wind_mph     = gw.wind_mph,
                    forecast_wind_dir_deg = gw.wind_dir_deg,
                    forecast_precip_in    = gw.precip_in,
                    forecast_humidity     = gw.humidity
                FROM {schema}.game_weather AS gw
                JOIN {schema}.games AS g ON g.game_id = gw.game_id
                WHERE fg.game_id = gw.game_id
                  AND g.game_date BETWEEN :start AND :end
                  AND gw.temp_f IS NOT NULL
            """),
            {"start": start.isoformat(), "end": end.isoformat()},
        )
        n = int(result.rowcount or 0)
    print(f"  Synced {n:,} rows from game_weather → features_game")
    return n


def load_games_needing_weather(
    engine, schema: str, start: date, end: date, skip_existing: bool,
) -> pd.DataFrame:
    skip_clause = "AND fg.forecast_temp_f IS NULL" if skip_existing else ""
    return pd.read_sql(
        text(f"""
            SELECT
                g.game_id,
                g.game_date,
                g.venue_id,
                g.home_team_id,
                g.first_pitch_utc,
                gw.commence_time AS gw_commence_time
            FROM {schema}.games g
            JOIN {schema}.features_game fg ON fg.game_id = g.game_id
            LEFT JOIN {schema}.game_weather gw ON gw.game_id = g.game_id
            WHERE g.game_date BETWEEN :start AND :end
              AND g.venue_id IS NOT NULL
              {skip_clause}
            ORDER BY g.game_date, g.game_id
        """),
        engine,
        params={"start": start.isoformat(), "end": end.isoformat()},
    )


def upsert_forecast_batch(engine, schema: str, rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    for col in FORECAST_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    cols = [c for c in df.columns if c != "game_id"]
    set_clause = ", ".join(f"{c} = s.{c}" for c in cols)
    with engine.begin() as conn:
        df.to_sql("_wx_hist_tmp", conn, schema=schema, if_exists="replace", index=False, method="multi")
        conn.execute(text(f"""
            UPDATE {schema}.features_game AS t
            SET {set_clause}
            FROM {schema}._wx_hist_tmp AS s
            WHERE t.game_id = s.game_id
        """))
        conn.execute(text(f"DROP TABLE IF EXISTS {schema}._wx_hist_tmp"))


def fetch_open_meteo_range(
    engine,
    schema: str,
    start: date,
    end: date,
    sleep_s: float,
    limit: int,
) -> None:
    games = load_games_needing_weather(engine, schema, start, end, skip_existing=True)
    if limit > 0:
        games = games.head(limit)
    print(f"  Games needing Open-Meteo fetch: {len(games):,}")

    session = requests.Session()
    schedule_cache: dict[date, dict[int, datetime]] = {}
    batch: list[dict] = []
    ok, skip = 0, 0

    for i, row in games.iterrows():
        gid = int(row["game_id"])
        gdate = pd.Timestamp(row["game_date"]).date()
        vid = int(row["venue_id"])
        htid = int(row["home_team_id"])

        coords = venue_coords(engine, schema, vid, htid)
        if coords is None:
            skip += 1
            continue
        lat, lon = coords

        ct = row.get("gw_commence_time") or row.get("first_pitch_utc")
        if pd.isna(ct):
            if gdate not in schedule_cache:
                schedule_cache[gdate] = {}
                # populate whole day lazily per game (inefficient but simple)
            if gid not in schedule_cache.get(gdate, {}):
                ct_found = fetch_commence_time(gid, gdate, session)
                schedule_cache.setdefault(gdate, {})[gid] = ct_found
                time.sleep(0.05)
            ct = schedule_cache.get(gdate, {}).get(gid)
        if ct is None or (isinstance(ct, float) and pd.isna(ct)):
            skip += 1
            continue
        if isinstance(ct, pd.Timestamp):
            ct = ct.to_pydatetime()

        wx = pick_closest_hour_open_meteo(lat, lon, ct, session)
        if not wx:
            skip += 1
            time.sleep(sleep_s)
            continue

        batch.append({"game_id": gid, **wx})
        ok += 1

        if len(batch) >= 200:
            upsert_forecast_batch(engine, schema, batch)
            batch = []
            print(f"    [{ok + skip:5d}] fetched={ok:,} skipped={skip:,}")

        time.sleep(sleep_s)

    if batch:
        upsert_forecast_batch(engine, schema, batch)
    print(f"  Open-Meteo done: fetched={ok:,}, skipped={skip:,}")


def print_coverage(engine, schema: str, start: date, end: date) -> None:
    df = pd.read_sql(
        text(f"""
            SELECT
                EXTRACT(YEAR FROM g.game_date)::int AS season,
                COUNT(*) AS games,
                COUNT(fg.forecast_temp_f) AS has_temp,
                COUNT(fg.forecast_wind_mph) AS has_wind,
                COUNT(fg.forecast_wind_dir_deg) AS has_dir
            FROM {schema}.games g
            JOIN {schema}.features_game fg ON fg.game_id = g.game_id
            WHERE g.game_date BETWEEN :start AND :end
            GROUP BY 1
            ORDER BY 1
        """),
        engine,
        params={"start": start.isoformat(), "end": end.isoformat()},
    )
    print("\n  Coverage by season:")
    print(f"  {'Season':>6}  {'Games':>6}  {'Temp':>6}  {'Wind':>6}  {'Dir':>6}")
    for _, r in df.iterrows():
        print(f"  {int(r['season']):>6}  {int(r['games']):>6}  "
              f"{int(r['has_temp']):>6}  {int(r['has_wind']):>6}  {int(r['has_dir']):>6}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", default="public")
    ap.add_argument("--start", default="2015-04-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--sleep", type=float, default=0.12, help="Seconds between Open-Meteo calls")
    ap.add_argument("--limit", type=int, default=0, help="Max games to fetch (0 = all)")
    ap.add_argument(
        "--sync-game-weather",
        action="store_true",
        help="Copy game_weather → features_game before fetching gaps",
    )
    ap.add_argument(
        "--sync-only",
        action="store_true",
        help="Only sync from game_weather; do not call Open-Meteo",
    )
    args = ap.parse_args()

    pg_dsn = os.getenv("PG_DSN")
    if not pg_dsn:
        raise RuntimeError("PG_DSN required")

    start, end = parse_date(args.start), parse_date(args.end)
    engine = create_engine(pg_dsn, pool_pre_ping=True)
    ensure_forecast_columns(engine, args.schema)

    print(f"Weather historical backfill: {start} → {end}")

    if args.sync_game_weather or args.sync_only:
        sync_from_game_weather(engine, args.schema, start, end)

    if not args.sync_only:
        fetch_open_meteo_range(
            engine, args.schema, start, end, args.sleep, args.limit,
        )

    print_coverage(engine, args.schema, start, end)


if __name__ == "__main__":
    main()
