import argparse
import os
import time
from datetime import datetime, date, timedelta, timezone
from typing import Dict, Optional, Tuple, List

import psycopg2
import psycopg2.extras
import requests


MLB_STATSAPI = "https://statsapi.mlb.com/api/v1"


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


# -----------------------------
# Venue lat/lon via MLB StatsAPI
# -----------------------------
def fetch_venue_latlon(_venue_id: int, query: str, session: requests.Session) -> Tuple[Optional[float], Optional[float]]:
    """
    Geocode a stadium/venue query using OpenStreetMap Nominatim (POI-friendly).
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": query,
        "format": "json",
        "limit": 1,
    }
    headers = {
        # Set something descriptive (Nominatim policy)
        "User-Agent": "mlb_model_weather_geocoder/1.0 (rodrigo)",
    }
    r = session.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    arr = r.json()
    if not arr:
        return None, None
    try:
        return float(arr[0]["lat"]), float(arr[0]["lon"])
    except Exception:
        return None, None

        
def populate_missing_venue_coords(conn, sleep_s: float = 0.15) -> None:
    """
    Fill venues.lat/lon for rows missing coords using MLB StatsAPI.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT venue_id, venue_name, geocode_query
            FROM venues
            WHERE lat IS NULL OR lon IS NULL
            ORDER BY venue_id;
        """)
        missing = cur.fetchall()

    if not missing:
        print("[venues] All venues already have lat/lon.")
        return

    print(f"[venues] Need coords for {len(missing)} venues...")
    
    sess = requests.Session()
    updates = []

    for i, (vid, vname, gq) in enumerate(missing, 1):
        query = gq or vname
        lat, lon = None, None
        try:
            lat, lon = fetch_venue_latlon(int(vid), str(query), sess)
        except Exception as e:
            print(f"[venues] venue_id={vid} geocode failed: {e}")

        if lat is None or lon is None:
            print(f"[venues] No coords for venue_id={vid} name={query!r}")

        updates.append((int(vid), lat, lon))

        if i % 25 == 0:
            print(f"[venues] fetched {i}/{len(missing)}...")
        time.sleep(sleep_s)

    with conn.cursor() as cur:
        for vid, lat, lon in updates:
            if lat is None or lon is None:
                continue
            cur.execute(
                "UPDATE venues SET lat=%s, lon=%s WHERE venue_id=%s",
                (lat, lon, vid),
            )
    conn.commit()
    print("[venues] Updated venue coords.")

# -----------------------------
# Get commence_time per game_id
# -----------------------------
def load_commence_times_from_odds(conn, start: date, end: date) -> Dict[int, datetime]:
    """
    Prefer commence_time from odds_ml (it is already in UTC tz).
    We take MIN(commence_time) per game_id (first pitch-ish).
    """
    q = """
    SELECT game_id, MIN(commence_time) AS ct
    FROM odds_ml
    WHERE game_id IS NOT NULL
      AND commence_time IS NOT NULL
      AND game_date BETWEEN %s AND %s
    GROUP BY game_id;
    """
    m: Dict[int, datetime] = {}
    with conn.cursor() as cur:
        cur.execute(q, (start, end))
        for game_id, ct in cur.fetchall():
            if ct is not None:
                m[int(game_id)] = ct
    return m


def load_commence_times_from_mlb_schedule_for_date(day: date, session: requests.Session) -> Dict[int, datetime]:
    """
    MLB schedule endpoint returns gamePk (your game_id) and gameDate (UTC).
    """
    url = f"{MLB_STATSAPI}/schedule"
    params = {"sportId": 1, "date": day.isoformat()}
    r = session.get(url, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()

    out: Dict[int, datetime] = {}
    for d in j.get("dates") or []:
        for g in d.get("games") or []:
            game_pk = g.get("gamePk")
            game_date_iso = g.get("gameDate")  # e.g. "2021-07-01T23:10:00Z"
            if game_pk is None or not game_date_iso:
                continue
            try:
                ct = datetime.fromisoformat(game_date_iso.replace("Z", "+00:00"))
                out[int(game_pk)] = ct
            except Exception:
                continue
    return out


def build_game_commence_time_map(conn, start: date, end: date, sleep_s: float = 0.1) -> Dict[int, datetime]:
    """
    For games in [start,end], create best-guess commence_time (UTC):
    1) odds_ml.commence_time when available
    2) MLB schedule per day fallback
    """
    odds_map = load_commence_times_from_odds(conn, start, end)

    # get all game_ids in range
    with conn.cursor() as cur:
        cur.execute("""
            SELECT game_id, game_date
            FROM games
            WHERE game_date BETWEEN %s AND %s
              AND game_type IN ('R','P');
        """, (start, end))
        rows = cur.fetchall()

    # Fill missing with schedule times
    need_days = set()
    game_day: Dict[int, date] = {}
    for gid, gd in rows:
        gid = int(gid)
        game_day[gid] = gd
        if gid not in odds_map:
            need_days.add(gd)

    print(f"[commence] have odds commence_time for {len(odds_map)} games; need schedule fallback for {len(need_days)} days")

    sess = requests.Session()
    for i, day in enumerate(sorted(need_days), 1):
        sch = load_commence_times_from_mlb_schedule_for_date(day, sess)
        # Only fill missing
        for gid, ct in sch.items():
            if gid in game_day and gid not in odds_map:
                odds_map[gid] = ct
        if i % 50 == 0:
            print(f"[commence] schedule fetched {i}/{len(need_days)} days...")
        time.sleep(sleep_s)

    return odds_map


# -----------------------------
# Weather Providers
# -----------------------------
def pick_closest_hour_from_open_meteo(lat: float, lon: float, ct_utc: datetime, session: requests.Session) -> Optional[dict]:
    """
    Uses Open-Meteo Historical API (no key).
    Requests +/- 3 hours around commence_time, then picks closest hourly datapoint (UTC).
    """
    # build time window
    start = (ct_utc - timedelta(hours=3)).replace(minute=0, second=0, microsecond=0)
    end = (ct_utc + timedelta(hours=3)).replace(minute=0, second=0, microsecond=0)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.date().isoformat(),
        "end_date": end.date().isoformat(),
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "surface_pressure",
            "wind_speed_10m",
            "wind_direction_10m",
        ]),
        "timezone": "UTC",
    }
    r = session.get(url, params=params, timeout=60)
    r.raise_for_status()
    j = r.json()

    hourly = j.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return None

    # Parse hours and pick closest
    target = ct_utc.replace(minute=0, second=0, microsecond=0)
    best_i = None
    best_dt = None
    for i, t in enumerate(times):
        # "2021-07-01T23:00"
        try:
            dt = datetime.fromisoformat(t).replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if dt < start or dt > end:
            continue
        if best_dt is None or abs((dt - target).total_seconds()) < abs((best_dt - target).total_seconds()):
            best_dt = dt
            best_i = i

    if best_i is None:
        return None

    def get(arr_name: str):
        arr = hourly.get(arr_name)
        if arr is None or best_i >= len(arr):
            return None
        return arr[best_i]

    return {
        "temp_f": c_to_f(get("temperature_2m")),
        "humidity": get("relative_humidity_2m"),
        "precip_in": mm_to_in(get("precipitation")),
        "pressure_mb": get("surface_pressure"),
        "wind_mph": kmh_to_mph(get("wind_speed_10m")),
        "wind_dir_deg": get("wind_direction_10m"),
        "raw": j,
    }


def c_to_f(c: Optional[float]) -> Optional[float]:
    if c is None:
        return None
    return float(c) * 9.0 / 5.0 + 32.0


def kmh_to_mph(kmh: Optional[float]) -> Optional[float]:
    if kmh is None:
        return None
    return float(kmh) * 0.621371


def mm_to_in(mm: Optional[float]) -> Optional[float]:
    if mm is None:
        return None
    return float(mm) / 25.4


# -----------------------------
# Backfill game_weather
# -----------------------------
def fetch_games_in_range(conn, start: date, end: date) -> List[Tuple[int, date, Optional[int]]]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT game_id, game_date, venue_id
            FROM games
            WHERE game_date BETWEEN %s AND %s
              AND game_type IN ('R','P')
              AND venue_id IS NOT NULL
            ORDER BY game_date, game_id;
        """, (start, end))
        rows = [(int(gid), gd, (int(vid) if vid is not None else None)) for gid, gd, vid in cur.fetchall()]
    return rows


def load_venue_coords(conn) -> Dict[int, Tuple[Optional[float], Optional[float]]]:
    with conn.cursor() as cur:
        cur.execute("SELECT venue_id, lat, lon FROM venues;")
        return {int(vid): (lat, lon) for vid, lat, lon in cur.fetchall()}


def backfill_weather_open_meteo(conn, start: date, end: date, sleep_s: float = 0.1, limit_games: int = 0) -> None:
    print(f"[weather] Backfill open-meteo from {start} to {end}")

    # Ensure venues have coords
    populate_missing_venue_coords(conn)

    venue_xy = load_venue_coords(conn)
    commence_map = build_game_commence_time_map(conn, start, end)

    games = fetch_games_in_range(conn, start, end)
    if limit_games and limit_games > 0:
        games = games[:limit_games]

    sess = requests.Session()

    upserts = []
    done = 0
    skipped = 0

    for (gid, gdate, venue_id) in games:
        ct = commence_map.get(gid)
        if ct is None:
            skipped += 1
            continue
        lat, lon = venue_xy.get(int(venue_id), (None, None))
        if lat is None or lon is None:
            skipped += 1
            continue

        try:
            w = pick_closest_hour_from_open_meteo(float(lat), float(lon), ct, sess)
        except Exception as e:
            print(f"[weather] gid={gid} failed: {e}")
            w = None

        if not w:
            skipped += 1
            continue

        upserts.append((
            gid,
            gdate,
            venue_id,
            ct,
            w.get("temp_f"),
            w.get("wind_mph"),
            w.get("wind_dir_deg"),
            w.get("humidity"),
            w.get("pressure_mb"),
            w.get("precip_in"),
            "open-meteo",
            psycopg2.extras.Json(w.get("raw")),
        ))

        done += 1
        if done % 200 == 0:
            flush_game_weather_upserts(conn, upserts)
            upserts = []
            print(f"[weather] upserted {done} games... (skipped={skipped})")

        time.sleep(sleep_s)

    if upserts:
        flush_game_weather_upserts(conn, upserts)

    print(f"[weather] DONE. upserted={done}, skipped={skipped}")


def flush_game_weather_upserts(conn, rows: list) -> None:
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO game_weather (
              game_id, game_date, venue_id, commence_time,
              temp_f, wind_mph, wind_dir_deg, humidity, pressure_mb, precip_in,
              source, raw
            )
            VALUES %s
            ON CONFLICT (game_id) DO UPDATE SET
              game_date = EXCLUDED.game_date,
              venue_id = EXCLUDED.venue_id,
              commence_time = EXCLUDED.commence_time,
              temp_f = EXCLUDED.temp_f,
              wind_mph = EXCLUDED.wind_mph,
              wind_dir_deg = EXCLUDED.wind_dir_deg,
              humidity = EXCLUDED.humidity,
              pressure_mb = EXCLUDED.pressure_mb,
              precip_in = EXCLUDED.precip_in,
              source = EXCLUDED.source,
              raw = EXCLUDED.raw,
              pulled_at = now();
            """,
            rows,
            page_size=500
        )
    conn.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pg_dsn", default=os.environ.get("PG_DSN"))
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--sleep", type=float, default=0.1)
    ap.add_argument("--limit_games", type=int, default=0)
    args = ap.parse_args()

    start = parse_date(args.start)
    end = parse_date(args.end)

    if not args.pg_dsn:
        raise ValueError("PG_DSN not set")

    conn = psycopg2.connect(args.pg_dsn)

    backfill_weather_open_meteo(
        conn,
        start=start,
        end=end,
        sleep_s=args.sleep,
        limit_games=args.limit_games,
    )

    conn.close()


if __name__ == "__main__":
    main()
