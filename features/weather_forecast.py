#!/usr/bin/env python3
"""
weather_forecast.py

Fetches day-of weather forecasts for MLB games using the Open-Meteo API
(free, no API key required). Updates features_game with forecast values
before inference runs.

Columns written to features_game:
  forecast_temp_f        — temperature at first pitch (°F)
  forecast_wind_mph      — wind speed at first pitch (mph)
  forecast_wind_dir_deg  — wind direction at first pitch (degrees)
  forecast_precip_in     — precipitation probability (0-1)
  forecast_humidity      — relative humidity (0-100)

These are used INSTEAD of game_weather actuals at inference time,
since actuals don't exist yet for today's games.

Usage:
    PG_DSN=... python weather_forecast.py --date 2026-03-25
    PG_DSN=... python weather_forecast.py  # defaults to today
"""

import os
import argparse
import requests
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------------
# Venue coordinates — keyed by venue_id from games table
# ---------------------------------------------------------------------------

VENUE_COORDS = {
    # MLB ballparks (lat, lon, timezone)
    1:    (33.8003, -117.8827, "America/Los_Angeles"),   # Angel Stadium
    2:    (39.2838, -76.6218,  "America/New_York"),      # Camden Yards
    3:    (42.3467, -71.0972,  "America/New_York"),      # Fenway Park
    4:    (41.8299, -87.6338,  "America/Chicago"),       # Guaranteed Rate Field
    5:    (41.4962, -81.6852,  "America/New_York"),      # Progressive Field
    7:    (39.0517, -94.4803,  "America/Chicago"),       # Kauffman Stadium
    10:   (37.7516, -122.2005, "America/Los_Angeles"),   # Oakland Coliseum
    12:   (27.7682, -82.6534,  "America/New_York"),      # Tropicana Field
    14:   (43.6414, -79.3894,  "America/New_York"),      # Rogers Centre
    15:   (33.4453, -112.0667, "America/Phoenix"),       # Chase Field
    17:   (41.9484, -87.6553,  "America/Chicago"),       # Wrigley Field
    19:   (39.7559, -104.9942, "America/Denver"),        # Coors Field
    22:   (34.0739, -118.2400, "America/Los_Angeles"),   # Dodger Stadium
    31:   (40.4469, -80.0058,  "America/New_York"),      # PNC Park
    32:   (43.0280, -87.9712,  "America/Chicago"),       # American Family Field
    109:  (33.4453, -112.0667, "America/Phoenix"),       # Chase Field (ARI)
    110:  (39.2838, -76.6218,  "America/New_York"),      # Camden Yards (BAL)
    111:  (42.3467, -71.0972,  "America/New_York"),      # Fenway (BOS)
    112:  (41.9484, -87.6553,  "America/Chicago"),       # Wrigley (CHC)
    113:  (39.0973, -84.5086,  "America/New_York"),      # Great American Ball Park
    114:  (41.4962, -81.6852,  "America/New_York"),      # Progressive Field
    115:  (39.7559, -104.9942, "America/Denver"),        # Coors Field
    116:  (42.3390, -83.0485,  "America/New_York"),      # Comerica Park
    117:  (29.7572, -95.3556,  "America/Chicago"),       # Minute Maid Park
    118:  (39.0517, -94.4803,  "America/Chicago"),       # Kauffman Stadium
    119:  (34.0739, -118.2400, "America/Los_Angeles"),   # Dodger Stadium
    120:  (38.8730, -77.0074,  "America/New_York"),      # Nationals Park
    121:  (40.7571, -73.8458,  "America/New_York"),      # Citi Field
    133:  (37.7516, -122.2005, "America/Los_Angeles"),   # Oakland Coliseum
    134:  (40.4469, -80.0058,  "America/New_York"),      # PNC Park
    135:  (32.7073, -117.1566, "America/Los_Angeles"),   # Petco Park
    136:  (37.7786, -122.3893, "America/Los_Angeles"),   # Oracle Park
    137:  (37.7786, -122.3893, "America/Los_Angeles"),   # Oracle Park (SF Giants)
    138:  (38.6226, -90.1928,  "America/Chicago"),       # Busch Stadium
    139:  (27.7682, -82.6534,  "America/New_York"),      # Tropicana Field
    140:  (32.7512, -97.0832,  "America/Chicago"),       # Globe Life Field
    141:  (43.6414, -79.3894,  "America/New_York"),      # Rogers Centre
    142:  (44.9817, -93.2775,  "America/Chicago"),       # Target Field
    143:  (39.9056, -75.1665,  "America/New_York"),      # Citizens Bank Park
    144:  (33.7350, -84.3900,  "America/New_York"),      # Truist Park
    145:  (41.8299, -87.6338,  "America/Chicago"),       # Guaranteed Rate
    146:  (25.7781, -80.2197,  "America/New_York"),      # loanDepot park
    147:  (40.8296, -73.9262,  "America/New_York"),      # Yankee Stadium
    158:  (43.0280, -87.9712,  "America/Chicago"),       # American Family Field
    # Additional venue IDs
    2392: (29.7572, -95.3556,  "America/Chicago"),       # Minute Maid Park
    2394: (42.3390, -83.0485,  "America/New_York"),      # Comerica Park
    2395: (37.7786, -122.3893, "America/Los_Angeles"),   # Oracle Park
    2536: (28.0836, -82.5292,  "America/New_York"),      # TD Ballpark
    2602: (39.0973, -84.5086,  "America/New_York"),      # Great American Ball Park
    2680: (32.7073, -117.1566, "America/Los_Angeles"),   # Petco Park
    2681: (39.9056, -75.1665,  "America/New_York"),      # Citizens Bank Park
    2735: (41.0534, -76.0936,  "America/New_York"),      # Muncy Bank Ballpark
    2889: (38.6226, -90.1928,  "America/Chicago"),       # Busch Stadium
    3289: (40.7571, -73.8458,  "America/New_York"),      # Citi Field
    3309: (38.8730, -77.0074,  "America/New_York"),      # Nationals Park
    3312: (44.9817, -93.2775,  "America/Chicago"),       # Target Field
    3313: (40.8296, -73.9262,  "America/New_York"),      # Yankee Stadium
    4169: (25.7781, -80.2197,  "America/New_York"),      # loanDepot park
    4705: (33.7350, -84.3900,  "America/New_York"),      # Truist Park
    5325: (32.7512, -97.0832,  "America/Chicago"),       # Globe Life Field
    680:  (47.5914, -122.3328, "America/Los_Angeles"),   # T-Mobile Park
}

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def celsius_to_fahrenheit(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def ms_to_mph(ms: float) -> float:
    return ms * 2.23694


def fetch_forecast(lat: float, lon: float, date_str: str,
                   game_hour_utc: int = 20) -> dict:
    params = {
        "latitude":        lat,
        "longitude":       lon,
        "hourly":          "temperature_2m,relative_humidity_2m,precipitation_probability,windspeed_10m,winddirection_10m",
        "temperature_unit":"celsius",
        "windspeed_unit":  "ms",
        "timezone":        "UTC",
        "forecast_days":   7,   # fetch 7 days, find the right one
    }
    try:
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"    Weather API error for ({lat}, {lon}): {e}")
        return {}

    hourly = data.get("hourly", {})
    times  = hourly.get("time", [])

    # Find the target date and hour
    target = f"{date_str}T{game_hour_utc:02d}:00"
    if target in times:
        idx = times.index(target)
    else:
        # Find closest time on the target date
        date_times = [t for t in times if t.startswith(date_str)]
        if not date_times:
            return {}
        target_idx = times.index(date_times[min(game_hour_utc, len(date_times)-1)])
        idx = target_idx

    try:
        temp_c   = hourly["temperature_2m"][idx]
        humidity = hourly["relative_humidity_2m"][idx]
        precip   = hourly["precipitation_probability"][idx]
        wind_ms  = hourly["windspeed_10m"][idx]
        wind_dir = hourly["winddirection_10m"][idx]
    except (KeyError, IndexError):
        return {}

    return {
        "forecast_temp_f":       float(celsius_to_fahrenheit(temp_c)) if temp_c is not None else None,
        "forecast_wind_mph":     float(ms_to_mph(wind_ms)) if wind_ms is not None else None,
        "forecast_wind_dir_deg": float(wind_dir) if wind_dir is not None else None,
        "forecast_precip_in":    float(precip) / 100.0 if precip is not None else None,
        "forecast_humidity":     float(humidity) if humidity is not None else None,
    }

def ensure_forecast_columns(engine, schema: str) -> None:
    """Add forecast columns to features_game if they don't exist."""
    cols = [
        "forecast_temp_f",
        "forecast_wind_mph",
        "forecast_wind_dir_deg",
        "forecast_precip_in",
        "forecast_humidity",
    ]
    with engine.begin() as conn:
        existing = pd.read_sql(
            text("SELECT column_name FROM information_schema.columns "
                 "WHERE table_schema = :s AND table_name = 'features_game'"),
            conn, params={"s": schema}
        )["column_name"].tolist()

        for col in cols:
            if col not in existing:
                conn.execute(text(
                    f"ALTER TABLE {schema}.features_game "
                    f"ADD COLUMN {col} DOUBLE PRECISION"
                ))
                print(f"  Added column: {col}")


def upsert_weather_forecasts(engine, schema: str, date_str: str) -> None:
    """
    For every game on date_str, fetch a weather forecast and upsert into features_game.
    Uses Open-Meteo (free, no API key).
    """
    games = pd.read_sql(text(f"""
        SELECT g.game_id, g.venue_id, g.home_team_id
        FROM {schema}.games g
        WHERE g.game_date = :d
          AND g.venue_id IS NOT NULL
    """), engine, params={"d": date_str})

    if games.empty:
        print(f"  No games found for {date_str}")
        return

    rows = []
    for _, g in games.iterrows():
        vid = int(g["venue_id"])
        coords = VENUE_COORDS.get(vid)

        if coords is None:
            # Try home_team_id as fallback
            coords = VENUE_COORDS.get(int(g["home_team_id"]))

        if coords is None:
            print(f"  No coordinates for venue_id={vid} — skipping weather")
            rows.append({"game_id": int(g["game_id"]),
                         "forecast_temp_f": None, "forecast_wind_mph": None,
                         "forecast_wind_dir_deg": None, "forecast_precip_in": None,
                         "forecast_humidity": None})
            continue

        lat, lon, tz = coords
        forecast = fetch_forecast(lat, lon, date_str)

        if not forecast:
            rows.append({"game_id": int(g["game_id"]),
                         "forecast_temp_f": None, "forecast_wind_mph": None,
                         "forecast_wind_dir_deg": None, "forecast_precip_in": None,
                         "forecast_humidity": None})
        else:
            rows.append({"game_id": int(g["game_id"]), **forecast})

        print(f"  game_id={g['game_id']} venue={vid}: "
              f"temp={forecast.get('forecast_temp_f', 'N/A'):.1f}°F "
              f"wind={forecast.get('forecast_wind_mph', 'N/A'):.1f}mph "
              f"dir={forecast.get('forecast_wind_dir_deg', 'N/A'):.0f}°"
              if forecast else f"  game_id={g['game_id']} venue={vid}: no forecast")

    if not rows:
        return

    df_out = pd.DataFrame(rows)
    # Ensure all forecast columns are float64, not object — prevents text type mismatch
    for col in ["forecast_temp_f", "forecast_wind_mph", "forecast_wind_dir_deg",
                "forecast_precip_in", "forecast_humidity"]:
        if col in df_out.columns:
            df_out[col] = pd.to_numeric(df_out[col], errors="coerce")
    cols   = [c for c in df_out.columns if c != "game_id"]
    set_clause = ", ".join(f"{c} = s.{c}" for c in cols)

    with engine.begin() as conn:
        df_out.to_sql("_weather_tmp", conn, schema=schema,
                      if_exists="replace", index=False, method="multi")
        conn.execute(text(f"""
            UPDATE {schema}.features_game AS t
            SET {set_clause}
            FROM {schema}._weather_tmp AS s
            WHERE t.game_id = s.game_id
        """))
        conn.execute(text(f"DROP TABLE IF EXISTS {schema}._weather_tmp"))

    print(f"  Weather forecasts upserted for {len(rows)} games.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date",   default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    ap.add_argument("--schema", default="public")
    args = ap.parse_args()

    pg_dsn = os.getenv("PG_DSN")
    if not pg_dsn:
        raise RuntimeError("PG_DSN env var required.")

    engine = create_engine(pg_dsn, pool_pre_ping=True)
    ensure_forecast_columns(engine, args.schema)
    upsert_weather_forecasts(engine, args.schema, args.date)