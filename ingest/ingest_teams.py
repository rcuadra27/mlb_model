#!/usr/bin/env python3
import os
import argparse
import requests
import pandas as pd
from sqlalchemy import create_engine, text

TEAMS_URL = "https://statsapi.mlb.com/api/v1/teams?sportId=1"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", default="public")
    ap.add_argument("--table", default="teams")
    args = ap.parse_args()

    pg_dsn = os.getenv("PG_DSN")
    if not pg_dsn:
        raise RuntimeError("PG_DSN env var required")

    r = requests.get(TEAMS_URL, timeout=30)
    r.raise_for_status()
    data = r.json()

    rows = []
    for t in data.get("teams", []):
        rows.append({
            "mlb_team_id": int(t["id"]),
            "team_name": t.get("name") or t.get("teamName") or str(t["id"]),
            "abbreviation": t.get("abbreviation"),
            "name_full": t.get("name"),
        })

    df = pd.DataFrame(rows).drop_duplicates(subset=["mlb_team_id"])
    if df.empty:
        raise RuntimeError("No teams returned from StatsAPI")

    engine = create_engine(pg_dsn, pool_pre_ping=True)
    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {args.schema}.{args.table} (
              mlb_team_id BIGINT PRIMARY KEY,
              team_name TEXT NOT NULL,
              abbreviation TEXT,
              name_full TEXT,
              updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """))
        upsert = text(f"""
          INSERT INTO {args.schema}.{args.table} (mlb_team_id, team_name, abbreviation, name_full)
          VALUES (:mlb_team_id, :team_name, :abbreviation, :name_full)
          ON CONFLICT (mlb_team_id) DO UPDATE SET
            team_name = EXCLUDED.team_name,
            abbreviation = EXCLUDED.abbreviation,
            name_full = EXCLUDED.name_full,
            updated_at = now();
        """)
        conn.execute(upsert, df.to_dict(orient="records"))

    print(f"Upserted {len(df)} teams into {args.schema}.{args.table}")

if __name__ == "__main__":
    main()