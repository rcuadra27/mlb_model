#!/usr/bin/env python3
"""
features/add_poisson_winprob.py

Compute Poisson/Skellam-based win probability from predicted runs:
  home_runs ~ Poisson(lambda_home)
  away_runs ~ Poisson(lambda_away)

p_home_win_final = P(home > away) + 0.5*P(home = away)

Requires columns in features_game:
  - home_runs_pred
  - away_runs_pred

Writes:
  - p_home_win_poisson
  - p_tie_poisson (diagnostic)

Usage:
  export PG_DSN="postgresql+psycopg2://USER:PASS@HOST:5432/mlb_model"
  python features/add_poisson_winprob.py --start 2015-04-01 --end 2024-12-31 --batch 50000
"""

import os
import argparse
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from scipy.stats import skellam


def ensure_columns(engine, schema: str):
    cols = {
        "p_home_win_poisson": "DOUBLE PRECISION",
        "p_tie_poisson": "DOUBLE PRECISION",
    }
    with engine.begin() as conn:
        existing = conn.execute(text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = :schema AND table_name = 'features_game'
        """), {"schema": schema}).fetchall()
        existing = {r[0] for r in existing}
        for c, typ in cols.items():
            if c not in existing:
                conn.execute(text(f"ALTER TABLE {schema}.features_game ADD COLUMN {c} {typ};"))
                print(f"Added column {schema}.features_game.{c}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", default="public")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--batch", type=int, default=50000)
    args = ap.parse_args()

    pg_dsn = os.getenv("PG_DSN")
    if not pg_dsn:
        raise RuntimeError("PG_DSN env var required")

    engine = create_engine(pg_dsn, pool_pre_ping=True)
    ensure_columns(engine, args.schema)

    df = pd.read_sql(
        text(f"""
            SELECT game_id, home_runs_pred, away_runs_pred
            FROM {args.schema}.features_game
            WHERE game_date BETWEEN :start AND :end
            ORDER BY game_id
        """),
        engine,
        params={"start": args.start, "end": args.end},
    )

    if df.empty:
        print("No games found.")
        return

    # guardrails: lambdas must be positive
    lam_h = np.clip(df["home_runs_pred"].astype(float).values, 0.05, 30.0)
    lam_a = np.clip(df["away_runs_pred"].astype(float).values, 0.05, 30.0)

    # Skellam on difference D = H - A
    p_tie = skellam.pmf(0, lam_h, lam_a)
    p_home_gt = 1.0 - skellam.cdf(0, lam_h, lam_a)  # P(D > 0)
    p_home = p_home_gt + 0.5 * p_tie

    out = pd.DataFrame({
        "game_id": df["game_id"].astype(int).values,
        "p_home_win_poisson": p_home.astype(float),
        "p_tie_poisson": p_tie.astype(float),
    })

    upd = text(f"""
        UPDATE {args.schema}.features_game
        SET
          p_home_win_poisson = :p_home_win_poisson,
          p_tie_poisson = :p_tie_poisson
        WHERE game_id = :game_id
    """)

    payload = out.to_dict(orient="records")
    print(f"Updating {len(payload):,} rows...")
    with engine.begin() as conn:
        conn.execute(text("SET LOCAL statement_timeout = '0'"))
        for i in range(0, len(payload), args.batch):
            conn.execute(upd, payload[i:i+args.batch])
            print(f"  {min(i+args.batch, len(payload)):,}/{len(payload):,}")

    print("Done. Sanity:")
    with engine.connect() as conn:
        r = conn.execute(text(f"""
            SELECT
              COUNT(*) n,
              AVG(p_home_win_poisson) avg_p,
              MIN(p_home_win_poisson) min_p,
              MAX(p_home_win_poisson) max_p,
              AVG(p_tie_poisson) avg_tie
            FROM {args.schema}.features_game
            WHERE game_date BETWEEN :start AND :end
              AND p_home_win_poisson IS NOT NULL
        """), {"start": args.start, "end": args.end}).fetchone()
    print(dict(r._mapping))


if __name__ == "__main__":
    main()
