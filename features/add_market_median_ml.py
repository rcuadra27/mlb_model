#!/usr/bin/env python3
"""
features/add_market_median_ml.py (fixed)

- Uses latest pulled_at per (game_id, sportsbook) as "closing"
- Filters invalid odds
- Computes vig-free fair probability per book
- Stores:
  p_home_median
  n_books_ml
  home_price_close_consensus
  away_price_close_consensus

Usage:
  export PG_DSN="postgresql+psycopg2://USER:PASS@HOST:5432/mlb_model"
  python features/add_market_median_ml.py --market h2h --start 2020-01-01 --end 2025-12-31
"""

import os
import argparse
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text


def ensure_columns(engine, schema: str):
    cols = {
        "p_home_median": "DOUBLE PRECISION",
        "n_books_ml": "INTEGER",
        "home_price_close_consensus": "INTEGER",
        "away_price_close_consensus": "INTEGER",
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


def american_to_implied(o: np.ndarray) -> np.ndarray:
    o = o.astype(float)
    p = np.full_like(o, np.nan, dtype=float)
    neg = o < 0
    pos = o > 0
    p[neg] = (-o[neg]) / ((-o[neg]) + 100.0)
    p[pos] = 100.0 / (o[pos] + 100.0)
    return p


def prob_to_american(p: np.ndarray) -> np.ndarray:
    """
    Convert probability to American odds (vig-free synthetic).
    p in (0,1)
    if p >= 0.5: odds = -100 * p/(1-p)
    else:        odds =  100 * (1-p)/p
    """
    p = np.clip(p.astype(float), 1e-6, 1 - 1e-6)
    odds = np.empty_like(p)
    fav = p >= 0.5
    odds[fav] = -100.0 * (p[fav] / (1.0 - p[fav]))
    odds[~fav] = 100.0 * ((1.0 - p[~fav]) / p[~fav])
    return np.rint(odds).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", default="public")
    ap.add_argument("--market", default="h2h")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--batch", type=int, default=50000)
    ap.add_argument("--max_abs_odds", type=int, default=2000)
    args = ap.parse_args()

    pg_dsn = os.getenv("PG_DSN")
    if not pg_dsn:
        raise RuntimeError("PG_DSN env var required")

    engine = create_engine(pg_dsn, pool_pre_ping=True)
    ensure_columns(engine, args.schema)

    df = pd.read_sql(
        text(f"""
            SELECT DISTINCT ON (game_id, sportsbook)
              game_id,
              sportsbook,
              home_price,
              away_price,
              pulled_at
            FROM {args.schema}.odds_ml
            WHERE market = :market
              AND game_date BETWEEN :start AND :end
              AND game_id IS NOT NULL
              AND home_price IS NOT NULL
              AND away_price IS NOT NULL
            ORDER BY game_id, sportsbook, pulled_at DESC
        """),
        engine,
        params={"market": args.market, "start": args.start, "end": args.end},
    )
    if df.empty:
        raise RuntimeError("No odds rows found. Check market/date range.")

    # ---- Clean odds ----
    hp = df["home_price"].astype(int).to_numpy()
    ap_ = df["away_price"].astype(int).to_numpy()

    def valid_odds(x):
        # must be <= -100 or >= +100, and within max_abs_odds
        return ((x <= -100) | (x >= 100)) & (np.abs(x) <= args.max_abs_odds)

    ok = valid_odds(hp) & valid_odds(ap_)
    df = df.loc[ok].copy()
    if df.empty:
        raise RuntimeError("All odds filtered out as invalid. Check ingestion.")

    hp = df["home_price"].astype(int).to_numpy()
    ap_ = df["away_price"].astype(int).to_numpy()

    p_home_imp = american_to_implied(hp)
    p_away_imp = american_to_implied(ap_)
    denom = p_home_imp + p_away_imp
    df["p_home_fair"] = p_home_imp / denom

    agg = df.groupby("game_id").agg(
        p_home_median=("p_home_fair", "median"),
        n_books_ml=("p_home_fair", "count"),
    ).reset_index()

    # synthetic consensus vig-free odds for grading / display
    home_odds = prob_to_american(agg["p_home_median"].to_numpy())
    away_odds = prob_to_american((1.0 - agg["p_home_median"]).to_numpy())
    agg["home_price_close_consensus"] = home_odds
    agg["away_price_close_consensus"] = away_odds

    upd = text(f"""
        UPDATE {args.schema}.features_game
        SET
          p_home_median = :p_home_median,
          n_books_ml = :n_books_ml,
          home_price_close_consensus = :home_price_close_consensus,
          away_price_close_consensus = :away_price_close_consensus
        WHERE game_id = :game_id
    """)

    payload = agg.to_dict(orient="records")
    print(f"Updating {len(payload):,} games with cleaned median market prob...")

    with engine.begin() as conn:
        conn.execute(text("SET LOCAL statement_timeout = '0'"))
        for i in range(0, len(payload), args.batch):
            conn.execute(upd, payload[i:i+args.batch])
            print(f"  {min(i+args.batch, len(payload)):,}/{len(payload):,}")

    with engine.connect() as conn:
        r = conn.execute(text(f"""
            SELECT
              COUNT(*) AS n_games,
              COUNT(p_home_median) AS n_with_market,
              AVG(p_home_median) AS avg_market_p,
              AVG(n_books_ml) AS avg_books,
              MIN(n_books_ml) AS min_books,
              MAX(n_books_ml) AS max_books,
              MIN(home_price_close_consensus) AS min_home_consensus,
              MAX(home_price_close_consensus) AS max_home_consensus
            FROM {args.schema}.features_game
            WHERE game_date BETWEEN :start AND :end
        """), {"start": args.start, "end": args.end}).fetchone()
    print("Sanity:", dict(r._mapping))


if __name__ == "__main__":
    main()
