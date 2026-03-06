#!/usr/bin/env python3
"""
Blend model probability with market probability.

p_blend = w * p_model + (1-w) * p_market

We optimize w on validation year (e.g., 2023) by minimizing logloss.
Then we apply to test year (e.g., 2024).
"""

import os
import argparse
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.metrics import log_loss


def load_df(engine, schema: str, start: str, end: str) -> pd.DataFrame:
    q = text(f"""
        SELECT
          f.game_id,
          f.game_date,
          f.p_home_win_poisson AS p_model,
          f.p_home_median AS p_market,
          g.home_runs,
          g.away_runs
        FROM {schema}.features_game f
        JOIN {schema}.games g USING (game_id)
        WHERE f.game_date BETWEEN :start AND :end
          AND f.p_home_win_poisson IS NOT NULL
          AND f.p_home_median IS NOT NULL
    """)
    return pd.read_sql(q, engine, params={"start": start, "end": end})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", default="public")
    ap.add_argument("--val_start", required=True)
    ap.add_argument("--val_end", required=True)
    ap.add_argument("--test_start", required=True)
    ap.add_argument("--test_end", required=True)
    args = ap.parse_args()

    pg_dsn = os.getenv("PG_DSN")
    engine = create_engine(pg_dsn, pool_pre_ping=True)

    # validation data
    val = load_df(engine, args.schema, args.val_start, args.val_end)
    val["y"] = (val["home_runs"] > val["away_runs"]).astype(int)

    # optimize w
    ws = np.linspace(0, 1, 101)
    best_w = None
    best_ll = 999

    for w in ws:
        p_blend = w * val["p_model"] + (1 - w) * val["p_market"]
        ll = log_loss(val["y"], p_blend)
        if ll < best_ll:
            best_ll = ll
            best_w = w

    print(f"Best w (validation): {best_w:.3f}")
    print(f"Validation logloss: {best_ll:.6f}")

    # test data
    test = load_df(engine, args.schema, args.test_start, args.test_end)
    test["y"] = (test["home_runs"] > test["away_runs"]).astype(int)

    test["p_blend"] = best_w * test["p_model"] + (1 - best_w) * test["p_market"]

    test_ll = log_loss(test["y"], test["p_blend"])
    print(f"Test logloss (blend): {test_ll:.6f}")

    # write back to DB
    with engine.begin() as conn:
        conn.execute(text("""
            ALTER TABLE public.features_game
            ADD COLUMN IF NOT EXISTS p_home_blend DOUBLE PRECISION;
        """))

        upd = text("""
            UPDATE public.features_game
            SET p_home_blend = :p_blend
            WHERE game_id = :game_id
        """)

        payload = test[["game_id", "p_blend"]].to_dict("records")
        conn.execute(upd, payload)

    print("Saved p_home_blend for test window.")


if __name__ == "__main__":
    main()