#!/usr/bin/env python3
"""
EV-based moneyline backtest graded at DraftKings closing odds (includes vig).

- Model prob: p_home_win_poisson
- Home EV: p_home * b_home - (1-p_home)
- Away EV: p_away * b_away - (1-p_away)
- Bet the side with higher EV if EV >= threshold

Usage:
  export PG_DSN="postgresql+psycopg2://USER:PASS@HOST:5432/mlb_model"
  python backtest/backtest_moneyline_ev_dk.py --start 2024-01-01 --end 2024-12-31 --min_books 6
"""

import os
import argparse
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text


def profit_if_win_1u(odds: float) -> float:
    if odds is None or np.isnan(odds) or odds == 0:
        return np.nan
    if odds > 0:
        return odds / 100.0
    return 100.0 / abs(odds)


def load_df(engine, schema: str, start: str, end: str, sportsbook: str = "DraftKings") -> pd.DataFrame:
    q = text(f"""
      WITH book_close AS (
        SELECT DISTINCT ON (game_id)
          game_id,
          home_price AS home_price_close,
          away_price AS away_price_close
        FROM {schema}.odds_ml
        WHERE market='h2h'
          AND sportsbook = :sportsbook
          AND game_date BETWEEN :start AND :end
          AND game_id IS NOT NULL
          AND home_price IS NOT NULL
          AND away_price IS NOT NULL
          AND ((home_price <= -100) OR (home_price >= 100))
          AND ((away_price <= -100) OR (away_price >= 100))
        ORDER BY game_id, pulled_at DESC
      )
      SELECT
        f.game_id,
        f.game_date,
        f.p_home_win_poisson,
        f.p_home_median,
        f.n_books_ml,
        b.home_price_close,
        b.away_price_close,
        g.home_runs,
        g.away_runs
      FROM {schema}.features_game f
      JOIN {schema}.games g USING (game_id)
      JOIN book_close b USING (game_id)
      WHERE f.game_date BETWEEN :start AND :end
        AND f.p_home_win_poisson IS NOT NULL
        AND f.n_books_ml IS NOT NULL
    """)
    return pd.read_sql(q, engine, params={"sportsbook": sportsbook, "start": start, "end": end})


def run_sweep(df: pd.DataFrame, ev_thresholds: list[float]) -> pd.DataFrame:
    d = df.copy()
    d["home_win"] = (d["home_runs"] > d["away_runs"]).astype(int)

    d["p_home"] = d["p_home_win_poisson"].astype(float)
    d["p_away"] = 1.0 - d["p_home"]

    d["b_home"] = d["home_price_close"].astype(float).apply(profit_if_win_1u)
    d["b_away"] = d["away_price_close"].astype(float).apply(profit_if_win_1u)

    d["ev_home"] = d["p_home"] * d["b_home"] - (1.0 - d["p_home"])
    d["ev_away"] = d["p_away"] * d["b_away"] - (1.0 - d["p_away"])

    out = []
    for thr in ev_thresholds:
        bets = d.copy()

        # choose higher EV side if above threshold
        best_side = np.where(bets["ev_home"] >= bets["ev_away"], 1, -1)
        # keep best_ev as a Series aligned to the original index so we can
        # subset it consistently with the filtered bets below
        best_ev = pd.Series(
            np.where(best_side == 1, bets["ev_home"], bets["ev_away"]),
            index=bets.index,
        )

        bets = bets[best_ev >= thr].copy()
        if bets.empty:
            out.append({"ev_threshold": thr, "n_bets": 0, "roi": np.nan})
            continue

        bets["bet_side"] = np.where(bets["ev_home"] >= bets["ev_away"], 1, -1)
        bets["odds"] = np.where(bets["bet_side"] == 1, bets["home_price_close"], bets["away_price_close"]).astype(float)
        bets["profit_if_win"] = bets["odds"].apply(profit_if_win_1u)

        bets["bet_win"] = np.where(bets["bet_side"] == 1, bets["home_win"] == 1, bets["home_win"] == 0)
        bets["profit"] = np.where(bets["bet_win"], bets["profit_if_win"], -1.0)

        roi = bets["profit"].sum() / len(bets)

        out.append({
            "ev_threshold": thr,
            "n_bets": int(len(bets)),
            "roi": float(roi),
            "win_rate": float(bets["bet_win"].mean()),
            "avg_ev": float(best_ev.loc[bets.index].mean()),
            "avg_profit_if_win": float(bets["profit_if_win"].mean()),
            "avg_books": float(bets["n_books_ml"].mean()),
        })

    return pd.DataFrame(out).sort_values("ev_threshold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", default="public")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--min_books", type=int, default=6)
    ap.add_argument("--sportsbook", default="DraftKings")
    args = ap.parse_args()

    pg_dsn = os.getenv("PG_DSN")
    if not pg_dsn:
        raise RuntimeError("PG_DSN env var required")
    engine = create_engine(pg_dsn, pool_pre_ping=True)

    df = load_df(engine, args.schema, args.start, args.end, args.sportsbook)
    df = df[df["n_books_ml"] >= args.min_books].copy()
    print(f"Rows after n_books_ml>={args.min_books}: {len(df):,}")

    ev_thresholds = [0.00, 0.005, 0.01, 0.02, 0.03, 0.05]
    res = run_sweep(df, ev_thresholds)

    print("\nEV threshold sweep (DraftKings close grading):")
    print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()