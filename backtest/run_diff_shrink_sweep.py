#!/usr/bin/env python3
"""
Sweep RUN_DIFF_SHRINK over a date range and compare:
  - average edge magnitude
  - average model home probability
  - calibration by probability bucket (ECE-like summary)
  - EV-based moneyline backtest ROI + win rate (using closing odds)

This script assumes `inference/inference.py` stores predictions in:
  public.inference_game_predictions
and now includes `run_diff_shrink`.
"""

import argparse
import os
import subprocess
import sys
from datetime import timedelta

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text


def profit_if_win_1u(odds: float) -> float:
    if odds is None or np.isnan(odds) or odds == 0:
        return np.nan
    if odds > 0:
        return float(odds) / 100.0
    return 100.0 / abs(float(odds))


def daterange_inclusive(start: str, end: str) -> list[str]:
    s = pd.to_datetime(start).date()
    e = pd.to_datetime(end).date()
    days: list[str] = []
    cur = s
    while cur <= e:
        days.append(cur.isoformat())
        cur = cur + timedelta(days=1)
    return days


def run_inference_for_range(
    *,
    repo_root: str,
    schema: str,
    start: str,
    end: str,
    shrink: float,
    team_model: str,
    team_features: str,
    market: str,
    min_books_infer: int,
    top_k: int,
    calibration_model: str,
) -> None:
    for day in daterange_inclusive(start, end):
        cmd = [
            sys.executable,
            os.path.join(repo_root, "inference", "inference.py"),
            "--schema",
            schema,
            "--date",
            day,
            "--market",
            market,
            "--min_books",
            str(min_books_infer),
            "--top_k",
            str(top_k),
            "--team_model",
            team_model,
            "--team_features",
            team_features,
            "--run_diff_shrink",
            str(shrink),
            "--calibration_model",
            calibration_model,
        ]
        proc = subprocess.run(cmd, cwd=repo_root)
        if proc.returncode != 0:
            raise RuntimeError(f"inference failed for day={day}, run_diff_shrink={shrink}, exit_code={proc.returncode}")


def load_latest_predictions_for_shrink(
    *,
    engine,
    schema: str,
    start: str,
    end: str,
    shrink: float,
    sportsbook: str,
    min_books: int,
) -> pd.DataFrame:
    # latest inference row per game_id for this shrink
    q = text(
        f"""
        WITH infer_latest AS (
          SELECT DISTINCT ON (game_id)
            game_id,
            game_date,
            home_team,
            away_team,
            run_diff_shrink,
            p_home_win_poisson,
            p_home_market_median,
            edge_home,
            n_books
          FROM {schema}.inference_game_predictions
          WHERE game_date BETWEEN :start AND :end
            AND ROUND(run_diff_shrink::numeric, 2) = ROUND(CAST(:shrink AS numeric), 2)
            AND p_home_win_poisson IS NOT NULL
          ORDER BY game_id, as_of_ts DESC
        ),
        book_close AS (
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
          i.game_id,
          i.game_date,
          i.home_team,
          i.away_team,
          i.run_diff_shrink,
          i.home_runs_pred,
          i.away_runs_pred,
          i.p_home_win_poisson,
          i.p_home_market_median,
          i.edge_home,
          i.n_books,
          b.home_price_close,
          b.away_price_close,
          g.home_runs,
          g.away_runs
        FROM infer_latest i
        JOIN book_close b USING (game_id)
        JOIN {schema}.games g USING (game_id)
        WHERE i.n_books >= :min_books
        """
    )
    return pd.read_sql(q, engine, params={"start": start, "end": end, "shrink": float(shrink), "sportsbook": sportsbook, "min_books": min_books})


def calibration_by_bucket(df: pd.DataFrame, bucket_count: int = 10) -> tuple[pd.DataFrame, float]:
    d = df.copy()
    d["home_win"] = (d["home_runs"] > d["away_runs"]).astype(int)

    # fixed-width buckets for readability: [0,0.1), ... , [0.9,1.0]
    bins = np.linspace(0.0, 1.0, bucket_count + 1)
    d["bucket"] = pd.cut(d["p_home_win_poisson"], bins=bins, include_lowest=True, right=True)

    grp = d.groupby("bucket", observed=True).agg(
        n=("bucket", "size"),
        mean_pred=("p_home_win_poisson", "mean"),
        win_rate=("home_win", "mean"),
    )
    grp = grp[grp["n"] > 0].copy()

    # ECE-like: weighted average absolute difference between mean_pred and win_rate
    total = grp["n"].sum()
    ece = float(((grp["n"] / total) * (grp["mean_pred"] - grp["win_rate"]).abs()).sum())

    grp = grp.sort_values("mean_pred")
    return grp.reset_index(drop=False), ece


def ev_backtest_roi(
    df: pd.DataFrame,
    *,
    ev_threshold: float,
    bet_threshold: float,
) -> dict:
    d = df.copy()
    d["home_win"] = (d["home_runs"] > d["away_runs"]).astype(int)

    p_home = d["p_home_win_poisson"].astype(float).to_numpy()
    p_away = 1.0 - p_home

    b_home = d["home_price_close"].astype(float).apply(profit_if_win_1u).to_numpy()
    b_away = d["away_price_close"].astype(float).apply(profit_if_win_1u).to_numpy()

    ev_home = p_home * b_home - (1.0 - p_home)
    ev_away = p_away * b_away - (1.0 - p_away)

    # Edge-based side selection (matches inference's best_edge/best_side semantics)
    edge_home = d["edge_home"].astype(float).to_numpy()
    best_side = np.where(edge_home >= 0, 1, -1)  # HOME if edge_home >= 0 else AWAY
    best_ev = np.where(best_side == 1, ev_home, ev_away)

    # Extra bet filters (from your screenshot)
    # 3) Minimum probability gap: abs(p_model - 0.5) > 0.05
    PROB_GAP_MIN = 0.05
    p_market_home = d["p_home_market_median"].astype(float).to_numpy()
    p_market_away = 1.0 - p_market_home

    p_model_side = np.where(best_side == 1, p_home, p_away)
    p_market_side = np.where(best_side == 1, p_market_home, p_market_away)

    min_confident = np.abs(p_model_side - 0.5) > PROB_GAP_MIN
    disagreement = np.abs(p_model_side - p_market_side) > float(bet_threshold)

    take = min_confident & disagreement & (best_ev >= ev_threshold)
    d = d.loc[take].copy()
    if d.empty:
        return {"n_bets": 0, "roi": np.nan, "win_rate": np.nan, "avg_ev": np.nan}

    # recompute arrays on taken subset
    idx = d.index
    best_side_taken = best_side[idx]
    profit_if_win = np.where(best_side_taken == 1, b_home[idx], b_away[idx])

    bet_win = np.where(best_side_taken == 1, d["home_win"].to_numpy() == 1, d["home_win"].to_numpy() == 0)
    profit = np.where(bet_win, profit_if_win, -1.0)

    roi = float(profit.sum() / len(profit))
    win_rate = float(bet_win.mean())
    avg_ev = float(best_ev[idx].mean())
    return {"n_bets": int(len(d)), "roi": roi, "win_rate": win_rate, "avg_ev": avg_ev}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", default="public")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--sportsbook", default="DraftKings")
    ap.add_argument("--min_books", type=int, default=5)
    ap.add_argument("--min_books_infer", type=int, default=5)
    ap.add_argument("--market", default="h2h")
    ap.add_argument("--top_k", type=int, default=0, help="Only for inference display; safe to keep 0 during sweeps.")
    ap.add_argument("--skip_inference", action="store_true", help="Do not call inference; only load existing predictions.")

    ap.add_argument("--ev_threshold", type=float, default=0.01, help="Place bet when best-side EV >= this threshold.")
    ap.add_argument(
        "--bet_threshold",
        type=float,
        default=0.03,
        help="Disagreement threshold X: bet only if abs(p_model_side - p_market_side) > this value.",
    )
    ap.add_argument("--bucket_count", type=int, default=10)

    ap.add_argument(
        "--shrink_values",
        default="0.35,0.50,0.65,0.80",
        help="Comma-separated list of RUN_DIFF_SHRINK values to test.",
    )
    ap.add_argument("--team_model", default="artifacts/runs_model_team_lgbm_optionA.joblib")
    ap.add_argument("--team_features", default="artifacts/runs_model_team_features_optionA.txt")
    ap.add_argument(
        "--calibration_model",
        default="artifacts/isotonic_home_win_poisson.joblib",
        help="Calibration model path passed through to inference.",
    )

    args = ap.parse_args()

    pg_dsn = os.getenv("PG_DSN")
    if not pg_dsn:
        raise RuntimeError("PG_DSN env var required")
    if any(x in pg_dsn for x in ["HOST", "USER", "PASS"]):
        raise RuntimeError(
            "PG_DSN appears to contain placeholder values (HOST/USER/PASS). "
            "Set PG_DSN to a real connection string before running the sweep."
        )
    engine = create_engine(pg_dsn, pool_pre_ping=True)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    team_model = args.team_model if os.path.isabs(args.team_model) else os.path.join(repo_root, args.team_model)
    team_features = args.team_features if os.path.isabs(args.team_features) else os.path.join(repo_root, args.team_features)

    shrink_values = [float(x.strip()) for x in args.shrink_values.split(",") if x.strip()]

    rows = []
    for shrink in shrink_values:
        print(f"\n=== RUN_DIFF_SHRINK={shrink:.2f} ===")

        if not args.skip_inference:
            run_inference_for_range(
                repo_root=repo_root,
                schema=args.schema,
                start=args.start,
                end=args.end,
                shrink=shrink,
                team_model=team_model,
                team_features=team_features,
                market=args.market,
                min_books_infer=args.min_books_infer,
                top_k=args.top_k,
                calibration_model=args.calibration_model,
            )

        df = load_latest_predictions_for_shrink(
            engine=engine,
            schema=args.schema,
            start=args.start,
            end=args.end,
            shrink=shrink,
            sportsbook=args.sportsbook,
            min_books=args.min_books,
        )
        if df.empty:
            print("No rows returned for this shrink (check min_books / date / odds availability).")
            rows.append(
                {
                    "shrink": shrink,
                    "n_games": 0,
                    "avg_abs_edge": np.nan,
                    "avg_p_home": np.nan,
                    "ece": np.nan,
                    "roi": np.nan,
                    "win_rate": np.nan,
                    "n_bets": 0,
                }
            )
            continue

        # Data-range sanity checks (critical diagnostics).
        n_dates = int(df["game_date"].nunique())
        min_date = df["game_date"].min()
        max_date = df["game_date"].max()
        n_games = int(len(df))
        print("\nData range checks:")
        print("unique game_date:", n_dates)
        print("min/max game_date:", min_date, max_date)
        print("n_games:", n_games)
        print("n_dates >= 20:", n_dates >= 20)
        print("\nGames per game_date:")
        print(df["game_date"].value_counts().sort_index().to_string())

        # Daily adjusted-runs diagnostic (uses same shrink transform as inference).
        run_diff_shrink = float(shrink)
        mid = (df["home_runs_pred"].astype(float) + df["away_runs_pred"].astype(float)) / 2.0
        diff = df["home_runs_pred"].astype(float) - df["away_runs_pred"].astype(float)
        diff_shrunk = diff * run_diff_shrink
        df["home_runs_pred_adj"] = (mid + diff_shrunk / 2.0).clip(lower=0.1)
        df["away_runs_pred_adj"] = (mid - diff_shrunk / 2.0).clip(lower=0.1)

        daily = df.groupby("game_date").agg({
            "home_runs_pred_adj": "mean",
            "away_runs_pred_adj": "mean",
        })
        daily["diff"] = daily["home_runs_pred_adj"] - daily["away_runs_pred_adj"]
        print("\nDaily largest home-away adjusted diffs:")
        print(daily.sort_values("diff", ascending=False).head(10).to_string())
        print("\nDaily most negative home-away adjusted diffs:")
        print(daily.sort_values("diff").head(10).to_string())
        print("\nOverall mean daily diff:", float(daily["diff"].mean()))

        df["home_win"] = (df["home_runs"] > df["away_runs"]).astype(int)
        avg_abs_edge = float(df["edge_home"].abs().mean())
        avg_p_home = float(df["p_home_win_poisson"].mean())

        calib, ece = calibration_by_bucket(df, bucket_count=args.bucket_count)
        bt = ev_backtest_roi(df, ev_threshold=args.ev_threshold, bet_threshold=args.bet_threshold)

        rows.append(
            {
                "shrink": shrink,
                "n_games": int(len(df)),
                "avg_abs_edge": avg_abs_edge,
                "avg_p_home": avg_p_home,
                "ece": ece,
                "roi": bt["roi"],
                "win_rate": bt["win_rate"],
                "n_bets": bt["n_bets"],
                "avg_ev": bt["avg_ev"],
                "bet_threshold": args.bet_threshold,
            }
        )

        print("Avg calibration by bucket (only non-empty):")
        # keep it compact
        print(calib.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"Backtest: n_bets={bt['n_bets']}, win_rate={bt['win_rate']}, roi={bt['roi']}, avg_ev={bt['avg_ev']}")

    res = pd.DataFrame(rows).sort_values("roi", ascending=False)
    print("\n=== SUMMARY (sorted by ROI) ===")
    print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()

