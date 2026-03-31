import argparse
import os

import numpy as np
import pandas as pd
import psycopg2
from sklearn.metrics import log_loss


def load_eval_df(conn, start=None, end=None, latest_only=False):
    where = ["g.home_runs IS NOT NULL", "g.away_runs IS NOT NULL"]
    params = []

    if start:
        where.append("p.game_date >= %s")
        params.append(start)
    if end:
        where.append("p.game_date <= %s")
        params.append(end)

    where_sql = "WHERE " + " AND ".join(where)

    if latest_only:
        sql = f"""
        WITH latest_preds AS (
            SELECT *
            FROM (
                SELECT
                    p.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY p.game_id
                        ORDER BY p.as_of_ts DESC
                    ) AS rn
                FROM inference_game_predictions p
            ) x
            WHERE rn = 1
        )
        SELECT
            p.as_of_ts,
            p.game_id,
            p.game_date,
            p.home_team,
            p.away_team,
            p.home_runs_pred,
            p.away_runs_pred,
            p.total_runs_pred,
            p.run_diff_pred,
            p.p_home_win_raw,
            p.p_home_win_poisson,
            g.home_runs AS home_runs_actual,
            g.away_runs AS away_runs_actual,
            g.home_win AS home_win_actual
        FROM latest_preds p
        JOIN games g
          ON p.game_id = g.game_id
        {where_sql}
        ORDER BY p.game_date, p.game_id;
        """
    else:
        sql = f"""
        SELECT
            p.as_of_ts,
            p.game_id,
            p.game_date,
            p.home_team,
            p.away_team,
            p.home_runs_pred,
            p.away_runs_pred,
            p.total_runs_pred,
            p.run_diff_pred,
            p.p_home_win_raw,
            p.p_home_win_poisson,
            g.home_runs AS home_runs_actual,
            g.away_runs AS away_runs_actual,
            g.home_win AS home_win_actual
        FROM inference_game_predictions p
        JOIN games g
          ON p.game_id = g.game_id
        {where_sql}
        ORDER BY p.game_date, p.game_id, p.as_of_ts DESC;
        """

    return pd.read_sql(sql, conn, params=params)


def compute_metrics(df: pd.DataFrame, prob_col: str):
    df = df.copy()

    # Use chosen probability column
    df["p_home_win_pred"] = df[prob_col]

    # Drop rows with missing prediction values
    needed = [
        "home_runs_pred",
        "away_runs_pred",
        "p_home_win_pred",
        "home_runs_actual",
        "away_runs_actual",
    ]
    df = df.dropna(subset=needed).copy()

    df["total_runs_actual"] = df["home_runs_actual"] + df["away_runs_actual"]
    df["home_win_actual_int"] = df["home_win_actual"].astype(int)
    df["predicted_home_win"] = (df["p_home_win_pred"] >= 0.5).astype(int)

    # Runs metrics
    rmse_home = np.sqrt(np.mean((df["home_runs_pred"] - df["home_runs_actual"]) ** 2))
    rmse_away = np.sqrt(np.mean((df["away_runs_pred"] - df["away_runs_actual"]) ** 2))
    rmse_total = np.sqrt(np.mean((df["total_runs_pred"] - df["total_runs_actual"]) ** 2))

    mae_home = np.mean(np.abs(df["home_runs_pred"] - df["home_runs_actual"]))
    mae_away = np.mean(np.abs(df["away_runs_pred"] - df["away_runs_actual"]))
    mae_total = np.mean(np.abs(df["total_runs_pred"] - df["total_runs_actual"]))

    bias_home = np.mean(df["home_runs_pred"] - df["home_runs_actual"])
    bias_away = np.mean(df["away_runs_pred"] - df["away_runs_actual"])
    bias_total = np.mean(df["total_runs_pred"] - df["total_runs_actual"])

    # Win metrics
    probs = df["p_home_win_pred"].clip(1e-6, 1 - 1e-6)
    logloss = log_loss(df["home_win_actual_int"], probs)
    brier = np.mean((probs - df["home_win_actual_int"]) ** 2)
    accuracy = np.mean(df["predicted_home_win"] == df["home_win_actual_int"])

    # Consistency check
    inconsistent = df[
        ((df["run_diff_pred"] > 0) & (df["p_home_win_pred"] < 0.5)) |
        ((df["run_diff_pred"] < 0) & (df["p_home_win_pred"] > 0.5))
    ].copy()

    inconsistency_rate = len(inconsistent) / len(df) if len(df) else np.nan

    # Calibration
    cal = df.copy()
    cal["prob_bucket"] = np.floor(cal["p_home_win_pred"] * 10) / 10
    calibration = (
        cal.groupby("prob_bucket", as_index=False)
        .agg(
            n_games=("game_id", "count"),
            avg_pred=("p_home_win_pred", "mean"),
            actual_home_win_rate=("home_win_actual_int", "mean"),
        )
        .sort_values("prob_bucket")
    )

    metrics = {
        "n_games": len(df),
        "probability_column_used": prob_col,
        "rmse_home": rmse_home,
        "rmse_away": rmse_away,
        "rmse_total": rmse_total,
        "mae_home": mae_home,
        "mae_away": mae_away,
        "mae_total": mae_total,
        "bias_home": bias_home,
        "bias_away": bias_away,
        "bias_total": bias_total,
        "logloss": logloss,
        "brier": brier,
        "accuracy": accuracy,
        "inconsistent_cases": len(inconsistent),
        "inconsistency_rate": inconsistency_rate,
    }

    return df, metrics, calibration, inconsistent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pg_dsn", default=os.environ.get("PG_DSN"))
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument(
        "--prob_col",
        default="p_home_win_poisson",
        choices=["p_home_win_poisson", "p_home_win_raw"],
        help="Which home win probability column to evaluate",
    )
    ap.add_argument(
        "--latest_only",
        action="store_true",
        help="Use only the latest prediction per game_id",
    )
    ap.add_argument(
        "--save_bad_cases",
        default="",
        help="Optional CSV path for inconsistent cases",
    )
    args = ap.parse_args()

    if not args.pg_dsn:
        raise RuntimeError("Missing PG_DSN")

    conn = psycopg2.connect(args.pg_dsn)
    df = load_eval_df(
        conn,
        start=args.start,
        end=args.end,
        latest_only=args.latest_only,
    )
    conn.close()

    if df.empty:
        print("No evaluation rows found.")
        return

    prob_col = args.prob_col
    if prob_col not in df.columns:
        raise KeyError(f"Column {prob_col!r} not in query result")
    if df[prob_col].notna().sum() == 0:
        print(f"Warning: {prob_col} is all null — metrics will be empty after dropna.")

    eval_df, metrics, calibration, inconsistent = compute_metrics(df, prob_col=prob_col)

    print("\n=== MODEL QUALITY METRICS ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    print("\n=== PREDICTION DISTRIBUTION ===")
    dist = eval_df[
        [
            "home_runs_pred",
            "away_runs_pred",
            "total_runs_pred",
            "p_home_win_pred",
        ]
    ].describe()
    print(dist.to_string())

    print("\n=== CALIBRATION TABLE ===")
    print(calibration.to_string(index=False))

    if len(inconsistent):
        print("\n=== SAMPLE INCONSISTENT CASES ===")
        sample_cols = [
            "game_date",
            "game_id",
            "away_team",
            "home_team",
            "home_runs_pred",
            "away_runs_pred",
            "run_diff_pred",
            "p_home_win_pred",
            "home_runs_actual",
            "away_runs_actual",
            "home_win_actual",
        ]
        print(inconsistent[sample_cols].head(20).to_string(index=False))

    if args.save_bad_cases:
        inconsistent.to_csv(args.save_bad_cases, index=False)
        print(f"\nSaved inconsistent cases to: {args.save_bad_cases}")


if __name__ == "__main__":
    main()