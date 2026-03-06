#!/usr/bin/env python3
"""
train_win_model_optionA.py

Option A: Direct home-win classifier using baseball + weather features only (no market/odds features).

Usage example:
  export PG_DSN="postgresql+psycopg2://user:pass@host:5432/dbname"
  python train_win_model_optionA.py \
    --schema public \
    --features_table features_game \
    --games_table games \
    --date_col game_date \
    --id_col game_id \
    --train_end 2023-12-31 \
    --val_end 2024-06-30 \
    --test_end 2024-12-31

Notes:
- Assumes features_game has at least: game_id, game_date, and feature columns.
- Target:
    - If features_game has a 'home_win' column, uses it.
    - Else joins games table and derives home_win from home_score/away_score (or similar).
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

import lightgbm as lgb
import joblib

from sqlalchemy import create_engine, text
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score


# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def pick_first_existing(cols: List[str], candidates: List[str]) -> Optional[str]:
    cols_set = set(cols)
    for c in candidates:
        if c in cols_set:
            return c
    return None


def split_by_date(
    df: pd.DataFrame,
    date_col: str,
    train_end: str,
    val_end: str,
    test_end: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col]).dt.date

    train_end_d = pd.to_datetime(train_end).date()
    val_end_d = pd.to_datetime(val_end).date()
    test_end_d = pd.to_datetime(test_end).date()

    train = d[d[date_col] <= train_end_d]
    val = d[(d[date_col] > train_end_d) & (d[date_col] <= val_end_d)]
    test = d[(d[date_col] > val_end_d) & (d[date_col] <= test_end_d)]

    return train, val, test


def make_report(split: str, y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "split": split,
        "n": int(len(y)),
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
        "mean_p_homewin": float(np.mean(p)),
        "mean_y": float(np.mean(y)),
    }


# -----------------------------
# Data loading
# -----------------------------
def load_training_frame(
    engine,
    schema: str,
    features_table: str,
    games_table: str,
    id_col: str,
    date_col: str
) -> pd.DataFrame:
    """
    Loads a dataframe from features_table.
    If home_win exists in features_table, uses it.
    Otherwise joins games_table to derive home_win from scores.

    The script tries common score column names in games_table:
      home_score: home_score, home_runs, home_team_score
      away_score: away_score, away_runs, away_team_score
    """

    # Pull a small sample of columns to inspect
    insp_sql = text(f"""
        SELECT *
        FROM {schema}.{features_table}
        LIMIT 1
    """)
    sample = pd.read_sql(insp_sql, engine)
    if sample.empty:
        raise RuntimeError(f"No rows found in {schema}.{features_table}")

    feat_cols = list(sample.columns)
    has_home_win = "home_win" in feat_cols

    if has_home_win:
        # Load full features with home_win target already present
        sql = text(f"""
            SELECT *
            FROM {schema}.{features_table}
        """)
        df = pd.read_sql(sql, engine)
        return df

    # Otherwise, we join games to get scores and derive home_win
    # First inspect games columns
    game_insp = pd.read_sql(text(f"SELECT * FROM {schema}.{games_table} LIMIT 1"), engine)
    if game_insp.empty:
        raise RuntimeError(f"No rows found in {schema}.{games_table}")

    game_cols = list(game_insp.columns)

    home_score_col = pick_first_existing(
        game_cols, ["home_score", "home_runs", "home_team_score", "score_home", "home_r"]
    )
    away_score_col = pick_first_existing(
        game_cols, ["away_score", "away_runs", "away_team_score", "score_away", "away_r"]
    )

    if home_score_col is None or away_score_col is None:
        raise RuntimeError(
            f"Could not find score columns in {schema}.{games_table}. "
            f"Looked for home in [home_score, home_runs, home_team_score, score_home, home_r] "
            f"and away in [away_score, away_runs, away_team_score, score_away, away_r]. "
            f"Found columns: {game_cols}"
        )

    # Join and derive
    sql = text(f"""
        SELECT
            f.*,
            CASE WHEN g.{home_score_col} > g.{away_score_col} THEN 1 ELSE 0 END AS home_win
        FROM {schema}.{features_table} f
        JOIN {schema}.{games_table} g
          ON f.{id_col} = g.{id_col}
    """)
    df = pd.read_sql(sql, engine)
    return df


# -----------------------------
# Main train routine
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", default="public")
    ap.add_argument("--features_table", default="features_game")
    ap.add_argument("--games_table", default="games")
    ap.add_argument("--id_col", default="game_id")
    ap.add_argument("--date_col", default="game_date")

    ap.add_argument("--train_end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--val_end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--test_end", required=True, help="YYYY-MM-DD")

    ap.add_argument("--artifacts_dir", default="artifacts")
    ap.add_argument("--model_out", default="win_model_home_lgbm_optionA.joblib")
    ap.add_argument("--metrics_out", default="win_model_metrics.json")
    ap.add_argument("--fi_out", default="win_model_feature_importance.csv")

    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--early_stopping_rounds", type=int, default=200)

    # If you know which columns should never be used as features, list them here
    ap.add_argument(
        "--drop_cols",
        default="game_id,game_date,home_team,away_team,home_team_id,away_team_id,season,home_runs,away_runs,home_score,away_score",
        help="Comma-separated columns to drop from features if present."
    )

    args = ap.parse_args()

    pg_dsn = os.getenv("PG_DSN")
    if not pg_dsn:
        raise RuntimeError("PG_DSN env var is required. Example: postgresql+psycopg2://user:pass@host:5432/dbname")

    ensure_dir(args.artifacts_dir)

    engine = create_engine(pg_dsn)

    print("Loading training frame...")
    df = load_training_frame(
        engine=engine,
        schema=args.schema,
        features_table=args.features_table,
        games_table=args.games_table,
        id_col=args.id_col,
        date_col=args.date_col,
    )

    if args.date_col not in df.columns:
        raise RuntimeError(f"date_col '{args.date_col}' not found in dataframe. Columns: {list(df.columns)[:50]}...")

    if "home_win" not in df.columns:
        raise RuntimeError("home_win target not found/created. Check your tables and join keys.")

    # Basic cleaning
    df = df.dropna(subset=["home_win"])
    df["home_win"] = df["home_win"].astype(int)

    # Time split
    train_df, val_df, test_df = split_by_date(df, args.date_col, args.train_end, args.val_end, args.test_end)

    print(f"Split sizes: train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise RuntimeError("One of the splits is empty. Check date ranges and date_col values.")

    # Build feature list
    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]
    always_drop = set(drop_cols + ["home_win"])  # target
    
    # Also drop datetime columns and other non-numeric columns that LightGBM can't handle
    datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    always_drop.update(datetime_cols)
    
    feature_cols = [c for c in df.columns if c not in always_drop]
    
    # Filter out object dtype columns (unless they're meant to be categorical)
    # Keep only numeric, bool, or category dtypes
    valid_dtypes = {'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
                    'float16', 'float32', 'float64', 'bool', 'category'}
    feature_cols = [c for c in feature_cols 
                    if str(df[c].dtype) in valid_dtypes or 
                       df[c].dtype.name == 'category' or
                       pd.api.types.is_numeric_dtype(df[c])]
    
    # Convert categorical columns (important for venue_id)
    # If venue_id exists, cast to category
    for cat_col in ["venue_id", "park_id", "home_team_id", "away_team_id"]:
        if cat_col in feature_cols:
            for part_name, part in (("train", train_df), ("val", val_df), ("test", test_df)):
                if cat_col in part.columns:
                    part.loc[:, cat_col] = part[cat_col].astype("category")


    X_train = train_df[feature_cols]
    y_train = train_df["home_win"].values
    X_val = val_df[feature_cols]
    y_val = val_df["home_win"].values
    X_test = test_df[feature_cols]
    y_test = test_df["home_win"].values

    # LightGBM model (Option A win model)
    win_model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=4000,
        learning_rate=0.02,
        num_leaves=31,              # cut in half
        min_data_in_leaf=500,       # stronger smoothing
        feature_fraction=0.6,
        bagging_fraction=0.6,
        bagging_freq=1,
        lambda_l1=1.0,
        lambda_l2=25.0,             # much stronger L2
        max_depth=6,                # limit tree depth
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining LightGBM binary win model (Option A)...")
    win_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=args.early_stopping_rounds, verbose=True)]
    )

    best_iter = getattr(win_model, "best_iteration_", None)
    print(f"\nBest iteration: {best_iter}")

    # Predictions
    p_tr = win_model.predict_proba(X_train)[:, 1]
    p_val = win_model.predict_proba(X_val)[:, 1]
    p_te = win_model.predict_proba(X_test)[:, 1]

    # Reports
    rep_train = make_report("train", y_train, p_tr)
    rep_val = make_report("val", y_val, p_val)
    rep_test = make_report("test", y_test, p_te)

    print("\nWin-model performance (LightGBM binary):")
    for rep in (rep_train, rep_val, rep_test):
        print(
            f"{rep['split']:>5}  "
            f"log_loss={rep['log_loss']:.6f}  "
            f"brier={rep['brier']:.6f}  "
            f"auc={rep['auc']:.6f}  "
            f"mean_p={rep['mean_p_homewin']:.6f}"
        )

    # Save model
    model_path = Path(args.artifacts_dir) / args.model_out
    joblib.dump(win_model, model_path)
    print(f"\nSaved model: {model_path}")

    # Feature importances
    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": win_model.booster_.feature_importance(importance_type="gain"),
        "importance_split": win_model.booster_.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)

    fi_path = Path(args.artifacts_dir) / args.fi_out
    fi.to_csv(fi_path, index=False)
    print(f"Saved feature importances: {fi_path}")

    # Metrics JSON
    metrics = {
        "best_iteration": int(best_iter) if best_iter is not None else None,
        "train": rep_train,
        "val": rep_val,
        "test": rep_test,
        "n_features": int(len(feature_cols)),
        "features_table": f"{args.schema}.{args.features_table}",
        "games_table": f"{args.schema}.{args.games_table}",
        "date_col": args.date_col,
        "splits": {"train_end": args.train_end, "val_end": args.val_end, "test_end": args.test_end},
    }

    metrics_path = Path(args.artifacts_dir) / args.metrics_out
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
