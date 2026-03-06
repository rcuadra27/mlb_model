#!/usr/bin/env python3
"""
models/train_runs_model_optionA.py

Option A runs model:
- Train two regressors: home_runs and away_runs
- Uses baseball + weather + lineup matchup features (no odds features)

Usage:
  export PG_DSN="postgresql+psycopg2://USER:PASS@HOST:5432/mlb_model"
  python models/train_runs_model_optionA.py \
    --train_end 2022-12-31 \
    --val_end 2023-12-31 \
    --test_end 2024-12-31
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

from sqlalchemy import create_engine, text
from sklearn.metrics import mean_absolute_error, mean_squared_error


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def pick_first_existing(cols: List[str], candidates: List[str]) -> Optional[str]:
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    return None


def split_by_date(df: pd.DataFrame, date_col: str, train_end: str, val_end: str, test_end: str):
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col]).dt.date
    train_end_d = pd.to_datetime(train_end).date()
    val_end_d = pd.to_datetime(val_end).date()
    test_end_d = pd.to_datetime(test_end).date()

    train = d[d[date_col] <= train_end_d]
    val = d[(d[date_col] > train_end_d) & (d[date_col] <= val_end_d)]
    test = d[(d[date_col] > val_end_d) & (d[date_col] <= test_end_d)]
    return train, val, test


def load_frame(engine, schema: str, features_table: str, games_table: str, id_col: str) -> pd.DataFrame:
    # inspect games columns for target names
    g_sample = pd.read_sql(text(f"SELECT * FROM {schema}.{games_table} LIMIT 1"), engine)
    if g_sample.empty:
        raise RuntimeError(f"No rows found in {schema}.{games_table}")
    g_cols = list(g_sample.columns)

    home_runs_col = pick_first_existing(g_cols, ["home_runs", "home_score", "home_team_score", "score_home", "home_r"])
    away_runs_col = pick_first_existing(g_cols, ["away_runs", "away_score", "away_team_score", "score_away", "away_r"])

    if home_runs_col is None or away_runs_col is None:
        raise RuntimeError(f"Could not find home/away runs columns in games. Found: {g_cols}")

    sql = text(f"""
        SELECT
          f.*,
          g.{home_runs_col} AS home_runs,
          g.{away_runs_col} AS away_runs
        FROM {schema}.{features_table} f
        JOIN {schema}.{games_table} g
          ON f.{id_col} = g.{id_col}
    """)
    df = pd.read_sql(sql, engine)
    return df


def report_reg(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return {
        "target": name,
        "mae": float(mae),
        "rmse": float(rmse),
        "mean_true": float(np.mean(y_true)),
        "mean_pred": float(np.mean(y_pred)),
    }


def train_one(
    X_train, y_train,
    X_val, y_val,
    random_state: int,
    early_stopping_rounds: int
) -> lgb.LGBMRegressor:
    # Tweedie works well for runs, with variance power ~1.1–1.4; 1.2 is a solid default.
    model = lgb.LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.2,
        n_estimators=6000,
        learning_rate=0.02,
        num_leaves=63,
        min_data_in_leaf=300,
        feature_fraction=0.7,
        bagging_fraction=0.7,
        bagging_freq=1,
        lambda_l1=0.5,
        lambda_l2=10.0,
        max_depth=7,
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True)]
    )
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", default="public")
    ap.add_argument("--features_table", default="features_game")
    ap.add_argument("--games_table", default="games")
    ap.add_argument("--id_col", default="game_id")
    ap.add_argument("--date_col", default="game_date")

    ap.add_argument("--train_end", required=True)
    ap.add_argument("--val_end", required=True)
    ap.add_argument("--test_end", required=True)

    ap.add_argument("--artifacts_dir", default="artifacts")
    ap.add_argument("--home_model_out", default="runs_model_home_lgbm_optionA.joblib")
    ap.add_argument("--away_model_out", default="runs_model_away_lgbm_optionA.joblib")
    ap.add_argument("--metrics_out", default="runs_model_metrics_optionA.json")
    ap.add_argument("--fi_home_out", default="runs_model_feature_importance_home.csv")
    ap.add_argument("--fi_away_out", default="runs_model_feature_importance_away.csv")

    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--early_stopping_rounds", type=int, default=200)

    ap.add_argument(
        "--drop_cols",
        default="game_id,game_date,home_team,away_team,home_team_id,away_team_id,season,home_runs,away_runs,home_score,away_score,home_win",
        help="Comma-separated columns to drop if present."
    )

    args = ap.parse_args()

    pg_dsn = os.getenv("PG_DSN")
    if not pg_dsn:
        raise RuntimeError("PG_DSN env var required")

    ensure_dir(args.artifacts_dir)
    engine = create_engine(pg_dsn, pool_pre_ping=True)

    print("Loading frame...")
    df = load_frame(engine, args.schema, args.features_table, args.games_table, args.id_col)

    # drop rows missing targets
    df = df.dropna(subset=["home_runs", "away_runs"])
    df["home_runs"] = pd.to_numeric(df["home_runs"], errors="coerce")
    df["away_runs"] = pd.to_numeric(df["away_runs"], errors="coerce")
    df = df.dropna(subset=["home_runs", "away_runs"])

    train_df, val_df, test_df = split_by_date(df, args.date_col, args.train_end, args.val_end, args.test_end)
    print(f"Split sizes: train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise RuntimeError("Empty split detected; check date boundaries.")

    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]
    always_drop = set(drop_cols)

    # Drop datetime columns
    datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    always_drop.update(datetime_cols)

    feature_cols = [c for c in df.columns if c not in always_drop]

    # Keep only numeric/bool/category
    def is_valid(c):
        return (
            pd.api.types.is_numeric_dtype(df[c]) or
            df[c].dtype == bool or
            df[c].dtype.name == "category"
        )

    feature_cols = [c for c in feature_cols if is_valid(c)]

    # Categorical casting (venue_id is super important)
    for cat_col in ["venue_id", "park_id", "home_team_id", "away_team_id"]:
        if cat_col in feature_cols:
            for part in (train_df, val_df, test_df):
                part.loc[:, cat_col] = part[cat_col].astype("category")

    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]
    X_test = test_df[feature_cols]

    print(f"n_features={len(feature_cols)}")

    # Train home runs model
    print("\nTraining home_runs Tweedie...")
    home_model = train_one(
        X_train, train_df["home_runs"].values,
        X_val, val_df["home_runs"].values,
        args.random_state, args.early_stopping_rounds
    )

    # Train away runs model
    print("\nTraining away_runs Tweedie...")
    away_model = train_one(
        X_train, train_df["away_runs"].values,
        X_val, val_df["away_runs"].values,
        args.random_state, args.early_stopping_rounds
    )

    # Predictions + metrics
    preds = {}
    metrics = {"splits": {"train_end": args.train_end, "val_end": args.val_end, "test_end": args.test_end}}

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        Xs = split_df[feature_cols]
        yh = home_model.predict(Xs)
        ya = away_model.predict(Xs)

        metrics[f"home_runs_{split_name}"] = report_reg("home_runs", split_df["home_runs"].values, yh)
        metrics[f"away_runs_{split_name}"] = report_reg("away_runs", split_df["away_runs"].values, ya)

    print("\nRuns-model performance (LightGBM Tweedie):")
    for split_name in ("train","val","test"):
        hr = metrics[f"home_runs_{split_name}"]
        ar = metrics[f"away_runs_{split_name}"]
        print(f"home_runs {split_name:>4}  mae={hr['mae']:.4f} rmse={hr['rmse']:.4f} mean_true={hr['mean_true']:.4f} mean_pred={hr['mean_pred']:.4f}")
        print(f"away_runs {split_name:>4}  mae={ar['mae']:.4f} rmse={ar['rmse']:.4f} mean_true={ar['mean_true']:.4f} mean_pred={ar['mean_pred']:.4f}")

    # Save models
    home_path = Path(args.artifacts_dir) / args.home_model_out
    away_path = Path(args.artifacts_dir) / args.away_model_out
    joblib.dump(home_model, home_path)
    joblib.dump(away_model, away_path)
    print(f"\nSaved:\n - {home_path}\n - {away_path}")

    # Save feature importance
    fi_home = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": home_model.booster_.feature_importance(importance_type="gain"),
        "importance_split": home_model.booster_.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)

    fi_away = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": away_model.booster_.feature_importance(importance_type="gain"),
        "importance_split": away_model.booster_.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)

    fi_home_path = Path(args.artifacts_dir) / args.fi_home_out
    fi_away_path = Path(args.artifacts_dir) / args.fi_away_out
    fi_home.to_csv(fi_home_path, index=False)
    fi_away.to_csv(fi_away_path, index=False)
    print(f"Saved feature importances:\n - {fi_home_path}\n - {fi_away_path}")

    # Save metrics JSON
    metrics["n_features"] = int(len(feature_cols))
    metrics["features"] = feature_cols
    metrics_path = Path(args.artifacts_dir) / args.metrics_out
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
