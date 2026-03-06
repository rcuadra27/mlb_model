#!/usr/bin/env python3
"""
features/add_run_preds_to_features_game.py

Adds model-predicted run features to public.features_game:
- home_runs_pred
- away_runs_pred
- run_diff_pred
- total_runs_pred

It loads the feature list from runs_model_metrics_optionA.json to guarantee the
columns match training.

Usage:
  export PG_DSN="postgresql+psycopg2://USER:PASS@HOST:5432/mlb_model"
  python features/add_run_preds_to_features_game.py \
    --start 2015-04-01 --end 2025-11-15 \
    --artifacts_dir artifacts \
    --runs_metrics runs_model_metrics_optionA.json \
    --home_model runs_model_home_lgbm_optionA.joblib \
    --away_model runs_model_away_lgbm_optionA.joblib \
    --batch 20000
"""

import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sqlalchemy import create_engine, text


def ensure_columns(engine, schema: str):
    cols = {
        "home_runs_pred": "DOUBLE PRECISION",
        "away_runs_pred": "DOUBLE PRECISION",
        "run_diff_pred": "DOUBLE PRECISION",
        "total_runs_pred": "DOUBLE PRECISION",
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


def fetch_features(engine, schema: str, start: str, end: str, feature_cols):
    # Only pull what we need: game_id, game_date, and features required by the runs models
    cols_sql = ", ".join([f'"{c}"' for c in (["game_id", "game_date"] + feature_cols)])
    sql = text(f"""
        SELECT {cols_sql}
        FROM {schema}.features_game
        WHERE game_date BETWEEN :start AND :end
        ORDER BY game_date, game_id
    """)
    return pd.read_sql(sql, engine, params={"start": start, "end": end})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", default="public")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--artifacts_dir", default="artifacts")
    ap.add_argument("--runs_metrics", default="runs_model_metrics_optionA.json")
    ap.add_argument("--home_model", default="runs_model_home_lgbm_optionA.joblib")
    ap.add_argument("--away_model", default="runs_model_away_lgbm_optionA.joblib")
    ap.add_argument("--batch", type=int, default=20000)
    args = ap.parse_args()

    pg_dsn = os.getenv("PG_DSN")
    if not pg_dsn:
        raise RuntimeError("PG_DSN env var required")

    engine = create_engine(pg_dsn, pool_pre_ping=True)

    artifacts = Path(args.artifacts_dir)
    metrics_path = artifacts / args.runs_metrics
    home_path = artifacts / args.home_model
    away_path = artifacts / args.away_model

    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    if not home_path.exists():
        raise FileNotFoundError(home_path)
    if not away_path.exists():
        raise FileNotFoundError(away_path)

    # Metrics file written by our trainer includes a list called "features".
    # Older pandas versions can choke on dict-shaped JSON via read_json, so
    # load it with the stdlib json module instead.
    import json
    with open(metrics_path, "r") as f:
        md = json.load(f)
    feature_cols = md.get("features")
    if not feature_cols:
        raise RuntimeError(f"'features' list not found in {metrics_path}. Open the JSON and confirm it contains 'features'.")

    print(f"Loaded {len(feature_cols)} feature columns from {metrics_path.name}")

    home_model = joblib.load(home_path)
    away_model = joblib.load(away_path)

    ensure_columns(engine, args.schema)

    df = fetch_features(engine, args.schema, args.start, args.end, feature_cols)
    if df.empty:
        print("No rows found in features_game for that date range.")
        return

    # Cast categorical columns if present (must match training behavior)
    for cat_col in ["venue_id", "park_id", "home_team_id", "away_team_id"]:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].astype("category")

    # Handle missing feature columns (should not happen if pipeline is consistent)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required feature columns in features_game: {missing[:20]} (and {len(missing)-20} more)" if len(missing) > 20 else f"Missing: {missing}")

    X = df[feature_cols]

    print("Predicting runs...")
    home_pred = home_model.predict(X)
    away_pred = away_model.predict(X)

    out = pd.DataFrame({
        "game_id": df["game_id"].astype(int).values,
        "home_runs_pred": home_pred.astype(float),
        "away_runs_pred": away_pred.astype(float),
    })
    out["run_diff_pred"] = out["home_runs_pred"] - out["away_runs_pred"]
    out["total_runs_pred"] = out["home_runs_pred"] + out["away_runs_pred"]

    upd_sql = text(f"""
        UPDATE {args.schema}.features_game
        SET
          home_runs_pred = :home_runs_pred,
          away_runs_pred = :away_runs_pred,
          run_diff_pred = :run_diff_pred,
          total_runs_pred = :total_runs_pred
        WHERE game_id = :game_id
    """)

    print(f"Updating {len(out):,} games in batches of {args.batch:,}...")
    payload = out.to_dict(orient="records")
    with engine.begin() as conn:
        conn.execute(text("SET LOCAL statement_timeout = '0'"))
        for i in range(0, len(payload), args.batch):
            conn.execute(upd_sql, payload[i:i+args.batch])
            print(f"  updated {min(i+args.batch, len(payload)):,}/{len(payload):,}")

    print("Done. Quick check:")
    with engine.connect() as conn:
        r = conn.execute(text(f"""
            SELECT
              COUNT(*) AS n,
              COUNT(home_runs_pred) AS n_home,
              COUNT(away_runs_pred) AS n_away,
              COUNT(run_diff_pred) AS n_diff
            FROM {args.schema}.features_game
            WHERE game_date BETWEEN :start AND :end
        """), {"start": args.start, "end": args.end}).fetchone()
    print(dict(r._mapping))


if __name__ == "__main__":
    main()
