#!/usr/bin/env python3
"""
Score runs_model_v9 on 2025 regular-season games (OOS vs train <= 2023).

Outputs one row per game with preds, actuals, SP names, park, and top feature values.
"""
from __future__ import annotations

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.train_runs_team_lgbm import (  # noqa: E402
    LOCKED_FEATURES,
    NAN_PASSTHROUGH_FEATURES,
    build_two_row_dataset,
    load_base,
)


def _top_feature_cols(model, feature_cols: list[str], k: int = 10) -> list[str]:
    booster = model.booster_ if hasattr(model, "booster_") else model
    imp = booster.feature_importance(importance_type="gain")
    pairs = sorted(zip(feature_cols, imp), key=lambda x: -x[1])
    return [n for n, _ in pairs[:k]]


def _prepare_X(df_team: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = df_team.drop(columns=["target_runs", "game_date"], errors="ignore").copy()
    if "game_id" in X.columns:
        X = X.drop(columns=["game_id"])
    X = X.loc[:, ~X.columns.duplicated()].copy()
    X = X[feature_cols].copy()
    non_passthrough = [c for c in feature_cols if c not in NAN_PASSTHROUGH_FEATURES]
    X[non_passthrough] = X[non_passthrough].fillna(X[non_passthrough].median())
    for c in ["team_id", "opp_id", "season"]:
        if c in X.columns:
            X[c] = X[c].astype("category")
    return X


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", default="public")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--model", default="artifacts/runs_model_v9.joblib")
    ap.add_argument("--features", default="artifacts/runs_model_v9_features.txt")
    ap.add_argument("--out", default="artifacts/v9_2025_holdout_eval.csv")
    ap.add_argument("--top_k", type=int, default=10)
    args = ap.parse_args()

    pg_dsn = os.getenv("PG_DSN")
    if not pg_dsn:
        raise RuntimeError("PG_DSN required")

    with open(args.features) as f:
        feature_cols = [ln.strip() for ln in f if ln.strip()]

    model = joblib.load(args.model)
    top_feats = _top_feature_cols(model, feature_cols, k=args.top_k)

    engine = create_engine(pg_dsn, pool_pre_ping=True)
    print(f"Loading base data for season {args.season}...")
    df_base = load_base(engine, args.schema)
    df_base = df_base[df_base["season"].astype(int) == args.season].copy()
    print(f"  {len(df_base)} finished games")

    # SP names + park from games / features
    meta = pd.read_sql(
        text(f"""
            SELECT
              g.game_id,
              g.game_date,
              g.home_team_name AS home_team,
              g.away_team_name AS away_team,
              sp.home_sp_name,
              sp.away_sp_name,
              g.venue_id AS park_id
            FROM {args.schema}.games g
            LEFT JOIN {args.schema}.game_starting_pitchers sp USING (game_id)
            WHERE g.season = :season
              AND g.home_runs IS NOT NULL
              AND g.away_runs IS NOT NULL
        """),
        engine,
        params={"season": args.season},
    )

    df_team = build_two_row_dataset(df_base)
    df_team = df_team[df_team["season"].astype(int) == args.season].copy()
    print(f"  {len(df_team)} team rows")

    X = _prepare_X(df_team, feature_cols)
    residuals = model.predict(X).astype(float)
    league_baseline = df_team["league_baseline"].values
    pred_raw = np.clip(residuals + league_baseline, 2.5, None)

    scored = df_team[["game_id", "is_home", "target_runs_raw"]].copy()
    scored["pred_runs"] = pred_raw
    for feat in top_feats:
        scored[feat] = X[feat].values

    home = scored[scored["is_home"] == 1].set_index("game_id")
    away = scored[scored["is_home"] == 0].set_index("game_id")

    home_cols = {
        "target_runs_raw": "actual_home_runs",
        "pred_runs": "pred_home_runs",
        **{f: f"home_{f}" for f in top_feats},
    }
    away_cols = {
        "target_runs_raw": "actual_away_runs",
        "pred_runs": "pred_away_runs",
        **{f: f"away_{f}" for f in top_feats},
    }

    out = meta.set_index("game_id").join(
        home[["target_runs_raw", "pred_runs"] + top_feats].rename(columns=home_cols),
        how="inner",
    ).join(
        away[["target_runs_raw", "pred_runs"] + top_feats].rename(columns=away_cols),
        how="inner",
    )

    out = out.reset_index()
    out["game_date"] = out["game_date"].astype(str).str[:10]

    cols = [
        "game_id",
        "game_date",
        "home_team",
        "away_team",
        "pred_home_runs",
        "pred_away_runs",
        "actual_home_runs",
        "actual_away_runs",
        "home_sp_name",
        "away_sp_name",
        "park_id",
    ]
    for feat in top_feats:
        cols.extend([f"home_{feat}", f"away_{feat}"])

    out = out[cols].sort_values(["game_date", "game_id"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)

    n = len(out)
    rmse_h = float(np.sqrt(np.mean((out["pred_home_runs"] - out["actual_home_runs"]) ** 2)))
    rmse_a = float(np.sqrt(np.mean((out["pred_away_runs"] - out["actual_away_runs"]) ** 2)))
    bias_h = float((out["pred_home_runs"] - out["actual_home_runs"]).mean())
    bias_a = float((out["pred_away_runs"] - out["actual_away_runs"]).mean())
    home_fav = (out["pred_home_runs"] > out["pred_away_runs"]).mean()

    print(f"\nWrote {n} games -> {args.out}")
    print(f"RMSE home/away: {rmse_h:.3f} / {rmse_a:.3f}")
    print(f"Bias  home/away: {bias_h:+.3f} / {bias_a:+.3f}")
    print(f"Pred home > away: {home_fav*100:.1f}%")


if __name__ == "__main__":
    main()
