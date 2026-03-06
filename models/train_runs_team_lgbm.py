#!/usr/bin/env python3
import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from lightgbm import LGBMRegressor

BASE_DROP_EXACT = {
    "game_id", "game_date", "season",
    "home_team_id", "away_team_id",
    "home_runs", "away_runs",
}

# Extra guardrails against leakage (even if your MV already excluded these)
DROP_SUBSTRINGS = [
    "price",         # market odds leakage
    "_pred",         # predicted columns
    "p_home", "p_away", "p_tie", "p_",  # any probability fields
]

def load_base(engine, schema: str) -> pd.DataFrame:
    df = pd.read_sql(text(f"SELECT * FROM {schema}.train_team_runs_base"), engine)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df

def drop_leaky(cols):
    out = []
    for c in cols:
        lc = c.lower()
        if any(s in lc for s in DROP_SUBSTRINGS):
            continue
        out.append(c)
    return out

def build_two_row_dataset(df_game: pd.DataFrame, home_prefix="home_", away_prefix="away_") -> pd.DataFrame:
    cols = df_game.columns.tolist()

    home_cols = [c for c in cols if c.startswith(home_prefix)]
    away_cols = [c for c in cols if c.startswith(away_prefix)]
    global_cols = [
        c for c in cols
        if (not c.startswith(home_prefix))
        and (not c.startswith(away_prefix))
        and (c not in BASE_DROP_EXACT)
    ]

    # Safety drop
    home_cols = drop_leaky(home_cols)
    away_cols = drop_leaky(away_cols)
    global_cols = drop_leaky(global_cols)

    # HOME ROWS
    home = df_game[["game_id", "game_date", "season", "home_team_id", "away_team_id", "home_runs"] + global_cols + home_cols + away_cols].copy()
    home = home.rename(columns={"home_team_id": "team_id", "away_team_id": "opp_id", "home_runs": "target_runs"})
    home["is_home"] = 1
    home = home.rename(columns={c: "team_" + c[len(home_prefix):] for c in home_cols})
    home = home.rename(columns={c: "opp_" + c[len(away_prefix):] for c in away_cols})

    # AWAY ROWS
    away = df_game[["game_id", "game_date", "season", "home_team_id", "away_team_id", "away_runs"] + global_cols + home_cols + away_cols].copy()
    away = away.rename(columns={"away_team_id": "team_id", "home_team_id": "opp_id", "away_runs": "target_runs"})
    away["is_home"] = 0
    away = away.rename(columns={c: "team_" + c[len(away_prefix):] for c in away_cols})
    away = away.rename(columns={c: "opp_" + c[len(home_prefix):] for c in home_cols})

    # Ensure both frames have the same, uniquely named columns before concatenation
    common_cols = [c for c in home.columns if c in away.columns]
    cols = []
    seen = set()
    for c in common_cols:
        if c not in seen:
            cols.append(c)
            seen.add(c)
    home = home[cols].copy()
    away = away[cols].copy()

    out = pd.concat([home, away], ignore_index=True)
    # Remove duplicate column names (fixes target_runs duplication bug)
    out = out.loc[:, ~out.columns.duplicated()].copy()
    out = out.dropna(subset=["target_runs"]).copy()
    return out

def train(df: pd.DataFrame, seed=42):
    # Robust y extraction even if target_runs was duplicated
    y_col = df.loc[:, "target_runs"]
    if isinstance(y_col, pd.DataFrame):
        y_col = y_col.iloc[:, 0]
    y = y_col.astype(float).to_numpy()

    X = df.drop(columns=["target_runs", "game_date"]).copy()
    if "game_id" in X.columns:
        X = X.drop(columns=["game_id"])
    X = X.loc[:, ~X.columns.duplicated()].copy()

    # Categoricals
    for c in ["team_id", "opp_id", "season", "is_home"]:
        if c in X.columns:
            X[c] = X[c].astype("category")

    season_int = X["season"].astype(int)
    train_mask = season_int <= 2022
    val_mask = season_int == 2023

    X_train = X.loc[train_mask]
    X_val = X.loc[val_mask]
    y_train = y[train_mask.to_numpy()]
    y_val = y[val_mask.to_numpy()]

    model = LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.1,
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=128,
        min_child_samples=40,
        subsample=0.85,
        subsample_freq=1,
        feature_fraction=0.85,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
    )

    return model, list(X.columns)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", default="public")
    ap.add_argument("--out", default="artifacts/runs_model_team_lgbm_optionA.joblib")
    ap.add_argument("--features_out", default="artifacts/runs_model_team_features_optionA.txt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pg_dsn = os.getenv("PG_DSN")
    if not pg_dsn:
        raise RuntimeError("PG_DSN env var is required (postgresql+psycopg2://...).")

    engine = create_engine(pg_dsn, pool_pre_ping=True)

    df_base = load_base(engine, args.schema)
    df_team = build_two_row_dataset(df_base)

    print("Base games:", len(df_base))
    print("Team rows:", len(df_team))
    print("Mean runs (home rows):", df_team.loc[df_team["is_home"] == 1, "target_runs"].mean())
    print("Mean runs (away rows):", df_team.loc[df_team["is_home"] == 0, "target_runs"].mean())

    model, feature_cols = train(df_team, seed=args.seed)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(model, args.out)

    with open(args.features_out, "w") as f:
        for c in feature_cols:
            f.write(c + "\n")

    print(f"\nSaved model -> {args.out}")
    print(f"Saved features -> {args.features_out}")

    fi = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    print("\nTop 25 features:")
    print(fi.head(25).to_string(index=False))
    if "is_home" in set(fi["feature"]):
        print("\nis_home importance:", fi.loc[fi["feature"] == "is_home", "importance"].iloc[0])

if __name__ == "__main__":
    main()