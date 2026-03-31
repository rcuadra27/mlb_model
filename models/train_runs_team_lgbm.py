#!/usr/bin/env python3
"""
train_runs_model.py

Trains a LightGBM Tweedie regressor to predict team runs scored per game.
Two rows per game (home + away). Game-level diff features (home - away) are
FLIPPED for the away row so every row's features are team-vs-opponent, not
always home-vs-away.
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from lightgbm import LGBMRegressor


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DROP_EXACT = {
    "game_id", "game_date", "season",
    "home_team_id", "away_team_id",
    "home_runs", "away_runs",
}

DROP_SUBSTRINGS = [
    "price",
    "_pred",
    "p_home",
    "p_away",
    "p_tie",
    "p_home_win",
    "p_home_median",
    "p_home_blend",
    "n_books",
]

LOCKED_FEATURES = [
    # ── Team strength — multiple windows ──────────────────────────────────
    "win_pct_diff",          # 30d
    "runs_for_diff",         # 30d
    "runs_against_diff",     # 30d
    "avg_runs_scored_60_diff",
    "avg_runs_allowed_60_diff",
    "runs_for_7d_diff",      # short-window momentum
    "runs_for_15d_diff",
    "runs_against_7d_diff",
    "runs_against_15d_diff",
    "win_pct_7d_diff",
    "win_pct_15d_diff",
    # ── Absolute scoring environment (both teams combined) ────────────────
    "total_offense_env",     # home_scored_60 + away_scored_60 — captures high-total games
    "total_defense_env",     # home_allowed_60 + away_allowed_60
    # ── Starting pitcher — legacy ─────────────────────────────────────────
    "sp_ra9_diff",
    "sp_ip_diff",
    "sp_era_last3_diff",     # ERA in last 3 starts — captures recent form
    "sp_k9_last5_diff",      # K/9 in last 5 starts
    "sp_days_rest_diff",     # rest days advantage
    # ── Starting pitcher — statcast 90d ───────────────────────────────────
    "sp_xwoba_against_diff",
    "sp_k_rate_diff",
    "sp_bb_rate_diff",
    "sp_gb_rate_diff",
    "sp_n_pa_diff",
    # ── Bullpen fatigue ───────────────────────────────────────────────────
    "bp_outs_3d_diff",
    "bp_hlev_3d_diff",
    "bp_b2b_diff",
    # ── Lineup quality — statcast 90d ────────────────────────────────────
    "lineup_xwoba_diff",
    "lineup_k_rate_diff",
    "lineup_bb_rate_diff",
    "lineup_n_pa_diff",
    "lineup_barrel_rate_diff",    # raw power — key for high-total games
    "lineup_hard_hit_rate_diff",
    # ── Matchup ───────────────────────────────────────────────────────────
    "lineup_skill_diff",
    "lineup_skill_diff_known",
    "matchup_diff",
    "matchup_diff_known",
    "lineup_vs_sp_score_diff",
    # ── Context ───────────────────────────────────────────────────────────
    "is_home",
    # ── Park factors ──────────────────────────────────────────────────────
    "park_runs_factor",          # prior-season 3yr average
    "park_runs_factor_blended",  # blend of prior + current season (updates weekly)
    # ── Weather — NaN passthrough ─────────────────────────────────────────
    "temp_f",
    "wind_mph",
    "wind_dir_sin",
    "wind_dir_cos",
    "humidity",
    "precip_in",
    #Umpire
    "umpire_runs_boost",    # umpire's historical runs/game vs league avg
    "umpire_n_games",       # sample size signal
]

# LightGBM handles NaN natively for these — do NOT impute with zero or median.
WEATHER_FEATURES = [
    "temp_f", "wind_mph", "wind_dir_sin", "wind_dir_cos",
    "humidity", "precip_in",
]

# Also pass statcast features through as NaN — sparse early in careers/season.
NAN_PASSTHROUGH_FEATURES = WEATHER_FEATURES + [
    # SP legacy — missing when pitcher unknown or no start history
    "sp_ra9_diff", "sp_ip_diff",
    "sp_era_last3_diff", "sp_k9_last5_diff", "sp_days_rest_diff",
    # SP statcast
    "sp_xwoba_against_diff", "sp_k_rate_diff", "sp_bb_rate_diff",
    "sp_gb_rate_diff", "sp_n_pa_diff",
    # Lineup statcast — sparse early in season / careers
    "lineup_xwoba_diff", "lineup_k_rate_diff", "lineup_bb_rate_diff",
    "lineup_n_pa_diff", "lineup_vs_sp_score_diff",
    "lineup_barrel_rate_diff", "lineup_hard_hit_rate_diff",
    # Short-window team stats — missing early in season
    "runs_for_7d_diff", "runs_for_15d_diff",
    "runs_against_7d_diff", "runs_against_15d_diff",
    "win_pct_7d_diff", "win_pct_15d_diff",
    # Scoring environment — missing when 60d cols missing
    "total_offense_env", "total_defense_env",
    # Blended park factor — missing until enough current-season games
    "park_runs_factor_blended",
    "umpire_runs_boost",    # null for new umpires or missing assignments
    "umpire_n_games",
]

# Computed as (home - away) at game level.
# Negated for the away team row so they read (team - opponent).
GAME_LEVEL_DIFF_COLS = [
    # Team strength diffs
    "sp_ra9_diff", "sp_ip_diff",
    "bp_outs_3d_diff", "bp_hlev_3d_diff", "bp_b2b_diff",
    "win_pct_diff", "runs_for_diff", "runs_against_diff",
    "avg_runs_scored_60_diff", "avg_runs_allowed_60_diff",
    # Short-window diffs
    "runs_for_7d_diff", "runs_for_15d_diff",
    "runs_against_7d_diff", "runs_against_15d_diff",
    "win_pct_7d_diff", "win_pct_15d_diff",
    # SP recent form diffs
    "sp_era_last3_diff", "sp_k9_last5_diff", "sp_days_rest_diff",
    # Matchup (precomputed home-away)
    "matchup_diff",
    # Statcast SP diffs
    "sp_xwoba_against_diff", "sp_k_rate_diff", "sp_bb_rate_diff",
    "sp_gb_rate_diff", "sp_n_pa_diff",
    # Statcast lineup diffs
    "lineup_xwoba_diff", "lineup_k_rate_diff", "lineup_bb_rate_diff",
    "lineup_n_pa_diff", "lineup_vs_sp_score_diff",
    "lineup_barrel_rate_diff", "lineup_hard_hit_rate_diff",
    # NOTE: total_offense_env, total_defense_env, park factors are NOT diffs —
    # they are global game-level features, same for both team rows. Do NOT flip.
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_base(engine, schema: str) -> pd.DataFrame:
    df = pd.read_sql(
        text(f"""
            SELECT
              f.*,
              g.home_runs,
              g.away_runs,
              gw.temp_f       AS weather_temp_f,
              gw.wind_mph     AS weather_wind_mph,
              gw.wind_dir_deg AS weather_wind_dir_deg,
              gw.humidity     AS weather_humidity,
              gw.precip_in    AS weather_precip_in
            FROM {schema}.features_game f
            JOIN {schema}.games g USING (game_id)
            LEFT JOIN {schema}.game_weather gw USING (game_id)
            WHERE g.home_runs IS NOT NULL
              AND g.away_runs IS NOT NULL
        """),
        engine,
    )
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Prefer game_weather values; fall back to whatever features_game stored.
    weather_map = {
        "temp_f":       "weather_temp_f",
        "wind_mph":     "weather_wind_mph",
        "wind_dir_deg": "weather_wind_dir_deg",
        "humidity":     "weather_humidity",
        "precip_in":    "weather_precip_in",
    }
    for canon, backup in weather_map.items():
        if backup not in df.columns:
            continue
        if canon not in df.columns:
            df[canon] = df[backup]
        else:
            df[canon] = df[canon].where(df[canon].notna(), df[backup])

    drop_cols = [c for c in weather_map.values() if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def drop_leaky(cols):
    out = []
    for c in cols:
        lc = c.lower()
        if any(s in lc for s in DROP_SUBSTRINGS):
            continue
        out.append(c)
    return out


def add_game_level_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute game-level diff features (home - away) on the raw game frame
    before splitting into team rows. Flipped for away rows in build_two_row_dataset.
    """
    out = df.copy()

    def add_diff(new_col, home_col, away_col):
        if home_col in out.columns and away_col in out.columns:
            out[new_col] = out[home_col].astype(float) - out[away_col].astype(float)

    add_diff("sp_ra9_diff",              "home_sp_ra9_last5",          "away_sp_ra9_last5")
    add_diff("sp_ip_diff",               "home_sp_ip_per_start_last5", "away_sp_ip_per_start_last5")
    add_diff("bp_outs_3d_diff",          "home_bp_outs_3d",            "away_bp_outs_3d")
    add_diff("bp_hlev_3d_diff",          "home_bp_hlev_outs_3d",       "away_bp_hlev_outs_3d")
    add_diff("win_pct_diff",             "home_win_pct_30",             "away_win_pct_30")
    add_diff("runs_for_diff",            "home_runs_for_30",            "away_runs_for_30")
    add_diff("runs_against_diff",        "home_runs_against_30",        "away_runs_against_30")
    add_diff("avg_runs_scored_60_diff",  "home_avg_runs_scored_60",     "away_avg_runs_scored_60")
    add_diff("avg_runs_allowed_60_diff", "home_avg_runs_allowed_60",    "away_avg_runs_allowed_60")

    # bp_b2b: try column name variants in order of preference
    for h, a in [
        ("home_bp_b2b_pitchers_3d", "away_bp_b2b_pitchers_3d"),
        ("home_bp_b2b",             "away_bp_b2b"),
        ("home_bp_b2b_3d",          "away_bp_b2b_3d"),
    ]:
        if h in out.columns and a in out.columns:
            out["bp_b2b_diff"] = out[h].astype(float) - out[a].astype(float)
            break

    if "bp_b2b_diff" in out.columns:
        out["bp_b2b_diff"] = out["bp_b2b_diff"].fillna(0.0)

    # matchup_diff, lineup_*_vs_sp, and statcast SP/lineup diffs are all
    # precomputed in features_game by build_features.py as (home - away).
    # They pass through as global columns and get flipped for away rows.

    # New statcast diff features (materialised by build_features.py)
    add_diff("sp_xwoba_against_diff",  "home_sp_xwoba_against_90",  "away_sp_xwoba_against_90")
    add_diff("sp_k_rate_diff",         "home_sp_k_rate_90",         "away_sp_k_rate_90")
    add_diff("sp_bb_rate_diff",        "home_sp_bb_rate_90",        "away_sp_bb_rate_90")
    add_diff("sp_gb_rate_diff",        "home_sp_gb_rate_90",        "away_sp_gb_rate_90")
    add_diff("sp_n_pa_diff",           "home_sp_n_pa_90",           "away_sp_n_pa_90")
    add_diff("lineup_xwoba_diff",      "home_lineup_xwoba_90",      "away_lineup_xwoba_90")
    add_diff("lineup_k_rate_diff",     "home_lineup_k_rate_90",     "away_lineup_k_rate_90")
    add_diff("lineup_bb_rate_diff",    "home_lineup_bb_rate_90",    "away_lineup_bb_rate_90")
    add_diff("lineup_n_pa_diff",       "home_lineup_n_pa_90",       "away_lineup_n_pa_90")
    add_diff("lineup_vs_sp_score_diff","home_lineup_vs_sp_score",   "away_lineup_vs_sp_score")

    # New lineup power diffs (from build_features.py)
    add_diff("lineup_barrel_rate_diff",   "home_lineup_barrel_rate_90",   "away_lineup_barrel_rate_90")
    add_diff("lineup_hard_hit_rate_diff", "home_lineup_hard_hit_rate_90", "away_lineup_hard_hit_rate_90")

    # New short-window team stat diffs
    add_diff("runs_for_7d_diff",     "home_runs_for_7d",     "away_runs_for_7d")
    add_diff("runs_for_15d_diff",    "home_runs_for_15d",    "away_runs_for_15d")
    add_diff("runs_against_7d_diff", "home_runs_against_7d", "away_runs_against_7d")
    add_diff("runs_against_15d_diff","home_runs_against_15d","away_runs_against_15d")
    add_diff("win_pct_7d_diff",      "home_win_pct_7d",      "away_win_pct_7d")
    add_diff("win_pct_15d_diff",     "home_win_pct_15d",     "away_win_pct_15d")

    # New SP recent form diffs
    add_diff("sp_era_last3_diff",  "home_sp_era_last3",  "away_sp_era_last3")
    add_diff("sp_k9_last5_diff",   "home_sp_k9_last5",   "away_sp_k9_last5")
    add_diff("sp_days_rest_diff",  "home_sp_days_rest",  "away_sp_days_rest")

    # Absolute scoring environment — global columns, NOT diffs, pass through as-is.
    # total_offense_env and total_defense_env are computed in build_features.py
    # from existing features_game 60d columns. They require no add_diff here.

    # Wind cyclic encoding from raw degrees
    if "wind_dir_deg" in out.columns:
        wd = out["wind_dir_deg"].astype(float)
        out["wind_dir_sin"] = np.sin(np.deg2rad(wd))
        out["wind_dir_cos"] = np.cos(np.deg2rad(wd))

    return out


def add_missingness_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary indicator columns for features with meaningful missingness.
    Call AFTER all diffs are finalised so source columns are present.
    """
    out = df.copy()
    for col in ["lineup_skill_diff", "matchup_diff"]:
        flag = f"{col}_known"
        if col in out.columns:
            out[flag] = out[col].notna().astype(float)
        else:
            out[flag] = 0.0
    return out


def add_team_level_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Diff features derived from team_* / opp_* columns after the home/away
    rename. Naturally team-vs-opponent for both rows.

    lineup_skill_diff and matchup_diff are precomputed in features_game and
    pass through as global columns — only recomputed here as a fallback if
    somehow absent.
    """
    out = df.copy()

    def first_existing(candidates):
        for c in candidates:
            if c in out.columns:
                return c
        return None

    def add_diff_if_missing(new_col, team_candidates, opp_candidates):
        if new_col in out.columns:
            return  # already present — do not overwrite
        tc = first_existing(team_candidates)
        oc = first_existing(opp_candidates)
        if tc is not None and oc is not None:
            out[new_col] = out[tc].astype(float) - out[oc].astype(float)

    add_diff_if_missing("lineup_skill_diff", ["team_lineup_skill"], ["opp_lineup_skill"])
    add_diff_if_missing("matchup_diff",      ["team_matchup"],      ["opp_matchup"])

    return out


# ---------------------------------------------------------------------------
# Two-row dataset builder
# ---------------------------------------------------------------------------

def build_two_row_dataset(df_game: pd.DataFrame) -> pd.DataFrame:
    # Step 1: game-level diffs (home - away) on the raw frame.
    df_game = add_game_level_diff_features(df_game)

    cols      = df_game.columns.tolist()
    home_cols = [c for c in cols if c.startswith("home_")]
    away_cols = [c for c in cols if c.startswith("away_")]
    global_cols = [
        c for c in cols
        if not c.startswith("home_")
        and not c.startswith("away_")
        and c not in BASE_DROP_EXACT
    ]

    home_cols   = drop_leaky(home_cols)
    away_cols   = drop_leaky(away_cols)
    global_cols = drop_leaky(global_cols)

    id_cols = ["game_id", "game_date", "season"]

    # ------------------------------------------------------------------
    # HOME rows — diff features are (home - away), correct as team-vs-opp
    # ------------------------------------------------------------------
    home = df_game[
        id_cols + ["home_team_id", "away_team_id", "home_runs"]
        + global_cols + home_cols + away_cols
    ].copy()
    home = home.rename(columns={
        "home_team_id": "team_id",
        "away_team_id": "opp_id",
        "home_runs":    "target_runs",
    })
    home["is_home"] = 1
    home = home.rename(columns={c: "team_" + c[5:] for c in home_cols})
    home = home.rename(columns={c: "opp_"  + c[5:] for c in away_cols})

    # ------------------------------------------------------------------
    # AWAY rows — flip game-level diffs so they read (team - opponent)
    # ------------------------------------------------------------------
    away = df_game[
        id_cols + ["home_team_id", "away_team_id", "away_runs"]
        + global_cols + home_cols + away_cols
    ].copy()
    away = away.rename(columns={
        "away_team_id": "team_id",
        "home_team_id": "opp_id",
        "away_runs":    "target_runs",
    })
    away["is_home"] = 0
    away = away.rename(columns={c: "team_" + c[5:] for c in away_cols})
    away = away.rename(columns={c: "opp_"  + c[5:] for c in home_cols})

    # KEY FIX: negate all game-level diffs for the away row
    for col in GAME_LEVEL_DIFF_COLS:
        if col in away.columns:
            away[col] = -away[col]

    # ------------------------------------------------------------------
    # Combine
    # ------------------------------------------------------------------
    common_cols = [c for c in home.columns if c in away.columns]
    seen, deduped = set(), []
    for c in common_cols:
        if c not in seen:
            deduped.append(c)
            seen.add(c)

    out = pd.concat([home[deduped], away[deduped]], ignore_index=True)
    out = out.loc[:, ~out.columns.duplicated()].copy()

    # Step 2: team-level diffs (only fills gaps, won't overwrite existing cols)
    out = add_team_level_diff_features(out)

    # Step 3: missingness flags — must come AFTER all diffs are finalised
    out = add_missingness_flags(out)

    out = out.dropna(subset=["target_runs"]).copy()

    # ── Residual target: subtract league mean so model learns deviations ──
    # This eliminates the absolute intercept problem — the model only needs
    # to learn how much each team deviates from the current league average.
    # league_avg_runs_60d is the per-team (not total) league average.
    # At inference: predicted_runs = model_output + league_avg_runs_60d
    TRAINING_LEAGUE_MEAN = 4.50  # fallback for early-season games (< 50 prior games)
    out["league_baseline"] = out["league_avg_runs_60d"].fillna(TRAINING_LEAGUE_MEAN)
    out["target_runs_raw"] = out["target_runs"].copy()  # keep for diagnostics
    out["target_runs"]     = out["target_runs"] - out["league_baseline"]

    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(df: pd.DataFrame, seed: int = 42):  # df must contain league_baseline and target_runs_raw
    y_col = df.loc[:, "target_runs"]
    if isinstance(y_col, pd.DataFrame):
        y_col = y_col.iloc[:, 0]
    y = y_col.astype(float).to_numpy()

    if "season" not in df.columns:
        raise RuntimeError("Expected 'season' column for time split.")
    season_int = df["season"].astype(int)

    X = df.drop(columns=["target_runs", "game_date"], errors="ignore").copy()
    if "game_id" in X.columns:
        X = X.drop(columns=["game_id"])
    X = X.loc[:, ~X.columns.duplicated()].copy()

    missing = [c for c in LOCKED_FEATURES if c not in X.columns]
    if missing:
        raise RuntimeError(
            "Missing required locked features in training dataframe: "
            + ", ".join(missing)
        )
    X = X[LOCKED_FEATURES].copy()

    # Impute non-passthrough features with column medians.
    # Weather and statcast NaNs pass through to LightGBM — genuinely missing
    # data (no lineup, no SP history) is more honestly represented as NaN
    # than as an imputed median.
    non_passthrough = [c for c in LOCKED_FEATURES if c not in NAN_PASSTHROUGH_FEATURES]
    X[non_passthrough] = X[non_passthrough].fillna(X[non_passthrough].median())

    # Validate
    all_null = [c for c in X.columns if X[c].isna().all()]
    if all_null:
        raise RuntimeError(
            "These locked features are entirely NULL: " + ", ".join(all_null)
        )

    constant = [c for c in X.columns if X[c].dropna().nunique() <= 1]
    if constant:
        print("Warning — constant features (low utility):", ", ".join(constant))

    na_pct = (X.isna().mean() * 100.0).sort_values(ascending=False)
    print("\nLocked feature NA%:")
    print(na_pct.to_string(float_format=lambda v: f"{v:.2f}"))

    for c in ["team_id", "opp_id", "season"]:
        if c in X.columns:
            X[c] = X[c].astype("category")

    train_mask = season_int <= 2022
    val_mask   = season_int == 2023

    X_train, X_val = X.loc[train_mask], X.loc[val_mask]
    y_train, y_val = y[train_mask.to_numpy()], y[val_mask.to_numpy()]

    print(f"\nTrain rows: {len(X_train)}  Val rows: {len(X_val)}")

    model = LGBMRegressor(
        objective="regression_l2",  # MSE — correct for residual prediction
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

    # ------------------------------------------------------------------
    # Diagnostics — convert residuals back to raw runs for interpretability
    # ------------------------------------------------------------------
    pred_residuals = model.predict(X).astype(float)

    # Get league baseline for all rows (stored in df before target was residualized)
    league_baseline = df["league_baseline"].values
    target_raw      = df["target_runs_raw"].values
    pred_raw        = pred_residuals + league_baseline  # convert back to raw runs

    tmp = pd.DataFrame({
        "is_home":        X["is_home"].astype(float).values,
        "target_runs":    target_raw,
        "prediction_raw": pred_raw,
        "residual":       pred_residuals,
    })
    print("\nBy is_home — mean target_runs (raw):")
    print(tmp.groupby("is_home")["target_runs"].mean().to_string())
    print("\nBy is_home — mean prediction (raw runs):")
    print(tmp.groupby("is_home")["prediction_raw"].mean().to_string())
    print("\nBy is_home — mean bias (pred - actual, raw runs):")
    tmp["bias"] = tmp["prediction_raw"] - tmp["target_runs"]
    print(tmp.groupby("is_home")["bias"].mean().to_string())
    print("\nBy is_home — mean residual predicted:")
    print(tmp.groupby("is_home")["residual"].mean().to_string())

    home_pred = pred_raw[X["is_home"].astype(float).values == 1.0]
    away_pred = pred_raw[X["is_home"].astype(float).values == 0.0]
    if len(home_pred) == len(away_pred):
        n_games = len(home_pred)
        print(f"\nMean home pred (raw): {home_pred.mean():.4f}  Mean away pred (raw): {away_pred.mean():.4f}")
        print(f"Games where home pred > away pred: {(home_pred > away_pred).sum()} / {n_games} "
              f"({(home_pred > away_pred).mean()*100:.1f}%)")

    return model, list(X.columns)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema",       default="public")
    ap.add_argument("--out",          default="artifacts/runs_model_team_lgbm.joblib")
    ap.add_argument("--features_out", default="artifacts/runs_model_team_features.txt")
    ap.add_argument("--seed",         type=int, default=42)
    args = ap.parse_args()

    pg_dsn = os.getenv("PG_DSN")
    if not pg_dsn:
        raise RuntimeError("PG_DSN env var is required (postgresql+psycopg2://...).")

    engine = create_engine(pg_dsn, pool_pre_ping=True)

    print("Loading base data...")
    df_base = load_base(engine, args.schema)
    print(f"Base games: {len(df_base)}")

    print("Building two-row dataset...")
    df_team = build_two_row_dataset(df_base)
    print(f"Team rows: {len(df_team)}")
    print(f"Mean runs (home rows, raw):     {df_team.loc[df_team['is_home'] == 1, 'target_runs_raw'].mean():.3f}")
    print(f"Mean runs (away rows, raw):     {df_team.loc[df_team['is_home'] == 0, 'target_runs_raw'].mean():.3f}")
    print(f"Mean runs (home rows, residual):{df_team.loc[df_team['is_home'] == 1, 'target_runs'].mean():.3f}")
    print(f"Mean runs (away rows, residual):{df_team.loc[df_team['is_home'] == 0, 'target_runs'].mean():.3f}")
    print(f"Mean league baseline:           {df_team['league_baseline'].mean():.3f}")

    print("\nLocked feature presence:")
    for c in LOCKED_FEATURES:
        status   = "YES" if c in df_team.columns else "MISSING"
        null_pct = df_team[c].isna().mean() * 100 if c in df_team.columns else 100.0
        print(f"  {c}: {status}  ({null_pct:.1f}% null)")

    model, feature_cols = train(df_team, seed=args.seed)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump(model, args.out)
    with open(args.features_out, "w") as f:
        for c in feature_cols:
            f.write(c + "\n")

    print(f"\nSaved model    -> {args.out}")
    print(f"Saved features -> {args.features_out}")

    fi = (
        pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
    )
    print("\nTop feature importances:")
    print(fi.to_string(index=False))


if __name__ == "__main__":
    main()