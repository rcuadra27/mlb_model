#!/usr/bin/env python3
"""
inference_runs_model.py

Runs the trained LightGBM runs model for a given date. Fetches moneyline
AND totals odds from The Odds API in a single call, applies isotonic
calibration, derives win probabilities and O/U predictions, then prints
two dashboards:

  1. All games  — expected runs per team, win probabilities, O/U prediction
  2. Value bets — ML and O/U edges that exceed configurable thresholds

Usage:
    PG_DSN=... ODDS_API_KEY=... python inference_runs_model.py \\
        --date 2025-04-01 \\
        --team_model artifacts/runs_model_v2.joblib \\
        --team_features artifacts/runs_model_v2_features.txt \\
        --calibrator artifacts/calibrator_isotonic.joblib

    # Skip calibration:
        --no_calibrate

    # Custom thresholds:
        --ml_edge_threshold 0.05 --ou_edge_threshold 0.05 --min_run_diff 0.8
"""

import os
import argparse
import requests

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from scipy.stats import skellam, poisson

# ---------------------------------------------------------------------------
# Constants — must match training script exactly
# ---------------------------------------------------------------------------

WEATHER_FEATURES = [
    "temp_f", "wind_mph", "wind_dir_sin", "wind_dir_cos",
    "humidity", "precip_in",
]

NAN_PASSTHROUGH_FEATURES = WEATHER_FEATURES + [
    "sp_ra9_diff", "sp_ip_diff",
    "sp_era_last3_diff", "sp_era_last5_diff", "sp_era_season_diff",
    "sp_whip_last5_diff",
    "sp_k9_last5_diff", "sp_days_rest_diff",
    "sp_pitches_last_start_diff", "sp_innings_season_diff",
    "sp_xwoba_against_diff", "sp_k_rate_diff", "sp_bb_rate_diff",
    "sp_gb_rate_diff", "sp_n_pa_diff",
    "lineup_xwoba_diff", "lineup_k_rate_diff", "lineup_bb_rate_diff",
    "lineup_n_pa_diff", "lineup_vs_sp_score_diff",
    "lineup_barrel_rate_diff", "lineup_hard_hit_rate_diff",
    "runs_for_7d_diff", "runs_for_15d_diff",
    "runs_against_7d_diff", "runs_against_15d_diff",
    "win_pct_7d_diff", "win_pct_15d_diff",
    "win_streak_diff", "days_since_last_game_diff",
    "total_offense_env", "total_defense_env",
    "park_runs_factor_blended",
    "league_avg_runs_60d",
    "umpire_runs_boost", "umpire_n_games",
    "umpire_k_rate_boost", "umpire_bb_rate_boost",
    "line_move_magnitude",
]

DROP_SUBSTRINGS = [
    "price", "_pred", "p_home", "p_away", "p_tie",
    "p_home_win", "p_home_median", "p_home_blend", "n_books",
]

GAME_LEVEL_DIFF_COLS = [
    "sp_ra9_diff", "sp_ip_diff",
    "bp_outs_1d_diff",
    "bp_outs_3d_diff",
    "bp_outs_5d_diff",
    "bp_hlev_outs_1d_diff",
    "bp_hlev_3d_diff",
    "bp_b2b_diff",
    "win_pct_diff", "runs_for_diff", "runs_against_diff",
    "avg_runs_scored_60_diff", "avg_runs_allowed_60_diff",
    "runs_for_7d_diff", "runs_for_15d_diff",
    "runs_against_7d_diff", "runs_against_15d_diff",
    "win_pct_7d_diff", "win_pct_15d_diff",
    "win_streak_diff", "days_since_last_game_diff",
    "sp_era_last3_diff", "sp_era_last5_diff", "sp_era_season_diff",
    "sp_whip_last5_diff",
    "sp_k9_last5_diff", "sp_days_rest_diff",
    "sp_pitches_last_start_diff", "sp_innings_season_diff",
    "matchup_diff",
    "sp_xwoba_against_diff", "sp_k_rate_diff", "sp_bb_rate_diff",
    "sp_gb_rate_diff", "sp_n_pa_diff",
    "lineup_xwoba_diff", "lineup_k_rate_diff", "lineup_bb_rate_diff",
    "lineup_n_pa_diff", "lineup_vs_sp_score_diff",
    "lineup_barrel_rate_diff", "lineup_hard_hit_rate_diff",
]

ODDS_API_SPORT = "baseball_mlb"

# Added to clipped model outputs (inference only). Fitted on 2024 held-out data (n=2374 games).
BIAS_CORRECTION_HOME = 0.0
BIAS_CORRECTION_AWAY = 0.0
TRAINING_LEAGUE_MEAN = 4.50

# Min |predicted total − market O/U line| to emit OVER/UNDER; else NULL (no directional pick).
OU_PRED_LINE_GAP = 0.5


# ---------------------------------------------------------------------------
# Odds helpers
# ---------------------------------------------------------------------------

def american_to_implied(odds: np.ndarray) -> np.ndarray:
    o = odds.astype(float)
    p = np.full_like(o, np.nan, dtype=float)
    neg, pos = o < 0, o > 0
    p[neg] = (-o[neg]) / ((-o[neg]) + 100.0)
    p[pos] = 100.0 / (o[pos] + 100.0)
    return p


def prob_to_american(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(float), 1e-6, 1 - 1e-6)
    odds = np.empty_like(p)
    fav = p >= 0.5
    odds[fav]  = -100.0 * (p[fav] / (1.0 - p[fav]))
    odds[~fav] = 100.0 * ((1.0 - p[~fav]) / p[~fav])
    return np.rint(odds).astype(int)


def profit_if_win_1u(odds: float) -> float:
    if pd.isna(odds) or odds == 0:
        return np.nan
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)


def implied_from_american(price) -> float:
    if pd.isna(price):
        return np.nan
    return float(american_to_implied(np.array([float(price)]))[0])


# ---------------------------------------------------------------------------
# Win probability — Skellam conditioned on no tie
# ---------------------------------------------------------------------------

def p_home_win_from_lambdas(lh: float, la: float) -> float:
    lh = max(1e-9, float(lh))
    la = max(1e-9, float(la))
    p_hw   = float(1.0 - skellam.cdf(0, lh, la))
    p_tie  = float(skellam.pmf(0, lh, la))
    return p_hw / max(1.0 - p_tie, 1e-9)


# ---------------------------------------------------------------------------
# O/U probability — Poisson total
# ---------------------------------------------------------------------------

def p_over_under(lh: float, la: float, ou_line: float):
    """Returns (p_over, p_under, p_push). Total ~ Poisson(lh + la)."""
    mu    = max(1e-9, float(lh) + float(la))
    line  = float(ou_line)
    floor = int(line)
    p_le_floor = float(poisson.cdf(floor, mu))

    if line == float(floor):
        p_push  = float(poisson.pmf(floor, mu))
        p_under = p_le_floor - p_push
        p_over  = 1.0 - p_le_floor
    else:
        p_push  = 0.0
        p_under = p_le_floor
        p_over  = 1.0 - p_le_floor

    return p_over, p_under, p_push


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def drop_leaky(cols):
    return [c for c in cols if not any(s in c.lower() for s in DROP_SUBSTRINGS)]


def add_game_level_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def add_diff(new_col, hc, ac):
        if hc in out.columns and ac in out.columns:
            out[new_col] = out[hc].astype(float) - out[ac].astype(float)

    # 1. Legacy team strength diffs
    add_diff("sp_ra9_diff",              "home_sp_ra9_last5",          "away_sp_ra9_last5")
    add_diff("sp_ip_diff",               "home_sp_ip_per_start_last5", "away_sp_ip_per_start_last5")
    add_diff("bp_outs_3d_diff",          "home_bp_outs_3d",            "away_bp_outs_3d")
    add_diff("bp_hlev_3d_diff",          "home_bp_hlev_outs_3d",       "away_bp_hlev_outs_3d")
    add_diff("win_pct_diff",             "home_win_pct_30",             "away_win_pct_30")
    add_diff("runs_for_diff",            "home_runs_for_30",            "away_runs_for_30")
    add_diff("runs_against_diff",        "home_runs_against_30",        "away_runs_against_30")
    add_diff("avg_runs_scored_60_diff",  "home_avg_runs_scored_60",     "away_avg_runs_scored_60")
    add_diff("avg_runs_allowed_60_diff", "home_avg_runs_allowed_60",    "away_avg_runs_allowed_60")

    # Short-window team stats
    add_diff("runs_for_7d_diff",      "home_runs_for_7d",      "away_runs_for_7d")
    add_diff("runs_for_15d_diff",     "home_runs_for_15d",     "away_runs_for_15d")
    add_diff("runs_against_7d_diff",  "home_runs_against_7d",  "away_runs_against_7d")
    add_diff("runs_against_15d_diff", "home_runs_against_15d", "away_runs_against_15d")
    add_diff("win_pct_7d_diff",       "home_win_pct_7d",       "away_win_pct_7d")
    add_diff("win_pct_15d_diff",      "home_win_pct_15d",      "away_win_pct_15d")
    # SP recent form
    add_diff("sp_era_last3_diff",     "home_sp_era_last3",     "away_sp_era_last3")
    add_diff("sp_era_last5_diff",          "home_sp_era_last5",          "away_sp_era_last5")
    add_diff("sp_era_season_diff",         "home_sp_era_season",         "away_sp_era_season")
    add_diff("sp_whip_last5_diff",         "home_sp_whip_last5",         "away_sp_whip_last5")
    add_diff("sp_pitches_last_start_diff", "home_sp_pitches_last_start", "away_sp_pitches_last_start")
    add_diff("sp_innings_season_diff",     "home_sp_innings_season",     "away_sp_innings_season")
    add_diff("win_streak_diff",            "home_win_streak",            "away_win_streak")
    add_diff("days_since_last_game_diff",  "home_days_since_last_game",  "away_days_since_last_game")
    add_diff("bp_outs_1d_diff",            "home_bp_outs_1d",            "away_bp_outs_1d")
    add_diff("bp_outs_5d_diff",            "home_bp_outs_5d",            "away_bp_outs_5d")
    add_diff("bp_hlev_outs_1d_diff",       "home_bp_hlev_outs_1d",       "away_bp_hlev_outs_1d")
    # Lineup power
    add_diff("lineup_barrel_rate_diff",   "home_lineup_barrel_rate_90",   "away_lineup_barrel_rate_90")
    add_diff("lineup_hard_hit_rate_diff", "home_lineup_hard_hit_rate_90", "away_lineup_hard_hit_rate_90")
    # Scoring environment — global features, NOT diffs, pass through as-is
    # total_offense_env and total_defense_env require no add_diff
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

    if "wind_dir_deg" in out.columns:
        wd = out["wind_dir_deg"].astype(float)
        out["wind_dir_sin"] = np.sin(np.deg2rad(wd))
        out["wind_dir_cos"] = np.cos(np.deg2rad(wd))

    return out


def add_missingness_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["lineup_skill_diff", "matchup_diff"]:
        flag = f"{col}_known"
        out[flag] = out[col].notna().astype(float) if col in out.columns else 0.0
    return out


def add_team_level_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def first(candidates):
        return next((c for c in candidates if c in out.columns), None)

    def add_if_missing(new_col, tc, oc):
        if new_col in out.columns:
            return
        t, o = first(tc), first(oc)
        if t and o:
            out[new_col] = out[t].astype(float) - out[o].astype(float)

    add_if_missing("lineup_skill_diff", ["team_lineup_skill"], ["opp_lineup_skill"])
    add_if_missing("matchup_diff",      ["team_matchup"],      ["opp_matchup"])
    return out


def build_team_rows(df_game: pd.DataFrame, feature_cols: list):
    base_cols   = ["game_id", "season"]
    global_cols = drop_leaky([
        c for c in df_game.columns
        if not c.startswith("home_") and not c.startswith("away_")
        and c not in ["game_date","home_team","away_team","home_team_id","away_team_id"]
    ])
    home_cols = drop_leaky([c for c in df_game.columns if c.startswith("home_")])
    away_cols = drop_leaky([c for c in df_game.columns if c.startswith("away_")])
    all_cols  = base_cols + global_cols + home_cols + away_cols

    def make_row(df, team_col, opp_col, target_home, team_rename, opp_rename, is_home_val):
        frame = df[all_cols].copy()
        frame["team_id"] = df[team_col].values
        frame["opp_id"]  = df[opp_col].values
        frame["is_home"] = is_home_val
        frame = frame.rename(columns={c: "team_" + c[5:] for c in team_rename})
        frame = frame.rename(columns={c: "opp_"  + c[5:] for c in opp_rename})
        frame = frame.loc[:, ~frame.columns.duplicated()].copy()
        frame = add_team_level_diff_features(frame)
        return frame

    H = make_row(df_game, "home_team_id", "away_team_id", True,  home_cols, away_cols, 1)
    A = make_row(df_game, "away_team_id", "home_team_id", False, away_cols, home_cols, 0)

    for col in GAME_LEVEL_DIFF_COLS:
        if col in A.columns:
            A[col] = pd.to_numeric(A[col], errors='coerce').multiply(-1)

    H = add_missingness_flags(H)
    A = add_missingness_flags(A)

    for frame in [H, A]:
        for c in feature_cols:
            if c not in frame.columns:
                frame[c] = np.nan

    XH = H[feature_cols].copy()
    XA = A[feature_cols].copy()

    for X in [XH, XA]:
        for c in ["team_id", "opp_id", "season"]:
            if c in X.columns:
                X[c] = X[c].astype("category")

    return XH, XA


def coerce_feature_dtypes_for_lgbm(X: pd.DataFrame) -> pd.DataFrame:
    """
    LightGBM requires int/float/bool — not object. SQL/nullable columns often
    arrive as object (strings or mixed); coerce numeric features to float64.
    """
    out = X.copy()
    cat_cols = {"team_id", "opp_id", "season"}
    for c in out.columns:
        if c in cat_cols:
            out[c] = out[c].astype("category")
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def fetch_games_and_features(engine, schema: str, date_str: str) -> pd.DataFrame:
    q = text(f"""
        SELECT g.game_id, g.game_date, g.home_team_id, g.away_team_id,
               th.team_name AS home_team, ta.team_name AS away_team,
               f.*,
               gsp.home_sp_name, gsp.away_sp_name,
               gw.temp_f AS gw_temp_f, gw.wind_mph AS gw_wind_mph,
               gw.wind_dir_deg AS gw_wind_dir_deg,
               gw.humidity AS gw_humidity, gw.precip_in AS gw_precip_in
      FROM {schema}.games g
        JOIN {schema}.features_game f ON f.game_id = g.game_id
        LEFT JOIN {schema}.teams th ON th.mlb_team_id = g.home_team_id
        LEFT JOIN {schema}.teams ta ON ta.mlb_team_id = g.away_team_id
        LEFT JOIN {schema}.game_starting_pitchers gsp ON gsp.game_id = g.game_id
        LEFT JOIN {schema}.game_weather gw ON gw.game_id = g.game_id
      WHERE g.game_date = :d
      ORDER BY g.game_id
    """)
    df = pd.read_sql(q, engine, params={"d": date_str})
    df = df.loc[:, ~df.columns.duplicated()].copy()

    for canon, backup, forecast in [
        ("temp_f",     "gw_temp_f",      "forecast_temp_f"),
        ("wind_mph",   "gw_wind_mph",    "forecast_wind_mph"),
        ("wind_dir_deg","gw_wind_dir_deg","forecast_wind_dir_deg"),
        ("humidity",   "gw_humidity",    "forecast_humidity"),
        ("precip_in",  "gw_precip_in",   "forecast_precip_in"),
    ]:
        # Use game_weather actuals first, then forecast fallback
        if canon not in df.columns:
            df[canon] = np.nan
        if backup in df.columns:
            df[canon] = df[canon].where(df[canon].notna(), df[backup])
        if forecast in df.columns:
            df[canon] = df[canon].where(df[canon].notna(), df[forecast])

    df = df.drop(columns=[
        "gw_temp_f","gw_wind_mph","gw_wind_dir_deg","gw_humidity","gw_precip_in"
    ], errors="ignore")

    if "season" not in df.columns:
        df["season"] = pd.to_datetime(df["game_date"]).dt.year
    return df


def fetch_odds_api(date_str: str, api_key: str) -> pd.DataFrame:
    """Single Odds API call — returns h2h + totals for all MLB games on date."""
    url = f"https://api.the-odds-api.com/v4/sports/{ODDS_API_SPORT}/odds/"
    params = {
        "apiKey":    api_key,
        "regions":   "us",
        "markets":   "h2h,totals",
        "oddsFormat":"american",
        "dateFormat":"iso",
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  Odds API error: {e}")
        return pd.DataFrame()

    records = []
    for g in resp.json():
        # Convert UTC to PT before date comparison — late PT games are next day UTC
        commence_utc = pd.to_datetime(g.get("commence_time", ""), utc=True)
        commence_pt = commence_utc.tz_convert("America/Los_Angeles").date()
        if str(commence_pt) != date_str:
            continue
        # Skip games that have already started — use stored closing odds instead
        if commence_utc <= pd.Timestamp.now(tz="UTC"):
            continue

        home_team = g.get("home_team", "")
        away_team = g.get("away_team", "")

        ml_h, ml_a = [], []
        ou_lines, ou_over_px, ou_under_px = [], [], []

        for bk in g.get("bookmakers", []):
            for mkt in bk.get("markets", []):
                if mkt["key"] == "h2h":
                    for o in mkt["outcomes"]:
                        if o["name"] == home_team:
                            ml_h.append(o["price"])
                        elif o["name"] == away_team:
                            ml_a.append(o["price"])
                elif mkt["key"] == "totals":
                    for o in mkt["outcomes"]:
                        if o["name"] == "Over":
                            ou_lines.append(o.get("point", np.nan))
                            ou_over_px.append(o["price"])
                        elif o["name"] == "Under":
                            ou_under_px.append(o["price"])

        # Moneyline
        p_home_fair = p_away_fair = np.nan
        home_px = away_px = np.nan
        n_ml = 0
        if ml_h and ml_a:
            n_ml = min(len(ml_h), len(ml_a))
            ph = american_to_implied(np.array(ml_h[:n_ml]))
            pa = american_to_implied(np.array(ml_a[:n_ml]))
            p_home_fair = float(np.median(ph / (ph + pa)))
            p_away_fair = 1.0 - p_home_fair
            home_px = int(prob_to_american(np.array([p_home_fair]))[0])
            away_px = int(prob_to_american(np.array([p_away_fair]))[0])

        # Totals
        ou_line = ou_over_price = ou_under_price = np.nan
        n_ou = 0
        if ou_lines and ou_over_px and ou_under_px:
            n_ou = min(len(ou_lines), len(ou_over_px), len(ou_under_px))
            ou_line        = float(np.median(ou_lines[:n_ou]))
            ou_over_price  = float(np.median(ou_over_px[:n_ou]))
            ou_under_price = float(np.median(ou_under_px[:n_ou]))

        records.append({
            "api_home_team":        home_team,
            "api_away_team":        away_team,
            "p_home_market_median": p_home_fair,
            "p_away_market_median": p_away_fair,
            "home_price_consensus": home_px,
            "away_price_consensus": away_px,
            "n_books_ml":           n_ml,
            "ou_line":              ou_line,
            "ou_over_price":        ou_over_price,
            "ou_under_price":       ou_under_price,
            "n_books_ou":           n_ou,
        })

    return pd.DataFrame(records)


def match_odds_to_games(df_games: pd.DataFrame, df_odds: pd.DataFrame) -> pd.DataFrame:
    """Match Odds API team names to DB team names (exact then partial)."""
    odds_cols = [
        "p_home_market_median", "p_away_market_median",
        "home_price_consensus", "away_price_consensus", "n_books_ml",
        "ou_line", "ou_over_price", "ou_under_price", "n_books_ou",
    ]

    if df_odds.empty:
        for col in odds_cols:
            df_games[col] = np.nan
        return df_games

    result = df_games.copy()
    for col in odds_cols:
        result[col] = np.nan

    lookup = {
        (r["api_away_team"].lower().strip(), r["api_home_team"].lower().strip()): r
        for _, r in df_odds.iterrows()
    }

    for idx, game in result.iterrows():
        ht = game["home_team"].lower().strip()
        at = game["away_team"].lower().strip()
        matched = lookup.get((at, ht))

        if matched is None:
            for (ka, kh), row in lookup.items():
                if set(ht.split()) & set(kh.split()) and set(at.split()) & set(ka.split()):
                    matched = row
                    break

        if matched is not None:
            for col in odds_cols:
                result.at[idx, col] = matched.get(col, np.nan)

    return result


# ---------------------------------------------------------------------------
# Output table
# ---------------------------------------------------------------------------

def ensure_output_table(engine, schema: str) -> None:
    with engine.begin() as conn:
        conn.execute(text(f"""
        CREATE TABLE IF NOT EXISTS {schema}.inference_game_predictions (
            as_of_ts                TIMESTAMPTZ      NOT NULL,
            game_id                 BIGINT           NOT NULL,
            game_date               DATE             NOT NULL,
            home_team               TEXT,
            away_team               TEXT,
            home_runs_pred          DOUBLE PRECISION,
            away_runs_pred          DOUBLE PRECISION,
            total_runs_pred         DOUBLE PRECISION,
            run_diff_pred           DOUBLE PRECISION,
            p_home_win_raw          DOUBLE PRECISION,
            p_home_win_poisson      DOUBLE PRECISION,
            p_away_win_poisson      DOUBLE PRECISION,
            p_home_market_median    DOUBLE PRECISION,
            p_away_market_median    DOUBLE PRECISION,
            n_books_ml              INTEGER,
            home_price_consensus    INTEGER,
            away_price_consensus    INTEGER,
            ou_line                 DOUBLE PRECISION,
            ou_over_price           DOUBLE PRECISION,
            ou_under_price          DOUBLE PRECISION,
            n_books_ou              INTEGER,
            p_over                  DOUBLE PRECISION,
            p_under                 DOUBLE PRECISION,
            ou_recommendation       TEXT,
            edge_home               DOUBLE PRECISION,
            edge_away               DOUBLE PRECISION,
            ou_edge_over            DOUBLE PRECISION,
            ou_edge_under           DOUBLE PRECISION,
            ev_home                 DOUBLE PRECISION,
            ev_away                 DOUBLE PRECISION,
            ev_over                 DOUBLE PRECISION,
            ev_under                DOUBLE PRECISION,
            is_value_ml_home        BOOLEAN,
            is_value_ml_away        BOOLEAN,
            is_value_ou_over        BOOLEAN,
            is_value_ou_under       BOOLEAN,
            PRIMARY KEY (as_of_ts, game_id)
        );
        """))


# ---------------------------------------------------------------------------
# Consistency check
# ---------------------------------------------------------------------------

def assert_predictions_consistent(df: pd.DataFrame) -> None:
    violations = []
    for i, row in df.iterrows():
        lh, la = float(row["home_runs_pred"]), float(row["away_runs_pred"])
        if abs(lh - la) < 0.01:
            continue
        p_act  = float(1.0 - skellam.cdf(0,lh,la) - skellam.pmf(0,lh,la)) / \
                 max(float(1.0 - skellam.pmf(0,lh,la)), 1e-9)
        p_swap = float(1.0 - skellam.cdf(0,la,lh) - skellam.pmf(0,la,lh)) / \
                 max(float(1.0 - skellam.pmf(0,la,lh)), 1e-9)
        if not ((lh > la) == (p_act > p_swap)):
            violations.append(f"  {row.get('home_team','?')} vs {row.get('away_team','?')}: lh={lh:.3f} la={la:.3f}")
    if violations:
        raise RuntimeError("CONSISTENCY VIOLATION:\n" + "\n".join(violations))
    print(f"  Consistency check passed: {len(df)} games.")


# ---------------------------------------------------------------------------
# Dashboards
# ---------------------------------------------------------------------------

def print_all_games(df: pd.DataFrame) -> None:
    W = 118
    print(f"\n{'⚾  ALL GAMES — ' + str(df['game_date'].iloc[0])[:10]:^{W}}")
    print("─" * W)
    print(f"{'AWAY TEAM':<22} {'HOME TEAM':<22} "
          f"{'AWAY SP':<20} {'HOME SP':<20}" 
          f"{'AWAY PRED':>9} {'HOME PRED':>9} {'TOTAL':>6} "
          f"{'LINE':>6} {'P_OVER':>7} {'P_UNDR':>7} "
          f"{'REC':>5} {'P_WIN_A':>8} {'P_WIN_H':>8}")
    print("─" * W)

    for _, r in df.iterrows():
        ap   = f"{r['away_runs_pred']:.2f}"    if pd.notna(r.get('away_runs_pred'))   else "—"
        hp   = f"{r['home_runs_pred']:.2f}"    if pd.notna(r.get('home_runs_pred'))   else "—"
        tot  = f"{r['total_runs_pred']:.1f}"   if pd.notna(r.get('total_runs_pred'))  else "—"
        line = f"{r['ou_line']:.1f}"           if pd.notna(r.get('ou_line'))          else "—"
        po   = f"{r['p_over']*100:.0f}%"       if pd.notna(r.get('p_over'))           else "—"
        pu   = f"{r['p_under']*100:.0f}%"      if pd.notna(r.get('p_under'))          else "—"
        rec  = r.get('ou_recommendation') or "—"
        pwa  = f"{r['p_away_win_poisson']*100:.0f}%" if pd.notna(r.get('p_away_win_poisson')) else "—"
        pwh  = f"{r['p_home_win_poisson']*100:.0f}%" if pd.notna(r.get('p_home_win_poisson')) else "—"

        away_sp = (r.get('away_sp_name') or '—')[:16]
        home_sp = (r.get('home_sp_name') or '—')[:16]

        print(f"{r['away_team']:<22} {r['home_team']:<22} "
              f"{away_sp:<18} {home_sp:<18} "
              f"{ap:>9} {hp:>9} {tot:>6} "
              f"{line:>6} {po:>7} {pu:>7} "
              f"{rec:>5} {pwa:>8} {pwh:>8}")

    print("─" * W)
    print("  REC = model O/U recommendation based on predicted total vs line")
    print("  P_WIN = calibrated win probability")


def print_value_bets(df: pd.DataFrame, ml_thresh: float,
                     ou_thresh: float, min_diff: float) -> None:
    W = 120
    print(f"\n{'💰  VALUE BETS':^{W}}")
    print("─" * W)
    print(f"  Filters: ML edge ≥ {ml_thresh:.0%}  |  O/U edge ≥ {ou_thresh:.0%}  "
          f"|  |run diff| ≥ {min_diff:.1f} runs (ML only)")
    print("─" * W)
    print(f"{'TYPE':<5} {'AWAY':<21} {'HOME':<21} {'SIDE':<6} "
          f"{'MDL%':>5} {'MKT%':>5} {'EDGE':>6} "
          f"{'LINE':>5} {'MDL_TOT':>8} {'REC':>5} "
          f"{'EV':>7}  {'ODDS':>6}")
    print("─" * W)

    any_bet = False

    for _, r in df.iterrows():
        away = r['away_team'][:19]
        home = r['home_team'][:19]
        rdiff = abs(float(r.get('run_diff_pred') or 0))

        # Moneyline value bets
        if rdiff >= min_diff:
            for side, pc, mc, ec, evc, prc in [
                ("HOME", "p_home_win_poisson", "p_home_market_median",
                 "edge_home", "ev_home", "home_price_consensus"),
                ("AWAY", "p_away_win_poisson", "p_away_market_median",
                 "edge_away", "ev_away", "away_price_consensus"),
            ]:
                edge = r.get(ec)
                if pd.isna(edge) or edge < ml_thresh:
                    continue
                mp  = r.get(pc, np.nan)
                mkp = r.get(mc, np.nan)
                ev  = r.get(evc, np.nan)
                px  = r.get(prc, np.nan)
                print(f"{'ML':<5} {away:<21} {home:<21} {side:<6} "
                      f"{mp*100:>4.1f}% {mkp*100:>4.1f}% {edge*100:>+5.1f}% "
                      f"{'—':>5} {'—':>8} {'—':>5} "
                      f"{ev:>+6.3f}u  {int(px):>+6d}" if pd.notna(ev) and pd.notna(px)
                      else
                      f"{'ML':<5} {away:<21} {home:<21} {side:<6} "
                      f"{mp*100:>4.1f}% {mkp*100:>4.1f}% {edge*100:>+5.1f}% "
                      f"{'—':>5} {'—':>8} {'—':>5} {'—':>7}  {'—':>6}")
                any_bet = True

        # O/U value bets
        ou = r.get('ou_line')
        if pd.notna(ou):
            tot = r.get('total_runs_pred', np.nan)
            rec = r.get('ou_recommendation', '—')
            for side, pc, ec, evc, prc in [
                ("OVER",  "p_over",  "ou_edge_over",  "ev_over",  "ou_over_price"),
                ("UNDER", "p_under", "ou_edge_under", "ev_under", "ou_under_price"),
            ]:
                edge = r.get(ec)
                if pd.isna(edge) or edge < ou_thresh:
                    continue
                mp  = r.get(pc, np.nan)
                px  = r.get(prc, np.nan)
                ev  = r.get(evc, np.nan)
                mkt = implied_from_american(px) if pd.notna(px) else np.nan
                tot_s = f"{tot:.1f}" if pd.notna(tot) else "—"
                ev_s  = f"{ev:+.3f}u" if pd.notna(ev) else "—"
                px_s  = f"{int(px):+d}" if pd.notna(px) else "—"
                mkt_s = f"{mkt*100:.1f}%" if pd.notna(mkt) else "—"
                print(f"{'O/U':<5} {away:<21} {home:<21} {side:<6} "
                      f"{mp*100:>4.1f}% {mkt_s:>5} {edge*100:>+5.1f}% "
                      f"{ou:>5.1f} {tot_s:>8} {rec:>5} "
                      f"{ev_s:>7}  {px_s:>6}")
                any_bet = True

    if not any_bet:
        print("  No value bets found at current thresholds.")
        print(f"  Try: --ml_edge_threshold 0.03 --ou_edge_threshold 0.03 --min_run_diff 0.5")
    print("─" * W)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date",               required=True)
    ap.add_argument("--schema",             default="public")
    ap.add_argument("--team_model",         required=True)
    ap.add_argument("--team_features",      required=True)
    ap.add_argument("--calibrator",         default=None)
    ap.add_argument("--no_calibrate",       action="store_true")
    ap.add_argument("--fill_missing",       action="store_true")
    ap.add_argument("--ml_edge_threshold",  type=float, default=0.05)
    ap.add_argument("--ou_edge_threshold",  type=float, default=0.05)
    ap.add_argument("--min_run_diff",       type=float, default=0.8)
    args = ap.parse_args()

    pg_dsn  = os.getenv("PG_DSN")
    api_key = os.getenv("ODDS_API_KEY")
    if not pg_dsn:
        raise RuntimeError("PG_DSN env var required.")

    engine = create_engine(pg_dsn, pool_pre_ping=True)
    ensure_output_table(engine, args.schema)

    # Calibrator
    calibrator = None
    if args.calibrator and not args.no_calibrate:
        if os.path.exists(args.calibrator):
            calibrator = joblib.load(args.calibrator)
            print(f"  Calibrator: {args.calibrator}")
        else:
            print(f"  Warning: calibrator not found at {args.calibrator}")

    # Games
    df = fetch_games_and_features(engine, args.schema, args.date)
    if df.empty:
        print(f"No games found for {args.date}")
        return
    print(f"Found {len(df)} games for {args.date}")
    df = add_game_level_diff_features(df)

    # Odds
    if api_key:
        print("Fetching odds from The Odds API...")
        df_odds = fetch_odds_api(args.date, api_key)
        n_matched = len(df_odds)
        print(f"  API returned {n_matched} games")
        df = match_odds_to_games(df, df_odds)

    if not api_key or n_matched < len(df):
        print("  Falling back to morning odds from features_game...")
        for col, fg_col in [
            ("p_home_market_median", "morning_p_home"),
            ("ou_line",              "morning_ou_line"),
        ]:
            if fg_col in df.columns and col in df.columns:
                df[col] = df[col].where(df[col].notna(), df[fg_col])
    # Derive away probability and consensus prices from morning pull
        if "morning_p_home" in df.columns:
            mask = df["p_home_market_median"].isna() & df["morning_p_home"].notna()
            df.loc[mask, "p_home_market_median"] = df.loc[mask, "morning_p_home"].astype(float)
            df.loc[mask, "p_away_market_median"] = (1.0 - df.loc[mask, "morning_p_home"].astype(float))
            # Convert to American odds for display
            for idx in df[mask].index:
                p = float(df.at[idx, "p_home_market_median"])
                df.at[idx, "home_price_consensus"] = int(
                    -100*p/(1-p) if p >= 0.5 else 100*(1-p)/p)
                p_a = 1.0 - p
                df.at[idx, "away_price_consensus"] = int(
                    -100*p_a/(1-p_a) if p_a >= 0.5 else 100*(1-p_a)/p_a)
    if not api_key:
        print("  ODDS_API_KEY not set — odds columns will be NaN")
        for col in ["p_home_market_median","p_away_market_median",
                    "home_price_consensus","away_price_consensus","n_books_ml",
                    "ou_line","ou_over_price","ou_under_price","n_books_ou"]:
            df[col] = np.nan

    # Model features
    # Ensure odds columns are float dtype before saving
    float_cols = ['p_home_market_median','p_away_market_median','ou_line',
                  'ou_over_price','ou_under_price','home_price_consensus','away_price_consensus']
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    team_model = joblib.load(args.team_model)
    with open(args.team_features) as f:
        feature_cols = [l.strip() for l in f if l.strip()]

    XH, XA = build_team_rows(df, feature_cols)

    combined = pd.concat([XH, XA]).isna().mean()
    pt_miss  = combined[~combined.index.isin(NAN_PASSTHROUGH_FEATURES)]
    pt_miss  = pt_miss[pt_miss > 0]
    wx_miss  = combined[combined.index.isin(NAN_PASSTHROUGH_FEATURES)]
    if not wx_miss.empty:
        print(f"  NaN-passthrough missing: {wx_miss.mean()*100:.1f}% avg")
    if not pt_miss.empty:
        if not args.fill_missing:
            raise RuntimeError(f"Missing features:\n{(pt_miss*100).round(1).to_string()}\nPass --fill_missing")
        non_pt = [c for c in feature_cols if c not in NAN_PASSTHROUGH_FEATURES]
        XH[non_pt] = XH[non_pt].fillna(0.0)
        XA[non_pt] = XA[non_pt].fillna(0.0)

    # Predict
    # Predict — model outputs residuals from league mean
    df = df.reset_index(drop=True)

    XH = coerce_feature_dtypes_for_lgbm(XH)
    XA = coerce_feature_dtypes_for_lgbm(XA)

    # Step 1: raw residual predictions
    home_residuals = team_model.predict(XH).astype(float)
    away_residuals = team_model.predict(XA).astype(float)

    # Step 2: get league baseline — per-team average over prior 60 days
    league_baseline = df["league_avg_runs_60d"].fillna(TRAINING_LEAGUE_MEAN).values

    # Step 3: add baseline back to get actual run predictions
    df["home_runs_pred"] = np.clip(home_residuals + league_baseline, 2.5, None)
    df["away_runs_pred"] = np.clip(away_residuals + league_baseline, 2.5, None)
    df["total_runs_pred"] = df["home_runs_pred"] + df["away_runs_pred"]
    df["run_diff_pred"]   = df["home_runs_pred"] - df["away_runs_pred"]

    # Win probability from corrected predictions
    df["p_home_win_raw"] = [
        p_home_win_from_lambdas(h, a)
        for h, a in zip(df["home_runs_pred"].values, df["away_runs_pred"].values)
    ]

    
    if calibrator is not None:
        df["p_home_win_poisson"] = calibrator.predict(df["p_home_win_raw"].values)
    else:
        df["p_home_win_poisson"] = df["p_home_win_raw"]
    df["p_away_win_poisson"] = 1.0 - df["p_home_win_poisson"]

    assert_predictions_consistent(df)

    # O/U
    p_over_l, p_under_l, rec_l = [], [], []
    for _, row in df.iterrows():
        ou = row.get("ou_line")
        if pd.notna(ou):
            po, pu, _ = p_over_under(row["home_runs_pred"], row["away_runs_pred"], ou)
            p_over_l.append(po)
            p_under_l.append(pu)
            t = row["total_runs_pred"]
            if abs(float(t) - float(ou)) < OU_PRED_LINE_GAP:
                rec_l.append(None)
            else:
                rec_l.append("OVER" if float(t) > float(ou) else "UNDER")
        else:
            p_over_l.append(np.nan)
            p_under_l.append(np.nan)
            rec_l.append(None)

    df["p_over"]            = p_over_l
    df["p_under"]           = p_under_l
    df["ou_recommendation"] = rec_l

    # Edges
    df["edge_home"]     = df["p_home_win_poisson"] - df["p_home_market_median"]
    df["edge_away"]     = df["p_away_win_poisson"] - df["p_away_market_median"]
    df["ou_edge_over"]  = df.apply(lambda r: r["p_over"]  - implied_from_american(r.get("ou_over_price"))
                                   if pd.notna(r.get("p_over")) else np.nan, axis=1)
    df["ou_edge_under"] = df.apply(lambda r: r["p_under"] - implied_from_american(r.get("ou_under_price"))
                                   if pd.notna(r.get("p_under")) else np.nan, axis=1)

    # EV
    for ev_col, p_col, price_col in [
        ("ev_home",  "p_home_win_poisson", "home_price_consensus"),
        ("ev_away",  "p_away_win_poisson", "away_price_consensus"),
        ("ev_over",  "p_over",             "ou_over_price"),
        ("ev_under", "p_under",            "ou_under_price"),
    ]:
        evs = []
        for i in df.index:
            px = df.at[i, price_col]
            p  = df.at[i, p_col]
            if pd.notna(px) and pd.notna(p):
                b = profit_if_win_1u(float(px))
                evs.append(float(p) * b - (1.0 - float(p)) if pd.notna(b) else np.nan)
            else:
                evs.append(np.nan)
        df[ev_col] = evs

    # Value flags
    df["is_value_ml_home"]  = (df["run_diff_pred"].abs() >= args.min_run_diff) & \
                               (df["edge_home"].fillna(-1)     >= args.ml_edge_threshold)
    df["is_value_ml_away"]  = (df["run_diff_pred"].abs() >= args.min_run_diff) & \
                               (df["edge_away"].fillna(-1)     >= args.ml_edge_threshold)
    df["is_value_ou_over"]  = df["ou_edge_over"].fillna(-1)  >= args.ou_edge_threshold
    df["is_value_ou_under"] = df["ou_edge_under"].fillna(-1) >= args.ou_edge_threshold

    # Save
    out_cols = [
        "game_id","game_date","home_team","away_team",
        "home_runs_pred","away_runs_pred","total_runs_pred","run_diff_pred",
        "p_home_win_raw","p_home_win_poisson","p_away_win_poisson",
        "p_home_market_median","p_away_market_median","n_books_ml",
        "home_price_consensus","away_price_consensus",
        "ou_line","ou_over_price","ou_under_price","n_books_ou",
        "p_over","p_under","ou_recommendation",
        "edge_home","edge_away","ou_edge_over","ou_edge_under",
        "ev_home","ev_away","ev_over","ev_under",
        "is_value_ml_home","is_value_ml_away","is_value_ou_over","is_value_ou_under",
    ]
    out = df[out_cols].copy()
    out.insert(0, "as_of_ts", pd.Timestamp.now("UTC").floor("min"))

    with engine.begin() as conn:
        tmp = f"_inf_tmp_{args.date.replace('-','')}"
        out.to_sql(tmp, conn, schema=args.schema,
                   if_exists="replace", index=False, method="multi")
        col_names  = ", ".join(out.columns)
        set_clause = ", ".join(f"{c} = EXCLUDED.{c}"
                               for c in out.columns if c not in ("as_of_ts","game_id"))
        conn.execute(text(f"""
            INSERT INTO {args.schema}.inference_game_predictions ({col_names})
            SELECT {col_names} FROM {args.schema}.{tmp}
            WHERE game_id NOT IN (
                SELECT game_id FROM {args.schema}.games
                WHERE LOWER(COALESCE(status, '')) IN ('final', 'game over', 'completed early')
            )
            ON CONFLICT (as_of_ts, game_id) DO UPDATE SET {set_clause}
        """))
        conn.execute(text(f"DROP TABLE IF EXISTS {args.schema}.{tmp}"))

    print(f"  Saved {len(out)} predictions for {args.date}")

    # Dashboards
    print_all_games(df)
    print_value_bets(df, args.ml_edge_threshold, args.ou_edge_threshold, args.min_run_diff)


if __name__ == "__main__":
    main()