#!/usr/bin/env python3
"""
build_features.py

Computes and materialises all engineered features into features_game.
Safe to run repeatedly — all writes use upsert (UPDATE from temp table).

Run once to backfill, then nightly before inference:
    PG_DSN=... python build_features.py --start 2015-04-01
    PG_DSN=... python build_features.py --date 2025-04-01

New / expanded columns in features_game
──────────────────────────────────────────────────────────────────────────────
EXISTING (unchanged):
  home/away_sp_xwoba_against_90, _k_rate_90, _bb_rate_90, _gb_rate_90, _n_pa_90
  home/away_lineup_xwoba_90, _k_rate_90, _bb_rate_90, _n_pa_90
  home/away_lineup_vs_sp_score
  park_runs_factor

NEW — Group 1: Multiple rolling windows for team offense/defense
  home/away_runs_for_7d, _15d          (short-window team offense)
  home/away_runs_against_7d, _15d      (short-window team defense)
  home/away_win_pct_7d, _15d           (recent momentum)

NEW — Group 2: Absolute scoring environment (not just diffs)
  total_offense_env                    (home_avg_scored_60 + away_avg_scored_60)
  total_defense_env                    (home_avg_allowed_60 + away_avg_allowed_60)

NEW — Group 3: SP recent form (last 3 starts) and rest
  home/away_sp_era_last3               (ERA in last 3 starts)
  home/away_sp_era_last5               (ERA in last 5 starts)          [NEW]
  home/away_sp_era_season              (ERA for full current season)    [NEW]
  home/away_sp_k9_last5                (K/9 in last 5 starts)
  home/away_sp_whip_last5              (WHIP in last 5 starts)         [FIXED - now real]
  home/away_sp_days_rest               (days since last start)

NEW — Group 4: Lineup power metrics
  home/away_lineup_barrel_rate_90      (barrel rate rolling 90d)
  home/away_lineup_hard_hit_rate_90    (hard hit rate rolling 90d)

NEW — Group 5: Current-season rolling park factor (blended with prior)
  park_runs_factor_current             (current season factor, updated weekly)
  park_runs_factor_blended             (blend of prior + current season)

NEW — Group 6: Bullpen workload features                               [NEW]
  home/away_bp_outs_1d                 (bullpen outs yesterday)
  home/away_bp_outs_3d                 (bullpen outs last 3 days)
  home/away_bp_outs_5d                 (bullpen outs last 5 days)
  home/away_bp_hlev_outs_1d            (high-leverage bullpen outs yesterday)
  home/away_bp_hlev_outs_3d            (high-leverage bullpen outs last 3 days)
──────────────────────────────────────────────────────────────────────────────
"""

import os
import argparse
import textwrap
import warnings
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SP_WINDOW_DAYS    = 365
BAT_WINDOW_DAYS   = 365
SP_LAST3_DAYS     = 365    # lookback for "last 3 starts" — catches 3 starts in ~45 days
MIN_SP_PA         = 50
MIN_BAT_PA        = 30
MIN_BAT_PA_POWER  = 50    # higher min for barrel/hard-hit (noisier metrics)
PARK_FACTOR_MIN_GAMES = 50
BLEND_FULL_GAMES  = 60

LEAGUE_AVG_ERA    = 4.50   # fallback when pitcher has no valid starts

PITCH_TYPES = ["ff", "si", "fc", "sl", "cu", "ch", "sp", "fs"]
PITCH_TYPE_MAP = {
    "FF": "ff", "SI": "si", "FC": "fc", "SL": "sl",
    "CU": "cu", "CH": "ch", "SP": "sp", "FS": "fs",
    "KC": "cu", "ST": "sl", "SV": "sl", "CS": "cu",
}

# All columns this script manages — used for schema migration
NEW_COLUMNS = [
    # Existing statcast SP
    "home_sp_xwoba_against_90", "away_sp_xwoba_against_90",
    "home_sp_k_rate_90",        "away_sp_k_rate_90",
    "home_sp_bb_rate_90",       "away_sp_bb_rate_90",
    "home_sp_gb_rate_90",       "away_sp_gb_rate_90",
    "home_sp_n_pa_90",          "away_sp_n_pa_90",
    # Existing statcast lineup
    "home_lineup_xwoba_90",     "away_lineup_xwoba_90",
    "home_lineup_k_rate_90",    "away_lineup_k_rate_90",
    "home_lineup_bb_rate_90",   "away_lineup_bb_rate_90",
    "home_lineup_n_pa_90",      "away_lineup_n_pa_90",
    # Existing matchup score
    "home_lineup_vs_sp_score",  "away_lineup_vs_sp_score",
    # Existing park factor
    "park_runs_factor",
    # Short-window team stats
    "home_runs_for_7d",    "away_runs_for_7d",
    "home_runs_for_15d",   "away_runs_for_15d",
    "home_runs_against_7d","away_runs_against_7d",
    "home_runs_against_15d","away_runs_against_15d",
    "home_win_pct_7d",     "away_win_pct_7d",
    "home_win_pct_15d",    "away_win_pct_15d",
    # Absolute scoring environment
    "total_offense_env",
    "total_defense_env",
    # SP recent form
    "home_sp_era_last3",          "away_sp_era_last3",
    "home_sp_era_last5",          "away_sp_era_last5",
    "home_sp_era_season",         "away_sp_era_season",
    "home_sp_k9_last5",           "away_sp_k9_last5",
    "home_sp_whip_last5",         "away_sp_whip_last5",
    "home_sp_days_rest",          "away_sp_days_rest",
    "home_sp_pitches_last_start", "away_sp_pitches_last_start",  # NEW
    "home_sp_innings_season",     "away_sp_innings_season",       # NEW
    # Team situational
    "home_win_streak",            "away_win_streak",               # NEW
    "home_days_since_last_game",  "away_days_since_last_game",     # NEW
    # Lineup power
    "home_lineup_barrel_rate_90",   "away_lineup_barrel_rate_90",
    "home_lineup_hard_hit_rate_90", "away_lineup_hard_hit_rate_90",
    # Blended park factor
    "park_runs_factor_current",
    "park_runs_factor_blended",
    "league_avg_runs_60d",
    # Bullpen workload  [NEW]
    "home_bp_outs_1d",      "away_bp_outs_1d",
    "home_bp_outs_3d",      "away_bp_outs_3d",
    "home_bp_outs_5d",      "away_bp_outs_5d",
    "home_bp_hlev_outs_1d", "away_bp_hlev_outs_1d",
    "home_bp_hlev_outs_3d", "away_bp_hlev_outs_3d",
]


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------

def ensure_columns(engine, schema: str) -> None:
    with engine.begin() as conn:
        existing = pd.read_sql(
            text("""
                SELECT column_name FROM information_schema.columns
                WHERE table_schema = :s AND table_name = 'features_game'
            """),
            conn, params={"s": schema}
        )["column_name"].tolist()

        added = []
        for col in NEW_COLUMNS:
            if col not in existing:
                conn.execute(text(
                    f"ALTER TABLE {schema}.features_game "
                    f"ADD COLUMN {col} DOUBLE PRECISION"
                ))
                added.append(col)
        if added:
            print(f"  Added {len(added)} new columns: {', '.join(added)}")
        else:
            print("  All columns already exist.")


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_games(engine, schema: str, start: str, end: str) -> pd.DataFrame:
    return pd.read_sql(text(f"""
        SELECT
            g.game_id, g.game_date, g.season,
            g.home_team_id, g.away_team_id,
            g.venue_id,
            g.home_runs, g.away_runs, g.home_win,
            gsp.home_sp_id, gsp.away_sp_id
        FROM {schema}.games g
        LEFT JOIN {schema}.game_starting_pitchers gsp USING (game_id)
        WHERE g.game_date BETWEEN :start AND :end
        ORDER BY g.game_date
    """), engine, params={"start": start, "end": end})


def load_statcast_for_window(engine, schema: str, d0: str, d1: str) -> pd.DataFrame:
    print(f"  Loading statcast {d0} → {d1} ...")
    df = pd.read_sql(text(f"""
        SELECT
            game_date, pitcher, batter,
            pitch_type, events, bb_type,
            estimated_woba_using_speedangle AS xwoba,
            woba_denom,
            launch_speed,
            launch_angle
        FROM {schema}.statcast_pitches
        WHERE game_date BETWEEN :d0 AND :d1
          AND game_date IS NOT NULL
    """), engine, params={"d0": d0, "d1": d1})
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["pitch_type_mapped"] = df["pitch_type"].map(PITCH_TYPE_MAP)
    has_launch = df["launch_speed"].notna() & df["launch_angle"].notna()
    is_fair_bip = df["bb_type"].notna()

    df["is_barrel"] = is_fair_bip & has_launch & (
        ((df["launch_speed"] >= 98) & df["launch_angle"].between(26, 30)) |
        ((df["launch_speed"] >= 99) & df["launch_angle"].between(25, 31)) |
        ((df["launch_speed"] >= 100) & df["launch_angle"].between(24, 33)) |
        ((df["launch_speed"] >= 101) & df["launch_angle"].between(23, 35)) |
        ((df["launch_speed"] >= 102) & df["launch_angle"].between(22, 38)) |
        ((df["launch_speed"] >= 103) & df["launch_angle"].between(19, 41)) |
        ((df["launch_speed"] >= 104) & df["launch_angle"].between(15, 45))
    )
    df["is_hard_hit"] = is_fair_bip & (df["launch_speed"] >= 95)
    df["is_tracked_bip"] = is_fair_bip & has_launch
    df["is_bip"] = is_fair_bip
    return df


def load_lineups_for_window(engine, schema: str, d0: str, d1: str) -> pd.DataFrame:
    return pd.read_sql(text(f"""
        SELECT game_id, game_date, team_id, is_home, batting_order, player_id
        FROM {schema}.game_lineups
        WHERE game_date BETWEEN :d0 AND :d1
          AND batting_order BETWEEN 1 AND 9
        ORDER BY game_id, team_id, batting_order
    """), engine, params={"d0": d0, "d1": d1})


def load_pitcher_pitchmix(engine, schema: str, d0: str, d1: str) -> pd.DataFrame:
    return pd.read_sql(text(f"""
        SELECT pitcher_id, as_of_date, n_pitches,
               pct_ff, pct_si, pct_fc, pct_sl,
               pct_cu, pct_ch, pct_sp, pct_fs
        FROM {schema}.pitcher_pitchmix_rolling
        WHERE as_of_date BETWEEN :d0 AND :d1 AND window_days = 365
    """), engine, params={"d0": d0, "d1": d1})


def load_batter_pitchtype(engine, schema: str, d0: str, d1: str) -> pd.DataFrame:
    return pd.read_sql(text(f"""
        SELECT batter_id, as_of_date, n_pitches,
               skill_ff, skill_si, skill_fc, skill_sl,
               skill_cu, skill_ch, skill_sp, skill_fs
        FROM {schema}.batter_vs_pitchtype_rolling
        WHERE as_of_date BETWEEN :d0 AND :d1 AND window_days = 365
    """), engine, params={"d0": d0, "d1": d1})


def load_team_game_history(engine, schema: str, d0: str, d1: str) -> pd.DataFrame:
    return pd.read_sql(text(f"""
        SELECT game_id, game_date, season,
               home_team_id, away_team_id,
               home_runs, away_runs, home_win
        FROM {schema}.games
        WHERE game_date BETWEEN :d0 AND :d1
          AND home_runs IS NOT NULL
        ORDER BY game_date
    """), engine, params={"d0": d0, "d1": d1})


def load_pitcher_start_history(engine, schema: str, d0: str, d1: str) -> pd.DataFrame:
    """Load pitcher start box scores including new hits/walks columns."""
    return pd.read_sql(text(f"""
        SELECT pitcher_id, game_date, team_id,
               innings_pitched, runs_allowed, outs_pitched,
               hits_allowed, walks_allowed, pitches_thrown
        FROM {schema}.pitcher_starts
        WHERE game_date BETWEEN :d0 AND :d1
          AND innings_pitched > 0          -- exclude pre-game / mid-game placeholders
        ORDER BY pitcher_id, game_date
    """), engine, params={"d0": d0, "d1": d1})


def load_pitcher_appearances_history(engine, schema: str, d0: str, d1: str) -> pd.DataFrame:
    """Load reliever appearances for bullpen workload features."""
    return pd.read_sql(text(f"""
        SELECT pa.pitcher_id, pa.game_id, pa.game_date, pa.team_id,
               pa.outs_pitched, pa.is_high_leverage
        FROM {schema}.pitcher_appearances pa
        WHERE pa.game_date BETWEEN :d0 AND :d1
          AND pa.is_starter = FALSE
          AND pa.outs_pitched IS NOT NULL
          AND pa.outs_pitched > 0
        ORDER BY pa.team_id, pa.game_date
    """), engine, params={"d0": d0, "d1": d1})


# ---------------------------------------------------------------------------
# Group 1: Short-window team stats
# ---------------------------------------------------------------------------

def compute_team_rolling_stats(
    history: pd.DataFrame,
    team_id: int,
    game_date: pd.Timestamp,
    window_days: int,
) -> dict:
    date_min = game_date - pd.Timedelta(days=window_days)
    mask_home = (
        (history["home_team_id"] == team_id) &
        (history["game_date"] >= date_min) &
        (history["game_date"] < game_date)
    )
    mask_away = (
        (history["away_team_id"] == team_id) &
        (history["game_date"] >= date_min) &
        (history["game_date"] < game_date)
    )

    home_g = history[mask_home]
    away_g = history[mask_away]

    runs_for     = list(home_g["home_runs"]) + list(away_g["away_runs"])
    runs_against = list(home_g["away_runs"]) + list(away_g["home_runs"])
    wins         = list(home_g["home_win"].astype(int)) + \
                   list((~away_g["home_win"]).astype(int))

    n = len(runs_for)
    if n == 0:
        return {"runs_for": np.nan, "runs_against": np.nan, "win_pct": np.nan, "n": 0}

    return {
        "runs_for":     float(np.mean(runs_for)),
        "runs_against": float(np.mean(runs_against)),
        "win_pct":      float(np.mean(wins)),
        "n":            n,
    }


# ---------------------------------------------------------------------------
# Group 2: Absolute scoring environment
# ---------------------------------------------------------------------------

def compute_scoring_environment(row: dict) -> dict:
    h60  = row.get("home_avg_runs_scored_60", np.nan)
    a60  = row.get("away_avg_runs_scored_60", np.nan)
    hd60 = row.get("home_avg_runs_allowed_60", np.nan)
    ad60 = row.get("away_avg_runs_allowed_60", np.nan)

    return {
        "total_offense_env": float(h60 + a60)   if pd.notna(h60)  and pd.notna(a60)  else np.nan,
        "total_defense_env": float(hd60 + ad60) if pd.notna(hd60) and pd.notna(ad60) else np.nan,
    }


# ---------------------------------------------------------------------------
# Group 3: SP recent form from pitcher_starts
# ---------------------------------------------------------------------------

def _reg_season_mask(starts: pd.DataFrame) -> pd.Series:
    """Boolean mask: exclude March spring training, keep late March (opening day) onward."""
    month = starts["game_date"].dt.month
    day   = starts["game_date"].dt.day
    return (month != 3) | (day >= 27)


def compute_sp_recent_form(
    starts: pd.DataFrame,
    pitcher_id: int,
    game_date: pd.Timestamp,
) -> dict:
    """
    Compute ERA in last 3 starts, last 5 starts, full current season,
    WHIP in last 5 starts, K/9 in last 5 starts, and days since last start.

    KEY FIX: innings_pitched > 0 filter already applied at load time,
    so no placeholder rows will appear here.
    """
    null = {
        "era_last3":          LEAGUE_AVG_ERA,
        "era_last5":          LEAGUE_AVG_ERA,
        "era_season":         LEAGUE_AVG_ERA,
        "whip_last5":         np.nan,
        "k9_last5":           np.nan,
        "days_rest":          np.nan,
        "pitches_last_start": np.nan,
        "innings_season":     np.nan,
    }

    if pd.isna(pitcher_id):
        return null

    reg_mask = _reg_season_mask(starts)
    mask = (
        (starts["pitcher_id"] == int(pitcher_id)) &
        (starts["game_date"] < game_date) &
        reg_mask
    )
    sp = starts[mask].sort_values("game_date")

    if sp.empty:
        return null

    # Days rest — capped at 30
    last_start = sp["game_date"].iloc[-1]
    null["days_rest"] = min(float((game_date - last_start).days), 30.0)

    # Pitches thrown in last start
    if "pitches_thrown" in sp.columns:
        last_pitches = sp["pitches_thrown"].iloc[-1]
        null["pitches_last_start"] = float(last_pitches) if pd.notna(last_pitches) else np.nan

    # Innings pitched in current season
    sp_current_ip = sp[sp["game_date"].dt.year == game_date.year]
    if len(sp_current_ip) > 0:
        total_ip = sp_current_ip["innings_pitched"].sum()
        null["innings_season"] = float(total_ip) if total_ip > 0 else np.nan

    # Helper: compute ERA from a slice, returns LEAGUE_AVG_ERA if insufficient
    def era_from_slice(df, min_ip=3.0):
        ip  = df["innings_pitched"].sum()
        ra  = df["runs_allowed"].sum()
        avg = ip / len(df) if len(df) > 0 else 0
        if ip == 0 or avg < 2.0:      # opener/reliever — skip
            return LEAGUE_AVG_ERA
        if ip >= min_ip:
            return float(ra / ip * 9)
        return LEAGUE_AVG_ERA

    # Helper: compute WHIP from a slice
    def whip_from_slice(df, min_ip=3.0):
        ip = df["innings_pitched"].sum()
        if ip < min_ip:
            return np.nan
        h  = df["hits_allowed"].sum()  if "hits_allowed"  in df.columns else np.nan
        bb = df["walks_allowed"].sum() if "walks_allowed" in df.columns else np.nan
        if pd.isna(h) or pd.isna(bb):
            return np.nan
        return float((h + bb) / ip)

    # Prefer current season for ERA calculations if 2+ starts available
    current_season  = game_date.year
    sp_current      = sp[sp["game_date"].dt.year == current_season]
    sp_for_era      = sp_current if len(sp_current) >= 2 else sp

    # ERA last 3
    null["era_last3"] = era_from_slice(sp_for_era.tail(3), min_ip=3.0)

    # ERA last 5
    null["era_last5"] = era_from_slice(sp_for_era.tail(5), min_ip=3.0)

    # ERA full current season
    if len(sp_current) >= 1:
        null["era_season"] = era_from_slice(sp_current, min_ip=3.0)

    # WHIP last 5 — now real since hits_allowed/walks_allowed are populated
    null["whip_last5"] = whip_from_slice(sp_for_era.tail(5), min_ip=3.0)

    # K/9 last 5 — still approximated from statcast k_rate (filled below in caller)
    null["k9_last5"] = np.nan

    return null


# ---------------------------------------------------------------------------
# Group 3b: SP K/9 from statcast
# ---------------------------------------------------------------------------

def compute_sp_statcast_form(
    statcast: pd.DataFrame,
    pitcher_id: int,
    game_date: pd.Timestamp,
    window_days: int = SP_WINDOW_DAYS,
) -> dict:
    date_min = game_date - pd.Timedelta(days=window_days)
    mask = (
        (statcast["pitcher"] == pitcher_id) &
        (statcast["game_date"] >= date_min) &
        (statcast["game_date"] < game_date)
    )
    sp_df = statcast[mask]

    null = {
        "xwoba_against": np.nan, "k_rate": np.nan,
        "bb_rate": np.nan,       "gb_rate": np.nan,
        "n_pa": 0,
    }

    if sp_df.empty:
        return null

    pa_df = sp_df[sp_df["woba_denom"] == 1]
    n_pa  = len(pa_df)

    if n_pa < MIN_SP_PA:
        null["n_pa"] = n_pa
        return null

    bip = sp_df[sp_df["is_bip"]]
    return {
        "xwoba_against": float(pa_df["xwoba"].mean()) if pa_df["xwoba"].notna().any() else np.nan,
        "k_rate":        float((pa_df["events"] == "strikeout").mean()),
        "bb_rate":       float((pa_df["events"] == "walk").mean()),
        "gb_rate":       float((bip["bb_type"] == "ground_ball").mean()) if len(bip) > 0 else np.nan,
        "n_pa":          n_pa,
    }


# ---------------------------------------------------------------------------
# Group 4: Lineup power metrics
# ---------------------------------------------------------------------------

def compute_lineup_power_metrics(
    statcast: pd.DataFrame,
    batter_ids: list,
    game_date: pd.Timestamp,
    window_days: int = BAT_WINDOW_DAYS,
) -> dict:
    date_min = game_date - pd.Timedelta(days=window_days)
    barrel_rates, hard_hit_rates = [], []

    for bid in batter_ids:
        mask = (
            (statcast["batter"] == bid) &
            (statcast["game_date"] >= date_min) &
            (statcast["game_date"] < game_date) &
            statcast["is_tracked_bip"]
        )
        b_df = statcast[mask]
        if len(b_df) < MIN_BAT_PA_POWER:
            continue
        barrel_rates.append(float(b_df["is_barrel"].mean()))
        hard_hit_rates.append(float(b_df["is_hard_hit"].mean()))

    return {
        "barrel_rate":   float(np.mean(barrel_rates))   if barrel_rates   else np.nan,
        "hard_hit_rate": float(np.mean(hard_hit_rates)) if hard_hit_rates else np.nan,
    }


# ---------------------------------------------------------------------------
# Existing: lineup quality + matchup score
# ---------------------------------------------------------------------------

def compute_lineup_rolling_metrics(
    statcast: pd.DataFrame,
    batter_ids: list,
    game_date: pd.Timestamp,
    window_days: int = BAT_WINDOW_DAYS,
) -> dict:
    date_min = game_date - pd.Timedelta(days=window_days)
    stats, total_pa = [], 0

    for bid in batter_ids:
        mask = (
            (statcast["batter"] == bid) &
            (statcast["game_date"] >= date_min) &
            (statcast["game_date"] < game_date)
        )
        b_df = statcast[mask]
        if b_df.empty:
            continue
        pa_df  = b_df[b_df["woba_denom"] == 1]
        n_pa   = len(pa_df)
        total_pa += n_pa
        if n_pa < MIN_BAT_PA:
            continue
        stats.append({
            "xwoba":   pa_df["xwoba"].mean(),
            "k_rate":  (pa_df["events"] == "strikeout").mean(),
            "bb_rate": (pa_df["events"] == "walk").mean(),
        })

    if not stats:
        return {"xwoba": np.nan, "k_rate": np.nan, "bb_rate": np.nan, "n_pa": total_pa}

    return {
        "xwoba":   float(np.mean([s["xwoba"]   for s in stats])),
        "k_rate":  float(np.mean([s["k_rate"]  for s in stats])),
        "bb_rate": float(np.mean([s["bb_rate"] for s in stats])),
        "n_pa":    total_pa,
    }


def compute_matchup_score(
    batter_ids: list,
    game_date: pd.Timestamp,
    sp_id: int,
    batter_pitchtype: pd.DataFrame,
    pitcher_pitchmix: pd.DataFrame,
) -> float:
    sp_mix = pitcher_pitchmix[
        (pitcher_pitchmix["pitcher_id"] == sp_id) &
        (pitcher_pitchmix["as_of_date"] < game_date)
    ].sort_values("as_of_date").tail(1)

    if sp_mix.empty:
        return np.nan

    sp_mix  = sp_mix.iloc[0]
    mix_vec = np.array([sp_mix.get(f"pct_{pt}", 0.0) or 0.0 for pt in PITCH_TYPES], dtype=float)
    mix_sum = mix_vec.sum()
    if mix_sum < 0.5:
        return np.nan
    mix_vec /= mix_sum

    scores = []
    for bid in batter_ids:
        b = batter_pitchtype[
            (batter_pitchtype["batter_id"] == bid) &
            (batter_pitchtype["as_of_date"] < game_date)
        ].sort_values("as_of_date").tail(1)

        if b.empty or b.iloc[0].get("n_pitches", 0) < MIN_BAT_PA:
            continue

        skill_vec = np.array([b.iloc[0].get(f"skill_{pt}", np.nan) or np.nan
                               for pt in PITCH_TYPES], dtype=float)
        active = (mix_vec > 0.05) & np.isfinite(skill_vec)
        if active.sum() == 0:
            continue
        scores.append(float(np.sum(mix_vec[active] * skill_vec[active])))

    return float(np.mean(scores)) if scores else np.nan


# ---------------------------------------------------------------------------
# Group 6: Bullpen workload features [NEW]
# ---------------------------------------------------------------------------

def compute_bullpen_workload(
    appearances: pd.DataFrame,
    team_id: int,
    game_date: pd.Timestamp,
) -> dict:
    """
    Compute bullpen outs over 1d, 3d, 5d windows strictly before game_date.
    Also computes high-leverage outs for 1d and 3d.
    """
    null = {
        "bp_outs_1d":      0,
        "bp_outs_3d":      0,
        "bp_outs_5d":      0,
        "bp_hlev_outs_1d": 0,
        "bp_hlev_outs_3d": 0,
    }

    team_apps = appearances[appearances["team_id"] == team_id]
    if team_apps.empty:
        return null

    for days, key, hlev_key in [
        (1, "bp_outs_1d",  "bp_hlev_outs_1d"),
        (3, "bp_outs_3d",  "bp_hlev_outs_3d"),
        (5, "bp_outs_5d",  None),
    ]:
        date_min = game_date - pd.Timedelta(days=days)
        window = team_apps[
            (team_apps["game_date"] >= date_min) &
            (team_apps["game_date"] < game_date)
        ]
        null[key] = int(window["outs_pitched"].sum())

        if hlev_key is not None:
            hlev = window[window["is_high_leverage"] == True]
            null[hlev_key] = int(hlev["outs_pitched"].sum())

    return null


# ---------------------------------------------------------------------------
# Group 7: Team situational features [NEW]
# ---------------------------------------------------------------------------

def compute_team_situational(
    history: pd.DataFrame,
    team_id: int,
    game_date: pd.Timestamp,
) -> dict:
    """
    Compute win streak and days since last game for a team, strictly before game_date.
    Win streak: positive = winning streak, negative = losing streak.
    """
    null = {"win_streak": 0, "days_since_last_game": np.nan}

    mask_home = (history["home_team_id"] == team_id) & (history["game_date"] < game_date)
    mask_away = (history["away_team_id"] == team_id) & (history["game_date"] < game_date)

    home_g = history[mask_home][["game_date", "home_win"]].rename(columns={"home_win": "win"})
    away_g = history[mask_away][["game_date", "home_win"]].copy()
    away_g["win"] = ~away_g["home_win"]
    away_g = away_g[["game_date", "win"]]

    team_games = pd.concat([home_g, away_g]).sort_values("game_date")
    if team_games.empty:
        return null

    # Days since last game
    last_game_date = team_games["game_date"].iloc[-1]
    null["days_since_last_game"] = float((game_date - last_game_date).days)

    # Win streak — count consecutive wins/losses from most recent game backward
    results = team_games["win"].tolist()
    if not results:
        return null

    last = results[-1]
    streak = 1 if last else -1
    for r in reversed(results[:-1]):
        if r == last:
            streak += (1 if last else -1)
        else:
            break

    null["win_streak"] = streak
    return null


def upsert_league_avg_runs(engine, schema: str, start: str, end: str) -> None:
    print("  Computing league average runs (60d rolling)...")

    df = pd.read_sql(text(f"""
        SELECT
            g2.game_id,
            g2.game_date,
            AVG((g1.home_runs + g1.away_runs) / 2.0) AS league_avg_runs_60d
        FROM {schema}.games g2
        JOIN {schema}.games g1
            ON g1.game_date BETWEEN g2.game_date - INTERVAL '60 days'
                                AND g2.game_date - INTERVAL '1 day'
            AND g1.home_runs IS NOT NULL
        WHERE g2.game_date BETWEEN :start AND :end
        GROUP BY g2.game_id, g2.game_date
        HAVING COUNT(g1.game_id) >= 50
    """), engine, params={"start": start, "end": end})

    if df.empty:
        print("  No games to update for league avg runs.")
        return

    with engine.begin() as conn:
        tmp = "_league_avg_tmp"
        df[["game_id", "league_avg_runs_60d"]].to_sql(
            tmp, conn, schema=schema,
            if_exists="replace", index=False, method="multi"
        )
        conn.execute(text(f"""
            UPDATE {schema}.features_game AS t
            SET league_avg_runs_60d = s.league_avg_runs_60d
            FROM {schema}.{tmp} AS s
            WHERE t.game_id = s.game_id
        """))
        conn.execute(text(f"DROP TABLE IF EXISTS {schema}.{tmp}"))

    print(f"  League avg runs upserted for {len(df)} games.")


# ---------------------------------------------------------------------------
# Group 5: Park factor
# ---------------------------------------------------------------------------

PARK_FACTOR_SEASONS = 3

def compute_prior_park_factors(engine, schema: str, ref_season: int) -> dict:
    df = pd.read_sql(text(f"""
        SELECT venue_id,
               AVG(home_runs + away_runs) AS park_avg,
               COUNT(*)                   AS n
        FROM {schema}.games
        WHERE season BETWEEN :s0 AND :s1
          AND home_runs IS NOT NULL
          AND venue_id IS NOT NULL
        GROUP BY venue_id
        HAVING COUNT(*) >= :min_g
    """), engine, params={
        "s0":    ref_season - PARK_FACTOR_SEASONS,
        "s1":    ref_season - 1,
        "min_g": PARK_FACTOR_MIN_GAMES,
    })
    if df.empty:
        return {}
    league_avg = df["park_avg"].mean()
    return {int(r["venue_id"]): float(r["park_avg"]) / league_avg for _, r in df.iterrows()}


def compute_current_season_park_factors(engine, schema: str, ref_season: int,
                                        as_of_date: str) -> dict:
    df = pd.read_sql(text(f"""
        SELECT venue_id,
               AVG(home_runs + away_runs) AS park_avg,
               COUNT(*)                   AS n
        FROM {schema}.games
        WHERE season = :season
          AND game_date < :d
          AND home_runs IS NOT NULL
          AND venue_id IS NOT NULL
        GROUP BY venue_id
        HAVING COUNT(*) >= :min_g
    """), engine, params={
        "season": ref_season,
        "d":      as_of_date,
        "min_g":  PARK_FACTOR_MIN_GAMES,
    })
    if df.empty:
        return {}, {}

    league_avg = df["park_avg"].mean()
    factors  = {int(r["venue_id"]): float(r["park_avg"]) / league_avg for _, r in df.iterrows()}
    n_games  = {int(r["venue_id"]): int(r["n"]) for _, r in df.iterrows()}
    return factors, n_games


def upsert_park_factors(engine, schema: str, start: str, end: str) -> None:
    games = pd.read_sql(text(f"""
        SELECT game_id, game_date, season, venue_id
        FROM {schema}.games
        WHERE game_date BETWEEN :s AND :e
          AND venue_id IS NOT NULL
        ORDER BY game_date
    """), engine, params={"s": start, "e": end})

    if games.empty:
        return

    games["game_date"] = pd.to_datetime(games["game_date"])

    prior_cache   = {}
    current_cache = {}
    rows = []

    for _, g in games.iterrows():
        season   = int(g["season"])
        gdate    = g["game_date"]
        date_str = gdate.strftime("%Y-%m-%d")
        vid      = int(g["venue_id"])

        if season not in prior_cache:
            prior_cache[season] = compute_prior_park_factors(engine, schema, season)
        prior_factor = prior_cache[season].get(vid, 1.0)

        week_key = (season, gdate.strftime("%Y-W%U"))
        if week_key not in current_cache:
            current_cache[week_key] = compute_current_season_park_factors(
                engine, schema, season, date_str
            )
        curr_factors, curr_n = current_cache[week_key]

        curr_factor  = curr_factors.get(vid, np.nan)
        curr_n_games = curr_n.get(vid, 0) if isinstance(curr_n, dict) else 0

        blend_weight = min(curr_n_games / BLEND_FULL_GAMES, 1.0) if curr_n_games > 0 else 0.0

        if pd.notna(curr_factor) and blend_weight > 0:
            blended = blend_weight * curr_factor + (1.0 - blend_weight) * prior_factor
        else:
            blended = prior_factor

        rows.append({
            "game_id":                  int(g["game_id"]),
            "park_runs_factor":         prior_factor,
            "park_runs_factor_current": float(curr_factor) if pd.notna(curr_factor) else np.nan,
            "park_runs_factor_blended": float(blended),
        })

    df_out = pd.DataFrame(rows)
    with engine.begin() as conn:
        tmp = "_park_tmp"
        df_out.to_sql(tmp, conn, schema=schema,
                      if_exists="replace", index=False, method="multi")
        conn.execute(text(f"""
            UPDATE {schema}.features_game AS t
            SET park_runs_factor         = s.park_runs_factor,
                park_runs_factor_current = s.park_runs_factor_current,
                park_runs_factor_blended = s.park_runs_factor_blended
            FROM {schema}.{tmp} AS s
            WHERE t.game_id = s.game_id
        """))
        conn.execute(text(f"DROP TABLE IF EXISTS {schema}.{tmp}"))

    print(f"  Park factors upserted for {len(rows)} games.")


# ---------------------------------------------------------------------------
# Upsert features
# ---------------------------------------------------------------------------

def upsert_features(engine, schema: str, rows: list) -> None:
    if not rows:
        return
    df   = pd.DataFrame(rows)
    cols = [c for c in df.columns if c != "game_id"]
    set_clause = ", ".join(f"{c} = s.{c}" for c in cols)

    with engine.begin() as conn:
        df.to_sql("_feat_tmp", conn, schema=schema,
                  if_exists="replace", index=False, method="multi")
        conn.execute(text(f"""
            UPDATE {schema}.features_game AS t
            SET {set_clause}
            FROM {schema}._feat_tmp AS s
            WHERE t.game_id = s.game_id
        """))
        conn.execute(text(f"DROP TABLE IF EXISTS {schema}._feat_tmp"))

    print(f"  Upserted {len(rows)} rows.")


# ---------------------------------------------------------------------------
# Team rolling stats (SQL window functions)
# ---------------------------------------------------------------------------

def upsert_team_rolling_stats(engine, schema: str, start: str, end: str) -> None:
    print("  Computing team rolling stats (30d, 60d)...")

    # Process one year at a time to avoid temp_file_limit on Cloud SQL
    start_dt = pd.to_datetime(start)
    end_dt   = pd.to_datetime(end)
    years    = range(start_dt.year, end_dt.year + 1)

    for year in years:
        year_start = max(pd.Timestamp(f"{year}-01-01"), start_dt).strftime("%Y-%m-%d")
        year_end   = min(pd.Timestamp(f"{year}-12-31"), end_dt).strftime("%Y-%m-%d")
        print(f"    Rolling stats: {year_start} → {year_end}")

        with engine.begin() as conn:
            conn.execute(text(f"""
                WITH team_games AS (
                    SELECT
                        game_id, game_date, season,
                        home_team_id AS team_id,
                        home_runs AS runs_for,
                        away_runs AS runs_against,
                        CASE WHEN home_runs > away_runs THEN 1.0 ELSE 0.0 END AS win
                    FROM {schema}.games
                    WHERE home_runs IS NOT NULL AND away_runs IS NOT NULL

                    UNION ALL

                    SELECT
                        game_id, game_date, season,
                        away_team_id AS team_id,
                        away_runs AS runs_for,
                        home_runs AS runs_against,
                        CASE WHEN away_runs > home_runs THEN 1.0 ELSE 0.0 END AS win
                    FROM {schema}.games
                    WHERE home_runs IS NOT NULL AND away_runs IS NOT NULL
                ),
                rolling AS (
                    SELECT
                        game_id, team_id,
                        AVG(win) OVER (
                            PARTITION BY team_id, season
                            ORDER BY game_date, game_id
                            ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
                        ) AS win_pct_30,
                        AVG(runs_for) OVER (
                            PARTITION BY team_id, season
                            ORDER BY game_date, game_id
                            ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
                        ) AS runs_for_30,
                        AVG(runs_against) OVER (
                            PARTITION BY team_id, season
                            ORDER BY game_date, game_id
                            ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
                        ) AS runs_against_30,
                        AVG(runs_for) OVER (
                            PARTITION BY team_id
                            ORDER BY game_date, game_id
                            ROWS BETWEEN 60 PRECEDING AND 1 PRECEDING
                        ) AS avg_runs_scored_60,
                        AVG(runs_against) OVER (
                            PARTITION BY team_id
                            ORDER BY game_date, game_id
                            ROWS BETWEEN 60 PRECEDING AND 1 PRECEDING
                        ) AS avg_runs_allowed_60
                    FROM team_games
                ),
                target_games AS (
                    SELECT g.game_id, g.game_date, g.home_team_id, g.away_team_id
                    FROM {schema}.games g
                    WHERE g.game_date BETWEEN :start AND :end
                ),
                prior_home AS (
                    SELECT DISTINCT ON (tg.game_id)
                        tg.game_id,
                        r.win_pct_30 AS home_win_pct_30,
                        r.runs_for_30 AS home_runs_for_30,
                        r.runs_against_30 AS home_runs_against_30,
                        r.avg_runs_scored_60 AS home_avg_runs_scored_60,
                        r.avg_runs_allowed_60 AS home_avg_runs_allowed_60
                    FROM target_games tg
                    JOIN team_games tg2 ON tg2.team_id = tg.home_team_id
                        AND tg2.game_date < tg.game_date
                    JOIN rolling r ON r.game_id = tg2.game_id AND r.team_id = tg.home_team_id
                    ORDER BY tg.game_id, tg2.game_date DESC, tg2.game_id DESC
                ),
                prior_away AS (
                    SELECT DISTINCT ON (tg.game_id)
                        tg.game_id,
                        r.win_pct_30 AS away_win_pct_30,
                        r.runs_for_30 AS away_runs_for_30,
                        r.runs_against_30 AS away_runs_against_30,
                        r.avg_runs_scored_60 AS away_avg_runs_scored_60,
                        r.avg_runs_allowed_60 AS away_avg_runs_allowed_60
                    FROM target_games tg
                    JOIN team_games tg2 ON tg2.team_id = tg.away_team_id
                        AND tg2.game_date < tg.game_date
                    JOIN rolling r ON r.game_id = tg2.game_id AND r.team_id = tg.away_team_id
                    ORDER BY tg.game_id, tg2.game_date DESC, tg2.game_id DESC
                )
                UPDATE {schema}.features_game AS fg
                SET
                    home_win_pct_30          = ph.home_win_pct_30,
                    home_runs_for_30         = ph.home_runs_for_30,
                    home_runs_against_30     = ph.home_runs_against_30,
                    home_avg_runs_scored_60  = ph.home_avg_runs_scored_60,
                    home_avg_runs_allowed_60 = ph.home_avg_runs_allowed_60,
                    away_win_pct_30          = pa.away_win_pct_30,
                    away_runs_for_30         = pa.away_runs_for_30,
                    away_runs_against_30     = pa.away_runs_against_30,
                    away_avg_runs_scored_60  = pa.away_avg_runs_scored_60,
                    away_avg_runs_allowed_60 = pa.away_avg_runs_allowed_60
                FROM prior_home ph
                JOIN prior_away pa ON pa.game_id = ph.game_id
                WHERE fg.game_id = ph.game_id
            """), {"start": year_start, "end": year_end})

    print(f"  Team rolling stats upserted for {start} → {end}")


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def process_date_range(engine, schema: str, start_date: str, end_date: str,
                       batch_days: int = 30) -> None:
    games = load_games(engine, schema, start_date, end_date)
    if games.empty:
        print(f"No games found between {start_date} and {end_date}.")
        return

    games["game_date"] = pd.to_datetime(games["game_date"])
    print(f"Found {len(games)} games to process ({start_date} → {end_date})")

    all_dates = sorted(games["game_date"].unique())
    batches   = [all_dates[i:i + batch_days] for i in range(0, len(all_dates), batch_days)]

    for batch_dates in batches:
        batch_start = batch_dates[0]
        batch_end   = batch_dates[-1]
        print(f"\nBatch {batch_start.date()} → {batch_end.date()}")

        lookback = batch_start - pd.Timedelta(days=max(SP_WINDOW_DAYS, BAT_WINDOW_DAYS, 60) + 1)
        d0 = lookback.strftime("%Y-%m-%d")
        d1 = batch_end.strftime("%Y-%m-%d")

        statcast = load_statcast_for_window(engine, schema, d0, d1)
        print(f"  Statcast rows: {len(statcast):,}")

        lineups = load_lineups_for_window(engine, schema,
                      batch_start.strftime("%Y-%m-%d"), d1)
        lineups["game_date"] = pd.to_datetime(lineups["game_date"])

        pitchmix = load_pitcher_pitchmix(engine, schema, d0, d1)
        pitchmix["as_of_date"] = pd.to_datetime(pitchmix["as_of_date"])

        batter_pt = load_batter_pitchtype(engine, schema, d0, d1)
        batter_pt["as_of_date"] = pd.to_datetime(batter_pt["as_of_date"])

        hist_lookback = batch_start - pd.Timedelta(days=65)
        team_history  = load_team_game_history(engine, schema,
                            hist_lookback.strftime("%Y-%m-%d"), d1)
        team_history["game_date"] = pd.to_datetime(team_history["game_date"])

        # SP starts — lookback 365d, innings_pitched > 0 already filtered in loader
        sp_lookback = batch_start - pd.Timedelta(days=SP_LAST3_DAYS)
        sp_starts   = load_pitcher_start_history(engine, schema,
                          sp_lookback.strftime("%Y-%m-%d"), d1)
        sp_starts["game_date"] = pd.to_datetime(sp_starts["game_date"])

        # Bullpen appearances — 5 day lookback is enough for workload features
        bp_lookback  = batch_start - pd.Timedelta(days=6)
        bp_apps      = load_pitcher_appearances_history(engine, schema,
                           bp_lookback.strftime("%Y-%m-%d"), d1)
        bp_apps["game_date"] = pd.to_datetime(bp_apps["game_date"])

        fg_batch = pd.read_sql(text(f"""
            SELECT game_id,
                   home_avg_runs_scored_60, home_avg_runs_allowed_60,
                   away_avg_runs_scored_60, away_avg_runs_allowed_60
            FROM {schema}.features_game
            WHERE game_id IN (
                SELECT game_id FROM {schema}.games
                WHERE game_date BETWEEN :s AND :e
            )
        """), engine, params={
            "s": batch_start.strftime("%Y-%m-%d"),
            "e": batch_end.strftime("%Y-%m-%d"),
        })
        fg_lookup = fg_batch.set_index("game_id").to_dict("index")

        lineup_lookup = {}
        for _, row in lineups.iterrows():
            gid  = row["game_id"]
            side = "home" if row["is_home"] else "away"
            lineup_lookup.setdefault(gid, {"home": [], "away": []})
            lineup_lookup[gid][side].append(row["player_id"])

        batch_games = games[
            (games["game_date"] >= batch_start) &
            (games["game_date"] <= batch_end)
        ]

        results = []
        for _, game in tqdm(batch_games.iterrows(), total=len(batch_games),
                            desc="  Computing features"):
            gid      = int(game["game_id"])
            gdate    = game["game_date"]
            home_sp  = game["home_sp_id"]
            away_sp  = game["away_sp_id"]
            home_tid = int(game["home_team_id"])
            away_tid = int(game["away_team_id"])
            home_bats = lineup_lookup.get(gid, {}).get("home", [])
            away_bats = lineup_lookup.get(gid, {}).get("away", [])

            row = {"game_id": gid}

            # ── SP statcast metrics ────────────────────────────────────────
            for side, sp_id in [("home", home_sp), ("away", away_sp)]:
                if pd.isna(sp_id) or sp_id is None:
                    for m in ["xwoba_against","k_rate","bb_rate","gb_rate","n_pa"]:
                        row[f"{side}_sp_{m}_90"] = np.nan
                    continue
                m = compute_sp_statcast_form(statcast, int(sp_id), gdate)
                row[f"{side}_sp_xwoba_against_90"] = m["xwoba_against"]
                row[f"{side}_sp_k_rate_90"]        = m["k_rate"]
                row[f"{side}_sp_bb_rate_90"]       = m["bb_rate"]
                row[f"{side}_sp_gb_rate_90"]       = m["gb_rate"]
                row[f"{side}_sp_n_pa_90"]          = m["n_pa"]

            # ── Lineup quality + matchup ───────────────────────────────────
            for side, bids, opp_sp in [
                ("home", home_bats, away_sp),
                ("away", away_bats, home_sp),
            ]:
                if not bids:
                    for m in ["xwoba","k_rate","bb_rate","n_pa"]:
                        row[f"{side}_lineup_{m}_90"] = np.nan
                    row[f"{side}_lineup_vs_sp_score"] = np.nan
                    continue
                lm = compute_lineup_rolling_metrics(statcast, bids, gdate)
                row[f"{side}_lineup_xwoba_90"]   = lm["xwoba"]
                row[f"{side}_lineup_k_rate_90"]  = lm["k_rate"]
                row[f"{side}_lineup_bb_rate_90"] = lm["bb_rate"]
                row[f"{side}_lineup_n_pa_90"]    = lm["n_pa"]

                if not pd.isna(opp_sp):
                    row[f"{side}_lineup_vs_sp_score"] = compute_matchup_score(
                        bids, gdate, int(opp_sp), batter_pt, pitchmix)
                else:
                    row[f"{side}_lineup_vs_sp_score"] = np.nan

            # ── Group 1: Short-window team stats ───────────────────────────
            for side, tid in [("home", home_tid), ("away", away_tid)]:
                for w in [7, 15]:
                    m = compute_team_rolling_stats(team_history, tid, gdate, w)
                    row[f"{side}_runs_for_{w}d"]     = m["runs_for"]
                    row[f"{side}_runs_against_{w}d"] = m["runs_against"]
                    row[f"{side}_win_pct_{w}d"]      = m["win_pct"]

            # ── Group 2: Absolute scoring environment ──────────────────────
            fg = fg_lookup.get(gid, {})
            env = compute_scoring_environment({
                "home_avg_runs_scored_60":  fg.get("home_avg_runs_scored_60"),
                "away_avg_runs_scored_60":  fg.get("away_avg_runs_scored_60"),
                "home_avg_runs_allowed_60": fg.get("home_avg_runs_allowed_60"),
                "away_avg_runs_allowed_60": fg.get("away_avg_runs_allowed_60"),
            })
            row["total_offense_env"] = env["total_offense_env"]
            row["total_defense_env"] = env["total_defense_env"]

            # ── Group 3: SP recent form ────────────────────────────────────
            for side, sp_id in [("home", home_sp), ("away", away_sp)]:
                if pd.isna(sp_id) or sp_id is None:
                    row[f"{side}_sp_era_last3"]          = np.nan
                    row[f"{side}_sp_era_last5"]          = np.nan
                    row[f"{side}_sp_era_season"]         = np.nan
                    row[f"{side}_sp_k9_last5"]           = np.nan
                    row[f"{side}_sp_whip_last5"]         = np.nan
                    row[f"{side}_sp_days_rest"]          = np.nan
                    row[f"{side}_sp_pitches_last_start"] = np.nan
                    row[f"{side}_sp_innings_season"]     = np.nan
                    continue

                form = compute_sp_recent_form(sp_starts, int(sp_id), gdate)
                row[f"{side}_sp_era_last3"]          = form["era_last3"]
                row[f"{side}_sp_era_last5"]          = form["era_last5"]
                row[f"{side}_sp_era_season"]         = form["era_season"]
                row[f"{side}_sp_whip_last5"]         = form["whip_last5"]
                row[f"{side}_sp_days_rest"]          = form["days_rest"]
                row[f"{side}_sp_pitches_last_start"] = form["pitches_last_start"]
                row[f"{side}_sp_innings_season"]     = form["innings_season"]

                # K/9 approximated from statcast k_rate * 27
                k_rate = row.get(f"{side}_sp_k_rate_90")
                row[f"{side}_sp_k9_last5"] = float(k_rate * 27) if pd.notna(k_rate) else np.nan

            # ── Group 4: Lineup power ──────────────────────────────────────
            for side, bids in [("home", home_bats), ("away", away_bats)]:
                if not bids:
                    row[f"{side}_lineup_barrel_rate_90"]   = np.nan
                    row[f"{side}_lineup_hard_hit_rate_90"] = np.nan
                    continue
                pm = compute_lineup_power_metrics(statcast, bids, gdate)
                row[f"{side}_lineup_barrel_rate_90"]   = pm["barrel_rate"]
                row[f"{side}_lineup_hard_hit_rate_90"] = pm["hard_hit_rate"]

            # ── Group 6: Bullpen workload ──────────────────────────────────
            for side, tid in [("home", home_tid), ("away", away_tid)]:
                bp = compute_bullpen_workload(bp_apps, tid, gdate)
                row[f"{side}_bp_outs_1d"]      = bp["bp_outs_1d"]
                row[f"{side}_bp_outs_3d"]      = bp["bp_outs_3d"]
                row[f"{side}_bp_outs_5d"]      = bp["bp_outs_5d"]
                row[f"{side}_bp_hlev_outs_1d"] = bp["bp_hlev_outs_1d"]
                row[f"{side}_bp_hlev_outs_3d"] = bp["bp_hlev_outs_3d"]

            # ── Group 7: Team situational ──────────────────────────────────
            for side, tid in [("home", home_tid), ("away", away_tid)]:
                sit = compute_team_situational(team_history, tid, gdate)
                row[f"{side}_win_streak"]           = sit["win_streak"]
                row[f"{side}_days_since_last_game"] = sit["days_since_last_game"]

            results.append(row)

        upsert_features(engine, schema, results)

    print("\nDone.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Materialise engineered features into features_game.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python build_features1.py                          # full backfill 2015→today
          python build_features1.py --start 2024-01-01      # from a specific date
          python build_features1.py --date 2025-04-01       # single date (nightly cron)
        """),
    )
    ap.add_argument("--schema",        default="public")
    ap.add_argument("--start",         default="2015-04-01")
    ap.add_argument("--end",           default=None)
    ap.add_argument("--date",          default=None)
    ap.add_argument("--batch-days",    type=int, default=30)
    ap.add_argument("--skip-statcast", action="store_true",
                    help="Skip statcast computation (park factors and team stats only)")
    ap.add_argument("--statcast-only", action="store_true",
                    help="Only statcast SP/lineup block (skip park/league/team rolling)")
    args = ap.parse_args()

    pg_dsn = os.getenv("PG_DSN")
    if not pg_dsn:
        raise RuntimeError("PG_DSN env var is required.")

    engine = create_engine(pg_dsn, pool_pre_ping=True)

    if args.date:
        start_date = end_date = args.date
    else:
        start_date = args.start
        end_date   = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")

    print(f"Schema: {args.schema}  Range: {start_date} → {end_date}")

    print("\nEnsuring columns exist...")
    ensure_columns(engine, args.schema)

    if args.statcast_only:
        print("\nStatcast-only mode — SP/lineup features for date range...")
        process_date_range(engine, args.schema, start_date, end_date,
                           batch_days=args.batch_days)
    else:
        print("\nComputing park factors...")
        upsert_park_factors(engine, args.schema, start_date, end_date)

        print("\nComputing league average runs...")
        upsert_league_avg_runs(engine, args.schema, start_date, end_date)

        print("\nComputing team rolling stats (30d, 60d)...")
        upsert_team_rolling_stats(engine, args.schema, start_date, end_date)

        if not args.skip_statcast:
            process_date_range(engine, args.schema, start_date, end_date,
                               batch_days=args.batch_days)
        else:
            print("Skipping statcast computation (--skip-statcast).")

    print("\nAll done.")


if __name__ == "__main__":
    main()