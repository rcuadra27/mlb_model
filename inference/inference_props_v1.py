"""
inference_props_v1.py — Production inference for player prop models.

Runs after inference_v10.py in the morning pipeline.
Outputs per-batter and per-pitcher prop probabilities for today's games.

Batter props (6 binary classifiers):
  p_hit          probability of recording >= 1 hit
  p_2plus_hits   probability of recording >= 2 hits
  p_hr           probability of recording a home run
  p_k            probability of recording a strikeout
  p_2plus_bases  probability of recording >= 2 total bases
  p_walk         probability of recording a walk

Pitcher props (Poisson K count model):
  lambda_k       expected strikeout count
  p_k0..p_k10   probability of exactly k strikeouts
  p_k10plus      probability of 10+ strikeouts
  p_over_N_5     probability of over N.5 strikeouts (N=0..9)

Writes to:
  public.player_prop_predictions   (batter rows)
  public.pitcher_prop_predictions  (SP rows)

Usage (add to run_daily.sh after inference_v10_total.py):
  python inference/inference_props_v1.py --date $TODAY \\
      --batter-model artifacts/props_v1_expanded.joblib \\
      --pitcher-model artifacts/pitcher_props_v1.joblib
"""
import argparse
import os
import sys
import logging
from datetime import date, datetime

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit
from sqlalchemy import create_engine, text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

try:
    from inference.pitcher_extras_inference import (
        ER_THRESHOLDS,
        HITS_THRESHOLDS,
        PITCHER_EXTRA_COLUMNS,
        WALKS_THRESHOLDS,
        run_extra_pitcher_model,
    )
except ImportError:
    from pitcher_extras_inference import (
        ER_THRESHOLDS,
        HITS_THRESHOLDS,
        PITCHER_EXTRA_COLUMNS,
        WALKS_THRESHOLDS,
        run_extra_pitcher_model,
    )

try:
    from inference.lineup_utils import confirmed_lineup_game_ids
except ImportError:
    from lineup_utils import confirmed_lineup_game_ids

class PoissonRegressor:
    """Poisson regression via scipy MLE — must match train_pitcher_props_v1.py exactly."""
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        n, p = X.shape
        w0 = np.zeros(p + 1)
        def neg_log_likelihood(w):
            intercept, coef = w[0], w[1:]
            eta = intercept + X @ coef
            lam = np.exp(np.clip(eta, -10, 10))
            nll = -(y * np.log(lam + 1e-10) - lam).sum()
            reg = 0.5 * self.alpha * (coef ** 2).sum()
            return nll + reg
        def grad(w):
            intercept, coef = w[0], w[1:]
            eta = intercept + X @ coef
            lam = np.exp(np.clip(eta, -10, 10))
            resid = lam - y
            return np.concatenate([[resid.sum()], X.T @ resid + self.alpha * coef])
        result = minimize(neg_log_likelihood, w0, jac=grad, method='L-BFGS-B',
                         options={'maxiter': 1000, 'ftol': 1e-9})
        self.intercept_ = result.x[0]
        self.coef_ = result.x[1:]
        return self

    def predict_lambda(self, X):
        eta = self.intercept_ + X @ self.coef_
        return np.exp(np.clip(eta, -10, 10))


# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULTS = {
    "batter_xwoba_season":      0.320,
    "batter_k_rate_season":     0.230,
    "batter_hr_rate_season":    0.036,
    "batter_hit_rate_30d":      0.245,
    "batter_iso_season":        0.080,
    "batter_hard_hit_rate_30d": 0.242,
    "batter_barrel_rate_30d":   0.072,
    "batter_xwoba_30d":         0.320,
    "matchup_score":            0.320,
    "sp_xwoba_against":         0.320,
    "platoon_advantage":        0,
    "sp_fastball_pct":          0.50,
    "batting_order":            5.0,
    "park_hr_factor":           1.0,
    "umpire_k_boost":           0.0,
    "sp_hr_rate_season":        0.035,
    "sp_k_rate_season":         0.230,
    # Pitcher defaults
    "sp_k9_last5":              7.0,
    "sp_xwoba_against_90":      0.320,
    "opp_lineup_k_rate_30d":    0.230,
    "opp_lineup_xwoba_90":      0.320,
    "sp_innings_season":        0.0,
    "park_runs_factor":         1.0,
    "sp_days_rest":             4.0,
    "sp_bb_rate_season":       0.085,
    "sp_bb9_last5":            3.0,
    "opp_lineup_bb_rate_30d":  0.085,
    "expected_bf":             22.0,
    "expected_ip":             5.5,
    "sp_er_last5":             2.5,
    "sp_xwoba_against_season": 0.320,
    "umpire_runs_boost":       0.0,
}

PITCH_TYPES = ['ff', 'si', 'fc', 'sl', 'cu', 'ch', 'sp', 'fs']

LEAGUE_HR_PA = DEFAULTS["batter_hr_rate_season"]
LEAGUE_SP_HR_BF = DEFAULTS["sp_hr_rate_season"]
PA_BY_ORDER = {1: 4.3, 2: 4.2, 3: 4.1, 4: 4.0, 5: 3.9, 6: 3.8, 7: 3.7, 8: 3.6, 9: 3.5}


def compute_game_hr_probability(df: pd.DataFrame) -> np.ndarray:
    """
    P(at least one HR this game) from per-PA HR rate, pitcher, park, and lineup spot.
    The ML classifier ranks well but compresses probabilities near the league base rate;
    this formula matches how HR props are quoted (e.g. 25–30% for elite sluggers).
    """
    hr_pa = pd.to_numeric(df.get("batter_hr_rate_season"), errors="coerce").fillna(LEAGUE_HR_PA)
    sp_hr = pd.to_numeric(df.get("sp_hr_rate_season"), errors="coerce").fillna(LEAGUE_SP_HR_BF)
    park = pd.to_numeric(df.get("park_hr_factor"), errors="coerce").fillna(1.0)
    iso = pd.to_numeric(df.get("batter_iso_season"), errors="coerce").fillna(0.080)
    platoon = pd.to_numeric(df.get("platoon_advantage"), errors="coerce").fillna(0)
    order = pd.to_numeric(df.get("batting_order"), errors="coerce").fillna(5).astype(int).clip(1, 9)

    platoon_mult = np.where(platoon > 0, 1.05, np.where(platoon < 0, 0.95, 1.0))
    power_boost = 1.0 + np.clip((iso.to_numpy() - 0.080) * 2.5, -0.15, 0.35)

    p_pa = hr_pa * (sp_hr / LEAGUE_SP_HR_BF) * park * platoon_mult * power_boost
    p_pa = np.clip(p_pa.to_numpy(), 0.006, 0.145)

    n_pa = order.map(PA_BY_ORDER).fillna(3.9).to_numpy()
    return 1.0 - np.power(1.0 - p_pa, n_pa)


def fill_feature_matrix(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Fill missing model inputs with training defaults (not zero)."""
    out = pd.DataFrame(index=df.index)
    for feat in features:
        default = DEFAULTS.get(feat, 0.0)
        if feat in df.columns:
            out[feat] = df[feat].fillna(default)
        else:
            out[feat] = default
    return out


def _patch_logistic_regression(clf):
    """Older pickled sklearn models may lack multi_class attrs required by 1.8+."""
    if not hasattr(clf, "coef_"):
        return clf
    if not hasattr(clf, "multi_class"):
        clf.multi_class = "ovr"
    if not hasattr(clf, "multi_class_"):
        clf.multi_class_ = getattr(clf, "multi_class", "ovr")
    return clf


def predict_binary_proba(clf, X: np.ndarray) -> np.ndarray:
    """Positive-class probability for binary LogisticRegression across sklearn versions."""
    clf = _patch_logistic_regression(clf)
    try:
        return clf.predict_proba(X)[:, 1]
    except AttributeError:
        pass
    if hasattr(clf, "decision_function"):
        return expit(clf.decision_function(X))
    z = np.asarray(clf.intercept_ + X @ clf.coef_.T).ravel()
    return expit(z)


# ── SQL ───────────────────────────────────────────────────────────────────────

# Today's confirmed lineups with matchup context
LINEUP_QUERY = """
SELECT
    gl.game_id,
    gl.game_date,
    gl.player_id                                        AS batter_id,
    gl.player_name                                      AS batter_name,
    gl.batting_order,
    gl.is_home,
    gl.bats                                             AS batter_hand,
    gl.team_id,
    -- Opposing SP
    CASE WHEN gl.is_home THEN gsp.away_sp_id
         ELSE gsp.home_sp_id END                        AS sp_id,
    CASE WHEN gl.is_home THEN gsp.away_sp_name
         ELSE gsp.home_sp_name END                      AS sp_name,
    -- SP xwOBA against (90-day)
    CASE WHEN gl.is_home THEN fg.away_sp_xwoba_against_90
         ELSE fg.home_sp_xwoba_against_90 END           AS sp_xwoba_against,
    -- Park
    COALESCE(fg.park_runs_factor_blended,
             fg.park_runs_factor, 1.0)                  AS park_hr_factor,
    -- Umpire
    fg.umpire_k_rate_boost,
    fg.umpire_bb_rate_boost,
    -- Team IDs for lineup quality lookup
    g.home_team_id,
    g.away_team_id
FROM public.game_lineups gl
JOIN public.game_starting_pitchers gsp ON gsp.game_id = gl.game_id
JOIN public.features_game fg ON fg.game_id = gl.game_id
JOIN public.games g ON g.game_id = gl.game_id
WHERE gl.game_date = :d
ORDER BY gl.game_id, gl.is_home DESC, gl.batting_order;
"""


def load_batter_lineups(engine, schema: str, inference_date: str, all_roster: bool) -> pd.DataFrame:
    """
    Confirmed games: today's nine from game_lineups (real batting_order).
    Unconfirmed games: active-roster proxy with NULL batting_order.
    --all-roster: roster proxy for every game (early inference).
    """
    with engine.connect() as conn:
        if all_roster:
            log.info("--all-roster: loading roster proxy for all games")
            roster = pd.read_sql(text(ALL_ROSTER_QUERY), conn, params={"d": inference_date})
            if len(roster) == 0:
                return roster
            roster["lineup_confirmed"] = False
            return roster

        confirmed_ids = confirmed_lineup_game_ids(engine, schema, inference_date)
        confirmed = pd.read_sql(text(LINEUP_QUERY), conn, params={"d": inference_date})
        roster = pd.read_sql(text(ALL_ROSTER_QUERY), conn, params={"d": inference_date})

    if confirmed_ids and len(confirmed):
        confirmed = confirmed[confirmed["game_id"].isin(confirmed_ids)].copy()
        confirmed["lineup_confirmed"] = True
        log.info(
            f"  {len(confirmed):,} confirmed lineup slots across "
            f"{confirmed['game_id'].nunique()} game(s)"
        )
    else:
        confirmed = pd.DataFrame()
        log.warning("No confirmed lineups yet — roster proxy for all games")

    if len(roster):
        if confirmed_ids:
            roster = roster[~roster["game_id"].isin(confirmed_ids)].copy()
        roster["lineup_confirmed"] = False
        if len(roster):
            log.info(
                f"  {len(roster):,} roster proxy slots across "
                f"{roster['game_id'].nunique()} unconfirmed game(s)"
            )

    if len(confirmed) and len(roster):
        return pd.concat([confirmed, roster], ignore_index=True)
    if len(confirmed):
        return confirmed
    return roster


ALL_ROSTER_QUERY = """
SELECT DISTINCT ON (gl.player_id, g.game_id)
    gl.player_id                                        AS batter_id,
    gl.player_name                                      AS batter_name,
    gl.team_id,
    gl.bats                                             AS batter_hand,
    NULL::integer                                       AS batting_order,
    (gl.team_id = g.home_team_id)                      AS is_home,
    g.game_id,
    g.game_date,
    g.home_team_id,
    g.away_team_id,
    CASE WHEN gl.team_id = g.home_team_id
         THEN gsp.away_sp_id
         ELSE gsp.home_sp_id END                        AS sp_id,
    CASE WHEN gl.team_id = g.home_team_id
         THEN gsp.away_sp_name
         ELSE gsp.home_sp_name END                      AS sp_name,
    CASE WHEN gl.team_id = g.home_team_id
         THEN fg.away_sp_xwoba_against_90
         ELSE fg.home_sp_xwoba_against_90 END           AS sp_xwoba_against,
    COALESCE(fg.park_runs_factor_blended,
             fg.park_runs_factor, 1.0)                  AS park_hr_factor,
    fg.umpire_k_rate_boost,
    fg.umpire_bb_rate_boost
FROM public.game_lineups gl
JOIN public.games g
    ON g.game_date = :d
    AND gl.team_id IN (g.home_team_id, g.away_team_id)
JOIN public.game_starting_pitchers gsp ON gsp.game_id = g.game_id
JOIN public.features_game fg ON fg.game_id = g.game_id
WHERE gl.game_date >= CAST(:d AS DATE) - INTERVAL '30 days'
  AND gl.game_date < :d
ORDER BY gl.player_id, g.game_id, gl.game_date DESC;
"""

SP_TODAY_QUERY = """
SELECT
    g.game_id,
    g.game_date,
    g.home_team_id,
    g.away_team_id,
    gsp.home_sp_id   AS sp_id,
    gsp.home_sp_name AS sp_name,
    TRUE             AS is_home
FROM public.games g
JOIN public.game_starting_pitchers gsp ON gsp.game_id = g.game_id
WHERE g.game_date = :d AND gsp.home_sp_id IS NOT NULL
UNION ALL
SELECT
    g.game_id,
    g.game_date,
    g.home_team_id,
    g.away_team_id,
    gsp.away_sp_id,
    gsp.away_sp_name,
    FALSE
FROM public.games g
JOIN public.game_starting_pitchers gsp ON gsp.game_id = g.game_id
WHERE g.game_date = :d AND gsp.away_sp_id IS NOT NULL
"""

# Season-to-date batter stats as of today
BATTER_SEASON_QUERY = """
WITH batter_game_stats AS (
    SELECT
        batter                                              AS batter_id,
        game_pk                                             AS game_id,
        game_date,
        EXTRACT(YEAR FROM game_date)::int                   AS season,
        COUNT(DISTINCT at_bat_number)                       AS pa,
        SUM(CASE WHEN events IN ('single','double','triple','home_run')
                 THEN 1 ELSE 0 END)                         AS hits,
        SUM(CASE WHEN events = 'home_run' THEN 1 ELSE 0 END) AS hrs,
        SUM(CASE WHEN events IN ('double','triple','home_run')
                 THEN 1 ELSE 0 END)                         AS xbh,
        SUM(CASE WHEN events = 'strikeout' THEN 1 ELSE 0 END) AS ks,
        AVG(CASE WHEN woba_denom > 0
                 THEN estimated_woba_using_speedangle END)   AS game_xwoba,
        SUM(CASE WHEN launch_speed >= 95 THEN 1 ELSE 0 END) AS hard_hits,
        COUNT(launch_speed)                                  AS bip,
        SUM(CASE
            WHEN launch_speed >= 98
             AND launch_angle BETWEEN 8 AND 50
             AND (
                 (launch_speed >= 116) OR
                 (launch_speed >= 110 AND launch_angle BETWEEN 18 AND 42) OR
                 (launch_speed >= 105 AND launch_angle BETWEEN 20 AND 38) OR
                 (launch_speed >= 100 AND launch_angle BETWEEN 22 AND 36) OR
                 (launch_speed >= 98  AND launch_angle BETWEEN 26 AND 30)
             )
            THEN 1 ELSE 0 END)                              AS barrels
    FROM public.statcast_pitches
    WHERE game_date < :d
      AND EXTRACT(YEAR FROM game_date)::int = :season
      AND batter IN ({batter_id_list})
      AND batter IS NOT NULL
    GROUP BY batter, game_pk, game_date
),
batter_season AS (
    SELECT
        batter_id,
        SUM(pa)     AS sd_pa,
        SUM(hits)   AS sd_hits,
        SUM(hrs)    AS sd_hrs,
        SUM(xbh)    AS sd_xbh,
        SUM(ks)     AS sd_ks,
        SUM(hard_hits) AS sd_hard_hits,
        SUM(bip)    AS sd_bip,
        SUM(barrels) AS sd_barrels,
        AVG(game_xwoba) AS sd_xwoba
    FROM batter_game_stats
    GROUP BY batter_id
),
batter_30d AS (
    SELECT
        batter                                              AS batter_id,
        SUM(CASE WHEN events IN ('single','double','triple','home_run')
                 THEN 1 ELSE 0 END)                         AS hits_30d,
        COUNT(DISTINCT at_bat_number)                       AS pa_30d,
        SUM(CASE WHEN launch_speed >= 95 THEN 1 ELSE 0 END) AS hard_hits_30d,
        COUNT(launch_speed)                                  AS bip_30d,
        SUM(CASE
            WHEN launch_speed >= 98
             AND launch_angle BETWEEN 8 AND 50
             AND (
                 (launch_speed >= 116) OR
                 (launch_speed >= 110 AND launch_angle BETWEEN 18 AND 42) OR
                 (launch_speed >= 105 AND launch_angle BETWEEN 20 AND 38) OR
                 (launch_speed >= 100 AND launch_angle BETWEEN 22 AND 36) OR
                 (launch_speed >= 98  AND launch_angle BETWEEN 26 AND 30)
             )
            THEN 1 ELSE 0 END)                              AS barrels_30d,
        AVG(CASE WHEN woba_denom > 0
                 THEN estimated_woba_using_speedangle END)   AS xwoba_30d
    FROM public.statcast_pitches
    WHERE game_date >= CAST(:d AS DATE) - INTERVAL '30 days'
      AND game_date < :d
      AND batter IN ({batter_id_list})
    GROUP BY batter
)
SELECT
    s.batter_id,
    -- Season-to-date (Bayesian shrinkage, prior=100 PA)
    (COALESCE(s.sd_hits, 0) + 100*0.245) / (COALESCE(s.sd_pa,0)+100) AS batter_hit_rate_season,
    (COALESCE(s.sd_hrs,  0) + 100*0.036) / (COALESCE(s.sd_pa,0)+100) AS batter_hr_rate_season,
    (COALESCE(s.sd_ks,   0) + 100*0.230) / (COALESCE(s.sd_pa,0)+100) AS batter_k_rate_season,
    (COALESCE(s.sd_xbh,  0) + 100*0.080) / (COALESCE(s.sd_pa,0)+100) AS batter_iso_season,
    COALESCE(s.sd_xwoba, 0.320)                                        AS batter_xwoba_season,
    -- 30-day rolling
    CASE WHEN COALESCE(r.pa_30d,0) >= 10
         THEN r.hits_30d::float / r.pa_30d
         ELSE 0.245 END                                                AS batter_hit_rate_30d,
    CASE WHEN COALESCE(r.bip_30d,0) >= 10
         THEN r.hard_hits_30d::float / r.bip_30d
         ELSE 0.242 END                                                AS batter_hard_hit_rate_30d,
    CASE WHEN COALESCE(r.bip_30d,0) >= 10
         THEN r.barrels_30d::float / r.bip_30d
         ELSE 0.072 END                                                AS batter_barrel_rate_30d,
    COALESCE(r.xwoba_30d, 0.320)                                       AS batter_xwoba_30d,
    COALESCE(s.sd_pa, 0)                                               AS batter_sd_pa
FROM batter_season s
LEFT JOIN batter_30d r ON r.batter_id = s.batter_id;
"""

# SP season-to-date K rate
# Filter to starter appearances only (>= 15 BF per game) to match training distribution
SP_SEASON_QUERY = """
WITH sp_game_stats AS (
    SELECT
        pitcher                                             AS pitcher_id,
        game_pk,
        SUM(CASE WHEN events = 'strikeout' THEN 1 ELSE 0 END) AS ks,
        COUNT(DISTINCT at_bat_number)                       AS bf
    FROM public.statcast_pitches
    WHERE game_date < :d
      AND EXTRACT(YEAR FROM game_date)::int = :season
      AND pitcher IN ({pitcher_id_list})
      AND events IS NOT NULL
    GROUP BY pitcher, game_pk
    HAVING COUNT(DISTINCT at_bat_number) >= 15
),
sp_stats AS (
    SELECT
        pitcher_id,
        SUM(ks) AS sd_ks,
        SUM(bf) AS sd_bf
    FROM sp_game_stats
    GROUP BY pitcher_id
)
SELECT
    pitcher_id,
    (COALESCE(sd_ks,0) + 200*0.230) / (COALESCE(sd_bf,0)+200) AS sp_k_rate_season,
    COALESCE(sd_bf, 0) AS sp_sd_bf
FROM sp_stats;
"""

# SP innings pitched this season (for workload feature)
SP_INNINGS_QUERY = """
SELECT
    pitcher_id,
    SUM(innings_pitched) AS sp_innings_season
FROM public.pitcher_starts
WHERE game_date < :d
  AND EXTRACT(YEAR FROM game_date)::int = :season
  AND pitcher_id IN ({pitcher_id_list})
GROUP BY pitcher_id;
"""

# Batter pitch type skills (nearest as_of_date)
BATTER_SKILLS_QUERY = """
SELECT DISTINCT ON (batter_id)
    batter_id,
    skill_ff, skill_si, skill_fc, skill_sl,
    skill_cu, skill_ch, skill_sp, skill_fs
FROM public.batter_vs_pitchtype_rolling
WHERE window_days = 365
  AND as_of_date <= :d
  AND batter_id IN ({batter_id_list})
ORDER BY batter_id, as_of_date DESC;
"""

# Pitcher pitch mix (nearest as_of_date)
PITCHER_MIX_QUERY = """
SELECT DISTINCT ON (pitcher_id)
    pitcher_id,
    pct_ff, pct_si, pct_fc, pct_sl,
    pct_cu, pct_ch, pct_sp, pct_fs
FROM public.pitcher_pitchmix_rolling
WHERE window_days = 365
  AND as_of_date <= :d
  AND pitcher_id IN ({pitcher_id_list})
ORDER BY pitcher_id, as_of_date DESC;
"""

# Pitcher hand from statcast
PITCHER_HAND_QUERY = """
SELECT DISTINCT ON (pitcher)
    pitcher AS pitcher_id,
    p_throws
FROM public.statcast_pitches
WHERE pitcher IN ({pitcher_id_list})
  AND p_throws IS NOT NULL
ORDER BY pitcher, game_date DESC;
"""

# Batter hand fallback from statcast
BATTER_HAND_QUERY = """
SELECT DISTINCT ON (batter)
    batter AS batter_id,
    stand AS batter_hand
FROM public.statcast_pitches
WHERE batter IN ({batter_id_list})
  AND stand IS NOT NULL
ORDER BY batter, game_date DESC;
"""

# Opposing lineup K rate and xwOBA for pitcher model
LINEUP_QUALITY_QUERY = """
SELECT
    fg.game_id,
    fg.home_lineup_k_rate_90,
    fg.away_lineup_k_rate_90,
    fg.home_lineup_xwoba_90,
    fg.away_lineup_xwoba_90,
    fg.home_sp_k9_last5,
    fg.away_sp_k9_last5,
    fg.home_sp_xwoba_against_90,
    fg.away_sp_xwoba_against_90,
    fg.home_sp_days_rest,
    fg.away_sp_days_rest,
    fg.home_sp_bb_rate_90,
    fg.away_sp_bb_rate_90,
    fg.park_runs_factor_blended AS park_runs_factor,
    fg.umpire_k_rate_boost,
    fg.umpire_runs_boost
FROM public.features_game fg
WHERE fg.game_date = :d;
"""

SP_BB_SEASON_QUERY = """
WITH sp_game_stats AS (
    SELECT
        pitcher AS pitcher_id,
        game_pk,
        SUM(CASE WHEN events IN ('walk','hit_by_pitch') THEN 1 ELSE 0 END) AS walks,
        COUNT(DISTINCT at_bat_number) AS bf
    FROM public.statcast_pitches
    WHERE game_date < :d
      AND EXTRACT(YEAR FROM game_date)::int = :season
      AND pitcher IN ({pitcher_id_list})
      AND events IS NOT NULL
    GROUP BY pitcher, game_pk
    HAVING COUNT(DISTINCT at_bat_number) >= 15
)
SELECT
    pitcher_id,
    (COALESCE(SUM(walks), 0) + 200 * 0.085) / (COALESCE(SUM(bf), 0) + 200) AS sp_bb_rate_season
FROM sp_game_stats
GROUP BY pitcher_id;
"""

SP_XWOBA_SEASON_QUERY = """
WITH sp_game_stats AS (
    SELECT
        pitcher AS pitcher_id,
        game_pk,
        COUNT(DISTINCT at_bat_number) AS bf,
        AVG(estimated_woba_using_speedangle) FILTER (
            WHERE woba_denom = 1 AND estimated_woba_using_speedangle IS NOT NULL
        ) AS xwoba_against_game
    FROM public.statcast_pitches
    WHERE game_date < :d
      AND EXTRACT(YEAR FROM game_date)::int = :season
      AND pitcher IN ({pitcher_id_list})
      AND events IS NOT NULL
    GROUP BY pitcher, game_pk
    HAVING COUNT(DISTINCT at_bat_number) >= 15
)
SELECT
    pitcher_id,
    SUM(xwoba_against_game * bf) / NULLIF(SUM(bf), 0) AS sp_xwoba_against_season
FROM sp_game_stats
WHERE xwoba_against_game IS NOT NULL
GROUP BY pitcher_id;
"""

SP_ROLLING_TODAY_QUERY = """
WITH last_bf AS (
    SELECT pitcher_id, AVG(bf)::float AS expected_bf
    FROM (
        SELECT pitcher AS pitcher_id, game_pk, game_date,
               COUNT(DISTINCT at_bat_number) AS bf,
               ROW_NUMBER() OVER (PARTITION BY pitcher ORDER BY game_date DESC, game_pk DESC) AS rn
        FROM public.statcast_pitches
        WHERE game_date < :d
          AND pitcher IN ({pitcher_id_list})
          AND events IS NOT NULL
        GROUP BY pitcher, game_pk, game_date
        HAVING COUNT(DISTINCT at_bat_number) >= 15
    ) x
    WHERE rn <= 5
    GROUP BY pitcher_id
),
last_starts AS (
    SELECT pitcher_id,
           AVG(innings_pitched)::float AS expected_ip,
           AVG(walks_allowed * 9.0 / NULLIF(innings_pitched, 0))::float AS sp_bb9_last5,
           AVG(earned_runs)::float AS sp_er_last5
    FROM (
        SELECT pitcher_id, game_date, innings_pitched, walks_allowed, earned_runs,
               ROW_NUMBER() OVER (PARTITION BY pitcher_id ORDER BY game_date DESC, game_id DESC) AS rn
        FROM public.pitcher_starts
        WHERE game_date < :d
          AND pitcher_id IN ({pitcher_id_list})
          AND innings_pitched IS NOT NULL
    ) s
    WHERE rn <= 5
    GROUP BY pitcher_id
)
SELECT
    COALESCE(b.pitcher_id, s.pitcher_id) AS pitcher_id,
    b.expected_bf,
    s.expected_ip,
    s.sp_bb9_last5,
    s.sp_er_last5
FROM last_bf b
FULL OUTER JOIN last_starts s ON s.pitcher_id = b.pitcher_id;
"""


# ── Feature computation ───────────────────────────────────────────────────────

def compute_matchup_score(batter_id, sp_id, batter_skills, pitcher_mix):
    DEFAULT = 0.320
    b = batter_skills.get(int(batter_id))
    p = pitcher_mix.get(int(sp_id)) if sp_id else None
    if b is None or p is None:
        return DEFAULT, 0.0

    total_w = weighted = 0.0
    fastball_pct = 0.0
    for pt in PITCH_TYPES:
        pct = p.get(f'pct_{pt}', 0.0) or 0.0
        skill = b.get(f'skill_{pt}', DEFAULT) or DEFAULT
        weighted += pct * skill
        total_w += pct
        if pt in ('ff', 'si'):
            fastball_pct += pct

    score = weighted / total_w if total_w > 0.05 else DEFAULT
    return score, fastball_pct


def get_platoon(batter_hand, sp_id, pitcher_hands):
    if not batter_hand or not sp_id:
        return 0
    ph = pitcher_hands.get(int(sp_id))
    if ph is None:
        return 0
    return 1 if batter_hand != ph else -1


def ensure_tables(engine, schema):
    """Create prop prediction tables if they don't exist."""
    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {schema}.player_prop_predictions (
                id              BIGSERIAL PRIMARY KEY,
                game_id         BIGINT NOT NULL,
                game_date       DATE NOT NULL,
                batter_id       INTEGER NOT NULL,
                batter_name     TEXT,
                team_id         INTEGER,
                batting_order   INTEGER,
                sp_id           BIGINT,
                sp_name         TEXT,
                as_of_ts        TIMESTAMPTZ NOT NULL,
                -- Batter prop probabilities
                p_hit           DOUBLE PRECISION,
                p_2plus_hits    DOUBLE PRECISION,
                p_hr            DOUBLE PRECISION,
                p_k             DOUBLE PRECISION,
                p_2plus_bases   DOUBLE PRECISION,
                p_walk          DOUBLE PRECISION,
                -- Key features for display/audit
                matchup_score   DOUBLE PRECISION,
                platoon_advantage INTEGER,
                batter_xwoba_season DOUBLE PRECISION,
                batter_hit_rate_30d DOUBLE PRECISION,
                -- Metadata
                model_version   TEXT DEFAULT 'v1'
            );
        """))
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {schema}.pitcher_prop_predictions (
                id              BIGSERIAL PRIMARY KEY,
                game_id         BIGINT NOT NULL,
                game_date       DATE NOT NULL,
                pitcher_id      BIGINT NOT NULL,
                pitcher_name    TEXT,
                is_home         BOOLEAN,
                as_of_ts        TIMESTAMPTZ NOT NULL,
                -- Poisson output
                lambda_k        DOUBLE PRECISION,
                -- Exact probabilities P(K=k)
                p_k0            DOUBLE PRECISION,
                p_k1            DOUBLE PRECISION,
                p_k2            DOUBLE PRECISION,
                p_k3            DOUBLE PRECISION,
                p_k4            DOUBLE PRECISION,
                p_k5            DOUBLE PRECISION,
                p_k6            DOUBLE PRECISION,
                p_k7            DOUBLE PRECISION,
                p_k8            DOUBLE PRECISION,
                p_k9            DOUBLE PRECISION,
                p_k10           DOUBLE PRECISION,
                p_k10plus       DOUBLE PRECISION,
                -- Over/under probabilities P(K > threshold)
                p_over_0_5      DOUBLE PRECISION,
                p_over_1_5      DOUBLE PRECISION,
                p_over_2_5      DOUBLE PRECISION,
                p_over_3_5      DOUBLE PRECISION,
                p_over_4_5      DOUBLE PRECISION,
                p_over_5_5      DOUBLE PRECISION,
                p_over_6_5      DOUBLE PRECISION,
                p_over_7_5      DOUBLE PRECISION,
                p_over_8_5      DOUBLE PRECISION,
                p_over_9_5      DOUBLE PRECISION,
                -- Key features for audit
                sp_k_rate_season DOUBLE PRECISION,
                sp_innings_season DOUBLE PRECISION,
                opp_lineup_k_rate DOUBLE PRECISION,
                -- Metadata
                model_version   TEXT DEFAULT 'v1'
            );
        """))
        conn.execute(text(f"""
            CREATE INDEX IF NOT EXISTS idx_player_props_game_date
                ON {schema}.player_prop_predictions (game_date);
            CREATE INDEX IF NOT EXISTS idx_pitcher_props_game_date
                ON {schema}.pitcher_prop_predictions (game_date);
        """))
        conn.execute(text(f"""
            ALTER TABLE {schema}.player_prop_predictions
                ADD COLUMN IF NOT EXISTS lineup_confirmed BOOLEAN DEFAULT FALSE
        """))
        for col in PITCHER_EXTRA_COLUMNS:
            conn.execute(text(
                f"ALTER TABLE {schema}.pitcher_prop_predictions "
                f"ADD COLUMN IF NOT EXISTS {col} DOUBLE PRECISION"
            ))
        conn.execute(text(f"""
            ALTER TABLE {schema}.pitcher_prop_predictions
                ADD COLUMN IF NOT EXISTS is_defaulted BOOLEAN DEFAULT FALSE
        """))
        conn.execute(text(f"""
            ALTER TABLE {schema}.pitcher_prop_predictions
                ADD COLUMN IF NOT EXISTS expected_ip DOUBLE PRECISION
        """))
    log.info("Tables ensured")


def sql_id_list(ids) -> str:
    """Comma-separated integer IDs for `IN ({list})` SQL templates."""
    if not ids:
        return "-1"
    return ",".join(str(int(i)) for i in ids)


def bind_id_lists(sql: str, batter_ids=None, pitcher_ids=None) -> str:
    """Substitute {batter_id_list} / {pitcher_id_list} placeholders."""
    return sql.format(
        batter_id_list=sql_id_list(batter_ids or []),
        pitcher_id_list=sql_id_list(pitcher_ids or []),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=str(date.today()))
    ap.add_argument("--batter-model",
                    default="artifacts/props_v1_expanded.joblib")
    ap.add_argument("--pitcher-model",
                    default="artifacts/pitcher_props_v1.joblib")
    ap.add_argument("--pitcher-walks-model",
                    default="artifacts/pitcher_walks_v1.joblib")
    ap.add_argument("--pitcher-hits-model",
                    default="artifacts/pitcher_hits_v1.joblib")
    ap.add_argument("--pitcher-er-model",
                    default="artifacts/pitcher_er_v1.joblib")
    ap.add_argument("--skip-pitcher-extras", action="store_true",
                    help="Run K model only; skip walks/hits/ER extras.")
    ap.add_argument("--schema", default="public")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--all-roster", action="store_true",
                    help="Run batter props for all players active in last 30 days, not just confirmed lineup. Use in early morning before lineups post.")
    ap.add_argument("--pitchers-only", action="store_true",
                    help="Skip batter props entirely. Only run pitcher K model.")
    args = ap.parse_args()

    pg_dsn = os.environ.get("PG_DSN")
    if not pg_dsn:
        log.error("PG_DSN not set"); sys.exit(1)

    inference_date = args.date
    season = int(inference_date[:4])
    now = datetime.utcnow()

    log.info(f"Props inference — date={inference_date}")

    # ── Load models ──
    log.info(f"Loading batter model: {args.batter_model}")
    batter_bundle = joblib.load(args.batter_model)
    batter_scaler  = batter_bundle['scaler']
    batter_models = {
        k: _patch_logistic_regression(m)
        for k, m in batter_bundle["models"].items()
    }
    batter_features = batter_bundle['features']
    log.info(f"  Batter features ({len(batter_features)}): {batter_features}")

    log.info(f"Loading pitcher model: {args.pitcher_model}")
    pitcher_bundle  = joblib.load(args.pitcher_model)
    pitcher_scaler  = pitcher_bundle['scaler']
    pitcher_model   = pitcher_bundle['model']
    pitcher_features = pitcher_bundle['features']
    log.info(f"  Pitcher features ({len(pitcher_features)}): {pitcher_features}")

    walks_bundle = hits_bundle = er_bundle = None
    if not args.skip_pitcher_extras:
        for label, path in [
            ("walks", args.pitcher_walks_model),
            ("hits", args.pitcher_hits_model),
            ("er", args.pitcher_er_model),
        ]:
            if os.path.isfile(path):
                log.info(f"Loading pitcher {label} model: {path}")
                try:
                    bundle = joblib.load(path)
                except Exception as exc:
                    log.error(
                        "Failed to load %s (%s). Re-export with the pipeline image "
                        "scikit-learn version or pin requirements.txt to the training version.",
                        path,
                        exc,
                    )
                    raise
                if label == "walks":
                    walks_bundle = bundle
                elif label == "hits":
                    hits_bundle = bundle
                else:
                    er_bundle = bundle
            else:
                log.warning(f"Pitcher {label} model not found at {path} — skipping")

    engine = create_engine(pg_dsn)

    with engine.connect() as conn:
        sc_diag = pd.read_sql(
            text("""
                SELECT
                    COUNT(*) AS total,
                    COUNT(*) FILTER (
                        WHERE EXTRACT(YEAR FROM game_date)::int = :season
                    ) AS season_rows,
                    COUNT(DISTINCT pitcher) FILTER (
                        WHERE EXTRACT(YEAR FROM game_date)::int = :season
                          AND pitcher IS NOT NULL
                    ) AS season_pitchers,
                    COUNT(DISTINCT batter) FILTER (
                        WHERE EXTRACT(YEAR FROM game_date)::int = :season
                          AND batter IS NOT NULL
                    ) AS season_batters
                FROM public.statcast_pitches
                WHERE events IS NOT NULL
            """),
            conn,
            params={"season": season},
        )
    log.info(f"Statcast diagnostic: {sc_diag.to_dict('records')[0]}")

    # ── Load today's lineups ──
    if args.pitchers_only:
        log.info("--pitchers-only: skipping batter props entirely")
        lineups = pd.DataFrame()
        batter_ids = []
    else:
        lineups = load_batter_lineups(engine, args.schema, inference_date, args.all_roster)
        if len(lineups) == 0:
            log.warning("No batter rows found — skipping batter props")
            batter_ids = []
        else:
            log.info(
                f"  {len(lineups):,} batter rows across "
                f"{lineups['game_id'].nunique()} games "
                f"({lineups['lineup_confirmed'].sum()} confirmed slots)"
            )
            batter_ids = lineups["batter_id"].dropna().astype(int).unique().tolist()

    if len(lineups) > 0:
        from features.active_roster import filter_eligible_batters
        removed = 0
        if args.all_roster:
            before = len(lineups)
            lineups = filter_eligible_batters(lineups, engine, inference_date, id_col="batter_id")
            removed = before - len(lineups)
        else:
            confirmed_part = lineups[lineups["lineup_confirmed"].fillna(False)].copy()
            proxy_part = lineups[~lineups["lineup_confirmed"].fillna(False)].copy()
            before = len(proxy_part)
            if len(proxy_part):
                proxy_part = filter_eligible_batters(
                    proxy_part, engine, inference_date, id_col="batter_id"
                )
            removed = before - len(proxy_part)
            if len(confirmed_part) and len(proxy_part):
                lineups = pd.concat([confirmed_part, proxy_part], ignore_index=True)
            elif len(confirmed_part):
                lineups = confirmed_part
            else:
                lineups = proxy_part
        if removed:
            log.info(f"  Active roster filter removed {removed:,} inactive/IL proxy slots")
        log.info(
            f"  {len(lineups):,} eligible batter rows across "
            f"{lineups['game_id'].nunique() if len(lineups) else 0} games"
        )
        batter_ids = lineups["batter_id"].dropna().astype(int).unique().tolist()

    if len(lineups) > 0:
        sp_ids = lineups['sp_id'].dropna().astype(int).unique().tolist()
    else:
        log.info("Loading today's starting pitchers for pitcher props...")
        with engine.connect() as conn:
            sp_starters = pd.read_sql(text(SP_TODAY_QUERY), conn, params={"d": inference_date})
        sp_ids = sp_starters['sp_id'].dropna().astype(int).unique().tolist()
    if args.all_roster:
        from features.active_roster import load_eligible_player_ids
        eligible = load_eligible_player_ids(engine, inference_date)
        if eligible:
            sp_ids = [int(x) for x in sp_ids if int(x) in eligible]
    sql_ids = {"batter_ids": batter_ids, "pitcher_ids": sp_ids}

    # ── Load batter season stats ──
    log.info("Loading batter season stats...")
    with engine.connect() as conn:
        batter_stats = pd.read_sql(text(
            bind_id_lists(BATTER_SEASON_QUERY, **sql_ids)
        ), conn, params={"d": inference_date, "season": season})
    log.info(f"  {len(batter_stats):,} batters with season stats")
    if batter_ids:
        matched_bat = set(batter_stats['batter_id'].astype(int))
        missing_bat = [x for x in batter_ids if x not in matched_bat]
        if missing_bat:
            log.warning(
                f"  {len(missing_bat)}/{len(batter_ids)} batters missing season stats "
                f"(will default xwOBA to {DEFAULTS['batter_xwoba_season']:.3f}); "
                f"sample IDs: {missing_bat[:8]}"
            )
        if len(batter_stats):
            at_default = (
                (batter_stats['batter_xwoba_season'] - DEFAULTS['batter_xwoba_season']).abs() < 0.001
            ).sum()
            log.info(
                f"  batter_xwoba_season at league default ({DEFAULTS['batter_xwoba_season']:.3f}): "
                f"{at_default}/{len(batter_stats)}"
            )

    # ── Load pitcher season K rate ──
    log.info("Loading pitcher season stats...")
    with engine.connect() as conn:
        sp_season = pd.read_sql(text(
            bind_id_lists(SP_SEASON_QUERY, **sql_ids)
        ), conn, params={"d": inference_date, "season": season})
    log.info(f"  {len(sp_season):,} pitchers with season stats")
    if sp_ids:
        matched_sp = set(sp_season['pitcher_id'].astype(int))
        missing_sp = [x for x in sp_ids if x not in matched_sp]
        if missing_sp:
            log.warning(
                f"  {len(missing_sp)}/{len(sp_ids)} SPs missing season K stats "
                f"(will default to {DEFAULTS['sp_k_rate_season']:.3f}); "
                f"sample IDs: {missing_sp[:8]}"
            )
        if len(sp_season):
            at_default = (
                (sp_season['sp_k_rate_season'] - DEFAULTS['sp_k_rate_season']).abs() < 0.001
            ).sum()
            log.info(
                f"  sp_k_rate_season at league default ({DEFAULTS['sp_k_rate_season']:.3f}): "
                f"{at_default}/{len(sp_season)}"
            )

    # ── Load pitcher innings ──
    log.info("Loading pitcher innings...")
    try:
        with engine.connect() as conn:
            sp_innings = pd.read_sql(text(
                bind_id_lists(SP_INNINGS_QUERY, **sql_ids)
            ), conn, params={"d": inference_date, "season": season})
        log.info(f"  {len(sp_innings):,} pitchers with innings data")
    except Exception as e:
        log.warning(f"  Could not load SP innings: {e}")
        sp_innings = pd.DataFrame(columns=['pitcher_id', 'sp_innings_season'])

    sp_bb_season = pd.DataFrame(columns=['pitcher_id', 'sp_bb_rate_season'])
    sp_xwoba_season = pd.DataFrame(columns=['pitcher_id', 'sp_xwoba_against_season'])
    sp_rolling = pd.DataFrame(columns=['pitcher_id', 'expected_bf', 'expected_ip', 'sp_bb9_last5', 'sp_er_last5'])
    if sp_ids:
        log.info("Loading pitcher BB / rolling context for extras...")
        with engine.connect() as conn:
            sp_bb_season = pd.read_sql(text(
                bind_id_lists(SP_BB_SEASON_QUERY, **sql_ids)
            ), conn, params={"d": inference_date, "season": season})
            sp_xwoba_season = pd.read_sql(text(
                bind_id_lists(SP_XWOBA_SEASON_QUERY, **sql_ids)
            ), conn, params={"d": inference_date, "season": season})
            sp_rolling = pd.read_sql(text(
                bind_id_lists(SP_ROLLING_TODAY_QUERY, **sql_ids)
            ), conn, params={"d": inference_date})

    # ── Load pitch type lookups ──
    log.info("Loading batter pitch skills and pitcher mix...")
    with engine.connect() as conn:
        batter_skills_df = pd.read_sql(text(
            bind_id_lists(BATTER_SKILLS_QUERY, **sql_ids)
        ), conn, params={"d": inference_date})
        pitcher_mix_df = pd.read_sql(text(
            bind_id_lists(PITCHER_MIX_QUERY, **sql_ids)
        ), conn, params={"d": inference_date})

    # Build lookups
    batter_skills = {}
    for _, row in batter_skills_df.iterrows():
        batter_skills[int(row['batter_id'])] = {
            f'skill_{pt}': row.get(f'skill_{pt}', DEFAULTS['matchup_score'])
            for pt in PITCH_TYPES}

    pitcher_mix = {}
    for _, row in pitcher_mix_df.iterrows():
        pitcher_mix[int(row['pitcher_id'])] = {
            f'pct_{pt}': row.get(f'pct_{pt}', 0.0)
            for pt in PITCH_TYPES}

    # ── Load handedness ──
    log.info("Loading handedness data...")
    with engine.connect() as conn:
        pitcher_hand_df = pd.read_sql(text(
            bind_id_lists(PITCHER_HAND_QUERY, **sql_ids)
        ), conn)
        batter_hand_df = pd.read_sql(text(
            bind_id_lists(BATTER_HAND_QUERY, **sql_ids)
        ), conn)

    pitcher_hands = dict(zip(
        pitcher_hand_df['pitcher_id'].astype(int), pitcher_hand_df['p_throws']))
    batter_hands_statcast = dict(zip(
        batter_hand_df['batter_id'].astype(int), batter_hand_df['batter_hand']))

    # ── Load lineup quality for pitcher model ──
    log.info("Loading lineup quality for pitcher model...")
    with engine.connect() as conn:
        lu_quality = pd.read_sql(text(LINEUP_QUALITY_QUERY), conn,
                                 params={"d": inference_date})

    # ── Build batter features ──
    if not args.pitchers_only:
        log.info("Building batter features...")

        # Merge stats
        df = lineups.merge(
            batter_stats, on='batter_id', how='left'
        ).merge(
            sp_season.rename(columns={'pitcher_id': 'sp_id'}),
            on='sp_id', how='left'
        )

        # Fill defaults
        for col, default in DEFAULTS.items():
            if col in df.columns:
                df[col] = df[col].fillna(default)

        # Batter hand: use game_lineups.bats first, fall back to statcast
        null_hand = df['batter_hand'].isna()
        if null_hand.sum() > 0:
            df.loc[null_hand, 'batter_hand'] = (
                df.loc[null_hand, 'batter_id'].map(
                    lambda x: batter_hands_statcast.get(int(x)) if pd.notna(x) else None))

        # Compute matchup score and platoon per batter
        matchup_scores = []
        fastball_pcts = []
        platoon_advantages = []

        for _, row in df.iterrows():
            bid = int(row['batter_id']) if pd.notna(row['batter_id']) else None
            sid = int(row['sp_id']) if pd.notna(row.get('sp_id')) else None

            ms, fb = compute_matchup_score(bid, sid, batter_skills, pitcher_mix)
            plat = get_platoon(row.get('batter_hand'), sid, pitcher_hands)

            matchup_scores.append(ms)
            fastball_pcts.append(fb)
            platoon_advantages.append(plat)

        df['matchup_score']     = matchup_scores
        df['sp_fastball_pct']   = fastball_pcts
        df['platoon_advantage'] = platoon_advantages

        # Fill remaining feature defaults
        df['batter_xwoba_season']      = df.get('batter_xwoba_season', pd.Series()).fillna(DEFAULTS['batter_xwoba_season'])
        df['batter_k_rate_season']     = df.get('batter_k_rate_season', pd.Series()).fillna(DEFAULTS['batter_k_rate_season'])
        df['batter_hr_rate_season']    = df.get('batter_hr_rate_season', pd.Series()).fillna(DEFAULTS['batter_hr_rate_season'])
        df['batter_hit_rate_30d']      = df.get('batter_hit_rate_30d', pd.Series()).fillna(DEFAULTS['batter_hit_rate_30d'])
        df['batter_iso_season']        = df.get('batter_iso_season', pd.Series()).fillna(DEFAULTS['batter_iso_season'])
        df['batter_hard_hit_rate_30d'] = df.get('batter_hard_hit_rate_30d', pd.Series()).fillna(DEFAULTS['batter_hard_hit_rate_30d'])
        df['batter_barrel_rate_30d']  = df.get('batter_barrel_rate_30d', pd.Series()).fillna(DEFAULTS['batter_barrel_rate_30d'])
        df['batter_xwoba_30d']         = df.get('batter_xwoba_30d', pd.Series()).fillna(DEFAULTS['batter_xwoba_30d'])
        df['sp_xwoba_against']         = df.get('sp_xwoba_against', pd.Series()).fillna(DEFAULTS['sp_xwoba_against'])
        df['park_hr_factor']           = df.get('park_hr_factor', pd.Series()).fillna(DEFAULTS['park_hr_factor'])
        df['umpire_k_boost']           = df.get('umpire_k_rate_boost', pd.Series()).fillna(DEFAULTS['umpire_k_boost'])
        df['sp_k_rate_season']         = df.get('sp_k_rate_season', pd.Series()).fillna(DEFAULTS['sp_k_rate_season'])
        df['sp_hr_rate_season']        = df.get('sp_hr_rate_season', pd.Series()).fillna(DEFAULTS['sp_hr_rate_season'])

        # Model features use default order 5 when unknown; stored batting_order stays NULL for proxy rows.
        df_infer = df.copy()
        df_infer['batting_order'] = (
            pd.to_numeric(df_infer['batting_order'], errors='coerce')
            .fillna(DEFAULTS['batting_order'])
        )

        log.info("Batter feature sample (first 5):")
        sample_cols = [
            'batter_name', 'batter_xwoba_season', 'batter_k_rate_season',
            'batter_hr_rate_season', 'batter_hit_rate_30d',
            'matchup_score', 'sp_xwoba_against',
        ]
        for _, row in df[sample_cols].head(5).iterrows():
            log.info(
                f"  {row['batter_name']}: "
                f"xwoba={row['batter_xwoba_season']:.3f} "
                f"k_rate={row['batter_k_rate_season']:.3f} "
                f"matchup={row['matchup_score']:.3f} "
                f"sp_xwoba={row['sp_xwoba_against']:.3f}"
            )

        # ── Run batter model ──
        log.info(f"Running batter model on {len(df):,} rows...")
        X_batter = batter_scaler.transform(fill_feature_matrix(df_infer, batter_features))

        target_map = {
            'recorded_hit':        'p_hit',
            'recorded_2plus':      'p_2plus_hits',
            'recorded_hr':         'p_hr',
            'recorded_k':          'p_k',
            'recorded_2plus_bases': 'p_2plus_bases',
            'recorded_walk':       'p_walk',
        }

        for target, col in target_map.items():
            if target in batter_models:
                df[col] = predict_binary_proba(batter_models[target], X_batter)
            else:
                df[col] = np.nan

        # Game-level HR prop: P(at least one HR), not miscalibrated classifier output
        df['p_hr'] = compute_game_hr_probability(df_infer)

        # Preview
        log.info("\nBatter prop predictions (sample):")
        log.info(f"  {'Batter':<25} {'Hit':>6} {'2+H':>6} {'HR':>6} {'K':>6} {'2+TB':>6} {'BB':>6}")
        for _, row in df.head(10).iterrows():
            log.info(f"  {str(row.get('batter_name','?')):<25} "
                     f"{row.get('p_hit',0)*100:>5.1f}% "
                     f"{row.get('p_2plus_hits',0)*100:>5.1f}% "
                     f"{row.get('p_hr',0)*100:>5.1f}% "
                     f"{row.get('p_k',0)*100:>5.1f}% "
                     f"{row.get('p_2plus_bases',0)*100:>5.1f}% "
                     f"{row.get('p_walk',0)*100:>5.1f}%")
    else:
        log.info("--pitchers-only: skipping all batter inference")

    # ── Build pitcher features ──
    log.info("\nBuilding pitcher features...")

    # Always load SPs from game_starting_pitchers — lineup rows carry the batter's
    # is_home flag but sp_id is the *opposing* starter, which inverts home/away.
    with engine.connect() as conn:
        sp_rows = pd.read_sql(text(SP_TODAY_QUERY), conn, params={"d": inference_date})
    sp_rows = sp_rows[sp_rows['sp_id'].notna()].copy()
    sp_rows['sp_id'] = sp_rows['sp_id'].astype(int)

    # Merge SP season stats
    sp_df = sp_rows.merge(
        sp_season.rename(columns={'pitcher_id': 'sp_id'}),
        on='sp_id', how='left'
    ).merge(
        sp_bb_season.rename(columns={'pitcher_id': 'sp_id'}),
        on='sp_id', how='left'
    ).merge(
        sp_xwoba_season.rename(columns={'pitcher_id': 'sp_id'}),
        on='sp_id', how='left'
    ).merge(
        sp_innings.rename(columns={'pitcher_id': 'sp_id'}),
        on='sp_id', how='left'
    ).merge(
        sp_rolling.rename(columns={'pitcher_id': 'sp_id'}),
        on='sp_id', how='left'
    ).merge(
        lu_quality, on='game_id', how='left'
    )

    # SP is home or away determines which lineup stats to use
    is_home_sp = sp_df['is_home']
    sp_df['sp_k9_last5'] = np.where(is_home_sp,
        sp_df.get('home_sp_k9_last5', pd.Series(np.nan, index=sp_df.index)),
        sp_df.get('away_sp_k9_last5', pd.Series(np.nan, index=sp_df.index)))
    sp_df['sp_xwoba_against_90'] = np.where(is_home_sp,
        sp_df.get('home_sp_xwoba_against_90', pd.Series(np.nan, index=sp_df.index)),
        sp_df.get('away_sp_xwoba_against_90', pd.Series(np.nan, index=sp_df.index)))
    sp_df['sp_bb_rate_90'] = np.where(is_home_sp,
        sp_df.get('home_sp_bb_rate_90', pd.Series(np.nan, index=sp_df.index)),
        sp_df.get('away_sp_bb_rate_90', pd.Series(np.nan, index=sp_df.index)))
    sp_df['sp_days_rest'] = np.where(is_home_sp,
        sp_df.get('home_sp_days_rest', pd.Series(np.nan, index=sp_df.index)),
        sp_df.get('away_sp_days_rest', pd.Series(np.nan, index=sp_df.index)))
    sp_df['opp_lineup_k_rate_30d'] = np.where(is_home_sp,
        sp_df.get('away_lineup_k_rate_90', pd.Series(np.nan, index=sp_df.index)),
        sp_df.get('home_lineup_k_rate_90', pd.Series(np.nan, index=sp_df.index)))
    sp_df['opp_lineup_bb_rate_30d'] = np.where(is_home_sp,
        sp_df.get('away_lineup_bb_rate_90', pd.Series(np.nan, index=sp_df.index)),
        sp_df.get('home_lineup_bb_rate_90', pd.Series(np.nan, index=sp_df.index)))
    sp_df['opp_lineup_xwoba_90'] = np.where(is_home_sp,
        sp_df.get('away_lineup_xwoba_90', pd.Series(np.nan, index=sp_df.index)),
        sp_df.get('home_lineup_xwoba_90', pd.Series(np.nan, index=sp_df.index)))

    # Fastball pct from pitcher mix
    sp_df['sp_fastball_pct'] = sp_df['sp_id'].map(
        lambda x: sum(pitcher_mix.get(int(x), {}).get(f'pct_{pt}', 0)
                      for pt in ('ff', 'si')) if pd.notna(x) else DEFAULTS['sp_fastball_pct'])

    # Fill defaults for pitcher
    for col, default in DEFAULTS.items():
        if col in sp_df.columns:
            sp_df[col] = sp_df[col].fillna(default)
        elif col in pitcher_features:
            sp_df[col] = default

    sp_df['sp_k_rate_season']   = sp_df['sp_k_rate_season'].fillna(DEFAULTS['sp_k_rate_season'])
    sp_df['sp_k9_last5']        = sp_df['sp_k9_last5'].fillna(DEFAULTS['sp_k9_last5'])
    sp_df['sp_days_rest']       = sp_df['sp_days_rest'].fillna(DEFAULTS['sp_days_rest'])
    # Cap sp_innings_season to training distribution range [0, 120]
    # During training this was cumulative IP before each game (0 early season → ~200 late)
    # At inference we have full season total which pushes late-season pitchers out of range
    # Cap at 120 IP (roughly 20 starts × 6 IP) which is the median SP workload at midseason
    sp_df['sp_innings_season_raw'] = sp_df['sp_innings_season'].fillna(DEFAULTS['sp_innings_season'])
    sp_df['sp_innings_season'] = sp_df['sp_innings_season_raw'].clip(0, 120)
    sp_df['opp_lineup_k_rate_30d'] = sp_df['opp_lineup_k_rate_30d'].fillna(DEFAULTS['opp_lineup_k_rate_30d'])
    sp_df['opp_lineup_bb_rate_30d'] = sp_df['opp_lineup_bb_rate_30d'].fillna(DEFAULTS['opp_lineup_bb_rate_30d'])
    sp_df['opp_lineup_xwoba_90']   = sp_df['opp_lineup_xwoba_90'].fillna(DEFAULTS['opp_lineup_xwoba_90'])
    sp_df['sp_bb_rate_season'] = sp_df['sp_bb_rate_season'].fillna(DEFAULTS['sp_bb_rate_season'])
    sp_df['sp_xwoba_against_season'] = sp_df['sp_xwoba_against_season'].fillna(sp_df['sp_xwoba_against_90']).fillna(DEFAULTS['sp_xwoba_against_season'])
    sp_df['sp_bb9_last5'] = sp_df['sp_bb9_last5'].fillna(DEFAULTS['sp_bb9_last5'])
    sp_df['expected_bf'] = sp_df['expected_bf'].fillna(DEFAULTS['expected_bf'])
    sp_df['expected_ip'] = sp_df['expected_ip'].fillna(DEFAULTS['expected_ip'])
    sp_df['sp_er_last5'] = sp_df['sp_er_last5'].fillna(DEFAULTS['sp_er_last5'])
    sp_df['umpire_k_boost']     = sp_df.get('umpire_k_rate_boost', pd.Series(0.0, index=sp_df.index)).fillna(0.0)
    sp_df['umpire_runs_boost']  = sp_df.get('umpire_runs_boost', pd.Series(0.0, index=sp_df.index)).fillna(0.0)
    sp_df['park_runs_factor']   = sp_df.get('park_runs_factor', pd.Series(1.0, index=sp_df.index)).fillna(1.0)
    sp_df['is_defaulted'] = (
        sp_df.get('sp_sd_bf', pd.Series(0, index=sp_df.index)).fillna(0) <= 0
    ) | sp_df['sp_k_rate_season'].isna()
    sp_df.loc[sp_df['is_defaulted'], 'sp_bb_rate_season'] = DEFAULTS['sp_bb_rate_season']

    log.info("Pitcher feature diagnostics:")
    for _, row in sp_df.iterrows():
        lam = row.get('lambda_k', 'not yet computed')
        lam_str = f"{lam:.2f}" if isinstance(lam, (int, float)) and pd.notna(lam) else str(lam)
        log.info(
            f"  {row.get('sp_name', '?')}: "
            f"k_rate={row['sp_k_rate_season']:.3f} "
            f"k9_last5={row['sp_k9_last5']:.1f} "
            f"innings={row['sp_innings_season']:.0f} "
            f"(raw={row.get('sp_innings_season_raw', row['sp_innings_season']):.0f}) "
            f"fastball={row['sp_fastball_pct']:.2f} "
            f"opp_k_rate={row['opp_lineup_k_rate_30d']:.3f} "
            f"lambda={lam_str}"
        )

    # ── Run pitcher model ──
    log.info(f"Running pitcher model on {len(sp_df):,} SPs...")
    X_pitcher = pitcher_scaler.transform(sp_df[pitcher_features].fillna(0))
    lambda_k = pitcher_model.predict_lambda(X_pitcher)
    sp_df['lambda_k'] = lambda_k

    # Exact PMF: P(K=k) for k=0..10
    for k in range(11):
        sp_df[f'p_k{k}'] = stats.poisson.pmf(k, lambda_k)
    sp_df['p_k10plus'] = 1.0 - stats.poisson.cdf(9, lambda_k)

    # Over/under: P(K > threshold) for common lines
    for thresh_int in range(10):  # 0.5 through 9.5
        thresh = thresh_int + 0.5
        col = f'p_over_{thresh_int}_5'
        sp_df[col] = 1.0 - stats.poisson.cdf(thresh_int, lambda_k)

    log.info("\nPitcher K predictions:")
    log.info(f"  {'Pitcher':<25} {'λ':>6} {'>4.5':>6} {'>5.5':>6} {'>6.5':>6} {'>7.5':>6}")
    for _, row in sp_df.iterrows():
        log.info(f"  {str(row.get('sp_name','?')):<25} "
                 f"{row['lambda_k']:>5.2f}  "
                 f"{row['p_over_4_5']*100:>5.1f}%  "
                 f"{row['p_over_5_5']*100:>5.1f}%  "
                 f"{row['p_over_6_5']*100:>5.1f}%  "
                 f"{row['p_over_7_5']*100:>5.1f}%")

    if walks_bundle is not None:
        run_extra_pitcher_model(sp_df, walks_bundle, "lambda_walks", "walks", WALKS_THRESHOLDS)
    if hits_bundle is not None:
        run_extra_pitcher_model(sp_df, hits_bundle, "lambda_hits", "hits", HITS_THRESHOLDS)
    if er_bundle is not None:
        run_extra_pitcher_model(sp_df, er_bundle, "lambda_er", "er", ER_THRESHOLDS)

    if args.dry_run:
        log.info("\nDRY RUN — not writing to Postgres")
        return

    # ── Write to Postgres ──
    log.info("\nEnsuring tables exist...")
    ensure_tables(engine, args.schema)

    # Delete today's existing predictions before inserting fresh ones
    with engine.begin() as conn:
        if not args.pitchers_only:
            conn.execute(text(
                f"DELETE FROM {args.schema}.player_prop_predictions "
                f"WHERE game_date = :d"), {"d": inference_date})
        conn.execute(text(
            f"DELETE FROM {args.schema}.pitcher_prop_predictions "
            f"WHERE game_date = :d"), {"d": inference_date})

    # Write batter predictions
    if not args.pitchers_only:
        batter_out = df[[
            'game_id', 'game_date', 'batter_id', 'batter_name', 'team_id',
            'batting_order', 'sp_id', 'sp_name',
            'p_hit', 'p_2plus_hits', 'p_hr', 'p_k', 'p_2plus_bases', 'p_walk',
            'matchup_score', 'platoon_advantage',
            'batter_xwoba_season', 'batter_hit_rate_30d',
            'lineup_confirmed',
        ]].copy()
        confirmed_mask = batter_out['lineup_confirmed'].fillna(False).astype(bool)
        batter_out.loc[~confirmed_mask, 'batting_order'] = None
        batter_out.loc[confirmed_mask, 'batting_order'] = (
            pd.to_numeric(batter_out.loc[confirmed_mask, 'batting_order'], errors='coerce')
            .astype('Int64')
        )
        batter_out['lineup_confirmed'] = confirmed_mask
        batter_out['as_of_ts'] = now
        batter_out['model_version'] = 'v1'
        batter_out.to_sql('player_prop_predictions', engine, schema=args.schema,
                          if_exists='append', index=False, method='multi')
        log.info(f"Wrote {len(batter_out):,} batter prop rows")

    # Write pitcher predictions
    k_cols = [f'p_k{k}' for k in range(11)] + ['p_k10plus']
    over_cols = [f'p_over_{i}_5' for i in range(10)]
    walks_cols = [f"p_walks_over_{str(t).replace('.', '_')}" for t in WALKS_THRESHOLDS]
    hits_cols = [f"p_hits_over_{str(t).replace('.', '_')}" for t in HITS_THRESHOLDS]
    er_cols = [f"p_er_over_{str(t).replace('.', '_')}" for t in ER_THRESHOLDS]
    extra_cols = ['lambda_walks', 'lambda_hits', 'lambda_er'] + walks_cols + hits_cols + er_cols
    pitcher_cols = [
        'game_id', 'game_date', 'sp_id', 'sp_name', 'is_home',
        'lambda_k',
        *k_cols,
        *over_cols,
        *extra_cols,
        'sp_k_rate_season', 'sp_innings_season', 'opp_lineup_k_rate_30d',
        'sp_bb_rate_season', 'sp_xwoba_against_90', 'opp_lineup_bb_rate_30d', 'opp_lineup_xwoba_90',
        'expected_ip', 'is_defaulted',
    ]
    pitcher_out = sp_df[[c for c in pitcher_cols if c in sp_df.columns]].rename(columns={
        'sp_id': 'pitcher_id',
        'sp_name': 'pitcher_name',
        'opp_lineup_k_rate_30d': 'opp_lineup_k_rate',
        'opp_lineup_bb_rate_30d': 'opp_lineup_bb_rate',
        'opp_lineup_xwoba_90': 'opp_lineup_xwoba',
    }).copy()
    pitcher_out['as_of_ts'] = now
    pitcher_out['model_version'] = 'v1'
    pitcher_out.to_sql('pitcher_prop_predictions', engine, schema=args.schema,
                       if_exists='append', index=False, method='multi')
    log.info(f"Wrote {len(pitcher_out):,} pitcher prop rows")

    log.info("Done.")


if __name__ == "__main__":
    main()