"""
Shared training utilities for pitcher Poisson prop models (K, walks, hits, ER).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, text

from models.poisson_regressor import PoissonRegressor

SP_MIN_BF = 15
LEAGUE_BB_RATE = 0.085
LEAGUE_XWOBA_AGAINST = 0.320
LEAGUE_XWOBA = 0.320
LEAGUE_ER_PER_START = 2.5
LEAGUE_WALKS_PER_START = 1.8
LEAGUE_HITS_PER_START = 5.2
LEAGUE_BF_PER_START = 22.0
LEAGUE_IP_PER_START = 5.5

SP_GAME_TARGETS_QUERY = """
SELECT
    sp.pitcher                                              AS pitcher_id,
    sp.game_pk                                              AS game_id,
    sp.game_date,
    EXTRACT(YEAR FROM sp.game_date)::int                    AS season,
    CASE WHEN sp.inning_topbot = 'Top' THEN g.away_team_id
         ELSE g.home_team_id END                            AS team_id,
    CASE WHEN sp.inning_topbot = 'Top' THEN g.home_team_id
         ELSE g.away_team_id END                            AS opp_team_id,
    SUM(CASE WHEN sp.events = 'strikeout' THEN 1 ELSE 0 END) AS k_count,
    COUNT(DISTINCT sp.at_bat_number)                        AS batters_faced,
    SUM(CASE WHEN sp.events IN ('walk','hit_by_pitch')
             THEN 1 ELSE 0 END)                             AS walks,
    SUM(CASE WHEN sp.events IN ('single','double','triple','home_run')
             THEN 1 ELSE 0 END)                             AS hits_allowed
FROM public.statcast_pitches sp
JOIN public.games g ON g.game_id = sp.game_pk
WHERE sp.game_date BETWEEN :start_date AND :end_date
  AND sp.pitcher IS NOT NULL
  AND sp.game_pk IS NOT NULL
  AND sp.inning <= 9
GROUP BY sp.pitcher, sp.game_pk, sp.game_date, g.away_team_id, g.home_team_id,
         CASE WHEN sp.inning_topbot = 'Top' THEN g.away_team_id ELSE g.home_team_id END,
         CASE WHEN sp.inning_topbot = 'Top' THEN g.home_team_id ELSE g.away_team_id END
HAVING COUNT(DISTINCT sp.at_bat_number) >= :min_bf
ORDER BY sp.game_date, sp.game_pk, sp.pitcher;
"""

SP_ER_TARGETS_QUERY = """
WITH sp_starters AS (
    SELECT
        sp.pitcher                                              AS pitcher_id,
        sp.game_pk                                              AS game_id,
        sp.game_date,
        EXTRACT(YEAR FROM sp.game_date)::int                    AS season,
        CASE WHEN sp.inning_topbot = 'Top' THEN g.away_team_id
             ELSE g.home_team_id END                            AS team_id,
        CASE WHEN sp.inning_topbot = 'Top' THEN g.home_team_id
             ELSE g.away_team_id END                            AS opp_team_id
    FROM public.statcast_pitches sp
    JOIN public.games g ON g.game_id = sp.game_pk
    WHERE sp.game_date BETWEEN :start_date AND :end_date
      AND sp.pitcher IS NOT NULL
      AND sp.game_pk IS NOT NULL
      AND sp.inning <= 9
    GROUP BY sp.pitcher, sp.game_pk, sp.game_date, g.away_team_id, g.home_team_id,
             CASE WHEN sp.inning_topbot = 'Top' THEN g.away_team_id ELSE g.home_team_id END,
             CASE WHEN sp.inning_topbot = 'Top' THEN g.home_team_id ELSE g.away_team_id END
    HAVING COUNT(DISTINCT sp.at_bat_number) >= :min_bf
),
starter_er AS (
    SELECT
        s.pitcher_id,
        s.game_id,
        COALESCE(ps.earned_runs, pa.earned_runs) AS er_count,
        COALESCE(ps.innings_pitched, pa.innings_pitched) AS innings_pitched
    FROM sp_starters s
    LEFT JOIN public.pitcher_starts ps
      ON ps.pitcher_id = s.pitcher_id AND ps.game_id = s.game_id
    LEFT JOIN (
        SELECT game_id, pitcher_id,
               MAX(earned_runs) AS earned_runs,
               MAX(innings_pitched) AS innings_pitched
        FROM public.pitcher_appearances
        WHERE is_starter IS TRUE AND earned_runs IS NOT NULL
        GROUP BY game_id, pitcher_id
    ) pa ON pa.pitcher_id = s.pitcher_id AND pa.game_id = s.game_id
)
SELECT
    s.pitcher_id,
    s.game_id,
    s.game_date,
    s.season,
    s.team_id,
    s.opp_team_id,
    er.er_count,
    er.innings_pitched
FROM sp_starters s
JOIN starter_er er
  ON er.pitcher_id = s.pitcher_id AND er.game_id = s.game_id
WHERE er.er_count IS NOT NULL
ORDER BY s.game_date, s.game_id, s.pitcher_id;
"""

SP_RATE_SEASON_QUERY = """
WITH sp_game_stats AS (
    SELECT
        pitcher AS pitcher_id,
        game_pk AS game_id,
        game_date,
        EXTRACT(YEAR FROM game_date)::int AS season,
        SUM(CASE WHEN events = 'strikeout' THEN 1 ELSE 0 END) AS ks,
        SUM(CASE WHEN events IN ('walk','hit_by_pitch') THEN 1 ELSE 0 END) AS walks,
        COUNT(DISTINCT at_bat_number) AS bf,
        AVG(CASE WHEN woba_denom = 1 AND estimated_woba_using_speedangle IS NOT NULL
                 THEN estimated_woba_using_speedangle END) AS xwoba_against_game
    FROM public.statcast_pitches
    WHERE game_date BETWEEN :start_date AND :end_date
      AND pitcher IS NOT NULL
      AND events IS NOT NULL
    GROUP BY pitcher, game_pk, game_date
    HAVING COUNT(DISTINCT at_bat_number) >= :min_bf
),
sp_cumulative AS (
    SELECT pitcher_id, game_id, game_date, season,
        SUM(ks) OVER w AS sd_ks,
        SUM(walks) OVER w AS sd_walks,
        SUM(bf) OVER w AS sd_bf,
        SUM(CASE WHEN xwoba_against_game IS NOT NULL THEN xwoba_against_game * bf ELSE 0 END)
            OVER w AS sd_xwoba_sum,
        SUM(CASE WHEN xwoba_against_game IS NOT NULL THEN bf ELSE 0 END)
            OVER w AS sd_xwoba_bf
    FROM sp_game_stats
    WINDOW w AS (
        PARTITION BY pitcher_id, season
        ORDER BY game_date, game_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    )
)
SELECT pitcher_id, game_id, game_date, season,
    (COALESCE(sd_ks, 0) + 200 * 0.230) / (COALESCE(sd_bf, 0) + 200) AS sp_k_rate_season,
    (COALESCE(sd_walks, 0) + 200 * 0.085) / (COALESCE(sd_bf, 0) + 200) AS sp_bb_rate_season,
    CASE WHEN COALESCE(sd_xwoba_bf, 0) > 0
         THEN sd_xwoba_sum / sd_xwoba_bf
         ELSE NULL END AS sp_xwoba_against_season,
    COALESCE(sd_bf, 0) AS sp_sd_bf
FROM sp_cumulative;
"""

SP_FEATURES_QUERY = """
SELECT
    fg.game_id,
    g.home_team_id,
    g.away_team_id,
    fg.home_sp_k9_last5,
    fg.away_sp_k9_last5,
    fg.home_sp_xwoba_against_90,
    fg.away_sp_xwoba_against_90,
    fg.home_sp_bb_rate_90,
    fg.away_sp_bb_rate_90,
    fg.home_sp_days_rest,
    fg.away_sp_days_rest,
    fg.umpire_k_rate_boost,
    fg.umpire_runs_boost,
    COALESCE(fg.park_runs_factor_blended, fg.park_runs_factor, 1.0) AS park_runs_factor,
    fg.home_lineup_k_rate_90,
    fg.away_lineup_k_rate_90,
    fg.home_lineup_bb_rate_90,
    fg.away_lineup_bb_rate_90,
    fg.home_lineup_xwoba_90,
    fg.away_lineup_xwoba_90
FROM public.features_game fg
JOIN public.games g ON g.game_id = fg.game_id
WHERE fg.game_date BETWEEN :start_date AND :end_date;
"""

SP_INNINGS_QUERY = """
SELECT ps.pitcher_id, ps.game_id, ps.game_date,
    SUM(ps.innings_pitched) OVER (
        PARTITION BY ps.pitcher_id, EXTRACT(YEAR FROM ps.game_date)::int
        ORDER BY ps.game_date, ps.game_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS sp_innings_season
FROM public.pitcher_starts ps
WHERE ps.game_date BETWEEN :start_date AND :end_date
  AND ps.innings_pitched IS NOT NULL;
"""

SP_ROLLING_QUERY = """
WITH sp_games AS (
    SELECT pitcher AS pitcher_id, game_pk AS game_id, game_date,
           COUNT(DISTINCT at_bat_number) AS batters_faced
    FROM public.statcast_pitches
    WHERE game_date BETWEEN :start_date AND :end_date
      AND pitcher IS NOT NULL
    GROUP BY pitcher, game_pk, game_date
    HAVING COUNT(DISTINCT at_bat_number) >= :min_bf
),
starts AS (
    SELECT pitcher_id, game_id, game_date,
           innings_pitched,
           walks_allowed,
           earned_runs
    FROM public.pitcher_starts
    WHERE game_date BETWEEN :start_date AND :end_date
)
SELECT
    g.pitcher_id,
    g.game_id,
    g.game_date,
    AVG(g.batters_faced) OVER (
        PARTITION BY g.pitcher_id
        ORDER BY g.game_date, g.game_id
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    ) AS expected_bf,
    AVG(s.innings_pitched) OVER (
        PARTITION BY g.pitcher_id
        ORDER BY g.game_date, g.game_id
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    ) AS expected_ip,
    AVG(s.walks_allowed * 9.0 / NULLIF(s.innings_pitched, 0)) OVER (
        PARTITION BY g.pitcher_id
        ORDER BY g.game_date, g.game_id
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    ) AS sp_bb9_last5,
    AVG(s.earned_runs) OVER (
        PARTITION BY g.pitcher_id
        ORDER BY g.game_date, g.game_id
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    ) AS sp_er_last5
FROM sp_games g
LEFT JOIN starts s
  ON s.pitcher_id = g.pitcher_id AND s.game_id = g.game_id;
"""


@dataclass
class PropModelSpec:
    name: str
    artifact: str
    target_col: str
    features: list[str]
    calib_thresholds: list[float]
    league_mean: float
    defaults: dict[str, float] = field(default_factory=dict)
    alpha_by_calibration: bool = False
    lambda_scale_calibrate: bool = False
    over_prob_calibrate: bool = False


WALKS_SPEC = PropModelSpec(
    name="walks",
    artifact="pitcher_walks_v1.joblib",
    target_col="walks",
    features=[
        "sp_bb_rate_season",
        "sp_bb9_last5",
        "opp_lineup_bb_rate_30d",
        "expected_bf",
        "sp_innings_season",
        "sp_days_rest",
    ],
    calib_thresholds=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
    league_mean=LEAGUE_WALKS_PER_START,
    defaults={
        "sp_bb_rate_season": LEAGUE_BB_RATE,
        "sp_bb9_last5": 3.0,
        "opp_lineup_bb_rate_30d": LEAGUE_BB_RATE,
        "expected_bf": LEAGUE_BF_PER_START,
        "sp_innings_season": 0.0,
        "sp_days_rest": 4.0,
    },
)

HITS_SPEC = PropModelSpec(
    name="hits",
    artifact="pitcher_hits_v1.joblib",
    target_col="hits_allowed",
    features=[
        "sp_xwoba_against_season",
        "sp_xwoba_against_90",
        "opp_lineup_xwoba_90",
        "expected_bf",
        "park_runs_factor",
        "sp_innings_season",
        "sp_days_rest",
    ],
    calib_thresholds=[3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
    league_mean=LEAGUE_HITS_PER_START,
    defaults={
        "sp_xwoba_against_season": LEAGUE_XWOBA_AGAINST,
        "sp_xwoba_against_90": LEAGUE_XWOBA_AGAINST,
        "opp_lineup_xwoba_90": LEAGUE_XWOBA,
        "expected_bf": LEAGUE_BF_PER_START,
        "park_runs_factor": 1.0,
        "sp_innings_season": 0.0,
        "sp_days_rest": 4.0,
    },
)

ER_SPEC = PropModelSpec(
    name="er",
    artifact="pitcher_er_v1.joblib",
    target_col="er_count",
    features=[
        "sp_er_last5",
        "sp_xwoba_against_season",
        "sp_xwoba_against_90",
        "opp_lineup_xwoba_90",
        "park_runs_factor",
        "umpire_runs_boost",
        "expected_ip",
        "sp_innings_season",
        "sp_days_rest",
    ],
    calib_thresholds=[1.5, 2.5, 3.5, 4.5, 5.5],
    league_mean=LEAGUE_ER_PER_START,
    defaults={
        "sp_er_last5": LEAGUE_ER_PER_START,
        "sp_xwoba_against_season": LEAGUE_XWOBA_AGAINST,
        "sp_xwoba_against_90": LEAGUE_XWOBA_AGAINST,
        "opp_lineup_xwoba_90": LEAGUE_XWOBA,
        "park_runs_factor": 1.0,
        "umpire_runs_boost": 0.0,
        "expected_ip": LEAGUE_IP_PER_START,
        "sp_innings_season": 0.0,
        "sp_days_rest": 4.0,
    },
    alpha_by_calibration=True,
    lambda_scale_calibrate=True,
    over_prob_calibrate=True,
)


def load_table(engine, query, params, label):
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params=params)
    print(f"  {label}: {len(df):,} rows")
    return df


def _asof_sort(frame: pd.DataFrame, by: str, on: str) -> pd.DataFrame:
    out = frame.copy()
    out[on] = pd.to_datetime(out[on], utc=False).dt.normalize()
    out[by] = pd.to_numeric(out[by], errors="coerce")
    out = out.dropna(subset=[by, on])
    out[by] = out[by].astype("int64")
    return out.sort_values(on, kind="mergesort").reset_index(drop=True)


def _raw_over_probs(lam, thresh):
    k_floor = int(thresh + 0.5)
    return 1.0 - stats.poisson.cdf(k_floor - 1, np.asarray(lam, dtype=float))


def _over_factor(over_calib_factors, thresh) -> float:
    if not over_calib_factors:
        return 1.0
    return float(
        over_calib_factors.get(str(thresh), over_calib_factors.get(thresh, 1.0))
    )


def evaluate_poisson(label, y_true, lam_pred, thresholds, over_calib_factors=None):
    y_true = np.asarray(y_true, dtype=float)
    lam = np.asarray(lam_pred, dtype=float)
    mae = mean_absolute_error(y_true, lam)
    rmse = np.sqrt(((y_true - lam) ** 2).mean())
    bias = (lam - y_true).mean()
    ll = stats.poisson.logpmf(y_true.astype(int), lam).mean()
    naive_lam = y_true.mean()
    naive_ll = stats.poisson.logpmf(y_true.astype(int), naive_lam).mean()

    print(f"\n  {'─'*65}")
    print(f"  {label}  (N={len(y_true):,})")
    print(f"  {'─'*65}")
    print(f"  Mean actual: {y_true.mean():.3f}  std: {y_true.std():.3f}")
    print(f"  Mean λ pred: {lam.mean():.3f}  std: {lam.std():.3f}")
    print(f"  MAE: {mae:.4f}  RMSE: {rmse:.4f}  Bias: {bias:+.4f}")
    print(f"  Log-lik: {ll:.4f}  naive: {naive_ll:.4f}  Δ: {ll - naive_ll:+.4f}")
    print("  Over/under calibration:")
    calib_rows = []
    max_gap = 0.0
    for thresh in thresholds:
        k_floor = int(thresh + 0.5)
        actual_over = (y_true >= k_floor).mean()
        factor = _over_factor(over_calib_factors, thresh)
        pred_p_over = np.clip(_raw_over_probs(lam, thresh) * factor, 0.0, 1.0).mean()
        gap = abs(actual_over - pred_p_over)
        max_gap = max(max_gap, gap)
        suffix = f"  (×{factor:.3f})" if factor != 1.0 else ""
        print(
            f"    Over {thresh}: actual={actual_over*100:.1f}%  "
            f"model={pred_p_over*100:.1f}%  gap={gap*100:.1f}pp{suffix}"
        )
        calib_rows.append({
            "line": f"Over {thresh}",
            "actual_pct": round(actual_over * 100, 1),
            "pred_pct": round(pred_p_over * 100, 1),
            "gap_pp": round(gap * 100, 1),
            "calib_factor": round(factor, 3),
        })
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "bias": float(bias),
        "log_lik": float(ll),
        "naive_log_lik": float(naive_ll),
        "log_lik_gain": float(ll - naive_ll),
        "max_calib_gap_pp": round(max_gap * 100, 1),
        "calibration": calib_rows,
    }


def merge_base_frame(engine, params, target_df: pd.DataFrame) -> pd.DataFrame:
    sp_stats = load_table(engine, SP_RATE_SEASON_QUERY, params, "SP season rates")
    fg_feats = load_table(
        engine,
        SP_FEATURES_QUERY,
        {"start_date": params["start_date"], "end_date": params["end_date"]},
        "features_game",
    )
    sp_innings = load_table(
        engine,
        SP_INNINGS_QUERY,
        {"start_date": params["start_date"], "end_date": params["end_date"]},
        "SP innings",
    )
    rolling = load_table(engine, SP_ROLLING_QUERY, params, "SP rolling context")

    df = target_df.merge(
        sp_stats[["pitcher_id", "game_id", "sp_k_rate_season", "sp_bb_rate_season",
                  "sp_xwoba_against_season", "sp_sd_bf"]],
        on=["pitcher_id", "game_id"],
        how="left",
    )
    df = df.merge(fg_feats, on="game_id", how="left")
    df = df.merge(sp_innings[["pitcher_id", "game_id", "sp_innings_season"]], on=["pitcher_id", "game_id"], how="left")
    df = df.merge(
        rolling[["pitcher_id", "game_id", "expected_bf", "expected_ip", "sp_bb9_last5", "sp_er_last5"]],
        on=["pitcher_id", "game_id"],
        how="left",
    )

    is_home = df["team_id"] == df["home_team_id"]
    df["sp_k9_last5"] = np.where(is_home, df["home_sp_k9_last5"], df["away_sp_k9_last5"])
    df["sp_xwoba_against_90"] = np.where(is_home, df["home_sp_xwoba_against_90"], df["away_sp_xwoba_against_90"])
    df["sp_bb_rate_90"] = np.where(is_home, df["home_sp_bb_rate_90"], df["away_sp_bb_rate_90"])
    df["sp_days_rest"] = np.where(is_home, df["home_sp_days_rest"], df["away_sp_days_rest"])
    df["opp_lineup_k_rate_30d"] = np.where(is_home, df["away_lineup_k_rate_90"], df["home_lineup_k_rate_90"])
    df["opp_lineup_bb_rate_30d"] = np.where(is_home, df["away_lineup_bb_rate_90"], df["home_lineup_bb_rate_90"])
    df["opp_lineup_xwoba_90"] = np.where(is_home, df["away_lineup_xwoba_90"], df["home_lineup_xwoba_90"])
    df["umpire_k_boost"] = pd.to_numeric(df["umpire_k_rate_boost"], errors="coerce").fillna(0.0)
    df["umpire_runs_boost"] = pd.to_numeric(df["umpire_runs_boost"], errors="coerce").fillna(0.0)
    df["is_defaulted"] = (
        df["sp_sd_bf"].fillna(0) <= 0
    ) | df["sp_bb_rate_season"].isna()
    return df


def _max_calib_gap_pp(y_true, lam, thresholds, over_calib_factors=None) -> float:
    y_true = np.asarray(y_true, dtype=float)
    lam = np.asarray(lam, dtype=float)
    max_gap = 0.0
    for thresh in thresholds:
        k_floor = int(thresh + 0.5)
        actual_over = (y_true >= k_floor).mean()
        factor = _over_factor(over_calib_factors, thresh)
        pred_p_over = np.clip(_raw_over_probs(lam, thresh) * factor, 0.0, 1.0).mean()
        max_gap = max(max_gap, abs(actual_over - pred_p_over))
    return max_gap * 100.0


def _tune_over_calib_factors(y_val, lam_val, thresholds) -> dict[str, float]:
    y_val = np.asarray(y_val, dtype=float)
    lam_val = np.asarray(lam_val, dtype=float)
    factors: dict[str, float] = {}
    for thresh in thresholds:
        k_floor = int(thresh + 0.5)
        raw = _raw_over_probs(lam_val, thresh)
        pred = float(raw.mean())
        actual = float((y_val >= k_floor).mean())
        if pred > 1e-6:
            factors[str(thresh)] = float(np.clip(actual / pred, 0.5, 1.5))
        else:
            factors[str(thresh)] = 1.0
    return factors


def _tune_lambda_scale(y_val, lam_val, thresholds) -> float:
    if len(y_val) == 0:
        return 1.0
    best_scale, best_gap = 1.0, float("inf")
    for scale in np.linspace(0.82, 1.02, 41):
        gap = _max_calib_gap_pp(y_val, lam_val * scale, thresholds)
        if gap < best_gap:
            best_gap, best_scale = gap, float(scale)
    return best_scale


def train_prop_model(
    df: pd.DataFrame,
    spec: PropModelSpec,
    train_end_season: int = 2022,
    val_season: int = 2023,
    test_season: int = 2024,
):
    for col, default in spec.defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(default)
        else:
            df[col] = default

    if spec.name == "walks":
        df.loc[df["is_defaulted"], "sp_bb_rate_season"] = spec.defaults["sp_bb_rate_season"]
    if spec.name in ("hits", "er"):
        df["sp_xwoba_against_season"] = df["sp_xwoba_against_season"].fillna(
            df["sp_xwoba_against_90"]
        ).fillna(spec.defaults.get("sp_xwoba_against_season", LEAGUE_XWOBA_AGAINST))

    df["sp_innings_season"] = df["sp_innings_season"].fillna(0.0).clip(0, 120)
    df = df.dropna(subset=spec.features + [spec.target_col]).copy()

    train = df[df["season"] <= train_end_season]
    val = df[df["season"] == val_season]
    test = df[df["season"] == test_season]
    print(f"  Train:{len(train):,}  Val:{len(val):,}  Test:{len(test):,}")
    if len(train) == 0:
        raise ValueError(
            f"No {spec.name} training rows for seasons <= {train_end_season}. "
            "Check target source tables and feature coverage."
        )

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train[spec.features])
    X_v = scaler.transform(val[spec.features])
    X_te = scaler.transform(test[spec.features])
    y_tr = train[spec.target_col].values
    y_v = val[spec.target_col].values
    y_te = test[spec.target_col].values

    best_alpha, best_ll, best_calib = 0.1, -np.inf, float("inf")
    alpha_grid = [0.001, 0.01, 0.1, 1.0, 10.0]
    for alpha in alpha_grid:
        m = PoissonRegressor(alpha=alpha)
        m.fit(X_tr, y_tr)
        lam_v = m.predict_lambda(X_v)
        ll = stats.poisson.logpmf(y_v.astype(int), lam_v).mean()
        calib_gap = _max_calib_gap_pp(y_v, lam_v, spec.calib_thresholds) if len(y_v) else float("inf")
        if spec.alpha_by_calibration:
            pick = calib_gap < best_calib or (calib_gap == best_calib and ll > best_ll)
        else:
            pick = ll > best_ll
        if pick:
            best_ll, best_alpha = ll, alpha
            best_calib = calib_gap

    model = PoissonRegressor(alpha=best_alpha)
    model.fit(X_tr, y_tr)
    lambda_scale = 1.0
    over_calib_factors: dict[str, float] = {}
    lam_v_scaled = model.predict_lambda(X_v)
    if spec.lambda_scale_calibrate and len(y_v) > 0:
        lambda_scale = _tune_lambda_scale(y_v, lam_v_scaled, spec.calib_thresholds)
        lam_v_scaled = lam_v_scaled * lambda_scale
        print(
            f"  Val lambda scale: {lambda_scale:.3f}  "
            f"(val max calib gap {_max_calib_gap_pp(y_v, lam_v_scaled, spec.calib_thresholds):.1f}pp)"
        )

    if spec.over_prob_calibrate and len(y_v) > 0:
        over_calib_factors = _tune_over_calib_factors(
            y_v, lam_v_scaled, spec.calib_thresholds
        )
        print(
            f"  Val over calib factors: {over_calib_factors}  "
            f"(val max calib gap {_max_calib_gap_pp(y_v, lam_v_scaled, spec.calib_thresholds, over_calib_factors):.1f}pp)"
        )

    lam_te = model.predict_lambda(X_te) * lambda_scale
    calib_note = ""
    if lambda_scale != 1.0:
        calib_note += f" scale={lambda_scale:.3f}"
    if over_calib_factors:
        calib_note += " over-calib"
    metrics = evaluate_poisson(
        f"{spec.name} Poisson α={best_alpha}{calib_note}",
        y_te,
        lam_te,
        spec.calib_thresholds,
        over_calib_factors=over_calib_factors or None,
    )
    metrics["lambda_scale"] = lambda_scale
    metrics["alpha"] = best_alpha
    metrics["over_calib_factors"] = over_calib_factors
    return scaler, model, metrics, test, lam_te, lambda_scale, over_calib_factors


def get_engine():
    pg_dsn = os.environ.get("PG_DSN")
    if not pg_dsn:
        raise RuntimeError("PG_DSN not set")
    return create_engine(pg_dsn)
