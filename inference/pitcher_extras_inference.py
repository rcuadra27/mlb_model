"""Inference-time feature defaults and Poisson over-line helpers for pitcher extras."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

PITCHER_EXTRA_DEFAULTS = {
    "sp_bb_rate_season": 0.085,
    "sp_bb9_last5": 3.0,
    "opp_lineup_bb_rate_30d": 0.085,
    "expected_bf": 22.0,
    "expected_ip": 5.5,
    "sp_er_last5": 2.5,
    "sp_xwoba_against_season": 0.320,
    "umpire_runs_boost": 0.0,
}

WALKS_THRESHOLDS = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
HITS_THRESHOLDS = [3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
ER_THRESHOLDS = [1.5, 2.5, 3.5, 4.5, 5.5]

PITCHER_EXTRA_COLUMNS = [
    "lambda_walks", "lambda_hits", "lambda_er",
    "sp_bb_rate_season", "sp_xwoba_against_90", "opp_lineup_bb_rate", "opp_lineup_xwoba",
] + [f"p_walks_over_{str(t).replace('.', '_')}" for t in WALKS_THRESHOLDS] \
  + [f"p_hits_over_{str(t).replace('.', '_')}" for t in HITS_THRESHOLDS] \
  + [f"p_er_over_{str(t).replace('.', '_')}" for t in ER_THRESHOLDS]


def fill_pitcher_extra_matrix(df: pd.DataFrame, features: list) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for feat in features:
        default = PITCHER_EXTRA_DEFAULTS.get(feat, 0.0)
        if feat in df.columns:
            out[feat] = df[feat].fillna(default)
        else:
            out[feat] = default
    return out


def add_poisson_over_lines(
    sp_df: pd.DataFrame,
    lam_col: str,
    prefix: str,
    thresholds: list[float],
    over_calib_factors: dict | None = None,
) -> None:
    lam = sp_df[lam_col].to_numpy(dtype=float)
    factors = over_calib_factors or {}
    for thresh in thresholds:
        k_floor = int(thresh + 0.5)
        col = f"p_{prefix}_over_{str(thresh).replace('.', '_')}"
        factor = float(factors.get(str(thresh), factors.get(thresh, 1.0)))
        sp_df[col] = np.clip(1.0 - stats.poisson.cdf(k_floor - 1, lam), 0.0, 1.0) * factor


def run_extra_pitcher_model(sp_df: pd.DataFrame, bundle: dict, lambda_col: str, prefix: str, thresholds: list[float]) -> None:
    features = bundle["features"]
    defaults = bundle.get("defaults") or {}
    X = fill_pitcher_extra_matrix(sp_df, features)
    for col, val in defaults.items():
        if col in X.columns:
            X[col] = X[col].fillna(val)
    lam = bundle["model"].predict_lambda(bundle["scaler"].transform(X))
    lam *= float(bundle.get("lambda_scale") or 1.0)
    sp_df[lambda_col] = lam
    add_poisson_over_lines(
        sp_df, lambda_col, prefix, thresholds,
        over_calib_factors=bundle.get("over_calib_factors"),
    )
