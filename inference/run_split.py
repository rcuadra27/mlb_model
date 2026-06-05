"""Strength-based split of predicted game total into home/away runs."""

from __future__ import annotations

import math

HOME_FIELD_WEIGHT_BOOST = 1.06
LEAGUE_RG_RA_DEFAULT = 4.5


def _safe_runs_rate(value, fallback: float) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return fallback
    if math.isnan(x) or math.isinf(x) or x <= 0:
        return fallback
    return x


def split_total(
    pred_total: float,
    *,
    away_runs_scored: float | None = None,
    home_runs_scored: float | None = None,
    away_runs_allowed: float | None = None,
    home_runs_allowed: float | None = None,
    league_avg_runs: float | None = None,
    home_field_weight_boost: float = HOME_FIELD_WEIGHT_BOOST,
) -> tuple[float, float]:
    """
    Split predicted total into home/away runs using matchup strength.

    Each side's weight = (team R/G) × (opponent RA/G). Home weight gets a modest
    HFA multiplier before normalization.
    """
    if pred_total is None or (isinstance(pred_total, float) and (math.isnan(pred_total) or pred_total <= 0)):
        return 0.0, 0.0

    lg = _safe_runs_rate(league_avg_runs, LEAGUE_RG_RA_DEFAULT)
    away_rs = _safe_runs_rate(away_runs_scored, lg)
    home_rs = _safe_runs_rate(home_runs_scored, lg)
    away_ra = _safe_runs_rate(away_runs_allowed, lg)
    home_ra = _safe_runs_rate(home_runs_allowed, lg)

    away_weight = away_rs * home_ra
    home_weight = home_rs * away_ra * home_field_weight_boost
    denom = away_weight + home_weight
    if denom <= 0:
        away_share = 0.47
    else:
        away_share = away_weight / denom

    away_runs = round(float(pred_total) * away_share, 3)
    home_runs = round(float(pred_total) - away_runs, 3)
    return home_runs, away_runs
