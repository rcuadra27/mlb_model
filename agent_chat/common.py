"""Shared helpers for the Hot Corner agent."""

from __future__ import annotations

import time
import unicodedata

from config import BQ_TABLE, PREDICTION_CACHE_TTL_SECONDS


def _stored_ou_is_pass_like(raw: object) -> bool:
    if raw is None:
        return False
    s = str(raw).strip().upper()
    if not s:
        return False
    return s in ("PUSH", "PASS")


def _row_to_dict(row) -> dict:
    out = dict(row)
    for k, v in list(out.items()):
        if hasattr(v, "isoformat"):
            out[k] = v.isoformat()
    return out


def _safe_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _cache_get(cache: dict, key):
    item = cache.get(key)
    if not item:
        return None
    expires_at, payload = item
    if expires_at <= time.time():
        cache.pop(key, None)
        return None
    return payload


def _cache_set(cache: dict, key, payload, ttl: int | None = None):
    cache[key] = (time.time() + (ttl if ttl is not None else PREDICTION_CACHE_TTL_SECONDS), payload)
    return payload


def _pct(val) -> float | None:
    if val is None:
        return None
    try:
        return round(float(val) * 100, 1)
    except (TypeError, ValueError):
        return None


def _latest_snapshot_cte(where: str) -> str:
    return f"""
        WITH latest AS (
            SELECT *,
                ROW_NUMBER() OVER (PARTITION BY game_id ORDER BY as_of_ts DESC) AS rn
            FROM `{BQ_TABLE}`
            WHERE {where}
        )
        SELECT * FROM latest WHERE rn = 1
    """


TEAM_ABBR_BY_NAME = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Oakland Athletics": "OAK",
    "Athletics": "OAK",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",
    "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}


BatterPropSpec = dict[str, str]
BATTER_TOP_PROP_MAP: dict[str, BatterPropSpec] = {
    "hit": {"column": "p_hit", "label": "1+ hit probability"},
    "hits": {"column": "p_hit", "label": "1+ hit probability"},
    "1_hit": {"column": "p_hit", "label": "1+ hit probability"},
    "1plus_hit": {"column": "p_hit", "label": "1+ hit probability"},
    "2plus_hits": {"column": "p_2plus_hits", "label": "2+ hits probability"},
    "2_hits": {"column": "p_2plus_hits", "label": "2+ hits probability"},
    "two_hits": {"column": "p_2plus_hits", "label": "2+ hits probability"},
    "hr": {"column": "p_hr", "label": "home run probability"},
    "homer": {"column": "p_hr", "label": "home run probability"},
    "home_run": {"column": "p_hr", "label": "home run probability"},
    "k": {"column": "p_k", "label": "batter strikeout probability"},
    "batter_k": {"column": "p_k", "label": "batter strikeout probability"},
    "strikeout": {"column": "p_k", "label": "batter strikeout probability"},
    "walk": {"column": "p_walk", "label": "walk probability"},
    "bb": {"column": "p_walk", "label": "walk probability"},
    "2plus_bases": {"column": "p_2plus_bases", "label": "2+ total bases probability"},
    "2_tb": {"column": "p_2plus_bases", "label": "2+ total bases probability"},
    "tb": {"column": "p_2plus_bases", "label": "2+ total bases probability"},
    "total_bases": {"column": "p_2plus_bases", "label": "2+ total bases probability"},
}

PITCHER_TOP_PROP_MAP: dict[str, BatterPropSpec] = {
    "expected_k": {"column": "lambda_k", "label": "expected strikeouts"},
    "pitcher_k": {"column": "lambda_k", "label": "expected strikeouts"},
    "lambda_k": {"column": "lambda_k", "label": "expected strikeouts"},
    "k_over_3_5": {"column": "p_over_3_5", "label": "pitcher over 3.5 K probability"},
    "k_over_4_5": {"column": "p_over_4_5", "label": "pitcher over 4.5 K probability"},
    "k_over_5_5": {"column": "p_over_5_5", "label": "pitcher over 5.5 K probability"},
    "k_over_6_5": {"column": "p_over_6_5", "label": "pitcher over 6.5 K probability"},
    "k_over_7_5": {"column": "p_over_7_5", "label": "pitcher over 7.5 K probability"},
    "k_over_8_5": {"column": "p_over_8_5", "label": "pitcher over 8.5 K probability"},
    "k_over_9_5": {"column": "p_over_9_5", "label": "pitcher over 9.5 K probability"},
}


def _normalize_prop_type(prop_type: str | None) -> str:
    raw = (prop_type or "hit").strip().lower()
    return raw.replace("+", "plus").replace(" ", "_").replace("-", "_").replace(".", "_")


def _normalize_name_for_match(name: str) -> str:
    """Lowercase, strip accents, collapse whitespace — for player name matching."""
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", str(name).strip().lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.split())


def _name_match_score(query: str, full_name: str) -> int:
    q = _normalize_name_for_match(query)
    n = _normalize_name_for_match(full_name)
    if not q or not n:
        return 0
    if q == n:
        return 100
    parts = n.split()
    last = parts[-1] if parts else ""
    q_tokens = q.split()
    if len(q_tokens) > 1:
        if all(tok in parts for tok in q_tokens):
            return 98
        if all(tok in n for tok in q_tokens):
            return 92
    if q == last:
        return 95
    if q in n:
        return 85
    if q in last:
        return 75
    if len(q_tokens) == 1 and q_tokens[0] == last:
        return 95
    return 0
