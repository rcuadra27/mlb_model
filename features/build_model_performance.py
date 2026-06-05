#!/usr/bin/env python3
"""Precompute Model Performance tab metrics into a small snapshot table."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine, text


V9_LAUNCH_DATE = "2026-04-14"
V10_LAUNCH_DATE = "2026-05-28"
V9_END_DATE = "2026-05-27"
V10_SMALL_SAMPLE_NOTE = (
    "v10 launched May 28 — these metrics are based on a limited early sample "
    "and will stabilize as more games are graded."
)
OU_PRED_LINE_GAP = 0.5
_ML_CALIBRATION_BUCKETS = [
    ("50-55%", 0.50, 0.55),
    ("55-60%", 0.55, 0.60),
    ("60-65%", 0.60, 0.65),
    ("65%+", 0.65, 1.01),
]
OU_ACCURACY_EXCLUDED_SLATE_DATES = frozenset({
    "2026-04-23",
    "2026-04-24",
    "2026-04-25",
    "2026-04-26",
})

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS public.model_performance_snapshot (
    snapshot_date DATE PRIMARY KEY,
    version TEXT NOT NULL DEFAULT 'v10',
    min_game_date DATE NOT NULL,
    games_graded INTEGER,
    calibration_error_pct DOUBLE PRECISION,
    brier_score DOUBLE PRECISION,
    brier_market DOUBLE PRECISION,
    accuracy_pct DOUBLE PRECISION,
    pitcher_k_starters_graded INTEGER,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_model_performance_snapshot_date
    ON public.model_performance_snapshot(snapshot_date DESC);
"""


def normalize_pg_dsn(dsn: str) -> str:
    if dsn.startswith("postgresql+psycopg2://"):
        return "postgresql://" + dsn[len("postgresql+psycopg2://") :]
    if dsn.startswith("postgres+psycopg2://"):
        return "postgres://" + dsn[len("postgres+psycopg2://") :]
    return dsn


def _model_version_sql(version: str) -> str:
    if version == "v10":
        return f"""(
                  p.model_version = 'v10'
                  OR (p.model_version IS NULL AND p.game_date >= DATE '{V10_LAUNCH_DATE}')
              )"""
    if version == "v9":
        # v9 window is defined by calendar dates; include rows even if mis-tagged v10.
        return "TRUE"
    raise ValueError(f"unknown model version filter: {version}")


def fetch_graded_games_from_pg(
    dsn: str,
    min_game_date: str,
    *,
    max_game_date: str | None = None,
    model_version: str = "v10",
) -> list[dict[str, Any]]:
    version_sql = _model_version_sql(model_version)
    max_clause = ""
    params: list[Any] = [min_game_date]
    if max_game_date:
        max_clause = " AND p.game_date <= %s::date"
        params.append(max_game_date)
    q = f"""
        WITH latest AS (
            SELECT DISTINCT ON (p.game_id)
                p.game_id,
                p.game_date::text AS game_date,
                p.home_team,
                p.away_team,
                p.p_home_win_poisson::float AS p_home,
                p.p_away_win_poisson::float AS p_away,
                p.p_home_market_median::float AS p_home_market,
                p.p_away_market_median::float AS p_away_market,
                p.total_runs_pred::float AS total_runs_pred,
                p.ou_recommendation,
                COALESCE(fg.closing_ou_line, fg.morning_ou_line, p.ou_line)::float AS ou_line,
                COALESCE(fg.closing_home_price, fg.morning_home_price)::int AS home_odds,
                COALESCE(fg.closing_away_price, fg.morning_away_price)::int AS away_odds,
                g.home_runs,
                g.away_runs,
                g.status
            FROM public.inference_game_predictions p
            LEFT JOIN public.features_game fg ON fg.game_id = p.game_id
            LEFT JOIN public.games g ON g.game_id = p.game_id AND g.game_date = p.game_date
            WHERE p.game_date >= %s::date
              {max_clause}
              AND {version_sql}
            ORDER BY p.game_id, p.as_of_ts DESC NULLS LAST
        )
        SELECT * FROM latest
        WHERE home_runs IS NOT NULL
          AND away_runs IS NOT NULL
          AND (
              LOWER(COALESCE(status, '')) LIKE 'final%%'
              OR LOWER(COALESCE(status, '')) = 'game over'
              OR LOWER(COALESCE(status, '')) LIKE 'completed%%'
          )
        ORDER BY game_date, game_id
    """
    conn = psycopg2.connect(normalize_pg_dsn(dsn), connect_timeout=25)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(q, tuple(params))
            return [dict(x) for x in cur.fetchall()]
    finally:
        conn.close()


def build_v9_history(pg_dsn: str) -> dict[str, Any]:
    raw_rows = fetch_graded_games_from_pg(
        pg_dsn,
        V9_LAUNCH_DATE,
        max_game_date=V9_END_DATE,
        model_version="v9",
    )
    headline, ml_calibration = compute_ml_headline_and_calibration(raw_rows)
    return {
        "version": "v9",
        "label": "Previous model (v9), Apr 14 – May 27",
        "min_game_date": V9_LAUNCH_DATE,
        "max_game_date": V9_END_DATE,
        "headline": headline,
        "ml_calibration": ml_calibration,
    }


def ml_pnl(pick_won: bool, pick_odds) -> float:
    if pick_odds is None:
        return 0.0
    if not pick_won:
        return -10.0
    odds = float(pick_odds)
    if odds > 0:
        return 10.0 * (odds / 100.0)
    if odds < 0:
        return 10.0 * (100.0 / abs(odds))
    return 0.0


def stored_ou_is_pass_like(raw) -> bool:
    s = str(raw or "").strip().upper()
    return s in ("PUSH", "PASS")


def grade_games(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    graded = []
    for r in raw_rows:
        game_date = r.get("game_date")
        game_id = r.get("game_id")
        p_home = r.get("p_home")
        p_away = r.get("p_away")
        home_runs = r.get("home_runs")
        away_runs = r.get("away_runs")
        if home_runs is None or away_runs is None:
            continue
        home_runs = int(home_runs)
        away_runs = int(away_runs)
        total_actual = home_runs + away_runs

        if p_home is not None and p_away is not None and home_runs != away_runs:
            p_home_f = float(p_home)
            p_away_f = float(p_away)
            conf = max(p_home_f, p_away_f)
            pick_home = p_home_f >= p_away_f
            pick_won = (pick_home and home_runs > away_runs) or ((not pick_home) and away_runs > home_runs)
            if conf < 0.55:
                bucket = "50-55%"
            elif conf < 0.60:
                bucket = "55-60%"
            elif conf < 0.65:
                bucket = "60-65%"
            else:
                bucket = "65%+"
            graded.append({
                "game_date": game_date,
                "game_id": game_id,
                "kind": "ml",
                "is_hit": 1 if pick_won else 0,
                "pnl_dollars": float(ml_pnl(pick_won, r.get("home_odds") if pick_home else r.get("away_odds"))),
                "conf": conf,
                "conf_bucket": bucket,
                "ou_pick": None,
            })

        total_pred = r.get("total_runs_pred")
        ou_line = r.get("ou_line")
        if total_pred is not None and ou_line is not None:
            tp = float(total_pred)
            line = float(ou_line)
            if abs(tp - line) < OU_PRED_LINE_GAP or stored_ou_is_pass_like(r.get("ou_recommendation")):
                continue
            ou_pick = "over" if tp > line else "under"
            half_line = (
                (abs(line * 10 - int(round(line * 10))) < 1e-6)
                and (int(round(line * 10)) % 10 != 0)
            )
            if (not half_line) and total_actual == int(round(line)):
                continue
            hit = (ou_pick == "over" and total_actual > line) or (ou_pick == "under" and total_actual < line)
            graded.append({
                "game_date": game_date,
                "game_id": game_id,
                "kind": "ou",
                "is_hit": 1 if hit else 0,
                "pnl_dollars": 10.0 * (100.0 / 110.0) if hit else -10.0,
                "conf": None,
                "conf_bucket": None,
                "ou_pick": ou_pick,
            })
    return graded


def game_date_key(value) -> str:
    return str(value or "")[:10]


def drop_ou_bets_on_excluded_slates(graded_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        r for r in graded_rows
        if (r.get("kind") or "ml") != "ou" or game_date_key(r.get("game_date")) not in OU_ACCURACY_EXCLUDED_SLATE_DATES
    ]


def compute_ml_headline_and_calibration(raw_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    buckets = {label: {"bucket": label, "n": 0, "wins": 0, "sum_pred": 0.0} for label, _, _ in _ML_CALIBRATION_BUCKETS}
    n = 0
    brier_model_sum = 0.0
    brier_market_sum = 0.0
    market_n = 0
    correct = 0
    home_wins = 0

    for r in raw_rows:
        if r.get("home_runs") is None or r.get("away_runs") is None or r.get("p_home") is None or r.get("p_away") is None:
            continue
        home_runs = int(r["home_runs"])
        away_runs = int(r["away_runs"])
        if home_runs == away_runs:
            continue
        n += 1
        home_won = home_runs > away_runs
        home_wins += 1 if home_won else 0
        p_home_f = float(r["p_home"])
        p_away_f = float(r["p_away"])
        pick_home = p_home_f >= p_away_f
        pick_won = (pick_home and home_won) or ((not pick_home) and (not home_won))
        correct += 1 if pick_won else 0
        pick_prob = p_home_f if pick_home else p_away_f
        y_home = 1.0 if home_won else 0.0
        brier_model_sum += (p_home_f - y_home) ** 2
        if r.get("p_home_market") is not None:
            brier_market_sum += (float(r["p_home_market"]) - y_home) ** 2
            market_n += 1
        for label, lo, hi in _ML_CALIBRATION_BUCKETS:
            if lo <= pick_prob < hi:
                buckets[label]["n"] += 1
                buckets[label]["wins"] += 1 if pick_won else 0
                buckets[label]["sum_pred"] += pick_prob
                break

    ece = 0.0
    calibration = []
    for label, _, _ in _ML_CALIBRATION_BUCKETS:
        b = buckets[label]
        if b["n"] > 0:
            pred_pct = 100.0 * b["sum_pred"] / b["n"]
            actual_pct = 100.0 * b["wins"] / b["n"]
            ece += (b["n"] / n) * abs(pred_pct / 100.0 - actual_pct / 100.0) if n else 0.0
            calibration.append({"bucket": label, "n": b["n"], "pred_pct": round(pred_pct, 1), "actual_pct": round(actual_pct, 1)})
        else:
            calibration.append({"bucket": label, "n": 0, "pred_pct": None, "actual_pct": None})

    return {
        "games_graded": n,
        "calibration_error_pct": round(100.0 * ece, 1) if n else None,
        "brier_score": round(brier_model_sum / n, 3) if n else None,
        "brier_market": round(brier_market_sum / market_n, 3) if market_n else None,
        "accuracy_pct": round(100.0 * correct / n, 1) if n else None,
        "pick_home_baseline_pct": round(100.0 * home_wins / n, 1) if n else None,
    }, calibration


def _calibration_lines_from_rows(rows: list[dict], thresholds: list[tuple[str, float, str]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    out = []
    for label, thresh, col in thresholds:
        k_floor = int(thresh + 0.5)
        preds = [float(r[col]) for r in rows if r.get(col) is not None]
        actuals = [1 if int(r["actual_count"]) >= k_floor else 0 for r in rows if r.get(col) is not None]
        out.append({
            "line": label,
            "n": len(preds),
            "pred_pct": round(100.0 * sum(preds) / len(preds), 1) if preds else None,
            "actual_pct": round(100.0 * sum(actuals) / len(actuals), 1) if actuals else None,
        })
    return out


def fetch_pitcher_k_calibration_from_pg(dsn: str, min_game_date: str) -> tuple[list[dict[str, Any]], int]:
    q = """
        WITH latest_props AS (
            SELECT DISTINCT ON (ppp.game_id, ppp.pitcher_id)
                ppp.game_id,
                ppp.pitcher_id,
                ppp.p_over_3_5,
                ppp.p_over_5_5,
                ppp.p_over_7_5
            FROM public.pitcher_prop_predictions ppp
            JOIN public.games g ON g.game_id = ppp.game_id AND g.game_date = ppp.game_date
            WHERE ppp.game_date >= %s::date
              AND g.home_runs IS NOT NULL
              AND g.away_runs IS NOT NULL
              AND (
                  LOWER(COALESCE(g.status, '')) LIKE 'final%%'
                  OR LOWER(COALESCE(g.status, '')) = 'game over'
                  OR LOWER(COALESCE(g.status, '')) LIKE 'completed%%'
              )
            ORDER BY ppp.game_id, ppp.pitcher_id, ppp.as_of_ts DESC NULLS LAST
        ),
        actual_ks AS (
            SELECT sp.pitcher AS pitcher_id, sp.game_pk AS game_id,
                   SUM(CASE WHEN sp.events = 'strikeout' THEN 1 ELSE 0 END) AS actual_count
            FROM public.statcast_pitches sp
            WHERE sp.game_date >= %s::date
              AND sp.pitcher IS NOT NULL
              AND sp.game_pk IS NOT NULL
            GROUP BY sp.pitcher, sp.game_pk
        )
        SELECT lp.p_over_3_5, lp.p_over_5_5, lp.p_over_7_5, ak.actual_count
        FROM latest_props lp
        JOIN actual_ks ak ON ak.game_id = lp.game_id AND ak.pitcher_id = lp.pitcher_id
        WHERE ak.actual_count IS NOT NULL
    """
    conn = psycopg2.connect(normalize_pg_dsn(dsn), connect_timeout=25)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(q, (min_game_date, min_game_date))
            rows = [dict(x) for x in cur.fetchall()]
    finally:
        conn.close()
    thresholds = [
        ("Over 3.5", 3.5, "p_over_3_5"),
        ("Over 5.5", 5.5, "p_over_5_5"),
        ("Over 7.5", 7.5, "p_over_7_5"),
    ]
    return _calibration_lines_from_rows(rows, thresholds), len(rows)


def fetch_pitcher_walks_calibration_from_pg(dsn: str, min_game_date: str) -> tuple[list[dict[str, Any]], int]:
    q = """
        WITH latest_props AS (
            SELECT DISTINCT ON (ppp.game_id, ppp.pitcher_id)
                ppp.game_id, ppp.pitcher_id,
                ppp.p_walks_over_1_5, ppp.p_walks_over_2_5, ppp.p_walks_over_3_5
            FROM public.pitcher_prop_predictions ppp
            JOIN public.games g ON g.game_id = ppp.game_id AND g.game_date = ppp.game_date
            WHERE ppp.game_date >= %s::date
              AND g.home_runs IS NOT NULL AND g.away_runs IS NOT NULL
              AND (
                  LOWER(COALESCE(g.status, '')) LIKE 'final%%'
                  OR LOWER(COALESCE(g.status, '')) = 'game over'
                  OR LOWER(COALESCE(g.status, '')) LIKE 'completed%%'
              )
            ORDER BY ppp.game_id, ppp.pitcher_id, ppp.as_of_ts DESC NULLS LAST
        ),
        actual_walks AS (
            SELECT sp.pitcher AS pitcher_id, sp.game_pk AS game_id,
                   SUM(CASE WHEN sp.events IN ('walk','hit_by_pitch') THEN 1 ELSE 0 END) AS actual_count
            FROM public.statcast_pitches sp
            WHERE sp.game_date >= %s::date AND sp.pitcher IS NOT NULL AND sp.game_pk IS NOT NULL
            GROUP BY sp.pitcher, sp.game_pk
        )
        SELECT lp.p_walks_over_1_5, lp.p_walks_over_2_5, lp.p_walks_over_3_5, aw.actual_count
        FROM latest_props lp
        JOIN actual_walks aw ON aw.game_id = lp.game_id AND aw.pitcher_id = lp.pitcher_id
        WHERE aw.actual_count IS NOT NULL
    """
    conn = psycopg2.connect(normalize_pg_dsn(dsn), connect_timeout=25)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(q, (min_game_date, min_game_date))
            rows = [dict(x) for x in cur.fetchall()]
    finally:
        conn.close()
    thresholds = [
        ("Over 1.5", 1.5, "p_walks_over_1_5"),
        ("Over 2.5", 2.5, "p_walks_over_2_5"),
        ("Over 3.5", 3.5, "p_walks_over_3_5"),
    ]
    return _calibration_lines_from_rows(rows, thresholds), len(rows)


def fetch_pitcher_hits_calibration_from_pg(dsn: str, min_game_date: str) -> tuple[list[dict[str, Any]], int]:
    q = """
        WITH latest_props AS (
            SELECT DISTINCT ON (ppp.game_id, ppp.pitcher_id)
                ppp.game_id, ppp.pitcher_id,
                ppp.p_hits_over_4_5, ppp.p_hits_over_5_5, ppp.p_hits_over_6_5
            FROM public.pitcher_prop_predictions ppp
            JOIN public.games g ON g.game_id = ppp.game_id AND g.game_date = ppp.game_date
            WHERE ppp.game_date >= %s::date
              AND g.home_runs IS NOT NULL AND g.away_runs IS NOT NULL
              AND (
                  LOWER(COALESCE(g.status, '')) LIKE 'final%%'
                  OR LOWER(COALESCE(g.status, '')) = 'game over'
                  OR LOWER(COALESCE(g.status, '')) LIKE 'completed%%'
              )
            ORDER BY ppp.game_id, ppp.pitcher_id, ppp.as_of_ts DESC NULLS LAST
        ),
        actual_hits AS (
            SELECT sp.pitcher AS pitcher_id, sp.game_pk AS game_id,
                   SUM(CASE WHEN sp.events IN ('single','double','triple','home_run')
                            THEN 1 ELSE 0 END) AS actual_count
            FROM public.statcast_pitches sp
            WHERE sp.game_date >= %s::date AND sp.pitcher IS NOT NULL AND sp.game_pk IS NOT NULL
            GROUP BY sp.pitcher, sp.game_pk
        )
        SELECT lp.p_hits_over_4_5, lp.p_hits_over_5_5, lp.p_hits_over_6_5, ah.actual_count
        FROM latest_props lp
        JOIN actual_hits ah ON ah.game_id = lp.game_id AND ah.pitcher_id = lp.pitcher_id
        WHERE ah.actual_count IS NOT NULL
    """
    conn = psycopg2.connect(normalize_pg_dsn(dsn), connect_timeout=25)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(q, (min_game_date, min_game_date))
            rows = [dict(x) for x in cur.fetchall()]
    finally:
        conn.close()
    thresholds = [
        ("Over 4.5", 4.5, "p_hits_over_4_5"),
        ("Over 5.5", 5.5, "p_hits_over_5_5"),
        ("Over 6.5", 6.5, "p_hits_over_6_5"),
    ]
    return _calibration_lines_from_rows(rows, thresholds), len(rows)


def fetch_pitcher_er_calibration_from_pg(dsn: str, min_game_date: str) -> tuple[list[dict[str, Any]], int]:
    q = """
        WITH latest_props AS (
            SELECT DISTINCT ON (ppp.game_id, ppp.pitcher_id)
                ppp.game_id, ppp.pitcher_id,
                ppp.p_er_over_2_5, ppp.p_er_over_3_5, ppp.p_er_over_4_5
            FROM public.pitcher_prop_predictions ppp
            JOIN public.games g ON g.game_id = ppp.game_id AND g.game_date = ppp.game_date
            WHERE ppp.game_date >= %s::date
              AND g.home_runs IS NOT NULL AND g.away_runs IS NOT NULL
              AND (
                  LOWER(COALESCE(g.status, '')) LIKE 'final%%'
                  OR LOWER(COALESCE(g.status, '')) = 'game over'
                  OR LOWER(COALESCE(g.status, '')) LIKE 'completed%%'
              )
            ORDER BY ppp.game_id, ppp.pitcher_id, ppp.as_of_ts DESC NULLS LAST
        ),
        actual_er AS (
            SELECT ps.pitcher_id, ps.game_id, ps.earned_runs AS actual_count
            FROM public.pitcher_starts ps
            WHERE ps.game_date >= %s::date AND ps.earned_runs IS NOT NULL
        )
        SELECT lp.p_er_over_2_5, lp.p_er_over_3_5, lp.p_er_over_4_5, ae.actual_count
        FROM latest_props lp
        JOIN actual_er ae ON ae.game_id = lp.game_id AND ae.pitcher_id = lp.pitcher_id
        WHERE ae.actual_count IS NOT NULL
    """
    conn = psycopg2.connect(normalize_pg_dsn(dsn), connect_timeout=25)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(q, (min_game_date, min_game_date))
            rows = [dict(x) for x in cur.fetchall()]
    finally:
        conn.close()
    thresholds = [
        ("Over 2.5", 2.5, "p_er_over_2_5"),
        ("Over 3.5", 3.5, "p_er_over_3_5"),
        ("Over 4.5", 4.5, "p_er_over_4_5"),
    ]
    return _calibration_lines_from_rows(rows, thresholds), len(rows)


def build_payload(pg_dsn: str, snapshot_date: str, min_game_date: str) -> dict[str, Any]:
    raw_rows = fetch_graded_games_from_pg(pg_dsn, min_game_date, model_version="v10")
    v9_history = build_v9_history(pg_dsn)
    rows = drop_ou_bets_on_excluded_slates(grade_games(raw_rows))
    headline, ml_calibration = compute_ml_headline_and_calibration(raw_rows)
    k_lines, k_n = fetch_pitcher_k_calibration_from_pg(pg_dsn, min_game_date)
    walks_lines, walks_n = fetch_pitcher_walks_calibration_from_pg(pg_dsn, min_game_date)
    hits_lines, hits_n = fetch_pitcher_hits_calibration_from_pg(pg_dsn, min_game_date)
    er_lines, er_n = fetch_pitcher_er_calibration_from_pg(pg_dsn, min_game_date)

    def new_overall():
        return {"bets": 0, "wins": 0, "win_pct": None, "net_dollars": 0.0, "roi_pct": None}

    def new_buckets():
        return {b: {"bucket": b, "bets": 0, "wins": 0, "net_dollars": 0.0} for b in ["50-55%", "55-60%", "60-65%", "65%+"]}

    ml_overall = new_overall()
    ou_overall = new_overall()
    ml_buckets = new_buckets()
    ml_daily: dict[str, dict[str, Any]] = {}
    ou_daily: dict[str, dict[str, Any]] = {}
    combined_daily: dict[str, dict[str, Any]] = {}
    ou_picks = {"over": 0, "under": 0}

    for r in rows:
        kind = r.get("kind") or "ml"
        hit = int(r["is_hit"])
        pnl = float(r["pnl_dollars"] or 0.0)
        day = str(r["game_date"])
        target = ml_overall if kind == "ml" else ou_overall
        target["bets"] += 1
        target["wins"] += hit
        target["net_dollars"] += pnl
        if kind == "ml":
            b = r["conf_bucket"]
            if b in ml_buckets:
                ml_buckets[b]["bets"] += 1
                ml_buckets[b]["wins"] += hit
                ml_buckets[b]["net_dollars"] += pnl
            ml_daily.setdefault(day, {"date": day, "bets": 0, "wins": 0, "net_dollars": 0.0})
            ml_daily[day]["bets"] += 1
            ml_daily[day]["wins"] += hit
            ml_daily[day]["net_dollars"] += pnl
        elif kind == "ou":
            if r.get("ou_pick") in ou_picks:
                ou_picks[r["ou_pick"]] += 1
            ou_daily.setdefault(day, {"date": day, "bets": 0, "wins": 0, "net_dollars": 0.0})
            ou_daily[day]["bets"] += 1
            ou_daily[day]["wins"] += hit
            ou_daily[day]["net_dollars"] += pnl
        combined_daily.setdefault(day, {"date": day, "bets": 0, "wins": 0, "net_dollars": 0.0})
        combined_daily[day]["bets"] += 1
        combined_daily[day]["wins"] += hit
        combined_daily[day]["net_dollars"] += pnl

    def finalize_overall(o: dict[str, Any]) -> None:
        if o["bets"] > 0:
            o["win_pct"] = round(100.0 * o["wins"] / o["bets"], 1)
            o["roi_pct"] = round(100.0 * o["net_dollars"] / (o["bets"] * 10.0), 1)
        o["net_dollars"] = round(o["net_dollars"], 2)

    finalize_overall(ml_overall)
    finalize_overall(ou_overall)

    bucket_rows = []
    for key in ["50-55%", "55-60%", "60-65%", "65%+"]:
        br = ml_buckets[key]
        if br["bets"] > 0:
            br["win_pct"] = round(100.0 * br["wins"] / br["bets"], 1)
            br["roi_pct"] = round(100.0 * br["net_dollars"] / (br["bets"] * 10.0), 1)
        else:
            br["win_pct"] = None
            br["roi_pct"] = None
        br["net_dollars"] = round(br["net_dollars"], 2)
        bucket_rows.append(br)

    def finalize_daily(daily_dict: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        out = []
        cum_rows = []
        cum = 0.0
        for day in sorted(daily_dict.keys()):
            dr = daily_dict[day]
            if dr["bets"] > 0:
                dr["win_pct"] = round(100.0 * dr["wins"] / dr["bets"], 1)
                dr["roi_pct"] = round(100.0 * dr["net_dollars"] / (dr["bets"] * 10.0), 1)
            else:
                dr["win_pct"] = None
                dr["roi_pct"] = None
            dr["net_dollars"] = round(dr["net_dollars"], 2)
            cum += dr["net_dollars"]
            dr["cumulative_dollars"] = round(cum, 2)
            out.append(dr)
            cum_rows.append({"date": dr["date"], "cumulative_dollars": dr["cumulative_dollars"]})
        return out, cum_rows

    ml_daily_rows, ml_cum_rows = finalize_daily(ml_daily)
    ou_daily_rows, ou_cum_rows = finalize_daily(ou_daily)
    combined_daily_rows, combined_cum_rows = finalize_daily(combined_daily)
    return {
        "headline": headline,
        "ml_calibration": ml_calibration,
        "pitcher_k_calibration": {"starters_graded": k_n, "lines": k_lines},
        "pitcher_walks_calibration": {"starters_graded": walks_n, "lines": walks_lines},
        "pitcher_hits_calibration": {"starters_graded": hits_n, "lines": hits_lines},
        "pitcher_er_calibration": {"starters_graded": er_n, "lines": er_lines},
        "overall": ml_overall,
        "buckets": bucket_rows,
        "daily": ml_daily_rows,
        "daily_cumulative": ml_cum_rows,
        "ou_overall": ou_overall,
        "ou_daily": ou_daily_rows,
        "ou_daily_cumulative": ou_cum_rows,
        "ou_pick_counts": ou_picks,
        "combined_daily": combined_daily_rows,
        "combined_daily_cumulative": combined_cum_rows,
        "v9_history": v9_history,
        "meta": {
            "source": "model_performance_snapshot",
            "version": "v10",
            "snapshot_date": snapshot_date,
            "min_game_date": min_game_date,
            "v10_launch_date": V10_LAUNCH_DATE,
            "v9_launch_date": V9_LAUNCH_DATE,
            "v9_end_date": V9_END_DATE,
            "small_sample_note": V10_SMALL_SAMPLE_NOTE,
            "ou_pricing_assumption": "standard -110 (10/11 on hit)",
            "ou_slate_dates_excluded_from_ou_stats": sorted(OU_ACCURACY_EXCLUDED_SLATE_DATES),
            "graded_games": len(raw_rows),
            "graded_bets": len(rows),
            "ou_excludes_pass": True,
            "ou_pass_gap_lt": OU_PRED_LINE_GAP,
        },
    }


def upsert_snapshot(engine, schema: str, snapshot_date: str, min_game_date: str, payload: dict[str, Any]) -> None:
    headline = payload.get("headline") or {}
    pitcher_k = payload.get("pitcher_k_calibration") or {}
    with engine.begin() as conn:
        conn.execute(text(CREATE_TABLE_SQL))
        conn.execute(
            text(f"""
                INSERT INTO {schema}.model_performance_snapshot (
                    snapshot_date, version, min_game_date, games_graded,
                    calibration_error_pct, brier_score, brier_market, accuracy_pct,
                    pitcher_k_starters_graded, payload, created_at
                )
                VALUES (
                    :snapshot_date, 'v10', :min_game_date, :games_graded,
                    :calibration_error_pct, :brier_score, :brier_market, :accuracy_pct,
                    :pitcher_k_starters_graded, CAST(:payload AS jsonb), NOW()
                )
                ON CONFLICT (snapshot_date) DO UPDATE SET
                    version = EXCLUDED.version,
                    min_game_date = EXCLUDED.min_game_date,
                    games_graded = EXCLUDED.games_graded,
                    calibration_error_pct = EXCLUDED.calibration_error_pct,
                    brier_score = EXCLUDED.brier_score,
                    brier_market = EXCLUDED.brier_market,
                    accuracy_pct = EXCLUDED.accuracy_pct,
                    pitcher_k_starters_graded = EXCLUDED.pitcher_k_starters_graded,
                    payload = EXCLUDED.payload,
                    created_at = NOW()
            """),
            {
                "snapshot_date": snapshot_date,
                "min_game_date": min_game_date,
                "games_graded": headline.get("games_graded"),
                "calibration_error_pct": headline.get("calibration_error_pct"),
                "brier_score": headline.get("brier_score"),
                "brier_market": headline.get("brier_market"),
                "accuracy_pct": headline.get("accuracy_pct"),
                "pitcher_k_starters_graded": pitcher_k.get("starters_graded"),
                "payload": json.dumps(payload, separators=(",", ":")),
            },
        )
    print(f"  Wrote model performance snapshot for {snapshot_date}")


def backfill_model_version_tags(engine, schema: str) -> None:
    """Correct v9/v10 tags on historical inference rows (one-time safe to re-run)."""
    with engine.begin() as conn:
        conn.execute(text(f"""
            UPDATE {schema}.inference_game_predictions
            SET model_version = 'v9'
            WHERE game_date >= DATE '{V9_LAUNCH_DATE}'
              AND game_date < DATE '{V10_LAUNCH_DATE}'
        """))
        conn.execute(text(f"""
            UPDATE {schema}.inference_game_predictions
            SET model_version = 'v10'
            WHERE game_date >= DATE '{V10_LAUNCH_DATE}'
              AND (model_version IS NULL OR model_version = '')
        """))
        conn.execute(text(f"""
            UPDATE {schema}.inference_game_predictions
            SET model_version = 'v9'
            WHERE game_date < DATE '{V9_LAUNCH_DATE}'
              AND (model_version IS NULL OR model_version = '')
        """))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--schema", default="public")
    ap.add_argument("--min-game-date", default=V10_LAUNCH_DATE)
    args = ap.parse_args()

    pg_dsn = os.getenv("PG_DSN")
    if not pg_dsn:
        raise RuntimeError("PG_DSN env var required.")
    engine = create_engine(pg_dsn, pool_pre_ping=True)
    backfill_model_version_tags(engine, args.schema)
    payload = build_payload(pg_dsn, args.date, args.min_game_date)
    upsert_snapshot(engine, args.schema, args.date, args.min_game_date, payload)


if __name__ == "__main__":
    main()
