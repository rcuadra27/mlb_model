#!/usr/bin/env python3
"""
Pre-compute daily top edges after props inference.

Usage:
    PG_DSN=... python features/build_edges.py --date 2026-05-29
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.edge_logic import compute_all_edges, _normalize_pitcher_name
from features.pitcher_prop_market_lines import fetch_pitcher_counting_stat_market_lines

LEAGUE_DEFAULT_SP_K_RATE = 0.230
DEFAULT_RATE_EPSILON = 0.001

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS public.daily_edges (
    id BIGSERIAL PRIMARY KEY,
    edge_date DATE NOT NULL,
    rank INTEGER NOT NULL,
    edge_type TEXT NOT NULL,
    prop_subtype TEXT,
    pick_description TEXT NOT NULL,
    detail_line TEXT NOT NULL,
    model_value DOUBLE PRECISION,
    comparison_value DOUBLE PRECISION,
    edge_magnitude DOUBLE PRECISION NOT NULL,
    direction TEXT NOT NULL,
    game_id BIGINT,
    player_id INTEGER,
    team_id INTEGER,
    team_abbr TEXT,
    team_name TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_daily_edges_date ON public.daily_edges(edge_date);
"""

ALTER_TABLE_SQL = """
ALTER TABLE public.daily_edges ADD COLUMN IF NOT EXISTS team_id INTEGER;
ALTER TABLE public.daily_edges ADD COLUMN IF NOT EXISTS team_abbr TEXT;
ALTER TABLE public.daily_edges ADD COLUMN IF NOT EXISTS team_name TEXT;
ALTER TABLE public.daily_edges ADD COLUMN IF NOT EXISTS rate_detail_line TEXT;
ALTER TABLE public.daily_edges ADD COLUMN IF NOT EXISTS market_line DOUBLE PRECISION;
ALTER TABLE public.daily_edges ADD COLUMN IF NOT EXISTS model_prob_pct DOUBLE PRECISION;
"""

GAMES_SQL = """
    SELECT
        p.game_id,
        p.away_team,
        p.home_team,
        p.p_away_win_poisson::float AS p_win_away,
        p.p_home_win_poisson::float AS p_win_home,
        p.p_away_market_median::float AS p_away_market,
        p.p_home_market_median::float AS p_home_market,
        p.edge_away::float AS edge_away,
        p.edge_home::float AS edge_home,
        COALESCE(ta.total_runs_pred, p.total_runs_pred)::float AS total_runs_pred,
        COALESCE(ta.ou_line, p.ou_line, fg.morning_ou_line, fg.closing_ou_line)::float AS ou_line,
        COALESCE(ta.ou_edge_over, p.ou_edge_over)::float AS ou_edge_over,
        COALESCE(ta.ou_edge_under, p.ou_edge_under)::float AS ou_edge_under,
        fg.morning_away_price,
        fg.morning_home_price,
        fg.closing_away_price,
        fg.closing_home_price
    FROM (
        SELECT DISTINCT ON (game_id) *
        FROM public.inference_game_predictions
        WHERE game_date = :d
        ORDER BY game_id, as_of_ts DESC
    ) p
    LEFT JOIN (
        SELECT DISTINCT ON (game_id)
            game_id,
            total_runs_pred,
            ou_line,
            ou_edge_over,
            ou_edge_under
        FROM public.inference_game_predictions
        WHERE game_date = :d
          AND total_runs_pred IS NOT NULL
        ORDER BY game_id, as_of_ts DESC
    ) ta USING (game_id)
    LEFT JOIN public.features_game fg USING (game_id)
    ORDER BY p.game_id
"""

BATTERS_SQL = """
    SELECT
        bpp.game_id,
        bpp.batter_id,
        bpp.batter_name,
        bpp.team_id,
        t.abbreviation AS team_abbr,
        t.team_name,
        bpp.p_hit::float,
        bpp.p_2plus_hits::float,
        bpp.p_hr::float,
        bpp.p_k::float,
        bpp.p_2plus_bases::float,
        bpp.p_walk::float,
        bpp.lineup_confirmed,
        bpp.batting_order
    FROM (
        SELECT *,
            ROW_NUMBER() OVER (
                PARTITION BY game_id, batter_id
                ORDER BY COALESCE(lineup_confirmed, FALSE) DESC, as_of_ts DESC
            ) AS rn
        FROM public.player_prop_predictions
        WHERE game_date = :d
    ) bpp
    LEFT JOIN public.teams t ON t.mlb_team_id = bpp.team_id
    WHERE bpp.rn = 1
      AND COALESCE(bpp.lineup_confirmed, FALSE) = TRUE
      AND bpp.batting_order IS NOT NULL
"""

PITCHERS_SQL = """
    SELECT
        ppp.game_id,
        ppp.pitcher_id,
        ppp.pitcher_name,
        CASE WHEN ppp.is_home THEN g.home_team_id ELSE g.away_team_id END AS team_id,
        t.abbreviation AS team_abbr,
        t.team_name,
        ppp.lambda_k::float,
        ppp.lambda_walks::float,
        ppp.lambda_hits::float,
        ppp.lambda_er::float,
        ppp.expected_ip::float,
        ppp.sp_k_rate_season::float,
        ppp.sp_innings_season::float,
        ppp.opp_lineup_k_rate::float
    FROM (
        SELECT *,
            ROW_NUMBER() OVER (PARTITION BY game_id, pitcher_id ORDER BY as_of_ts DESC) AS rn
        FROM public.pitcher_prop_predictions
        WHERE game_date = :d
    ) ppp
    JOIN public.games g ON g.game_id = ppp.game_id AND g.game_date = ppp.game_date
    LEFT JOIN public.teams t
      ON t.mlb_team_id = CASE WHEN ppp.is_home THEN g.home_team_id ELSE g.away_team_id END
    WHERE ppp.rn = 1
"""

SP_RATE_SQL = """
    WITH k_by_game AS (
        SELECT sp.pitcher AS pitcher_id, sp.game_pk AS game_id,
            SUM(CASE WHEN sp.events = 'strikeout' THEN 1 ELSE 0 END) AS ks
        FROM public.statcast_pitches sp
        GROUP BY sp.pitcher, sp.game_pk
    )
    SELECT
        ps.pitcher_id,
        ROUND((SUM(COALESCE(k.ks, 0)) * 9.0 / NULLIF(SUM(ps.innings_pitched), 0))::numeric, 2) AS k9,
        ROUND((SUM(ps.walks_allowed) * 9.0 / NULLIF(SUM(ps.innings_pitched), 0))::numeric, 2) AS bb9,
        ROUND((SUM(ps.hits_allowed) * 9.0 / NULLIF(SUM(ps.innings_pitched), 0))::numeric, 2) AS h9,
        ROUND((SUM(ps.earned_runs) * 9.0 / NULLIF(SUM(ps.innings_pitched), 0))::numeric, 2) AS era
    FROM public.pitcher_starts ps
    JOIN public.games g ON g.game_id = ps.game_id
    LEFT JOIN k_by_game k ON k.pitcher_id = ps.pitcher_id AND k.game_id = ps.game_id
    WHERE g.home_runs IS NOT NULL AND g.away_runs IS NOT NULL
      AND ps.pitcher_id = ANY(%s)
      AND EXTRACT(YEAR FROM ps.game_date)::int = EXTRACT(YEAR FROM %s::date)::int
      AND ps.game_date < %s::date
      AND ps.innings_pitched > 0
    GROUP BY ps.pitcher_id
    HAVING SUM(ps.innings_pitched) >= 9
"""

PERSONAL_RATES_SQL = """
    WITH batter_game AS (
        SELECT
            sp.batter AS batter_id,
            sp.game_pk,
            MAX(CASE WHEN sp.events IN ('single','double','triple','home_run') THEN 1 ELSE 0 END) AS had_hit,
            SUM(CASE WHEN sp.events IN ('single','double','triple','home_run') THEN 1 ELSE 0 END) AS hits,
            MAX(CASE WHEN sp.events = 'home_run' THEN 1 ELSE 0 END) AS had_hr,
            MAX(CASE WHEN sp.events = 'strikeout' THEN 1 ELSE 0 END) AS had_k,
            MAX(CASE WHEN sp.events = 'walk' THEN 1 ELSE 0 END) AS had_walk,
            SUM(CASE
                WHEN sp.events = 'single' THEN 1
                WHEN sp.events = 'double' THEN 2
                WHEN sp.events = 'triple' THEN 3
                WHEN sp.events = 'home_run' THEN 4
                ELSE 0 END) AS total_bases,
            COUNT(DISTINCT sp.at_bat_number) AS pa
        FROM public.statcast_pitches sp
        WHERE sp.game_date < %s::date
          AND EXTRACT(YEAR FROM sp.game_date)::int = %s
          AND sp.batter = ANY(%s)
          AND sp.batter IS NOT NULL
        GROUP BY sp.batter, sp.game_pk
    ),
    agg AS (
        SELECT
            batter_id,
            SUM(pa) AS sd_pa,
            COUNT(*) AS games,
            SUM(had_hit)::float / NULLIF(COUNT(*), 0) AS personal_p_hit,
            SUM(CASE WHEN hits >= 2 THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0) AS personal_p_2plus_hits,
            SUM(had_hr)::float / NULLIF(COUNT(*), 0) AS personal_p_hr,
            SUM(had_k)::float / NULLIF(COUNT(*), 0) AS personal_p_k,
            SUM(CASE WHEN total_bases >= 2 THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0) AS personal_p_2plus_bases,
            SUM(had_walk)::float / NULLIF(COUNT(*), 0) AS personal_p_walk
        FROM batter_game
        GROUP BY batter_id
    )
    SELECT * FROM agg
"""


def _normalize_pg_dsn(dsn: str) -> str:
    if dsn.startswith("postgresql+psycopg2://"):
        return "postgresql://" + dsn[len("postgresql+psycopg2://") :]
    if dsn.startswith("postgres+psycopg2://"):
        return "postgres://" + dsn[len("postgres+psycopg2://") :]
    return dsn


def fetch_pitcher_sp_avgs(dsn: str, pitcher_ids: list, date_str: str) -> dict[str, dict[int, float]]:
    ids = [int(x) for x in pitcher_ids if x is not None]
    if not ids:
        return {"k": {}, "walks": {}, "hits": {}, "er": {}}
    dsn = _normalize_pg_dsn(dsn)
    conn = psycopg2.connect(dsn, connect_timeout=25)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(SP_RATE_SQL, (ids, date_str, date_str))
            rows = cur.fetchall()
    finally:
        conn.close()
    k, walks, hits, er = {}, {}, {}, {}
    for r in rows:
        pid = int(r["pitcher_id"])
        if r.get("k9") is not None:
            k[pid] = float(r["k9"])
        if r.get("bb9") is not None:
            walks[pid] = float(r["bb9"])
        if r.get("h9") is not None:
            hits[pid] = float(r["h9"])
        if r.get("era") is not None:
            er[pid] = float(r["era"])
    return {"k": k, "walks": walks, "hits": hits, "er": er}


def fetch_personal_rates(dsn: str, batter_ids: list, date_str: str) -> dict:
    ids = [int(x) for x in batter_ids if x is not None]
    if not ids:
        return {}
    dsn = _normalize_pg_dsn(dsn)
    season = int(date_str[:4])
    conn = psycopg2.connect(dsn, connect_timeout=25)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(PERSONAL_RATES_SQL, (date_str, season, ids))
            rows = cur.fetchall()
    finally:
        conn.close()
    out = {}
    for r in rows:
        bid = int(r["batter_id"])
        out[bid] = {
            "sd_pa": int(r["sd_pa"] or 0),
            "personal_p_hit": float(r["personal_p_hit"]) if r["personal_p_hit"] is not None else None,
            "personal_p_2plus_hits": float(r["personal_p_2plus_hits"]) if r["personal_p_2plus_hits"] is not None else None,
            "personal_p_hr": float(r["personal_p_hr"]) if r["personal_p_hr"] is not None else None,
            "personal_p_k": float(r["personal_p_k"]) if r["personal_p_k"] is not None else None,
            "personal_p_2plus_bases": float(r["personal_p_2plus_bases"]) if r["personal_p_2plus_bases"] is not None else None,
            "personal_p_walk": float(r["personal_p_walk"]) if r["personal_p_walk"] is not None else None,
        }
    return out


def upsert_edges(date_str: str, schema: str, engine, edges: list[dict[str, Any]]) -> None:
    with engine.begin() as conn:
        conn.execute(text(CREATE_TABLE_SQL))
        conn.execute(text(ALTER_TABLE_SQL))
        conn.execute(text(f"DELETE FROM {schema}.daily_edges WHERE edge_date = :d"), {"d": date_str})
        if not edges:
            print(f"  No edges for {date_str}")
            return
        rows = []
        for e in edges:
            mag = e.get("edge_magnitude")
            if mag is None:
                print(f"  Skipping edge rank {e.get('rank')}: null edge_magnitude ({e.get('pick_description')})")
                continue
            try:
                mag_f = float(mag)
            except (TypeError, ValueError):
                print(f"  Skipping edge rank {e.get('rank')}: invalid edge_magnitude ({e.get('pick_description')})")
                continue
            if mag_f != mag_f:  # NaN
                print(f"  Skipping edge rank {e.get('rank')}: NaN edge_magnitude ({e.get('pick_description')})")
                continue
            rows.append({
                "edge_date": date_str,
                "rank": e["rank"],
                "edge_type": e["edge_type"],
                "prop_subtype": e.get("prop_subtype"),
                "pick_description": e["pick_description"],
                "detail_line": e["detail_line"],
                "rate_detail_line": e.get("rate_detail_line"),
                "market_line": e.get("market_line"),
                "model_prob_pct": e.get("model_prob_pct"),
                "model_value": e.get("model_value"),
                "comparison_value": e.get("comparison_value"),
                "edge_magnitude": mag_f,
                "direction": e["direction"],
                "game_id": e.get("game_id"),
                "player_id": e.get("player_id"),
                "team_id": e.get("team_id"),
                "team_abbr": e.get("team_abbr"),
                "team_name": e.get("team_name"),
            })
        pd.DataFrame(rows).to_sql(
            "daily_edges",
            conn,
            schema=schema,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=100,
        )
    print(f"  Wrote {len(edges)} edge rows for {date_str}")


def build_edges(date_str: str, schema: str, engine, pg_dsn: str) -> list:
    from features.active_roster import (
        filter_eligible_pitchers_df,
        load_eligible_player_ids,
    )

    params = {"d": date_str}
    games = pd.read_sql(text(GAMES_SQL), engine, params=params).to_dict(orient="records")
    batters_raw = pd.read_sql(text(BATTERS_SQL), engine, params=params).to_dict(orient="records")
    print(f"  {len(batters_raw)} confirmed-lineup batter prop row(s) eligible for edges")
    eligible = load_eligible_player_ids(engine, date_str, schema)
    if eligible:
        before = len(batters_raw)
        batters = [b for b in batters_raw if b.get("batter_id") is not None and int(b["batter_id"]) in eligible]
        skipped = before - len(batters)
        if skipped:
            print(f"  Active roster filter removed {skipped} batter edge candidate(s)")
    else:
        batters = batters_raw
        print("  Warning: no active roster snapshot — batter edges unfiltered")
    pitchers_raw = pd.read_sql(text(PITCHERS_SQL), engine, params=params).to_dict(orient="records")
    if eligible:
        before_p = len(pitchers_raw)
        pitchers_raw = filter_eligible_pitchers_df(pitchers_raw, eligible)
        skipped_p = before_p - len(pitchers_raw)
        if skipped_p:
            print(f"  Active roster filter removed {skipped_p} pitcher edge candidate(s)")
    pitchers = []
    skipped_defaulted = []
    for row in pitchers_raw:
        sp_k_rate = row.get("sp_k_rate_season")
        sp_ip = row.get("sp_innings_season")
        is_default_rate = sp_k_rate is None or abs(float(sp_k_rate) - LEAGUE_DEFAULT_SP_K_RATE) <= DEFAULT_RATE_EPSILON
        is_no_sp_history = sp_ip is None or float(sp_ip) <= 0
        if is_default_rate or is_no_sp_history:
            skipped_defaulted.append(row.get("pitcher_name") or row.get("pitcher_id"))
            continue
        pitchers.append(row)
    if skipped_defaulted:
        print(
            "  Skipped defaulted pitcher K edge inputs: "
            + ", ".join(str(x) for x in skipped_defaulted[:8])
            + (f" (+{len(skipped_defaulted) - 8} more)" if len(skipped_defaulted) > 8 else "")
        )

    pitcher_ids = [p.get("pitcher_id") for p in pitchers]
    batter_ids = [b.get("batter_id") for b in batters]
    sp_avgs = fetch_pitcher_sp_avgs(pg_dsn, pitcher_ids, date_str)
    personal_rates = fetch_personal_rates(pg_dsn, batter_ids, date_str)

    games_by_id = {
        int(g["game_id"]): {"away_team": g.get("away_team"), "home_team": g.get("home_team")}
        for g in games
        if g.get("game_id") is not None
    }
    market_lines = fetch_pitcher_counting_stat_market_lines(date_str, games_by_id)
    for p in pitchers:
        key = (int(p["game_id"]), _normalize_pitcher_name(p.get("pitcher_name")))
        p["_market_lines"] = market_lines.get(key, {})

    return compute_all_edges(games, batters, pitchers, personal_rates, sp_avgs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    ap.add_argument("--schema", default="public")
    args = ap.parse_args()

    if not args.date or not str(args.date).strip():
        raise SystemExit("--date is required (e.g. 2026-05-29)")

    pg_dsn = os.getenv("PG_DSN")
    if not pg_dsn:
        raise RuntimeError("PG_DSN env var required.")

    engine = create_engine(pg_dsn, pool_pre_ping=True)
    print(f"Building daily edges for {args.date}...")
    edges = build_edges(args.date, args.schema, engine, pg_dsn)
    upsert_edges(args.date, args.schema, engine, edges)
    by_type: dict[str, int] = {}
    for e in edges:
        t = e.get("edge_type")
        by_type[t] = by_type.get(t, 0) + 1
    for t, n in sorted(by_type.items()):
        print(f"    {t}: {n}")


if __name__ == "__main__":
    main()
