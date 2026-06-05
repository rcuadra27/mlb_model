import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

import functions_framework
from google.cloud import bigquery
import psycopg2
from psycopg2.extras import RealDictCursor


# Min |predicted total − market O/U line| to grade an O/U pick (below = no bet / model push).
OU_PRED_LINE_GAP = 0.5
V9_LAUNCH_DATE = "2026-04-14"
V10_LAUNCH_DATE = "2026-05-28"
V9_END_DATE = "2026-05-27"
V10_SMALL_SAMPLE_NOTE = (
    "v10 launched May 28 — these metrics are based on a limited early sample "
    "and will stabilize as more games are graded."
)
_TZ_PT = ZoneInfo("America/Los_Angeles")
_CACHE_TTL_SECONDS = 300
_TTL_CACHE = {}


def _today_pacific() -> str:
    return datetime.now(_TZ_PT).date().isoformat()


def _cache_get(key):
    item = _TTL_CACHE.get(key)
    if not item:
        return None
    expires_at, payload = item
    if expires_at <= time.time():
        _TTL_CACHE.pop(key, None)
        return None
    return payload


def _cache_set(key, payload):
    _TTL_CACHE[key] = (time.time() + _CACHE_TTL_SECONDS, payload)
    return payload


def _cached(key, builder):
    cached = _cache_get(key)
    if cached is not None:
        return cached
    return _cache_set(key, builder())


def _format_pitch_time_pt(iso_ts) -> str | None:
    if not iso_ts:
        return None
    try:
        s = str(iso_ts).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
        local = dt.astimezone(_TZ_PT)
        hour = local.strftime("%I").lstrip("0") or "12"
        return f"{hour}:{local.strftime('%M %p')} PT"
    except (TypeError, ValueError):
        return None


def _fmt_american(odds) -> str | None:
    if odds is None:
        return None
    try:
        o = int(round(float(odds)))
    except (TypeError, ValueError):
        return None
    return f"+{o}" if o > 0 else str(o)


def _safe_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _normalize_pg_dsn(dsn: str) -> str:
    """Strip SQLAlchemy +psycopg2 for raw psycopg2."""
    if dsn.startswith("postgresql+psycopg2://"):
        return "postgresql://" + dsn[len("postgresql+psycopg2://") :]
    if dsn.startswith("postgres+psycopg2://"):
        return "postgres://" + dsn[len("postgres+psycopg2://") :]
    return dsn


def _is_confirmed_lineup_batter(row: dict) -> bool:
    confirmed = row.get("lineup_confirmed")
    if confirmed in (True, "true", "t", 1, "1"):
        return row.get("batting_order") is not None
    return False


def _filter_batters_for_confirmed_lineups(batters: list) -> list:
    """Once a game has a confirmed lineup, drop roster-proxy / non-starter rows."""
    if not batters:
        return batters
    confirmed_games = {
        b["game_id"] for b in batters if _is_confirmed_lineup_batter(b)
    }
    if not confirmed_games:
        return batters
    return [
        b for b in batters
        if b.get("game_id") not in confirmed_games or _is_confirmed_lineup_batter(b)
    ]


def _fetch_graded_games_from_pg(dsn: str, min_game_date: str) -> list:
    """
    Same row shape as _fetch_graded_games_from_bq, from Cloud SQL
    (updates as soon as scores land in public.games).
    """
    dsn = _normalize_pg_dsn(dsn.strip())
    q = """
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
              AND (
                  p.model_version = 'v10'
                  OR (p.model_version IS NULL AND p.game_date >= DATE '{V10_LAUNCH_DATE}')
              )
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
    conn = psycopg2.connect(dsn, connect_timeout=25)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(q, (min_game_date,))
            return [dict(x) for x in cur.fetchall()]
    finally:
        conn.close()


def _fetch_graded_games_v9_from_pg(dsn: str) -> list:
    """Graded games from the v9 window (Apr 14 – May 27)."""
    dsn = _normalize_pg_dsn(dsn.strip())
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
            WHERE p.game_date >= DATE '{V9_LAUNCH_DATE}'
              AND p.game_date <= DATE '{V9_END_DATE}'
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
    conn = psycopg2.connect(dsn, connect_timeout=25)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(q)
            return [dict(x) for x in cur.fetchall()]
    finally:
        conn.close()


def _build_v9_history(raw_rows: list | None = None, pg_dsn: str | None = None, client=None) -> dict:
    rows = raw_rows
    if rows is None and pg_dsn:
        try:
            rows = _fetch_graded_games_v9_from_pg(pg_dsn)
        except Exception:
            rows = None
    if rows is None:
        try:
            if client is None:
                client = bigquery.Client()
            rows = _fetch_graded_games_v9_from_bq(client)
        except Exception:
            rows = []
    rows = rows or []
    headline, ml_calibration = _compute_ml_headline_and_calibration(rows)
    return {
        "version": "v9",
        "label": "Previous model (v9), Apr 14 – May 27",
        "min_game_date": V9_LAUNCH_DATE,
        "max_game_date": V9_END_DATE,
        "headline": headline,
        "ml_calibration": ml_calibration,
    }
def _row_to_dict(row):
    out = dict(row)
    for k, v in out.items():
        if hasattr(v, "isoformat"):
            out[k] = v.isoformat()
        elif v is None:
            out[k] = None
    return out


def _is_sane_ml_price(price) -> bool:
    if price is None:
        return False
    try:
        p = int(price)
    except (TypeError, ValueError):
        return False
    return p != 0 and abs(p) <= 500


def _is_sane_market_prob(prob) -> bool:
    if prob is None:
        return False
    try:
        p = float(prob)
    except (TypeError, ValueError):
        return False
    return 0.08 <= p <= 0.92


def _fetch_pregame_odds_from_pg(dsn: str, date_str: str | None) -> dict[int, dict]:
    """
    First inference snapshot per game with sane pre-game ML prices (frozen at morning run).
    features_game.morning_* can be overwritten mid-game; consensus on the earliest snapshot cannot.
    """
    dsn = _normalize_pg_dsn(dsn.strip())
    date_filter = "AND game_date = %s::date" if date_str else ""
    params: tuple = (date_str,) if date_str else ()
    q = f"""
        SELECT
            game_id,
            home_price_consensus,
            away_price_consensus,
            p_home_market_median,
            p_away_market_median,
            model_version,
            as_of_ts
        FROM public.inference_game_predictions
        WHERE TRUE {date_filter}
        ORDER BY game_id, as_of_ts ASC
    """
    conn = psycopg2.connect(dsn, connect_timeout=25)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(q, params)
            out: dict[int, dict] = {}
            for row in cur.fetchall():
                gid = int(row["game_id"])
                entry = out.setdefault(gid, {"model_version": row.get("model_version")})

                hp = row.get("home_price_consensus")
                ap = row.get("away_price_consensus")
                if (
                    "pregame_home_price" not in entry
                    and _is_sane_ml_price(hp)
                    and _is_sane_ml_price(ap)
                ):
                    entry["pregame_home_price"] = int(hp)
                    entry["pregame_away_price"] = int(ap)

                mph = row.get("p_home_market_median")
                mpa = row.get("p_away_market_median")
                if (
                    "market_p_home" not in entry
                    and _is_sane_market_prob(mph)
                    and _is_sane_market_prob(mpa)
                ):
                    entry["market_p_home"] = round(float(mph) * 100, 1)
                    entry["market_p_away"] = round(float(mpa) * 100, 1)

                if row.get("model_version"):
                    entry["model_version"] = row.get("model_version")

            return {
                gid: entry
                for gid, entry in out.items()
                if len(entry) > 1
            }
    finally:
        conn.close()


def _apply_pregame_odds_overlay(games: list, frozen: dict[int, dict]) -> list:
    if not frozen:
        return games
    for g in games:
        fro = frozen.get(int(g.get("game_id") or 0))
        if not fro:
            continue
        g.update(fro)
    return games


def _run_daily_games_query(client, date):
    date_filter = f"AND game_date = '{date}'" if date else ""
    query = f"""
        WITH latest AS (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY game_id
                    ORDER BY COALESCE(lineup_pending, FALSE) ASC, as_of_ts DESC
                ) AS rn
            FROM `mlb-model-491223.mlb_model_logs.daily_games`
            WHERE TRUE {date_filter}
        )
        SELECT
            CAST(game_date AS STRING)                           AS game_date,
            game_id,
            away_team,
            home_team,
            away_sp_name,
            home_sp_name,
            ROUND(CAST(away_runs_pred AS FLOAT64), 2)          AS away_runs_pred,
            ROUND(CAST(home_runs_pred AS FLOAT64), 2)          AS home_runs_pred,
            ROUND(CAST(total_runs_pred AS FLOAT64), 2)         AS total_runs_pred,
            ROUND(CAST(p_win_away AS FLOAT64) * 100, 1)        AS p_win_away,
            ROUND(CAST(p_win_home AS FLOAT64) * 100, 1)        AS p_win_home,
            COALESCE(CAST(ou_line AS FLOAT64), CAST(morning_ou_line AS FLOAT64)) AS ou_line,
            ou_recommendation,
            ROUND(CAST(ou_edge_over AS FLOAT64) * 100, 1)      AS ou_edge_over,
            ROUND(CAST(ou_edge_under AS FLOAT64) * 100, 1)     AS ou_edge_under,
            ROUND(CAST(edge_away AS FLOAT64) * 100, 1)         AS edge_away,
            ROUND(CAST(edge_home AS FLOAT64) * 100, 1)         AS edge_home,
            is_value_ml_away,
            is_value_ml_home,
            is_value_ou_over,
            is_value_ou_under,
            ROUND(CAST(p_home_market_median AS FLOAT64) * 100, 1)  AS market_p_home,
            ROUND(CAST(p_away_market_median AS FLOAT64) * 100, 1)  AS market_p_away,
            CAST(morning_p_home AS FLOAT64)                    AS morning_p_home,
            CAST(closing_p_home AS FLOAT64)                    AS closing_p_home,
            CAST(morning_ou_line AS FLOAT64)                   AS morning_ou_line,
            CAST(closing_ou_line AS FLOAT64)                   AS closing_ou_line,
            CAST(total_line_move AS FLOAT64)                   AS total_line_move,
            CAST(home_line_move AS FLOAT64)                    AS home_line_move,
            CAST(sharp_action_home AS INT64)                   AS sharp_action_home,
            CAST(morning_home_price AS INT64)                  AS morning_home_price,
            CAST(morning_away_price AS INT64)                  AS morning_away_price,
            CAST(closing_home_price AS INT64)                  AS closing_home_price,
            CAST(closing_away_price AS INT64)                  AS closing_away_price,
            CAST(n_books_ml AS INT64)                          AS n_books_ml,
            CAST(n_books_ou AS INT64)                          AS n_books_ou,
            first_pitch_utc,
            CAST(away_runs AS INT64)                           AS away_runs,
            CAST(home_runs AS INT64)                           AS home_runs,
            status,
            COALESCE(lineup_pending, FALSE)                    AS lineup_pending
        FROM latest
        WHERE rn = 1
        ORDER BY first_pitch_utc ASC NULLS LAST
    """
    return [_row_to_dict(r) for r in client.query(query).result()]


def _fetch_players_from_pg(dsn: str, date_str: str | None) -> dict:
    """Player/pitcher props from Cloud SQL (same shape as BQ players view)."""
    dsn = _normalize_pg_dsn(dsn.strip())
    date_clause = "WHERE game_date = %s::date" if date_str else "WHERE TRUE"
    params: tuple = (date_str,) if date_str else ()

    batters_q = f"""
        SELECT
            bpp.game_date::text AS game_date,
            bpp.game_id,
            bpp.batter_id,
            bpp.batter_name,
            bpp.team_id,
            bpp.batting_order,
            bpp.sp_id,
            bpp.sp_name,
            bpp.p_hit::float,
            bpp.p_2plus_hits::float,
            bpp.p_hr::float,
            bpp.p_k::float,
            bpp.p_2plus_bases::float,
            bpp.p_walk::float,
            bpp.matchup_score::float,
            bpp.platoon_advantage,
            bpp.batter_xwoba_season::float,
            bpp.batter_hit_rate_30d::float,
            bpp.lineup_confirmed,
            gl.bats AS batter_hand,
            CASE WHEN bpp.team_id = g.home_team_id THEN TRUE ELSE FALSE END AS is_home,
            g.away_team_name AS away_team,
            g.home_team_name AS home_team
        FROM (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY game_id, batter_id
                    ORDER BY COALESCE(lineup_confirmed, FALSE) DESC, as_of_ts DESC
                ) AS rn
            FROM public.player_prop_predictions
            {date_clause}
        ) bpp
        JOIN public.games g
          ON g.game_id = bpp.game_id AND g.game_date = bpp.game_date
        LEFT JOIN public.game_lineups gl
          ON gl.game_id = bpp.game_id AND gl.player_id = bpp.batter_id
        WHERE bpp.rn = 1
        ORDER BY bpp.game_id, is_home DESC, bpp.batting_order
    """

    pitchers_q = f"""
        SELECT
            ppp.game_date::text AS game_date,
            ppp.game_id,
            ppp.pitcher_id,
            ppp.pitcher_name,
            ppp.is_home,
            ppp.lambda_k::float,
            ppp.lambda_walks::float,
            ppp.lambda_hits::float,
            ppp.lambda_er::float,
            ppp.p_k0::float, ppp.p_k1::float, ppp.p_k2::float, ppp.p_k3::float,
            ppp.p_k4::float, ppp.p_k5::float, ppp.p_k6::float, ppp.p_k7::float,
            ppp.p_k8::float, ppp.p_k9::float, ppp.p_k10::float,
            ppp.p_k10plus::float,
            ppp.p_over_0_5::float, ppp.p_over_1_5::float, ppp.p_over_2_5::float,
            ppp.p_over_3_5::float, ppp.p_over_4_5::float, ppp.p_over_5_5::float,
            ppp.p_over_6_5::float, ppp.p_over_7_5::float,
            ppp.p_over_8_5::float, ppp.p_over_9_5::float,
            ppp.p_walks_over_0_5::float, ppp.p_walks_over_1_5::float, ppp.p_walks_over_2_5::float,
            ppp.p_walks_over_3_5::float, ppp.p_walks_over_4_5::float, ppp.p_walks_over_5_5::float,
            ppp.p_hits_over_3_5::float, ppp.p_hits_over_4_5::float, ppp.p_hits_over_5_5::float,
            ppp.p_hits_over_6_5::float, ppp.p_hits_over_7_5::float, ppp.p_hits_over_8_5::float,
            ppp.p_er_over_1_5::float, ppp.p_er_over_2_5::float, ppp.p_er_over_3_5::float,
            ppp.p_er_over_4_5::float, ppp.p_er_over_5_5::float,
            ppp.sp_k_rate_season::float,
            ppp.sp_innings_season::float,
            ppp.opp_lineup_k_rate::float,
            ppp.expected_ip::float,
            g.away_team_name AS away_team,
            g.home_team_name AS home_team
        FROM (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY game_id, pitcher_id
                    ORDER BY as_of_ts DESC
                ) AS rn
            FROM public.pitcher_prop_predictions
            {date_clause}
        ) ppp
        JOIN public.games g
          ON g.game_id = ppp.game_id AND g.game_date = ppp.game_date
        WHERE ppp.rn = 1
        ORDER BY ppp.game_id, ppp.is_home DESC
    """

    conn = psycopg2.connect(dsn, connect_timeout=25)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(batters_q, params)
            batters = [dict(x) for x in cur.fetchall()]
            cur.execute(pitchers_q, params)
            pitchers = [dict(x) for x in cur.fetchall()]
    finally:
        conn.close()

    for row in batters + pitchers:
        for k, v in list(row.items()):
            if hasattr(v, "isoformat"):
                row[k] = str(v)[:10] if k == "game_date" else v.isoformat()

    return {"batters": batters, "pitchers": pitchers, "source": "postgres"}


def _run_players_bq_query(client, date_str: str | None) -> dict:
    date_filter = f"AND game_date = DATE('{date_str}')" if date_str else ""
    query = f"""
        WITH latest_batters AS (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY game_id, batter_id
                    ORDER BY COALESCE(lineup_confirmed, FALSE) DESC, as_of_ts DESC
                ) AS rn
            FROM `mlb-model-491223.mlb_model_logs.player_prop_predictions`
            WHERE TRUE {date_filter}
        ),
        latest_pitchers AS (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY game_id, pitcher_id
                    ORDER BY as_of_ts DESC
                ) AS rn
            FROM `mlb-model-491223.mlb_model_logs.pitcher_prop_predictions`
            WHERE TRUE {date_filter}
        )
        SELECT *
        FROM (
            SELECT
                'batter' AS row_type,
                CAST(game_date AS STRING) AS game_date,
                game_id,
                batter_id,
                batter_name,
                team_id,
                batting_order,
                sp_id,
                sp_name,
                p_hit,
                p_2plus_hits,
                p_hr,
                p_k,
                p_2plus_bases,
                p_walk,
                matchup_score,
                platoon_advantage,
                batter_xwoba_season,
                batter_hit_rate_30d,
                lineup_confirmed,
                is_home,
                away_team,
                home_team,
                CAST(NULL AS INT64) AS pitcher_id,
                CAST(NULL AS STRING) AS pitcher_name,
                CAST(NULL AS FLOAT64) AS lambda_k,
                CAST(NULL AS FLOAT64) AS lambda_walks,
                CAST(NULL AS FLOAT64) AS lambda_hits,
                CAST(NULL AS FLOAT64) AS lambda_er,
                CAST(NULL AS FLOAT64) AS p_k0,
                CAST(NULL AS FLOAT64) AS p_k1,
                CAST(NULL AS FLOAT64) AS p_k2,
                CAST(NULL AS FLOAT64) AS p_k3,
                CAST(NULL AS FLOAT64) AS p_k4,
                CAST(NULL AS FLOAT64) AS p_k5,
                CAST(NULL AS FLOAT64) AS p_k6,
                CAST(NULL AS FLOAT64) AS p_k7,
                CAST(NULL AS FLOAT64) AS p_k8,
                CAST(NULL AS FLOAT64) AS p_k9,
                CAST(NULL AS FLOAT64) AS p_k10,
                CAST(NULL AS FLOAT64) AS p_k10plus,
                CAST(NULL AS FLOAT64) AS p_over_0_5,
                CAST(NULL AS FLOAT64) AS p_over_1_5,
                CAST(NULL AS FLOAT64) AS p_over_2_5,
                CAST(NULL AS FLOAT64) AS p_over_3_5,
                CAST(NULL AS FLOAT64) AS p_over_4_5,
                CAST(NULL AS FLOAT64) AS p_over_5_5,
                CAST(NULL AS FLOAT64) AS p_over_6_5,
                CAST(NULL AS FLOAT64) AS p_over_7_5,
                CAST(NULL AS FLOAT64) AS p_over_8_5,
                CAST(NULL AS FLOAT64) AS p_over_9_5,
                CAST(NULL AS FLOAT64) AS p_walks_over_0_5,
                CAST(NULL AS FLOAT64) AS p_walks_over_1_5,
                CAST(NULL AS FLOAT64) AS p_walks_over_2_5,
                CAST(NULL AS FLOAT64) AS p_walks_over_3_5,
                CAST(NULL AS FLOAT64) AS p_walks_over_4_5,
                CAST(NULL AS FLOAT64) AS p_walks_over_5_5,
                CAST(NULL AS FLOAT64) AS p_hits_over_3_5,
                CAST(NULL AS FLOAT64) AS p_hits_over_4_5,
                CAST(NULL AS FLOAT64) AS p_hits_over_5_5,
                CAST(NULL AS FLOAT64) AS p_hits_over_6_5,
                CAST(NULL AS FLOAT64) AS p_hits_over_7_5,
                CAST(NULL AS FLOAT64) AS p_hits_over_8_5,
                CAST(NULL AS FLOAT64) AS p_er_over_1_5,
                CAST(NULL AS FLOAT64) AS p_er_over_2_5,
                CAST(NULL AS FLOAT64) AS p_er_over_3_5,
                CAST(NULL AS FLOAT64) AS p_er_over_4_5,
                CAST(NULL AS FLOAT64) AS p_er_over_5_5,
                CAST(NULL AS FLOAT64) AS sp_k_rate_season,
                CAST(NULL AS FLOAT64) AS sp_innings_season,
                CAST(NULL AS FLOAT64) AS opp_lineup_k_rate,
                CAST(NULL AS FLOAT64) AS expected_ip
            FROM latest_batters
            WHERE rn = 1
            UNION ALL
            SELECT
                'pitcher' AS row_type,
                CAST(game_date AS STRING) AS game_date,
                game_id,
                CAST(NULL AS INT64) AS batter_id,
                CAST(NULL AS STRING) AS batter_name,
                CAST(NULL AS INT64) AS team_id,
                CAST(NULL AS INT64) AS batting_order,
                CAST(NULL AS INT64) AS sp_id,
                CAST(NULL AS STRING) AS sp_name,
                CAST(NULL AS FLOAT64) AS p_hit,
                CAST(NULL AS FLOAT64) AS p_2plus_hits,
                CAST(NULL AS FLOAT64) AS p_hr,
                CAST(NULL AS FLOAT64) AS p_k,
                CAST(NULL AS FLOAT64) AS p_2plus_bases,
                CAST(NULL AS FLOAT64) AS p_walk,
                CAST(NULL AS FLOAT64) AS matchup_score,
                CAST(NULL AS INT64) AS platoon_advantage,
                CAST(NULL AS FLOAT64) AS batter_xwoba_season,
                CAST(NULL AS FLOAT64) AS batter_hit_rate_30d,
                CAST(NULL AS BOOL) AS lineup_confirmed,
                is_home,
                away_team,
                home_team,
                pitcher_id,
                pitcher_name,
                lambda_k,
                lambda_walks,
                lambda_hits,
                lambda_er,
                p_k0, p_k1, p_k2, p_k3, p_k4, p_k5, p_k6, p_k7, p_k8, p_k9, p_k10,
                p_k10plus,
                p_over_0_5, p_over_1_5, p_over_2_5, p_over_3_5,
                p_over_4_5, p_over_5_5, p_over_6_5, p_over_7_5,
                p_over_8_5, p_over_9_5,
                p_walks_over_0_5, p_walks_over_1_5, p_walks_over_2_5, p_walks_over_3_5, p_walks_over_4_5, p_walks_over_5_5,
                p_hits_over_3_5, p_hits_over_4_5, p_hits_over_5_5, p_hits_over_6_5, p_hits_over_7_5, p_hits_over_8_5,
                p_er_over_1_5, p_er_over_2_5, p_er_over_3_5, p_er_over_4_5, p_er_over_5_5,
                sp_k_rate_season,
                sp_innings_season,
                opp_lineup_k_rate,
                expected_ip
            FROM latest_pitchers
            WHERE rn = 1
        )
        ORDER BY game_id, row_type, is_home DESC, batting_order
    """

    rows = [_row_to_dict(r) for r in client.query(query).result()]
    batters = [{k: v for k, v in r.items() if k != "row_type"} for r in rows if r.get("row_type") == "batter"]
    pitchers = [{k: v for k, v in r.items() if k != "row_type"} for r in rows if r.get("row_type") == "pitcher"]
    return {"batters": batters, "pitchers": pitchers, "source": "bigquery"}


def _run_players_view(client, date_str: str | None) -> dict:
    games = _run_daily_games_query(client, date_str)
    pg_dsn = (os.environ.get("PG_DSN") or "").strip()

    props = {"batters": [], "pitchers": [], "source": "none"}
    try:
        props = _run_players_bq_query(client, date_str)
    except Exception as exc:
        print(f"players BQ query failed: {exc}")

    if not props.get("batters") and not props.get("pitchers") and pg_dsn:
        try:
            props = _fetch_players_from_pg(pg_dsn, date_str)
        except Exception as exc:
            print(f"players PG fallback failed: {exc}")

    if pg_dsn:
        try:
            frozen = _fetch_pregame_odds_from_pg(pg_dsn, date_str)
            games = _apply_pregame_odds_overlay(games, frozen)
        except Exception as exc:
            print(f"pregame odds overlay failed: {exc}")

    return {
        "games": games,
        "batters": _filter_batters_for_confirmed_lineups(props.get("batters") or []),
        "pitchers": props.get("pitchers") or [],
        "meta": {"source": props.get("source", "none")},
    }


def _fetch_graded_games_from_bq(client, min_game_date):
    """
    Reads graded games from the BigQuery mirror of Postgres
    (`mlb_model_logs.daily_games`) written by the daily pipeline.

    One row per game_id, using the most-recent prediction snapshot, filtered to:
      - game_date >= @min_game_date
      - both final-run columns populated
    """
    query = f"""
        WITH latest AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY game_id
                    ORDER BY as_of_ts DESC
                ) AS rn
            FROM `mlb-model-491223.mlb_model_logs.daily_games`
            WHERE game_date >= DATE('{min_game_date}')
              AND game_date >= DATE('{V10_LAUNCH_DATE}')
        )
        SELECT
            CAST(game_date AS STRING)                                      AS game_date,
            game_id,
            home_team,
            away_team,
            CAST(p_win_home AS FLOAT64)                                    AS p_home,
            CAST(p_win_away AS FLOAT64)                                    AS p_away,
            CAST(p_home_market_median AS FLOAT64)                          AS p_home_market,
            CAST(p_away_market_median AS FLOAT64)                          AS p_away_market,
            CAST(total_runs_pred AS FLOAT64)                               AS total_runs_pred,
            ou_recommendation,
            COALESCE(
                CAST(closing_ou_line AS FLOAT64),
                CAST(morning_ou_line AS FLOAT64),
                CAST(ou_line AS FLOAT64)
            )                                                              AS ou_line,
            COALESCE(
                CAST(morning_home_price AS INT64),
                CAST(closing_home_price AS INT64)
            )                                                              AS home_odds,
            COALESCE(
                CAST(morning_away_price AS INT64),
                CAST(closing_away_price AS INT64)
            )                                                              AS away_odds,
            CAST(home_runs AS INT64)                                       AS home_runs,
            CAST(away_runs AS INT64)                                       AS away_runs,
            status
        FROM latest
        WHERE rn = 1
          AND home_runs IS NOT NULL
          AND away_runs IS NOT NULL
          AND (
              LOWER(IFNULL(status, '')) LIKE 'final%'
              OR LOWER(IFNULL(status, '')) = 'game over'
              OR LOWER(IFNULL(status, '')) LIKE 'completed%'
          )
        ORDER BY game_date ASC, game_id ASC
    """
    return [_row_to_dict(r) for r in client.query(query).result()]


def _fetch_graded_games_v9_from_bq(client) -> list:
    query = f"""
        WITH latest AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY game_id
                    ORDER BY as_of_ts DESC
                ) AS rn
            FROM `mlb-model-491223.mlb_model_logs.daily_games`
            WHERE game_date >= DATE('{V9_LAUNCH_DATE}')
              AND game_date <= DATE('{V9_END_DATE}')
        )
        SELECT
            CAST(game_date AS STRING)                                      AS game_date,
            game_id,
            home_team,
            away_team,
            CAST(p_win_home AS FLOAT64)                                    AS p_home,
            CAST(p_win_away AS FLOAT64)                                    AS p_away,
            CAST(p_home_market_median AS FLOAT64)                          AS p_home_market,
            CAST(p_away_market_median AS FLOAT64)                          AS p_away_market,
            CAST(total_runs_pred AS FLOAT64)                               AS total_runs_pred,
            ou_recommendation,
            COALESCE(
                CAST(closing_ou_line AS FLOAT64),
                CAST(morning_ou_line AS FLOAT64),
                CAST(ou_line AS FLOAT64)
            )                                                              AS ou_line,
            COALESCE(
                CAST(morning_home_price AS INT64),
                CAST(closing_home_price AS INT64)
            )                                                              AS home_odds,
            COALESCE(
                CAST(morning_away_price AS INT64),
                CAST(closing_away_price AS INT64)
            )                                                              AS away_odds,
            CAST(home_runs AS INT64)                                       AS home_runs,
            CAST(away_runs AS INT64)                                       AS away_runs,
            status
        FROM latest
        WHERE rn = 1
          AND home_runs IS NOT NULL
          AND away_runs IS NOT NULL
          AND (
              LOWER(IFNULL(status, '')) LIKE 'final%'
              OR LOWER(IFNULL(status, '')) = 'game over'
              OR LOWER(IFNULL(status, '')) LIKE 'completed%'
          )
        ORDER BY game_date ASC, game_id ASC
    """
    return [_row_to_dict(r) for r in client.query(query).result()]


def _ml_pnl(pick_won, pick_odds):
    """American-odds P&L on a $10 stake."""
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


def _stored_ou_is_pass_like(raw) -> bool:
    """True when inference stored Pass / PUSH (exclude from accuracy O/U grading)."""
    if raw is None:
        return False
    s = str(raw).strip().upper()
    if not s:
        return False
    return s in ("PUSH", "PASS")


def _grade_games(raw_rows):
    """
    Produces a flat list of graded bet rows (one per {ml|ou} pick).
    Each row: {game_date, game_id, kind, is_hit, pnl_dollars, conf, conf_bucket, ou_pick, ou_line, total_actual, ou_result}
    """
    graded = []
    for r in raw_rows:
        game_date = r.get("game_date")
        game_id = r.get("game_id")
        p_home = r.get("p_home")
        p_away = r.get("p_away")
        home_runs = r.get("home_runs")
        away_runs = r.get("away_runs")
        home_odds = r.get("home_odds")
        away_odds = r.get("away_odds")
        total_pred = r.get("total_runs_pred")
        ou_line = r.get("ou_line")

        if home_runs is None or away_runs is None:
            continue
        home_runs = int(home_runs)
        away_runs = int(away_runs)
        total_actual = home_runs + away_runs

        # --- Moneyline (skip ties and missing predictions)
        if p_home is not None and p_away is not None and home_runs != away_runs:
            p_home_f = float(p_home)
            p_away_f = float(p_away)
            conf = max(p_home_f, p_away_f)
            pick_home = p_home_f >= p_away_f
            pick_won = (pick_home and home_runs > away_runs) or ((not pick_home) and away_runs > home_runs)
            pick_odds = home_odds if pick_home else away_odds
            pnl = _ml_pnl(pick_won, pick_odds)
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
                "pnl_dollars": float(pnl),
                "conf": conf,
                "conf_bucket": bucket,
                "ou_pick": None,
                "ou_line": None,
                "total_actual": total_actual,
                "ou_result": None,
            })

        # --- Over/Under: Pass / PUSH grades nothing (omit from picks, W-L, P&L — same as UI)
        if total_pred is not None and ou_line is not None:
            tp = float(total_pred)
            line = float(ou_line)
            stored_rec = r.get("ou_recommendation")

            ou_pass = abs(tp - line) < OU_PRED_LINE_GAP or _stored_ou_is_pass_like(stored_rec)

            ou_pick = None
            ou_result = None

            if not ou_pass:
                ou_pick = "over" if tp > line else "under"
                half_line = (
                    (abs(line * 10 - int(round(line * 10))) < 1e-6)
                    and (int(round(line * 10)) % 10 != 0)
                )
                if (not half_line) and total_actual == int(round(line)):
                    ou_result = "push"
                elif ou_pick == "over" and total_actual > line:
                    ou_result = "hit"
                elif ou_pick == "under" and total_actual < line:
                    ou_result = "hit"
                else:
                    ou_result = "miss"

            if ou_result in ("hit", "miss"):
                if ou_result == "hit":
                    pnl = 10.0 * (100.0 / 110.0)
                else:
                    pnl = -10.0
                graded.append({
                    "game_date": game_date,
                    "game_id": game_id,
                    "kind": "ou",
                    "is_hit": 1 if ou_result == "hit" else 0,
                    "pnl_dollars": float(pnl),
                    "conf": None,
                    "conf_bucket": None,
                    "ou_pick": ou_pick,
                    "ou_line": line,
                    "total_actual": total_actual,
                    "ou_result": ou_result,
                })
    return graded


def _game_date_key(d) -> str:
    if d is None:
        return ""
    s = d if isinstance(d, str) else str(d)
    return s[:10] if len(s) >= 10 else s


# Model Accuracy tab (O/U only): omit graded O/U picks on these slate dates. Moneyline unchanged.
OU_ACCURACY_EXCLUDED_SLATE_DATES = frozenset(
    {
        "2026-04-23",
        "2026-04-24",
        "2026-04-25",
        "2026-04-26",
    }
)


def _drop_ou_bets_on_excluded_slates(graded_rows: list) -> list:
    if not OU_ACCURACY_EXCLUDED_SLATE_DATES:
        return graded_rows
    out = []
    for r in graded_rows:
        if (r.get("kind") or "ml") != "ou":
            out.append(r)
            continue
        if _game_date_key(r.get("game_date")) in OU_ACCURACY_EXCLUDED_SLATE_DATES:
            continue
        out.append(r)
    return out


_ML_CALIBRATION_BUCKETS = [
    ("50-55%", 0.50, 0.55),
    ("55-60%", 0.55, 0.60),
    ("60-65%", 0.60, 0.65),
    ("65%+", 0.65, 1.01),
]


def _compute_ml_headline_and_calibration(raw_rows: list) -> tuple[dict, list]:
    """Headline ML metrics + per-bucket predicted vs actual win rates (v10 games only)."""
    buckets = {
        label: {"bucket": label, "n": 0, "wins": 0, "sum_pred": 0.0}
        for label, _, _ in _ML_CALIBRATION_BUCKETS
    }
    n = 0
    brier_model_sum = 0.0
    brier_market_sum = 0.0
    market_n = 0
    correct = 0
    home_wins = 0

    for r in raw_rows:
        home_runs = r.get("home_runs")
        away_runs = r.get("away_runs")
        p_home = r.get("p_home")
        p_away = r.get("p_away")
        if home_runs is None or away_runs is None or p_home is None or p_away is None:
            continue
        home_runs = int(home_runs)
        away_runs = int(away_runs)
        if home_runs == away_runs:
            continue

        n += 1
        home_won = home_runs > away_runs
        if home_won:
            home_wins += 1

        p_home_f = float(p_home)
        p_away_f = float(p_away)
        pick_home = p_home_f >= p_away_f
        pick_won = (pick_home and home_won) or ((not pick_home) and (not home_won))
        if pick_won:
            correct += 1

        pick_prob = p_home_f if pick_home else p_away_f
        y_home = 1.0 if home_won else 0.0
        brier_model_sum += (p_home_f - y_home) ** 2

        p_mkt_home = r.get("p_home_market")
        if p_mkt_home is not None:
            brier_market_sum += (float(p_mkt_home) - y_home) ** 2
            market_n += 1

        for label, lo, hi in _ML_CALIBRATION_BUCKETS:
            if lo <= pick_prob < hi:
                buckets[label]["n"] += 1
                buckets[label]["wins"] += (1 if pick_won else 0)
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
            calibration.append({
                "bucket": label,
                "n": b["n"],
                "pred_pct": round(pred_pct, 1),
                "actual_pct": round(actual_pct, 1),
            })
        else:
            calibration.append({"bucket": label, "n": 0, "pred_pct": None, "actual_pct": None})

    ece_pct = round(100.0 * ece, 1) if n else None
    brier_model = round(brier_model_sum / n, 3) if n else None
    brier_market = round(brier_market_sum / market_n, 3) if market_n else None

    headline = {
        "games_graded": n,
        "calibration_error_pct": ece_pct,
        "brier_score": brier_model,
        "brier_market": brier_market,
        "accuracy_pct": round(100.0 * correct / n, 1) if n else None,
        "pick_home_baseline_pct": round(100.0 * home_wins / n, 1) if n else None,
    }
    return headline, calibration


def _fetch_pitcher_k_calibration_from_pg(dsn: str, min_game_date: str) -> list:
    """Actual vs predicted over-K rates for common SP lines (requires statcast + props in PG)."""
    dsn = _normalize_pg_dsn(dsn.strip())
    q = """
        WITH latest_props AS (
            SELECT DISTINCT ON (ppp.game_id, ppp.pitcher_id)
                ppp.game_id,
                ppp.pitcher_id,
                ppp.p_over_3_5,
                ppp.p_over_5_5,
                ppp.p_over_7_5
            FROM public.pitcher_prop_predictions ppp
            JOIN public.games g
              ON g.game_id = ppp.game_id AND g.game_date = ppp.game_date
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
            SELECT
                sp.pitcher AS pitcher_id,
                sp.game_pk AS game_id,
                SUM(CASE WHEN sp.events = 'strikeout' THEN 1 ELSE 0 END) AS k_count
            FROM public.statcast_pitches sp
            WHERE sp.game_date >= %s::date
              AND sp.pitcher IS NOT NULL
              AND sp.game_pk IS NOT NULL
            GROUP BY sp.pitcher, sp.game_pk
        )
        SELECT
            lp.p_over_3_5,
            lp.p_over_5_5,
            lp.p_over_7_5,
            ak.k_count
        FROM latest_props lp
        JOIN actual_ks ak
          ON ak.game_id = lp.game_id AND ak.pitcher_id = lp.pitcher_id
        WHERE ak.k_count IS NOT NULL
    """
    conn = psycopg2.connect(dsn, connect_timeout=25)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(q, (min_game_date, min_game_date))
            rows = [dict(x) for x in cur.fetchall()]
    finally:
        conn.close()

    if not rows:
        return [], 0

    thresholds = [
        ("Over 3.5", 3.5, "p_over_3_5"),
        ("Over 5.5", 5.5, "p_over_5_5"),
        ("Over 7.5", 7.5, "p_over_7_5"),
    ]
    out = []
    n = len(rows)
    for label, thresh, col in thresholds:
        k_floor = int(thresh + 0.5)
        preds = [float(r[col]) for r in rows if r.get(col) is not None]
        actuals = [1 if int(r["k_count"]) >= k_floor else 0 for r in rows if r.get(col) is not None]
        if not preds:
            out.append({"line": label, "n": 0, "pred_pct": None, "actual_pct": None})
            continue
        pred_pct = round(100.0 * sum(preds) / len(preds), 1)
        actual_pct = round(100.0 * sum(actuals) / len(actuals), 1)
        out.append({"line": label, "n": len(preds), "pred_pct": pred_pct, "actual_pct": actual_pct})
    return out, n


def _fetch_pitcher_prop_calibration_from_pg(
    dsn: str,
    min_game_date: str,
    pred_cols: list[tuple[str, float, str]],
    actual_sql: str,
) -> tuple[list, int]:
    """Generic O/U calibration: predicted over-rate vs actual for graded SP starts."""
    dsn = _normalize_pg_dsn(dsn.strip())
    pred_select = ", ".join(f"ppp.{col}" for _, _, col in pred_cols)
    q = f"""
        WITH latest_props AS (
            SELECT DISTINCT ON (ppp.game_id, ppp.pitcher_id)
                ppp.game_id,
                ppp.pitcher_id,
                {pred_select}
            FROM public.pitcher_prop_predictions ppp
            JOIN public.games g
              ON g.game_id = ppp.game_id AND g.game_date = ppp.game_date
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
        actuals AS (
            {actual_sql}
        )
        SELECT lp.*, a.actual_count
        FROM latest_props lp
        JOIN actuals a
          ON a.game_id = lp.game_id AND a.pitcher_id = lp.pitcher_id
        WHERE a.actual_count IS NOT NULL
    """
    conn = psycopg2.connect(dsn, connect_timeout=25)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(q, (min_game_date, min_game_date))
            rows = [dict(x) for x in cur.fetchall()]
    finally:
        conn.close()
    if not rows:
        return [], 0
    out = []
    for label, thresh, col in pred_cols:
        k_floor = int(thresh + 0.5)
        preds = [float(r[col]) for r in rows if r.get(col) is not None]
        actuals = [1 if int(r["actual_count"]) >= k_floor else 0 for r in rows if r.get(col) is not None]
        if not preds:
            out.append({"line": label, "n": 0, "pred_pct": None, "actual_pct": None})
            continue
        out.append({
            "line": label,
            "n": len(preds),
            "pred_pct": round(100.0 * sum(preds) / len(preds), 1),
            "actual_pct": round(100.0 * sum(actuals) / len(actuals), 1),
        })
    return out, len(rows)


def _fetch_pitcher_walks_calibration_from_pg(dsn: str, min_game_date: str) -> tuple[list, int]:
    return _fetch_pitcher_prop_calibration_from_pg(
        dsn,
        min_game_date,
        [
            ("Over 1.5", 1.5, "p_walks_over_1_5"),
            ("Over 2.5", 2.5, "p_walks_over_2_5"),
            ("Over 3.5", 3.5, "p_walks_over_3_5"),
        ],
        """
            SELECT sp.pitcher AS pitcher_id, sp.game_pk AS game_id,
                   SUM(CASE WHEN sp.events IN ('walk','hit_by_pitch') THEN 1 ELSE 0 END) AS actual_count
            FROM public.statcast_pitches sp
            WHERE sp.game_date >= %s::date
              AND sp.pitcher IS NOT NULL AND sp.game_pk IS NOT NULL
            GROUP BY sp.pitcher, sp.game_pk
        """,
    )


def _fetch_pitcher_hits_calibration_from_pg(dsn: str, min_game_date: str) -> tuple[list, int]:
    return _fetch_pitcher_prop_calibration_from_pg(
        dsn,
        min_game_date,
        [
            ("Over 4.5", 4.5, "p_hits_over_4_5"),
            ("Over 5.5", 5.5, "p_hits_over_5_5"),
            ("Over 6.5", 6.5, "p_hits_over_6_5"),
        ],
        """
            SELECT sp.pitcher AS pitcher_id, sp.game_pk AS game_id,
                   SUM(CASE WHEN sp.events IN ('single','double','triple','home_run')
                            THEN 1 ELSE 0 END) AS actual_count
            FROM public.statcast_pitches sp
            WHERE sp.game_date >= %s::date
              AND sp.pitcher IS NOT NULL AND sp.game_pk IS NOT NULL
            GROUP BY sp.pitcher, sp.game_pk
        """,
    )


def _fetch_pitcher_er_calibration_from_pg(dsn: str, min_game_date: str) -> tuple[list, int]:
    return _fetch_pitcher_prop_calibration_from_pg(
        dsn,
        min_game_date,
        [
            ("Over 2.5", 2.5, "p_er_over_2_5"),
            ("Over 3.5", 3.5, "p_er_over_3_5"),
            ("Over 4.5", 4.5, "p_er_over_4_5"),
        ],
        """
            SELECT ps.pitcher_id, ps.game_id, ps.earned_runs AS actual_count
            FROM public.pitcher_starts ps
            WHERE ps.game_date >= %s::date AND ps.earned_runs IS NOT NULL
        """,
    )


def _run_accuracy_query(client=None):
    """
    Prefers Postgres (Cloud SQL) when PG_DSN is set so accuracy updates as soon
    as scores land; otherwise uses the BigQuery mirror. v10 only: game_date >= launch.
    """
    min_game_date = V10_LAUNCH_DATE
    source_meta = "BigQuery mlb-model-491223.mlb_model_logs.daily_games (mirror of Postgres)"
    raw_rows = None
    pg_dsn = (os.environ.get("PG_DSN") or "").strip()
    if pg_dsn:
        try:
            raw_rows = _fetch_graded_games_from_pg(pg_dsn, min_game_date)
            source_meta = "Postgres Cloud SQL (inference + games; updates when scores are ingested)"
        except Exception:
            raw_rows = None
    if raw_rows is None:
        if client is None:
            client = bigquery.Client()
        raw_rows = _fetch_graded_games_from_bq(client, min_game_date)
        source_meta = "BigQuery mlb-model-491223.mlb_model_logs.daily_games (mirror of Postgres)"
    rows = _drop_ou_bets_on_excluded_slates(_grade_games(raw_rows))

    headline, ml_calibration = _compute_ml_headline_and_calibration(raw_rows)

    pitcher_k_calibration = {"starters_graded": 0, "lines": []}
    pitcher_walks_calibration = {"starters_graded": 0, "lines": []}
    pitcher_hits_calibration = {"starters_graded": 0, "lines": []}
    pitcher_er_calibration = {"starters_graded": 0, "lines": []}
    if pg_dsn:
        try:
            k_lines, k_n = _fetch_pitcher_k_calibration_from_pg(pg_dsn, min_game_date)
            pitcher_k_calibration = {"starters_graded": k_n, "lines": k_lines}
            walks_lines, walks_n = _fetch_pitcher_walks_calibration_from_pg(pg_dsn, min_game_date)
            pitcher_walks_calibration = {"starters_graded": walks_n, "lines": walks_lines}
            hits_lines, hits_n = _fetch_pitcher_hits_calibration_from_pg(pg_dsn, min_game_date)
            pitcher_hits_calibration = {"starters_graded": hits_n, "lines": hits_lines}
            er_lines, er_n = _fetch_pitcher_er_calibration_from_pg(pg_dsn, min_game_date)
            pitcher_er_calibration = {"starters_graded": er_n, "lines": er_lines}
        except Exception:
            pitcher_k_calibration = {"starters_graded": 0, "lines": []}
            pitcher_walks_calibration = {"starters_graded": 0, "lines": []}
            pitcher_hits_calibration = {"starters_graded": 0, "lines": []}
            pitcher_er_calibration = {"starters_graded": 0, "lines": []}

    def _new_overall():
        return {"bets": 0, "wins": 0, "win_pct": None, "net_dollars": 0.0, "roi_pct": None}

    def _new_buckets():
        return {
            "50-55%": {"bucket": "50-55%", "bets": 0, "wins": 0, "net_dollars": 0.0},
            "55-60%": {"bucket": "55-60%", "bets": 0, "wins": 0, "net_dollars": 0.0},
            "60-65%": {"bucket": "60-65%", "bets": 0, "wins": 0, "net_dollars": 0.0},
            "65%+":   {"bucket": "65%+",   "bets": 0, "wins": 0, "net_dollars": 0.0},
        }

    ml_overall = _new_overall()
    ou_overall = _new_overall()
    ml_buckets = _new_buckets()
    ml_daily = {}
    ou_daily = {}
    combined_daily = {}
    ou_picks = {"over": 0, "under": 0}

    for r in rows:
        kind = r.get("kind") or "ml"
        hit = int(r["is_hit"])
        pnl = float(r["pnl_dollars"] or 0.0)
        d = r["game_date"]
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

            if d not in ml_daily:
                ml_daily[d] = {"date": d, "bets": 0, "wins": 0, "net_dollars": 0.0}
            ml_daily[d]["bets"] += 1
            ml_daily[d]["wins"] += hit
            ml_daily[d]["net_dollars"] += pnl
        elif kind == "ou":
            pick = r.get("ou_pick")
            if pick in ou_picks:
                ou_picks[pick] += 1
            if d not in ou_daily:
                ou_daily[d] = {"date": d, "bets": 0, "wins": 0, "net_dollars": 0.0}
            ou_daily[d]["bets"] += 1
            ou_daily[d]["wins"] += hit
            ou_daily[d]["net_dollars"] += pnl

        if d not in combined_daily:
            combined_daily[d] = {"date": d, "bets": 0, "wins": 0, "net_dollars": 0.0}
        combined_daily[d]["bets"] += 1
        combined_daily[d]["wins"] += hit
        combined_daily[d]["net_dollars"] += pnl

    def _finalize_overall(o):
        if o["bets"] > 0:
            o["win_pct"] = round(100.0 * o["wins"] / o["bets"], 1)
            o["roi_pct"] = round(100.0 * o["net_dollars"] / (o["bets"] * 10.0), 1)
        o["net_dollars"] = round(o["net_dollars"], 2)

    _finalize_overall(ml_overall)
    _finalize_overall(ou_overall)

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

    def _finalize_daily(daily_dict):
        out = []
        cum = 0.0
        cum_rows = []
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

    ml_daily_rows, ml_cum_rows = _finalize_daily(ml_daily)
    ou_daily_rows, ou_cum_rows = _finalize_daily(ou_daily)
    combined_daily_rows, combined_cum_rows = _finalize_daily(combined_daily)

    if client is None:
        client = bigquery.Client()
    v9_history = _build_v9_history(pg_dsn=pg_dsn if pg_dsn else None, client=client)

    return {
        "headline": headline,
        "ml_calibration": ml_calibration,
        "pitcher_k_calibration": pitcher_k_calibration,
        "pitcher_walks_calibration": pitcher_walks_calibration,
        "pitcher_hits_calibration": pitcher_hits_calibration,
        "pitcher_er_calibration": pitcher_er_calibration,
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
            "source": source_meta,
            "version": "v10",
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


def _run_accuracy_snapshot_query(client=None, date_str: str | None = None):
    date_str = date_str or _today_pacific()
    if client is None:
        client = bigquery.Client()
    query = f"""
        SELECT payload_json
        FROM `mlb-model-491223.mlb_model_logs.model_performance_snapshot`
        WHERE snapshot_date <= DATE('{date_str}')
        ORDER BY snapshot_date DESC
        LIMIT 1
    """
    rows = list(client.query(query).result())
    if not rows:
        return _run_accuracy_query(client)
    payload = json.loads(rows[0]["payload_json"])
    payload.setdefault("meta", {})
    snap_min = payload["meta"].get("min_game_date")
    if snap_min != V10_LAUNCH_DATE:
        return _run_accuracy_query(client)
    if not payload.get("v9_history"):
        pg_dsn = (os.environ.get("PG_DSN") or "").strip()
        if pg_dsn:
            try:
                payload["v9_history"] = _build_v9_history(pg_dsn=pg_dsn, client=client)
            except Exception:
                pass
    payload["meta"]["source"] = "bigquery.model_performance_snapshot"
    return payload


def _fetch_daily_edges_bq(client, date_str: str) -> list:
    query = f"""
        SELECT rank, edge_type, prop_subtype, pick_description, detail_line,
               rate_detail_line, market_line, model_prob_pct,
               model_value, comparison_value, edge_magnitude, direction,
               game_id, player_id, team_id, team_abbr, team_name
        FROM `mlb-model-491223.mlb_model_logs.daily_edges`
        WHERE edge_date = DATE('{date_str}')
        ORDER BY rank
    """
    return [_row_to_dict(r) for r in client.query(query).result()]


def _format_edge_api_row(row: dict) -> dict:
    edge_type = row.get("edge_type")
    direction = row.get("direction") or "over"
    mag = _safe_float(row.get("edge_magnitude"))
    model_v = _safe_float(row.get("model_value"))

    if edge_type == "ml":
        stat_label, edge_label = "MODEL", "EDGE"
        stat_value = f"{model_v:.1f}%" if model_v is not None else "—"
        edge_value = f"+{mag:.1f}" if mag is not None else "—"
    elif edge_type == "total":
        stat_label, edge_label = "PROJ", "EDGE"
        stat_value = f"{model_v:.1f}" if model_v is not None else "—"
        if mag is not None:
            edge_value = f"{'+' if direction == 'over' else '-'}{mag:.1f}"
        else:
            edge_value = "—"
    elif edge_type in ("k", "walks", "hits", "er"):
        market_line = _safe_float(row.get("market_line"))
        model_prob_pct = _safe_float(row.get("model_prob_pct"))
        stat_label, edge_label = "MODEL", "LINE"
        stat_value = f"{model_prob_pct:.1f}%" if model_prob_pct is not None else "—"
        edge_value = f"{market_line:.1f}" if market_line is not None else "—"
    else:
        stat_label, edge_label = "MODEL", "VS AVG"
        stat_value = f"{model_v:.1f}%" if model_v is not None else "—"
        if mag is not None:
            edge_value = f"+{mag:.1f}"
        else:
            edge_value = "—"

    return {
        "type": edge_type,
        "subtype": row.get("prop_subtype"),
        "direction": direction,
        "rank": row.get("rank"),
        "title": row.get("pick_description"),
        "detail": row.get("detail_line"),
        "rate_detail": row.get("rate_detail_line"),
        "market_line": _safe_float(row.get("market_line")),
        "model_prob_pct": _safe_float(row.get("model_prob_pct")),
        "stat_label": stat_label,
        "stat_value": stat_value,
        "edge_label": edge_label,
        "edge_value": edge_value,
        "game_id": row.get("game_id"),
        "player_id": row.get("player_id"),
        "team_id": row.get("team_id"),
        "team_abbr": row.get("team_abbr"),
        "team_name": row.get("team_name"),
        "model_value_num": model_v,
        "comparison_value_num": _safe_float(row.get("comparison_value")),
    }


def _run_edges_view(client, date_str: str | None) -> dict:
    date_str = date_str or _today_pacific()
    rows = _fetch_daily_edges_bq(client, date_str)
    edges = [_format_edge_api_row(r) for r in rows]
    return {
        "date": date_str,
        "edges": edges,
        "meta": {"count": len(edges), "source": "bigquery.daily_edges"},
    }


def _fetch_daily_trends_bq(client, date_str: str) -> list:
    query = f"""
        SELECT trend_type, rank, name, meta, team_id, team_abbr, team_name,
               value_primary, value_secondary, value_label, direction
        FROM `mlb-model-491223.mlb_model_logs.daily_trends`
        WHERE trend_date = DATE('{date_str}')
        ORDER BY trend_type, rank
    """
    return [_row_to_dict(r) for r in client.query(query).result()]


def _format_trends_payload(rows: list, date_str: str) -> dict:
    hottest: list = []
    most_hr_last10: list = []
    most_hits_last10: list = []
    hitting_streaks: list = []
    cold_bats_last10: list = []
    k_leaders: list = []
    best_era_last3: list = []
    cold_pitchers: list = []
    best_bullpens_last7: list = []
    teams_trending: list = []
    line_moves: list = []

    for row in rows:
        t = row.get("trend_type")
        rank = row.get("rank")
        name = row.get("name")
        meta = row.get("meta")
        vp = _safe_float(row.get("value_primary"))
        vs = _safe_float(row.get("value_secondary"))
        direction = row.get("direction") or "neutral"
        value_label = row.get("value_label")
        team_payload = {
            "team_id": row.get("team_id"),
            "team_abbr": row.get("team_abbr"),
            "team_name": row.get("team_name") or meta,
        }

        if t == "hot_hitters":
            hottest.append({
                "rank": rank,
                "player_name": name,
                **team_payload,
                "xwoba_14d": vp,
                "pa": int(vs) if vs is not None else None,
                "meta": meta,
            })
        elif t == "most_hr_last10":
            most_hr_last10.append({
                "rank": rank,
                "player_name": name,
                **team_payload,
                "hr": int(vp) if vp is not None else None,
                "pa": int(vs) if vs is not None else None,
                "meta": meta,
            })
        elif t == "most_hits_last10":
            most_hits_last10.append({
                "rank": rank,
                "player_name": name,
                **team_payload,
                "hits": int(vp) if vp is not None else None,
                "pa": int(vs) if vs is not None else None,
                "meta": meta,
            })
        elif t == "hitting_streaks":
            hitting_streaks.append({
                "rank": rank,
                "player_name": name,
                **team_payload,
                "streak": int(vp) if vp is not None else None,
                "meta": meta,
            })
        elif t == "cold_bats_last10":
            cold_bats_last10.append({
                "rank": rank,
                "player_name": name,
                **team_payload,
                "strikeouts": int(vp) if vp is not None else None,
                "hits": int(vs) if vs is not None else None,
                "meta": value_label or meta,
            })
        elif t == "k_leaders":
            k_leaders.append({
                "rank": rank,
                "pitcher_name": name,
                **team_payload,
                "total_k": int(vp) if vp is not None else None,
                "k_per_start": vs,
                "meta": f"{meta or '—'} · {vs} K/start" if vs is not None else meta,
            })
        elif t == "best_era_last3":
            best_era_last3.append({
                "rank": rank,
                "pitcher_name": name,
                **team_payload,
                "era": vp,
                "runs_allowed": int(vs) if vs is not None else None,
                "meta": value_label or meta,
            })
        elif t == "cold_pitchers":
            cold_pitchers.append({
                "rank": rank,
                "pitcher_name": name,
                **team_payload,
                "era": vp,
                "runs_allowed": int(vs) if vs is not None else None,
                "meta": value_label or meta,
            })
        elif t == "best_bullpens_last7":
            best_bullpens_last7.append({
                "rank": rank,
                "team_name": row.get("team_name") or name,
                "team_id": row.get("team_id"),
                "team_abbr": row.get("team_abbr"),
                "era": vp,
                "innings": vs,
                "meta": value_label or meta,
            })
        elif t == "team_form":
            wl = value_label or "0-0"
            parts = wl.split("-")
            wins = int(parts[0]) if parts and parts[0].isdigit() else 0
            losses = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
            run_diff = int(vs) if vs is not None else 0
            streak_n = int(vp) if vp is not None else 0
            teams_trending.append({
                "rank": rank,
                "team_name": row.get("team_name") or name,
                "team_id": row.get("team_id"),
                "team_abbr": row.get("team_abbr"),
                "wins": wins,
                "losses": losses,
                "run_diff": run_diff,
                "win_streak": streak_n,
                "streak": f"W{streak_n}" if streak_n else "—",
                "meta": meta,
            })
        elif t == "line_moves":
            line_moves.append({
                "rank": rank,
                "matchup": name,
                "description": meta,
                "magnitude": round(vp, 1) if vp is not None else None,
                "direction": direction,
                "meta": meta,
            })

    return {
        "hottest_hitters": hottest,
        "most_hr_last10": most_hr_last10,
        "most_hits_last10": most_hits_last10,
        "hitting_streaks": hitting_streaks,
        "cold_bats_last10": cold_bats_last10,
        "k_leaders": k_leaders,
        "best_era_last3": best_era_last3,
        "cold_pitchers": cold_pitchers,
        "best_bullpens_last7": best_bullpens_last7,
        "teams_trending": teams_trending,
        "line_moves": line_moves,
        "meta": {"date": date_str, "source": "daily_trends"},
    }


def _fetch_ml_edges_mini(client, date_str: str) -> list:
    """Top 3 edges (any type) for Trends tab mini-panel."""
    rows = _fetch_daily_edges_bq(client, date_str)
    return [_format_edge_api_row(r) for r in rows[:3]]


def _run_trends_view(client, date_str: str | None = None) -> dict:
    date_str = date_str or _today_pacific()
    rows = _fetch_daily_trends_bq(client, date_str)
    payload = _format_trends_payload(rows, date_str)
    payload["model_edges"] = _fetch_ml_edges_mini(client, date_str)
    return payload


def _run_standings_view(client, date_str: str | None = None) -> dict:
    date_str = date_str or _today_pacific()
    query = f"""
        WITH standings_current AS (
            SELECT *
            FROM `mlb-model-491223.mlb_model_logs.standings`
            WHERE snapshot_date = DATE('{date_str}')
        ),
        projections AS (
            SELECT *
            FROM `mlb-model-491223.mlb_model_logs.standings_projections`
            WHERE snapshot_date = DATE('{date_str}')
        )
        SELECT
            CAST(c.snapshot_date AS STRING) AS snapshot_date,
            c.season,
            c.league_id,
            c.league_name,
            c.division_id,
            c.division_name,
            c.division_name_short,
            c.team_id,
            c.team_name,
            c.abbreviation,
            c.rank,
            c.wins,
            c.losses,
            c.pct,
            c.games_back,
            c.streak,
            c.last_10,
            c.run_diff,
            c.runs_scored,
            c.runs_allowed,
            p.projected_wins,
            p.projected_losses,
            p.projected_record,
            p.playoff_odds,
            p.remaining_games,
            p.simulations
        FROM standings_current c
        LEFT JOIN projections p USING (snapshot_date, team_id)
        ORDER BY c.league_id, c.division_id, c.rank
    """
    rows = [_row_to_dict(r) for r in client.query(query).result()]
    return {
        "date": date_str,
        "standings": rows,
        "meta": {"count": len(rows), "source": "bigquery.standings", "simulations": rows[0].get("simulations") if rows else None},
    }


def _transaction_category(transaction_type: str | None, description: str | None) -> str:
    text = f"{transaction_type or ''} {description or ''}".lower()
    if any(x in text for x in ("injured", "injury", " il", "10-day", "15-day", "60-day")):
        return "injury"
    if any(x in text for x in ("recalled", "selected", "called up", "contract selected", "active roster")):
        return "callup"
    if "trade" in text or "traded" in text:
        return "trade"
    if any(x in text for x in ("sign", "signed", "claimed", "waiver")):
        return "signing"
    if any(x in text for x in ("designated", "dfa", "assigned", "optioned")):
        return "dfa"
    return "other"


def _run_transactions_view(client, end_date: str | None = None, days: int = 14) -> dict:
    end_date = end_date or _today_pacific()
    days = max(1, min(60, int(days or 14)))
    query = f"""
        SELECT
            transaction_id,
            CAST(transaction_date AS STRING) AS transaction_date,
            team_id,
            team_name,
            player_id,
            player_name,
            transaction_type,
            type_code,
            description
        FROM `mlb-model-491223.mlb_model_logs.transactions`
        WHERE transaction_date BETWEEN DATE_SUB(DATE('{end_date}'), INTERVAL {days - 1} DAY) AND DATE('{end_date}')
        ORDER BY transaction_date DESC, transaction_id DESC
    """
    rows = []
    teams = {}
    categories = set()
    for r in client.query(query).result():
        row = _row_to_dict(r)
        category = _transaction_category(row.get("transaction_type"), row.get("description"))
        row["category"] = category
        rows.append(row)
        categories.add(category)
        if row.get("team_id") and row.get("team_name"):
            teams[int(row["team_id"])] = row["team_name"]
    return {
        "date": end_date,
        "days": days,
        "transactions": rows,
        "teams": [{"team_id": tid, "team_name": name} for tid, name in sorted(teams.items(), key=lambda x: x[1])],
        "categories": sorted(categories),
        "meta": {"count": len(rows), "source": "bigquery.transactions"},
    }


@functions_framework.http
def get_daily_predictions(request):
    if request.method == "OPTIONS":
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET",
            "Access-Control-Allow-Headers": "Content-Type",
        }
        return ("", 204, headers)

    headers = {
        "Access-Control-Allow-Origin": "*",
        "Content-Type": "application/json; charset=utf-8",
    }
    view = (request.args.get("view", "") or "").strip().lower()
    if view == "odds_board":
        return ({"columns": [], "error": "odds_board_disabled"}, 410, headers)
    if view == "accuracy":
        try:
            date = request.args.get("date", None) or _today_pacific()
            client = bigquery.Client()
            payload = _cached(f"accuracy:{date}", lambda: _run_accuracy_snapshot_query(client, date))
            return (payload, 200, headers)
        except Exception as exc:
            err_body = {"error": "accuracy_query_failed", "message": str(exc)}
            return (err_body, 500, headers)
    if view == "players":
        try:
            date = request.args.get("date", None)
            client = bigquery.Client()
            payload = _cached(f"players:{date or _today_pacific()}", lambda: _run_players_view(client, date))
            return (payload, 200, headers)
        except Exception as exc:
            err_body = {"error": "players_query_failed", "message": str(exc)}
            return (err_body, 500, headers)
    if view == "edges":
        try:
            date = request.args.get("date", None)
            client = bigquery.Client()
            payload = _cached(f"edges:{date or _today_pacific()}", lambda: _run_edges_view(client, date))
            return (payload, 200, headers)
        except Exception as exc:
            err_body = {"error": "edges_query_failed", "message": str(exc)}
            return (err_body, 500, headers)
    if view == "trends":
        try:
            date = request.args.get("date", None)
            client = bigquery.Client()
            payload = _cached(f"trends:{date or _today_pacific()}", lambda: _run_trends_view(client, date))
            return (payload, 200, headers)
        except Exception as exc:
            err_body = {"error": "trends_query_failed", "message": str(exc)}
            return (err_body, 500, headers)
    if view == "standings":
        try:
            date = request.args.get("date", None)
            client = bigquery.Client()
            payload = _cached(f"standings:{date or _today_pacific()}", lambda: _run_standings_view(client, date))
            return (payload, 200, headers)
        except Exception as exc:
            err_body = {"error": "standings_query_failed", "message": str(exc)}
            return (err_body, 500, headers)
    if view == "transactions":
        try:
            date = request.args.get("date", None)
            days = int(request.args.get("days", "14") or 14)
            client = bigquery.Client()
            payload = _cached(f"transactions:{date or _today_pacific()}:{days}", lambda: _run_transactions_view(client, date, days))
            return (payload, 200, headers)
        except Exception as exc:
            err_body = {"error": "transactions_query_failed", "message": str(exc)}
            return (err_body, 500, headers)

    client = bigquery.Client()
    date = request.args.get("date", None)
    def _build_games_payload():
        games = _run_daily_games_query(client, date)
        pg_dsn = (os.environ.get("PG_DSN") or "").strip()
        if pg_dsn:
            try:
                frozen = _fetch_pregame_odds_from_pg(pg_dsn, date)
                games = _apply_pregame_odds_overlay(games, frozen)
            except Exception as exc:
                print(f"pregame odds overlay failed: {exc}")
        return {"games": games}
    return (_cached(f"games:{date or _today_pacific()}", _build_games_payload), 200, headers)