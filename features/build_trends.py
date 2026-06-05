#!/usr/bin/env python3
"""
Pre-compute daily trend snapshots (hot hitters, K leaders, team form, line moves).

Run once per day in the morning pipeline after build_features1.py:
    PG_DSN=... python features/build_trends.py --date 2026-05-28
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS public.daily_trends (
    id BIGSERIAL PRIMARY KEY,
    trend_date DATE NOT NULL,
    trend_type TEXT NOT NULL,
    rank INTEGER NOT NULL,
    name TEXT NOT NULL,
    meta TEXT,
    team_id INTEGER,
    team_abbr TEXT,
    team_name TEXT,
    value_primary DOUBLE PRECISION,
    value_secondary DOUBLE PRECISION,
    value_label TEXT,
    direction TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_daily_trends_date ON public.daily_trends(trend_date);
"""

ALTER_TABLE_SQL = """
ALTER TABLE public.daily_trends ADD COLUMN IF NOT EXISTS team_id INTEGER;
ALTER TABLE public.daily_trends ADD COLUMN IF NOT EXISTS team_abbr TEXT;
ALTER TABLE public.daily_trends ADD COLUMN IF NOT EXISTS team_name TEXT;
"""


HOT_HITTERS_SQL = """
WITH agg AS (
    SELECT
        sp.batter AS batter_id,
        AVG(sp.estimated_woba_using_speedangle) AS xwoba_14d,
        COUNT(DISTINCT sp.at_bat_number) AS pa
    FROM public.statcast_pitches sp
    WHERE sp.game_date >= CAST(:d AS DATE) - INTERVAL '14 days'
      AND sp.game_date < CAST(:d AS DATE)
      AND EXTRACT(YEAR FROM sp.game_date) = EXTRACT(YEAR FROM CAST(:d AS DATE))
      AND sp.woba_denom > 0
      AND sp.estimated_woba_using_speedangle IS NOT NULL
      AND sp.batter IS NOT NULL
    GROUP BY sp.batter
    HAVING COUNT(DISTINCT sp.at_bat_number) >= 30
),
names AS (
    SELECT DISTINCT ON (player_id)
        player_id,
        player_name,
        team_id
    FROM public.game_lineups
    WHERE EXTRACT(YEAR FROM game_date) = EXTRACT(YEAR FROM CAST(:d AS DATE))
    ORDER BY player_id, game_date DESC
)
SELECT
    COALESCE(n.player_name, 'Batter ' || agg.batter_id::text) AS name,
    COALESCE(t.team_name, '—') AS meta,
    ROUND(agg.xwoba_14d::numeric, 3)::float AS value_primary,
    agg.pa::float AS value_secondary,
    'xwOBA' AS value_label,
    'neutral' AS direction
FROM agg
LEFT JOIN names n ON n.player_id = agg.batter_id
LEFT JOIN public.teams t ON t.mlb_team_id = n.team_id
ORDER BY agg.xwoba_14d DESC
LIMIT 8
"""


K_LEADERS_SQL = """
WITH season_starters AS (
    SELECT ps.pitcher_id
    FROM public.pitcher_starts ps
    JOIN public.games g ON g.game_id = ps.game_id
    WHERE EXTRACT(YEAR FROM ps.game_date) = EXTRACT(YEAR FROM CAST(:d AS DATE))
      AND g.home_runs IS NOT NULL
      AND g.away_runs IS NOT NULL
    GROUP BY ps.pitcher_id
    HAVING COUNT(*) >= 3
),
ranked_starts AS (
    SELECT
        ps.pitcher_id,
        ps.game_id,
        ps.game_date,
        ps.team_id,
        ROW_NUMBER() OVER (
            PARTITION BY ps.pitcher_id
            ORDER BY ps.game_date DESC
        ) AS start_num
    FROM public.pitcher_starts ps
    JOIN season_starters ss ON ss.pitcher_id = ps.pitcher_id
    JOIN public.games g ON g.game_id = ps.game_id
    WHERE EXTRACT(YEAR FROM ps.game_date) = EXTRACT(YEAR FROM CAST(:d AS DATE))
      AND g.home_runs IS NOT NULL
      AND g.away_runs IS NOT NULL
),
rolled AS (
    SELECT
        rs.pitcher_id,
        SUM(CASE WHEN sp.events = 'strikeout' THEN 1 ELSE 0 END) AS total_k,
        COUNT(DISTINCT rs.game_id) AS starts,
        ROUND(
            SUM(CASE WHEN sp.events = 'strikeout' THEN 1 ELSE 0 END)::numeric
            / COUNT(DISTINCT rs.game_id), 1
        ) AS k_per_start
    FROM ranked_starts rs
    JOIN public.statcast_pitches sp
        ON sp.game_pk = rs.game_id
       AND sp.pitcher = rs.pitcher_id
    WHERE rs.start_num <= 3
      AND sp.events IS NOT NULL
    GROUP BY rs.pitcher_id
),
latest_team AS (
    SELECT pitcher_id, team_id
    FROM ranked_starts
    WHERE start_num = 1
),
names AS (
    SELECT DISTINCT ON (pitcher_id)
        pitcher_id,
        pitcher_name
    FROM (
        SELECT gsp.home_sp_id AS pitcher_id, gsp.home_sp_name AS pitcher_name, gsp.updated_at
        FROM public.game_starting_pitchers gsp
        WHERE gsp.home_sp_id IS NOT NULL AND gsp.home_sp_name IS NOT NULL
        UNION ALL
        SELECT gsp.away_sp_id, gsp.away_sp_name, gsp.updated_at
        FROM public.game_starting_pitchers gsp
        WHERE gsp.away_sp_id IS NOT NULL AND gsp.away_sp_name IS NOT NULL
        UNION ALL
        SELECT ppp.pitcher_id, ppp.pitcher_name, ppp.as_of_ts AS updated_at
        FROM public.pitcher_prop_predictions ppp
        WHERE ppp.pitcher_name IS NOT NULL
    ) src
    ORDER BY pitcher_id, updated_at DESC NULLS LAST
)
SELECT
    COALESCE(n.pitcher_name, 'SP ' || r.pitcher_id::text) AS name,
    COALESCE(t.team_name, '—') AS meta,
    r.total_k::float AS value_primary,
    r.k_per_start::float AS value_secondary,
    (r.starts::text || ' starts') AS value_label,
    'neutral' AS direction
FROM rolled r
LEFT JOIN latest_team lt ON lt.pitcher_id = r.pitcher_id
LEFT JOIN names n ON n.pitcher_id = r.pitcher_id
LEFT JOIN public.teams t ON t.mlb_team_id = lt.team_id
WHERE r.total_k > 0
ORDER BY r.total_k DESC, r.k_per_start DESC
LIMIT 8
"""


COLD_PITCHERS_SQL = """
WITH eligible AS (
    SELECT ps.pitcher_id
    FROM public.pitcher_starts ps
    JOIN public.games g ON g.game_id = ps.game_id
    WHERE EXTRACT(YEAR FROM ps.game_date) = 2026
      AND ps.game_date >= CAST(:d AS DATE) - INTERVAL '14 days'
      AND ps.game_date < CAST(:d AS DATE)
      AND g.home_runs IS NOT NULL
      AND g.away_runs IS NOT NULL
      AND ps.innings_pitched IS NOT NULL
      AND ps.innings_pitched > 0
      AND ps.earned_runs IS NOT NULL
    GROUP BY ps.pitcher_id
    HAVING COUNT(*) >= 3
),
ranked AS (
    SELECT
        ps.*,
        ROW_NUMBER() OVER (PARTITION BY ps.pitcher_id ORDER BY ps.game_date DESC) AS start_num
    FROM public.pitcher_starts ps
    JOIN eligible e ON e.pitcher_id = ps.pitcher_id
    JOIN public.games g ON g.game_id = ps.game_id
    WHERE EXTRACT(YEAR FROM ps.game_date) = 2026
      AND ps.game_date >= CAST(:d AS DATE) - INTERVAL '14 days'
      AND ps.game_date < CAST(:d AS DATE)
      AND g.home_runs IS NOT NULL
      AND g.away_runs IS NOT NULL
      AND ps.innings_pitched IS NOT NULL
      AND ps.innings_pitched > 0
      AND ps.earned_runs IS NOT NULL
),
rolled AS (
    SELECT
        pitcher_id,
        COUNT(*) AS starts,
        SUM(COALESCE(runs_allowed, 0)) AS runs_allowed,
        SUM(COALESCE(earned_runs, 0)) AS earned_runs,
        SUM(innings_pitched) AS innings_pitched,
        ROUND((SUM(COALESCE(earned_runs, 0)) * 9.0 / NULLIF(SUM(innings_pitched), 0))::numeric, 2) AS era_last3
    FROM ranked
    WHERE start_num <= 3
    GROUP BY pitcher_id
    HAVING COUNT(*) >= 3 AND SUM(innings_pitched) > 0
),
latest_team AS (
    SELECT pitcher_id, team_id
    FROM ranked
    WHERE start_num = 1
),
names AS (
    SELECT DISTINCT ON (pitcher_id)
        pitcher_id,
        pitcher_name
    FROM (
        SELECT gsp.home_sp_id AS pitcher_id, gsp.home_sp_name AS pitcher_name, gsp.updated_at
        FROM public.game_starting_pitchers gsp
        WHERE gsp.home_sp_id IS NOT NULL AND gsp.home_sp_name IS NOT NULL
        UNION ALL
        SELECT gsp.away_sp_id, gsp.away_sp_name, gsp.updated_at
        FROM public.game_starting_pitchers gsp
        WHERE gsp.away_sp_id IS NOT NULL AND gsp.away_sp_name IS NOT NULL
        UNION ALL
        SELECT ppp.pitcher_id, ppp.pitcher_name, ppp.as_of_ts AS updated_at
        FROM public.pitcher_prop_predictions ppp
        WHERE ppp.pitcher_name IS NOT NULL
    ) src
    ORDER BY pitcher_id, updated_at DESC NULLS LAST
)
SELECT
    COALESCE(n.pitcher_name, 'SP ' || r.pitcher_id::text) AS name,
    COALESCE(t.team_name, '—') AS meta,
    r.era_last3::float AS value_primary,
    r.earned_runs::float AS value_secondary,
    (r.earned_runs::text || ' ER in ' || ROUND(r.innings_pitched::numeric, 1)::text || ' IP') AS value_label,
    'down' AS direction
FROM rolled r
LEFT JOIN latest_team lt ON lt.pitcher_id = r.pitcher_id
LEFT JOIN names n ON n.pitcher_id = r.pitcher_id
LEFT JOIN public.teams t ON t.mlb_team_id = lt.team_id
ORDER BY r.era_last3 DESC, r.earned_runs DESC
LIMIT 8
"""


BEST_ERA_LAST3_SQL = """
WITH eligible AS (
    SELECT ps.pitcher_id
    FROM public.pitcher_starts ps
    JOIN public.games g ON g.game_id = ps.game_id
    WHERE EXTRACT(YEAR FROM ps.game_date) = 2026
      AND ps.game_date >= CAST(:d AS DATE) - INTERVAL '14 days'
      AND ps.game_date < CAST(:d AS DATE)
      AND g.home_runs IS NOT NULL
      AND g.away_runs IS NOT NULL
      AND ps.innings_pitched IS NOT NULL
      AND ps.innings_pitched > 0
      AND ps.earned_runs IS NOT NULL
    GROUP BY ps.pitcher_id
    HAVING COUNT(*) >= 3
),
ranked AS (
    SELECT
        ps.*,
        ROW_NUMBER() OVER (PARTITION BY ps.pitcher_id ORDER BY ps.game_date DESC) AS start_num
    FROM public.pitcher_starts ps
    JOIN eligible e ON e.pitcher_id = ps.pitcher_id
    JOIN public.games g ON g.game_id = ps.game_id
    WHERE EXTRACT(YEAR FROM ps.game_date) = 2026
      AND ps.game_date >= CAST(:d AS DATE) - INTERVAL '14 days'
      AND ps.game_date < CAST(:d AS DATE)
      AND g.home_runs IS NOT NULL
      AND g.away_runs IS NOT NULL
      AND ps.innings_pitched IS NOT NULL
      AND ps.innings_pitched > 0
      AND ps.earned_runs IS NOT NULL
),
rolled AS (
    SELECT
        pitcher_id,
        COUNT(*) AS starts,
        SUM(COALESCE(runs_allowed, 0)) AS runs_allowed,
        SUM(COALESCE(earned_runs, 0)) AS earned_runs,
        SUM(innings_pitched) AS innings_pitched,
        ROUND((SUM(COALESCE(earned_runs, 0)) * 9.0 / NULLIF(SUM(innings_pitched), 0))::numeric, 2) AS era_last3
    FROM ranked
    WHERE start_num <= 3
    GROUP BY pitcher_id
    HAVING COUNT(*) >= 3 AND SUM(innings_pitched) > 0
),
latest_team AS (
    SELECT pitcher_id, team_id
    FROM ranked
    WHERE start_num = 1
),
names AS (
    SELECT DISTINCT ON (pitcher_id)
        pitcher_id,
        pitcher_name
    FROM (
        SELECT gsp.home_sp_id AS pitcher_id, gsp.home_sp_name AS pitcher_name, gsp.updated_at
        FROM public.game_starting_pitchers gsp
        WHERE gsp.home_sp_id IS NOT NULL AND gsp.home_sp_name IS NOT NULL
        UNION ALL
        SELECT gsp.away_sp_id, gsp.away_sp_name, gsp.updated_at
        FROM public.game_starting_pitchers gsp
        WHERE gsp.away_sp_id IS NOT NULL AND gsp.away_sp_name IS NOT NULL
        UNION ALL
        SELECT ppp.pitcher_id, ppp.pitcher_name, ppp.as_of_ts AS updated_at
        FROM public.pitcher_prop_predictions ppp
        WHERE ppp.pitcher_name IS NOT NULL
    ) src
    ORDER BY pitcher_id, updated_at DESC NULLS LAST
)
SELECT
    COALESCE(n.pitcher_name, 'SP ' || r.pitcher_id::text) AS name,
    COALESCE(t.team_name, '—') AS meta,
    r.era_last3::float AS value_primary,
    r.earned_runs::float AS value_secondary,
    (r.starts::text || ' starts') AS value_label,
    'up' AS direction
FROM rolled r
LEFT JOIN latest_team lt ON lt.pitcher_id = r.pitcher_id
LEFT JOIN names n ON n.pitcher_id = r.pitcher_id
LEFT JOIN public.teams t ON t.mlb_team_id = lt.team_id
ORDER BY r.era_last3 ASC, r.innings_pitched DESC
LIMIT 8
"""


BATTER_GAME_BASE_SQL = """
WITH team_games AS (
    SELECT team_id, game_id, game_date
    FROM (
        SELECT g.home_team_id AS team_id, g.game_id, g.game_date
        FROM public.games g
        WHERE EXTRACT(YEAR FROM g.game_date) = 2026
          AND g.game_date < CAST(:d AS DATE)
          AND g.home_runs IS NOT NULL
          AND g.away_runs IS NOT NULL
        UNION ALL
        SELECT g.away_team_id, g.game_id, g.game_date
        FROM public.games g
        WHERE EXTRACT(YEAR FROM g.game_date) = 2026
          AND g.game_date < CAST(:d AS DATE)
          AND g.home_runs IS NOT NULL
          AND g.away_runs IS NOT NULL
    ) x
),
last10_team_games AS (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY team_id ORDER BY game_date DESC, game_id DESC) AS rn
    FROM team_games
),
names AS (
    SELECT DISTINCT ON (player_id)
        player_id,
        player_name,
        team_id
    FROM public.game_lineups
    WHERE EXTRACT(YEAR FROM game_date) = 2026
    ORDER BY player_id, game_date DESC
),
batter_games AS (
    SELECT
        sp.batter AS batter_id,
        sp.game_pk AS game_id,
        sp.game_date,
        n.team_id,
        SUM(CASE WHEN sp.events IN ('single','double','triple','home_run') THEN 1 ELSE 0 END) AS hits,
        SUM(CASE WHEN sp.events = 'home_run' THEN 1 ELSE 0 END) AS hrs,
        SUM(CASE WHEN sp.events = 'strikeout' THEN 1 ELSE 0 END) AS ks,
        COUNT(DISTINCT sp.at_bat_number) AS pa
    FROM public.statcast_pitches sp
    JOIN names n ON n.player_id = sp.batter
    WHERE EXTRACT(YEAR FROM sp.game_date) = 2026
      AND sp.game_date < CAST(:d AS DATE)
      AND sp.batter IS NOT NULL
      AND sp.events IS NOT NULL
    GROUP BY sp.batter, sp.game_pk, sp.game_date, n.team_id
)
"""


MOST_HR_LAST10_SQL = BATTER_GAME_BASE_SQL + """
SELECT
    COALESCE(n.player_name, 'Batter ' || bg.batter_id::text) AS name,
    COALESCE(t.team_name, '—') AS meta,
    SUM(bg.hrs)::float AS value_primary,
    SUM(bg.pa)::float AS value_secondary,
    'HR last 10' AS value_label,
    'up' AS direction
FROM batter_games bg
JOIN last10_team_games ltg
  ON ltg.team_id = bg.team_id AND ltg.game_id = bg.game_id AND ltg.rn <= 10
LEFT JOIN names n ON n.player_id = bg.batter_id
LEFT JOIN public.teams t ON t.mlb_team_id = bg.team_id
GROUP BY bg.batter_id, n.player_name, t.team_name
HAVING SUM(bg.hrs) > 0
ORDER BY SUM(bg.hrs) DESC, SUM(bg.hits) DESC, SUM(bg.pa) ASC
LIMIT 8
"""


MOST_HITS_LAST10_SQL = BATTER_GAME_BASE_SQL + """
SELECT
    COALESCE(n.player_name, 'Batter ' || bg.batter_id::text) AS name,
    COALESCE(t.team_name, '—') AS meta,
    SUM(bg.hits)::float AS value_primary,
    SUM(bg.pa)::float AS value_secondary,
    'H last 10' AS value_label,
    'up' AS direction
FROM batter_games bg
JOIN last10_team_games ltg
  ON ltg.team_id = bg.team_id AND ltg.game_id = bg.game_id AND ltg.rn <= 10
LEFT JOIN names n ON n.player_id = bg.batter_id
LEFT JOIN public.teams t ON t.mlb_team_id = bg.team_id
GROUP BY bg.batter_id, n.player_name, t.team_name
HAVING SUM(bg.hits) > 0
ORDER BY SUM(bg.hits) DESC, SUM(bg.hrs) DESC, SUM(bg.pa) ASC
LIMIT 8
"""


COLD_BATS_LAST10_SQL = BATTER_GAME_BASE_SQL + """
SELECT
    COALESCE(n.player_name, 'Batter ' || bg.batter_id::text) AS name,
    COALESCE(t.team_name, '—') AS meta,
    SUM(bg.ks)::float AS value_primary,
    SUM(bg.hits)::float AS value_secondary,
    (SUM(bg.hits)::text || ' H') AS value_label,
    'down' AS direction
FROM batter_games bg
JOIN last10_team_games ltg
  ON ltg.team_id = bg.team_id AND ltg.game_id = bg.game_id AND ltg.rn <= 10
LEFT JOIN names n ON n.player_id = bg.batter_id
LEFT JOIN public.teams t ON t.mlb_team_id = bg.team_id
GROUP BY bg.batter_id, n.player_name, t.team_name
HAVING SUM(bg.ks) > 0
ORDER BY SUM(bg.ks) DESC, SUM(bg.hits) ASC, SUM(bg.pa) DESC
LIMIT 8
"""


HITTING_STREAKS_SQL = """
WITH names AS (
    SELECT DISTINCT ON (player_id)
        player_id,
        player_name,
        team_id
    FROM public.game_lineups
    WHERE EXTRACT(YEAR FROM game_date) = 2026
    ORDER BY player_id, game_date DESC
),
batter_games AS (
    SELECT
        sp.batter AS batter_id,
        sp.game_pk AS game_id,
        sp.game_date,
        n.team_id,
        SUM(CASE WHEN sp.events IN ('single','double','triple','home_run') THEN 1 ELSE 0 END) AS hits
    FROM public.statcast_pitches sp
    JOIN names n ON n.player_id = sp.batter
    WHERE EXTRACT(YEAR FROM sp.game_date) = 2026
      AND sp.game_date < CAST(:d AS DATE)
      AND sp.batter IS NOT NULL
      AND sp.events IS NOT NULL
    GROUP BY sp.batter, sp.game_pk, sp.game_date, n.team_id
),
ordered AS (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY batter_id ORDER BY game_date DESC, game_id DESC) AS rn
    FROM batter_games
),
first_miss AS (
    SELECT batter_id, MIN(rn) AS miss_rn
    FROM ordered
    WHERE hits = 0
    GROUP BY batter_id
),
active AS (
    SELECT
        o.batter_id,
        MAX(o.team_id) AS team_id,
        COUNT(*) AS streak_len
    FROM ordered o
    LEFT JOIN first_miss fm ON fm.batter_id = o.batter_id
    WHERE o.hits > 0
      AND (fm.miss_rn IS NULL OR o.rn < fm.miss_rn)
    GROUP BY o.batter_id
    HAVING COUNT(*) >= 3
)
SELECT
    COALESCE(n.player_name, 'Batter ' || a.batter_id::text) AS name,
    COALESCE(t.team_name, '—') AS meta,
    a.streak_len::float AS value_primary,
    NULL::float AS value_secondary,
    'games' AS value_label,
    'up' AS direction
FROM active a
LEFT JOIN names n ON n.player_id = a.batter_id
LEFT JOIN public.teams t ON t.mlb_team_id = a.team_id
ORDER BY a.streak_len DESC, n.player_name
LIMIT 8
"""


BEST_BULLPENS_LAST7_SQL = """
WITH bullpen_appearances AS (
    SELECT
        pa.team_id,
        pa.game_id,
        pa.innings_pitched,
        pa.earned_runs
    FROM public.pitcher_appearances pa
    JOIN public.games g ON g.game_id = pa.game_id
    WHERE EXTRACT(YEAR FROM pa.game_date) = 2026
      AND pa.game_date >= CAST(:d AS DATE) - INTERVAL '7 days'
      AND pa.game_date < CAST(:d AS DATE)
      AND g.home_runs IS NOT NULL
      AND g.away_runs IS NOT NULL
      AND COALESCE(pa.is_starter, FALSE) = FALSE
      AND pa.innings_pitched IS NOT NULL
      AND pa.innings_pitched > 0
      AND pa.earned_runs IS NOT NULL
),
agg AS (
    SELECT
        team_id,
        COUNT(DISTINCT game_id) AS games,
        SUM(COALESCE(earned_runs, 0)) AS earned_runs,
        SUM(innings_pitched) AS bp_ip
    FROM bullpen_appearances
    GROUP BY team_id
    HAVING SUM(innings_pitched) >= 8
)
SELECT
    t.team_name AS name,
    (a.games::text || ' games') AS meta,
    ROUND((a.earned_runs * 9.0 / NULLIF(a.bp_ip, 0))::numeric, 2)::float AS value_primary,
    a.bp_ip::float AS value_secondary,
    'ERA last 7d' AS value_label,
    'up' AS direction
FROM agg a
JOIN public.teams t ON t.mlb_team_id = a.team_id
ORDER BY value_primary ASC, a.bp_ip DESC
LIMIT 8
"""


TEAM_FORM_SQL = """
WITH team_games AS (
    SELECT
        g.game_date,
        g.home_team_id AS team_id,
        g.home_runs AS runs_for,
        g.away_runs AS runs_against,
        CASE WHEN g.home_runs > g.away_runs THEN 1 ELSE 0 END AS win
    FROM public.games g
    WHERE g.home_runs IS NOT NULL AND g.away_runs IS NOT NULL
      AND g.game_date < CAST(:d AS DATE)
    UNION ALL
    SELECT
        g.game_date,
        g.away_team_id,
        g.away_runs,
        g.home_runs,
        CASE WHEN g.away_runs > g.home_runs THEN 1 ELSE 0 END
    FROM public.games g
    WHERE g.home_runs IS NOT NULL AND g.away_runs IS NOT NULL
      AND g.game_date < CAST(:d AS DATE)
),
last10 AS (
    SELECT
        team_id,
        game_date,
        win,
        runs_for,
        runs_against,
        ROW_NUMBER() OVER (PARTITION BY team_id ORDER BY game_date DESC) AS rn
    FROM team_games
),
agg AS (
    SELECT
        team_id,
        SUM(win) AS wins,
        COUNT(*) - SUM(win) AS losses,
        SUM(runs_for - runs_against) AS run_diff
    FROM last10
    WHERE rn <= 10
    GROUP BY team_id
    HAVING COUNT(*) >= 5
),
streak AS (
    SELECT
        tg.team_id,
        CASE
            WHEN MAX(CASE WHEN tg.rn = 1 THEN tg.win END) = 0 THEN 0
            ELSE COALESCE(MIN(CASE WHEN tg.win = 0 THEN tg.rn END) - 1, COUNT(*) FILTER (WHERE tg.win = 1))
        END AS win_streak
    FROM (
        SELECT team_id, win,
               ROW_NUMBER() OVER (PARTITION BY team_id ORDER BY game_date DESC) AS rn
        FROM team_games
    ) tg
    GROUP BY tg.team_id
)
SELECT
    t.team_name AS name,
    agg.wins::text || '-' || agg.losses::text AS value_label,
    COALESCE(st.win_streak, 0)::float AS value_primary,
    agg.run_diff::float AS value_secondary,
    CASE
        WHEN agg.run_diff > 0 THEN 'up'
        WHEN agg.run_diff < 0 THEN 'down'
        ELSE 'neutral'
    END AS direction,
    agg.wins::text || '-' || agg.losses::text || ' · '
        || CASE WHEN agg.run_diff > 0 THEN '+' ELSE '' END || agg.run_diff::text || ' run diff' AS meta
FROM agg
JOIN public.teams t ON t.mlb_team_id = agg.team_id
LEFT JOIN streak st ON st.team_id = agg.team_id
ORDER BY agg.wins DESC, agg.run_diff DESC
LIMIT 8
"""


LINE_MOVES_SQL = """
SELECT
    g.away_team_name || ' @ ' || g.home_team_name AS name,
    fg.total_line_move,
    fg.home_line_move,
    fg.morning_ou_line,
    fg.closing_ou_line,
    fg.morning_home_price,
    fg.closing_home_price,
    g.home_team_name
FROM public.features_game fg
JOIN public.games g ON g.game_id = fg.game_id
WHERE g.game_date = CAST(:d AS DATE)
ORDER BY ABS(COALESCE(fg.home_line_move, 0)) + ABS(COALESCE(fg.total_line_move, 0)) DESC
LIMIT 12
"""


def _line_move_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        tlm = r.get("total_line_move")
        hlm = r.get("home_line_move")
        tlm_f = float(tlm) if pd.notna(tlm) else 0.0
        hlm_f = float(hlm) if pd.notna(hlm) else 0.0
        combined = abs(tlm_f) + abs(hlm_f)
        if combined < 0.01:
            continue

        desc_parts = []
        direction = "neutral"
        magnitude = combined

        if abs(tlm_f) >= 0.3:
            old_l = r.get("morning_ou_line")
            new_l = r.get("closing_ou_line")
            if pd.notna(old_l) and pd.notna(new_l):
                desc_parts.append(f"Total {float(old_l):.1f} → {float(new_l):.1f}")
            else:
                desc_parts.append(f"Total move {tlm_f:+.1f}")
            if abs(tlm_f) >= abs(hlm_f):
                direction = "up" if tlm_f > 0 else "down"
                magnitude = abs(tlm_f)

        if abs(hlm_f) >= 5:
            old_p = r.get("morning_home_price")
            new_p = r.get("closing_home_price")
            home = r.get("home_team_name") or "Home"
            abbr = str(home).split()[-1][:3].upper()
            if pd.notna(old_p) and pd.notna(new_p):
                desc_parts.append(f"{abbr} {int(old_p):+d} → {int(new_p):+d}")
            else:
                desc_parts.append(f"Home ML move {hlm_f:+.0f}")
            if abs(hlm_f) > abs(tlm_f):
                direction = "up" if hlm_f > 0 else "down"
                magnitude = abs(hlm_f)

        rows.append({
            "name": r["name"],
            "meta": " · ".join(desc_parts) if desc_parts else "Line movement",
            "value_primary": float(magnitude),
            "value_secondary": combined,
            "value_label": "move",
            "direction": direction,
        })
    return rows[:6]


def _rows_from_df(df: pd.DataFrame, trend_type: str, trend_date: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, row in enumerate(df.to_dict(orient="records"), start=1):
        out.append({
            "trend_date": trend_date,
            "trend_type": trend_type,
            "rank": i,
            "name": row.get("name"),
            "meta": row.get("meta"),
            "value_primary": row.get("value_primary"),
            "value_secondary": row.get("value_secondary"),
            "value_label": row.get("value_label"),
            "direction": row.get("direction") or "neutral",
        })
    return out


PLAYER_TREND_TYPES = {
    "hot_hitters",
    "most_hr_last10",
    "most_hits_last10",
    "hitting_streaks",
    "cold_bats_last10",
    "k_leaders",
    "best_era_last3",
    "cold_pitchers",
}

TEAM_TREND_TYPES = {"best_bullpens_last7", "team_form"}


def _add_team_identity(df: pd.DataFrame, engine, schema: str) -> pd.DataFrame:
    if df.empty:
        return df

    teams = pd.read_sql(text(f"""
        SELECT mlb_team_id AS team_id, team_name, abbreviation AS team_abbr
        FROM {schema}.teams
    """), engine)
    by_name = {
        str(row["team_name"]).strip().lower(): row
        for row in teams.to_dict(orient="records")
        if row.get("team_name")
    }
    out = df.copy()
    for col in ("team_id", "team_abbr", "team_name"):
        if col not in out.columns:
            out[col] = None

    def source_team_name(row) -> str | None:
        trend_type = row.get("trend_type")
        if trend_type in PLAYER_TREND_TYPES:
            return row.get("meta")
        if trend_type in TEAM_TREND_TYPES:
            return row.get("name")
        return None

    for idx, row in out.iterrows():
        raw = source_team_name(row)
        if not raw:
            continue
        match = by_name.get(str(raw).strip().lower())
        if not match:
            continue
        out.at[idx, "team_id"] = int(match["team_id"]) if match.get("team_id") is not None else None
        out.at[idx, "team_abbr"] = match.get("team_abbr")
        out.at[idx, "team_name"] = match.get("team_name")
    return out


def build_trends(date_str: str, schema: str, engine) -> pd.DataFrame:
    params = {"d": date_str}
    sections: list[dict[str, Any]] = []

    hot = pd.read_sql(text(HOT_HITTERS_SQL), engine, params=params)
    sections.extend(_rows_from_df(hot, "hot_hitters", date_str))

    most_hr = pd.read_sql(text(MOST_HR_LAST10_SQL), engine, params=params)
    sections.extend(_rows_from_df(most_hr, "most_hr_last10", date_str))

    most_hits = pd.read_sql(text(MOST_HITS_LAST10_SQL), engine, params=params)
    sections.extend(_rows_from_df(most_hits, "most_hits_last10", date_str))

    streaks = pd.read_sql(text(HITTING_STREAKS_SQL), engine, params=params)
    sections.extend(_rows_from_df(streaks, "hitting_streaks", date_str))

    cold_bats = pd.read_sql(text(COLD_BATS_LAST10_SQL), engine, params=params)
    sections.extend(_rows_from_df(cold_bats, "cold_bats_last10", date_str))

    k_leaders = pd.read_sql(text(K_LEADERS_SQL), engine, params=params)
    sections.extend(_rows_from_df(k_leaders, "k_leaders", date_str))

    best_era = pd.read_sql(text(BEST_ERA_LAST3_SQL), engine, params=params)
    sections.extend(_rows_from_df(best_era, "best_era_last3", date_str))

    cold = pd.read_sql(text(COLD_PITCHERS_SQL), engine, params=params)
    sections.extend(_rows_from_df(cold, "cold_pitchers", date_str))

    bullpens = pd.read_sql(text(BEST_BULLPENS_LAST7_SQL), engine, params=params)
    sections.extend(_rows_from_df(bullpens, "best_bullpens_last7", date_str))

    teams = pd.read_sql(text(TEAM_FORM_SQL), engine, params=params)
    sections.extend(_rows_from_df(teams, "team_form", date_str))

    line_df = pd.read_sql(text(LINE_MOVES_SQL), engine, params=params)
    line_rows = _line_move_rows(line_df)
    for i, row in enumerate(line_rows, start=1):
        sections.append({
            "trend_date": date_str,
            "trend_type": "line_moves",
            "rank": i,
            **row,
        })

    return _add_team_identity(pd.DataFrame(sections), engine, schema)


def upsert_trends(date_str: str, schema: str, engine, df: pd.DataFrame) -> None:
    with engine.begin() as conn:
        conn.execute(text(CREATE_TABLE_SQL))
        conn.execute(text(ALTER_TABLE_SQL))
        conn.execute(
            text(f"DELETE FROM {schema}.daily_trends WHERE trend_date = :d"),
            {"d": date_str},
        )
        if df.empty:
            print(f"  No trend rows for {date_str}")
            return
        df.to_sql(
            "daily_trends",
            conn,
            schema=schema,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=100,
        )
    print(f"  Wrote {len(df)} trend rows for {date_str}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    ap.add_argument("--schema", default="public")
    args = ap.parse_args()

    pg_dsn = os.getenv("PG_DSN")
    if not pg_dsn:
        raise RuntimeError("PG_DSN env var required.")

    engine = create_engine(pg_dsn, pool_pre_ping=True)
    print(f"Building daily trends for {args.date}...")
    df = build_trends(args.date, args.schema, engine)
    upsert_trends(args.date, args.schema, engine, df)
    for t in (
        "hot_hitters",
        "most_hr_last10",
        "most_hits_last10",
        "hitting_streaks",
        "cold_bats_last10",
        "k_leaders",
        "best_era_last3",
        "cold_pitchers",
        "best_bullpens_last7",
        "team_form",
        "line_moves",
    ):
        n = len(df[df["trend_type"] == t]) if not df.empty else 0
        print(f"    {t}: {n}")


if __name__ == "__main__":
    main()
