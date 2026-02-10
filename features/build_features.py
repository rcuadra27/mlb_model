import os
import psycopg2
from psycopg2.extras import execute_values


UPSERT_SQL = """
INSERT INTO features_game (
  game_id, game_date, season, home_team_id, away_team_id,
  home_win_pct_30, away_win_pct_30,
  home_runs_for_30, home_runs_against_30,
  away_runs_for_30, away_runs_against_30,
  home_sp_ra9_last5, away_sp_ra9_last5,
  home_sp_ip_per_start_last5, away_sp_ip_per_start_last5,
  home_win
)
VALUES %s
ON CONFLICT (game_id) DO UPDATE SET
  game_date = EXCLUDED.game_date,
  season = EXCLUDED.season,
  home_team_id = EXCLUDED.home_team_id,
  away_team_id = EXCLUDED.away_team_id,
  home_win_pct_30 = EXCLUDED.home_win_pct_30,
  away_win_pct_30 = EXCLUDED.away_win_pct_30,
  home_runs_for_30 = EXCLUDED.home_runs_for_30,
  home_runs_against_30 = EXCLUDED.home_runs_against_30,
  away_runs_for_30 = EXCLUDED.away_runs_for_30,
  away_runs_against_30 = EXCLUDED.away_runs_against_30,
  -- starter fields remain NULL for team-only step
  home_win = EXCLUDED.home_win,
  updated_at = now();
"""


def main():
    dsn = os.environ["PG_DSN"]
    conn = psycopg2.connect(dsn)

    # Pull all games in chronological order
    with conn.cursor() as cur:
        cur.execute("""
            SELECT game_id, game_date, season, home_team_id, away_team_id, home_runs, away_runs, home_win
            FROM games
            WHERE home_runs IS NOT NULL AND away_runs IS NOT NULL
            ORDER BY game_date, game_id;
        """)
        games = cur.fetchall()

    # We'll compute rolling features using SQL window queries per team for correctness and speed.
    # Easiest: materialize a "team_games" view in SQL and compute windowed rolling stats there,
    # then join back to each game.
    #
    # We'll do it in one SQL query and upsert results.

    q = """
    WITH team_games AS (
      SELECT
        game_id,
        game_date,
        season,
        home_team_id AS team_id,
        1 AS is_home,
        home_runs AS runs_for,
        away_runs AS runs_against,
        CASE WHEN home_runs > away_runs THEN 1 ELSE 0 END AS win
      FROM games
      WHERE home_runs IS NOT NULL AND away_runs IS NOT NULL

      UNION ALL

      SELECT
        game_id,
        game_date,
        season,
        away_team_id AS team_id,
        0 AS is_home,
        away_runs AS runs_for,
        home_runs AS runs_against,
        CASE WHEN away_runs > home_runs THEN 1 ELSE 0 END AS win
      FROM games
      WHERE home_runs IS NOT NULL AND away_runs IS NOT NULL
    ),
    rolling AS (
      SELECT
        game_id,
        team_id,
        game_date,
        season,
        AVG(win) OVER (
          PARTITION BY team_id
          ORDER BY game_date, game_id
          ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
        ) AS win_pct_30,
        AVG(runs_for) OVER (
          PARTITION BY team_id
          ORDER BY game_date, game_id
          ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
        ) AS runs_for_30,
        AVG(runs_against) OVER (
          PARTITION BY team_id
          ORDER BY game_date, game_id
          ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
        ) AS runs_against_30
      FROM team_games
    )
    SELECT
      g.game_id,
      g.game_date,
      g.season,
      g.home_team_id,
      g.away_team_id,

      rh.win_pct_30 AS home_win_pct_30,
      ra.win_pct_30 AS away_win_pct_30,
      rh.runs_for_30 AS home_runs_for_30,
      rh.runs_against_30 AS home_runs_against_30,
      ra.runs_for_30 AS away_runs_for_30,
      ra.runs_against_30 AS away_runs_against_30,

      NULL::REAL AS home_sp_ra9_last5,
      NULL::REAL AS away_sp_ra9_last5,
      NULL::REAL AS home_sp_ip_per_start_last5,
      NULL::REAL AS away_sp_ip_per_start_last5,

      g.home_win
    FROM games g
    LEFT JOIN rolling rh ON rh.game_id = g.game_id AND rh.team_id = g.home_team_id
    LEFT JOIN rolling ra ON ra.game_id = g.game_id AND ra.team_id = g.away_team_id
    WHERE g.home_runs IS NOT NULL AND g.away_runs IS NOT NULL
    ORDER BY g.game_date, g.game_id;
    """

    with conn.cursor() as cur:
        cur.execute(q)
        rows = cur.fetchall()

    cols = [
        "game_id", "game_date", "season", "home_team_id", "away_team_id",
        "home_win_pct_30", "away_win_pct_30",
        "home_runs_for_30", "home_runs_against_30",
        "away_runs_for_30", "away_runs_against_30",
        "home_sp_ra9_last5", "away_sp_ra9_last5",
        "home_sp_ip_per_start_last5", "away_sp_ip_per_start_last5",
        "home_win"
    ]

    # Batch upserts
    batch = []
    for r in rows:
        batch.append(list(r))
        if len(batch) >= 2000:
            with conn.cursor() as cur:
                execute_values(cur, UPSERT_SQL, batch, page_size=2000)
            conn.commit()
            batch.clear()

    if batch:
        with conn.cursor() as cur:
            execute_values(cur, UPSERT_SQL, batch, page_size=2000)
        conn.commit()

    conn.close()
    print(f"Built features for {len(rows)} games into features_game")


if __name__ == "__main__":
    main()
