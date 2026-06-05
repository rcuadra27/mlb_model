"""
Shared SQL for choosing display snapshots from inference_game_predictions.

Totals and O/U fields are anchored to the latest row with a non-null total_runs_pred
so lineup refresh runs (new v10 rows with NaN totals) cannot clobber morning totals.
"""


def totals_anchor_cte(schema: str) -> str:
    return f"""
        totals_anchor AS (
            SELECT DISTINCT ON (game_id)
                game_id,
                home_runs_pred,
                away_runs_pred,
                total_runs_pred,
                ou_line,
                ou_recommendation,
                ou_edge_over,
                ou_edge_under,
                ev_over,
                ev_under,
                is_value_ou_over,
                is_value_ou_under
            FROM {schema}.inference_game_predictions
            WHERE game_date = :d
              AND total_runs_pred IS NOT NULL
            ORDER BY game_id, as_of_ts DESC
        )"""


def best_pred_cte(schema: str) -> str:
    return f"""
        confirmed AS (
            SELECT game_id
            FROM {schema}.game_lineups
            WHERE game_date = :d
            GROUP BY game_id
            HAVING COUNT(DISTINCT is_home) = 2
        ),
        best_pred AS (
            SELECT DISTINCT ON (p.game_id) p.*
            FROM {schema}.inference_game_predictions p
            LEFT JOIN confirmed c ON c.game_id = p.game_id
            WHERE p.game_date = :d
            ORDER BY p.game_id,
                CASE
                    WHEN c.game_id IS NOT NULL
                         AND COALESCE(p.lineup_pending, FALSE) = FALSE
                    THEN 0
                    WHEN COALESCE(p.lineup_pending, FALSE) = TRUE
                    THEN 1
                    ELSE 2
                END,
                p.as_of_ts DESC
        )"""
