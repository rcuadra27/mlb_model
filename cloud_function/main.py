import functions_framework
from google.cloud import bigquery

@functions_framework.http
def get_daily_predictions(request):
    if request.method == "OPTIONS":
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET",
            "Access-Control-Allow-Headers": "Content-Type",
        }
        return ("", 204, headers)

    headers = {"Access-Control-Allow-Origin": "*"}
    client = bigquery.Client()

    date = request.args.get("date", None)
    date_filter = f"AND game_date = '{date}'" if date else ""

    query = f"""
        WITH latest AS (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY game_id
                    ORDER BY as_of_ts DESC
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
            CAST(away_runs AS INT64)                               AS away_runs,
            CAST(home_runs AS INT64)                               AS home_runs,
            status
        FROM latest
        WHERE rn = 1
        ORDER BY first_pitch_utc ASC NULLS LAST
    """

    results = client.query(query).result()
    games = []
    for row in results:
        g = dict(row)
        for k, v in g.items():
            if hasattr(v, "isoformat"):
                g[k] = v.isoformat()
            elif v is None:
                g[k] = None
        games.append(g)

    return ({"games": games}, 200, headers)