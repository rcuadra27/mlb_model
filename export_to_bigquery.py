#!/usr/bin/env python3
"""
export_to_bigquery.py

Exports today's inference predictions + odds from Postgres to BigQuery.
Run after inference completes each day.

Usage:
    PG_DSN=... GOOGLE_CLOUD_PROJECT=mlb-model-491223 python export_to_bigquery.py --date 2026-04-01
"""

import os
import argparse
import pandas as pd
from sqlalchemy import create_engine, text
from google.cloud import bigquery

BQ_DATASET = "mlb_model_logs"
BQ_TABLE   = "daily_games"

def export(date_str: str, schema: str, engine, bq_client):
    print(f"Exporting {date_str} to BigQuery...")

    df = pd.read_sql(text(f"""
        SELECT
            p.game_date,
            p.game_id,
            p.away_team,
            p.home_team,
            sp.away_sp_name,
            sp.home_sp_name,
            p.away_runs_pred,
            p.home_runs_pred,
            p.total_runs_pred,
            p.p_away_win_poisson  AS p_win_away,
            p.p_home_win_poisson  AS p_win_home,
            p.ou_line,
            p.ou_recommendation,
            p.ou_edge_over,
            p.ou_edge_under,
            p.edge_away,
            p.edge_home,
            p.ev_away,
            p.ev_home,
            p.ev_over,
            p.ev_under,
            p.is_value_ml_away,
            p.is_value_ml_home,
            p.is_value_ou_over,
            p.is_value_ou_under,
            p.p_away_market_median,
            p.p_home_market_median,
            p.n_books_ml,
            p.n_books_ou,
            fg.morning_p_home,
            fg.closing_p_home,
            fg.morning_ou_line,
            fg.closing_ou_line,
            fg.total_line_move,
        fg.home_line_move,
            fg.sharp_action_home,
        fg.morning_home_price,
        fg.morning_away_price,
        fg.closing_home_price,
        fg.closing_away_price,
            p.as_of_ts,
            g.first_pitch_utc,
            CAST(g.away_runs AS FLOAT8) AS away_runs,
            CAST(g.home_runs AS FLOAT8) AS home_runs,
            g.status
        FROM (
            SELECT DISTINCT ON (game_id) *
            FROM {schema}.inference_game_predictions
            WHERE game_date = :d
            ORDER BY game_id, as_of_ts DESC
        ) p
        LEFT JOIN {schema}.game_starting_pitchers sp USING (game_id)
        LEFT JOIN {schema}.features_game fg USING (game_id)
        LEFT JOIN {schema}.games g ON g.game_id = p.game_id AND g.game_date = :d
        JOIN (
            SELECT game_id
            FROM public.game_lineups
            WHERE game_date = :d
            GROUP BY game_id
            HAVING COUNT(DISTINCT is_home) = 2
        ) confirmed_lineups ON confirmed_lineups.game_id = p.game_id
        ORDER BY p.game_id
    """), engine, params={"d": date_str})

    if df.empty:
        print(f"  No predictions found for {date_str} — nothing to export")
        return

    print(f"  Found {len(df)} games")

    # Convert date columns to string for BQ compatibility
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    if "as_of_ts" in df.columns:
        df["as_of_ts"] = df["as_of_ts"].astype(str)

    # Explicit schema — never use autodetect to avoid type conflicts
    from google.cloud.bigquery import SchemaField, TimePartitioning
    bq_schema = [
       SchemaField("game_date", "DATE"),
        SchemaField("game_id", "INTEGER"),
        SchemaField("away_team", "STRING"),
        SchemaField("home_team", "STRING"),
        SchemaField("away_sp_name", "STRING"),
        SchemaField("home_sp_name", "STRING"),
        SchemaField("away_runs_pred", "FLOAT"),
        SchemaField("home_runs_pred", "FLOAT"),
        SchemaField("total_runs_pred", "FLOAT"),
        SchemaField("p_win_away", "FLOAT"),
        SchemaField("p_win_home", "FLOAT"),
        SchemaField("ou_line", "FLOAT"),
        SchemaField("ou_recommendation", "STRING"),
        SchemaField("ou_edge_over", "FLOAT"),
        SchemaField("ou_edge_under", "FLOAT"),
        SchemaField("edge_away", "FLOAT"),
        SchemaField("edge_home", "FLOAT"),
        SchemaField("ev_away", "FLOAT"),
        SchemaField("ev_home", "FLOAT"),
        SchemaField("ev_over", "FLOAT"),
        SchemaField("ev_under", "FLOAT"),
        SchemaField("is_value_ml_away", "BOOLEAN"),
        SchemaField("is_value_ml_home", "BOOLEAN"),
        SchemaField("is_value_ou_over", "BOOLEAN"),
        SchemaField("is_value_ou_under", "BOOLEAN"),
        SchemaField("p_away_market_median", "FLOAT"),
        SchemaField("p_home_market_median", "FLOAT"),
        SchemaField("n_books_ml", "FLOAT"),
        SchemaField("n_books_ou", "FLOAT"),
        SchemaField("morning_p_home", "FLOAT"),
        SchemaField("closing_p_home", "FLOAT"),
        SchemaField("morning_ou_line", "FLOAT"),
        SchemaField("closing_ou_line", "FLOAT"),
        SchemaField("total_line_move", "FLOAT"),
        SchemaField("home_line_move", "FLOAT"),
        SchemaField("sharp_action_home", "FLOAT"),
        SchemaField("morning_home_price", "INTEGER"),
        SchemaField("morning_away_price", "INTEGER"),
        SchemaField("closing_home_price", "FLOAT"),
        SchemaField("closing_away_price", "FLOAT"),
        SchemaField("as_of_ts", "STRING"),
        SchemaField("first_pitch_utc", "TIMESTAMP"),
        SchemaField("away_runs", "FLOAT"),
        SchemaField("home_runs", "FLOAT"),
        SchemaField("status", "STRING"),
    ]
    base_ref = f"{bq_client.project}.{BQ_DATASET}.{BQ_TABLE}"
    date_nodash = date_str.replace("-", "")
    try:
        bq_client.get_table(base_ref)
        table_ref = f"{base_ref}${date_nodash}"
    except Exception:
        table_ref = base_ref
    job_config = bigquery.LoadJobConfig(
        schema=bq_schema,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        time_partitioning=TimePartitioning(field="game_date"),
    )
    job = bq_client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()

    print(f"  Exported {len(df)} games to {table_ref}")
    print(f"  Value bets — ML: {df['is_value_ml_away'].sum() + df['is_value_ml_home'].sum()} | O/U: {df['is_value_ou_over'].sum() + df['is_value_ou_under'].sum()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    ap.add_argument("--schema", default="public")
    args = ap.parse_args()

    pg_dsn = os.getenv("PG_DSN")
    if not pg_dsn:
        raise RuntimeError("PG_DSN env var required.")

    engine    = create_engine(pg_dsn, pool_pre_ping=True)
    bq_client = bigquery.Client(project="mlb-model-491223")

    export(args.date, args.schema, engine, bq_client)


if __name__ == "__main__":
    main()
