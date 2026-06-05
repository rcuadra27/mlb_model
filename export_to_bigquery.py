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

from inference.prediction_snapshots import best_pred_cte, totals_anchor_cte

BQ_DATASET = "mlb_model_logs"
BQ_TABLE   = "daily_games"
BQ_PLAYER_PROPS = "player_prop_predictions"
BQ_PITCHER_PROPS = "pitcher_prop_predictions"
BQ_DAILY_TRENDS = "daily_trends"
BQ_DAILY_EDGES = "daily_edges"
BQ_STANDINGS = "standings"
BQ_STANDINGS_PROJECTIONS = "standings_projections"
BQ_TRANSACTIONS = "transactions"
BQ_MODEL_PERFORMANCE = "model_performance_snapshot"


def _ensure_bq_schema(bq_client, table_id: str, schema_fields) -> None:
    """Add any missing columns to an existing BQ table."""
    try:
        table = bq_client.get_table(table_id)
    except Exception:
        return
    existing = {f.name for f in table.schema}
    new_fields = [f for f in schema_fields if f.name not in existing]
    if not new_fields:
        return
    table.schema = list(table.schema) + new_fields
    bq_client.update_table(table, ["schema"])
    print(f"  Added BQ columns to {table_id}: {[f.name for f in new_fields]}")


def _load_partitioned_table(bq_client, table_name, date_str, df, schema_fields, partition_field="game_date"):
    """Load a date-partitioned BQ table (same pattern as daily_games)."""
    from google.cloud.bigquery import TimePartitioning

    if df.empty:
        print(f"  No rows for {table_name} on {date_str} — skipping")
        return

    if partition_field in df.columns:
        df[partition_field] = pd.to_datetime(df[partition_field]).dt.date
    if "as_of_ts" in df.columns:
        df["as_of_ts"] = df["as_of_ts"].astype(str)

    base_ref = f"{bq_client.project}.{BQ_DATASET}.{table_name}"
    date_nodash = date_str.replace("-", "")
    table_ref = base_ref
    time_partitioning = TimePartitioning(field=partition_field)
    clustering_fields = None

    try:
        table = bq_client.get_table(base_ref)
        table_ref = f"{base_ref}${date_nodash}"
        if table.time_partitioning:
            time_partitioning = table.time_partitioning
        if table.clustering_fields:
            clustering_fields = list(table.clustering_fields)
    except Exception:
        pass

    job_kwargs = {
        "schema": schema_fields,
        "write_disposition": bigquery.WriteDisposition.WRITE_TRUNCATE,
        "time_partitioning": time_partitioning,
        "schema_update_options": [bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION],
    }
    if clustering_fields:
        job_kwargs["clustering_fields"] = clustering_fields

    job_config = bigquery.LoadJobConfig(**job_kwargs)
    job = bq_client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()
    print(f"  Exported {len(df):,} rows to {table_ref}")

    if not clustering_fields and partition_field == "game_date":
        try:
            table = bq_client.get_table(base_ref)
            if list(table.clustering_fields or []) != ["game_date", "game_id"]:
                table.clustering_fields = ["game_date", "game_id"]
                bq_client.update_table(table, ["clustering_fields"])
                print(f"  Updated clustering on {base_ref}")
        except Exception as exc:
            print(f"  Warning: could not update clustering on {base_ref}: {exc}")


def _load_trends_partitioned_table(bq_client, table_name, date_str, df, schema_fields):
    """Load daily_trends partitioned on trend_date."""
    from google.cloud.bigquery import TimePartitioning

    if df.empty:
        print(f"  No rows for {table_name} on {date_str} — skipping")
        return

    if "trend_date" in df.columns:
        df["trend_date"] = pd.to_datetime(df["trend_date"]).dt.date
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)

    base_ref = f"{bq_client.project}.{BQ_DATASET}.{table_name}"
    date_nodash = date_str.replace("-", "")
    table_ref = base_ref
    time_partitioning = TimePartitioning(field="trend_date")
    clustering_fields = None

    try:
        table = bq_client.get_table(base_ref)
        existing = {field.name for field in table.schema}
        additions = [field for field in schema_fields if field.name not in existing]
        if additions:
            table.schema = list(table.schema) + additions
            bq_client.update_table(table, ["schema"])
            print(f"  Added schema fields to {base_ref}: {', '.join(f.name for f in additions)}")
        table_ref = f"{base_ref}${date_nodash}"
        if table.time_partitioning:
            time_partitioning = table.time_partitioning
        if table.clustering_fields:
            clustering_fields = list(table.clustering_fields)
    except Exception:
        pass

    job_kwargs = {
        "schema": schema_fields,
        "write_disposition": bigquery.WriteDisposition.WRITE_TRUNCATE,
        "time_partitioning": time_partitioning,
    }
    if clustering_fields:
        job_kwargs["clustering_fields"] = clustering_fields

    job_config = bigquery.LoadJobConfig(**job_kwargs)
    job = bq_client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()
    print(f"  Exported {len(df):,} rows to {table_ref}")


def export_player_props(date_str: str, schema: str, engine, bq_client):
    from google.cloud.bigquery import SchemaField

    print(f"Exporting player props for {date_str}...")
    df = pd.read_sql(text(f"""
        WITH latest AS (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY game_id, batter_id
                    ORDER BY COALESCE(lineup_confirmed, FALSE) DESC, as_of_ts DESC
                ) AS rn
            FROM {schema}.player_prop_predictions
            WHERE game_date = :d
        )
        SELECT
            l.game_date,
            l.game_id,
            l.batter_id,
            l.batter_name,
            l.team_id,
            l.batting_order,
            l.sp_id,
            l.sp_name,
            l.p_hit,
            l.p_2plus_hits,
            l.p_hr,
            l.p_k,
            l.p_2plus_bases,
            l.p_walk,
            l.matchup_score,
            l.platoon_advantage,
            l.batter_xwoba_season,
            l.batter_hit_rate_30d,
            CAST(l.lineup_confirmed AS BOOL) AS lineup_confirmed,
            CASE WHEN l.team_id = g.home_team_id THEN TRUE ELSE FALSE END AS is_home,
            g.away_team_name AS away_team,
            g.home_team_name AS home_team,
            l.as_of_ts
        FROM latest l
        JOIN {schema}.games g
          ON g.game_id = l.game_id AND g.game_date = l.game_date
        WHERE l.rn = 1
        ORDER BY l.game_id, is_home DESC, l.batting_order
    """), engine, params={"d": date_str})

    schema_fields = [
        SchemaField("game_date", "DATE"),
        SchemaField("game_id", "INTEGER"),
        SchemaField("batter_id", "INTEGER"),
        SchemaField("batter_name", "STRING"),
        SchemaField("team_id", "INTEGER"),
        SchemaField("batting_order", "INTEGER"),
        SchemaField("sp_id", "INTEGER"),
        SchemaField("sp_name", "STRING"),
        SchemaField("p_hit", "FLOAT"),
        SchemaField("p_2plus_hits", "FLOAT"),
        SchemaField("p_hr", "FLOAT"),
        SchemaField("p_k", "FLOAT"),
        SchemaField("p_2plus_bases", "FLOAT"),
        SchemaField("p_walk", "FLOAT"),
        SchemaField("matchup_score", "FLOAT"),
        SchemaField("platoon_advantage", "INTEGER"),
        SchemaField("batter_xwoba_season", "FLOAT"),
        SchemaField("batter_hit_rate_30d", "FLOAT"),
        SchemaField("lineup_confirmed", "BOOL"),
        SchemaField("is_home", "BOOLEAN"),
        SchemaField("away_team", "STRING"),
        SchemaField("home_team", "STRING"),
        SchemaField("as_of_ts", "STRING"),
    ]
    _load_partitioned_table(bq_client, BQ_PLAYER_PROPS, date_str, df, schema_fields)


def export_pitcher_props(date_str: str, schema: str, engine, bq_client):
    from google.cloud.bigquery import SchemaField

    print(f"Exporting pitcher props for {date_str}...")
    df = pd.read_sql(text(f"""
        WITH latest AS (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY game_id, pitcher_id
                    ORDER BY as_of_ts DESC
                ) AS rn
            FROM {schema}.pitcher_prop_predictions
            WHERE game_date = :d
        )
        SELECT
            l.game_date,
            l.game_id,
            l.pitcher_id,
            l.pitcher_name,
            l.is_home,
            l.lambda_k,
            l.lambda_walks,
            l.lambda_hits,
            l.lambda_er,
            l.p_k0, l.p_k1, l.p_k2, l.p_k3, l.p_k4,
            l.p_k5, l.p_k6, l.p_k7, l.p_k8, l.p_k9, l.p_k10,
            l.p_k10plus,
            l.p_over_0_5, l.p_over_1_5, l.p_over_2_5, l.p_over_3_5,
            l.p_over_4_5, l.p_over_5_5, l.p_over_6_5, l.p_over_7_5,
            l.p_over_8_5, l.p_over_9_5,
            l.p_walks_over_0_5, l.p_walks_over_1_5, l.p_walks_over_2_5, l.p_walks_over_3_5, l.p_walks_over_4_5, l.p_walks_over_5_5,
            l.p_hits_over_3_5, l.p_hits_over_4_5, l.p_hits_over_5_5, l.p_hits_over_6_5, l.p_hits_over_7_5, l.p_hits_over_8_5,
            l.p_er_over_1_5, l.p_er_over_2_5, l.p_er_over_3_5, l.p_er_over_4_5, l.p_er_over_5_5,
            l.sp_k_rate_season,
            l.sp_innings_season,
            l.opp_lineup_k_rate,
            l.expected_ip,
            g.away_team_name AS away_team,
            g.home_team_name AS home_team,
            l.as_of_ts
        FROM latest l
        JOIN {schema}.games g
          ON g.game_id = l.game_id AND g.game_date = l.game_date
        WHERE l.rn = 1
        ORDER BY l.game_id, l.is_home DESC
    """), engine, params={"d": date_str})

    schema_fields = [
        SchemaField("game_date", "DATE"),
        SchemaField("game_id", "INTEGER"),
        SchemaField("pitcher_id", "INTEGER"),
        SchemaField("pitcher_name", "STRING"),
        SchemaField("is_home", "BOOLEAN"),
        SchemaField("lambda_k", "FLOAT"),
        SchemaField("lambda_walks", "FLOAT"),
        SchemaField("lambda_hits", "FLOAT"),
        SchemaField("lambda_er", "FLOAT"),
        SchemaField("p_k0", "FLOAT"),
        SchemaField("p_k1", "FLOAT"),
        SchemaField("p_k2", "FLOAT"),
        SchemaField("p_k3", "FLOAT"),
        SchemaField("p_k4", "FLOAT"),
        SchemaField("p_k5", "FLOAT"),
        SchemaField("p_k6", "FLOAT"),
        SchemaField("p_k7", "FLOAT"),
        SchemaField("p_k8", "FLOAT"),
        SchemaField("p_k9", "FLOAT"),
        SchemaField("p_k10", "FLOAT"),
        SchemaField("p_k10plus", "FLOAT"),
        SchemaField("p_over_0_5", "FLOAT"),
        SchemaField("p_over_1_5", "FLOAT"),
        SchemaField("p_over_2_5", "FLOAT"),
        SchemaField("p_over_3_5", "FLOAT"),
        SchemaField("p_over_4_5", "FLOAT"),
        SchemaField("p_over_5_5", "FLOAT"),
        SchemaField("p_over_6_5", "FLOAT"),
        SchemaField("p_over_7_5", "FLOAT"),
        SchemaField("p_over_8_5", "FLOAT"),
        SchemaField("p_over_9_5", "FLOAT"),
        SchemaField("p_walks_over_0_5", "FLOAT"),
        SchemaField("p_walks_over_1_5", "FLOAT"),
        SchemaField("p_walks_over_2_5", "FLOAT"),
        SchemaField("p_walks_over_3_5", "FLOAT"),
        SchemaField("p_walks_over_4_5", "FLOAT"),
        SchemaField("p_walks_over_5_5", "FLOAT"),
        SchemaField("p_hits_over_3_5", "FLOAT"),
        SchemaField("p_hits_over_4_5", "FLOAT"),
        SchemaField("p_hits_over_5_5", "FLOAT"),
        SchemaField("p_hits_over_6_5", "FLOAT"),
        SchemaField("p_hits_over_7_5", "FLOAT"),
        SchemaField("p_hits_over_8_5", "FLOAT"),
        SchemaField("p_er_over_1_5", "FLOAT"),
        SchemaField("p_er_over_2_5", "FLOAT"),
        SchemaField("p_er_over_3_5", "FLOAT"),
        SchemaField("p_er_over_4_5", "FLOAT"),
        SchemaField("p_er_over_5_5", "FLOAT"),
        SchemaField("sp_k_rate_season", "FLOAT"),
        SchemaField("sp_innings_season", "FLOAT"),
        SchemaField("opp_lineup_k_rate", "FLOAT"),
        SchemaField("expected_ip", "FLOAT"),
        SchemaField("away_team", "STRING"),
        SchemaField("home_team", "STRING"),
        SchemaField("as_of_ts", "STRING"),
    ]
    _load_partitioned_table(bq_client, BQ_PITCHER_PROPS, date_str, df, schema_fields)


def export_daily_trends(date_str: str, schema: str, engine, bq_client):
    from google.cloud.bigquery import SchemaField

    print(f"Exporting daily trends for {date_str}...")
    df = pd.read_sql(text(f"""
        SELECT
            trend_date,
            trend_type,
            rank,
            name,
            meta,
            team_id,
            team_abbr,
            team_name,
            value_primary,
            value_secondary,
            value_label,
            direction,
            created_at
        FROM {schema}.daily_trends
        WHERE trend_date = :d
        ORDER BY trend_type, rank
    """), engine, params={"d": date_str})

    schema_fields = [
        SchemaField("trend_date", "DATE"),
        SchemaField("trend_type", "STRING"),
        SchemaField("rank", "INTEGER"),
        SchemaField("name", "STRING"),
        SchemaField("meta", "STRING"),
        SchemaField("team_id", "INTEGER"),
        SchemaField("team_abbr", "STRING"),
        SchemaField("team_name", "STRING"),
        SchemaField("value_primary", "FLOAT"),
        SchemaField("value_secondary", "FLOAT"),
        SchemaField("value_label", "STRING"),
        SchemaField("direction", "STRING"),
        SchemaField("created_at", "TIMESTAMP"),
    ]
    _load_trends_partitioned_table(bq_client, BQ_DAILY_TRENDS, date_str, df, schema_fields)


def _load_edges_partitioned_table(bq_client, table_name, date_str, df, schema_fields):
    """Load daily_edges partitioned on edge_date."""
    from google.cloud.bigquery import TimePartitioning

    if df.empty:
        print(f"  No rows for {table_name} on {date_str} — skipping")
        return

    if "edge_date" in df.columns:
        df["edge_date"] = pd.to_datetime(df["edge_date"]).dt.date
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)

    base_ref = f"{bq_client.project}.{BQ_DATASET}.{table_name}"
    date_nodash = date_str.replace("-", "")
    table_ref = base_ref
    time_partitioning = TimePartitioning(field="edge_date")
    clustering_fields = None

    try:
        table = bq_client.get_table(base_ref)
        existing = {field.name for field in table.schema}
        additions = [field for field in schema_fields if field.name not in existing]
        if additions:
            table.schema = list(table.schema) + additions
            bq_client.update_table(table, ["schema"])
            print(f"  Added schema fields to {base_ref}: {', '.join(f.name for f in additions)}")
        table_ref = f"{base_ref}${date_nodash}"
        if table.time_partitioning:
            time_partitioning = table.time_partitioning
        if table.clustering_fields:
            clustering_fields = list(table.clustering_fields)
    except Exception:
        pass

    job_kwargs = {
        "schema": schema_fields,
        "write_disposition": bigquery.WriteDisposition.WRITE_TRUNCATE,
        "time_partitioning": time_partitioning,
        "schema_update_options": [bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION],
    }
    if clustering_fields:
        job_kwargs["clustering_fields"] = clustering_fields

    job_config = bigquery.LoadJobConfig(**job_kwargs)
    job = bq_client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()
    print(f"  Exported {len(df):,} rows to {table_ref}")


def export_daily_edges(date_str: str, schema: str, engine, bq_client):
    from google.cloud.bigquery import SchemaField

    print(f"Exporting daily edges for {date_str}...")
    df = pd.read_sql(text(f"""
        SELECT
            edge_date,
            rank,
            edge_type,
            prop_subtype,
            pick_description,
            detail_line,
            rate_detail_line,
            market_line,
            model_prob_pct,
            model_value,
            comparison_value,
            edge_magnitude,
            direction,
            game_id,
            player_id,
            team_id,
            team_abbr,
            team_name,
            created_at
        FROM {schema}.daily_edges
        WHERE edge_date = :d
        ORDER BY rank
    """), engine, params={"d": date_str})

    schema_fields = [
        SchemaField("edge_date", "DATE"),
        SchemaField("rank", "INTEGER"),
        SchemaField("edge_type", "STRING"),
        SchemaField("prop_subtype", "STRING"),
        SchemaField("pick_description", "STRING"),
        SchemaField("detail_line", "STRING"),
        SchemaField("rate_detail_line", "STRING"),
        SchemaField("market_line", "FLOAT"),
        SchemaField("model_prob_pct", "FLOAT"),
        SchemaField("model_value", "FLOAT"),
        SchemaField("comparison_value", "FLOAT"),
        SchemaField("edge_magnitude", "FLOAT"),
        SchemaField("direction", "STRING"),
        SchemaField("game_id", "INTEGER"),
        SchemaField("player_id", "INTEGER"),
        SchemaField("team_id", "INTEGER"),
        SchemaField("team_abbr", "STRING"),
        SchemaField("team_name", "STRING"),
        SchemaField("created_at", "TIMESTAMP"),
    ]
    _load_edges_partitioned_table(bq_client, BQ_DAILY_EDGES, date_str, df, schema_fields)


def export_standings(date_str: str, schema: str, engine, bq_client):
    from google.cloud.bigquery import SchemaField

    print(f"Exporting standings for {date_str}...")
    df = pd.read_sql(text(f"""
        SELECT
            snapshot_date,
            season,
            league_id,
            league_name,
            division_id,
            division_name,
            division_name_short,
            team_id,
            team_name,
            abbreviation,
            rank,
            wins,
            losses,
            pct,
            games_back,
            streak,
            last_10,
            run_diff,
            runs_scored,
            runs_allowed,
            created_at
        FROM {schema}.standings
        WHERE snapshot_date = :d
        ORDER BY league_id, division_id, rank
    """), engine, params={"d": date_str})

    schema_fields = [
        SchemaField("snapshot_date", "DATE"),
        SchemaField("season", "INTEGER"),
        SchemaField("league_id", "INTEGER"),
        SchemaField("league_name", "STRING"),
        SchemaField("division_id", "INTEGER"),
        SchemaField("division_name", "STRING"),
        SchemaField("division_name_short", "STRING"),
        SchemaField("team_id", "INTEGER"),
        SchemaField("team_name", "STRING"),
        SchemaField("abbreviation", "STRING"),
        SchemaField("rank", "INTEGER"),
        SchemaField("wins", "INTEGER"),
        SchemaField("losses", "INTEGER"),
        SchemaField("pct", "FLOAT"),
        SchemaField("games_back", "STRING"),
        SchemaField("streak", "STRING"),
        SchemaField("last_10", "STRING"),
        SchemaField("run_diff", "INTEGER"),
        SchemaField("runs_scored", "INTEGER"),
        SchemaField("runs_allowed", "INTEGER"),
        SchemaField("created_at", "TIMESTAMP"),
    ]
    _load_partitioned_table(bq_client, BQ_STANDINGS, date_str, df, schema_fields, partition_field="snapshot_date")


def export_standings_projections(date_str: str, schema: str, engine, bq_client):
    from google.cloud.bigquery import SchemaField

    print(f"Exporting standings projections for {date_str}...")
    df = pd.read_sql(text(f"""
        SELECT
            snapshot_date,
            season,
            team_id,
            team_name,
            projected_wins,
            projected_losses,
            projected_record,
            playoff_odds,
            remaining_games,
            simulations,
            created_at
        FROM {schema}.standings_projections
        WHERE snapshot_date = :d
        ORDER BY playoff_odds DESC, projected_wins DESC
    """), engine, params={"d": date_str})

    schema_fields = [
        SchemaField("snapshot_date", "DATE"),
        SchemaField("season", "INTEGER"),
        SchemaField("team_id", "INTEGER"),
        SchemaField("team_name", "STRING"),
        SchemaField("projected_wins", "FLOAT"),
        SchemaField("projected_losses", "FLOAT"),
        SchemaField("projected_record", "STRING"),
        SchemaField("playoff_odds", "FLOAT"),
        SchemaField("remaining_games", "INTEGER"),
        SchemaField("simulations", "INTEGER"),
        SchemaField("created_at", "TIMESTAMP"),
    ]
    _load_partitioned_table(
        bq_client,
        BQ_STANDINGS_PROJECTIONS,
        date_str,
        df,
        schema_fields,
        partition_field="snapshot_date",
    )


def export_transactions(date_str: str, schema: str, engine, bq_client):
    from google.cloud.bigquery import SchemaField, TimePartitioning

    print(f"Exporting transactions through {date_str}...")
    df = pd.read_sql(text(f"""
        SELECT
            transaction_id,
            transaction_date,
            team_id,
            team_name,
            player_id,
            player_name,
            transaction_type,
            type_code,
            description,
            created_at
        FROM {schema}.transactions
        WHERE transaction_date BETWEEN CAST(:d AS DATE) - INTERVAL '60 days' AND CAST(:d AS DATE)
        ORDER BY transaction_date DESC, transaction_id DESC
    """), engine, params={"d": date_str})

    schema_fields = [
        SchemaField("transaction_id", "INTEGER"),
        SchemaField("transaction_date", "DATE"),
        SchemaField("team_id", "INTEGER"),
        SchemaField("team_name", "STRING"),
        SchemaField("player_id", "INTEGER"),
        SchemaField("player_name", "STRING"),
        SchemaField("transaction_type", "STRING"),
        SchemaField("type_code", "STRING"),
        SchemaField("description", "STRING"),
        SchemaField("created_at", "TIMESTAMP"),
    ]
    if df.empty:
        print(f"  No transactions through {date_str} — skipping")
        return

    df["transaction_date"] = pd.to_datetime(df["transaction_date"]).dt.date
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)

    table_ref = f"{bq_client.project}.{BQ_DATASET}.{BQ_TRANSACTIONS}"
    delete_sql = f"""
        DELETE FROM `{table_ref}`
        WHERE transaction_date BETWEEN DATE_SUB(DATE(@date_str), INTERVAL 60 DAY) AND DATE(@date_str)
    """
    delete_job = bq_client.query(
        delete_sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("date_str", "STRING", date_str)]
        ),
    )
    try:
        delete_job.result()
    except Exception as exc:
        print(f"  Warning: could not delete existing transaction rows before load: {exc}")

    job_config = bigquery.LoadJobConfig(
        schema=schema_fields,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        time_partitioning=TimePartitioning(field="transaction_date"),
    )
    job = bq_client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()
    print(f"  Exported {len(df):,} rows to {table_ref}")


def export_model_performance(date_str: str, schema: str, engine, bq_client):
    from google.cloud.bigquery import SchemaField

    print(f"Exporting model performance snapshot for {date_str}...")
    df = pd.read_sql(text(f"""
        SELECT
            snapshot_date,
            version,
            min_game_date,
            games_graded,
            calibration_error_pct,
            brier_score,
            brier_market,
            accuracy_pct,
            pitcher_k_starters_graded,
            payload::text AS payload_json,
            created_at
        FROM {schema}.model_performance_snapshot
        WHERE snapshot_date = :d
    """), engine, params={"d": date_str})

    schema_fields = [
        SchemaField("snapshot_date", "DATE"),
        SchemaField("version", "STRING"),
        SchemaField("min_game_date", "DATE"),
        SchemaField("games_graded", "INTEGER"),
        SchemaField("calibration_error_pct", "FLOAT"),
        SchemaField("brier_score", "FLOAT"),
        SchemaField("brier_market", "FLOAT"),
        SchemaField("accuracy_pct", "FLOAT"),
        SchemaField("pitcher_k_starters_graded", "INTEGER"),
        SchemaField("payload_json", "STRING"),
        SchemaField("created_at", "TIMESTAMP"),
    ]
    _load_partitioned_table(
        bq_client,
        BQ_MODEL_PERFORMANCE,
        date_str,
        df,
        schema_fields,
        partition_field="snapshot_date",
    )


def _daily_games_bq_schema():
    from google.cloud.bigquery import SchemaField

    return [
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
        SchemaField("lineup_pending", "BOOLEAN"),
        SchemaField("first_pitch_utc", "TIMESTAMP"),
        SchemaField("away_runs", "FLOAT"),
        SchemaField("home_runs", "FLOAT"),
        SchemaField("status", "STRING"),
    ]


def _daily_games_sql(schema: str, game_ids: list[int] | None = None) -> str:
    gid_clause = ""
    if game_ids:
        gid_clause = " AND g.game_id = ANY(:gids)"
    return f"""
        WITH {totals_anchor_cte(schema)},
        {best_pred_cte(schema)}
        SELECT
            p.game_date,
            p.game_id,
            p.away_team,
            p.home_team,
            sp.away_sp_name,
            sp.home_sp_name,
            COALESCE(ta.away_runs_pred, p.away_runs_pred) AS away_runs_pred,
            COALESCE(ta.home_runs_pred, p.home_runs_pred) AS home_runs_pred,
            COALESCE(ta.total_runs_pred, p.total_runs_pred) AS total_runs_pred,
            p.p_away_win_poisson  AS p_win_away,
            p.p_home_win_poisson  AS p_win_home,
            COALESCE(ta.ou_line, p.ou_line, fg.morning_ou_line, fg.closing_ou_line) AS ou_line,
            COALESCE(ta.ou_recommendation, p.ou_recommendation) AS ou_recommendation,
            COALESCE(ta.ou_edge_over, p.ou_edge_over) AS ou_edge_over,
            COALESCE(ta.ou_edge_under, p.ou_edge_under) AS ou_edge_under,
            p.edge_away,
            p.edge_home,
            p.ev_away,
            p.ev_home,
            COALESCE(ta.ev_over, p.ev_over) AS ev_over,
            COALESCE(ta.ev_under, p.ev_under) AS ev_under,
            p.is_value_ml_away,
            p.is_value_ml_home,
            COALESCE(ta.is_value_ou_over, p.is_value_ou_over) AS is_value_ou_over,
            COALESCE(ta.is_value_ou_under, p.is_value_ou_under) AS is_value_ou_under,
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
            COALESCE(p.lineup_pending, FALSE) AS lineup_pending,
            g.first_pitch_utc,
            CAST(g.away_runs AS FLOAT8) AS away_runs,
            CAST(g.home_runs AS FLOAT8) AS home_runs,
            g.status
        FROM {schema}.games g
        INNER JOIN best_pred p ON p.game_id = g.game_id AND p.game_date = g.game_date
        LEFT JOIN totals_anchor ta ON ta.game_id = p.game_id
        LEFT JOIN {schema}.game_starting_pitchers sp ON sp.game_id = g.game_id
        LEFT JOIN {schema}.features_game fg ON fg.game_id = g.game_id
        WHERE g.game_date = :d{gid_clause}
        ORDER BY g.game_id
    """


def _fetch_daily_games_df(
    date_str: str,
    schema: str,
    engine,
    game_ids: list[int] | None = None,
) -> pd.DataFrame:
    params: dict = {"d": date_str}
    if game_ids:
        params["gids"] = game_ids
    df = pd.read_sql(
        text(_daily_games_sql(schema, game_ids)),
        engine,
        params=params,
    )
    if df.empty:
        return df
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    if "as_of_ts" in df.columns:
        df["as_of_ts"] = df["as_of_ts"].astype(str)
    return df


def _write_daily_games_partition(date_str: str, df: pd.DataFrame, bq_client) -> None:
    from google.cloud.bigquery import TimePartitioning

    if df.empty:
        print(f"  No rows to write for {date_str} — skipping")
        return

    bq_schema = _daily_games_bq_schema()
    base_ref = f"{bq_client.project}.{BQ_DATASET}.{BQ_TABLE}"
    date_nodash = date_str.replace("-", "")
    _ensure_bq_schema(bq_client, base_ref, bq_schema)
    try:
        bq_client.get_table(base_ref)
        table_ref = f"{base_ref}${date_nodash}"
    except Exception:
        table_ref = base_ref
    job_config = bigquery.LoadJobConfig(
        schema=bq_schema,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        time_partitioning=TimePartitioning(field="game_date"),
        schema_update_options=[
            bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION,
        ],
    )
    job = bq_client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()
    print(f"  Exported {len(df)} games to {table_ref}")
    if "is_value_ml_away" in df.columns:
        print(
            f"  Value bets — ML: {df['is_value_ml_away'].sum() + df['is_value_ml_home'].sum()} "
            f"| O/U: {df['is_value_ou_over'].sum() + df['is_value_ou_under'].sum()}"
        )


def _fetch_existing_daily_games_bq(bq_client, date_str: str) -> pd.DataFrame:
    table = f"{bq_client.project}.{BQ_DATASET}.{BQ_TABLE}"
    q = f"SELECT * FROM `{table}` WHERE game_date = @d"
    job = bq_client.query(
        q,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("d", "DATE", date_str),
            ]
        ),
    )
    return job.to_dataframe()


def export(date_str: str, schema: str, engine, bq_client):
    print(f"Exporting {date_str} to BigQuery...")
    df = _fetch_daily_games_df(date_str, schema, engine)
    if df.empty:
        print(f"  No predictions found for {date_str} — nothing to export")
        return
    print(f"  Found {len(df)} games")
    ou_pop = df["ou_line"].notna().sum() if "ou_line" in df.columns else 0
    print(f"  Games with ou_line: {ou_pop}/{len(df)}")
    _write_daily_games_partition(date_str, df, bq_client)


def export_lineup_refresh(
    date_str: str,
    schema: str,
    engine,
    bq_client,
    game_ids: list[int],
) -> None:
    """
    Merge lineup-dependent game rows into the existing BQ partition without
    recomputing totals from a full-table export of stale inference rows.
    """
    if not game_ids:
        print("  lineup_refresh: no game_ids — skipping")
        return

    print(f"  lineup_refresh: merging {len(game_ids)} game(s) into daily_games for {date_str}")
    existing = _fetch_existing_daily_games_bq(bq_client, date_str)
    fresh = _fetch_daily_games_df(date_str, schema, engine, game_ids=game_ids)
    if fresh.empty:
        print("  lineup_refresh: no postgres rows for game_ids — skipping")
        return

    gid_set = set(int(g) for g in game_ids)
    if not existing.empty and "game_id" in existing.columns:
        keep = existing[~existing["game_id"].isin(gid_set)]
        merged = pd.concat([keep, fresh], ignore_index=True)
    else:
        merged = fresh

    if "game_date" in merged.columns:
        merged["game_date"] = pd.to_datetime(merged["game_date"]).dt.date
    merged = merged.sort_values("game_id").reset_index(drop=True)
    _write_daily_games_partition(date_str, merged, bq_client)


def _parse_game_ids(raw: str | None) -> list[int]:
    if not raw or not str(raw).strip():
        return []
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    ap.add_argument("--schema", default="public")
    ap.add_argument(
        "--only",
        choices=(
            "edges",
            "trends",
            "derived",
            "props",
            "standings",
            "transactions",
            "model_performance",
            "lineup_refresh",
        ),
        help="Skip main exports; lineup_refresh = merge game rows only (requires --game-ids)",
    )
    ap.add_argument(
        "--game-ids",
        default="",
        help="Comma-separated game_ids for --only lineup_refresh",
    )
    args = ap.parse_args()

    if not args.date or not str(args.date).strip():
        raise SystemExit("--date is required (e.g. 2026-05-29)")

    pg_dsn = os.getenv("PG_DSN")
    if not pg_dsn:
        raise RuntimeError("PG_DSN env var required.")

    engine    = create_engine(pg_dsn, pool_pre_ping=True)
    bq_client = bigquery.Client(project="mlb-model-491223")

    if args.only == "edges":
        export_daily_edges(args.date, args.schema, engine, bq_client)
        return
    if args.only == "trends":
        export_daily_trends(args.date, args.schema, engine, bq_client)
        return
    if args.only == "derived":
        export_daily_trends(args.date, args.schema, engine, bq_client)
        export_daily_edges(args.date, args.schema, engine, bq_client)
        return
    if args.only == "props":
        export_player_props(args.date, args.schema, engine, bq_client)
        export_pitcher_props(args.date, args.schema, engine, bq_client)
        return
    if args.only == "standings":
        export_standings(args.date, args.schema, engine, bq_client)
        export_standings_projections(args.date, args.schema, engine, bq_client)
        return
    if args.only == "transactions":
        export_transactions(args.date, args.schema, engine, bq_client)
        return
    if args.only == "model_performance":
        export_model_performance(args.date, args.schema, engine, bq_client)
        return
    if args.only == "lineup_refresh":
        game_ids = _parse_game_ids(args.game_ids)
        if not game_ids:
            raise SystemExit("--only lineup_refresh requires --game-ids (comma-separated)")
        export_lineup_refresh(args.date, args.schema, engine, bq_client, game_ids)
        return

    export(args.date, args.schema, engine, bq_client)
    export_player_props(args.date, args.schema, engine, bq_client)
    export_pitcher_props(args.date, args.schema, engine, bq_client)
    export_daily_trends(args.date, args.schema, engine, bq_client)
    export_daily_edges(args.date, args.schema, engine, bq_client)
    export_standings(args.date, args.schema, engine, bq_client)
    export_standings_projections(args.date, args.schema, engine, bq_client)
    export_transactions(args.date, args.schema, engine, bq_client)
    export_model_performance(args.date, args.schema, engine, bq_client)


if __name__ == "__main__":
    main()
