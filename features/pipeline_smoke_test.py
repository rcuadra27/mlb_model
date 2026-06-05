#!/usr/bin/env python3
"""Post-pipeline row-count smoke test (Postgres + BigQuery) for one slate date."""

from __future__ import annotations

import argparse
import json
import os
import sys

from google.cloud import bigquery
from sqlalchemy import create_engine, text

# Minimum rows expected on a normal MLB slate (fewer → alert).
MIN_BQ_ROWS = {
    "bq.daily_trends": 3,
    "bq.daily_edges": 3,
    "bq.standings": 15,
}


def pg_counts(dsn: str, date_str: str) -> dict[str, int]:
    engine = create_engine(dsn, pool_pre_ping=True)
    queries = {
        "pg.games": "SELECT COUNT(*) FROM public.games WHERE game_date = :d",
        "pg.inference_game_predictions": (
            "SELECT COUNT(*) FROM public.inference_game_predictions WHERE game_date = :d"
        ),
        "pg.player_prop_predictions": (
            "SELECT COUNT(*) FROM public.player_prop_predictions WHERE game_date = :d"
        ),
        "pg.pitcher_prop_predictions": (
            "SELECT COUNT(*) FROM public.pitcher_prop_predictions WHERE game_date = :d"
        ),
        "pg.daily_trends": "SELECT COUNT(*) FROM public.daily_trends WHERE trend_date = :d",
        "pg.daily_edges": "SELECT COUNT(*) FROM public.daily_edges WHERE edge_date = :d",
        "pg.standings": "SELECT COUNT(*) FROM public.standings WHERE snapshot_date = :d",
    }
    out: dict[str, int] = {}
    with engine.connect() as conn:
        for label, q in queries.items():
            try:
                out[label] = int(conn.execute(text(q), {"d": date_str}).scalar() or 0)
            except Exception as exc:
                out[label] = -1
                print(f"  {label}: ERROR ({exc})", file=sys.stderr)
    return out


def bq_counts(date_str: str) -> dict[str, int]:
    client = bigquery.Client(project="mlb-model-491223")
    checks = {
        "bq.daily_games": ("game_date", "mlb-model-491223.mlb_model_logs.daily_games"),
        "bq.daily_trends": ("trend_date", "mlb-model-491223.mlb_model_logs.daily_trends"),
        "bq.daily_edges": ("edge_date", "mlb-model-491223.mlb_model_logs.daily_edges"),
        "bq.standings": ("snapshot_date", "mlb-model-491223.mlb_model_logs.standings"),
        "bq.player_prop_predictions": (
            "game_date",
            "mlb-model-491223.mlb_model_logs.player_prop_predictions",
        ),
        "bq.pitcher_prop_predictions": (
            "game_date",
            "mlb-model-491223.mlb_model_logs.pitcher_prop_predictions",
        ),
    }
    out: dict[str, int] = {}
    for label, (date_col, table) in checks.items():
        q = f"SELECT COUNT(*) AS n FROM `{table}` WHERE {date_col} = DATE(@d)"
        job = client.query(
            q,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("d", "STRING", date_str)]
            ),
        )
        out[label] = int(next(job.result()).n)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="Slate date YYYY-MM-DD (Pacific)")
    ap.add_argument(
        "--fail-on-empty",
        action="store_true",
        help="Exit 1 if any critical table has zero rows",
    )
    ap.add_argument(
        "--notify",
        action="store_true",
        help="Send PIPELINE_ALERT notification on failure (uses pipeline_alert.py)",
    )
    ap.add_argument(
        "--json-counts",
        action="store_true",
        help="Print row counts as JSON on stdout (last line)",
    )
    args = ap.parse_args()
    date_str = args.date

    print(f"\n=== Pipeline smoke test for {date_str} (PT slate) ===")

    pg_dsn = os.environ.get("PG_DSN")
    counts: dict[str, int] = {}
    if pg_dsn:
        print("\nPostgres:")
        pg = pg_counts(pg_dsn, date_str)
        counts.update(pg)
        for k, v in sorted(pg.items()):
            flag = " *** EMPTY ***" if v == 0 else (" *** ERROR ***" if v < 0 else "")
            print(f"  {k}: {v}{flag}")
    else:
        print("\nPostgres: skipped (PG_DSN not set)")

    print("\nBigQuery:")
    bq = bq_counts(date_str)
    counts.update(bq)
    for k, v in sorted(bq.items()):
        flag = " *** EMPTY ***" if v == 0 else ""
        print(f"  {k}: {v}{flag}")

    critical = [
        "pg.inference_game_predictions",
        "bq.daily_games",
        "pg.standings",
        "bq.standings",
        "pg.daily_trends",
        "bq.daily_trends",
        "pg.daily_edges",
        "bq.daily_edges",
        "pg.player_prop_predictions",
        "bq.player_prop_predictions",
    ]
    empty = [k for k in critical if counts.get(k) == 0]
    low = [
        k for k, min_n in MIN_BQ_ROWS.items()
        if counts.get(k, 0) >= 0 and counts.get(k, 0) < min_n
    ]
    if empty:
        print(f"\nWARNING: zero-row critical tables: {', '.join(empty)}", file=sys.stderr)
    if low:
        print(f"\nWARNING: below minimum row counts: {', '.join(low)}", file=sys.stderr)
    if empty or low:
        if args.fail_on_empty:
            if args.notify:
                from features.pipeline_alert import notify

                notify(
                    "morning_inference",
                    "[MLB Pipeline] Smoke test failed",
                    f"Smoke test failed for {date_str}. "
                    f"Empty: {', '.join(empty) or 'none'}. "
                    f"Low: {', '.join(low) or 'none'}.",
                    severity="critical",
                )
            return 1
    else:
        print("\nAll critical tables have rows.")
    if args.json_counts:
        print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
