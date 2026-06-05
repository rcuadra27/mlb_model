#!/usr/bin/env python3
"""Fetch recent MLB transactions into Postgres for the dashboard Transactions tab."""

from __future__ import annotations

import argparse
import datetime as dt
import os
from typing import Any

import requests
from sqlalchemy import create_engine, text


TRANSACTIONS_URL = "https://statsapi.mlb.com/api/v1/transactions"
MLB_ORG_NAMES = {
    109: "Arizona Diamondbacks",
    110: "Baltimore Orioles",
    111: "Boston Red Sox",
    112: "Chicago Cubs",
    113: "Cincinnati Reds",
    114: "Cleveland Guardians",
    115: "Colorado Rockies",
    116: "Detroit Tigers",
    117: "Houston Astros",
    118: "Kansas City Royals",
    119: "Los Angeles Dodgers",
    120: "Washington Nationals",
    121: "New York Mets",
    133: "Athletics",
    134: "Pittsburgh Pirates",
    135: "San Diego Padres",
    136: "Seattle Mariners",
    137: "San Francisco Giants",
    138: "St. Louis Cardinals",
    139: "Tampa Bay Rays",
    140: "Texas Rangers",
    141: "Toronto Blue Jays",
    142: "Minnesota Twins",
    143: "Philadelphia Phillies",
    144: "Atlanta Braves",
    145: "Chicago White Sox",
    146: "Miami Marlins",
    147: "New York Yankees",
    158: "Milwaukee Brewers",
    108: "Los Angeles Angels",
}
MLB_ORG_BY_NAME = {name.lower(): {"id": tid, "name": name} for tid, name in MLB_ORG_NAMES.items()}

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS public.transactions (
    transaction_id BIGINT NOT NULL,
    transaction_date DATE NOT NULL,
    team_id INTEGER,
    team_name TEXT,
    player_id INTEGER,
    player_name TEXT NOT NULL,
    transaction_type TEXT,
    type_code TEXT,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (transaction_id)
);
CREATE INDEX IF NOT EXISTS idx_transactions_date
    ON public.transactions(transaction_date DESC, team_id, transaction_type);
"""


def normalize_pg_dsn(dsn: str) -> str:
    if dsn.startswith("postgresql+psycopg2://"):
        return "postgresql://" + dsn[len("postgresql+psycopg2://") :]
    if dsn.startswith("postgres+psycopg2://"):
        return "postgres://" + dsn[len("postgres+psycopg2://") :]
    return dsn


def fetch_transactions(start_date: str, end_date: str) -> dict[str, Any]:
    params = {
        "startDate": start_date,
        "endDate": end_date,
        "sportId": 1,
    }
    resp = requests.get(TRANSACTIONS_URL, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _team_from_transaction(txn: dict[str, Any]) -> dict[str, Any]:
    for key in ("team", "toTeam", "fromTeam"):
        team = txn.get(key)
        if isinstance(team, dict) and team.get("id") in MLB_ORG_NAMES:
            return {"id": int(team["id"]), "name": MLB_ORG_NAMES[int(team["id"])]}
    desc = (txn.get("description") or "").lower()
    for org_name, org in sorted(MLB_ORG_BY_NAME.items(), key=lambda item: len(item[0]), reverse=True):
        if org_name in desc:
            return org
    for key in ("team", "toTeam", "fromTeam"):
        team = txn.get(key)
        if isinstance(team, dict) and team.get("id"):
            return team
    return {}


def _transaction_date(txn: dict[str, Any]) -> str | None:
    raw = txn.get("date") or txn.get("effectiveDate") or txn.get("resolutionDate")
    if not raw:
        return None
    return str(raw)[:10]


def _transaction_type(txn: dict[str, Any]) -> str | None:
    return txn.get("typeDesc") or txn.get("type") or txn.get("typeCode")


def parse_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for txn in payload.get("transactions") or []:
        txn_id = txn.get("id")
        txn_date = _transaction_date(txn)
        person = txn.get("person") or {}
        player_name = person.get("fullName") or txn.get("playerName")
        if txn_id is None or not txn_date or not player_name:
            continue

        team = _team_from_transaction(txn)
        rows.append({
            "transaction_id": int(txn_id),
            "transaction_date": txn_date,
            "team_id": int(team["id"]) if team.get("id") is not None else None,
            "team_name": team.get("name"),
            "player_id": int(person["id"]) if person.get("id") is not None else None,
            "player_name": player_name,
            "transaction_type": _transaction_type(txn),
            "type_code": txn.get("typeCode"),
            "description": txn.get("description") or "",
        })
    return rows


def upsert_rows(engine, rows: list[dict[str, Any]], schema: str, start_date: str, end_date: str) -> None:
    with engine.begin() as conn:
        conn.execute(text(CREATE_TABLE_SQL))
        conn.execute(
            text(f"DELETE FROM {schema}.transactions WHERE transaction_date BETWEEN :start_date AND :end_date"),
            {"start_date": start_date, "end_date": end_date},
        )
        if rows:
            conn.execute(
                text(f"""
                    INSERT INTO {schema}.transactions (
                        transaction_id, transaction_date, team_id, team_name, player_id,
                        player_name, transaction_type, type_code, description
                    )
                    VALUES (
                        :transaction_id, :transaction_date, :team_id, :team_name, :player_id,
                        :player_name, :transaction_type, :type_code, :description
                    )
                    ON CONFLICT (transaction_id) DO UPDATE SET
                        transaction_date = EXCLUDED.transaction_date,
                        team_id = EXCLUDED.team_id,
                        team_name = EXCLUDED.team_name,
                        player_id = EXCLUDED.player_id,
                        player_name = EXCLUDED.player_name,
                        transaction_type = EXCLUDED.transaction_type,
                        type_code = EXCLUDED.type_code,
                        description = EXCLUDED.description,
                        created_at = NOW()
                """),
                rows,
            )
    print(f"  Wrote {len(rows)} transactions for {start_date} through {end_date}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--schema", default="public")
    args = ap.parse_args()

    end_date = args.end or args.date
    if args.start:
        start_date = args.start
    else:
        start_date = (dt.date.fromisoformat(end_date) - dt.timedelta(days=max(1, args.days) - 1)).isoformat()

    pg_dsn = os.getenv("PG_DSN")
    if not pg_dsn:
        raise RuntimeError("PG_DSN env var required.")

    engine = create_engine(normalize_pg_dsn(pg_dsn), pool_pre_ping=True)
    print(f"Fetching MLB transactions {start_date} through {end_date}...")
    rows = parse_rows(fetch_transactions(start_date, end_date))
    upsert_rows(engine, rows, args.schema, start_date, end_date)


if __name__ == "__main__":
    main()
