#!/usr/bin/env python3
"""Fetch MLB active rosters and IL status into Postgres for props/edges eligibility."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import time
from typing import Any

import requests
from sqlalchemy import create_engine, text

from features.active_roster import IL_STATUS_CODES

ROSTER_ACTIVE_URL = "https://statsapi.mlb.com/api/v1/teams/{team_id}/roster/active"
ROSTER_40MAN_URL = "https://statsapi.mlb.com/api/v1/teams/{team_id}/roster/40Man"

MLB_TEAM_IDS = (
    108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
    133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146,
    147, 158,
)

CREATE_ACTIVE_SQL = """
CREATE TABLE IF NOT EXISTS public.active_rosters (
    snapshot_date DATE NOT NULL,
    team_id INTEGER NOT NULL,
    player_id INTEGER NOT NULL,
    player_name TEXT NOT NULL,
    position_abbr TEXT,
    status_code TEXT,
    status_desc TEXT,
    fetched_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (snapshot_date, team_id, player_id)
);
CREATE INDEX IF NOT EXISTS idx_active_rosters_date_player
    ON public.active_rosters(snapshot_date, player_id);
"""

CREATE_IL_SQL = """
CREATE TABLE IF NOT EXISTS public.player_il_status (
    snapshot_date DATE NOT NULL,
    player_id INTEGER NOT NULL,
    team_id INTEGER,
    player_name TEXT,
    status_code TEXT NOT NULL,
    status_desc TEXT,
    fetched_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (snapshot_date, player_id)
);
CREATE INDEX IF NOT EXISTS idx_player_il_status_date
    ON public.player_il_status(snapshot_date);
"""


def normalize_pg_dsn(dsn: str) -> str:
    if dsn.startswith("postgresql+psycopg2://"):
        return "postgresql://" + dsn[len("postgresql+psycopg2://") :]
    if dsn.startswith("postgres+psycopg2://"):
        return "postgres://" + dsn[len("postgres+psycopg2://") :]
    return dsn


def _get_team_ids(engine, schema: str) -> list[int]:
    with engine.connect() as conn:
        rows = conn.execute(
            text(f"SELECT mlb_team_id FROM {schema}.teams ORDER BY mlb_team_id")
        ).fetchall()
    if rows:
        return [int(r[0]) for r in rows]
    return list(MLB_TEAM_IDS)


def _fetch_roster(url: str, team_id: int, session: requests.Session) -> list[dict[str, Any]]:
    resp = session.get(url.format(team_id=team_id), timeout=30)
    resp.raise_for_status()
    return resp.json().get("roster") or []


def _parse_active(snapshot_date: str, team_id: int, roster: list[dict[str, Any]]) -> list[dict]:
    rows = []
    for entry in roster:
        person = entry.get("person") or {}
        pid = person.get("id")
        name = person.get("fullName")
        if pid is None or not name:
            continue
        status = entry.get("status") or {}
        pos = entry.get("position") or {}
        rows.append({
            "snapshot_date": snapshot_date,
            "team_id": team_id,
            "player_id": int(pid),
            "player_name": name,
            "position_abbr": pos.get("abbreviation"),
            "status_code": status.get("code"),
            "status_desc": status.get("description"),
        })
    return rows


def _parse_il(snapshot_date: str, team_id: int, roster: list[dict[str, Any]]) -> list[dict]:
    rows = []
    for entry in roster:
        person = entry.get("person") or {}
        pid = person.get("id")
        name = person.get("fullName")
        status = entry.get("status") or {}
        code = status.get("code") or ""
        if pid is None or code not in IL_STATUS_CODES:
            continue
        rows.append({
            "snapshot_date": snapshot_date,
            "player_id": int(pid),
            "team_id": team_id,
            "player_name": name,
            "status_code": code,
            "status_desc": status.get("description"),
        })
    return rows


def fetch_all_rosters(
    team_ids: list[int], snapshot_date: str, sleep: float = 0.15
) -> tuple[list[dict], list[dict]]:
    session = requests.Session()
    session.headers.update({"User-Agent": "mlb-model-active-roster/1"})
    active_rows: list[dict] = []
    il_rows: list[dict] = []
    for i, team_id in enumerate(team_ids):
        active = _fetch_roster(ROSTER_ACTIVE_URL, team_id, session)
        active_rows.extend(_parse_active(snapshot_date, team_id, active))
        forty = _fetch_roster(ROSTER_40MAN_URL, team_id, session)
        il_rows.extend(_parse_il(snapshot_date, team_id, forty))
        if sleep and i + 1 < len(team_ids):
            time.sleep(sleep)
    return active_rows, il_rows


def upsert(
    engine,
    schema: str,
    snapshot_date: str,
    active_rows: list[dict],
    il_rows: list[dict],
) -> None:
    for row in active_rows:
        row["snapshot_date"] = snapshot_date
    for row in il_rows:
        row["snapshot_date"] = snapshot_date

    with engine.begin() as conn:
        conn.execute(text(CREATE_ACTIVE_SQL))
        conn.execute(text(CREATE_IL_SQL))
        conn.execute(
            text(f"DELETE FROM {schema}.active_rosters WHERE snapshot_date = :d"),
            {"d": snapshot_date},
        )
        conn.execute(
            text(f"DELETE FROM {schema}.player_il_status WHERE snapshot_date = :d"),
            {"d": snapshot_date},
        )
        if active_rows:
            conn.execute(
                text(f"""
                    INSERT INTO {schema}.active_rosters (
                        snapshot_date, team_id, player_id, player_name,
                        position_abbr, status_code, status_desc
                    ) VALUES (
                        :snapshot_date, :team_id, :player_id, :player_name,
                        :position_abbr, :status_code, :status_desc
                    )
                """),
                active_rows,
            )
        if il_rows:
            conn.execute(
                text(f"""
                    INSERT INTO {schema}.player_il_status (
                        snapshot_date, player_id, team_id, player_name,
                        status_code, status_desc
                    ) VALUES (
                        :snapshot_date, :player_id, :team_id, :player_name,
                        :status_code, :status_desc
                    )
                """),
                il_rows,
            )
    print(
        f"  Wrote {len(active_rows)} active roster rows and "
        f"{len(il_rows)} IL status rows for {snapshot_date}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    ap.add_argument("--schema", default="public")
    ap.add_argument("--sleep", type=float, default=0.15, help="Pause between team API calls")
    args = ap.parse_args()

    pg_dsn = os.getenv("PG_DSN")
    if not pg_dsn:
        raise RuntimeError("PG_DSN env var required.")

    engine = create_engine(normalize_pg_dsn(pg_dsn), pool_pre_ping=True)
    team_ids = _get_team_ids(engine, args.schema)
    print(f"Fetching active rosters for {len(team_ids)} teams (date={args.date})...")
    active_rows, il_rows = fetch_all_rosters(team_ids, args.date, sleep=args.sleep)
    upsert(engine, args.schema, args.date, active_rows, il_rows)


if __name__ == "__main__":
    main()
