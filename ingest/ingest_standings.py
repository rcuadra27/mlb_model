#!/usr/bin/env python3
"""Fetch current MLB standings into Postgres for the dashboard Standings tab."""

from __future__ import annotations

import argparse
import datetime as dt
import os
from typing import Any

import requests
from sqlalchemy import create_engine, text


STANDINGS_URL = "https://statsapi.mlb.com/api/v1/standings"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS public.standings (
    snapshot_date DATE NOT NULL,
    season INTEGER NOT NULL,
    league_id INTEGER NOT NULL,
    league_name TEXT,
    division_id INTEGER NOT NULL,
    division_name TEXT NOT NULL,
    division_name_short TEXT,
    team_id INTEGER NOT NULL,
    team_name TEXT NOT NULL,
    abbreviation TEXT,
    rank INTEGER,
    wins INTEGER NOT NULL,
    losses INTEGER NOT NULL,
    pct DOUBLE PRECISION,
    games_back TEXT,
    streak TEXT,
    last_10 TEXT,
    run_diff INTEGER,
    runs_scored INTEGER,
    runs_allowed INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (snapshot_date, team_id)
);
CREATE INDEX IF NOT EXISTS idx_standings_snapshot_division
    ON public.standings(snapshot_date, league_id, division_id, rank);
"""


def normalize_pg_dsn(dsn: str) -> str:
    if dsn.startswith("postgresql+psycopg2://"):
        return "postgresql://" + dsn[len("postgresql+psycopg2://") :]
    if dsn.startswith("postgres+psycopg2://"):
        return "postgres://" + dsn[len("postgres+psycopg2://") :]
    return dsn


def fetch_standings(season: int) -> dict[str, Any]:
    params = {
        "leagueId": "103,104",
        "season": str(season),
        "group": "division",
        "hydrate": "division,league,team",
    }
    resp = requests.get(STANDINGS_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _record_value(record: dict[str, Any], record_type: str) -> str | None:
    records = record.get("records") or {}
    if isinstance(records, dict):
        splits = records.get("splitRecords") or []
    else:
        splits = records
    for split in splits:
        if not isinstance(split, dict):
            continue
        if split.get("type") == record_type:
            wins = split.get("wins")
            losses = split.get("losses")
            if wins is not None and losses is not None:
                return f"{wins}-{losses}"
    return None


def _streak_value(record: dict[str, Any]) -> str | None:
    streak = record.get("streak") or {}
    code = streak.get("streakCode")
    if code:
        return str(code)
    st_type = (streak.get("streakType") or "").strip()[:1].upper()
    n = streak.get("streakNumber")
    if st_type and n is not None:
        return f"{st_type}{n}"
    return None


def parse_rows(payload: dict[str, Any], snapshot_date: str, season: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in payload.get("records") or []:
        league = rec.get("league") or {}
        division = rec.get("division") or {}
        league_id = int(league.get("id") or rec.get("leagueId") or 0)
        division_id = int(division.get("id") or rec.get("divisionId") or 0)
        for tr in rec.get("teamRecords") or []:
            team = tr.get("team") or {}
            team_id = team.get("id")
            if team_id is None:
                continue
            wins = int(tr.get("wins") or 0)
            losses = int(tr.get("losses") or 0)
            pct_raw = tr.get("winningPercentage")
            rows.append({
                "snapshot_date": snapshot_date,
                "season": season,
                "league_id": league_id,
                "league_name": league.get("name"),
                "division_id": division_id,
                "division_name": division.get("name") or rec.get("divisionName") or "",
                "division_name_short": division.get("nameShort") or division.get("abbreviation"),
                "team_id": int(team_id),
                "team_name": team.get("name") or "",
                "abbreviation": team.get("abbreviation"),
                "rank": int(tr.get("divisionRank")) if tr.get("divisionRank") else None,
                "wins": wins,
                "losses": losses,
                "pct": float(pct_raw) if pct_raw not in (None, "") else (wins / max(1, wins + losses)),
                "games_back": str(tr.get("gamesBack") or "-"),
                "streak": _streak_value(tr),
                "last_10": _record_value(tr, "lastTen"),
                "run_diff": int(tr.get("runDifferential")) if tr.get("runDifferential") is not None else None,
                "runs_scored": int(tr.get("runsScored")) if tr.get("runsScored") is not None else None,
                "runs_allowed": int(tr.get("runsAllowed")) if tr.get("runsAllowed") is not None else None,
            })
    return rows


def upsert_rows(engine, rows: list[dict[str, Any]], schema: str, snapshot_date: str) -> None:
    with engine.begin() as conn:
        conn.execute(text(CREATE_TABLE_SQL))
        conn.execute(text(f"DELETE FROM {schema}.standings WHERE snapshot_date = :d"), {"d": snapshot_date})
        if rows:
            conn.execute(
                text(f"""
                    INSERT INTO {schema}.standings (
                        snapshot_date, season, league_id, league_name, division_id,
                        division_name, division_name_short, team_id, team_name,
                        abbreviation, rank, wins, losses, pct, games_back, streak,
                        last_10, run_diff, runs_scored, runs_allowed
                    )
                    VALUES (
                        :snapshot_date, :season, :league_id, :league_name, :division_id,
                        :division_name, :division_name_short, :team_id, :team_name,
                        :abbreviation, :rank, :wins, :losses, :pct, :games_back, :streak,
                        :last_10, :run_diff, :runs_scored, :runs_allowed
                    )
                """),
                rows,
            )
    print(f"  Wrote {len(rows)} standings rows for {snapshot_date}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--schema", default="public")
    args = ap.parse_args()

    season = args.season or int(args.date[:4])
    pg_dsn = os.getenv("PG_DSN")
    if not pg_dsn:
        raise RuntimeError("PG_DSN env var required.")

    engine = create_engine(normalize_pg_dsn(pg_dsn), pool_pre_ping=True)
    print(f"Fetching MLB standings for {season}...")
    rows = parse_rows(fetch_standings(season), args.date, season)
    upsert_rows(engine, rows, args.schema, args.date)


if __name__ == "__main__":
    main()
