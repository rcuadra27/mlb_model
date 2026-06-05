#!/usr/bin/env python3
"""Project final records and playoff odds from model win probabilities."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import random
from collections import defaultdict
from typing import Any

import requests
from sqlalchemy import create_engine, text


N_SIMULATIONS = 10_000
PLAYOFF_TEAMS_PER_LEAGUE = 6
SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
PYTHAG_EXPONENT = 1.83
TRUE_TALENT_REGRESSION_GAMES = 60

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS public.standings_projections (
    snapshot_date DATE NOT NULL,
    season INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    team_name TEXT NOT NULL,
    projected_wins DOUBLE PRECISION NOT NULL,
    projected_losses DOUBLE PRECISION NOT NULL,
    projected_record TEXT NOT NULL,
    playoff_odds DOUBLE PRECISION NOT NULL,
    remaining_games INTEGER NOT NULL,
    simulations INTEGER NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (snapshot_date, team_id)
);
CREATE INDEX IF NOT EXISTS idx_standings_projections_snapshot
    ON public.standings_projections(snapshot_date, playoff_odds DESC);
"""


def normalize_pg_dsn(dsn: str) -> str:
    if dsn.startswith("postgresql+psycopg2://"):
        return "postgresql://" + dsn[len("postgresql+psycopg2://") :]
    if dsn.startswith("postgres+psycopg2://"):
        return "postgres://" + dsn[len("postgres+psycopg2://") :]
    return dsn


def safe_prob(v: Any) -> float:
    try:
        p = float(v)
    except (TypeError, ValueError):
        return 0.5
    if p > 1.0:
        p /= 100.0
    return max(0.05, min(0.95, p))


def fetch_standings(engine, schema: str, date_str: str) -> list[dict[str, Any]]:
    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT
                snapshot_date,
                season,
                league_id,
                division_id,
                team_id,
                team_name,
                wins,
                losses,
                runs_scored,
                runs_allowed
            FROM {schema}.standings
            WHERE snapshot_date = :d
        """), {"d": date_str}).mappings().all()
    return [dict(r) for r in rows]


def _season_end_date(season: int) -> str:
    return dt.date(season, 10, 15).isoformat()


def fetch_model_probs(engine, schema: str, date_str: str) -> dict[int, float]:
    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            WITH latest AS (
                SELECT
                    p.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY p.game_id
                        ORDER BY p.as_of_ts DESC NULLS LAST
                    ) AS rn
                FROM {schema}.inference_game_predictions p
                WHERE p.game_date >= CAST(:d AS DATE)
            )
            SELECT game_id, p_home_win_poisson
            FROM latest
            WHERE rn = 1
        """), {"d": date_str}).mappings().all()
    return {int(r["game_id"]): safe_prob(r.get("p_home_win_poisson")) for r in rows}


def pythagorean_win_pct(runs_scored: Any, runs_allowed: Any, exponent: float = PYTHAG_EXPONENT) -> float:
    try:
        rs = float(runs_scored)
        ra = float(runs_allowed)
    except (TypeError, ValueError):
        return 0.5
    if rs <= 0 or ra <= 0:
        return 0.5
    rs_exp = rs ** exponent
    ra_exp = ra ** exponent
    return max(0.05, min(0.95, rs_exp / (rs_exp + ra_exp)))


def regressed_true_talent(
    runs_scored: Any,
    runs_allowed: Any,
    games_played: Any,
    regression_games: int = TRUE_TALENT_REGRESSION_GAMES,
) -> float:
    pyth = pythagorean_win_pct(runs_scored, runs_allowed)
    try:
        gp = max(0.0, float(games_played))
    except (TypeError, ValueError):
        gp = 0.0
    regressed = ((pyth * gp) + (0.500 * regression_games)) / max(1.0, gp + regression_games)
    return max(0.05, min(0.95, regressed))


def log5_prob(a: float, b: float) -> float:
    denom = a + b - (2 * a * b)
    if denom <= 0:
        return 0.5
    return (a - (a * b)) / denom


def team_strengths_from_standings(standings: list[dict[str, Any]]) -> dict[int, float]:
    strengths = {}
    for row in standings:
        games_played = int(row.get("wins") or 0) + int(row.get("losses") or 0)
        strengths[int(row["team_id"])] = regressed_true_talent(
            row.get("runs_scored"),
            row.get("runs_allowed"),
            games_played,
        )
    return strengths


def fallback_home_prob(home_team_id: int, away_team_id: int, strengths: dict[int, float]) -> float:
    home_strength = strengths.get(home_team_id, 0.5)
    away_strength = strengths.get(away_team_id, 0.5)
    neutral_home = log5_prob(home_strength, away_strength)
    return max(0.05, min(0.95, neutral_home + 0.035))


def fetch_remaining_games(
    engine,
    schema: str,
    date_str: str,
    season: int,
    standings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    model_probs = fetch_model_probs(engine, schema, date_str)
    strengths = team_strengths_from_standings(standings)
    params = {
        "sportId": 1,
        "startDate": date_str,
        "endDate": _season_end_date(season),
        "season": season,
        "gameTypes": "R",
    }
    resp = requests.get(SCHEDULE_URL, params=params, timeout=60)
    resp.raise_for_status()
    rows = []
    for day in resp.json().get("dates") or []:
        game_date = day.get("date")
        for g in day.get("games") or []:
            status = (g.get("status") or {}).get("detailedState") or ""
            if any(x in status.lower() for x in ("final", "completed", "postponed", "cancelled")):
                continue
            teams = g.get("teams") or {}
            home = (teams.get("home") or {}).get("team") or {}
            away = (teams.get("away") or {}).get("team") or {}
            game_id = int(g["gamePk"])
            if not home.get("id") or not away.get("id"):
                continue
            home_team_id = int(home["id"])
            away_team_id = int(away["id"])
            model_prob = model_probs.get(game_id)
            p_home_win = model_prob if model_prob is not None else fallback_home_prob(
                home_team_id,
                away_team_id,
                strengths,
            )
            rows.append({
                "game_id": game_id,
                "game_date": game_date,
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "home_team_name": home.get("name"),
                "away_team_name": away.get("name"),
                "p_home_win": p_home_win,
                "has_model_prob": model_prob is not None,
                "prob_source": "model" if model_prob is not None else "pythag_log5",
                "home_true_talent": strengths.get(home_team_id),
                "away_true_talent": strengths.get(away_team_id),
            })
    return rows


def fetch_remaining_games_from_db(engine, schema: str, date_str: str, season: int) -> list[dict[str, Any]]:
    with engine.connect() as conn:
        rows = conn.execute(text(f"""
        WITH latest AS (
            SELECT
                p.*,
                ROW_NUMBER() OVER (
                    PARTITION BY p.game_id
                    ORDER BY p.as_of_ts DESC NULLS LAST
                ) AS rn
            FROM {schema}.inference_game_predictions p
            WHERE p.game_date >= :d::date
        )
        SELECT
            g.game_id,
            g.game_date,
            g.home_team_id,
            g.away_team_id,
            g.home_team_name,
            g.away_team_name,
            COALESCE(l.p_home_win_poisson, 0.5) AS p_home_win,
            COALESCE(l.p_away_win_poisson, 0.5) AS p_away_win
        FROM {schema}.games g
        LEFT JOIN latest l ON l.game_id = g.game_id AND l.rn = 1
        WHERE g.season = :season
          AND g.game_date >= CAST(:d AS DATE)
          AND g.home_runs IS NULL
          AND g.away_runs IS NULL
          AND LOWER(COALESCE(g.status, '')) NOT LIKE 'final%%'
          AND LOWER(COALESCE(g.status, '')) NOT LIKE 'completed%%'
          AND LOWER(COALESCE(g.status, '')) NOT LIKE 'postponed%%'
          AND LOWER(COALESCE(g.status, '')) NOT LIKE 'cancelled%%'
        ORDER BY g.game_date, g.game_id
    """), {"d": date_str, "season": season}).mappings().all()
    return [dict(r) for r in rows]


def playoff_teams(wins_by_team: dict[int, float], standings_by_team: dict[int, dict[str, Any]]) -> set[int]:
    qualified: set[int] = set()
    by_league_division: dict[tuple[int, int], list[int]] = defaultdict(list)
    by_league: dict[int, list[int]] = defaultdict(list)

    for tid, row in standings_by_team.items():
        by_league_division[(int(row["league_id"]), int(row["division_id"]))].append(tid)
        by_league[int(row["league_id"])].append(tid)

    for teams in by_league_division.values():
        winner = max(teams, key=lambda tid: (wins_by_team.get(tid, 0), random.random()))
        qualified.add(winner)

    for league_id, teams in by_league.items():
        remaining = [tid for tid in teams if tid not in qualified]
        remaining.sort(key=lambda tid: (wins_by_team.get(tid, 0), random.random()), reverse=True)
        league_qualified = [tid for tid in qualified if int(standings_by_team[tid]["league_id"]) == league_id]
        slots = max(0, PLAYOFF_TEAMS_PER_LEAGUE - len(league_qualified))
        qualified.update(remaining[:slots])

    return qualified


def build_projection_rows(
    standings: list[dict[str, Any]],
    games: list[dict[str, Any]],
    date_str: str,
    season: int,
    simulations: int,
) -> list[dict[str, Any]]:
    by_team = {int(r["team_id"]): r for r in standings}
    projected_extra = {tid: 0.0 for tid in by_team}
    remaining_count = {tid: 0 for tid in by_team}
    game_inputs = []

    for g in games:
        home = int(g["home_team_id"])
        away = int(g["away_team_id"])
        if home not in by_team or away not in by_team:
            continue
        p_home = safe_prob(g.get("p_home_win"))
        raw_away = g.get("p_away_win")
        if raw_away is not None:
            p_away = safe_prob(raw_away)
        else:
            p_away = 1.0 - p_home
        if raw_away is not None and abs((p_home + p_away) - 1.0) > 0.05:
            total = p_home + p_away
            p_home = p_home / total if total else 0.5
        p_home = max(0.05, min(0.95, p_home))
        projected_extra[home] += p_home
        projected_extra[away] += 1.0 - p_home
        remaining_count[home] += 1
        remaining_count[away] += 1
        game_inputs.append((home, away, p_home))

    playoff_hits = {tid: 0 for tid in by_team}
    rng = random.Random(f"{date_str}-{season}-hot-corner")
    old_state = random.getstate()
    random.setstate(rng.getstate())
    try:
        for _ in range(simulations):
            wins = {tid: float(r["wins"]) for tid, r in by_team.items()}
            for home, away, p_home in game_inputs:
                if random.random() < p_home:
                    wins[home] += 1
                else:
                    wins[away] += 1
            for tid in playoff_teams(wins, by_team):
                playoff_hits[tid] += 1
    finally:
        random.setstate(old_state)

    rows = []
    for tid, row in by_team.items():
        cur_w = int(row["wins"])
        cur_l = int(row["losses"])
        rem = remaining_count.get(tid, 0)
        proj_w = cur_w + projected_extra.get(tid, 0.0)
        proj_l = cur_l + (rem - projected_extra.get(tid, 0.0))
        rec_w = int(round(proj_w))
        rec_l = int(round(proj_l))
        rows.append({
            "snapshot_date": date_str,
            "season": season,
            "team_id": tid,
            "team_name": row["team_name"],
            "projected_wins": round(proj_w, 2),
            "projected_losses": round(proj_l, 2),
            "projected_record": f"{rec_w}-{rec_l}",
            "playoff_odds": round(playoff_hits[tid] / simulations, 4) if simulations else 0.0,
            "remaining_games": rem,
            "simulations": simulations,
        })
    return rows


def diagnostic_game_rows(
    standings: list[dict[str, Any]],
    games: list[dict[str, Any]],
    team_query: str,
    limit: int,
) -> list[dict[str, Any]]:
    by_team = {int(r["team_id"]): r for r in standings}
    team_query_l = team_query.lower()
    matches = [
        (tid, r) for tid, r in by_team.items()
        if team_query_l in str(r.get("team_name") or "").lower()
    ]
    if not matches:
        raise RuntimeError(f"No standings team matched {team_query!r}")
    team_id, team_row = matches[0]
    print(
        f"DIAG team={team_row['team_name']} team_id={team_id} "
        f"current_record={int(team_row['wins'])}-{int(team_row['losses'])}"
    )

    remaining = []
    for g in games:
        home = int(g["home_team_id"])
        away = int(g["away_team_id"])
        if team_id not in (home, away):
            continue
        raw_home = safe_prob(g.get("p_home_win"))
        raw_away_value = g.get("p_away_win")
        raw_away = safe_prob(raw_away_value) if raw_away_value is not None else 1.0 - raw_home
        effective_home = raw_home
        normalized = False
        if raw_away_value is not None and abs((raw_home + raw_away) - 1.0) > 0.05:
            total = raw_home + raw_away
            effective_home = raw_home / total if total else 0.5
            normalized = True
        effective_home = max(0.05, min(0.95, effective_home))
        team_raw = raw_home if team_id == home else 1.0 - raw_home
        team_effective = effective_home if team_id == home else 1.0 - effective_home
        remaining.append({
            "date": g.get("game_date"),
            "game_id": g.get("game_id"),
            "matchup": f"{g.get('away_team_name')} @ {g.get('home_team_name')}",
            "home": team_id == home,
            "source": g.get("prob_source") or ("model" if g.get("has_model_prob") else "pythag_log5"),
            "raw_home": raw_home,
            "raw_away_value": raw_away_value,
            "effective_home": effective_home,
            "team_raw": team_raw,
            "team_effective": team_effective,
            "normalized": normalized,
            "home_true_talent": g.get("home_true_talent"),
            "away_true_talent": g.get("away_true_talent"),
        })

    print(f"DIAG remaining_games={len(remaining)}")
    print("DIAG playoff_rule=3 division winners + top 3 non-division-winners per league")
    print("DIAG current_records_carried_forward=yes")
    print("DIAG remaining_schedule_source=MLB Stats API schedule endpoint")
    print(f"DIAG next_{limit}_games:")
    for i, r in enumerate(remaining[:limit], start=1):
        side = "home" if r["home"] else "away"
        print(
            "DIAG_GAME "
            f"{i} date={r['date']} game_id={r['game_id']} side={side} "
            f"source={r['source']} normalized={r['normalized']} "
            f"team_raw={r['team_raw']:.3f} team_effective={r['team_effective']:.3f} "
            f"home_true_talent={r['home_true_talent'] if r['home_true_talent'] is not None else 'n/a'} "
            f"away_true_talent={r['away_true_talent'] if r['away_true_talent'] is not None else 'n/a'} "
            f"raw_home={r['raw_home']:.3f} raw_away_field={r['raw_away_value']} "
            f"matchup={r['matchup']}"
        )
    return remaining


def upsert_rows(engine, schema: str, date_str: str, rows: list[dict[str, Any]]) -> None:
    with engine.begin() as conn:
        conn.execute(text(CREATE_TABLE_SQL))
        conn.execute(text(f"DELETE FROM {schema}.standings_projections WHERE snapshot_date = :d"), {"d": date_str})
        if rows:
            conn.execute(
                text(f"""
                    INSERT INTO {schema}.standings_projections (
                        snapshot_date, season, team_id, team_name, projected_wins,
                        projected_losses, projected_record, playoff_odds,
                        remaining_games, simulations
                    )
                    VALUES (
                        :snapshot_date, :season, :team_id, :team_name, :projected_wins,
                        :projected_losses, :projected_record, :playoff_odds,
                        :remaining_games, :simulations
                    )
                """),
                rows,
            )
    print(f"  Wrote {len(rows)} standings projection rows for {date_str}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--schema", default="public")
    ap.add_argument("--simulations", type=int, default=N_SIMULATIONS)
    ap.add_argument("--diagnose-team", help="Print simulation inputs for a team, e.g. Dodgers")
    ap.add_argument("--diagnose-games", type=int, default=10)
    ap.add_argument("--diagnose-only", action="store_true", help="Print diagnostics and skip writing projections")
    args = ap.parse_args()

    season = args.season or int(args.date[:4])
    pg_dsn = os.getenv("PG_DSN")
    if not pg_dsn:
        raise RuntimeError("PG_DSN env var required.")

    engine = create_engine(normalize_pg_dsn(pg_dsn), pool_pre_ping=True)
    standings = fetch_standings(engine, args.schema, args.date)
    if not standings:
        raise RuntimeError(f"No standings rows found for {args.date}; run ingest_standings.py first.")
    games = fetch_remaining_games(engine, args.schema, args.date, season, standings)
    if args.diagnose_team:
        diagnostic_game_rows(standings, games, args.diagnose_team, args.diagnose_games)
        if args.diagnose_only:
            return
    rows = build_projection_rows(standings, games, args.date, season, args.simulations)
    upsert_rows(engine, args.schema, args.date, rows)


if __name__ == "__main__":
    main()
