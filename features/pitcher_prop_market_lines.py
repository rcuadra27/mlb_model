"""Fetch posted pitcher K/walks/hits/ER lines from The Odds API (median across books)."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import requests

ODDS_EVENTS_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/events"
ODDS_EVENT_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/events/{event_id}/odds"

EDGE_TYPE_TO_MARKET = {
    "k": "pitcher_strikeouts",
    "walks": "pitcher_walks",
    "hits": "pitcher_hits_allowed",
    "er": "pitcher_earned_runs",
}


def _normalize_name(name: str | None) -> str:
    return (name or "").strip().lower()


def _team_names_match(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    x, y = a.strip().lower(), b.strip().lower()
    if x == y or x in y or y in x:
        return True
    x_last, y_last = x.split()[-1], y.split()[-1]
    return x_last == y_last and len(x_last) > 2


def _event_on_date(commence_time: str | None, date_str: str) -> bool:
    if not commence_time:
        return False
    try:
        dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
        pt_date = dt.astimezone(ZoneInfo("America/Los_Angeles")).date()
        return str(pt_date) == date_str
    except (TypeError, ValueError):
        return False


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    mid = len(vals) // 2
    if len(vals) % 2:
        return float(vals[mid])
    return float((vals[mid - 1] + vals[mid]) / 2.0)


def _parse_event_player_lines(event_payload: dict) -> dict[str, dict[str, float]]:
    """Return {pitcher_name_lower: {k|walks|hits|er: line}} for one event."""
    market_to_edge = {v: k for k, v in EDGE_TYPE_TO_MARKET.items()}
    collected: dict[str, dict[str, list[float]]] = {}

    for book in event_payload.get("bookmakers") or []:
        for market in book.get("markets") or []:
            edge_type = market_to_edge.get(market.get("key"))
            if not edge_type:
                continue
            for outcome in market.get("outcomes") or []:
                if (outcome.get("name") or "").lower() != "over":
                    continue
                player = _normalize_name(outcome.get("description"))
                point = outcome.get("point")
                if not player or point is None:
                    continue
                try:
                    line = float(point)
                except (TypeError, ValueError):
                    continue
                collected.setdefault(player, {}).setdefault(edge_type, []).append(line)

    out: dict[str, dict[str, float]] = {}
    for player, by_type in collected.items():
        lines = {}
        for edge_type, vals in by_type.items():
            med = _median(vals)
            if med is not None:
                lines[edge_type] = med
        if lines:
            out[player] = lines
    return out


def _match_event_id(
    events: list[dict],
    away_team: str | None,
    home_team: str | None,
    date_str: str,
) -> str | None:
    for ev in events:
        if not _event_on_date(ev.get("commence_time"), date_str):
            continue
        if _team_names_match(ev.get("away_team"), away_team) and _team_names_match(ev.get("home_team"), home_team):
            return ev.get("id")
    return None


def fetch_pitcher_counting_stat_market_lines(
    date_str: str,
    games_by_id: dict[int, dict[str, Any]],
) -> dict[tuple[int, str], dict[str, float]]:
    """
    Return {(game_id, pitcher_name_lower): {k|walks|hits|er: posted_line}}.
    Best-effort; returns {} when ODDS_API_KEY is missing or the API fails.
    """
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key or not games_by_id:
        return {}

    try:
        resp = requests.get(
            ODDS_EVENTS_URL,
            params={"apiKey": api_key},
            timeout=15,
        )
        resp.raise_for_status()
        events = resp.json()
    except requests.RequestException as exc:
        print(f"  Warning: Odds API events fetch failed ({exc}); counting-stat market lines skipped.")
        return {}

    markets = ",".join(EDGE_TYPE_TO_MARKET.values())
    out: dict[tuple[int, str], dict[str, float]] = {}

    for game_id, game in games_by_id.items():
        event_id = _match_event_id(events, game.get("away_team"), game.get("home_team"), date_str)
        if not event_id:
            continue
        try:
            r2 = requests.get(
                ODDS_EVENT_URL.format(event_id=event_id),
                params={
                    "apiKey": api_key,
                    "regions": "us",
                    "markets": markets,
                    "oddsFormat": "american",
                },
                timeout=15,
            )
            r2.raise_for_status()
            player_lines = _parse_event_player_lines(r2.json())
        except requests.RequestException as exc:
            print(f"  Warning: Odds API props fetch failed for game {game_id} ({exc})")
            continue

        for player, lines in player_lines.items():
            out[(int(game_id), player)] = lines

    if out:
        print(f"  Loaded posted counting-stat lines for {len(out)} pitcher(s) from Odds API")
    return out
