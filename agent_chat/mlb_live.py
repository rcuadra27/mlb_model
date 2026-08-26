"""MLB Stats API live/final status, scores, and schedule (dashboard parity)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from common import _cache_get, _cache_set
from config import (
    MLB_FEED_CACHE_TTL_SECONDS,
    MLB_FEED_URL,
    MLB_SCHEDULE_CACHE_TTL_SECONDS,
    MLB_SCHEDULE_URL,
    _FEED_CACHE,
    _SCHEDULE_CACHE,
)


def _fetch_mlb_game_feed(game_id: int) -> dict:
    key = str(int(game_id))
    cached = _cache_get(_FEED_CACHE, key)
    if cached is not None:
        return cached
    url = MLB_FEED_URL.format(game_id=int(game_id))
    req = urllib.request.Request(url, headers={"User-Agent": "mlb-agent-chat/1"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return {"error": "mlb_feed_http_error", "status": exc.code, "game_id": game_id}
    except Exception as exc:
        return {"error": "mlb_feed_failed", "message": str(exc)[:200], "game_id": game_id}
    return _cache_set(_FEED_CACHE, key, data, MLB_FEED_CACHE_TTL_SECONDS)


def _is_live_status(status: str | None) -> bool:
    s = (status or "").lower()
    if "final" in s or "game over" in s or "completed early" in s:
        return False
    return "progress" in s or "warmup" in s or "delayed" in s or s == "live"


def _is_mlb_game_finished(detailed: str | None, abstract: str | None, coded: str | None) -> bool:
    abs_l = (abstract or "").strip().lower()
    if abs_l == "final":
        return True
    code = (coded or "").strip().upper()
    if code == "F":
        return True
    d = (detailed or "").strip().lower()
    return d in ("final", "game over") or "completed early" in d or "game over" in d


def _is_postponed_or_cancelled(detailed: str | None, abstract: str | None) -> bool:
    abs_l = (abstract or "").strip().lower()
    if "postponed" in abs_l:
        return True
    d = (detailed or "").strip().lower()
    return any(x in d for x in ("postponed", "cancelled", "canceled"))


def _parse_game_status_from_feed(feed: dict) -> dict:
    status = (feed.get("gameData") or {}).get("status") or {}
    abstract = (status.get("abstractGameState") or "").strip()
    detailed = (status.get("detailedState") or "").strip()
    coded = (status.get("codedGameState") or "").strip()
    ls = (feed.get("liveData") or {}).get("linescore") or {}
    current_inning = ls.get("currentInning")
    inning_state = (ls.get("inningState") or "").strip()

    if _is_postponed_or_cancelled(detailed, abstract):
        return {
            "game_status": "postponed",
            "is_final": False,
            "is_live": False,
            "status_detail": detailed or abstract or "Postponed",
        }

    is_final = _is_mlb_game_finished(detailed, abstract, coded)
    is_live = (not is_final) and _is_live_status(detailed or abstract)

    if is_final:
        return {
            "game_status": "final",
            "is_final": True,
            "is_live": False,
            "status_detail": detailed or abstract or "Final",
        }

    if is_live:
        detail = detailed or abstract or "In progress"
        if current_inning and not detailed:
            half = ""
            if inning_state.lower().startswith("top"):
                half = "Top"
            elif inning_state.lower().startswith("bot"):
                half = "Bottom"
            detail = f"{half} {current_inning}".strip() if half else f"Inning {current_inning}"
        return {
            "game_status": "in_progress",
            "is_final": False,
            "is_live": True,
            "status_detail": detail,
            "current_inning": current_inning,
            "inning_state": inning_state or None,
        }

    abs_l = abstract.lower()
    if abs_l in ("preview", "scheduled", "pre-game") or coded in ("P", "S"):
        return {
            "game_status": "scheduled",
            "is_final": False,
            "is_live": False,
            "status_detail": detailed or abstract or "Scheduled",
        }

    return {
        "game_status": "scheduled",
        "is_final": False,
        "is_live": False,
        "status_detail": detailed or abstract or "Not started",
    }


def _parse_schedule_entry(g: dict) -> dict:
    status_obj = g.get("status") or {}
    detailed = (status_obj.get("detailedState") or "").strip()
    abstract = (status_obj.get("abstractGameState") or "").strip()
    coded = (status_obj.get("codedGameState") or "").strip()
    ls = g.get("linescore") or {}
    teams = ls.get("teams") or {}
    away_runs = teams.get("away", {}).get("runs")
    home_runs = teams.get("home", {}).get("runs")

    if _is_postponed_or_cancelled(detailed, abstract):
        game_status = "postponed"
        is_final = False
        is_live = False
    elif _is_mlb_game_finished(detailed, abstract, coded):
        game_status = "final"
        is_final = True
        is_live = False
    elif _is_live_status(detailed or abstract):
        game_status = "in_progress"
        is_final = False
        is_live = True
    else:
        game_status = "scheduled"
        is_final = False
        is_live = False

    return {
        "game_id": int(g.get("gamePk")),
        "game_status": game_status,
        "is_final": is_final,
        "is_live": is_live,
        "status_detail": detailed or abstract or game_status.replace("_", " ").title(),
        "away_runs": int(away_runs) if away_runs is not None else None,
        "home_runs": int(home_runs) if home_runs is not None else None,
    }


def _fetch_mlb_schedule_map(date_str: str) -> dict[int, dict]:
    """MLB schedule + linescore — same hydrate the dashboard useLiveScores uses."""
    cached = _cache_get(_SCHEDULE_CACHE, date_str)
    if cached is not None:
        return cached
    url = MLB_SCHEDULE_URL.format(date=date_str)
    req = urllib.request.Request(url, headers={"User-Agent": "mlb-agent-chat/1"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        return _cache_set(_SCHEDULE_CACHE, date_str, {}, MLB_SCHEDULE_CACHE_TTL_SECONDS)

    out: dict[int, dict] = {}
    for day in data.get("dates") or []:
        for g in day.get("games") or []:
            pk = g.get("gamePk")
            if pk is not None:
                out[int(pk)] = _parse_schedule_entry(g)
    return _cache_set(_SCHEDULE_CACHE, date_str, out, MLB_SCHEDULE_CACHE_TTL_SECONDS)


def _parse_bq_game_status(game_row: dict | None) -> dict:
    raw = ((game_row or {}).get("status") or "").strip()
    s = raw.lower()
    if _is_postponed_or_cancelled(raw, None):
        return {
            "game_status": "postponed",
            "is_final": False,
            "is_live": False,
            "status_detail": raw or "Postponed",
        }
    if "final" in s or s == "game over" or "completed" in s:
        return {
            "game_status": "final",
            "is_final": True,
            "is_live": False,
            "status_detail": raw or "Final",
        }
    if _is_live_status(raw):
        return {
            "game_status": "in_progress",
            "is_final": False,
            "is_live": True,
            "status_detail": raw or "In progress",
        }
    return {
        "game_status": "scheduled",
        "is_final": False,
        "is_live": False,
        "status_detail": raw or "Scheduled",
    }


def _first_int_runs(*sources) -> int | None:
    for src in sources:
        if src is None:
            continue
        try:
            return int(src)
        except (TypeError, ValueError):
            continue
    return None


def _pick_finished_game_runs(mlb_runs, db_runs) -> int | None:
    """Prefer MLB linescore for finals; fall back to BigQuery snapshot."""
    return _first_int_runs(mlb_runs, db_runs)


def _resolve_game_outcome(
    game_row: dict | None,
    feed: dict | None = None,
    schedule_row: dict | None = None,
) -> dict:
    """Merge BQ + MLB schedule + feed into one outcome (dashboard parity)."""
    away_team = (game_row or {}).get("away_team") or "Away"
    home_team = (game_row or {}).get("home_team") or "Home"

    if schedule_row:
        status = {
            "game_status": schedule_row.get("game_status"),
            "is_final": schedule_row.get("is_final"),
            "is_live": schedule_row.get("is_live"),
            "status_detail": schedule_row.get("status_detail"),
        }
    elif feed and not feed.get("error"):
        status = _parse_game_status_from_feed(feed)
    else:
        status = _parse_bq_game_status(game_row)

    bq_status = _parse_bq_game_status(game_row)
    if schedule_row and schedule_row.get("is_final"):
        status = {
            "game_status": "final",
            "is_final": True,
            "is_live": False,
            "status_detail": schedule_row.get("status_detail") or "Final",
        }
    elif bq_status.get("is_final") and not status.get("is_final"):
        status = bq_status

    sched_away = (schedule_row or {}).get("away_runs")
    sched_home = (schedule_row or {}).get("home_runs")
    ls = ((feed or {}).get("liveData") or {}).get("linescore") or {}
    teams = ls.get("teams") or {}
    feed_away = teams.get("away", {}).get("runs")
    feed_home = teams.get("home", {}).get("runs")
    bq_away = (game_row or {}).get("away_runs")
    bq_home = (game_row or {}).get("home_runs")

    if status.get("is_final"):
        away_runs = _pick_finished_game_runs(sched_away, _pick_finished_game_runs(feed_away, bq_away))
        home_runs = _pick_finished_game_runs(sched_home, _pick_finished_game_runs(feed_home, bq_home))
    else:
        away_runs = _first_int_runs(sched_away, feed_away, bq_away)
        home_runs = _first_int_runs(sched_home, feed_home, bq_home)

    score_line = None
    if away_runs is not None and home_runs is not None:
        score_line = f"{away_team} {away_runs}, {home_team} {home_runs}"

    winner = None
    if status.get("is_final") and away_runs is not None and home_runs is not None:
        if away_runs > home_runs:
            winner = away_team
        elif home_runs > away_runs:
            winner = home_team

    return {
        **status,
        "away_team": away_team,
        "home_team": home_team,
        "away_runs": away_runs,
        "home_runs": home_runs,
        "score_line": score_line,
        "winner": winner,
    }


def _enrich_game_with_outcome(game: dict, schedule_map: dict[int, dict]) -> dict:
    gid = game.get("game_id")
    schedule_row = schedule_map.get(int(gid)) if gid is not None else None
    outcome = _resolve_game_outcome(game, schedule_row=schedule_row)
    enriched = dict(game)
    enriched.update({
        "game_status": outcome.get("game_status"),
        "is_final": outcome.get("is_final"),
        "is_live": outcome.get("is_live"),
        "status_detail": outcome.get("status_detail"),
        "away_runs": outcome.get("away_runs"),
        "home_runs": outcome.get("home_runs"),
        "score_line": outcome.get("score_line"),
        "winner": outcome.get("winner"),
    })
    return enriched
