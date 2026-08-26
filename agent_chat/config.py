"""Shared agent config, clients, and caches."""

from __future__ import annotations

import datetime as dt
import os
from zoneinfo import ZoneInfo

from anthropic import Anthropic
from google.cloud import bigquery
from sqlalchemy import create_engine

MODEL = os.environ.get("MODEL", "claude-haiku-4-5")
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")
DAILY_MSG_LIMIT = int(os.environ.get("DAILY_MSG_LIMIT", "40"))
MAX_TOOL_ROUNDS = int(os.environ.get("MAX_TOOL_ROUNDS", "4"))
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "500"))
BQ_TABLE = "mlb-model-491223.mlb_model_logs.daily_games"
PREDICTION_CACHE_TTL_SECONDS = int(os.environ.get("PREDICTION_CACHE_TTL_SECONDS", "300"))

OU_PRED_LINE_GAP = 0.5
V10_LAUNCH_DATE = "2026-05-28"

_INPUT_USD_PER_MTOK = float(os.environ.get("AGENT_CHAT_INPUT_USD_PER_MTOK", "1"))
_OUTPUT_USD_PER_MTOK = float(os.environ.get("AGENT_CHAT_OUTPUT_USD_PER_MTOK", "5"))

_TZ_PT = ZoneInfo("America/Los_Angeles")

_BQ = bigquery.Client()
_GAMES_CACHE: dict[str, tuple[float, dict]] = {}
_PROPS_CACHE: dict[int, tuple[float, dict]] = {}
_TOP_PROPS_CACHE: dict[str, tuple[float, dict]] = {}
_SLATE_BATTERS_CACHE: dict[str, tuple[float, list]] = {}
_PLAYER_PROP_CACHE: dict[str, tuple[float, dict]] = {}
_FEED_CACHE: dict[str, tuple[float, dict]] = {}
_SCHEDULE_CACHE: dict[str, tuple[float, dict]] = {}
MLB_FEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live"
MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}&gameTypes=R&hydrate=linescore"
MLB_FEED_CACHE_TTL_SECONDS = 60
MLB_SCHEDULE_CACHE_TTL_SECONDS = 60
_ANTHROPIC: Anthropic | None = None
_PG_ENGINE = None


def _today_pacific_iso() -> str:
    return dt.datetime.now(_TZ_PT).date().isoformat()


def _anthropic() -> Anthropic:
    global _ANTHROPIC
    if _ANTHROPIC is None:
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        _ANTHROPIC = Anthropic(api_key=key)
    return _ANTHROPIC


def _pg():
    global _PG_ENGINE
    if _PG_ENGINE is None:
        dsn = os.environ.get("PG_DSN", "").strip()
        if not dsn:
            raise RuntimeError("PG_DSN not set")
        _PG_ENGINE = create_engine(dsn, pool_pre_ping=True, pool_size=1, max_overflow=0)
    return _PG_ENGINE
