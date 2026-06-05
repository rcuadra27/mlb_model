"""Helpers for confirmed vs pre-lineup (early) inference."""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Engine


def confirmed_lineup_game_ids(
    engine: Engine, schema: str, game_date: str
) -> set[int]:
    """game_ids with both home and away rows in game_lineups."""
    q = text(f"""
        SELECT game_id
        FROM {schema}.game_lineups
        WHERE game_date = :d
        GROUP BY game_id
        HAVING COUNT(DISTINCT is_home) = 2
    """)
    with engine.connect() as conn:
        rows = conn.execute(q, {"d": game_date}).fetchall()
    return {int(r[0]) for r in rows if r[0] is not None}
