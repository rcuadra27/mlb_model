"""Active roster + IL eligibility helpers for props inference and edge building."""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

# 40-man status codes for injured/inactive lists (not on active roster).
IL_STATUS_CODES = frozenset({"D7", "D10", "D15", "D60", "BRV", "R60", "FME", "DED"})

_IL_PLACE_RE = re.compile(
    r"placed on the (?:10|15|60)-day injured list|placed on the injured list",
    re.I,
)
_IL_CLEAR_RE = re.compile(
    r"activated from the (?:10|15|60)-day injured list|"
    r"activated from the injured list|reinstated from the injured list",
    re.I,
)


def load_active_player_ids(engine: Engine, snapshot_date: str, schema: str = "public") -> set[int]:
    """Player IDs on MLB active roster for snapshot_date."""
    q = text(f"""
        SELECT player_id
        FROM {schema}.active_rosters
        WHERE snapshot_date = :d
    """)
    with engine.connect() as conn:
        rows = conn.execute(q, {"d": snapshot_date}).fetchall()
    return {int(r[0]) for r in rows if r[0] is not None}


def load_il_player_ids_from_status(engine: Engine, snapshot_date: str, schema: str = "public") -> set[int]:
    """Player IDs with an IL stint on 40-man roster snapshot."""
    q = text(f"""
        SELECT player_id
        FROM {schema}.player_il_status
        WHERE snapshot_date = :d
    """)
    with engine.connect() as conn:
        rows = conn.execute(q, {"d": snapshot_date}).fetchall()
    return {int(r[0]) for r in rows if r[0] is not None}


def load_il_player_ids_from_transactions(
    engine: Engine, as_of_date: str, schema: str = "public"
) -> set[int]:
    """
    Players whose most recent IL-related transaction through as_of_date is a placement
    (not activation/reinstatement).
    """
    q = text(f"""
        SELECT player_id, transaction_date, description, transaction_type, transaction_id
        FROM {schema}.transactions
        WHERE player_id IS NOT NULL
          AND transaction_date <= :d
          AND (
              LOWER(COALESCE(description, '') || ' ' || COALESCE(transaction_type, ''))
              ~* 'injured|injury| il|10-day|15-day|60-day|activated|reinstated'
          )
        ORDER BY player_id, transaction_date DESC, transaction_id DESC
    """)
    with engine.connect() as conn:
        rows = conn.execute(q, {"d": as_of_date}).fetchall()

    on_il: set[int] = set()
    seen: set[int] = set()
    for player_id, _txn_date, description, transaction_type, _txn_id in rows:
        pid = int(player_id)
        if pid in seen:
            continue
        seen.add(pid)
        blob = f"{transaction_type or ''} {description or ''}"
        if _IL_CLEAR_RE.search(blob):
            continue
        if _IL_PLACE_RE.search(blob) or _transaction_text_is_il(blob):
            on_il.add(pid)
    return on_il


def _transaction_text_is_il(text_blob: str) -> bool:
    lower = text_blob.lower()
    if any(x in lower for x in ("activated from", "reinstated from")):
        return False
    return any(
        x in lower
        for x in (
            "injured list",
            "10-day il",
            "15-day il",
            "60-day il",
            " placed on il",
        )
    )


def load_eligible_player_ids(engine: Engine, snapshot_date: str, schema: str = "public") -> set[int]:
    """
    Eligible = on active roster AND not on IL (40-man status or transaction-derived).
    """
    active = load_active_player_ids(engine, snapshot_date, schema)
    if not active:
        return set()
    il_status = load_il_player_ids_from_status(engine, snapshot_date, schema)
    il_txn = load_il_player_ids_from_transactions(engine, snapshot_date, schema)
    ineligible = il_status | il_txn
    return active - ineligible


def filter_player_ids(player_ids: Iterable[int | None], eligible: set[int]) -> list[int]:
    return [int(x) for x in player_ids if x is not None and int(x) in eligible]


def filter_eligible_batters(
    df: pd.DataFrame,
    engine: Engine,
    snapshot_date: str,
    id_col: str = "batter_id",
    schema: str = "public",
) -> pd.DataFrame:
    if df.empty or id_col not in df.columns:
        return df
    eligible = load_eligible_player_ids(engine, snapshot_date, schema)
    if not eligible:
        return df.iloc[0:0].copy()
    out = df[df[id_col].astype(int).isin(eligible)].copy()
    return out


def filter_eligible_pitchers_df(
    rows: list[dict],
    eligible: set[int],
    id_col: str = "pitcher_id",
) -> list[dict]:
    if not rows or not eligible:
        return [] if not eligible and rows else rows
    return [r for r in rows if r.get(id_col) is not None and int(r[id_col]) in eligible]
