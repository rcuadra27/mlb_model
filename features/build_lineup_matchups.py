#!/usr/bin/env python3
"""
features/build_lineup_matchups.py

Build lineup strength + pitch-mix matchup features and update public.features_game.

Features written:
- home_lineup_skill
- away_lineup_skill
- home_vs_away_sp_matchup
- away_vs_home_sp_matchup
- lineup_skill_diff
- matchup_diff

Usage:
  export PG_DSN="postgresql+psycopg2://USER:PASS@localhost:5432/mlb_model"
  python features/build_lineup_matchups.py --start 2015-04-01 --end 2025-11-15 --window_days 365 --top_n 9 --batch 5000
"""

import os
import argparse
import datetime as dt
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text


PITCH_COLS_MIX = ["pct_ff","pct_si","pct_fc","pct_sl","pct_cu","pct_ch","pct_sp","pct_fs"]
PITCH_COLS_SKILL = ["skill_ff","skill_si","skill_fc","skill_sl","skill_cu","skill_ch","skill_sp","skill_fs"]


def dot_matchup(pmix: np.ndarray, bskill: np.ndarray) -> float:
    if pmix.shape != bskill.shape:
        return float("nan")
    if np.any(np.isnan(pmix)) or np.any(np.isnan(bskill)):
        return float("nan")
    return float(np.dot(pmix, bskill))


def ensure_columns(engine, schema: str = "public"):
    wanted = {
        "home_lineup_skill": "DOUBLE PRECISION",
        "away_lineup_skill": "DOUBLE PRECISION",
        "home_vs_away_sp_matchup": "DOUBLE PRECISION",
        "away_vs_home_sp_matchup": "DOUBLE PRECISION",
        "lineup_skill_diff": "DOUBLE PRECISION",
        "matchup_diff": "DOUBLE PRECISION",
    }
    with engine.begin() as conn:
        existing = conn.execute(text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = :schema AND table_name = 'features_game'
        """), {"schema": schema}).fetchall()
        existing = {c[0] for c in existing}

        for col, typ in wanted.items():
            if col not in existing:
                conn.execute(text(f'ALTER TABLE {schema}.features_game ADD COLUMN {col} {typ};'))
                print(f"Added column {schema}.features_game.{col}")


def fetch_base(engine, schema: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    sql = text(f"""
        SELECT f.game_id, f.game_date, sp.home_sp_id, sp.away_sp_id
        FROM {schema}.features_game f
        LEFT JOIN {schema}.game_starting_pitchers sp USING (game_id)
        WHERE f.game_date BETWEEN :start AND :end
        ORDER BY f.game_date, f.game_id
    """)
    return pd.read_sql(sql, engine, params={"start": start, "end": end})


def fetch_lineups(engine, schema: str, game_ids: List[int], top_n: int) -> pd.DataFrame:
    sql = text(f"""
        SELECT game_id, is_home, batting_order, player_id
        FROM {schema}.game_lineups
        WHERE game_id = ANY(:gids)
          AND batting_order BETWEEN 1 AND :top_n
        ORDER BY game_id, is_home, batting_order
    """)
    return pd.read_sql(sql, engine, params={"gids": game_ids, "top_n": top_n})


def build_latest_lookup(engine, schema: str, table: str, id_col: str, date_col: str,
                        window_days: int, ids: List[int], min_date: dt.date, max_date: dt.date,
                        cols: List[str]) -> pd.DataFrame:
    """
    Pull all rollups for ids in date range so we can do "latest <= as_of" lookups in pandas.
    """
    sql = text(f"""
        SELECT {id_col} AS entity_id,
               {date_col} AS as_of_date,
               {",".join(cols)}
        FROM {schema}.{table}
        WHERE window_days = :w
          AND {id_col} = ANY(:ids)
          AND {date_col} BETWEEN :min_d AND :max_d
        ORDER BY {id_col}, {date_col}
    """)
    return pd.read_sql(sql, engine, params={"w": window_days, "ids": ids, "min_d": min_date, "max_d": max_date})


def latest_row(df: pd.DataFrame, entity_id: int, asof: dt.date) -> Optional[pd.Series]:
    """
    df must be sorted by (entity_id, as_of_date).
    Returns latest row with as_of_date <= asof.
    """
    sub = df[df["entity_id"] == entity_id]
    if sub.empty:
        return None
    sub = sub[sub["as_of_date"] <= pd.Timestamp(asof)]
    if sub.empty:
        return None
    return sub.iloc[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", default="public")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--window_days", type=int, default=365)
    ap.add_argument("--top_n", type=int, default=9)
    ap.add_argument("--batch", type=int, default=5000)
    args = ap.parse_args()

    pg_dsn = os.getenv("PG_DSN")
    if not pg_dsn:
        raise RuntimeError("PG_DSN env var required")

    engine = create_engine(pg_dsn, pool_pre_ping=True)

    start_d = pd.to_datetime(args.start).date()
    end_d = pd.to_datetime(args.end).date()

    ensure_columns(engine, args.schema)

    base = fetch_base(engine, args.schema, start_d, end_d)
    if base.empty:
        print("No games found in features_game for that range.")
        return

    # We'll process in batches of game_ids to keep memory stable.
    game_ids_all = base["game_id"].astype(int).tolist()

    # Precompute date bounds for rollups (we use game_date-1 and 365d window)
    min_asof = start_d - dt.timedelta(days=366)
    max_asof = end_d

    for i in range(0, len(game_ids_all), args.batch):
        gids = game_ids_all[i:i+args.batch]
        batch_df = base[base["game_id"].isin(gids)].copy()
        batch_df["game_date"] = pd.to_datetime(batch_df["game_date"]).dt.date
        batch_df["asof"] = batch_df["game_date"].apply(lambda d: d - dt.timedelta(days=1))

        lineups = fetch_lineups(engine, args.schema, gids, args.top_n)
        if lineups.empty:
            print(f"[batch {i}] No lineups for these games; skipping.")
            continue

        # Build lineup map
        lineups = lineups.sort_values(["game_id", "is_home", "batting_order"])
        lineup_map: Dict[Tuple[int, bool], List[int]] = {}
        for (gid, is_home), g in lineups.groupby(["game_id", "is_home"]):
            lineup_map[(int(gid), bool(is_home))] = g["player_id"].astype(int).tolist()

        # Collect unique batters and pitchers in this batch
        batters = sorted(set(lineups["player_id"].astype(int).tolist()))
        pitchers = sorted(set(
            batch_df["home_sp_id"].dropna().astype(int).tolist()
            + batch_df["away_sp_id"].dropna().astype(int).tolist()
        ))

        # Pull rollups for these entities across needed date window.
        # Convert as_of_date to timestamp for comparisons.
        batter_roll = build_latest_lookup(
            engine, args.schema,
            table="batter_vs_pitchtype_rolling",
            id_col="batter_id",
            date_col="as_of_date",
            window_days=args.window_days,
            ids=batters,
            min_date=min_asof,
            max_date=max_asof,
            cols=PITCH_COLS_SKILL
        )
        pitcher_roll = build_latest_lookup(
            engine, args.schema,
            table="pitcher_pitchmix_rolling",
            id_col="pitcher_id",
            date_col="as_of_date",
            window_days=args.window_days,
            ids=pitchers,
            min_date=min_asof,
            max_date=max_asof,
            cols=PITCH_COLS_MIX
        )

        if batter_roll.empty or pitcher_roll.empty:
            print(f"[batch {i}] Missing rollups (batter_roll empty={batter_roll.empty}, pitcher_roll empty={pitcher_roll.empty}); skipping.")
            continue

        batter_roll["as_of_date"] = pd.to_datetime(batter_roll["as_of_date"])
        pitcher_roll["as_of_date"] = pd.to_datetime(pitcher_roll["as_of_date"])

        out_rows = []

        for r in batch_df.itertuples(index=False):
            gid = int(r.game_id)
            asof = r.asof

            home_batters = lineup_map.get((gid, True), [])[:args.top_n]
            away_batters = lineup_map.get((gid, False), [])[:args.top_n]

            def avg_skill_vec(b_list: List[int]) -> np.ndarray:
                vecs = []
                for b in b_list:
                    row = latest_row(batter_roll, int(b), asof)
                    if row is None:
                        continue
                    v = row[PITCH_COLS_SKILL].to_numpy(dtype=float)
                    if np.any(np.isnan(v)):
                        continue
                    vecs.append(v)
                if not vecs:
                    return np.full(len(PITCH_COLS_SKILL), np.nan, dtype=float)
                return np.nanmean(np.vstack(vecs), axis=0)

            home_skill_vec = avg_skill_vec(home_batters)
            away_skill_vec = avg_skill_vec(away_batters)

            def mix_vec(pid) -> np.ndarray:
                if pd.isna(pid):
                    return np.full(len(PITCH_COLS_MIX), np.nan, dtype=float)
                row = latest_row(pitcher_roll, int(pid), asof)
                if row is None:
                    return np.full(len(PITCH_COLS_MIX), np.nan, dtype=float)
                v = row[PITCH_COLS_MIX].to_numpy(dtype=float)
                if np.any(np.isnan(v)):
                    return np.full(len(PITCH_COLS_MIX), np.nan, dtype=float)
                return v

            home_sp_mix = mix_vec(r.home_sp_id)
            away_sp_mix = mix_vec(r.away_sp_id)

            home_lineup_skill = float(np.nanmean(home_skill_vec)) if np.any(~np.isnan(home_skill_vec)) else np.nan
            away_lineup_skill = float(np.nanmean(away_skill_vec)) if np.any(~np.isnan(away_skill_vec)) else np.nan

            home_vs_away_sp = dot_matchup(away_sp_mix, home_skill_vec)
            away_vs_home_sp = dot_matchup(home_sp_mix, away_skill_vec)

            out_rows.append({
                "game_id": gid,
                "home_lineup_skill": None if np.isnan(home_lineup_skill) else home_lineup_skill,
                "away_lineup_skill": None if np.isnan(away_lineup_skill) else away_lineup_skill,
                "home_vs_away_sp_matchup": None if np.isnan(home_vs_away_sp) else home_vs_away_sp,
                "away_vs_home_sp_matchup": None if np.isnan(away_vs_home_sp) else away_vs_home_sp,
                "lineup_skill_diff": None if (np.isnan(home_lineup_skill) or np.isnan(away_lineup_skill)) else (home_lineup_skill - away_lineup_skill),
                "matchup_diff": None if (np.isnan(home_vs_away_sp) or np.isnan(away_vs_home_sp)) else (home_vs_away_sp - away_vs_home_sp),
            })

        out = pd.DataFrame(out_rows)

        upd_sql = text(f"""
            UPDATE {args.schema}.features_game
            SET
                home_lineup_skill = :home_lineup_skill,
                away_lineup_skill = :away_lineup_skill,
                home_vs_away_sp_matchup = :home_vs_away_sp_matchup,
                away_vs_home_sp_matchup = :away_vs_home_sp_matchup,
                lineup_skill_diff = :lineup_skill_diff,
                matchup_diff = :matchup_diff
            WHERE game_id = :game_id
        """)

        payload = out.to_dict(orient="records")

        with engine.begin() as conn:
            conn.execute(text("SET LOCAL statement_timeout = '0'"))
            #executemany: SQLAlchemy will send as batch
            conn.execute(upd_sql, payload)

        print(f"[batch {i}] updated {len(payload):,} games")


    print("Done updating matchup features in features_game.")


if __name__ == "__main__":
    main()
