"""
train_props_v1.py — Player prop probability model (expanded feature set).

Trains three binary classifiers per batter per game:
  - recorded_hit:    batter gets >= 1 hit
  - recorded_2plus:  batter gets >= 2 hits
  - recorded_hr:     batter hits a home run

Base features (always included):
  batter_xwoba_season      season-to-date xwOBA
  batter_k_rate_season     season-to-date K rate
  batter_hr_rate_season    season-to-date HR/PA
  batter_hit_rate_30d      rolling 30-day hit rate
  matchup_score            batter skill vs pitcher pitch mix
  sp_xwoba_against         opposing SP quality (90-day)
  platoon_advantage        handedness matchup

Optional features (gated on AUC improvement):
  batter_barrel_rate_30d   barrel rate last 30 days (best power predictor)
  batter_hard_hit_rate_30d hard hit rate (exit velo >= 95) last 30 days
  batter_xwoba_30d         rolling 30-day xwOBA (recent expected quality)
  batter_iso_season        isolated power season-to-date
  sp_hr_rate_season        opposing SP HR allowed rate this season
  sp_k_rate_season         opposing SP K rate this season
  sp_fastball_pct          opposing SP fastball usage (from pitchmix_rolling)
  park_hr_factor           park HR factor
  batting_order            lineup position (more PA at top)
  umpire_k_boost           umpire strikeout tendency

Usage:
  PG_DSN=... python models/train_props_v1.py
  PG_DSN=... python models/train_props_v1.py --extra-feature batter_barrel_rate_30d
  PG_DSN=... python models/train_props_v1.py --all-features  (test everything)
  PG_DSN=... python models/train_props_v1.py --forward-select (greedy forward selection)
"""
import argparse
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
import warnings
warnings.filterwarnings("ignore")

# ── Feature definitions ───────────────────────────────────────────────────────

BASE_FEATURES = [
    "batter_xwoba_season",
    "batter_k_rate_season",
    "batter_hr_rate_season",
    "batter_hit_rate_30d",
    "matchup_score",
    "sp_xwoba_against",
    "platoon_advantage",
]

OPTIONAL_FEATURES = {
    "batter_barrel_rate_30d":   ["batter_barrel_rate_30d"],
    "batter_hard_hit_rate_30d": ["batter_hard_hit_rate_30d"],
    "batter_xwoba_30d":         ["batter_xwoba_30d"],
    "batter_iso_season":        ["batter_iso_season"],
    "sp_hr_rate_season":        ["sp_hr_rate_season"],
    "sp_k_rate_season":         ["sp_k_rate_season"],
    "sp_fastball_pct":          ["sp_fastball_pct"],
    "park_hr_factor":           ["park_hr_factor"],
    "batting_order":            ["batting_order"],
    "umpire_k_boost":           ["umpire_k_boost"],
}

ALL_OPTIONAL = list(OPTIONAL_FEATURES.keys())

# ── SQL queries ───────────────────────────────────────────────────────────────

TARGETS_QUERY = """
SELECT
    sp.batter                                          AS batter_id,
    sp.game_pk                                         AS game_id,
    sp.game_date,
    EXTRACT(YEAR FROM sp.game_date)::int               AS season,
    MAX(CASE WHEN sp.events IN ('single','double','triple','home_run')
             THEN 1 ELSE 0 END)                        AS recorded_hit,
    CASE WHEN SUM(CASE WHEN sp.events IN ('single','double','triple','home_run')
                       THEN 1 ELSE 0 END) >= 2
         THEN 1 ELSE 0 END                             AS recorded_2plus,
    MAX(CASE WHEN sp.events = 'home_run'
             THEN 1 ELSE 0 END)                        AS recorded_hr,
    MAX(CASE WHEN sp.events = 'strikeout'
             THEN 1 ELSE 0 END)                        AS recorded_k,
    CASE WHEN SUM(
        CASE sp.events
            WHEN 'single'   THEN 1
            WHEN 'double'   THEN 2
            WHEN 'triple'   THEN 3
            WHEN 'home_run' THEN 4
            ELSE 0
        END) >= 2 THEN 1 ELSE 0 END                    AS recorded_2plus_bases,
    MAX(CASE WHEN sp.events IN ('walk','hit_by_pitch')
             THEN 1 ELSE 0 END)                        AS recorded_walk,
    COUNT(DISTINCT sp.at_bat_number)                   AS n_pa,
    SUM(CASE WHEN sp.events IN ('single','double','triple','home_run')
             THEN 1 ELSE 0 END)                        AS n_hits
FROM public.statcast_pitches sp
WHERE sp.game_date BETWEEN :start_date AND :end_date
  AND sp.batter IS NOT NULL
  AND sp.game_pk IS NOT NULL
GROUP BY sp.batter, sp.game_pk, sp.game_date
HAVING COUNT(DISTINCT sp.at_bat_number) >= 1
ORDER BY sp.game_date, sp.game_pk, sp.batter;
"""

# Season-to-date batter stats with rolling windows
BATTER_STATS_QUERY = """
WITH batter_game_stats AS (
    SELECT
        batter                                              AS batter_id,
        game_pk                                             AS game_id,
        game_date,
        EXTRACT(YEAR FROM game_date)::int                   AS season,
        COUNT(DISTINCT at_bat_number)                       AS pa,
        SUM(CASE WHEN events IN ('single','double','triple','home_run')
                 THEN 1 ELSE 0 END)                         AS hits,
        SUM(CASE WHEN events = 'home_run' THEN 1 ELSE 0 END) AS hrs,
        SUM(CASE WHEN events IN ('double','triple','home_run')
                 THEN 1 ELSE 0 END)                         AS xbh,
        SUM(CASE WHEN events = 'strikeout' THEN 1 ELSE 0 END) AS ks,
        -- xwOBA (only on balls in play / plate appearances with woba_denom)
        AVG(CASE WHEN woba_denom > 0
                 THEN estimated_woba_using_speedangle END)   AS game_xwoba,
        -- Batted ball quality (only non-null launch speeds = balls in play)
        AVG(launch_speed)                                   AS avg_exit_velo,
        -- Hard hit: exit velo >= 95
        SUM(CASE WHEN launch_speed >= 95 THEN 1 ELSE 0 END) AS hard_hits,
        COUNT(launch_speed)                                  AS balls_in_play,
        -- Barrel: exit velo >= 98 AND launch angle 26-30, sliding scale for higher velo
        SUM(CASE
            WHEN launch_speed >= 98
             AND launch_angle BETWEEN 8 AND 50
             AND (
                 (launch_speed >= 116) OR
                 (launch_speed >= 110 AND launch_angle BETWEEN 18 AND 42) OR
                 (launch_speed >= 105 AND launch_angle BETWEEN 20 AND 38) OR
                 (launch_speed >= 100 AND launch_angle BETWEEN 22 AND 36) OR
                 (launch_speed >= 98  AND launch_angle BETWEEN 26 AND 30)
             )
            THEN 1 ELSE 0 END)                              AS barrels
    FROM public.statcast_pitches
    WHERE batter IS NOT NULL
      AND game_pk IS NOT NULL
      AND game_date BETWEEN :start_date AND :end_date
    GROUP BY batter, game_pk, game_date
),
batter_cumulative AS (
    SELECT
        batter_id, game_id, game_date, season,
        -- Season cumulative (before this game)
        SUM(pa)     OVER w_season AS sd_pa,
        SUM(hits)   OVER w_season AS sd_hits,
        SUM(hrs)    OVER w_season AS sd_hrs,
        SUM(xbh)    OVER w_season AS sd_xbh,
        SUM(ks)     OVER w_season AS sd_ks,
        AVG(game_xwoba) OVER w_season AS sd_xwoba,
        -- 30-day rolling (before this game) — single ORDER BY for RANGE
        SUM(hits)          OVER w_30d AS roll_hits_30d,
        SUM(pa)            OVER w_30d AS roll_pa_30d,
        SUM(hard_hits)     OVER w_30d AS roll_hard_hits_30d,
        SUM(balls_in_play) OVER w_30d AS roll_bip_30d,
        SUM(barrels)       OVER w_30d AS roll_barrels_30d,
        AVG(game_xwoba)    OVER w_30d AS roll_xwoba_30d
    FROM batter_game_stats
    WINDOW
        w_season AS (
            PARTITION BY batter_id, season
            ORDER BY game_date, game_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ),
        w_30d AS (
            PARTITION BY batter_id
            ORDER BY game_date
            RANGE BETWEEN INTERVAL '30 days' PRECEDING
                      AND INTERVAL '1 day'  PRECEDING
        )
)
SELECT
    batter_id, game_id, game_date, season,
    -- Season-to-date (Bayesian shrinkage toward league average, prior=100 PA)
    (COALESCE(sd_hits, 0) + 100 * 0.245) / (COALESCE(sd_pa, 0) + 100) AS batter_hit_rate_season,
    (COALESCE(sd_hrs,  0) + 100 * 0.036) / (COALESCE(sd_pa, 0) + 100) AS batter_hr_rate_season,
    (COALESCE(sd_ks,   0) + 100 * 0.230) / (COALESCE(sd_pa, 0) + 100) AS batter_k_rate_season,
    COALESCE(sd_xwoba, 0.320)                                           AS batter_xwoba_season,
    -- ISO season-to-date: (XBH proxy) / PA, shrunk
    (COALESCE(sd_xbh, 0) + 100 * 0.080) / (COALESCE(sd_pa, 0) + 100)  AS batter_iso_season,
    -- 30-day rolling
    CASE WHEN COALESCE(roll_pa_30d, 0) >= 10
         THEN roll_hits_30d::float / roll_pa_30d
         ELSE 0.245 END                                                  AS batter_hit_rate_30d,
    CASE WHEN COALESCE(roll_bip_30d, 0) >= 10
         THEN roll_hard_hits_30d::float / roll_bip_30d
         ELSE 0.242 END                                                  AS batter_hard_hit_rate_30d,
    CASE WHEN COALESCE(roll_bip_30d, 0) >= 10
         THEN roll_barrels_30d::float / roll_bip_30d
         ELSE 0.072 END                                                  AS batter_barrel_rate_30d,
    COALESCE(roll_xwoba_30d, 0.320)                                     AS batter_xwoba_30d,
    COALESCE(sd_pa, 0)                                                   AS batter_sd_pa
FROM batter_cumulative;
"""

# SP season stats from statcast
SP_STATS_QUERY = """
WITH sp_game_stats AS (
    SELECT
        pitcher                                             AS pitcher_id,
        game_pk                                             AS game_id,
        game_date,
        SUM(CASE WHEN events = 'home_run'    THEN 1 ELSE 0 END) AS hrs_allowed,
        SUM(CASE WHEN events = 'strikeout'   THEN 1 ELSE 0 END) AS ks,
        COUNT(DISTINCT at_bat_number)                           AS bf
    FROM public.statcast_pitches
    WHERE game_date BETWEEN :start_date AND :end_date
      AND pitcher IS NOT NULL
      AND events IS NOT NULL
    GROUP BY pitcher, game_pk, game_date
),
sp_cumulative AS (
    SELECT
        pitcher_id, game_id, game_date,
        EXTRACT(YEAR FROM game_date)::int AS season,
        SUM(hrs_allowed) OVER w_season AS sd_hrs,
        SUM(ks)          OVER w_season AS sd_ks,
        SUM(bf)          OVER w_season AS sd_bf
    FROM sp_game_stats
    WINDOW w_season AS (
        PARTITION BY pitcher_id, EXTRACT(YEAR FROM game_date)::int
        ORDER BY game_date, game_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    )
)
SELECT
    pitcher_id, game_id, game_date, season,
    -- HR rate allowed (shrunk toward league avg ~3.5%)
    (COALESCE(sd_hrs, 0) + 200 * 0.035) / (COALESCE(sd_bf, 0) + 200) AS sp_hr_rate_season,
    -- K rate allowed (shrunk toward league avg ~23%)
    (COALESCE(sd_ks,  0) + 200 * 0.230) / (COALESCE(sd_bf, 0) + 200) AS sp_k_rate_season
FROM sp_cumulative;
"""

MATCHUP_QUERY = """
SELECT
    gl.game_id,
    gl.player_id                                        AS batter_id,
    gl.batting_order,
    gl.is_home,
    gl.bats                                             AS batter_hand,
    CASE WHEN gl.is_home THEN gsp.away_sp_id
         ELSE gsp.home_sp_id END                        AS sp_id,
    CASE WHEN gl.is_home THEN fg.away_sp_xwoba_against_90
         ELSE fg.home_sp_xwoba_against_90 END           AS sp_xwoba_against,
    COALESCE(fg.park_runs_factor_blended,
             fg.park_runs_factor, 1.0)                  AS park_hr_factor,
    fg.umpire_k_rate_boost                              AS umpire_k_boost
FROM public.game_lineups gl
JOIN public.game_starting_pitchers gsp ON gsp.game_id = gl.game_id
JOIN public.features_game fg ON fg.game_id = gl.game_id
WHERE gl.game_date BETWEEN :start_date AND :end_date;
"""

BATTER_PITCHTYPE_QUERY = """
SELECT batter_id, as_of_date,
    COALESCE(skill_ff, 0.320) AS skill_ff,
    COALESCE(skill_si, 0.320) AS skill_si,
    COALESCE(skill_fc, 0.320) AS skill_fc,
    COALESCE(skill_sl, 0.320) AS skill_sl,
    COALESCE(skill_cu, 0.320) AS skill_cu,
    COALESCE(skill_ch, 0.320) AS skill_ch,
    COALESCE(skill_sp, 0.320) AS skill_sp,
    COALESCE(skill_fs, 0.320) AS skill_fs
FROM public.batter_vs_pitchtype_rolling
WHERE window_days = 365
  AND as_of_date BETWEEN :start_date AND :end_date;
"""

PITCHER_PITCHMIX_QUERY = """
SELECT pitcher_id, as_of_date,
    COALESCE(pct_ff, 0) AS pct_ff,
    COALESCE(pct_si, 0) AS pct_si,
    COALESCE(pct_fc, 0) AS pct_fc,
    COALESCE(pct_sl, 0) AS pct_sl,
    COALESCE(pct_cu, 0) AS pct_cu,
    COALESCE(pct_ch, 0) AS pct_ch,
    COALESCE(pct_sp, 0) AS pct_sp,
    COALESCE(pct_fs, 0) AS pct_fs
FROM public.pitcher_pitchmix_rolling
WHERE window_days = 365
  AND as_of_date BETWEEN :start_date AND :end_date;
"""

PITCHER_HAND_QUERY = """
SELECT pitcher AS pitcher_id,
       MODE() WITHIN GROUP (ORDER BY p_throws) AS p_throws
FROM public.statcast_pitches
WHERE p_throws IS NOT NULL
GROUP BY pitcher;
"""

PITCH_TYPES = ['ff', 'si', 'fc', 'sl', 'cu', 'ch', 'sp', 'fs']


# ── Data loading ──────────────────────────────────────────────────────────────

def load_table(engine, query, params, label):
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params=params)
    print(f"  {label}: {len(df):,} rows")
    return df


# ── Matchup score ─────────────────────────────────────────────────────────────

def _asof_sort(frame: pd.DataFrame, by: str, on: str) -> pd.DataFrame:
    """Normalize keys and sort for merge_asof(on=...) within a single entity group."""
    out = frame.copy()
    out[on] = pd.to_datetime(out[on], utc=False).dt.normalize()
    out[by] = pd.to_numeric(out[by], errors='coerce')
    out = out.dropna(subset=[by, on])
    out[by] = out[by].astype('int64')
    return out.sort_values(on, kind='mergesort').reset_index(drop=True)


def _merge_asof_by_group(
    left: pd.DataFrame,
    right: pd.DataFrame,
    by: str,
    on: str = 'game_date_dt',
    direction: str = 'backward',
) -> pd.DataFrame:
    """
    merge_asof per entity group (no `by=` — avoids pandas global sort bug).
    Pre-groups right side once for speed on ~400k rows.
    """
    right_groups = {
        int(k): _asof_sort(grp, by, on).drop(columns=[by])
        for k, grp in right.groupby(by, sort=False)
    }
    parts = []
    for key, left_grp in left.groupby(by, sort=False):
        key = int(key)
        rg = right_groups.get(key)
        ls = _asof_sort(left_grp, by, on)
        if rg is None or rg.empty:
            parts.append(ls)
            continue
        parts.append(pd.merge_asof(ls, rg, on=on, direction=direction))
    return pd.concat(parts, ignore_index=True)


def build_matchup_score(df, batter_pt_df, pitcher_mix_df):
    """
    Vectorized matchup score: batter skill vs pitcher pitch mix.
    Uses as_of_date nearest but <= game_date for each entity.
    """
    DEFAULT = 0.320
    skill_cols = [f'skill_{pt}' for pt in PITCH_TYPES]
    pct_cols = [f'pct_{pt}' for pt in PITCH_TYPES]

    df = df.copy().reset_index(drop=True)
    df['_row_id'] = np.arange(len(df), dtype=np.int64)

    # ── Batter skill as-of lookup ──
    left_b = df[['_row_id', 'batter_id', 'game_date']].rename(
        columns={'game_date': 'game_date_dt'}
    )
    right_b = batter_pt_df.copy()
    right_b['game_date_dt'] = right_b['as_of_date']
    right_b = (
        right_b[['batter_id', 'game_date_dt'] + skill_cols]
        .drop_duplicates(subset=['batter_id', 'game_date_dt'], keep='last')
    )
    batter_merged = _merge_asof_by_group(left_b, right_b, by='batter_id')

    # ── Pitcher mix as-of lookup ──
    has_sp = df['sp_id'].notna()
    pitcher_merged = pd.DataFrame({'_row_id': pd.Series(dtype='int64')})

    if has_sp.any():
        left_p = (
            df.loc[has_sp, ['_row_id', 'sp_id', 'game_date']]
            .rename(columns={'sp_id': 'pitcher_id', 'game_date': 'game_date_dt'})
        )
        right_p = pitcher_mix_df.copy()
        right_p['game_date_dt'] = right_p['as_of_date']
        right_p = (
            right_p[['pitcher_id', 'game_date_dt'] + pct_cols]
            .drop_duplicates(subset=['pitcher_id', 'game_date_dt'], keep='last')
        )
        pitcher_merged = _merge_asof_by_group(left_p, right_p, by='pitcher_id')

    # ── Join back on row id (many batters share one game_id) ──
    b = (
        batter_merged.set_index('_row_id')[skill_cols]
        .reindex(df['_row_id'])
        .fillna(DEFAULT)
    )
    p = (
        pitcher_merged.set_index('_row_id')[pct_cols]
        .reindex(df['_row_id'])
        .fillna(0.0)
        if not pitcher_merged.empty and '_row_id' in pitcher_merged.columns
        else pd.DataFrame(0.0, index=df.index, columns=pct_cols)
    )

    total_weight = sum(p[c] for c in pct_cols)
    weighted = pd.Series(0.0, index=df.index)
    for pt in PITCH_TYPES:
        weighted = weighted + p[f'pct_{pt}'] * b[f'skill_{pt}']

    score = pd.Series(DEFAULT, index=df.index, dtype=float)
    has_weight = total_weight > 0.05
    score.loc[has_weight] = (weighted / total_weight).loc[has_weight]
    df['matchup_score'] = score.values
    df['sp_fastball_pct'] = (p['pct_ff'] + p['pct_si']).values

    return df.drop(columns=['_row_id'], errors='ignore')


# ── Metrics ───────────────────────────────────────────────────────────────────

def evaluate_classifier(label, y_true, y_pred, target_name, verbose=True):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan
    brier = brier_score_loss(y_true, y_pred)
    naive = float(y_true.mean() * (1 - y_true.mean()))

    if verbose:
        print(f"\n  {'─'*65}")
        print(f"  {label} — {target_name}  (N={len(y_true):,})")
        print(f"  {'─'*65}")
        print(f"  Base rate: {y_true.mean()*100:.2f}%  "
              f"AUC: {auc:.4f}  Brier: {brier:.4f} (naive={naive:.4f})")
        print(f"  Pred range: [{y_pred.min()*100:.1f}%, {y_pred.max()*100:.1f}%]  "
              f"Mean: {y_pred.mean()*100:.2f}%")

    return {'auc': float(auc), 'brier': float(brier),
            'naive_brier': naive, 'n': int(len(y_true)),
            'base_rate': float(y_true.mean())}


def train_one(X_tr, y_tr, X_v, y_v, X_te, y_te, target):
    """Train logistic, tune C on val AUC, evaluate on test."""
    best_C, best_auc = 1.0, 0.0
    for C in [0.01, 0.1, 1.0, 10.0]:
        m = LogisticRegression(C=C, max_iter=1000, solver='lbfgs')
        m.fit(X_tr, y_tr)
        v = roc_auc_score(y_v, m.predict_proba(X_v)[:, 1])
        if v > best_auc:
            best_auc, best_C = v, C
    model = LogisticRegression(C=best_C, max_iter=1000, solver='lbfgs')
    model.fit(X_tr, y_tr)
    p_te = model.predict_proba(X_te)[:, 1]
    r = evaluate_classifier(f"C={best_C}", y_te, p_te, target)
    return model, p_te, r


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='artifacts/')
    ap.add_argument('--train_end_season', type=int, default=2022)
    ap.add_argument('--val_season', type=int, default=2023)
    ap.add_argument('--test_season', type=int, default=2024)
    ap.add_argument('--earliest_season', type=int, default=2015)
    ap.add_argument('--extra-feature', dest='extra_features',
                    action='append', default=[],
                    choices=ALL_OPTIONAL)
    ap.add_argument('--all-features', action='store_true',
                    help='Include all optional features')
    ap.add_argument('--forward-select', action='store_true',
                    help='Greedy forward selection on val AUC (avg across 3 targets)')
    ap.add_argument('--min-pa', type=int, default=2)
    args = ap.parse_args()

    if args.all_features:
        args.extra_features = ALL_OPTIONAL

    pg_dsn = os.environ.get('PG_DSN')
    if not pg_dsn:
        print("ERROR: PG_DSN not set", file=sys.stderr); sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    engine = create_engine(pg_dsn)

    start_date = f'{args.earliest_season}-01-01'
    end_date   = f'{args.test_season}-12-31'
    params = {'start_date': start_date, 'end_date': end_date}

    print(f"\n{'='*70}")
    print(f"Props v1 training")
    print(f"{'='*70}")
    print(f"Mode: {'forward-select' if args.forward_select else 'all-features' if args.all_features else 'specified'}")
    print(f"Extra features: {args.extra_features or 'none'}")

    # ── Load targets ──
    print(f"\n[1/7] Loading targets...")
    targets = load_table(engine, TARGETS_QUERY, params, "batter-game outcomes")
    targets = targets[targets['n_pa'] >= args.min_pa].copy()
    print(f"  After min_pa={args.min_pa}: {len(targets):,} rows")
    print(f"  Base rates — "
          f"hit:{targets['recorded_hit'].mean()*100:.1f}% "
          f"2+:{targets['recorded_2plus'].mean()*100:.1f}% "
          f"hr:{targets['recorded_hr'].mean()*100:.1f}% "
          f"k:{targets['recorded_k'].mean()*100:.1f}% "
          f"2+TB:{targets['recorded_2plus_bases'].mean()*100:.1f}% "
          f"walk:{targets['recorded_walk'].mean()*100:.1f}%")

    # ── Load batter stats ──
    print(f"\n[2/7] Loading batter season + rolling stats...")
    batter_stats = load_table(engine, BATTER_STATS_QUERY, params, "batter stats")

    # ── Load SP stats from statcast ──
    print(f"\n[3/7] Loading SP season stats...")
    sp_stats = load_table(engine, SP_STATS_QUERY, params, "SP stats")

    # ── Load matchup context ──
    print(f"\n[4/7] Loading matchup context...")
    matchup = load_table(engine, MATCHUP_QUERY, params, "matchup context")
    print(f"  bats coverage: {matchup['batter_hand'].notna().mean()*100:.1f}%")

    # ── Load pitch type tables ──
    print(f"\n[5/7] Loading pitch type skill/mix tables...")
    batter_pt  = load_table(engine, BATTER_PITCHTYPE_QUERY, params, "batter pitch skills")
    pitcher_mix = load_table(engine, PITCHER_PITCHMIX_QUERY, params, "pitcher pitch mix")

    # Pitcher hand
    with engine.connect() as conn:
        pitcher_hand_df = pd.read_sql(text(PITCHER_HAND_QUERY), conn)
    pitcher_hand = dict(zip(pitcher_hand_df['pitcher_id'].astype(int),
                            pitcher_hand_df['p_throws']))
    print(f"  Pitcher hand: {len(pitcher_hand):,} pitchers")

    # ── Merge everything ──
    print(f"\n[6/7] Merging and computing features...")

    df = targets.merge(
        batter_stats[[
            'batter_id', 'game_id',
            'batter_xwoba_season', 'batter_k_rate_season',
            'batter_hr_rate_season', 'batter_hit_rate_30d',
            'batter_iso_season', 'batter_hard_hit_rate_30d',
            'batter_barrel_rate_30d', 'batter_xwoba_30d', 'batter_sd_pa',
        ]],
        on=['batter_id', 'game_id'], how='left'
    ).merge(
        matchup[[
            'game_id', 'batter_id', 'sp_id', 'sp_xwoba_against',
            'batting_order', 'park_hr_factor', 'batter_hand', 'umpire_k_boost',
        ]],
        on=['game_id', 'batter_id'], how='left'
    )

    # Fill batter stat defaults
    df['batter_xwoba_season']      = df['batter_xwoba_season'].fillna(0.320)
    df['batter_k_rate_season']     = df['batter_k_rate_season'].fillna(0.230)
    df['batter_hr_rate_season']    = df['batter_hr_rate_season'].fillna(0.036)
    df['batter_hit_rate_30d']      = df['batter_hit_rate_30d'].fillna(0.245)
    df['batter_iso_season']        = df['batter_iso_season'].fillna(0.080)
    df['batter_hard_hit_rate_30d'] = df['batter_hard_hit_rate_30d'].fillna(0.242)
    df['batter_barrel_rate_30d']   = df['batter_barrel_rate_30d'].fillna(0.072)
    df['batter_xwoba_30d']         = df['batter_xwoba_30d'].fillna(0.320)
    df['sp_xwoba_against']         = df['sp_xwoba_against'].fillna(0.320)
    df['park_hr_factor']           = df['park_hr_factor'].fillna(1.0)
    df['batting_order']            = df['batting_order'].fillna(5.0)
    df['umpire_k_boost']           = df['umpire_k_boost'].fillna(0.0)

    # Fill batter_hand from statcast if missing (game_lineups.bats may be empty)
    null_bh = df['batter_hand'].isna()
    if null_bh.sum() > 0:
        print(f"  batter_hand missing for {null_bh.sum():,} rows — filling from statcast stand")
        with engine.connect() as conn:
            bh_df = pd.read_sql(text(
                "SELECT batter AS batter_id, MODE() WITHIN GROUP "
                "(ORDER BY stand) AS stand FROM public.statcast_pitches "
                "WHERE stand IS NOT NULL GROUP BY batter"
            ), conn)
        bh_map = dict(zip(bh_df['batter_id'].astype(int), bh_df['stand']))
        df.loc[null_bh, 'batter_hand'] = (
            df.loc[null_bh, 'batter_id'].map(
                lambda x: bh_map.get(int(x)) if pd.notna(x) else None))
        filled = df['batter_hand'].notna().sum() - (~null_bh).sum()
        print(f"  Filled {filled:,} batter_hand values from statcast")

    # Platoon advantage
    def get_platoon(row):
        bh = row.get('batter_hand')
        sp = row.get('sp_id')
        if pd.isna(bh) or pd.isna(sp): return 0
        ph = pitcher_hand.get(int(sp))
        if ph is None: return 0
        return 1 if bh != ph else -1

    df['platoon_advantage'] = df.apply(get_platoon, axis=1)
    plat = df['platoon_advantage'].value_counts()
    print(f"  Platoon: adv={plat.get(1,0):,} dis={plat.get(-1,0):,} unk={plat.get(0,0):,}")

    # Vectorized matchup score + fastball pct
    print(f"  Computing matchup scores (vectorized)...")
    df = build_matchup_score(df, batter_pt, pitcher_mix)
    print(f"  Matchup: mean={df['matchup_score'].mean():.3f} std={df['matchup_score'].std():.3f}")

    # Merge SP stats (join on sp_id + game_id — need date-based lookup)
    # SP stats are cumulative per pitcher per game — merge on game_id + pitcher_id
    df_has_sp = df['sp_id'].notna()
    if df_has_sp.sum() > 0:
        sp_sub = sp_stats.copy()
        # Match by pitcher_id and game_id
        sp_sub = sp_sub.rename(columns={'pitcher_id': 'sp_id', 'game_id': 'sp_game_id'})
        # For each batter row, find the SP stat row matching that game's SP
        sp_lookup = sp_sub.set_index('sp_game_id')[['sp_id','sp_hr_rate_season','sp_k_rate_season']]
        # Join on the game_id of the SP appearance — same game_id
        df = df.merge(
            sp_stats.rename(columns={'pitcher_id': 'sp_id_match',
                                     'game_id': 'sp_game_match'})[
                ['sp_id_match', 'sp_game_match', 'sp_hr_rate_season', 'sp_k_rate_season']],
            left_on=['game_id', 'sp_id'],
            right_on=['sp_game_match', 'sp_id_match'],
            how='left'
        )
        df['sp_hr_rate_season'] = df['sp_hr_rate_season'].fillna(0.035)
        df['sp_k_rate_season']  = df['sp_k_rate_season'].fillna(0.230)
    else:
        df['sp_hr_rate_season'] = 0.035
        df['sp_k_rate_season']  = 0.230

    df = df.dropna(subset=BASE_FEATURES).copy()
    print(f"  Final rows after dropna: {len(df):,}")

    train = df[df['season'] <= args.train_end_season]
    val   = df[df['season'] == args.val_season]
    test  = df[df['season'] == args.test_season]
    print(f"  Train:{len(train):,}  Val:{len(val):,}  Test:{len(test):,}")

    TARGETS = [
        'recorded_hit', 'recorded_2plus', 'recorded_hr',
        'recorded_k', 'recorded_2plus_bases', 'recorded_walk',
    ]

    # ── Training ──
    print(f"\n[7/7] Training...")

    def run_feature_set(feature_cols, label=""):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(train[feature_cols])
        X_v  = scaler.transform(val[feature_cols])
        X_te = scaler.transform(test[feature_cols])
        results = {}
        models = {}
        for tgt in TARGETS:
            m, p, r = train_one(
                X_tr, train[tgt].values,
                X_v,  val[tgt].values,
                X_te, test[tgt].values,
                tgt
            )
            results[tgt] = r
            models[tgt] = m
        return scaler, models, results

    if args.forward_select:
        print("\n  === FORWARD SELECTION ===")
        # Start with base features
        current_features = list(BASE_FEATURES)
        _, _, base_results = run_feature_set(current_features, "base")
        base_avg_auc = np.mean([base_results[t]['auc'] for t in TARGETS])
        print(f"\n  Base AUC (avg): {base_avg_auc:.4f}  features: {current_features}")

        remaining = list(ALL_OPTIONAL)
        while remaining:
            best_feat, best_auc, best_results = None, base_avg_auc, None
            for feat in remaining:
                candidate = current_features + OPTIONAL_FEATURES[feat]
                try:
                    _, _, r = run_feature_set(candidate)
                    avg = np.mean([r[t]['auc'] for t in TARGETS])
                    print(f"    + {feat:<30} avg_AUC={avg:.4f} "
                          f"(hit={r['recorded_hit']['auc']:.4f} "
                          f"2+={r['recorded_2plus']['auc']:.4f} "
                          f"hr={r['recorded_hr']['auc']:.4f})")
                    if avg > best_auc:
                        best_auc, best_feat, best_results = avg, feat, r
                except Exception as e:
                    print(f"    + {feat}: ERROR {e}")

            if best_feat:
                current_features += OPTIONAL_FEATURES[best_feat]
                remaining.remove(best_feat)
                base_avg_auc = best_auc
                print(f"\n  ✓ Added {best_feat} → avg AUC {best_auc:.4f}")
                print(f"    Current: {current_features}")
            else:
                print(f"\n  No more improvements. Stopping.")
                break

        print(f"\n  === FINAL FEATURE SET ({len(current_features)} features) ===")
        print(f"  {current_features}")
        final_features = current_features
    else:
        # Add specified extra features
        extra_cols = [c for ef in args.extra_features
                      for c in OPTIONAL_FEATURES.get(ef, [])]
        final_features = BASE_FEATURES + extra_cols

    # Final model with chosen features
    print(f"\n  === FINAL MODEL — {len(final_features)} features ===")
    scaler, models, results = run_feature_set(final_features)

    print(f"\n  Coefficients:")
    for tgt in TARGETS:
        print(f"\n  {tgt}:")
        coefs = sorted(zip(final_features, models[tgt].coef_[0]),
                       key=lambda x: abs(x[1]), reverse=True)
        for f, c in coefs:
            print(f"    {f:<35} {c:>+8.4f}")

    print(f"\n{'='*70}")
    print(f"  SUMMARY — 2024 OOS holdout")
    print(f"{'='*70}")
    print(f"  {'Target':<20} {'Base%':>7} {'AUC':>8} {'Brier':>8} {'NaiveBrier':>12}")
    print(f"  {'-'*60}")
    for tgt, r in results.items():
        beats = "✓" if r['brier'] < r['naive_brier'] else "✗"
        print(f"  {tgt:<20} {r['base_rate']*100:>6.1f}% "
              f"{r['auc']:>8.4f} {r['brier']:>8.4f} "
              f"{r['naive_brier']:>12.4f} {beats}")

    # Base comparison
    print(f"\n  Base (7 feat) reference AUCs: hit=0.5548  2+=0.5685  hr=0.6025  k=new  2+TB=new  walk=new")

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    stem = 'props_v1_expanded' if (args.extra_features or args.forward_select or args.all_features) else 'props_v1'
    bundle = {
        'scaler': scaler, 'models': models,
        'features': final_features, 'targets': TARGETS,
        'train_end_season': args.train_end_season,
        'test_season': args.test_season, 'metrics': results,
    }
    joblib.dump(bundle, os.path.join(args.out_dir, f'{stem}.joblib'))

    all_targets = ['recorded_hit', 'recorded_2plus', 'recorded_hr',
                   'recorded_k', 'recorded_2plus_bases', 'recorded_walk']
    out_cols = [t for t in all_targets if t in test.columns]
    out = test[['batter_id', 'game_id', 'game_date', 'season'] + out_cols].copy()
    for tgt, m in models.items():
        scaler2 = StandardScaler().fit(train[final_features])
        out[f'p_{tgt}'] = m.predict_proba(scaler.transform(test[final_features]))[:, 1]
    out.to_csv(os.path.join(args.out_dir, f'{stem}_2024_eval.csv'), index=False)

    with open(os.path.join(args.out_dir, f'{stem}_metrics.json'), 'w') as f:
        json.dump({'features': final_features, 'metrics': results}, f, indent=2)

    print(f"\n  Saved: artifacts/{stem}.*")
    print(f"\nDone.\n")


if __name__ == '__main__':
    main()