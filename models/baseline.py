"""
Baseline win-probability model — v10 floor.

Goal: prove that a 5-feature logistic regression can beat v9 (53.66% pick
accuracy, Brier 0.2696, correlation 0.079 on 2025 OOS) on the same 2025
holdout — using STABLE season-to-date features instead of the short rolling
windows that dominate v9.

v10 Step 1: sp_xwoba_against_90 replaces sp_era_last3 (90d statcast, more stable).
Locked v10 feature set (6): base + lineup_xwoba_diff; SP/lineup decomposed for LGBM splits.
Step 3: shallow LightGBM on locked set — num_leaves=16, reg_lambda=5.0, min_data_in_leaf=100.
Step 4: raw logistic on same features is the calibration bar (Brier ~0.2449, ECE ~1.76%).

Key design choice: We compute season-to-date team strength from the games
table directly, NOT from features_game (which only has 30-day rolling
windows). This is the central hypothesis: stable, slow-moving signals beat
noisy short-window signals for MLB game-level prediction.

Splits:
  Train: seasons <= 2023
  Val:   2024 (used to fit isotonic calibrator)
  Test:  2025 (final OOS holdout, same 2,430 games we diagnosed)

Usage:
  PG_DSN=postgresql+psycopg2://... python baseline_v10_floor.py
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
from sklearn.isotonic import IsotonicRegression
from lightgbm import LGBMClassifier, early_stopping

# Production bar — locked shallow LGBM on V10_LOCKED_FEATURES (2025 OOS, raw, no isotonic).
LGBM_PRODUCTION_BAR = {
    'brier': 0.2445,
    'ece': 0.0122,
    'pick_accuracy': 0.5514,
}
# Legacy logit reference (5-feature floor).
LOGIT_BAR = {'brier': 0.2449, 'ece': 0.0267, 'pick_accuracy': 0.5556}


# ============================================================================
# DATA LOADING
# ============================================================================

GAMES_QUERY = """
SELECT
    g.game_id,
    g.game_date,
    EXTRACT(YEAR FROM g.game_date)::int AS season,
    g.home_team_id,
    g.away_team_id,
    g.home_runs,
    g.away_runs,
    g.status
FROM public.games g
WHERE g.game_date BETWEEN :start_date AND :end_date
  AND g.home_runs IS NOT NULL
  AND g.away_runs IS NOT NULL
  AND g.home_runs != g.away_runs
  AND (
      LOWER(COALESCE(g.status, '')) LIKE 'final%%'
      OR LOWER(COALESCE(g.status, '')) = 'game over'
      OR LOWER(COALESCE(g.status, '')) LIKE 'completed%%'
  )
ORDER BY g.game_date, g.game_id;
"""

FEATURES_QUERY = """
SELECT
    fg.game_id,
    fg.home_sp_xwoba_against_90,
    fg.away_sp_xwoba_against_90,
    fg.park_runs_factor,
    fg.park_runs_factor_blended,
    fg.home_avg_runs_scored_60,
    fg.home_avg_runs_allowed_60,
    fg.away_avg_runs_scored_60,
    fg.away_avg_runs_allowed_60,
    fg.home_bp_outs_3d,
    fg.away_bp_outs_3d,
    fg.umpire_runs_boost,
    fg.home_lineup_xwoba_90,
    fg.away_lineup_xwoba_90,
    fg.home_sp_k_rate_90,
    fg.away_sp_k_rate_90,
    fg.home_win_pct_30,
    fg.away_win_pct_30,
    fg.home_sp_days_rest,
    fg.away_sp_days_rest,
    fg.home_bp_hlev_outs_3d,
    fg.away_bp_hlev_outs_3d,
    fg.sharp_action_home,
    fg.line_move_magnitude
FROM public.features_game fg
WHERE fg.game_date BETWEEN :start_date AND :end_date;
"""

# Legacy 5-feature floor (logit bar reference).
BASE_FEATURES = ['win_pct_diff', 'run_diff_pg_diff', 'sp_xwoba_diff', 'park_factor', 'is_home_const']

# Locked v10 — SP + lineup decomposed (drop sp_xwoba_matchup_diff; trees learn interaction).
V10_LOCKED_FEATURES = [
    'win_pct_diff',
    'run_diff_pg_diff',
    'sp_xwoba_diff',
    'lineup_xwoba_diff',
    'park_factor',
    'is_home_const',
]

# Optional add-ons — use --extra-feature one at a time, or compose via --features.
OPTIONAL_FEATURES = {
    'run_env_matchup': 'run_env_matchup',
    'bp_outs_3d_diff': 'bp_outs_3d_diff',
    'umpire_runs_boost': 'umpire_runs_boost',
    'sp_xwoba_matchup_diff': 'sp_xwoba_matchup_diff',
    'lineup_xwoba_diff': 'lineup_xwoba_diff',
    'sp_k_rate_diff': 'sp_k_rate_diff',
    'win_pct_30d_diff': 'win_pct_30d_diff',
    'sp_days_rest_diff': 'sp_days_rest_diff',
    'bp_hlev_outs_3d_diff': 'bp_hlev_outs_3d_diff',
    'sharp_action_home': 'sharp_action_home',
    'line_move_magnitude': 'line_move_magnitude',
}

KNOWN_FEATURES = BASE_FEATURES + list(OPTIONAL_FEATURES.values())


def load_data(engine, start_date: str, end_date: str):
    with engine.connect() as conn:
        games = pd.read_sql(text(GAMES_QUERY), conn,
                            params={'start_date': start_date, 'end_date': end_date})
        feats = pd.read_sql(text(FEATURES_QUERY), conn,
                            params={'start_date': start_date, 'end_date': end_date})
    print(f"  Loaded {len(games):,} games and {len(feats):,} feature rows")
    return games, feats


# ============================================================================
# SEASON-TO-DATE FEATURE ENGINEERING
# ============================================================================

def compute_season_to_date(games: pd.DataFrame) -> pd.DataFrame:
    """For each game, compute season-to-date stats for both teams as of the
    morning of that game (i.e., excluding the game itself).
    """
    games = games.sort_values(['season', 'game_date', 'game_id']).reset_index(drop=True).copy()

    # Long-form team-game log
    home_log = games.rename(columns={
        'home_team_id': 'team_id', 'away_team_id': 'opp_id',
        'home_runs': 'runs_for', 'away_runs': 'runs_against'
    })[['game_id', 'game_date', 'season', 'team_id', 'runs_for', 'runs_against']]
    home_log['won'] = (home_log['runs_for'] > home_log['runs_against']).astype(int)
    home_log['is_home_row'] = True

    away_log = games.rename(columns={
        'away_team_id': 'team_id', 'home_team_id': 'opp_id',
        'away_runs': 'runs_for', 'home_runs': 'runs_against'
    })[['game_id', 'game_date', 'season', 'team_id', 'runs_for', 'runs_against']]
    away_log['won'] = (away_log['runs_for'] > away_log['runs_against']).astype(int)
    away_log['is_home_row'] = False

    log = pd.concat([home_log, away_log], ignore_index=True)
    log = log.sort_values(['season', 'team_id', 'game_date', 'game_id']).reset_index(drop=True)

    # Cumulative stats EXCLUDING the current game (shift(1) then cumsum)
    grp = log.groupby(['season', 'team_id'])
    log['sd_games'] = grp.cumcount()
    log['sd_wins'] = grp['won'].transform(lambda s: s.shift(1).cumsum().fillna(0)).astype(int)
    log['sd_runs_for'] = grp['runs_for'].transform(lambda s: s.shift(1).cumsum().fillna(0))
    log['sd_runs_against'] = grp['runs_against'].transform(lambda s: s.shift(1).cumsum().fillna(0))

    home_sd = log[log['is_home_row']][['game_id', 'sd_games', 'sd_wins',
                                        'sd_runs_for', 'sd_runs_against']].rename(
        columns={'sd_games': 'home_sd_games', 'sd_wins': 'home_sd_wins',
                 'sd_runs_for': 'home_sd_runs_for', 'sd_runs_against': 'home_sd_runs_against'})
    away_sd = log[~log['is_home_row']][['game_id', 'sd_games', 'sd_wins',
                                         'sd_runs_for', 'sd_runs_against']].rename(
        columns={'sd_games': 'away_sd_games', 'sd_wins': 'away_sd_wins',
                 'sd_runs_for': 'away_sd_runs_for', 'sd_runs_against': 'away_sd_runs_against'})

    out = games.merge(home_sd, on='game_id').merge(away_sd, on='game_id')
    return out


def _compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all candidate model features (one row per game)."""
    PRIOR_GAMES = 30
    PRIOR_WIN_PCT = 0.5
    league_xwoba_default = 0.320
    league_lineup_xwoba = 0.320
    league_k_rate = 0.22

    df['home_win_pct'] = (df['home_sd_wins'] + PRIOR_GAMES * PRIOR_WIN_PCT) / \
                          (df['home_sd_games'] + PRIOR_GAMES)
    df['away_win_pct'] = (df['away_sd_wins'] + PRIOR_GAMES * PRIOR_WIN_PCT) / \
                          (df['away_sd_games'] + PRIOR_GAMES)
    df['win_pct_diff'] = df['home_win_pct'] - df['away_win_pct']

    df['home_rd_pg'] = (df['home_sd_runs_for'] - df['home_sd_runs_against']) / \
                       (df['home_sd_games'] + PRIOR_GAMES)
    df['away_rd_pg'] = (df['away_sd_runs_for'] - df['away_sd_runs_against']) / \
                       (df['away_sd_games'] + PRIOR_GAMES)
    df['run_diff_pg_diff'] = df['home_rd_pg'] - df['away_rd_pg']

    df['home_sp_xwoba'] = df['home_sp_xwoba_against_90'].fillna(league_xwoba_default).clip(0.15, 0.50)
    df['away_sp_xwoba'] = df['away_sp_xwoba_against_90'].fillna(league_xwoba_default).clip(0.15, 0.50)
    df['sp_xwoba_diff'] = df['away_sp_xwoba'] - df['home_sp_xwoba']

    home_lu = df['home_lineup_xwoba_90'].fillna(league_lineup_xwoba).clip(0.15, 0.50)
    away_lu = df['away_lineup_xwoba_90'].fillna(league_lineup_xwoba).clip(0.15, 0.50)
    df['lineup_xwoba_diff'] = home_lu - away_lu
    df['sp_xwoba_matchup_diff'] = (
        (away_lu - df['home_sp_xwoba']) - (home_lu - df['away_sp_xwoba'])
    )

    df['park_factor'] = df['park_runs_factor_blended'].fillna(df['park_runs_factor']).fillna(1.0)
    df['is_home_const'] = 1.0

    df['run_env_matchup'] = (
        (df['home_avg_runs_scored_60'] - df['away_avg_runs_allowed_60'])
        - (df['away_avg_runs_scored_60'] - df['home_avg_runs_allowed_60'])
    )
    df['bp_outs_3d_diff'] = df['home_bp_outs_3d'].fillna(0) - df['away_bp_outs_3d'].fillna(0)
    df['umpire_runs_boost'] = df['umpire_runs_boost'].fillna(0.0)
    df['sp_k_rate_diff'] = (
        df['home_sp_k_rate_90'].fillna(league_k_rate).clip(0.05, 0.45)
        - df['away_sp_k_rate_90'].fillna(league_k_rate).clip(0.05, 0.45)
    )
    df['win_pct_30d_diff'] = (
        df['home_win_pct_30'].fillna(0.500).clip(0.0, 1.0)
        - df['away_win_pct_30'].fillna(0.500).clip(0.0, 1.0)
    )
    df['sp_days_rest_diff'] = (
        df['home_sp_days_rest'].fillna(5.0).clip(0, 30)
        - df['away_sp_days_rest'].fillna(5.0).clip(0, 30)
    )
    df['bp_hlev_outs_3d_diff'] = (
        df['home_bp_hlev_outs_3d'].fillna(0) - df['away_bp_hlev_outs_3d'].fillna(0)
    )
    df['sharp_action_home'] = df['sharp_action_home'].fillna(0).astype(float)
    df['line_move_magnitude'] = df['line_move_magnitude'].fillna(0.0).clip(lower=0.0)
    return df


def engineer_features(
    games_sd: pd.DataFrame,
    feats: pd.DataFrame,
    extra_features: list[str] | None = None,
    feature_cols_override: list[str] | None = None,
    locked_base: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Compute v10 features. One row per game."""
    extra_features = extra_features or []
    unknown_extra = [f for f in extra_features if f not in OPTIONAL_FEATURES]
    if unknown_extra:
        raise ValueError(f"Unknown optional features: {unknown_extra}. "
                         f"Choose from: {list(OPTIONAL_FEATURES)}")

    df = games_sd.merge(feats, on='game_id', how='left').copy()
    df['home_win'] = (df['home_runs'] > df['away_runs']).astype(int)
    df = _compute_derived_features(df)

    if feature_cols_override is not None:
        unknown = [f for f in feature_cols_override if f not in KNOWN_FEATURES]
        if unknown:
            raise ValueError(f"Unknown features: {unknown}. Choose from: {KNOWN_FEATURES}")
        feature_cols = feature_cols_override
    elif extra_features:
        base = V10_LOCKED_FEATURES if locked_base else BASE_FEATURES
        added = [
            OPTIONAL_FEATURES[f] for f in extra_features
            if OPTIONAL_FEATURES[f] not in base
        ]
        feature_cols = base + added
    else:
        feature_cols = V10_LOCKED_FEATURES

    required = feature_cols + ['home_win']
    before = len(df)
    df = df.dropna(subset=required).copy()
    print(f"  Feature set ({len(feature_cols)}): {feature_cols}")
    print(f"  Dropped {before - len(df):,} rows with missing features; {len(df):,} remain")
    return df, feature_cols


# ============================================================================
# EVALUATION
# ============================================================================

def wilson_ci(p, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - half, c + half)


def evaluate(p_pred: np.ndarray, y: np.ndarray, label: str, verbose=True):
    p_pred = np.asarray(p_pred)
    y = np.asarray(y).astype(int)
    n = len(y)

    pick_home = (p_pred >= 0.5).astype(int)
    pick_correct = (pick_home == y).astype(int)
    p_pick = np.where(p_pred >= 0.5, p_pred, 1.0 - p_pred)
    brier = ((p_pick - pick_correct) ** 2).mean()

    bins = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.01]
    bucket = pd.cut(p_pick, bins=bins, right=False, include_lowest=True)
    rel = pd.DataFrame({'bucket': bucket, 'p': p_pick, 'y': pick_correct}) \
        .groupby('bucket', observed=True).agg(n=('y', 'size'), mp=('p', 'mean'),
                                              ma=('y', 'mean')).reset_index()
    ece = (rel['n'] / n * (rel['mp'] - rel['ma']).abs()).sum()

    if verbose:
        print(f"\n{'=' * 80}")
        print(f"  {label}  (N = {n:,})")
        print(f"{'=' * 80}")
        print(f"Pick accuracy:        {pick_correct.mean() * 100:.2f}%")
        print(f"Always-home baseline: {y.mean() * 100:.2f}% "
              f"(edge: {(pick_correct.mean() - y.mean()) * 100:+.2f} pts)")
        print(f"Mean p_pick:          {p_pick.mean() * 100:.2f}%")
        print(f"  Overconfidence gap: {(p_pick.mean() - pick_correct.mean()) * 100:+.2f} pts")
        print(f"Brier (pick side):    {brier:.4f}  (0.25 = random)")
        print(f"ECE (10 buckets):     {ece * 100:.2f}%")
        print(f"Predicted prob range: [{p_pred.min() * 100:.1f}%, {p_pred.max() * 100:.1f}%]")
        print(f"\n{'Bucket':>15} {'N':>5} {'Pred%':>8} {'Actual%':>9} {'95% CI':>16} {'Gap':>7}")
        for _, r in rel.iterrows():
            lo, hi = wilson_ci(r['ma'], r['n'])
            print(f"{str(r['bucket']):>15} {r['n']:>5} {r['mp'] * 100:>7.1f}% "
                  f"{r['ma'] * 100:>8.1f}% [{lo * 100:>5.1f},{hi * 100:>5.1f}] "
                  f"{(r['mp'] - r['ma']) * 100:>+6.1f}")

    return {'n': int(n),
            'pick_accuracy': float(pick_correct.mean()),
            'always_home_baseline': float(y.mean()),
            'edge': float(pick_correct.mean() - y.mean()),
            'mean_p_pick': float(p_pick.mean()),
            'brier': float(brier),
            'ece': float(ece),
            'p_min': float(p_pred.min()),
            'p_max': float(p_pred.max())}


def fit_model(
    model_type: str,
    X_train,
    y_train,
    X_val,
    y_val,
    seed: int = 42,
    feature_cols: list[str] | None = None,
):
    """Fit logistic (legacy bar) or shallow LightGBM (production default)."""
    if model_type == 'logit':
        model = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
        model.fit(X_train, y_train)
        predict_proba = lambda X: model.predict_proba(X)[:, 1]
        return model, predict_proba

    if model_type == 'lgbm':
        if feature_cols is None:
            raise ValueError("feature_cols required for LGBM")
        X_train_df = pd.DataFrame(X_train, columns=feature_cols)
        X_val_df = pd.DataFrame(X_val, columns=feature_cols)
        model = LGBMClassifier(
            objective='binary',
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=16,
            min_child_samples=100,
            subsample=0.8,
            subsample_freq=1,
            feature_fraction=1.0,
            reg_alpha=0.1,
            reg_lambda=5.0,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(
            X_train_df, y_train,
            eval_set=[(X_val_df, y_val)],
            eval_metric='binary_logloss',
            callbacks=[early_stopping(stopping_rounds=30, verbose=False)],
        )
        predict_proba = lambda X: model.predict_proba(
            pd.DataFrame(X, columns=feature_cols) if not isinstance(X, pd.DataFrame) else X
        )[:, 1]
        return model, predict_proba

    raise ValueError(f"Unknown model_type: {model_type}")


def print_model_summary(model_type: str, model, feature_cols: list[str]):
    if model_type == 'logit':
        print(f"\n  Intercept: {model.intercept_[0]:+.4f}  "
              f"(implied baseline home prob = "
              f"{1 / (1 + np.exp(-model.intercept_[0])) * 100:.2f}%)")
        print("  Coefficients (on standardized features):")
        for fname, coef in zip(feature_cols, model.coef_[0]):
            print(f"    {fname:25s}  {coef:+.4f}")
    elif model_type == 'lgbm':
        print(f"\n  Best iteration: {model.best_iteration_}")
        print("  Feature importances (gain):")
        imp = model.booster_.feature_importance(importance_type='gain')
        for fname, g in sorted(zip(feature_cols, imp), key=lambda x: -x[1]):
            print(f"    {fname:25s}  {g:>10.1f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='artifacts/')
    ap.add_argument('--train_end_season', type=int, default=2023)
    ap.add_argument('--val_season', type=int, default=2024)
    ap.add_argument('--test_season', type=int, default=2025)
    ap.add_argument('--earliest_season', type=int, default=2015)
    ap.add_argument(
        '--features',
        nargs='+',
        choices=KNOWN_FEATURES,
        default=None,
        help='Override feature list entirely (e.g. swap sp_xwoba_diff for sp_xwoba_matchup_diff)',
    )
    ap.add_argument(
        '--extra-feature',
        action='append',
        choices=list(OPTIONAL_FEATURES),
        default=[],
        help='Step 2: add optional features; gate each on 2025 raw Brier vs logit bar',
    )
    ap.add_argument(
        '--model',
        choices=['logit', 'lgbm'],
        default='lgbm',
        help='Production default: shallow LGBM on V10_LOCKED_FEATURES',
    )
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--artifact-name', default=None,
                    help='Override artifact stem (default: baseline_v10_{model})')
    ap.add_argument(
        '--legacy-five',
        action='store_true',
        help='Use legacy 5-feature base instead of locked v10 set (experiments only)',
    )
    args = ap.parse_args()

    pg_dsn = os.environ.get('PG_DSN')
    if not pg_dsn:
        print("ERROR: PG_DSN env var not set", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    engine = create_engine(pg_dsn)

    start_date = f'{args.earliest_season}-01-01'
    end_date = f'{args.test_season}-12-31'

    print(f"\n[1/5] Loading data {start_date} → {end_date}...")
    games, feats = load_data(engine, start_date, end_date)

    print(f"\n[2/5] Computing season-to-date team strength features...")
    games_sd = compute_season_to_date(games)

    if args.features and args.extra_feature:
        print("ERROR: use --features OR --extra-feature, not both", file=sys.stderr)
        sys.exit(1)
    if args.legacy_five and (args.features or args.extra_feature):
        print("ERROR: --legacy-five cannot combine with --features or --extra-feature",
              file=sys.stderr)
        sys.exit(1)

    feature_override = args.features
    extra = args.extra_feature
    if args.legacy_five:
        feature_override = BASE_FEATURES

    print(f"\n[3/5] Engineering final feature set...")
    df, feature_cols = engineer_features(
        games_sd, feats,
        extra_features=extra,
        feature_cols_override=feature_override,
        locked_base=not args.legacy_five,
    )

    train = df[df['season'] <= args.train_end_season].copy()
    val = df[df['season'] == args.val_season].copy()
    test = df[df['season'] == args.test_season].copy()
    print(f"\n  Train: {len(train):,} games (seasons {args.earliest_season}-{args.train_end_season})")
    print(f"  Val:   {len(val):,} games (season {args.val_season})")
    print(f"  Test:  {len(test):,} games (season {args.test_season})")

    if len(test) == 0:
        print("ERROR: no test data — check season filters", file=sys.stderr)
        sys.exit(1)

    print(f"\n[4/5] Fitting {args.model} model...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_cols])
    y_train = train['home_win'].values
    X_val = scaler.transform(val[feature_cols])
    y_val = val['home_win'].values
    X_test = scaler.transform(test[feature_cols])

    model, predict_proba = fit_model(
        args.model, X_train, y_train, X_val, y_val,
        seed=args.seed, feature_cols=feature_cols,
    )
    print_model_summary(args.model, model, feature_cols)

    p_val_raw = predict_proba(X_val)
    p_test_raw = predict_proba(X_test)

    eval_val = evaluate(p_val_raw, val['home_win'].values,
                        f"VAL {args.val_season} (raw)", verbose=False)
    eval_test_raw = evaluate(p_test_raw, test['home_win'].values,
                             f"TEST {args.test_season} (raw, headline)")

    print(f"\n  Fitting isotonic calibrator on val ({args.val_season})...")
    iso = IsotonicRegression(out_of_bounds='clip', increasing=True)
    iso.fit(p_val_raw, val['home_win'].values)
    p_test_cal = iso.predict(p_test_raw)

    eval_test_cal = evaluate(p_test_cal, test['home_win'].values,
                             f"TEST {args.test_season} (isotonic-calibrated on {args.val_season})")

    prod_bar = LGBM_PRODUCTION_BAR if args.model == 'lgbm' else LOGIT_BAR
    bar_label = 'lgbm-bar' if args.model == 'lgbm' else 'logit-bar'
    beats_bar = (
        eval_test_raw['brier'] < prod_bar['brier']
        and eval_test_raw['ece'] <= prod_bar['ece']
    )
    print(f"\n{'=' * 80}")
    print(f"  HEAD-TO-HEAD: {args.model} vs v9 / {bar_label} on {args.test_season} holdout")
    print(f"{'=' * 80}")
    print(f"{'Metric':<25} {'v9':>12} {bar_label:>12} {args.model + '-raw':>15} {'cal':>10}")
    print(f"{'-' * 78}")
    print(f"{'Pick accuracy':<25} {'53.66%':>12} "
          f"{prod_bar['pick_accuracy']*100:>11.2f}% "
          f"{eval_test_raw['pick_accuracy'] * 100:>14.2f}% "
          f"{eval_test_cal['pick_accuracy'] * 100:>9.2f}%")
    print(f"{'Brier (pick side)':<25} {'0.2696':>12} "
          f"{prod_bar['brier']:>12.4f} "
          f"{eval_test_raw['brier']:>15.4f} "
          f"{eval_test_cal['brier']:>9.4f}")
    print(f"{'ECE':<25} {'11.46%':>12} "
          f"{prod_bar['ece']*100:>11.2f}% "
          f"{eval_test_raw['ece'] * 100:>14.2f}% "
          f"{eval_test_cal['ece'] * 100:>8.2f}%")
    print(f"{'Beats production bar (raw)':<25} {'':>12} {'':>12} "
          f"{'YES' if beats_bar else 'NO':>15}")
    print(f"{'-' * 78}")

    is_production_lock = (
        args.model == 'lgbm'
        and not args.extra_feature
        and not args.features
        and not args.legacy_five
    )
    artifact_stem = args.artifact_name or (
        'baseline_v10_production' if is_production_lock else f'baseline_v10_{args.model}'
    )
    print(f"\n[5/5] Saving artifacts to {args.out_dir}...")
    joblib.dump({'scaler': scaler, 'model': model, 'calibrator': iso,
                 'model_type': args.model,
                 'features': feature_cols,
                 'prior_games': 30, 'prior_win_pct': 0.5},
                os.path.join(args.out_dir, f'{artifact_stem}.joblib'))

    out_df = test[['game_id', 'game_date', 'home_team_id', 'away_team_id',
                   'home_runs', 'away_runs', 'home_win'] + feature_cols].copy()
    out_df['p_home_win_raw'] = p_test_raw
    out_df['p_home_win_cal'] = p_test_cal
    out_df.to_csv(os.path.join(args.out_dir,
                                f'{artifact_stem}_{args.test_season}_eval.csv'),
                  index=False)

    metrics = {
        'config': {
            'train_end_season': args.train_end_season,
            'val_season': args.val_season,
            'test_season': args.test_season,
            'model_type': args.model,
            'features': feature_cols,
            'features_override': args.features,
            'extra_features': args.extra_feature,
            'n_train': len(train), 'n_val': len(val), 'n_test': len(test),
            'logit_bar': LOGIT_BAR,
            'lgbm_production_bar': LGBM_PRODUCTION_BAR,
            'production_bar': prod_bar,
            'beats_production_bar_raw': beats_bar,
            'production_locked': is_production_lock,
        },
        'val_raw': eval_val,
        'test_raw': eval_test_raw,
        'test_cal': eval_test_cal,
        'v9_reference': {'pick_accuracy': 0.5366, 'edge': -0.0062,
                         'brier': 0.2696, 'ece': 0.1146}
    }
    if args.model == 'logit':
        metrics['coefficients'] = {
            f: float(c) for f, c in zip(feature_cols, model.coef_[0])
        }
        metrics['intercept'] = float(model.intercept_[0])
    with open(os.path.join(args.out_dir, f'{artifact_stem}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  Saved:")
    print(f"    {args.out_dir}{artifact_stem}.joblib")
    print(f"    {args.out_dir}{artifact_stem}_{args.test_season}_eval.csv")
    print(f"    {args.out_dir}{artifact_stem}_metrics.json")
    print(f"\nDone.\n")


if __name__ == '__main__':
    main()