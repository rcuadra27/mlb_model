"""
train_pitcher_props_v1.py — Pitcher strikeout count model.

Predicts the distribution P(K=0), P(K=1), ..., P(K=10+) for each
starting pitcher per game using Poisson regression.

Output at inference time:
  - lambda (expected K count)
  - P(K >= k) for k in 1..10 — for over/under lines
  - P(K == k) for k in 0..10 — for exact count display

Splits:
  Train: seasons <= 2022
  Val:   2023
  Test:  2024 (OOS holdout)

Usage:
  PG_DSN=... python models/train_pitcher_props_v1.py
  PG_DSN=... python models/train_pitcher_props_v1.py --forward-select
"""
import argparse
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine, text
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from models.poisson_regressor import PoissonRegressor
import warnings
warnings.filterwarnings("ignore")


# ── Features ──────────────────────────────────────────────────────────────────

BASE_FEATURES = [
    "sp_k_rate_season",        # pitcher K rate season-to-date (strongest signal)
    "sp_k9_last5",             # recent K/9 from features_game
    "sp_xwoba_against_90",     # overall stuff quality
    "opp_lineup_k_rate_30d",   # how often the opposing lineup strikes out
    "umpire_k_boost",          # umpire zone tendency
]

OPTIONAL_FEATURES = {
    "sp_fastball_pct":     ["sp_fastball_pct"],
    "opp_lineup_xwoba":    ["opp_lineup_xwoba_90"],
    "park_factor":         ["park_runs_factor"],
    "sp_days_rest":        ["sp_days_rest"],
    "sp_innings_season":   ["sp_innings_season"],  # workload proxy (affects depth)
}

ALL_OPTIONAL = list(OPTIONAL_FEATURES.keys())

# SP threshold: minimum batters faced to be counted as a starter appearance
SP_MIN_BF = 15

# ── SQL ───────────────────────────────────────────────────────────────────────

# One row per SP per game: K count target + basic context
SP_GAME_TARGETS_QUERY = """
SELECT
    sp.pitcher                                              AS pitcher_id,
    sp.game_pk                                              AS game_id,
    sp.game_date,
    EXTRACT(YEAR FROM sp.game_date)::int                    AS season,
    -- home or away team ID for this pitcher
    CASE WHEN sp.inning_topbot = 'Top' THEN g.away_team_id
         ELSE g.home_team_id END                            AS team_id,
    CASE WHEN sp.inning_topbot = 'Top' THEN g.home_team_id
         ELSE g.away_team_id END                            AS opp_team_id,
    -- Target: strikeout count
    SUM(CASE WHEN sp.events = 'strikeout' THEN 1 ELSE 0 END) AS k_count,
    COUNT(DISTINCT sp.at_bat_number)                        AS batters_faced,
    -- Other outcomes for reference
    SUM(CASE WHEN sp.events IN ('walk','hit_by_pitch')
             THEN 1 ELSE 0 END)                             AS walks,
    SUM(CASE WHEN sp.events IN ('single','double','triple','home_run')
             THEN 1 ELSE 0 END)                             AS hits_allowed
FROM public.statcast_pitches sp
JOIN public.games g ON g.game_id = sp.game_pk
WHERE sp.game_date BETWEEN :start_date AND :end_date
  AND sp.pitcher IS NOT NULL
  AND sp.game_pk IS NOT NULL
  AND sp.inning <= 9
GROUP BY sp.pitcher, sp.game_pk, sp.game_date, g.away_team_id, g.home_team_id,
         CASE WHEN sp.inning_topbot = 'Top' THEN g.away_team_id ELSE g.home_team_id END,
         CASE WHEN sp.inning_topbot = 'Top' THEN g.home_team_id ELSE g.away_team_id END
HAVING COUNT(DISTINCT sp.at_bat_number) >= :min_bf
ORDER BY sp.game_date, sp.game_pk, sp.pitcher;
"""

# SP season-to-date K rate from statcast (same pattern as batter stats)
SP_SEASON_STATS_QUERY = """
WITH sp_game_stats AS (
    SELECT
        pitcher                                             AS pitcher_id,
        game_pk                                             AS game_id,
        game_date,
        EXTRACT(YEAR FROM game_date)::int                   AS season,
        SUM(CASE WHEN events = 'strikeout' THEN 1 ELSE 0 END) AS ks,
        COUNT(DISTINCT at_bat_number)                       AS bf
    FROM public.statcast_pitches
    WHERE game_date BETWEEN :start_date AND :end_date
      AND pitcher IS NOT NULL
      AND events IS NOT NULL
    GROUP BY pitcher, game_pk, game_date
    HAVING COUNT(DISTINCT at_bat_number) >= :min_bf
),
sp_cumulative AS (
    SELECT
        pitcher_id, game_id, game_date, season,
        SUM(ks) OVER w_season AS sd_ks,
        SUM(bf) OVER w_season AS sd_bf
    FROM sp_game_stats
    WINDOW w_season AS (
        PARTITION BY pitcher_id, season
        ORDER BY game_date, game_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    )
)
SELECT
    pitcher_id,
    game_id,
    game_date,
    season,
    -- K rate season-to-date (shrunk toward league avg 23%, prior=200 BF)
    (COALESCE(sd_ks, 0) + 200 * 0.230) /
        (COALESCE(sd_bf, 0) + 200)          AS sp_k_rate_season,
    COALESCE(sd_bf, 0)                       AS sp_sd_bf
FROM sp_cumulative;
"""

# Features from features_game for the SP
SP_FEATURES_QUERY = """
SELECT
    fg.game_id,
    g.home_team_id,
    g.away_team_id,
    -- Home SP features
    fg.home_sp_k9_last5,
    fg.home_sp_xwoba_against_90,
    fg.home_sp_days_rest,
    -- Away SP features
    fg.away_sp_k9_last5,
    fg.away_sp_xwoba_against_90,
    fg.away_sp_days_rest,
    -- Environment
    fg.umpire_k_rate_boost,
    COALESCE(fg.park_runs_factor_blended, fg.park_runs_factor, 1.0) AS park_runs_factor,
    -- Lineup quality (both sides)
    fg.home_lineup_k_rate_90,
    fg.away_lineup_k_rate_90,
    fg.home_lineup_xwoba_90,
    fg.away_lineup_xwoba_90
FROM public.features_game fg
JOIN public.games g ON g.game_id = fg.game_id
WHERE fg.game_date BETWEEN :start_date AND :end_date;
"""

# SP innings pitched this season (workload/depth proxy)
SP_INNINGS_QUERY = """
SELECT
    ps.pitcher_id,
    ps.game_id,
    ps.game_date,
    -- Season-to-date IP before this game
    SUM(ps.innings_pitched) OVER (
        PARTITION BY ps.pitcher_id, EXTRACT(YEAR FROM ps.game_date)::int
        ORDER BY ps.game_date, ps.game_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS sp_innings_season
FROM public.pitcher_starts ps
WHERE ps.game_date BETWEEN :start_date AND :end_date
  AND ps.innings_pitched IS NOT NULL
ORDER BY ps.game_date, ps.game_id, ps.pitcher_id;
"""

PITCHER_PITCHMIX_QUERY = """
SELECT pitcher_id, as_of_date,
    COALESCE(pct_ff, 0) + COALESCE(pct_si, 0) AS sp_fastball_pct
FROM public.pitcher_pitchmix_rolling
WHERE window_days = 365
  AND as_of_date BETWEEN :start_date AND :end_date;
"""


# ── Data loading ──────────────────────────────────────────────────────────────

def load_table(engine, query, params, label):
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params=params)
    print(f"  {label}: {len(df):,} rows")
    return df


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
    """merge_asof per entity group (avoids pandas global sort bug with `by=`)."""
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


# ── Metrics ───────────────────────────────────────────────────────────────────

def evaluate_poisson(label, y_true, lam_pred):
    y_true = np.asarray(y_true, dtype=float)
    lam = np.asarray(lam_pred, dtype=float)

    mae = mean_absolute_error(y_true, lam)
    rmse = np.sqrt(((y_true - lam) ** 2).mean())
    bias = (lam - y_true).mean()

    # Log-likelihood (higher is better)
    ll = stats.poisson.logpmf(y_true.astype(int), lam).mean()
    # Naive: always predict mean
    naive_lam = y_true.mean()
    naive_ll = stats.poisson.logpmf(y_true.astype(int), naive_lam).mean()

    print(f"\n  {'─'*65}")
    print(f"  {label}  (N={len(y_true):,})")
    print(f"  {'─'*65}")
    print(f"  Mean K actual: {y_true.mean():.3f}  std: {y_true.std():.3f}")
    print(f"  Mean λ pred:   {lam.mean():.3f}  std: {lam.std():.3f}")
    print(f"  MAE:    {mae:.4f}  RMSE: {rmse:.4f}  Bias: {bias:+.4f}")
    print(f"  Log-lik (mean): {ll:.4f}  (naive={naive_ll:.4f}  Δ={ll-naive_ll:+.4f})")
    print(f"  λ range: [{lam.min():.2f}, {lam.max():.2f}]")

    # Calibration: % correct over/under at common thresholds
    print(f"  Over/under calibration:")
    for thresh in [3.5, 4.5, 5.5, 6.5, 7.5]:
        k_floor = int(thresh + 0.5)
        actual_over = (y_true >= k_floor).mean()
        pred_p_over = (1.0 - stats.poisson.cdf(k_floor - 1, lam)).mean()
        print(f"    Over {thresh}: actual={actual_over*100:.1f}%  "
              f"model avg pred={pred_p_over*100:.1f}%")

    return {'mae': float(mae), 'rmse': float(rmse), 'bias': float(bias),
            'log_lik': float(ll), 'naive_log_lik': float(naive_ll),
            'log_lik_gain': float(ll - naive_ll)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='artifacts/')
    ap.add_argument('--train_end_season', type=int, default=2022)
    ap.add_argument('--val_season', type=int, default=2023)
    ap.add_argument('--test_season', type=int, default=2024)
    ap.add_argument('--earliest_season', type=int, default=2015)
    ap.add_argument('--min-bf', type=int, default=15,
                    help='Min batters faced to count as SP appearance')
    ap.add_argument('--extra-feature', dest='extra_features',
                    action='append', default=[],
                    choices=ALL_OPTIONAL)
    ap.add_argument('--all-features', action='store_true')
    ap.add_argument('--forward-select', action='store_true')
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
    params = {'start_date': start_date, 'end_date': end_date,
              'min_bf': args.min_bf}

    print(f"\n{'='*70}")
    print(f"Pitcher props v1 — SP strikeout count model (Poisson)")
    print(f"{'='*70}")
    print(f"Min BF threshold: {args.min_bf}")

    # ── Load targets ──
    print(f"\n[1/6] Loading SP game targets...")
    targets = load_table(engine, SP_GAME_TARGETS_QUERY, params, "SP game outcomes")
    print(f"  K count distribution:")
    dist = targets['k_count'].value_counts().sort_index()
    for k, n in dist.items():
        if k <= 12:
            print(f"    K={k:>2}: {n:>5} ({n/len(targets)*100:.1f}%)")
    print(f"  Mean K: {targets['k_count'].mean():.3f}  "
          f"Std: {targets['k_count'].std():.3f}")

    # ── Load SP season stats ──
    print(f"\n[2/6] Loading SP season-to-date K rate...")
    sp_stats = load_table(engine, SP_SEASON_STATS_QUERY, params, "SP season stats")

    # ── Load features_game SP features ──
    print(f"\n[3/6] Loading SP game features...")
    fg_feats = load_table(engine, SP_FEATURES_QUERY,
                          {'start_date': start_date, 'end_date': end_date},
                          "SP game features")

    # ── Load optional features ──
    print(f"\n[4/6] Loading optional features...")

    # Pitcher pitch mix (fastball pct)
    pm_params = {'start_date': start_date, 'end_date': end_date}
    pitcher_mix = load_table(engine, PITCHER_PITCHMIX_QUERY, pm_params, "pitcher mix")
    pitcher_mix['as_of_date'] = pd.to_datetime(pitcher_mix['as_of_date'])
    pitcher_mix = pitcher_mix.sort_values(['pitcher_id', 'as_of_date'])

    # Pitcher starts for innings (optional)
    try:
        sp_innings = load_table(engine, SP_INNINGS_QUERY,
                                {'start_date': start_date, 'end_date': end_date},
                                "SP innings")
    except Exception as e:
        print(f"  SP innings unavailable ({e}) — skipping")
        sp_innings = None

    # ── Merge everything ──
    print(f"\n[5/6] Merging features...")

    df = targets.merge(
        sp_stats[['pitcher_id', 'game_id', 'sp_k_rate_season', 'sp_sd_bf']],
        on=['pitcher_id', 'game_id'], how='left'
    )

    # Join features_game — need to know if this pitcher was home or away SP
    # Determine home/away for each pitcher row
    df = df.merge(
        fg_feats[['game_id', 'home_team_id', 'away_team_id',
                  'home_sp_k9_last5', 'away_sp_k9_last5',
                  'home_sp_xwoba_against_90', 'away_sp_xwoba_against_90',
                  'home_sp_days_rest', 'away_sp_days_rest',
                  'umpire_k_rate_boost', 'park_runs_factor',
                  'home_lineup_k_rate_90', 'away_lineup_k_rate_90',
                  'home_lineup_xwoba_90', 'away_lineup_xwoba_90']],
        on='game_id', how='left'
    )

    # Select the right home/away features based on team_id
    is_home = df['team_id'] == df['home_team_id']
    df['sp_k9_last5'] = np.where(is_home,
                                  df['home_sp_k9_last5'],
                                  df['away_sp_k9_last5'])
    df['sp_xwoba_against_90'] = np.where(is_home,
                                          df['home_sp_xwoba_against_90'],
                                          df['away_sp_xwoba_against_90'])
    df['sp_days_rest'] = np.where(is_home,
                                   df['home_sp_days_rest'],
                                   df['away_sp_days_rest'])
    # Opposing lineup features
    df['opp_lineup_k_rate_30d'] = np.where(is_home,
                                            df['away_lineup_k_rate_90'],
                                            df['home_lineup_k_rate_90'])
    df['opp_lineup_xwoba_90'] = np.where(is_home,
                                          df['away_lineup_xwoba_90'],
                                          df['home_lineup_xwoba_90'])

    # Fill defaults
    df['sp_k_rate_season']     = df['sp_k_rate_season'].fillna(0.230)
    df['sp_k9_last5']          = df['sp_k9_last5'].fillna(7.0)
    df['sp_xwoba_against_90']  = df['sp_xwoba_against_90'].fillna(0.320)
    df['opp_lineup_k_rate_30d'] = df['opp_lineup_k_rate_30d'].fillna(0.230)
    df['opp_lineup_xwoba_90']  = df['opp_lineup_xwoba_90'].fillna(0.320)
    df['umpire_k_boost']       = df['umpire_k_rate_boost'].fillna(0.0)
    df['park_runs_factor']     = df['park_runs_factor'].fillna(1.0)
    df['sp_days_rest']         = df['sp_days_rest'].fillna(4.0)

    # Fastball pct nearest-date lookup
    left_fb = df[['game_id', 'pitcher_id', 'game_date']].rename(
        columns={'game_date': 'game_date_dt'}
    )
    right_fb = pitcher_mix[['pitcher_id', 'as_of_date', 'sp_fastball_pct']].rename(
        columns={'as_of_date': 'game_date_dt'}
    ).drop_duplicates(subset=['pitcher_id', 'game_date_dt'], keep='last')
    fb_merged = _merge_asof_by_group(left_fb, right_fb, by='pitcher_id')
    df = df.merge(
        fb_merged[['game_id', 'pitcher_id', 'sp_fastball_pct']],
        on=['game_id', 'pitcher_id'],
        how='left',
    )
    df['sp_fastball_pct'] = df['sp_fastball_pct'].fillna(0.5)

    # SP innings (optional)
    if sp_innings is not None:
        df = df.merge(
            sp_innings[['pitcher_id', 'game_id', 'sp_innings_season']],
            on=['pitcher_id', 'game_id'], how='left'
        )
        df['sp_innings_season'] = df['sp_innings_season'].fillna(0.0)

    df = df.dropna(subset=BASE_FEATURES).copy()
    print(f"  Final rows: {len(df):,}")

    train = df[df['season'] <= args.train_end_season]
    val   = df[df['season'] == args.val_season]
    test  = df[df['season'] == args.test_season]
    print(f"  Train:{len(train):,}  Val:{len(val):,}  Test:{len(test):,}")

    # ── Training ──
    print(f"\n[6/6] Training Poisson regression...")

    def run_feature_set(feature_cols):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(train[feature_cols])
        X_v  = scaler.transform(val[feature_cols])
        X_te = scaler.transform(test[feature_cols])
        y_tr = train['k_count'].values
        y_v  = val['k_count'].values
        y_te = test['k_count'].values

        # Tune alpha on val log-likelihood
        best_alpha, best_ll = 0.1, -np.inf
        for alpha in [0.01, 0.1, 1.0, 10.0]:
            m = PoissonRegressor(alpha=alpha)
            m.fit(X_tr, y_tr)
            lam_v = m.predict_lambda(X_v)
            ll = stats.poisson.logpmf(y_v.astype(int), lam_v).mean()
            if ll > best_ll:
                best_ll, best_alpha = ll, alpha

        model = PoissonRegressor(alpha=best_alpha)
        model.fit(X_tr, y_tr)
        lam_te = model.predict_lambda(X_te)
        r = evaluate_poisson(f"Poisson α={best_alpha}", y_te, lam_te)
        return scaler, model, r

    if args.forward_select:
        print("\n  === FORWARD SELECTION ===")
        current = list(BASE_FEATURES)
        scaler, model, base_r = run_feature_set(current)
        base_ll = base_r['log_lik']
        print(f"\n  Base log-lik: {base_ll:.4f}")

        remaining = list(ALL_OPTIONAL)
        while remaining:
            best_feat, best_ll_val, best_r = None, base_ll, None
            for feat in remaining:
                candidate = current + OPTIONAL_FEATURES[feat]
                if not all(c in df.columns for c in OPTIONAL_FEATURES[feat]):
                    print(f"    + {feat}: SKIP (column missing)")
                    continue
                try:
                    _, _, r = run_feature_set(candidate)
                    print(f"    + {feat:<25} log-lik={r['log_lik']:.4f}  "
                          f"MAE={r['mae']:.4f}  Δll={r['log_lik']-base_ll:+.4f}")
                    if r['log_lik'] > best_ll_val:
                        best_ll_val, best_feat, best_r = r['log_lik'], feat, r
                except Exception as e:
                    print(f"    + {feat}: ERROR {e}")

            if best_feat:
                current += OPTIONAL_FEATURES[best_feat]
                remaining.remove(best_feat)
                base_ll = best_ll_val
                print(f"\n  ✓ Added {best_feat} → log-lik {best_ll_val:.4f}")
            else:
                print(f"\n  No more improvements. Stopping.")
                break

        final_features = current
    else:
        extra_cols = [c for ef in args.extra_features
                      for c in OPTIONAL_FEATURES.get(ef, [])]
        final_features = BASE_FEATURES + extra_cols

    print(f"\n  === FINAL MODEL — {len(final_features)} features ===")
    print(f"  {final_features}")
    scaler, model, results = run_feature_set(final_features)

    print(f"\n  Coefficients (on standardized features):")
    for fname, coef in sorted(zip(final_features, model.coef_),
                               key=lambda x: abs(x[1]), reverse=True):
        print(f"    {fname:<35} {coef:>+8.4f}")
    print(f"    {'intercept':<35} {model.intercept_:>+8.4f}")
    print(f"    → baseline λ = exp({model.intercept_:.3f}) = "
          f"{np.exp(model.intercept_):.2f} Ks")

    # Sample probability distributions for illustration
    print(f"\n  Sample P(K=k) distributions on test set:")
    X_te = scaler.transform(test[final_features])
    lam_te = model.predict_lambda(X_te)
    for pct, label in [(10, "Low (10th pct)"), (50, "Median"), (90, "Elite (90th pct)")]:
        lam_ex = np.percentile(lam_te, pct)
        pk = [stats.poisson.pmf(k, lam_ex) * 100 for k in range(11)]
        print(f"    λ={lam_ex:.2f} ({label}): "
              f"{' '.join(f'K{k}={p:.1f}%' for k, p in enumerate(pk[:9]))}")

    # Save
    stem = 'pitcher_props_v1'
    bundle = {
        'scaler': scaler, 'model': model,
        'features': final_features,
        'train_end_season': args.train_end_season,
        'test_season': args.test_season,
        'metrics': results,
        'min_bf': args.min_bf,
    }
    joblib.dump(bundle, os.path.join(args.out_dir, f'{stem}.joblib'))

    # Save test predictions
    X_te_df = test[['pitcher_id', 'game_id', 'game_date', 'season',
                     'k_count', 'batters_faced']].copy()
    X_te_df['lambda_pred'] = lam_te
    prob_df = model.predict_proba_k(X_te)
    over_df = model.predict_over_k(X_te)
    out = pd.concat([X_te_df.reset_index(drop=True),
                     prob_df, over_df], axis=1)
    out.to_csv(os.path.join(args.out_dir, f'{stem}_2024_eval.csv'), index=False)

    with open(os.path.join(args.out_dir, f'{stem}_metrics.json'), 'w') as f:
        json.dump({'features': final_features, 'metrics': results}, f, indent=2)

    print(f"\n  Saved: artifacts/{stem}.*")
    print(f"\nDone.\n")


if __name__ == '__main__':
    main()