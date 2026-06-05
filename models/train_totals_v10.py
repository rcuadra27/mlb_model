"""
train_totals_v10.py — v10-total: total runs prediction model.

Target: home_runs + away_runs (regression)
Goal:   beat v9 total RMSE of 4.6834 on 2025 OOS holdout.
        Get as close as possible to O/U line RMSE (market floor).

Design:
  - One row per game (not two like v9)
  - Target: actual total runs scored
  - Features: game-environment signals only (no market/odds features)
  - At inference: compare pred_total vs morning_ou_line → over/under rec
  - Train: seasons <= 2023
  - Val:   2024 (early stopping + feature gating)
  - Test:  2025 (final OOS holdout)

Usage:
  PG_DSN=... python models/train_totals_v10.py
  PG_DSN=... python models/train_totals_v10.py --extra-feature forecast_temp_f
  PG_DSN=... python models/train_totals_v10.py --extra-feature forecast_wind
  PG_DSN=... python models/train_totals_v10.py --extra-feature umpire_runs_boost
  PG_DSN=... python models/train_totals_v10.py --extra-feature sp_xwoba_total
  PG_DSN=... python models/train_totals_v10.py --use-lgbm
"""
import argparse
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# ── Base feature set ─────────────────────────────────────────────────────────
# These are the stable, game-environment signals.
# Wind is encoded as two components (sin, cos) rather than raw degrees.
BASE_FEATURES = [
    "league_avg_total",      # league_avg_runs_60d × 2
    "total_offense_env",     # combined team offense environment
    "total_defense_env",     # combined pitching environment
    "park_runs_factor",      # park run inflation/suppression
]

# Optional features gated on RMSE improvement
OPTIONAL_FEATURES = {
    "forecast_temp_f":   ["temp_f_centered"],       # temperature centered at 72°F
    "forecast_wind":     ["wind_out_sin", "wind_mph_scaled"],  # wind encoding
    "umpire_runs_boost": ["umpire_runs_boost"],
    "sp_xwoba_total":    ["sp_xwoba_total"],         # sum of both SP xwOBA-against
}

# ── Reference bars (from diagnostic) ─────────────────────────────────────────
V9_TOTAL_RMSE = 4.6834
GRAND_MEAN_RMSE = 4.5934   # always-predict-mean baseline

# ── SQL ──────────────────────────────────────────────────────────────────────
QUERY = """
SELECT
    fg.game_id,
    fg.game_date,
    fg.season,
    fg.home_team_id,
    fg.away_team_id,
    g.home_runs,
    g.away_runs,
    -- Base features
    fg.league_avg_runs_60d,
    fg.total_offense_env,
    fg.total_defense_env,
    COALESCE(fg.park_runs_factor_blended, fg.park_runs_factor) AS park_runs_factor,
    -- Weather
    fg.forecast_temp_f,
    fg.forecast_wind_mph,
    fg.forecast_wind_dir_deg,
    fg.forecast_precip_in,
    -- Umpire
    fg.umpire_runs_boost,
    fg.umpire_n_games,
    -- SP xwOBA (for optional feature)
    fg.home_sp_xwoba_against_90,
    fg.away_sp_xwoba_against_90
FROM public.features_game fg
JOIN public.games g
    ON g.game_id = fg.game_id
WHERE g.home_runs IS NOT NULL
  AND g.away_runs  IS NOT NULL
  AND (
      LOWER(COALESCE(g.status,'')) LIKE 'final%%'
      OR LOWER(COALESCE(g.status,'')) = 'game over'
      OR LOWER(COALESCE(g.status,'')) LIKE 'completed%%'
  )
  AND fg.season BETWEEN :year_min AND :year_max
ORDER BY fg.game_date, fg.game_id;
"""


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data(engine, year_min: int, year_max: int) -> pd.DataFrame:
    with engine.connect() as conn:
        df = pd.read_sql(text(QUERY), conn,
                         params={"year_min": year_min, "year_max": year_max})
    print(f"  Loaded {len(df):,} games ({year_min}–{year_max})")
    return df


# ── Feature engineering ──────────────────────────────────────────────────────

def engineer(df: pd.DataFrame, extra_features: list) -> pd.DataFrame:
    df = df.copy()

    # Target
    df["actual_total"] = df["home_runs"] + df["away_runs"]

    # ── Base features ──

    # league_avg_total: per-game league average × 2 (both teams)
    # Fallback to grand mean if null
    LEAGUE_MEAN_FALLBACK = 9.0
    df["league_avg_total"] = (df["league_avg_runs_60d"] * 2).fillna(LEAGUE_MEAN_FALLBACK)

    # total_offense_env / total_defense_env: fill with 0 (neutral) if null
    df["total_offense_env"] = df["total_offense_env"].fillna(0.0)
    df["total_defense_env"] = df["total_defense_env"].fillna(0.0)

    # park_runs_factor: fill with 1.0 (neutral park) if null
    df["park_runs_factor"] = df["park_runs_factor"].fillna(1.0)

    # ── Optional features ──

    if "forecast_temp_f" in extra_features:
        # Center at 72°F (comfortable baseball weather baseline)
        # Cold games have negative values → lower scoring
        TEMP_CENTER = 72.0
        df["temp_f_centered"] = df["forecast_temp_f"].fillna(TEMP_CENTER) - TEMP_CENTER

    if "forecast_wind" in extra_features:
        # Wind encoding without park-specific outfield direction:
        # wind_out_sin ≈ component blowing toward outfield (approximate)
        # Use raw sin/cos of wind direction
        # sin(dir): positive = easterly wind
        # cos(dir): positive = northerly wind (typically blowing in at most parks)
        # We use sin as the primary "blowing out" proxy since most parks
        # face roughly south/southwest
        DEFAULT_WIND = 0.0
        DEFAULT_DIR = 180.0  # assume neutral (southerly, typical park orientation)

        wind = df["forecast_wind_mph"].fillna(DEFAULT_WIND)
        direction = df["forecast_wind_dir_deg"].fillna(DEFAULT_DIR)
        dir_rad = np.deg2rad(direction)

        # Directional component: wind speed × sin(direction)
        # Positive = easterly component (tends to blow out at most parks)
        df["wind_out_sin"] = wind * np.sin(dir_rad)
        df["wind_mph_scaled"] = wind  # raw speed for magnitude signal

    if "umpire_runs_boost" in extra_features:
        # Fill with 0 (neutral umpire) for missing
        # Only trust umpires with >= 20 games of history
        df["umpire_runs_boost"] = np.where(
            df["umpire_n_games"].fillna(0) >= 20,
            df["umpire_runs_boost"].fillna(0.0),
            0.0
        )

    if "sp_xwoba_total" in extra_features:
        # Sum of both starters' xwOBA-against — higher = worse pitching matchup = more runs
        DEFAULT_XWOBA = 0.320
        home_xwoba = df["home_sp_xwoba_against_90"].fillna(DEFAULT_XWOBA)
        away_xwoba = df["away_sp_xwoba_against_90"].fillna(DEFAULT_XWOBA)
        df["sp_xwoba_total"] = home_xwoba + away_xwoba

    # Drop rows missing the base features or target
    required = ["actual_total"] + BASE_FEATURES
    before = len(df)
    df = df.dropna(subset=required).copy()
    if before > len(df):
        print(f"  Dropped {before - len(df):,} rows with missing base features")

    return df


def get_feature_cols(extra_features: list) -> list:
    cols = list(BASE_FEATURES)
    for ef in extra_features:
        cols.extend(OPTIONAL_FEATURES.get(ef, [ef]))
    return cols


# ── Metrics ──────────────────────────────────────────────────────────────────

def rmse(y_true, y_pred):
    return float(np.sqrt(((np.asarray(y_true) - np.asarray(y_pred)) ** 2).mean()))

def mae(y_true, y_pred):
    return float(np.abs(np.asarray(y_true) - np.asarray(y_pred)).mean())

def bias(y_true, y_pred):
    return float((np.asarray(y_pred) - np.asarray(y_true)).mean())

def over_under_acc(y_true, y_pred, ou_line):
    """Directional accuracy: did pred and actual agree on over/under the line?"""
    mask = ou_line.notna() & (y_true != ou_line)  # exclude pushes
    if mask.sum() == 0:
        return np.nan, 0
    pred_over = y_pred[mask] > ou_line[mask]
    actual_over = y_true[mask] > ou_line[mask]
    return float((pred_over == actual_over).mean()), int(mask.sum())


def evaluate(label: str, y_true, y_pred, ou_line=None):
    r = rmse(y_true, y_pred)
    m = mae(y_true, y_pred)
    b = bias(y_true, y_pred)
    print(f"\n  {'─'*72}")
    print(f"  {label}  (N={len(y_true):,})")
    print(f"  {'─'*72}")
    print(f"  RMSE: {r:.4f}   MAE: {m:.4f}   Bias: {b:+.4f}")
    print(f"  Pred range: [{np.min(y_pred):.2f}, {np.max(y_pred):.2f}]")
    print(f"  Actual range: [{np.min(y_true):.0f}, {np.max(y_true):.0f}]")
    if ou_line is not None:
        acc, n = over_under_acc(pd.Series(y_true), pd.Series(y_pred), pd.Series(ou_line))
        print(f"  O/U direction acc: {acc*100:.1f}% on {n} games with O/U line")
    return r


def print_comparison(test_rmse_raw, test_rmse_lgbm=None):
    print(f"\n  {'='*72}")
    print(f"  HEAD-TO-HEAD: v10-total vs baselines (2025 OOS)")
    print(f"  {'='*72}")
    print(f"  {'Model':<35} {'RMSE':>8}  {'vs v9':>10}  {'vs mean':>10}")
    print(f"  {'-'*65}")
    baselines = [
        ("Always-grand-mean",        GRAND_MEAN_RMSE),
        ("v9 total (reference bar)", V9_TOTAL_RMSE),
    ]
    for name, r in baselines:
        print(f"  {name:<35} {r:>8.4f}  {'—':>10}  {'—':>10}")
    models = [("v10-total linear (raw)", test_rmse_raw)]
    if test_rmse_lgbm:
        models.append(("v10-total LGBM (raw)", test_rmse_lgbm))
    for name, r in models:
        vs_v9 = r - V9_TOTAL_RMSE
        vs_mean = r - GRAND_MEAN_RMSE
        gate = "✓ BEATS V9" if r < V9_TOTAL_RMSE else "✗ misses bar"
        print(f"  {name:<35} {r:>8.4f}  {vs_v9:>+9.4f}  {vs_mean:>+9.4f}  {gate}")
    print(f"  {'-'*65}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="artifacts/")
    ap.add_argument("--train_end_season", type=int, default=2023)
    ap.add_argument("--val_season", type=int, default=2024)
    ap.add_argument("--test_season", type=int, default=2025)
    ap.add_argument("--earliest_season", type=int, default=2015)
    ap.add_argument("--extra-feature", dest="extra_features",
                    action="append", default=[],
                    choices=list(OPTIONAL_FEATURES.keys()),
                    help="Add optional feature (can repeat)")
    ap.add_argument("--use-lgbm", action="store_true",
                    help="Also train shallow LGBM and compare")
    args = ap.parse_args()

    pg_dsn = os.environ.get("PG_DSN")
    if not pg_dsn:
        print("ERROR: PG_DSN not set", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    engine = create_engine(pg_dsn)

    feature_cols = get_feature_cols(args.extra_features)
    print(f"\n{'='*72}")
    print(f"v10-total training")
    print(f"{'='*72}")
    print(f"Base features:  {BASE_FEATURES}")
    print(f"Extra features: {args.extra_features or 'none'}")
    print(f"All features:   {feature_cols}")
    print(f"LGBM:           {args.use_lgbm}")

    # ── Load ──
    print(f"\n[1/5] Loading data {args.earliest_season}–{args.test_season}...")
    raw = load_data(engine, args.earliest_season, args.test_season)

    print(f"\n[2/5] Engineering features...")
    df = engineer(raw, args.extra_features)

    # Print feature coverage
    print(f"\n  Feature coverage on full dataset ({len(df):,} games):")
    for col in feature_cols:
        if col in df.columns:
            n = df[col].notna().sum()
            print(f"    {col:<35} {n:>6}/{len(df):>6} ({n/len(df)*100:.1f}%)")

    train = df[df["season"] <= args.train_end_season].copy()
    val   = df[df["season"] == args.val_season].copy()
    test  = df[df["season"] == args.test_season].copy()
    print(f"\n  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")

    if len(test) == 0:
        print("ERROR: no test data", file=sys.stderr)
        sys.exit(1)

    # ── Naive baselines on test ──
    print(f"\n[3/5] Naive baselines on 2025 test set...")
    grand_mean = train["actual_total"].mean()
    print(f"  Train set grand mean: {grand_mean:.3f}")
    evaluate("Always grand-mean (train mean)", test["actual_total"],
             np.full(len(test), grand_mean))
    evaluate("league_avg_total × 2 (feature)",
             test["actual_total"], test["league_avg_total"])

    # ── Linear regression ──
    print(f"\n[4/5] Fitting Ridge regression...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_cols])
    X_val   = scaler.transform(val[feature_cols])
    X_test  = scaler.transform(test[feature_cols])
    y_train = train["actual_total"].values
    y_val   = val["actual_total"].values
    y_test  = test["actual_total"].values

    # Tune alpha on val set
    best_alpha, best_val_rmse = 1.0, np.inf
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        m = Ridge(alpha=alpha)
        m.fit(X_train, y_train)
        vr = rmse(y_val, m.predict(X_val))
        if vr < best_val_rmse:
            best_val_rmse = vr
            best_alpha = alpha

    print(f"  Best Ridge alpha: {best_alpha}  (val RMSE: {best_val_rmse:.4f})")
    linear = Ridge(alpha=best_alpha)
    linear.fit(X_train, y_train)

    print(f"\n  Coefficients (on standardized features):")
    for fname, coef in zip(feature_cols, linear.coef_):
        print(f"    {fname:<35} {coef:>+10.4f}")
    print(f"    {'intercept':<35} {linear.intercept_:>+10.4f}")

    p_val_lin  = linear.predict(X_val)
    p_test_lin = linear.predict(X_test)

    evaluate("Ridge — val 2024", y_val, p_val_lin)
    test_rmse_lin = evaluate("Ridge — test 2025 (headline)", y_test, p_test_lin)

    # ── Optional LGBM ──
    test_rmse_lgbm = None
    if args.use_lgbm:
        print(f"\n[4b/5] Fitting shallow LGBM...")
        lgbm_model = lgb.LGBMRegressor(
            objective="regression_l2",
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=16,
            min_data_in_leaf=80,
            reg_alpha=0.1,
            reg_lambda=5.0,
            subsample=0.8,
            feature_fraction=1.0,
            verbose=-1,
        )
        lgbm_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False),
                       lgb.log_evaluation(period=-1)],
        )
        n_trees = lgbm_model.best_iteration_
        print(f"  LGBM: {n_trees} trees (early stop)")
        print(f"  Feature importance (gain):")
        imp = lgbm_model.booster_.feature_importance(importance_type="gain")
        for fname, i in sorted(zip(feature_cols, imp), key=lambda x: -x[1]):
            print(f"    {fname:<35} {i:>10.1f}")

        p_test_lgbm = lgbm_model.predict(X_test)
        test_rmse_lgbm = evaluate("LGBM — test 2025", y_test, p_test_lgbm)

    # ── Head-to-head ──
    print_comparison(test_rmse_lin, test_rmse_lgbm)

    # ── Save ──
    print(f"\n[5/5] Saving artifacts...")
    extra_tag = ("_" + "_".join(args.extra_features)) if args.extra_features else ""
    stem = f"totals_v10{extra_tag}"

    bundle = {
        "scaler": scaler,
        "model": linear,
        "features": feature_cols,
        "extra_features": args.extra_features,
        "grand_mean": grand_mean,
        "train_end_season": args.train_end_season,
        "val_season": args.val_season,
        "test_season": args.test_season,
        "test_rmse": test_rmse_lin,
    }
    if args.use_lgbm and test_rmse_lgbm:
        bundle["lgbm_model"] = lgbm_model
        bundle["test_rmse_lgbm"] = test_rmse_lgbm

    joblib.dump(bundle, os.path.join(args.out_dir, f"{stem}.joblib"))

    # Save test predictions
    out_df = test[["game_id", "game_date", "home_team_id", "away_team_id",
                   "home_runs", "away_runs", "actual_total"] + feature_cols].copy()
    out_df["pred_total_linear"] = p_test_lin
    if test_rmse_lgbm:
        out_df["pred_total_lgbm"] = p_test_lgbm
    out_df.to_csv(os.path.join(args.out_dir, f"{stem}_2025_eval.csv"), index=False)

    metrics = {
        "features": feature_cols,
        "extra_features": args.extra_features,
        "ridge_alpha": best_alpha,
        "val_rmse_linear": float(best_val_rmse),
        "test_rmse_linear": float(test_rmse_lin),
        "test_rmse_lgbm": float(test_rmse_lgbm) if test_rmse_lgbm else None,
        "v9_reference_rmse": V9_TOTAL_RMSE,
        "grand_mean_rmse": GRAND_MEAN_RMSE,
        "beats_v9": test_rmse_lin < V9_TOTAL_RMSE,
    }
    with open(os.path.join(args.out_dir, f"{stem}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  {args.out_dir}{stem}.joblib")
    print(f"  {args.out_dir}{stem}_2025_eval.csv")
    print(f"  {args.out_dir}{stem}_metrics.json")
    print(f"\nDone.\n")


if __name__ == "__main__":
    main()