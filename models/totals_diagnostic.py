"""
v10-total diagnostic — establish naive baselines before building anything.

Answers:
  1. What RMSE does "always predict league average" get? (true floor)
  2. What RMSE does "always predict the O/U line" get? (market floor)
  3. What RMSE did v9's run model get on 2025?
  4. What does the actual total distribution look like?
  5. What's the O/U line accuracy (over/under hit rate)?

Uses:
  - v9_2025_holdout_eval.csv  (has pred_home_runs, pred_away_runs, actual runs)
  - baseline_v10_production_2025_eval.csv (has actual home_runs, away_runs)
  - features_game (via Postgres for O/U lines + env features)

Run:
  PG_DSN=... python models/totals_diagnostic.py \
      --v9_csv artifacts/v9_2025_holdout_eval.csv \
      --v10_csv artifacts/baseline_v10_production_2025_eval.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text


OU_QUERY = """
SELECT
    fg.game_id,
    fg.closing_ou_line,
    fg.morning_ou_line,
    COALESCE(fg.closing_ou_line, fg.morning_ou_line) AS ou_line,
    fg.league_avg_runs_60d,
    fg.total_offense_env,
    fg.total_defense_env,
    fg.park_runs_factor_blended,
    fg.park_runs_factor,
    fg.forecast_temp_f,
    fg.forecast_wind_mph,
    fg.forecast_wind_dir_deg,
    fg.umpire_runs_boost,
    fg.umpire_n_games
FROM public.features_game fg
JOIN public.games g ON g.game_id = fg.game_id
WHERE g.game_date >= '2025-01-01'
  AND g.game_date < '2026-01-01';
"""


def rmse(y_true, y_pred):
    return np.sqrt(((y_true - y_pred) ** 2).mean())


def mae(y_true, y_pred):
    return np.abs(y_true - y_pred).mean()


def bias(y_true, y_pred):
    return (y_pred - y_true).mean()


def print_metrics(label, y_true, y_pred, n=None):
    n = n or len(y_true)
    print(f"  {label:<40s}  N={n:>5}  "
          f"RMSE={rmse(y_true, y_pred):.4f}  "
          f"MAE={mae(y_true, y_pred):.4f}  "
          f"Bias={bias(y_true, y_pred):+.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v9_csv", default="artifacts/v9_2025_holdout_eval.csv")
    ap.add_argument("--v10_csv", default="artifacts/baseline_v10_production_2025_eval.csv")
    args = ap.parse_args()

    pg_dsn = os.environ.get("PG_DSN")
    if not pg_dsn:
        print("ERROR: PG_DSN not set", file=sys.stderr)
        sys.exit(1)

    v9 = pd.read_csv(args.v9_csv)
    v10 = pd.read_csv(args.v10_csv)
    print(f"\nv9 holdout:  {len(v9):,} games")
    print(f"v10 holdout: {len(v10):,} games")

    v9["actual_total"] = v9["actual_home_runs"] + v9["actual_away_runs"]
    v9["pred_total_v9"] = v9["pred_home_runs"] + v9["pred_away_runs"]
    v10["actual_total"] = v10["home_runs"] + v10["away_runs"]
    v10_notie = v10[v10["home_runs"] != v10["away_runs"]].copy()

    print("\nLoading O/U lines and features from Postgres...")
    engine = create_engine(pg_dsn)
    with engine.connect() as conn:
        feat = pd.read_sql(text(OU_QUERY), conn)
    print(f"Feature rows loaded: {len(feat):,}")
    print(f"  Has closing_ou_line: {feat['closing_ou_line'].notna().sum():,}")
    print(f"  Has morning_ou_line: {feat['morning_ou_line'].notna().sum():,}")
    print(f"  Has ou_line (either): {feat['ou_line'].notna().sum():,}")
    print(f"  Has league_avg_runs_60d: {feat['league_avg_runs_60d'].notna().sum():,}")
    print(f"  Has total_offense_env: {feat['total_offense_env'].notna().sum():,}")
    print(f"  Has total_defense_env: {feat['total_defense_env'].notna().sum():,}")
    print(f"  Has forecast_temp_f: {feat['forecast_temp_f'].notna().sum():,}")
    print(f"  Has forecast_wind_mph: {feat['forecast_wind_mph'].notna().sum():,}")
    print(f"  Has umpire_runs_boost: {feat['umpire_runs_boost'].notna().sum():,}")

    v9m = v9.merge(feat, on="game_id", how="inner")
    v10m = v10_notie.merge(feat, on="game_id", how="inner")
    print(f"\nAfter merge with features:")
    print(f"  v9:  {len(v9m):,} games")
    print(f"  v10: {len(v10m):,} games")

    if len(v9m) == 0:
        print("ERROR: v9 CSV did not merge with Postgres features — check game_id overlap",
              file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*70}")
    print("ACTUAL TOTAL RUNS — 2025 DISTRIBUTION")
    print(f"{'='*70}")
    at = v9m["actual_total"]
    print(f"  Mean:   {at.mean():.3f}")
    print(f"  Median: {at.median():.3f}")
    print(f"  Std:    {at.std():.3f}")
    print(f"  Min:    {at.min():.0f}  Max: {at.max():.0f}")
    print("  Percentiles:")
    for p in [10, 25, 50, 75, 90, 95]:
        print(f"    {p:>3}%: {at.quantile(p/100):.1f}")

    print(f"\n{'='*70}")
    print("NAIVE BASELINES — total runs prediction")
    print(f"{'='*70}")

    grand_mean_actual = v9m["actual_total"].mean()
    print(f"\n  Grand mean (2025 actual): {grand_mean_actual:.3f}")
    print_metrics("Always predict grand mean (2025)",
                  v9m["actual_total"], np.full(len(v9m), grand_mean_actual))

    has_league = v9m["league_avg_runs_60d"].notna()
    if has_league.sum() > 0:
        sub = v9m[has_league]
        league_pred = sub["league_avg_runs_60d"] * 2
        print_metrics("league_avg_runs_60d × 2",
                      sub["actual_total"], league_pred, n=len(sub))
    else:
        print("  league_avg_runs_60d: no data")

    has_ou = v9m["ou_line"].notna()
    if has_ou.sum() > 0:
        sub = v9m[has_ou]
        print_metrics("Closing/morning O/U line (market)",
                      sub["actual_total"], sub["ou_line"], n=len(sub))
    else:
        print("  O/U line: no data — check odds tables")

    print_metrics("v9 pred_total (sum of team run preds)",
                  v9m["actual_total"], v9m["pred_total_v9"])

    print(f"\n{'='*70}")
    print("O/U LINE ACCURACY — how often does actual beat/miss the line?")
    print(f"{'='*70}")
    if has_ou.sum() > 0:
        sub = v9m[has_ou].copy()
        sub["over"] = (sub["actual_total"] > sub["ou_line"]).astype(int)
        sub["push"] = (sub["actual_total"] == sub["ou_line"]).astype(int)
        sub["under"] = (sub["actual_total"] < sub["ou_line"]).astype(int)
        print(f"  Games: {len(sub):,}")
        print(f"  Over:  {sub['over'].sum():>4} ({sub['over'].mean()*100:.1f}%)")
        print(f"  Push:  {sub['push'].sum():>4} ({sub['push'].mean()*100:.1f}%)")
        print(f"  Under: {sub['under'].sum():>4} ({sub['under'].mean()*100:.1f}%)")
        print(f"  Mean actual:  {sub['actual_total'].mean():.3f}")
        print(f"  Mean ou_line: {sub['ou_line'].mean():.3f}")
        print(f"  Bias of line: {(sub['ou_line'] - sub['actual_total']).mean():+.3f} runs")
        print("\n  O/U line residual distribution:")
        resid = sub["actual_total"] - sub["ou_line"]
        for p in [10, 25, 50, 75, 90]:
            print(f"    {p:>3}%: {resid.quantile(p/100):+.2f}")

    print(f"\n{'='*70}")
    print("FEATURE AVAILABILITY FOR v10-TOTAL")
    print(f"{'='*70}")
    features_to_check = [
        "league_avg_runs_60d",
        "total_offense_env",
        "total_defense_env",
        "park_runs_factor_blended",
        "forecast_temp_f",
        "forecast_wind_mph",
        "forecast_wind_dir_deg",
        "umpire_runs_boost",
    ]
    print(f"  {'Feature':<35} {'N':>6}  {'%':>6}  {'Mean':>8}  {'Std':>8}")
    for col in features_to_check:
        if col in feat.columns:
            n = feat[col].notna().sum()
            pct = n / len(feat) * 100 if len(feat) else 0
            mn = feat[col].mean() if n > 0 else np.nan
            sd = feat[col].std() if n > 0 else np.nan
            print(f"  {col:<35} {n:>6}  {pct:>5.1f}%  {mn:>8.3f}  {sd:>8.3f}")
        else:
            print(f"  {col:<35} NOT IN SCHEMA")

    print(f"\n{'='*70}")
    print("WIND DIRECTION — distribution (0=N, 90=E, 180=S, 270=W)")
    print(f"{'='*70}")
    if feat["forecast_wind_dir_deg"].notna().sum() > 0:
        wd = feat["forecast_wind_dir_deg"].dropna()
        print(f"  N games with wind dir: {len(wd):,}")
        print(f"  Range: [{wd.min():.0f}°, {wd.max():.0f}°]")
        wd_rad = np.deg2rad(wd)
        print(f"  sin(dir) mean: {np.sin(wd_rad).mean():+.3f}")
        print(f"  cos(dir) mean: {np.cos(wd_rad).mean():+.3f}")
    else:
        print("  No wind direction data")

    print(f"\n{'='*70}")
    print("FEATURE CORRELATIONS WITH ACTUAL TOTAL RUNS")
    print(f"{'='*70}")
    corr_features = [
        "league_avg_runs_60d", "total_offense_env", "total_defense_env",
        "park_runs_factor_blended", "forecast_temp_f", "forecast_wind_mph",
        "umpire_runs_boost", "ou_line",
    ]
    v9m["league_avg_total"] = v9m["league_avg_runs_60d"] * 2
    corr_features_ext = corr_features + ["league_avg_total", "pred_total_v9"]
    print(f"  {'Feature':<35} {'Corr':>8}  {'N':>6}")
    for col in corr_features_ext:
        if col in v9m.columns:
            sub = v9m[["actual_total", col]].dropna()
            if len(sub) > 10:
                c = sub.corr().iloc[0, 1]
                print(f"  {col:<35} {c:>+8.4f}  {len(sub):>6}")

    print(f"\n{'='*70}")
    print("SUMMARY — what to beat")
    print(f"{'='*70}")
    grand = v9m["actual_total"].mean()
    rmse_grand = rmse(v9m["actual_total"], np.full(len(v9m), grand))
    rmse_v9 = rmse(v9m["actual_total"], v9m["pred_total_v9"])
    rmse_ou = (rmse(v9m.loc[has_ou, "actual_total"], v9m.loc[has_ou, "ou_line"])
               if has_ou.sum() > 0 else np.nan)
    rmse_lg = (rmse(v9m.loc[has_league, "actual_total"],
                    v9m.loc[has_league, "league_avg_runs_60d"] * 2)
               if has_league.sum() > 0 else np.nan)

    print(f"  Always-grand-mean RMSE:     {rmse_grand:.4f}  ← true floor")
    print(f"  league_avg_runs_60d×2 RMSE: {rmse_lg:.4f}  ← rolling floor")
    print(f"  O/U line RMSE:              {rmse_ou:.4f}  ← market floor (hardest to beat)")
    print(f"  v9 total RMSE:              {rmse_v9:.4f}  ← bar to beat")
    print()
    print("  v10-total must beat v9 RMSE on 2025 OOS.")
    print("  Getting within 0.10 of O/U line RMSE would be exceptional.")
    print("  O/U recommendation only needs: pred_total vs morning_ou_line direction.")


if __name__ == "__main__":
    main()
