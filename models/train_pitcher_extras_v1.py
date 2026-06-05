"""
Train pitcher walks, hits allowed, and earned runs Poisson models (v1).

Usage:
  PG_DSN=... python models/train_pitcher_extras_v1.py
  PG_DSN=... python models/train_pitcher_extras_v1.py --only walks
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

# Allow `python models/train_pitcher_extras_v1.py` from repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
from scipy import stats

from models.pitcher_poisson_common import (
    ER_SPEC,
    HITS_SPEC,
    SP_ER_TARGETS_QUERY,
    SP_GAME_TARGETS_QUERY,
    SP_MIN_BF,
    WALKS_SPEC,
    get_engine,
    load_table,
    merge_base_frame,
    train_prop_model,
)


def _save_bundle(out_dir, spec, scaler, model, metrics, test, lam_te, lambda_scale=1.0, over_calib_factors=None):
    stem = spec.artifact.replace(".joblib", "")
    bundle = {
        "scaler": scaler,
        "model": model,
        "features": spec.features,
        "prop_type": spec.name,
        "metrics": metrics,
        "calib_thresholds": spec.calib_thresholds,
        "defaults": spec.defaults,
        "min_bf": SP_MIN_BF,
        "lambda_scale": lambda_scale,
        "over_calib_factors": over_calib_factors or {},
    }
    joblib.dump(bundle, os.path.join(out_dir, spec.artifact))

    out = test[["pitcher_id", "game_id", "game_date", "season", spec.target_col]].copy()
    out["lambda_pred"] = lam_te
    over_cols = {}
    lam = np.asarray(lam_te, dtype=float)
    factors = over_calib_factors or {}
    for thresh in spec.calib_thresholds:
        k_floor = int(thresh + 0.5)
        col = f"p_over_{str(thresh).replace('.', '_')}"
        factor = float(factors.get(str(thresh), factors.get(thresh, 1.0)))
        over_cols[col] = np.clip(1.0 - stats.poisson.cdf(k_floor - 1, lam), 0.0, 1.0) * factor
    out = pd.concat([out.reset_index(drop=True), pd.DataFrame(over_cols).reset_index(drop=True)], axis=1)
    out.to_csv(os.path.join(out_dir, f"{stem}_2024_eval.csv"), index=False)
    with open(os.path.join(out_dir, f"{stem}_metrics.json"), "w") as f:
        json.dump({"features": spec.features, "metrics": metrics}, f, indent=2)
    print(f"  Saved {spec.artifact}")


def _maybe_backfill_er_targets(args) -> None:
    if not args.backfill_er_targets:
        return
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [
        sys.executable,
        os.path.join(repo_root, "ingest", "backfill_pitcher_starts.py"),
        "--start",
        f"{args.earliest_season}-01-01",
    ]
    print(f"Backfilling starter ER box scores: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=repo_root, env=os.environ.copy())


def _ensure_er_targets(er_df, args) -> None:
    if len(er_df) > 0:
        return
    msg = (
        "No ER training rows: pitcher_starts.earned_runs is missing for historical SP starts.\n"
        f"Run: PG_DSN=... python ingest/backfill_pitcher_starts.py --start {args.earliest_season}-01-01\n"
        "Then re-run training, or pass --backfill-er-targets to fetch ER from the MLB box score API first."
    )
    needs_er = args.only in (None, "er")
    if needs_er:
        raise SystemExit(msg)


def train_one(spec, base_df, er_df, args):
    print(f"\n{'='*70}\nTraining {spec.name} model\n{'='*70}")
    if spec.name == "er":
        target_df = er_df.copy()
    else:
        target_df = base_df.copy()
    df = merge_base_frame(get_engine(), args.params, target_df)
    if spec.name == "er" and len(df) == 0:
        raise SystemExit(
            "ER merge produced 0 rows after joining features. "
            "Backfill earned_runs via ingest/backfill_pitcher_starts.py and retry."
        )
    scaler, model, metrics, test, lam_te, lambda_scale, over_calib_factors = train_prop_model(
        df,
        spec,
        train_end_season=args.train_end_season,
        val_season=args.val_season,
        test_season=args.test_season,
    )
    if args.max_calib_gap_pp is not None and metrics["max_calib_gap_pp"] > args.max_calib_gap_pp:
        print(
            f"  WARNING: max calibration gap {metrics['max_calib_gap_pp']:.1f}pp "
            f"> limit {args.max_calib_gap_pp:.1f}pp"
        )
        if args.require_calibration:
            raise SystemExit(f"{spec.name} model failed calibration gate")
    _save_bundle(
        args.out_dir, spec, scaler, model, metrics, test, lam_te,
        lambda_scale, over_calib_factors,
    )
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="artifacts/")
    ap.add_argument("--train_end_season", type=int, default=2022)
    ap.add_argument("--val_season", type=int, default=2023)
    ap.add_argument("--test_season", type=int, default=2024)
    ap.add_argument("--earliest_season", type=int, default=2015)
    ap.add_argument("--min-bf", type=int, default=SP_MIN_BF)
    ap.add_argument("--only", choices=["walks", "hits", "er"], default=None)
    ap.add_argument("--max-calib-gap-pp", type=float, default=8.0)
    ap.add_argument("--require-calibration", action="store_true")
    ap.add_argument(
        "--backfill-er-targets",
        action="store_true",
        help="Fetch missing starter earned_runs from MLB box scores before ER training.",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    args.params = {
        "start_date": f"{args.earliest_season}-01-01",
        "end_date": f"{args.test_season}-12-31",
        "min_bf": args.min_bf,
    }
    needs_er = args.only in (None, "er")
    if needs_er:
        _maybe_backfill_er_targets(args)

    engine = get_engine()

    print("Loading SP game targets...")
    base_df = load_table(engine, SP_GAME_TARGETS_QUERY, args.params, "SP targets")
    er_df = load_table(engine, SP_ER_TARGETS_QUERY, args.params, "SP ER targets")
    _ensure_er_targets(er_df, args)

    specs = [WALKS_SPEC, HITS_SPEC, ER_SPEC]
    if args.only:
        specs = [s for s in specs if s.name == args.only]

    summary = {}
    for spec in specs:
        summary[spec.name] = train_one(spec, base_df, er_df, args)

    with open(os.path.join(args.out_dir, "pitcher_extras_v1_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nDone.\n")


if __name__ == "__main__":
    main()
