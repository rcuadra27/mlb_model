#!/usr/bin/env python3
"""
calibrate.py

Fits and compares three calibration methods on 2023 out-of-sample predictions:
  1. Isotonic regression  — current approach, compresses tails aggressively
  2. Platt scaling        — logistic regression on raw probabilities, smooth sigmoid
  3. Beta calibration     — fits a beta distribution, preserves tail spread better

Evaluates all three on 2024 and 2025 held-out data, then saves the best
performer as the production calibrator.

Key problem being solved:
  Isotonic regression maps many different raw probabilities to the same output
  value (e.g. 0.562) for bad teams. This creates fake "edge" vs the market
  because the model appears to disagree by 15%+ when it's really just a
  calibration artifact. Platt scaling and beta calibration produce smooth,
  monotonic transformations that preserve the full range of raw probabilities.

Usage:
    PG_DSN=... python calibration/calibrate.py
    PG_DSN=... python calibration/calibrate.py \\
        --cal_start 2023-04-01 --cal_end 2023-09-30 \\
        --eval_years 2024 2025 \\
        --out artifacts/calibrator_v4.joblib
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin


# ---------------------------------------------------------------------------
# Beta calibration
# ---------------------------------------------------------------------------

class BetaCalibrator(BaseEstimator, TransformerMixin):
    """
    Beta calibration as described in Kull et al. (2017).
    Fits: p_cal = 1 / (1 + exp(-(a * log(p) + b * log(1-p) + c)))
    where a, b, c are fit by logistic regression on log-odds features.

    Advantages over isotonic:
    - Smooth, strictly monotonic transformation
    - Preserves the full range of raw probabilities
    - No step-function artifacts that cluster predictions at specific values
    - Better suited for sports win probabilities which have natural beta structure
    """

    def __init__(self):
        self.lr_ = None

    def _features(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.column_stack([
            np.log(p),
            np.log(1 - p),
        ])

    def fit(self, p: np.ndarray, y: np.ndarray):
        X = self._features(np.asarray(p, dtype=float))
        self.lr_ = LogisticRegression(C=1e9, solver="lbfgs", max_iter=1000)
        self.lr_.fit(X, np.asarray(y, dtype=float))
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        X = self._features(np.asarray(p, dtype=float))
        return self.lr_.predict_proba(X)[:, 1]

    def __call__(self, p):
        return self.predict(np.atleast_1d(p))


# ---------------------------------------------------------------------------
# Platt scaling
# ---------------------------------------------------------------------------

class PlattCalibrator(BaseEstimator, TransformerMixin):
    """
    Platt scaling: fits logistic regression on raw probability as single feature.
    p_cal = sigmoid(a * p_raw + b)
    Simpler than beta calibration but still smooth and monotonic.
    """

    def __init__(self):
        self.lr_ = None

    def fit(self, p: np.ndarray, y: np.ndarray):
        X = np.asarray(p, dtype=float).reshape(-1, 1)
        self.lr_ = LogisticRegression(C=1e9, solver="lbfgs", max_iter=1000)
        self.lr_.fit(X, np.asarray(y, dtype=float))
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        X = np.asarray(p, dtype=float).reshape(-1, 1)
        return self.lr_.predict_proba(X)[:, 1]

    def __call__(self, p):
        return self.predict(np.atleast_1d(p))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def brier_score(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def log_loss(p, y, eps=1e-7):
    p = np.clip(np.asarray(p), eps, 1 - eps)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def accuracy(p, y):
    return float(((np.asarray(p) > 0.5) == np.asarray(y).astype(bool)).mean())


def calibration_table(p, y, n_buckets=10):
    p, y = np.asarray(p), np.asarray(y)
    buckets = np.linspace(0, 1, n_buckets + 1)
    rows = []
    for i in range(n_buckets):
        mask = (p >= buckets[i]) & (p < buckets[i + 1])
        if mask.sum() == 0:
            continue
        rows.append({
            "bucket":   f"{buckets[i]:.0%}–{buckets[i+1]:.0%}",
            "n":        int(mask.sum()),
            "avg_pred": float(p[mask].mean()),
            "actual":   float(y[mask].mean()),
            "error":    float(p[mask].mean() - y[mask].mean()),
        })
    return pd.DataFrame(rows)


def print_calibration_table(ct: pd.DataFrame, label: str):
    print(f"\n  {label}:")
    print(f"  {'bucket':<10} {'n':>5} {'avg_pred':>9} {'actual':>8} {'error':>8}")
    print("  " + "-" * 44)
    for _, r in ct.iterrows():
        err_str = f"{r['error']:+.3f}"
        print(f"  {r['bucket']:<10} {r['n']:>5} {r['avg_pred']:>9.3f} "
              f"{r['actual']:>8.3f} {err_str:>8}")


def probability_spread(p, label):
    p = np.asarray(p)
    print(f"\n  {label}:")
    print(f"  {'pct':<6} {'value':>8}    range: {p.min():.3f} – {p.max():.3f}"
          f"    unique(3dp): {len(np.unique(np.round(p, 3)))}")
    for pct in [5, 25, 50, 75, 95]:
        print(f"  p{pct:<5} {np.percentile(p, pct):>8.3f}")


# ---------------------------------------------------------------------------
# Data loading — uses p_home_win_raw (pre-calibration Skellam probability)
# ---------------------------------------------------------------------------

def load_predictions(engine, schema, start, end):
    return pd.read_sql(text(f"""
        WITH latest AS (
            SELECT DISTINCT ON (game_id)
                game_id, game_date, p_home_win_raw
            FROM {schema}.inference_game_predictions
            WHERE game_date BETWEEN :start AND :end
              AND p_home_win_raw IS NOT NULL
            ORDER BY game_id, as_of_ts DESC
        )
        SELECT l.game_id, l.game_date, l.p_home_win_raw,
               g.home_win
        FROM latest l
        JOIN {schema}.games g USING (game_id)
        WHERE g.home_win IS NOT NULL
    """), engine, params={"start": start, "end": end})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema",     default="public")
    ap.add_argument("--cal_start",  default="2023-04-01")
    ap.add_argument("--cal_end",    default="2023-09-30")
    ap.add_argument("--eval_years", nargs="+", default=["2024", "2025"])
    ap.add_argument("--out",        default="artifacts/calibrator_v4.joblib")
    args = ap.parse_args()

    pg_dsn = os.getenv("PG_DSN")
    if not pg_dsn:
        raise RuntimeError("PG_DSN env var is required.")

    engine = create_engine(pg_dsn, pool_pre_ping=True)

    # ------------------------------------------------------------------
    # Load 2023 calibration set
    # ------------------------------------------------------------------
    print(f"Loading calibration data ({args.cal_start} → {args.cal_end})...")
    cal = load_predictions(engine, args.schema, args.cal_start, args.cal_end)
    print(f"  {len(cal)} games")

    p_raw_cal = cal["p_home_win_raw"].values
    y_cal     = cal["home_win"].astype(float).values

    # ------------------------------------------------------------------
    # Fit all three calibrators
    # ------------------------------------------------------------------
    print("\nFitting calibrators on 2023 data...")

    iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
    iso.fit(p_raw_cal, y_cal)
    print("  ✓ Isotonic regression")

    platt = PlattCalibrator()
    platt.fit(p_raw_cal, y_cal)
    print("  ✓ Platt scaling")

    beta = BetaCalibrator()
    beta.fit(p_raw_cal, y_cal)
    print("  ✓ Beta calibration")

    calibrators = {
        "Isotonic": (iso,   lambda p: iso.predict(np.atleast_1d(p))),
        "Platt":    (platt, platt.predict),
        "Beta":     (beta,  beta.predict),
    }

    # ------------------------------------------------------------------
    # Show probability spread on 2023 — key diagnostic for compression
    # ------------------------------------------------------------------
    print("\n--- Probability spread on 2023 calibration set ---")
    print("(watch for 'unique' count — isotonic collapses many values into few)")
    probability_spread(p_raw_cal, "Raw Skellam")
    for name, (_, pred_fn) in calibrators.items():
        probability_spread(pred_fn(p_raw_cal), name)

    # ------------------------------------------------------------------
    # Evaluate on each held-out year
    # ------------------------------------------------------------------
    year_scores = {name: [] for name in calibrators}

    for year in args.eval_years:
        start = f"{year}-04-01"
        end   = f"{year}-09-30"

        print(f"\n{'=' * 65}")
        print(f"HELD-OUT: {year}")
        print("=" * 65)

        ev = load_predictions(engine, args.schema, start, end)
        if ev.empty:
            print(f"  No data for {year} — skipping")
            continue
        print(f"  {len(ev)} games")

        p_raw_ev = ev["p_home_win_raw"].values
        y_ev     = ev["home_win"].astype(float).values

        # Metrics table
        raw_brier = brier_score(p_raw_ev, y_ev)
        print(f"\n  {'Method':<12} {'Accuracy':>10} {'Brier':>8} "
              f"{'LogLoss':>9} {'ΔBrier':>8}")
        print("  " + "-" * 50)
        print(f"  {'Raw':<12} {accuracy(p_raw_ev, y_ev):>10.4f} "
              f"{raw_brier:>8.4f} {log_loss(p_raw_ev, y_ev):>9.4f} {'—':>8}")

        for name, (_, pred_fn) in calibrators.items():
            p_c  = pred_fn(p_raw_ev)
            bs   = brier_score(p_c, y_ev)
            year_scores[name].append(bs)
            print(f"  {name:<12} {accuracy(p_c, y_ev):>10.4f} "
                  f"{bs:>8.4f} {log_loss(p_c, y_ev):>9.4f} "
                  f"{bs - raw_brier:>+8.4f}")

        # Calibration tables
        print(f"\n  --- Calibration tables ({year}) ---")
        print_calibration_table(calibration_table(p_raw_ev, y_ev), "Raw")
        for name, (_, pred_fn) in calibrators.items():
            print_calibration_table(
                calibration_table(pred_fn(p_raw_ev), y_ev), name)

        # Spread
        print(f"\n  --- Probability spread ({year}) ---")
        probability_spread(p_raw_ev, "Raw")
        for name, (_, pred_fn) in calibrators.items():
            probability_spread(pred_fn(p_raw_ev), name)

    # ------------------------------------------------------------------
    # Pick winner by average Brier score
    # ------------------------------------------------------------------
    print(f"\n{'=' * 65}")
    print("SUMMARY — Average Brier score across eval years")
    print("=" * 65)
    print(f"  {'Method':<12} {'Avg Brier':>12}")
    print("  " + "-" * 26)

    avg_scores = {
        name: float(np.mean(scores)) if scores else float("inf")
        for name, scores in year_scores.items()
    }
    ranked = sorted(avg_scores.items(), key=lambda x: x[1])
    for rank, (name, score) in enumerate(ranked, 1):
        marker = "  ← winner" if rank == 1 else ""
        print(f"  {name:<12} {score:>12.6f}{marker}")

    winner_name, _ = ranked[0]
    winner_clf     = calibrators[winner_name][0]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump(winner_clf, args.out)
    print(f"\nSaved winner ({winner_name}) → {args.out}")

    # Save all three for reference
    base = args.out.replace(".joblib", "")
    for name, (clf, _) in calibrators.items():
        path = f"{base}_{name.lower()}.joblib"
        joblib.dump(clf, path)
        print(f"Saved {name} → {path}")

    print(f"\nIf the winner changed from isotonic, update inference.py:")
    print(f"  --calibrator {args.out}")


if __name__ == "__main__":
    main()