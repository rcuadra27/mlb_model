"""
inference_v10_total.py — Production inference for v10-total (over/under model).

Runs after inference_v10.py in the morning pipeline.
Predicts total runs, compares vs morning_ou_line, writes ou_recommendation
and fills home_runs_pred / away_runs_pred in inference_game_predictions.

Features (must match training exactly):
    league_avg_total      = league_avg_runs_60d × 2
    total_offense_env     from features_game
    total_defense_env     from features_game
    park_runs_factor      park_runs_factor_blended ?? park_runs_factor
    umpire_runs_boost     features_game (0 if < 20 games history)
    sp_xwoba_total        home_sp_xwoba_against_90 + away_sp_xwoba_against_90

O/U recommendation logic:
    pred_total > morning_ou_line + EDGE_THRESHOLD → 'over'
    pred_total < morning_ou_line - EDGE_THRESHOLD → 'under'
    otherwise                                     → 'none' (no edge)

Usage (add to run_daily.sh after inference_v10.py):
    python inference/inference_v10_total.py --date $TODAY \\
        --model artifacts/totals_v10_umpire_runs_boost_sp_xwoba_total.joblib
"""
import argparse
import math
import os
import sys
import logging
from datetime import date, datetime

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

try:
    from inference.run_split import split_total
except ImportError:
    from run_split import split_total

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

XWOBA_DEFAULT      = 0.320   # fallback when SP xwOBA is NULL
LEAGUE_AVG_DEFAULT = 9.0     # fallback when league_avg_runs_60d is NULL
PARK_DEFAULT       = 1.0     # neutral park
UMPIRE_MIN_GAMES   = 20      # min games before trusting umpire tendency

# Edge threshold for O/U recommendation (in runs).
# Only flag over/under when model disagrees with line by this much.
# 0.3 runs is conservative — about 1 standard error on the prediction.
EDGE_THRESHOLD = 0.30

# Rolling window for model-vs-line bias (pred tends to sit above morning totals).
LINE_BIAS_LOOKBACK_DAYS = 21
LINE_BIAS_MIN_GAMES = 40
# Fallback when history is thin (recent prod ~0.35 runs vs morning line).
LINE_BIAS_DEFAULT = 0.35
LINE_BIAS_MAX = 0.75

# These must match training feature order in the joblib bundle exactly.
EXPECTED_FEATURES = [
    "league_avg_total",
    "total_offense_env",
    "total_defense_env",
    "park_runs_factor",
    "umpire_runs_boost",
    "sp_xwoba_total",
]

# ── SQL ───────────────────────────────────────────────────────────────────────

GAMES_QUERY = """
SELECT
    g.game_id,
    g.game_date,
    g.home_team_id,
    g.away_team_id,
    th.team_name  AS home_team,
    ta.team_name  AS away_team,
    fg.league_avg_runs_60d,
    fg.total_offense_env,
    fg.total_defense_env,
    COALESCE(fg.park_runs_factor_blended, fg.park_runs_factor) AS park_runs_factor,
    fg.umpire_runs_boost,
    fg.umpire_n_games,
    fg.home_sp_xwoba_against_90,
    fg.away_sp_xwoba_against_90,
    fg.morning_ou_line,
    fg.closing_ou_line,
    fg.home_avg_runs_scored_60,
    fg.away_avg_runs_scored_60,
    fg.home_avg_runs_allowed_60,
    fg.away_avg_runs_allowed_60,
    fg.forecast_temp_f,
    fg.forecast_wind_mph,
    fg.forecast_wind_dir_deg
FROM {schema}.games g
JOIN {schema}.features_game fg
    ON fg.game_id = g.game_id
JOIN {schema}.teams th ON th.mlb_team_id = g.home_team_id
JOIN {schema}.teams ta ON ta.mlb_team_id = g.away_team_id
WHERE g.game_date = :d
ORDER BY g.game_id;
"""

LINE_BIAS_QUERY = """
SELECT AVG(p.total_runs_pred - COALESCE(fg.morning_ou_line, fg.closing_ou_line, p.ou_line)) AS bias
FROM {schema}.inference_game_predictions p
JOIN {schema}.games g ON g.game_id = p.game_id AND g.game_date = p.game_date
LEFT JOIN {schema}.features_game fg ON fg.game_id = p.game_id
WHERE p.game_date >= CAST(:d AS date) - INTERVAL '{days} days'
  AND p.game_date < CAST(:d AS date)
  AND p.total_runs_pred IS NOT NULL
  AND COALESCE(fg.morning_ou_line, fg.closing_ou_line, p.ou_line) IS NOT NULL
  AND g.home_runs IS NOT NULL
  AND g.away_runs IS NOT NULL
  AND (
      LOWER(COALESCE(g.status, '')) LIKE 'final%%'
      OR LOWER(COALESCE(g.status, '')) = 'game over'
      OR LOWER(COALESCE(g.status, '')) LIKE 'completed%%'
  );
"""

# Update the row written by inference_v10.py — same game_id / as_of_ts,
# so we update rather than insert.
UPDATE_QUERY = """
UPDATE {schema}.inference_game_predictions AS p
SET
    home_runs_pred     = :home_runs_pred,
    away_runs_pred     = :away_runs_pred,
    total_runs_pred    = :total_runs_pred,
    ou_recommendation  = :ou_recommendation,
    ou_edge_over       = :ou_edge_over,
    ou_edge_under      = :ou_edge_under
FROM (
    SELECT DISTINCT ON (game_id) game_id, as_of_ts
    FROM {schema}.inference_game_predictions
    WHERE game_date = CAST(:game_date AS date)
    ORDER BY game_id, as_of_ts DESC
) AS latest
WHERE p.game_id = latest.game_id
  AND p.as_of_ts = latest.as_of_ts
  AND p.game_id = :game_id
  AND p.game_id NOT IN (
      SELECT game_id FROM {schema}.games
      WHERE LOWER(COALESCE(status,'')) IN ('final','game over','completed early')
  );
"""


# ── Feature engineering ───────────────────────────────────────────────────────

def build_features(today: pd.DataFrame) -> pd.DataFrame:
    df = today.copy()

    df["league_avg_total"] = (df["league_avg_runs_60d"] * 2).fillna(LEAGUE_AVG_DEFAULT)
    df["total_offense_env"] = df["total_offense_env"].fillna(0.0)
    df["total_defense_env"] = df["total_defense_env"].fillna(0.0)
    df["park_runs_factor"]  = df["park_runs_factor"].fillna(PARK_DEFAULT)

    # Umpire: only trust with >= UMPIRE_MIN_GAMES history
    df["umpire_runs_boost"] = np.where(
        df["umpire_n_games"].fillna(0) >= UMPIRE_MIN_GAMES,
        df["umpire_runs_boost"].fillna(0.0),
        0.0,
    )

    # SP xwOBA sum — higher = worse combined pitching = more runs expected
    home_x = df["home_sp_xwoba_against_90"].fillna(XWOBA_DEFAULT)
    away_x = df["away_sp_xwoba_against_90"].fillna(XWOBA_DEFAULT)
    df["sp_xwoba_total"] = home_x + away_x

    # Log nulls that were filled
    for col in ["home_sp_xwoba_against_90", "away_sp_xwoba_against_90",
                "total_offense_env", "total_defense_env"]:
        n_null = today[col].isna().sum()
        if n_null:
            log.warning(f"  {col}: {n_null} NULL → default")

    return df


# ── O/U recommendation ────────────────────────────────────────────────────────

def fetch_line_bias_correction(engine, schema: str, as_of_date: str, lookback_days: int = LINE_BIAS_LOOKBACK_DAYS) -> float:
    """
    Mean (model total − morning O/U line) on recent completed games.
    Subtracted from raw predictions so O/U edges center on the market, not only 'Over'.
    """
    q = LINE_BIAS_QUERY.format(schema=schema, days=int(lookback_days))
    with engine.connect() as conn:
        row = conn.execute(text(q), {"d": as_of_date}).fetchone()
    if not row or row[0] is None:
        log.info(f"  Line bias: no history → default {LINE_BIAS_DEFAULT:.2f} runs")
        return LINE_BIAS_DEFAULT
    bias = float(row[0])
    if math.isnan(bias) or math.isinf(bias):
        return LINE_BIAS_DEFAULT
    bias = max(0.0, min(LINE_BIAS_MAX, bias))
    log.info(f"  Line bias correction (last {lookback_days}d): {bias:.3f} runs")
    return bias


def make_ou_recommendation(pred_total: float, ou_line: float, line_bias: float = 0.0) -> tuple:
    """
    Returns (recommendation, edge_runs).
    recommendation: 'over' | 'under' | 'none'
    edge_runs: (pred_total - line_bias) - ou_line (positive = lean over)
    """
    if pd.isna(ou_line):
        return None, np.nan
    edge = pred_total - line_bias - ou_line
    if edge > EDGE_THRESHOLD:
        return "OVER", round(edge, 3)
    elif edge < -EDGE_THRESHOLD:
        return "UNDER", round(abs(edge), 3)
    return None, round(edge, 3)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="v10-total O/U inference")
    ap.add_argument("--date", default=str(date.today()))
    ap.add_argument("--model",
                    default="artifacts/totals_v10_umpire_runs_boost_sp_xwoba_total.joblib")
    ap.add_argument("--schema", default="public")
    ap.add_argument("--edge-threshold", type=float, default=EDGE_THRESHOLD,
                    help="Minimum run edge to flag over/under (default 0.30)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print predictions without writing to Postgres")
    args = ap.parse_args()

    pg_dsn = os.environ.get("PG_DSN")
    if not pg_dsn:
        log.error("PG_DSN not set")
        sys.exit(1)

    edge_threshold = args.edge_threshold
    log.info(f"v10-total inference — date={args.date}  threshold={edge_threshold} runs")

    # ── Load model ──
    log.info(f"Loading model: {args.model}")
    bundle = joblib.load(args.model)
    features = bundle["features"]
    assert features == EXPECTED_FEATURES, \
        f"Feature mismatch.\nExpected: {EXPECTED_FEATURES}\nGot:      {features}"
    scaler = bundle["scaler"]
    model  = bundle["model"]
    log.info(f"Model loaded — features: {features}")
    log.info(f"Training RMSE: {bundle.get('test_rmse', 'N/A'):.4f} (2025 OOS)")

    engine = create_engine(pg_dsn)
    line_bias = fetch_line_bias_correction(engine, args.schema, args.date)
    bundle_bias = bundle.get("line_bias_correction")
    if bundle_bias is not None:
        try:
            line_bias = max(0.0, min(LINE_BIAS_MAX, float(bundle_bias)))
            log.info(f"  Line bias from model bundle: {line_bias:.3f} runs")
        except (TypeError, ValueError):
            pass

    # ── Load today's games ──
    log.info("Loading today's games from features_game...")
    with engine.connect() as conn:
        today = pd.read_sql(
            text(GAMES_QUERY.format(schema=args.schema)),
            conn, params={"d": args.date}
        )
    log.info(f"Games: {len(today)} scheduled for {args.date}")
    if len(today) == 0:
        log.warning("No games — exiting")
        sys.exit(0)

    # O/U line availability
    has_morning_ou = today["morning_ou_line"].notna().sum()
    has_closing_ou = today["closing_ou_line"].notna().sum()
    log.info(f"O/U lines — morning: {has_morning_ou}/{len(today)}, "
             f"closing: {has_closing_ou}/{len(today)}")

    # Use morning line for recommendation (available at inference time)
    today["ou_line"] = today["morning_ou_line"]

    # ── Build features ──
    log.info("Building features...")
    df = build_features(today)

    # ── Predict ──
    log.info("Running v10-total model...")
    X = scaler.transform(df[features])
    pred_totals = model.predict(X)

    # ── Build results ──
    results = []
    log.info("\nPredictions:")
    log.info(f"  {'Game':<35} {'Pred':>6}  {'Line':>6}  {'Edge':>6}  {'Rec':<6}")
    log.info(f"  {'─'*62}")

    for i, (_, row) in enumerate(df.iterrows()):
        pred_raw = float(pred_totals[i])
        pred_total = round(pred_raw - line_bias, 3)
        ou_line    = row["ou_line"]
        rec, edge  = make_ou_recommendation(pred_raw, ou_line, line_bias)
        home_pred, away_pred = split_total(
            pred_total,
            away_runs_scored=row.get("away_avg_runs_scored_60"),
            home_runs_scored=row.get("home_avg_runs_scored_60"),
            away_runs_allowed=row.get("away_avg_runs_allowed_60"),
            home_runs_allowed=row.get("home_avg_runs_allowed_60"),
            league_avg_runs=row.get("league_avg_runs_60d"),
        )
        ou_edge_over = edge if rec == "OVER" else None
        ou_edge_under = edge if rec == "UNDER" else None
        ou_line_out = None if pd.isna(ou_line) else round(float(ou_line), 3)

        game_label = f"{row['away_team']} @ {row['home_team']}"
        line_str   = f"{ou_line:.1f}" if not pd.isna(ou_line) else " N/A"
        edge_str   = f"{edge:+.2f}" if not pd.isna(edge) else "  N/A"
        rec_str    = rec or "none"
        log.info(f"  {game_label:<35} {pred_total:>5.2f}  {line_str:>6}  "
                 f"{edge_str:>6}  {rec_str:<6}")

        results.append({
            "game_id":           row["game_id"],
            "game_date":         args.date,
            "home_runs_pred":    home_pred,
            "away_runs_pred":    away_pred,
            "total_runs_pred":   pred_total,
            "ou_line":           ou_line_out,
            "ou_recommendation": rec,
            "ou_edge_over":      ou_edge_over,
            "ou_edge_under":     ou_edge_under,
        })

    results_df = pd.DataFrame(results)

    # Summary
    over_ct  = (results_df["ou_recommendation"] == "OVER").sum()
    under_ct = (results_df["ou_recommendation"] == "UNDER").sum()
    none_ct  = results_df["ou_recommendation"].isna().sum()
    log.info(f"\n  O/U summary: {over_ct} over, {under_ct} under, {none_ct} no-edge")
    log.info(f"  Pred total range: [{pred_totals.min():.2f}, {pred_totals.max():.2f}]")
    log.info(f"  Mean pred total: {pred_totals.mean():.2f}")

    if args.dry_run:
        log.info("DRY RUN — not writing to Postgres")
        print(results_df.to_string(index=False))
        return

    # ── Write to Postgres ──
    # Update rows already written by inference_v10.py (latest snapshot per game).
    log.info(f"Updating {len(results_df)} rows in inference_game_predictions...")
    updated = 0
    with engine.begin() as conn:
        for _, row in results_df.iterrows():
            result = conn.execute(
                text(UPDATE_QUERY.format(schema=args.schema)),
                {
                    "game_id":           int(row["game_id"]),
                    "game_date":         args.date,
                    "home_runs_pred":    row["home_runs_pred"],
                    "away_runs_pred":    row["away_runs_pred"],
                    "total_runs_pred":   row["total_runs_pred"],
                    "ou_line":           row["ou_line"],
                    "ou_recommendation": row["ou_recommendation"],
                    "ou_edge_over":      row["ou_edge_over"],
                    "ou_edge_under":     row["ou_edge_under"],
                }
            )
            updated += result.rowcount

    log.info(f"Updated {updated}/{len(results_df)} rows")
    if updated < len(results_df):
        log.warning(f"{len(results_df) - updated} games not updated "
                    f"(already final or not yet in inference_game_predictions)")
    log.info("Done.")


if __name__ == "__main__":
    main()