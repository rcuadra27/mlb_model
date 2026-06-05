"""
inference_v10.py — Production inference for v10 moneyline model.

Replaces inference.py for ML win-probability predictions.
Writes to inference_game_predictions (Postgres) using the same column names
as v9 so export_to_bigquery.py and the dashboard need zero changes.

Key differences from v9:
  - No LightGBM run model, no Skellam win probability
  - Direct 6-feature LGBM → p_home_win
  - Season-to-date features computed at inference time from games table
  - Statcast features (sp_xwoba, lineup_xwoba) pulled from features_game
    with 0.320 fallback if NULL (matches training behavior)
  - O/U predictions (home_runs_pred, away_runs_pred, total_runs_pred) are
    written as NULL — v10-total will fill these when built

Usage (drop-in replacement in run_daily.sh):
  python inference/inference_v10.py --date $TODAY \
      --model artifacts/baseline_v10_production.joblib \
      [--schema public] [--fill_missing]
"""
import argparse
import os
import sys
import logging
from datetime import date, datetime

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from lineup_utils import confirmed_lineup_game_ids

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

# Fallback xwOBA when Statcast features are NULL (matches baseline.py training default)
XWOBA_DEFAULT = 0.320

# Fallback park factor
PARK_FACTOR_DEFAULT = 1.0

# Bayesian shrinkage prior — must match baseline.py exactly
PRIOR_GAMES = 30
PRIOR_WIN_PCT = 0.5

# Model feature names — must match artifacts/baseline_v10_production.joblib exactly
V10_FEATURES = [
    "win_pct_diff",
    "run_diff_pg_diff",
    "sp_xwoba_diff",
    "lineup_xwoba_diff",
    "park_factor",
    "is_home_const",
]


# ── Data loading ─────────────────────────────────────────────────────────────

TODAY_GAMES_QUERY = """
SELECT
    g.game_id,
    g.game_date,
    g.home_team_id,
    g.away_team_id,
    th.team_name  AS home_team,
    ta.team_name  AS away_team,
    -- Statcast features (may be NULL if --skip-statcast was used)
    fg.home_sp_xwoba_against_90,
    fg.away_sp_xwoba_against_90,
    fg.home_lineup_xwoba_90,
    fg.away_lineup_xwoba_90,
    fg.park_runs_factor_blended,
    fg.park_runs_factor,
    -- Market odds for edge / value flag computation
    fg.morning_p_home,
    fg.closing_p_home,
    fg.morning_home_price,
    fg.morning_away_price,
    fg.closing_home_price,
    fg.closing_away_price,
    fg.sharp_action_home,
    fg.line_move_magnitude,
    gsp.home_sp_name,
    gsp.away_sp_name
FROM {schema}.games g
JOIN {schema}.features_game fg
    ON fg.game_id = g.game_id
JOIN {schema}.teams th
    ON th.mlb_team_id = g.home_team_id
JOIN {schema}.teams ta
    ON ta.mlb_team_id = g.away_team_id
LEFT JOIN {schema}.game_starting_pitchers gsp
    ON gsp.game_id = g.game_id
WHERE g.game_date = :d
ORDER BY g.game_id;
"""

# Pull all finished games for the current season to compute season-to-date stats
SEASON_HISTORY_QUERY = """
SELECT
    g.game_id,
    g.game_date,
    g.home_team_id,
    g.away_team_id,
    g.home_runs,
    g.away_runs
FROM {schema}.games g
WHERE EXTRACT(YEAR FROM g.game_date) = :season
  AND g.game_date < :d
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


def load_today_games(engine, inference_date: str, schema: str) -> pd.DataFrame:
    with engine.connect() as conn:
        df = pd.read_sql(
            text(TODAY_GAMES_QUERY.format(schema=schema)),
            conn,
            params={"d": inference_date},
        )
    log.info(f"Today's games: {len(df)} scheduled for {inference_date}")
    return df


def load_season_history(engine, inference_date: str, schema: str) -> pd.DataFrame:
    season = int(inference_date[:4])
    with engine.connect() as conn:
        df = pd.read_sql(
            text(SEASON_HISTORY_QUERY.format(schema=schema)),
            conn,
            params={"season": season, "d": inference_date},
        )
    log.info(f"Season history: {len(df)} finished games before {inference_date}")
    return df


# ── Feature engineering ──────────────────────────────────────────────────────

def compute_season_to_date(history: pd.DataFrame) -> dict:
    """
    Compute season-to-date stats per team from finished games.
    Returns dict: team_id → {sd_games, sd_wins, sd_runs_for, sd_runs_against}
    Mirrors baseline.py compute_season_to_date exactly.
    """
    stats = {}

    for _, row in history.iterrows():
        h = int(row["home_team_id"])
        a = int(row["away_team_id"])
        hr = int(row["home_runs"])
        ar = int(row["away_runs"])

        for team_id, runs_for, runs_against, won in [
            (h, hr, ar, hr > ar),
            (a, ar, hr, ar > hr),
        ]:
            if team_id not in stats:
                stats[team_id] = {"sd_games": 0, "sd_wins": 0,
                                   "sd_runs_for": 0, "sd_runs_against": 0}
            s = stats[team_id]
            s["sd_games"] += 1
            s["sd_wins"] += int(won)
            s["sd_runs_for"] += runs_for
            s["sd_runs_against"] += runs_against

    return stats


def shrunk_win_pct(wins, games, prior_games=PRIOR_GAMES, prior_pct=PRIOR_WIN_PCT):
    return (wins + prior_games * prior_pct) / (games + prior_games)


def shrunk_run_diff_pg(runs_for, runs_against, games, prior_games=PRIOR_GAMES):
    return (runs_for - runs_against) / (games + prior_games)


def build_features(today: pd.DataFrame, sd_stats: dict) -> pd.DataFrame:
    """
    Compute the 6 V10_FEATURES for today's games.
    NULLs in Statcast columns → XWOBA_DEFAULT (matches training behavior).
    """
    rows = []
    for _, g in today.iterrows():
        h_id = int(g["home_team_id"])
        a_id = int(g["away_team_id"])

        h = sd_stats.get(h_id, {"sd_games": 0, "sd_wins": 0,
                                  "sd_runs_for": 0, "sd_runs_against": 0})
        a = sd_stats.get(a_id, {"sd_games": 0, "sd_wins": 0,
                                  "sd_runs_for": 0, "sd_runs_against": 0})

        # Feature 1: win pct diff (shrunk toward .500)
        win_pct_diff = (
            shrunk_win_pct(h["sd_wins"], h["sd_games"])
            - shrunk_win_pct(a["sd_wins"], a["sd_games"])
        )

        # Feature 2: run differential per game diff (shrunk toward 0)
        run_diff_pg_diff = (
            shrunk_run_diff_pg(h["sd_runs_for"], h["sd_runs_against"], h["sd_games"])
            - shrunk_run_diff_pg(a["sd_runs_for"], a["sd_runs_against"], a["sd_games"])
        )

        # Feature 3: SP xwOBA diff (away - home; positive = home SP better)
        home_sp_xwoba = g["home_sp_xwoba_against_90"]
        away_sp_xwoba = g["away_sp_xwoba_against_90"]
        if pd.isna(home_sp_xwoba):
            home_sp_xwoba = XWOBA_DEFAULT
            log.debug(f"  game {g['game_id']}: home SP xwOBA NULL → {XWOBA_DEFAULT}")
        if pd.isna(away_sp_xwoba):
            away_sp_xwoba = XWOBA_DEFAULT
            log.debug(f"  game {g['game_id']}: away SP xwOBA NULL → {XWOBA_DEFAULT}")
        sp_xwoba_diff = away_sp_xwoba - home_sp_xwoba

        # Feature 4: lineup xwOBA diff (home - away; positive = home lineup better)
        home_lu_xwoba = g["home_lineup_xwoba_90"]
        away_lu_xwoba = g["away_lineup_xwoba_90"]
        if pd.isna(home_lu_xwoba):
            home_lu_xwoba = XWOBA_DEFAULT
            log.debug(f"  game {g['game_id']}: home lineup xwOBA NULL → {XWOBA_DEFAULT}")
        if pd.isna(away_lu_xwoba):
            away_lu_xwoba = XWOBA_DEFAULT
            log.debug(f"  game {g['game_id']}: away lineup xwOBA NULL → {XWOBA_DEFAULT}")
        lineup_xwoba_diff = home_lu_xwoba - away_lu_xwoba

        # Feature 5: park factor (prefer blended)
        park_factor = g["park_runs_factor_blended"]
        if pd.isna(park_factor):
            park_factor = g["park_runs_factor"]
        if pd.isna(park_factor):
            park_factor = PARK_FACTOR_DEFAULT

        # Feature 6: constant (intercept proxy)
        is_home_const = 1.0

        rows.append({
            "game_id": g["game_id"],
            "game_date": g["game_date"],
            "home_team_id": h_id,
            "away_team_id": a_id,
            "home_team": g["home_team"],
            "away_team": g["away_team"],
            "home_sp_name": g.get("home_sp_name"),
            "away_sp_name": g.get("away_sp_name"),
            "morning_p_home": g.get("morning_p_home"),
            "closing_p_home": g.get("closing_p_home"),
            "morning_home_price": g.get("morning_home_price"),
            "morning_away_price": g.get("morning_away_price"),
            "closing_home_price": g.get("closing_home_price"),
            "closing_away_price": g.get("closing_away_price"),
            "sharp_action_home": g.get("sharp_action_home"),
            "line_move_magnitude": g.get("line_move_magnitude"),
            # Features
            "win_pct_diff": win_pct_diff,
            "run_diff_pg_diff": run_diff_pg_diff,
            "sp_xwoba_diff": sp_xwoba_diff,
            "lineup_xwoba_diff": lineup_xwoba_diff,
            "park_factor": park_factor,
            "is_home_const": is_home_const,
            # SD stats for audit
            "home_sd_games": h["sd_games"],
            "away_sd_games": a["sd_games"],
        })

    return pd.DataFrame(rows)


# ── Model inference ──────────────────────────────────────────────────────────

def run_inference(features_df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Apply v10 model. Returns df with p_home_win, p_away_win, confidence_tier."""
    scaler = bundle["scaler"]
    model = bundle["model"]

    X = scaler.transform(features_df[V10_FEATURES])
    p_home = model.predict_proba(X)[:, 1]
    p_away = 1.0 - p_home

    features_df = features_df.copy()
    features_df["p_home_win"] = p_home
    features_df["p_away_win"] = p_away

    # Confidence tier based on edge analysis findings:
    # [0.50-0.53) → skip, [0.53-0.56) → mild, [0.56-0.60) → moderate
    # [0.60-0.65) → high, [0.65+) → elite
    p_pick = np.maximum(p_home, p_away)
    tiers = np.where(p_pick < 0.53, "skip",
             np.where(p_pick < 0.56, "mild",
             np.where(p_pick < 0.60, "moderate",
             np.where(p_pick < 0.65, "high", "elite"))))
    features_df["confidence_tier"] = tiers

    return features_df


# ── Edge / value flags ───────────────────────────────────────────────────────

def compute_edge(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute edge_home, edge_away, ev_home, ev_away, is_value_ml_home/away.
    Uses morning odds (available at inference time).
    Mirrors v9 logic so dashboard value flags work unchanged.
    """
    df = df.copy()

    def implied_prob(price):
        if pd.isna(price):
            return np.nan
        p = float(price)
        return abs(p) / (abs(p) + 100) if p < 0 else 100 / (p + 100)

    # De-vig morning line
    df["morn_p_home_raw"] = df["morning_home_price"].apply(implied_prob)
    df["morn_p_away_raw"] = df["morning_away_price"].apply(implied_prob)
    total = df["morn_p_home_raw"] + df["morn_p_away_raw"]
    df["market_p_home"] = df["morn_p_home_raw"] / total
    df["market_p_away"] = df["morn_p_away_raw"] / total

    # Edge = model - market (de-vigged)
    df["edge_home"] = df["p_home_win"] - df["market_p_home"]
    df["edge_away"] = df["p_away_win"] - df["market_p_away"]

    # EV at morning price (per $100 bet)
    def ev(edge, price):
        if pd.isna(price) or pd.isna(edge):
            return np.nan
        p = float(price)
        win_amt = p if p > 0 else 100 * (100 / abs(p))
        return edge * win_amt - (1 - edge) * 100

    df["ev_home"] = df.apply(
        lambda r: ev(r["p_home_win"], r["morning_home_price"]), axis=1)
    df["ev_away"] = df.apply(
        lambda r: ev(r["p_away_win"], r["morning_away_price"]), axis=1)

    # Value flags: model sees edge > 3% (conservative threshold)
    df["is_value_ml_home"] = (df["edge_home"] > 0.03).fillna(False).astype(bool)
    df["is_value_ml_away"] = (df["edge_away"] > 0.03).fillna(False).astype(bool)

    return df


# ── Postgres write ───────────────────────────────────────────────────────────

def build_output(df: pd.DataFrame, inference_date: str, lineup_pending: bool = False) -> pd.DataFrame:
    """
    Build rows matching inference_game_predictions (v9 columns + v10 metadata).
    ML probs go to p_home_win_poisson / p_away_win_poisson for export compat.
    """
    now = datetime.utcnow()
    n = len(df)

    out = pd.DataFrame({
        "as_of_ts": [now] * n,
        "game_id": df["game_id"].values,
        "game_date": pd.to_datetime(
            df["game_date"] if "game_date" in df.columns else inference_date
        ),
        "home_team": df["home_team"].values,
        "away_team": df["away_team"].values,
        "home_runs_pred": np.nan,
        "away_runs_pred": np.nan,
        "total_runs_pred": np.nan,
        "run_diff_pred": np.nan,
        "p_home_win_raw": df["p_home_win"].values,
        "p_home_win_poisson": df["p_home_win"].values,
        "p_away_win_poisson": df["p_away_win"].values,
        "p_home_market_median": df.get("market_p_home", pd.Series([np.nan] * n)).values,
        "p_away_market_median": df.get("market_p_away", pd.Series([np.nan] * n)).values,
        "home_price_consensus": pd.array(
            df.get("morning_home_price", pd.Series([None] * n)),
            dtype=pd.Int64Dtype(),
        ),
        "away_price_consensus": pd.array(
            df.get("morning_away_price", pd.Series([None] * n)),
            dtype=pd.Int64Dtype(),
        ),
        "ou_recommendation": None,
        "edge_home": df.get("edge_home"),
        "edge_away": df.get("edge_away"),
        "ev_home": df.get("ev_home"),
        "ev_away": df.get("ev_away"),
        "is_value_ml_home": df.get("is_value_ml_home", False),
        "is_value_ml_away": df.get("is_value_ml_away", False),
        "model_version": "v10",
        "confidence_tier": df["confidence_tier"].values,
        "lineup_pending": lineup_pending,
    })
    return out


def ensure_v10_columns(engine, schema: str) -> None:
    with engine.begin() as conn:
        conn.execute(text(f"""
            ALTER TABLE {schema}.inference_game_predictions
                ADD COLUMN IF NOT EXISTS model_version TEXT,
                ADD COLUMN IF NOT EXISTS confidence_tier TEXT,
                ADD COLUMN IF NOT EXISTS lineup_pending BOOLEAN DEFAULT FALSE
        """))
        conn.execute(text(f"""
            UPDATE {schema}.inference_game_predictions
            SET model_version = 'v9'
            WHERE game_date >= DATE '2026-04-14'
              AND game_date < DATE '2026-05-28'
        """))
        conn.execute(text(f"""
            UPDATE {schema}.inference_game_predictions
            SET model_version = 'v10'
            WHERE game_date >= DATE '2026-05-28'
              AND (model_version IS NULL OR model_version = '')
        """))
        conn.execute(text(f"""
            UPDATE {schema}.inference_game_predictions
            SET model_version = 'v9'
            WHERE game_date < DATE '2026-04-14'
              AND (model_version IS NULL OR model_version = '')
        """))


def write_to_postgres(out: pd.DataFrame, engine, schema: str, inference_date: str):
    """Upsert predictions; skip games already marked final (same guard as v9)."""
    ensure_v10_columns(engine, schema)
    tmp = f"_v10_inf_tmp_{inference_date.replace('-', '')}"
    col_names = ", ".join(out.columns)
    set_clause = ", ".join(
        f"{c} = EXCLUDED.{c}"
        for c in out.columns
        if c not in ("as_of_ts", "game_id")
    )

    with engine.begin() as conn:
        out.to_sql(tmp, conn, schema=schema,
                   if_exists="replace", index=False, method="multi")
        conn.execute(text(f"""
            INSERT INTO {schema}.inference_game_predictions ({col_names})
            SELECT {col_names} FROM {schema}.{tmp}
            WHERE game_id NOT IN (
                SELECT game_id FROM {schema}.games
                WHERE LOWER(COALESCE(status, '')) IN ('final', 'game over', 'completed early')
            )
            ON CONFLICT (as_of_ts, game_id) DO UPDATE SET {set_clause}
        """))
        conn.execute(text(f"DROP TABLE IF EXISTS {schema}.{tmp}"))

    log.info(f"Wrote {len(out)} predictions to {schema}.inference_game_predictions")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="v10 moneyline inference")
    ap.add_argument("--date", default=str(date.today()),
                    help="Inference date YYYY-MM-DD (default: today)")
    ap.add_argument("--model", default="artifacts/baseline_v10_production.joblib",
                    help="Path to v10 model bundle")
    ap.add_argument("--schema", default="public")
    ap.add_argument("--fill_missing", action="store_true",
                    help="Fill NULL Statcast features with defaults instead of skipping")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print predictions without writing to Postgres")
    ap.add_argument("--no-lineup", action="store_true",
                    help="Mark predictions as lineup_pending (early run before confirmed lineups)")
    args = ap.parse_args()

    pg_dsn = os.environ.get("PG_DSN")
    if not pg_dsn:
        log.error("PG_DSN not set")
        sys.exit(1)

    log.info(f"v10 inference — date={args.date}  model={args.model}")

    # ── Load model bundle ──
    log.info(f"Loading model: {args.model}")
    bundle = joblib.load(args.model)
    assert bundle["features"] == V10_FEATURES, \
        f"Feature mismatch: expected {V10_FEATURES}, got {bundle['features']}"
    log.info(f"Model loaded. Features: {V10_FEATURES}")

    engine = create_engine(pg_dsn)

    # ── Load data ──
    log.info("Loading today's games...")
    today = load_today_games(engine, args.date, args.schema)
    if len(today) == 0:
        log.warning(f"No games found for {args.date} — exiting")
        sys.exit(0)

    log.info("Loading season history for season-to-date features...")
    history = load_season_history(engine, args.date, args.schema)

    # ── Compute season-to-date stats ──
    log.info("Computing season-to-date team stats...")
    sd_stats = compute_season_to_date(history)
    log.info(f"Teams with season history: {len(sd_stats)}")

    # ── Build features ──
    log.info("Building features...")
    features_df = build_features(today, sd_stats)

    # Warn if many Statcast features are NULL
    null_sp = features_df["sp_xwoba_diff"].isna().sum()
    null_lu = features_df["lineup_xwoba_diff"].isna().sum()
    if null_sp > 0 or null_lu > 0:
        log.warning(f"NULL Statcast features — SP xwOBA: {null_sp}, "
                    f"lineup xwOBA: {null_lu} games → using {XWOBA_DEFAULT} default")

    # ── Run model ──
    log.info("Running v10 model...")
    results = run_inference(features_df, bundle)

    # ── Compute edge / value flags ──
    log.info("Computing edge and value flags...")
    results = compute_edge(results)

    # ── Build output ──
    out = build_output(results, args.date, lineup_pending=args.no_lineup)

    if not args.no_lineup:
        confirmed = confirmed_lineup_game_ids(engine, args.schema, args.date)
        before = len(out)
        out = out[out["game_id"].isin(confirmed)].copy()
        skipped = before - len(out)
        if skipped:
            log.info(
                f"Confirmed-lineup upsert only: writing {len(out)} game(s), "
                f"preserving early predictions for {skipped} game(s) without full lineups"
            )
        if len(out) == 0:
            log.warning(
                "No confirmed lineups yet — skipping v10 write "
                "(early/pre-lineup predictions remain in Postgres)"
            )
            return

    # ── Preview ──
    log.info("\nPredictions:")
    for _, r in results.iterrows():
        pick = r["home_team"] if r["p_home_win"] >= 0.5 else r["away_team"]
        p_pick = max(r["p_home_win"], r["p_away_win"])
        tier = r["confidence_tier"]
        log.info(f"  {r['away_team']} @ {r['home_team']}  →  "
                 f"{pick} {p_pick*100:.1f}%  [{tier}]  "
                 f"edge_home={r.get('edge_home', float('nan'))*100:+.1f}%")

    if args.dry_run:
        log.info("DRY RUN — not writing to Postgres")
        print(out.to_string())
        return

    # ── Write to Postgres ──
    write_to_postgres(out, engine, args.schema, args.date)
    log.info("Done.")


if __name__ == "__main__":
    main()