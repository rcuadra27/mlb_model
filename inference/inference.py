#!/usr/bin/env python3
"""
Run inference for a given MLB date and save results to Postgres.

Outputs:
- Predicted runs: home_runs_pred, away_runs_pred
- Poisson win prob: p_home_win_poisson
- Market (median across books, vig-free): p_home_market_median, n_books
- Edge + EV (computed using consensus fair odds derived from p_home_market_median)

Saves to:
  public.inference_game_predictions

Usage:
  export PG_DSN="postgresql+psycopg2://USER:PASS@HOST:5432/mlb_model"

  python inference/run_inference_for_date.py \
    --date 2024-07-01 \
    --min_books 6 \
    --top_k 10 \
    --home_model artifacts/runs_model_home_lgbm_optionA.joblib \
    --away_model artifacts/runs_model_away_lgbm_optionA.joblib
"""

import os
import math
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import joblib
from sqlalchemy import create_engine, text


# -----------------------------
# Odds helpers
# -----------------------------
def american_to_implied(odds: np.ndarray) -> np.ndarray:
    """American odds -> implied probability (includes vig)."""
    o = odds.astype(float)
    p = np.full_like(o, np.nan, dtype=float)
    neg = o < 0
    pos = o > 0
    p[neg] = (-o[neg]) / ((-o[neg]) + 100.0)
    p[pos] = 100.0 / (o[pos] + 100.0)
    return p


def prob_to_american(p: np.ndarray) -> np.ndarray:
    """Probability -> American odds (vig-free synthetic)."""
    p = np.clip(p.astype(float), 1e-6, 1 - 1e-6)
    odds = np.empty_like(p)
    fav = p >= 0.5
    odds[fav] = -100.0 * (p[fav] / (1.0 - p[fav]))
    odds[~fav] = 100.0 * ((1.0 - p[~fav]) / p[~fav])
    return np.rint(odds).astype(int)


def profit_if_win_1u(odds: float) -> float:
    """Return profit on a 1-unit stake if bet wins, given American odds."""
    if odds is None or (isinstance(odds, float) and np.isnan(odds)) or odds == 0:
        return np.nan
    if odds > 0:
        return float(odds) / 100.0
    return 100.0 / abs(float(odds))


# -----------------------------
# Poisson win probability
# -----------------------------
def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def p_home_win_from_lambdas(lambda_home: float, lambda_away: float) -> float:
    """
    P(Home wins) when runs are independent Poisson(lambda_home), Poisson(lambda_away).
    Prefer Skellam (SciPy) if available; fallback to Normal approx on difference.

    Normal approx:
      D = H - A ~ Normal(mu=lambda_h-lambda_a, var=lambda_h+lambda_a)
      P(H>A) = P(D >= 1) approx P(D > 0.5) with continuity correction
    """
    lh = max(1e-9, float(lambda_home))
    la = max(1e-9, float(lambda_away))

    # Try SciPy Skellam if installed
    try:
        from scipy.stats import skellam  # type: ignore
        # P(H > A) = 1 - P(H-A <= 0) = 1 - CDF(0)
        return float(1.0 - skellam.cdf(0, lh, la))
    except Exception:
        mu = lh - la
        var = lh + la
        sd = math.sqrt(max(1e-12, var))
        z = (0.5 - mu) / sd
        return float(1.0 - normal_cdf(z))


# -----------------------------
# Team-model inference (two rows per game)
# -----------------------------
def drop_leaky_feature_cols(cols):
    bad_substrings = ["price", "_pred", "p_home", "p_away", "p_tie", "p_"]
    out = []
    for c in cols:
        lc = c.lower()
        if any(s in lc for s in bad_substrings):
            continue
        out.append(c)
    return out


def _uniq(seq):
    """Return list of unique elements preserving order."""
    out = []
    seen = set()
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def build_inference_team_rows(df_game: pd.DataFrame, feature_cols: list,
                              home_prefix: str = "home_", away_prefix: str = "away_") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      keys_df: columns [game_id, is_home, team_id, opp_id] aligned with X rows
      X: model feature matrix with columns == feature_cols
    """
    cols = df_game.columns.tolist()

    home_cols = [c for c in cols if c.startswith(home_prefix)]
    away_cols = [c for c in cols if c.startswith(away_prefix)]
    global_cols = [c for c in cols if (not c.startswith(home_prefix)) and (not c.startswith(away_prefix))]

    # Keep only useful globals (don't duplicate base keys; game_date not used in model)
    base_keys = ["game_id", "season", "home_team_id", "away_team_id"]
    global_keep = []
    for c in global_cols:
        if c in ["game_id", "season", "game_date"]:
            continue
        if c in ["home_team", "away_team"]:
            continue
        global_keep.append(c)

    # Safety: remove leaky cols
    home_cols = drop_leaky_feature_cols(home_cols)
    away_cols = drop_leaky_feature_cols(away_cols)
    global_keep = drop_leaky_feature_cols(global_keep)

    home_select = _uniq(base_keys + global_keep + home_cols + away_cols)
    away_select = _uniq(base_keys + global_keep + home_cols + away_cols)

    # HOME ROWS
    home = df_game[home_select].copy()
    home = home.rename(columns={"home_team_id": "team_id", "away_team_id": "opp_id"})
    home["is_home"] = 1
    home = home.rename(columns={c: "team_" + c[len(home_prefix):] for c in home_cols})
    home = home.rename(columns={c: "opp_" + c[len(away_prefix):] for c in away_cols})

    # AWAY ROWS
    away = df_game[away_select].copy()
    away = away.rename(columns={"away_team_id": "team_id", "home_team_id": "opp_id"})
    away["is_home"] = 0
    away = away.rename(columns={c: "team_" + c[len(away_prefix):] for c in away_cols})
    away = away.rename(columns={c: "opp_" + c[len(home_prefix):] for c in home_cols})

    # Defensive dedupe before concat
    home = home.loc[:, ~home.columns.duplicated()].copy()
    away = away.loc[:, ~away.columns.duplicated()].copy()

    team_rows = pd.concat([home, away], ignore_index=True)
    team_rows = team_rows.loc[:, ~team_rows.columns.duplicated()].copy()

    # Keys aligned with row order
    keys = team_rows[["game_id", "is_home", "team_id", "opp_id"]].copy()

    # Ensure all required feature columns exist
    for c in feature_cols:
        if c not in team_rows.columns:
            team_rows[c] = np.nan

    X = team_rows[feature_cols].copy()

    # Categoricals
    for c in ["team_id", "opp_id", "season", "is_home"]:
        if c in X.columns:
            X[c] = X[c].astype("category")

    return keys, X


def predict_game_runs_from_team_model(df_game: pd.DataFrame, team_model, feature_cols: list) -> pd.DataFrame:
    """
    Bulletproof: build HOME rows and AWAY rows separately and predict each.
    This removes any possibility of ordering / concat / merge bugs.
    """
    cols = df_game.columns.tolist()
    home_cols = [c for c in cols if c.startswith("home_")]
    away_cols = [c for c in cols if c.startswith("away_")]
    global_cols = [c for c in cols if (not c.startswith("home_")) and (not c.startswith("away_"))]

    # Keep only useful globals (exclude names/date; keep season/game_id)
    global_keep = []
    for c in global_cols:
        if c in ["game_id", "season"]:
            global_keep.append(c)
        elif c in ["game_date", "home_team", "away_team"]:
            continue
        else:
            global_keep.append(c)

    # Safety: remove leakage
    home_cols = drop_leaky_feature_cols(home_cols)
    away_cols = drop_leaky_feature_cols(away_cols)
    global_keep = drop_leaky_feature_cols(global_keep)

    def uniq(seq):
        out = []
        seen = set()
        for x in seq:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    base_keys = ["game_id", "season", "home_team_id", "away_team_id"]

    # -------------------
    # HOME feature rows (team = home, opp = away)
    # -------------------
    home_select = uniq(base_keys + global_keep + home_cols + away_cols)
    H = df_game[home_select].copy()
    H = H.rename(columns={"home_team_id": "team_id", "away_team_id": "opp_id"})
    H["is_home"] = 1
    H = H.rename(columns={c: "team_" + c[len("home_"):] for c in home_cols})
    H = H.rename(columns={c: "opp_" + c[len("away_"):] for c in away_cols})
    H = H.loc[:, ~H.columns.duplicated()].copy()

    # Ensure all required feature columns exist
    for c in feature_cols:
        if c not in H.columns:
            H[c] = np.nan
    XH = H[feature_cols].copy()
    for c in ["team_id", "opp_id", "season", "is_home"]:
        if c in XH.columns:
            XH[c] = XH[c].astype("category")

    home_pred = team_model.predict(XH).astype(float)

    # -------------------
    # AWAY feature rows (team = away, opp = home)
    # -------------------
    away_select = uniq(base_keys + global_keep + home_cols + away_cols)
    A = df_game[away_select].copy()
    A = A.rename(columns={"away_team_id": "team_id", "home_team_id": "opp_id"})
    A["is_home"] = 0
    A = A.rename(columns={c: "team_" + c[len("away_"):] for c in away_cols})
    A = A.rename(columns={c: "opp_" + c[len("home_"):] for c in home_cols})
    A = A.loc[:, ~A.columns.duplicated()].copy()

    for c in feature_cols:
        if c not in A.columns:
            A[c] = np.nan
    XA = A[feature_cols].copy()
    for c in ["team_id", "opp_id", "season", "is_home"]:
        if c in XA.columns:
            XA[c] = XA[c].astype("category")

    away_pred = team_model.predict(XA).astype(float)

    # -------------------
    out = df_game.copy()
    out["home_runs_pred"] = home_pred
    out["away_runs_pred"] = away_pred
    out["total_runs_pred"] = out["home_runs_pred"] + out["away_runs_pred"]
    out["run_diff_pred"] = out["home_runs_pred"] - out["away_runs_pred"]

    # Extra sanity (should be 0)
    bad_home = (H["team_id"].astype(int).values != df_game["home_team_id"].astype(int).values).sum()
    bad_away = (A["team_id"].astype(int).values != df_game["away_team_id"].astype(int).values).sum()
    print("SANITY mapping bad_home:", bad_home, "bad_away:", bad_away)

    return out


# -----------------------------
# DB / model utilities
# -----------------------------
@dataclass
class MarketAgg:
    game_id: int
    p_home_market_median: float
    n_books: int
    home_price_consensus: int
    away_price_consensus: int


def ensure_output_table(engine, schema: str) -> None:
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {schema}.inference_game_predictions (
      as_of_ts TIMESTAMPTZ NOT NULL,
      game_id BIGINT NOT NULL,
      game_date DATE NOT NULL,

      home_team TEXT,
      away_team TEXT,

      home_runs_pred DOUBLE PRECISION,
      away_runs_pred DOUBLE PRECISION,
      total_runs_pred DOUBLE PRECISION,
      run_diff_pred DOUBLE PRECISION,

      p_home_win_poisson DOUBLE PRECISION,
      p_away_win_poisson DOUBLE PRECISION,

      p_home_market_median DOUBLE PRECISION,
      p_away_market_median DOUBLE PRECISION,
      n_books INTEGER,

      home_price_consensus INTEGER,
      away_price_consensus INTEGER,

      edge_home DOUBLE PRECISION,
      edge_away DOUBLE PRECISION,

      ev_home DOUBLE PRECISION,
      ev_away DOUBLE PRECISION,

      PRIMARY KEY (as_of_ts, game_id)
    );

    CREATE INDEX IF NOT EXISTS idx_infer_game_date ON {schema}.inference_game_predictions (game_date);
    CREATE INDEX IF NOT EXISTS idx_infer_game_id ON {schema}.inference_game_predictions (game_id);
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))

def fetch_games_and_features(engine, schema: str, date_str: str) -> pd.DataFrame:
    col_q = text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = :schema
          AND table_name = :table
        ORDER BY ordinal_position
    """)
    feat_cols_all = pd.read_sql(col_q, engine, params={"schema": schema, "table": "features_game"})["column_name"].tolist()

    # avoid duplicates with games columns
    drop = {"game_id", "game_date", "home_team_id", "away_team_id"}
    feat_cols = [c for c in feat_cols_all if c not in drop]
    feat_select = ",\n        ".join([f"f.{c}" for c in feat_cols])

    q = text(f"""
      WITH tnames AS (
        SELECT mlb_team_id, team_name
        FROM {schema}.teams
      )
      SELECT
        g.game_id,
        g.game_date,
        g.home_team_id,
        g.away_team_id,
        COALESCE(th.team_name, g.home_team_id::text) AS home_team,
        COALESCE(ta.team_name, g.away_team_id::text) AS away_team,
        {feat_select}
      FROM {schema}.games g
      JOIN {schema}.features_game f ON f.game_id = g.game_id
      LEFT JOIN tnames th ON th.mlb_team_id = g.home_team_id
      LEFT JOIN tnames ta ON ta.mlb_team_id = g.away_team_id
      WHERE g.game_date = :d
      ORDER BY g.game_id
    """)

    return pd.read_sql(q, engine, params={"d": date_str})
def load_market_median(engine, schema: str, date_str: str, market: str = "h2h", max_abs_odds: int = 2000) -> pd.DataFrame:
    """
    Compute median across books using latest pulled_at per (game_id, sportsbook) for the date.
    Steps:
      1) select latest row per (game_id, sportsbook)
      2) filter invalid odds
      3) convert to implied probs, remove vig per book -> p_home_fair
      4) median across books -> p_home_market_median, n_books
      5) derive consensus odds from p_home_market_median (vig-free synthetic, for display/EV)
    """
    q = text(f"""
      WITH latest AS (
        SELECT DISTINCT ON (game_id, sportsbook)
          game_id,
          sportsbook,
          home_price,
          away_price,
          pulled_at
        FROM {schema}.odds_ml
        WHERE market = :market
          AND game_date = :d
          AND game_id IS NOT NULL
          AND home_price IS NOT NULL
          AND away_price IS NOT NULL
        ORDER BY game_id, sportsbook, pulled_at DESC
      )
      SELECT game_id, sportsbook, home_price, away_price
      FROM latest
    """)
    df = pd.read_sql(q, engine, params={"market": market, "d": date_str})
    df = df.loc[:, ~df.columns.duplicated()].copy()
    if df.empty:
        return df

    hp = df["home_price"].astype(int).to_numpy()
    ap = df["away_price"].astype(int).to_numpy()

    def valid_odds(x: np.ndarray) -> np.ndarray:
        return ((x <= -100) | (x >= 100)) & (np.abs(x) <= max_abs_odds)

    ok = valid_odds(hp) & valid_odds(ap)
    df = df.loc[ok].copy()
    if df.empty:
        return df

    hp = df["home_price"].astype(int).to_numpy()
    ap = df["away_price"].astype(int).to_numpy()

    p_home_imp = american_to_implied(hp)
    p_away_imp = american_to_implied(ap)
    denom = p_home_imp + p_away_imp
    df["p_home_fair"] = p_home_imp / denom

    agg = df.groupby("game_id").agg(
        p_home_market_median=("p_home_fair", "median"),
        n_books=("p_home_fair", "count"),
    ).reset_index()

    home_cons = prob_to_american(agg["p_home_market_median"].to_numpy())
    away_cons = prob_to_american((1.0 - agg["p_home_market_median"]).to_numpy())
    agg["home_price_consensus"] = home_cons
    agg["away_price_consensus"] = away_cons
    agg["p_away_market_median"] = 1.0 - agg["p_home_market_median"]

    return agg


def get_model_feature_names(model) -> List[str]:
    """
    Works for LightGBM sklearn wrapper saved via joblib.
    """
    # LGBMRegressor has .booster_ after fitting, and booster has feature_name()
    if hasattr(model, "booster_") and model.booster_ is not None:
        return list(model.booster_.feature_name())
    if hasattr(model, "feature_name_"):
        return list(model.feature_name_)
    # Fallback: can't infer reliably
    raise RuntimeError("Could not infer feature names from model. Ensure it's a fitted LightGBM model.")


def prepare_X(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required feature columns for model: {missing[:25]} (and {max(0, len(missing)-25)} more)")

    X = df[feature_names].copy()

    # LightGBM can handle categorical if dtype is category
    for cat_col in ["venue_id", "park_id", "home_team_id", "away_team_id"]:
        if cat_col in X.columns:
            X[cat_col] = X[cat_col].astype("category")

    # Ensure numeric for everything else
    for c in X.columns:
        if X[c].dtype == "object":
            # Most of your feature cols should be numeric; objects are usually IDs/strings that should not be here.
            raise RuntimeError(f"Non-numeric feature column detected in model inputs: {c} (dtype=object).")

    return X


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", default="public")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--market", default="h2h")
    ap.add_argument("--min_books", type=int, default=5)
    ap.add_argument("--top_k", type=int, default=10)

    ap.add_argument("--home_model", default="artifacts/runs_model_home_lgbm_optionA.joblib")
    ap.add_argument("--away_model", default="artifacts/runs_model_away_lgbm_optionA.joblib")
    ap.add_argument("--team_model", default="artifacts/runs_model_team_lgbm_optionA.joblib")
    ap.add_argument("--team_features", default="artifacts/runs_model_team_features_optionA.txt")

    args = ap.parse_args()

    pg_dsn = os.getenv("PG_DSN")
    if not pg_dsn:
        raise RuntimeError("PG_DSN env var is required (postgresql+psycopg2://...).")

    engine = create_engine(pg_dsn, pool_pre_ping=True)

    ensure_output_table(engine, args.schema)

    # Load game/features rows
    df = fetch_games_and_features(engine, args.schema, args.date)
    if df.empty:
        print(f"No games found for {args.date} in {args.schema}.games (or missing join in features_game).")
        return

    # Market median (vig-free) from odds snapshots
    mkt = load_market_median(engine, args.schema, args.date, market=args.market)
    if mkt.empty:
        print(f"Warning: No odds found for {args.date}. Market columns will be NULL; value bets won't be ranked.")
        df = df.copy()
        df["p_home_market_median"] = np.nan
        df["p_away_market_median"] = np.nan
        df["n_books"] = np.nan
        df["home_price_consensus"] = np.nan
        df["away_price_consensus"] = np.nan
    else:
        df = df.merge(mkt, on="game_id", how="left")

    # Load single team runs model + feature list
    team_model = joblib.load(args.team_model)
    with open(args.team_features, "r") as f:
        team_feature_cols = [line.strip() for line in f if line.strip()]

    # Add season to df (if not already there)
    if "season" not in df.columns:
        df["season"] = pd.to_datetime(df["game_date"]).dt.year

    # Predict runs using the single team model
    df = predict_game_runs_from_team_model(df, team_model, team_feature_cols)

    # Clip negative run preds (Tweedie can output small negatives)
    df["home_runs_pred"] = df["home_runs_pred"].clip(lower=0.1)
    df["away_runs_pred"] = df["away_runs_pred"].clip(lower=0.1)
    df["total_runs_pred"] = df["home_runs_pred"] + df["away_runs_pred"]
    df["run_diff_pred"] = df["home_runs_pred"] - df["away_runs_pred"]

    # Temporary home-advantage correction for bias testing (do not overwrite home_runs_pred) for bias testing (do not overwrite home_runs_pred)
    HOME_ADV_RUNS = 0.0  # temporary correction test
    df["home_runs_pred_adj"] = df["home_runs_pred"] + HOME_ADV_RUNS
    df["away_runs_pred_adj"] = df["away_runs_pred"]

    # Poisson home win probability (using adjusted runs for correction test).
    # Team model outputs home_runs_pred for away row and away_runs_pred for home row, so swap lambdas.
    df["p_home_win_poisson"] = [
        p_home_win_from_lambdas(a, h)
        for h, a in zip(df["home_runs_pred_adj"].values, df["away_runs_pred_adj"].values)
    ]
    df["p_away_win_poisson"] = 1.0 - df["p_home_win_poisson"]

    # Edge vs market
    df["edge_home"] = df["p_home_win_poisson"] - df["p_home_market_median"]
    df["edge_away"] = df["p_away_win_poisson"] - df["p_away_market_median"]

    # EV vs consensus (vig-free) odds derived from p_market_median
    # NOTE: Since consensus odds are vig-free, this EV is an "edge signal", not a true book-execution EV.
    df["ev_home"] = np.nan
    df["ev_away"] = np.nan
    for i in range(len(df)):
        hp = df.loc[i, "home_price_consensus"]
        ap = df.loc[i, "away_price_consensus"]
        if pd.isna(hp) or pd.isna(ap):
            continue
        b_home = profit_if_win_1u(float(hp))
        b_away = profit_if_win_1u(float(ap))
        p_home = float(df.loc[i, "p_home_win_poisson"])
        p_away = 1.0 - p_home
        df.loc[i, "ev_home"] = p_home * b_home - (1.0 - p_home)
        df.loc[i, "ev_away"] = p_away * b_away - (1.0 - p_away)

    # Aliases and best-side columns for insert (same logic as ranked display below)
    df["p_home_model"] = df["p_home_win_poisson"]
    df["p_away_model"] = df["p_away_win_poisson"]
    EDGE_MIN = 0.01

    df["best_side"] = np.where(df["edge_home"] >= df["edge_away"], "HOME", "AWAY")
    df["best_edge"] = np.maximum(df["edge_home"], df["edge_away"])

    # If not positive enough, mark as no bet
    df.loc[df["best_edge"] < EDGE_MIN, "best_side"] = "NO_BET"
    df.loc[df["best_edge"] < EDGE_MIN, ["best_edge", "best_p_model", "best_p_market", "best_ev"]] = np.nan

    df["best_p_model"] = np.where(
        df["best_side"] == "HOME", df["p_home_win_poisson"],
        np.where(df["best_side"] == "AWAY", df["p_away_win_poisson"], np.nan)
    )
    df["best_p_market"] = np.where(
        df["best_side"] == "HOME", df["p_home_market_median"],
        np.where(df["best_side"] == "AWAY", df["p_away_market_median"], np.nan)
    )
    df["best_ev"] = np.where(
        df["best_side"] == "HOME", df["ev_home"],
        np.where(df["best_side"] == "AWAY", df["ev_away"], np.nan)
    )

    # Save to Postgres (append with as_of_ts)
    as_of = pd.Timestamp.now("UTC")

    out_cols = [
        "game_id", "game_date", "home_team", "away_team",
        "home_runs_pred", "away_runs_pred", "total_runs_pred", "run_diff_pred",
        "p_home_win_poisson", "p_away_win_poisson",

        # model final
        "p_home_model", "p_away_model",

        # market
        "p_home_market_median", "p_away_market_median", "n_books",
        "home_price_consensus", "away_price_consensus",

        # per-side diagnostics
        "edge_home", "edge_away", "ev_home", "ev_away",

        # frontend recommendation
        "best_side", "best_edge", "best_ev", "best_p_model", "best_p_market",
    ]
    out = df[out_cols].copy()
    out.insert(0, "as_of_ts", as_of)

    # Replace NaN with None so Postgres gets NULL; cast int-like cols to int (avoids out-of-range from nan)
    records = out.to_dict(orient="records")
    int_cols = {"game_id", "n_books", "home_price_consensus", "away_price_consensus"}
    for r in records:
        for k, v in r.items():
            if pd.isna(v):
                r[k] = None
            elif k in int_cols and v is not None:
                try:
                    r[k] = int(v)
                except (ValueError, TypeError):
                    r[k] = None

    insert_sql = text(f"""
        INSERT INTO {args.schema}.inference_game_predictions (
            as_of_ts, game_id, game_date,
            home_team, away_team,
            home_runs_pred, away_runs_pred, total_runs_pred, run_diff_pred,
            p_home_win_poisson, p_away_win_poisson,

            p_home_model, p_away_model,

            p_home_market_median, p_away_market_median, n_books,
            home_price_consensus, away_price_consensus,
            edge_home, edge_away,
            ev_home, ev_away,

            best_side, best_edge, best_ev, best_p_model, best_p_market
        ) VALUES (
            :as_of_ts, :game_id, :game_date,
            :home_team, :away_team,
            :home_runs_pred, :away_runs_pred, :total_runs_pred, :run_diff_pred,
            :p_home_win_poisson, :p_away_win_poisson,

            :p_home_model, :p_away_model,

            :p_home_market_median, :p_away_market_median, :n_books,
            :home_price_consensus, :away_price_consensus,
            :edge_home, :edge_away,
            :ev_home, :ev_away,

            :best_side, :best_edge, :best_ev, :best_p_model, :best_p_market
        )
    """)

    with engine.begin() as conn:
        conn.execute(text("SET LOCAL statement_timeout = '0'"))
        conn.execute(insert_sql, records)

    print(f"Saved {len(out):,} rows to {args.schema}.inference_game_predictions for date={args.date} as_of_ts={as_of}.")

    # Print Top K value bets using the SAME best_* already computed on df
    ranked = df.copy()

    ranked = ranked[ranked["n_books"].fillna(0).astype(int) >= args.min_books].copy()
    if ranked.empty:
        print(f"No games met min_books={args.min_books}; falling back to min_books=1 for display.")
        ranked = df[df["n_books"].fillna(0).astype(int) >= 1].copy()

    ranked = ranked.dropna(subset=["p_home_market_median", "p_home_win_poisson"])

    # only bets (not NO_BET) and already thresholded by EDGE_MIN earlier
    ranked = ranked[ranked["best_side"] != "NO_BET"].copy()

    if ranked.empty:
        print(f"No games met EDGE_MIN={EDGE_MIN} and min_books={args.min_books} for {args.date}.")
        return

    ranked = ranked.sort_values("best_edge", ascending=False).head(args.top_k)

    top = ranked[[
        "game_id", "game_date", "away_team", "home_team",
        "best_side", "best_edge", "best_p_model", "best_p_market", "best_ev", "n_books"
    ]].copy()

    print("\nTop value bets (ranked by model-market edge; median across books):")
    print(top.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Sanity checks
    df["sum_market"] = df["p_home_market_median"] + df["p_away_market_median"]
    df["sum_model"]  = df["p_home_win_poisson"] + df["p_away_win_poisson"]

    print("\nSANITY CHECKS:")
    print("market sum min/mean/max:",
        df["sum_market"].min(), df["sum_market"].mean(), df["sum_market"].max())
    print("model sum min/mean/max:",
        df["sum_model"].min(), df["sum_model"].mean(), df["sum_model"].max())

    print("\nEDGE MEANS:")
    print("edge_home mean:", df["edge_home"].mean(), "edge_away mean:", df["edge_away"].mean())

    print("\nTOP 5 biggest edges (abs) with both sides shown:")
    tmp = df[[
        "away_team","home_team",
        "p_home_win_poisson","p_home_market_median","edge_home",
        "p_away_win_poisson","p_away_market_median","edge_away",
        "n_books"
    ]].copy()
    tmp["abs_max_edge"] = np.maximum(tmp["edge_home"].abs(), tmp["edge_away"].abs())
    print(tmp.sort_values("abs_max_edge", ascending=False).head(5).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    g = df[df["game_id"] == 745852].iloc[0]
    print("\nGAME 745852 CHECK")
    print(g["away_team"], "@", g["home_team"])
    print("home_runs_pred:", g["home_runs_pred"], "away_runs_pred:", g["away_runs_pred"])
    print("p_home_win (as-is):", g["p_home_win_poisson"])
    print("p_home_win (if swapped lambdas):", p_home_win_from_lambdas(g["away_runs_pred"], g["home_runs_pred"]))
    print("p_home_market:", g["p_home_market_median"])

    print("run_diff_pred mean:", df["run_diff_pred"].mean())
    print("run_diff_pred std :", df["run_diff_pred"].std())
    print("run_diff_pred min/max:", df["run_diff_pred"].min(), df["run_diff_pred"].max())
    print("Mean market home prob:", df["p_home_market_median"].mean())
    print("Mean model home prob :", df["p_home_win_poisson"].mean())

    print("Mean home_runs_pred:", df["home_runs_pred"].mean())
    print("Mean away_runs_pred:", df["away_runs_pred"].mean())
    print("Mean (home-away) predicted runs:", (df["home_runs_pred"] - df["away_runs_pred"]).mean())

if __name__ == "__main__":
    main()