"""
Market edge analysis — v10 vs closing line (2025 holdout).

Answers the key question: are v10's disagreements with the market profitable?
Overall accuracy (55.14%) is not what determines profitability — edge on the
specific games where the model disagrees with the market is what matters.

Usage:
    PG_DSN=postgresql+psycopg2://... python models/edge_analysis.py \
        --eval_csv artifacts/baseline_v10_production_2025_eval.csv

Requires closing_p_home from features_game (already in your schema).
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text


CLOSING_QUERY = """
SELECT
    oc.game_id,
    -- De-vig via normalization (p_home_median + p_away_median ~ 1.044, not 1.0)
    oc.p_home_median / (oc.p_home_median + oc.p_away_median) AS p_market_home,
    oc.p_away_median / (oc.p_home_median + oc.p_away_median) AS p_market_away,
    -- Keep raw for reference
    oc.p_home_median AS p_home_raw,
    oc.p_away_median AS p_away_raw
FROM public.odds_closing_consensus oc
JOIN public.games g ON g.game_id = oc.game_id
WHERE g.season = :season;
"""


def american_to_prob(odds):
    """Raw implied probability from American odds (includes vig)."""
    if pd.isna(odds):
        return np.nan
    o = float(odds)
    return abs(o) / (abs(o) + 100) if o < 0 else 100 / (o + 100)


def devig(p_home_raw, p_away_raw):
    """Remove vig via normalization. Returns (p_home_devig, p_away_devig)."""
    total = p_home_raw + p_away_raw
    if pd.isna(total) or total <= 0:
        return np.nan, np.nan
    return p_home_raw / total, p_away_raw / total


def pnl_flat(correct, odds, stake=10.0):
    """P&L for a flat $10 bet given American odds."""
    if pd.isna(odds):
        return np.nan
    o = float(odds)
    win_amt = stake * (o / 100.0) if o > 0 else stake * (100.0 / abs(o))
    return win_amt if correct else -stake


def wilson_ci(p, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - half, c + half


def evaluate_bucket(sub, label, stake=10.0):
    """Print stats for a subset of games."""
    n = len(sub)
    if n == 0:
        print(f"  {label}: no games")
        return None

    acc = sub['pick_correct'].mean()
    lo, hi = wilson_ci(acc, n)
    mean_edge = sub['edge'].mean()
    mean_model = sub['p_model_pick'].mean()
    mean_market = sub['p_market_pick'].mean()

    # ROI using closing odds on the picked side
    # ROI approximation: assume -110 juice (52.4% break-even)
    wins = sub['pick_correct'].sum()
    losses = n - wins
    approx_pnl = wins * 9.09 - losses * 10.0  # -110 juice: win .09 per 0
    approx_roi = approx_pnl / (n * 10.0) * 100
    roi_str = f"~${approx_pnl:+.0f} ROI ~{approx_roi:+.1f}% (at -110)"

    print(f"  {label:<30} N={n:>4}  acc={acc*100:.1f}% [{lo*100:.1f},{hi*100:.1f}]  "
          f"model={mean_model*100:.1f}%  market={mean_market*100:.1f}%  "
          f"edge={mean_edge*100:+.1f}pts  {roi_str}")

    return {'label': label, 'n': n, 'acc': acc, 'ci_lo': lo, 'ci_hi': hi,
            'mean_edge': mean_edge, 'roi_str': roi_str}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval_csv', required=True,
                    help='Path to baseline_v10_production_2025_eval.csv')
    ap.add_argument('--season', type=int, default=2025)
    ap.add_argument('--stake', type=float, default=10.0)
    args = ap.parse_args()

    pg_dsn = os.environ.get('PG_DSN')
    if not pg_dsn:
        print("ERROR: PG_DSN not set", file=sys.stderr)
        sys.exit(1)

    # --- Load v10 predictions ---
    print(f"\nLoading v10 eval: {args.eval_csv}")
    df = pd.read_csv(args.eval_csv)
    print(f"  {len(df):,} games")

    # p_home_win_raw is what the LGBM produces; use that
    if 'p_home_win_raw' not in df.columns:
        print("ERROR: p_home_win_raw not in eval CSV — re-run baseline.py to regenerate",
              file=sys.stderr)
        sys.exit(1)

    df['p_model_home'] = df['p_home_win_raw']
    df['p_model_away'] = 1.0 - df['p_model_home']
    df['pick_home'] = (df['p_model_home'] >= 0.5).astype(int)
    df['pick_correct'] = (df['pick_home'] == df['home_win']).astype(int)

    # --- Load closing odds ---
    print(f"Loading closing odds from features_game (season {args.season})...")
    engine = create_engine(pg_dsn)
    with engine.connect() as conn:
        odds = pd.read_sql(text(CLOSING_QUERY), conn, params={'season': args.season})
    print(f"  {len(odds):,} games with closing odds")

    df = df.merge(odds, on='game_id', how='left')
    has_odds = df['p_market_home'].notna()
    print(f"  Matched: {has_odds.sum():,} / {len(df):,} games have closing odds\n")

    # p_market_home / p_market_away already de-vigged in SQL query
    df['p_market_away'] = df['p_market_away'].fillna(1.0 - df['p_market_home'])

    # --- Edge: model vs market, from the picked side's perspective ---
    df['p_model_pick'] = np.where(df['pick_home'] == 1,
                                   df['p_model_home'], df['p_model_away'])
    df['p_market_pick'] = np.where(df['pick_home'] == 1,
                                    df['p_market_home'], df['p_market_away'])
    df['edge'] = df['p_model_pick'] - df['p_market_pick']

    # No American odds available — ROI will show N/A
    df['picked_odds'] = np.nan

    # Whether model agrees with market favorite
    df['market_favors_home'] = (df['p_market_home'] >= 0.5).astype(int)
    df['model_agrees_market'] = (df['pick_home'] == df['market_favors_home']).astype(int)

    # =========================================================================
    print("=" * 90)
    print("OVERALL — v10 vs market (2025 OOS)")
    print("=" * 90)
    mdf = df[has_odds].copy()
    evaluate_bucket(mdf, "All games with odds")

    print()
    print("=" * 90)
    print("AGREES vs DISAGREES WITH MARKET FAVORITE")
    print("=" * 90)
    evaluate_bucket(mdf[mdf['model_agrees_market'] == 1], "Agrees with market")
    evaluate_bucket(mdf[mdf['model_agrees_market'] == 0], "Disagrees with market")

    print()
    print("=" * 90)
    print("EDGE BUCKETS — picks where model diverges from market")
    print("  Positive edge = model more confident than market on picked side")
    print("=" * 90)

    edge_bins = [
        ("All picks (any edge)",        mdf['edge'] >= -99),
        ("Model edge < 2% (skip zone)", mdf['edge'].abs() < 0.02),
        ("Edge 2–4%",                   (mdf['edge'] >= 0.02) & (mdf['edge'] < 0.04)),
        ("Edge 4–6%",                   (mdf['edge'] >= 0.04) & (mdf['edge'] < 0.06)),
        ("Edge 6–8%",                   (mdf['edge'] >= 0.06) & (mdf['edge'] < 0.08)),
        ("Edge 8%+",                    mdf['edge'] >= 0.08),
        ("Edge > 4% (bet zone)",        mdf['edge'] >= 0.04),
        ("Edge > 6% (high conviction)", mdf['edge'] >= 0.06),
    ]
    rows = []
    for label, mask in edge_bins:
        r = evaluate_bucket(mdf[mask], label)
        if r:
            rows.append(r)

    print()
    print("=" * 90)
    print("MODEL CONFIDENCE BUCKETS (regardless of market)")
    print("=" * 90)
    conf_bins = pd.cut(mdf['p_model_pick'],
                       bins=[0.5, 0.53, 0.56, 0.60, 0.65, 0.70, 1.01],
                       right=False, include_lowest=True)
    for bucket, sub in mdf.groupby(conf_bins, observed=True):
        if len(sub) == 0:
            continue
        evaluate_bucket(sub, str(bucket))

    print()
    print("=" * 90)
    print("EDGE DISTRIBUTION SUMMARY")
    print("=" * 90)
    print(f"  Mean edge (all picks):    {mdf['edge'].mean()*100:+.2f}%")
    print(f"  Median edge:              {mdf['edge'].median()*100:+.2f}%")
    print(f"  Std edge:                 {mdf['edge'].std()*100:.2f}%")
    print(f"  Edge > 4% (bet zone):     {(mdf['edge'] >= 0.04).sum():>4} games "
          f"({(mdf['edge'] >= 0.04).mean()*100:.1f}% of all picks)")
    print(f"  Edge > 6%:                {(mdf['edge'] >= 0.06).sum():>4} games "
          f"({(mdf['edge'] >= 0.06).mean()*100:.1f}% of all picks)")
    print(f"  Edge < -4% (fade zone):   {(mdf['edge'] < -0.04).sum():>4} games "
          f"({(mdf['edge'] < -0.04).mean()*100:.1f}% of all picks)")
    print()

    print("=" * 90)
    print("INTERPRETATION GUIDE")
    print("=" * 90)
    print("  Edge > 0: model more confident than market on picked side")
    print("  Edge < 0: model less confident than market — market sees something we don't")
    print("  Break-even at -110 juice: 52.4% accuracy needed")
    print("  A profitable strategy needs: acc > 52.4% AND volume > ~100 bets/season")
    print("  Look for: edge > 4% bucket with acc > 54% and n > 50 for statistical validity")


if __name__ == '__main__':
    main()