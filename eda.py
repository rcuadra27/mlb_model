#!/usr/bin/env python3
"""
eda_runs_model.py

Pre-training EDA for the MLB runs model. Runs a suite of checks and
produces a self-contained HTML report: eda_report.html

Usage:
    PG_DSN=postgresql+psycopg2://... python eda_runs_model.py
    PG_DSN=postgresql+psycopg2://... python eda_runs_model.py --schema public --out eda_report.html
"""

import os
import io
import base64
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import scipy.stats as stats
from sqlalchemy import create_engine, text

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Must match training script exactly
# ---------------------------------------------------------------------------
LOCKED_FEATURES = [
    "win_pct_diff", "runs_for_diff", "runs_against_diff",
    "avg_runs_scored_60_diff", "avg_runs_allowed_60_diff",
    "sp_ra9_diff", "sp_ip_diff",
    "bp_outs_3d_diff", "bp_hlev_3d_diff", "bp_b2b_diff",
    "lineup_skill_diff", "matchup_diff",
    "is_home",
    "temp_f", "wind_mph", "wind_dir_sin", "wind_dir_cos",
    "humidity", "precip_in",
]

GAME_LEVEL_DIFF_COLS = [
    "sp_ra9_diff", "sp_ip_diff",
    "bp_outs_3d_diff", "bp_hlev_3d_diff", "bp_b2b_diff",
    "win_pct_diff", "runs_for_diff", "runs_against_diff",
    "avg_runs_scored_60_diff", "avg_runs_allowed_60_diff",
]

DIFF_FEATURES = [f for f in LOCKED_FEATURES if f.endswith("_diff")]
WEATHER_FEATURES = ["temp_f", "wind_mph", "wind_dir_sin", "wind_dir_cos", "humidity", "precip_in"]

STYLE = {
    "bg":        "#0f1117",
    "surface":   "#1a1d27",
    "border":    "#2a2d3a",
    "accent":    "#4f8ef7",
    "accent2":   "#f7914f",
    "green":     "#4fcf8e",
    "red":       "#f74f4f",
    "text":      "#e8eaf0",
    "muted":     "#7b7f94",
    "font":      "IBM Plex Mono",
}

plt.rcParams.update({
    "figure.facecolor":  STYLE["bg"],
    "axes.facecolor":    STYLE["surface"],
    "axes.edgecolor":    STYLE["border"],
    "axes.labelcolor":   STYLE["text"],
    "axes.titlecolor":   STYLE["text"],
    "xtick.color":       STYLE["muted"],
    "ytick.color":       STYLE["muted"],
    "text.color":        STYLE["text"],
    "grid.color":        STYLE["border"],
    "grid.linewidth":    0.5,
    "font.family":       "monospace",
    "figure.dpi":        110,
})


# ---------------------------------------------------------------------------
# DB loading
# ---------------------------------------------------------------------------

def load_data(engine, schema: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw game frame and build the two-row team dataset."""
    print("Loading base data from DB...")
    df_game = pd.read_sql(text(f"""
        SELECT
            f.*,
            g.home_runs,
            g.away_runs,
            g.game_date,
            g.home_team_id,
            g.away_team_id,
            gw.temp_f       AS gw_temp_f,
            gw.wind_mph     AS gw_wind_mph,
            gw.wind_dir_deg AS gw_wind_dir_deg,
            gw.humidity     AS gw_humidity,
            gw.precip_in    AS gw_precip_in
        FROM {schema}.features_game f
        JOIN {schema}.games g USING (game_id)
        LEFT JOIN {schema}.game_weather gw USING (game_id)
        WHERE g.home_runs IS NOT NULL
          AND g.away_runs IS NOT NULL
        ORDER BY g.game_date
    """), engine)

    df_game = df_game.loc[:, ~df_game.columns.duplicated()].copy()

    # Canonical weather columns
    for canon, backup in {
        "temp_f": "gw_temp_f", "wind_mph": "gw_wind_mph",
        "wind_dir_deg": "gw_wind_dir_deg", "humidity": "gw_humidity",
        "precip_in": "gw_precip_in",
    }.items():
        if canon not in df_game.columns:
            df_game[canon] = df_game.get(backup)
        else:
            df_game[canon] = df_game[canon].where(df_game[canon].notna(), df_game.get(backup))

    df_game["game_date"] = pd.to_datetime(df_game["game_date"])
    if "season" not in df_game.columns:
        df_game["season"] = df_game["game_date"].dt.year

    df_team = build_two_row_dataset(df_game)
    print(f"  Games: {len(df_game):,}   Team rows: {len(df_team):,}")
    return df_game, df_team


def add_game_level_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def add_diff(new_col, home_col, away_col):
        if home_col in out.columns and away_col in out.columns:
            out[new_col] = out[home_col].astype(float) - out[away_col].astype(float)

    add_diff("sp_ra9_diff",     "home_sp_ra9_last5",          "away_sp_ra9_last5")
    add_diff("sp_ip_diff",      "home_sp_ip_per_start_last5", "away_sp_ip_per_start_last5")
    add_diff("bp_outs_3d_diff", "home_bp_outs_3d",            "away_bp_outs_3d")
    add_diff("bp_hlev_3d_diff", "home_bp_hlev_outs_3d",       "away_bp_hlev_outs_3d")

    for h, a in [
        ("home_bp_b2b_pitchers_3d", "away_bp_b2b_pitchers_3d"),
        ("home_bp_b2b",             "away_bp_b2b"),
        ("home_bp_b2b_3d",          "away_bp_b2b_3d"),
    ]:
        if h in out.columns and a in out.columns:
            out["bp_b2b_diff"] = out[h].astype(float) - out[a].astype(float)
            break

    if "bp_b2b_diff" in out.columns:
        out["bp_b2b_diff"] = out["bp_b2b_diff"].fillna(0.0)

    add_diff("win_pct_diff",      "home_win_pct_30",     "away_win_pct_30")
    add_diff("runs_for_diff",     "home_runs_for_30",    "away_runs_for_30")
    add_diff("runs_against_diff", "home_runs_against_30","away_runs_against_30")
    add_diff("avg_runs_scored_60_diff",  "home_avg_runs_scored_60",  "away_avg_runs_scored_60")
    add_diff("avg_runs_allowed_60_diff", "home_avg_runs_allowed_60", "away_avg_runs_allowed_60")

    if "wind_dir_deg" in out.columns:
        wd = out["wind_dir_deg"].astype(float)
        out["wind_dir_sin"] = np.sin(np.deg2rad(wd))
        out["wind_dir_cos"] = np.cos(np.deg2rad(wd))

    return out


def add_team_level_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def first_existing(candidates):
        for c in candidates:
            if c in out.columns:
                return c
        return None

    def add_diff(new_col, team_cands, opp_cands):
        tc = first_existing(team_cands)
        oc = first_existing(opp_cands)
        if tc and oc:
            out[new_col] = out[tc].astype(float) - out[oc].astype(float)

    add_diff("lineup_skill_diff", ["team_lineup_skill"], ["opp_lineup_skill"])
    if "matchup_diff" not in out.columns:
        add_diff("matchup_diff", ["team_matchup"], ["opp_matchup"])
    return out


def build_two_row_dataset(df_game: pd.DataFrame) -> pd.DataFrame:
    df_game = add_game_level_diff_features(df_game)

    cols      = df_game.columns.tolist()
    home_cols = [c for c in cols if c.startswith("home_")]
    away_cols = [c for c in cols if c.startswith("away_")]
    skip      = {"game_id","game_date","season","home_team_id","away_team_id","home_runs","away_runs"}
    global_cols = [c for c in cols if not c.startswith("home_") and not c.startswith("away_") and c not in skip]

    id_cols = ["game_id", "game_date", "season"]

    home = df_game[id_cols + ["home_team_id","away_team_id","home_runs"] + global_cols + home_cols + away_cols].copy()
    home = home.rename(columns={"home_team_id":"team_id","away_team_id":"opp_id","home_runs":"target_runs"})
    home["is_home"] = 1
    home = home.rename(columns={c: "team_" + c[5:] for c in home_cols})
    home = home.rename(columns={c: "opp_"  + c[5:] for c in away_cols})

    away = df_game[id_cols + ["home_team_id","away_team_id","away_runs"] + global_cols + home_cols + away_cols].copy()
    away = away.rename(columns={"away_team_id":"team_id","home_team_id":"opp_id","away_runs":"target_runs"})
    away["is_home"] = 0
    away = away.rename(columns={c: "team_" + c[5:] for c in away_cols})
    away = away.rename(columns={c: "opp_"  + c[5:] for c in home_cols})

    # KEY: flip game-level diffs for away rows
    for col in GAME_LEVEL_DIFF_COLS:
        if col in away.columns:
            away[col] = -away[col]

    common = [c for c in home.columns if c in away.columns]
    seen, deduped = set(), []
    for c in common:
        if c not in seen:
            deduped.append(c)
            seen.add(c)

    out = pd.concat([home[deduped], away[deduped]], ignore_index=True)
    out = out.loc[:, ~out.columns.duplicated()].copy()
    out = add_team_level_diff_features(out)
    out = out.dropna(subset=["target_runs"]).copy()
    return out


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def section_img(b64: str, caption: str = "") -> str:
    cap = f'<p class="caption">{caption}</p>' if caption else ""
    return f'<div class="chart-wrap"><img src="data:image/png;base64,{b64}" />{cap}</div>'


# ---------------------------------------------------------------------------
# EDA sections
# ---------------------------------------------------------------------------

def eda_target_distribution(df_game: pd.DataFrame, df_team: pd.DataFrame) -> str:
    print("  [1/7] Target distribution...")
    fig = plt.figure(figsize=(14, 9))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # --- run score histogram ---
    ax1 = fig.add_subplot(gs[0, :2])
    bins = np.arange(-0.5, 25.5, 1)
    h_runs = df_game["home_runs"].dropna().astype(int)
    a_runs = df_game["away_runs"].dropna().astype(int)
    ax1.hist(h_runs, bins=bins, alpha=0.75, color=STYLE["accent"],  label=f"Home  μ={h_runs.mean():.2f}")
    ax1.hist(a_runs, bins=bins, alpha=0.75, color=STYLE["accent2"], label=f"Away  μ={a_runs.mean():.2f}")
    ax1.set_title("Run score distribution")
    ax1.set_xlabel("Runs scored")
    ax1.set_ylabel("Games")
    ax1.legend(fontsize=9)
    ax1.grid(True, axis="y")

    # --- total runs histogram ---
    ax2 = fig.add_subplot(gs[0, 2])
    totals = (df_game["home_runs"] + df_game["away_runs"]).dropna()
    ax2.hist(totals, bins=30, color=STYLE["green"], alpha=0.85)
    ax2.set_title(f"Total runs  μ={totals.mean():.2f}")
    ax2.set_xlabel("Total runs")
    ax2.grid(True, axis="y")

    # --- Poisson fit overlay ---
    ax3 = fig.add_subplot(gs[1, :2])
    all_runs = df_team["target_runs"].dropna().astype(float)
    ax3.hist(all_runs, bins=bins, density=True, alpha=0.65, color=STYLE["accent"], label="Observed")
    lam = all_runs.mean()
    xs  = np.arange(0, 22)
    ax3.plot(xs, stats.poisson.pmf(xs, lam), "o-", color=STYLE["accent2"],
             lw=2, ms=5, label=f"Poisson(λ={lam:.2f})")
    # Tweedie (approx negative binomial for overdispersion check)
    var  = all_runs.var()
    if var > lam:
        r = lam**2 / (var - lam)
        p = lam / var
        ax3.plot(xs, stats.nbinom.pmf(xs, r, p), "s--", color=STYLE["green"],
                 lw=1.5, ms=4, label=f"NegBinom (overdispersed)")
    ax3.set_title("Observed vs Poisson fit")
    ax3.set_xlabel("Runs")
    ax3.set_ylabel("Density")
    ax3.legend(fontsize=9)
    ax3.grid(True, axis="y")

    # --- by season ---
    ax4 = fig.add_subplot(gs[1, 2])
    if "season" in df_game.columns:
        seas = df_game.groupby("season")[["home_runs","away_runs"]].mean()
        x = np.arange(len(seas))
        ax4.bar(x - 0.2, seas["home_runs"], 0.4, color=STYLE["accent"],  label="Home")
        ax4.bar(x + 0.2, seas["away_runs"],  0.4, color=STYLE["accent2"], label="Away")
        ax4.set_xticks(x)
        ax4.set_xticklabels(seas.index.astype(str), rotation=45, fontsize=8)
        ax4.set_title("Avg runs by season")
        ax4.legend(fontsize=8)
        ax4.grid(True, axis="y")

    fig.suptitle("1 — Target distribution", fontsize=13, color=STYLE["text"], y=1.01)
    b64 = fig_to_b64(fig)

    # Stats table
    skew_h = float(stats.skew(h_runs))
    kurt_h = float(stats.kurtosis(h_runs))
    _, pval = stats.kstest(h_runs, "poisson", args=(h_runs.mean(),))
    rows = [
        ("Home runs mean",  f"{h_runs.mean():.3f}"),
        ("Away runs mean",  f"{a_runs.mean():.3f}"),
        ("Home advantage",  f"{h_runs.mean() - a_runs.mean():.3f} runs/game"),
        ("Total runs mean", f"{totals.mean():.3f}"),
        ("Skewness (home)", f"{skew_h:.3f}"),
        ("Kurtosis (home)", f"{kurt_h:.3f}"),
        ("Poisson KS p-val",f"{pval:.4f}  {'⚠ overdispersed' if pval < 0.05 else '✓ ok'}"),
        ("Tweedie power",   "1.1 (set)  — increase toward 1.5 if overdispersion is severe"),
    ]
    return section_img(b64) + stat_table(rows)


def eda_diff_sign_sanity(df_team: pd.DataFrame) -> str:
    print("  [2/7] Diff feature sign sanity...")
    available = [c for c in GAME_LEVEL_DIFF_COLS if c in df_team.columns]
    if not available:
        return "<p>No game-level diff columns found in team dataset.</p>"

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.flatten()
    rows = []

    for i, col in enumerate(available):
        ax = axes[i]
        home_vals = df_team.loc[df_team["is_home"] == 1, col].dropna()
        away_vals = df_team.loc[df_team["is_home"] == 0, col].dropna()

        ax.hist(home_vals, bins=40, alpha=0.7, color=STYLE["accent"],  density=True, label="Home row")
        ax.hist(away_vals, bins=40, alpha=0.7, color=STYLE["accent2"], density=True, label="Away row")
        ax.axvline(0, color=STYLE["muted"], lw=1, ls="--")
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("value", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, axis="y")

        h_mean = home_vals.mean()
        a_mean = away_vals.mean()
        mirror = abs(h_mean + a_mean) < 0.05 * (abs(h_mean) + abs(a_mean) + 1e-9)
        rows.append((col,
                     f"{h_mean:.4f}",
                     f"{a_mean:.4f}",
                     "✓ mirrored" if mirror else "⚠ NOT mirrored — check diff flip"))

    for j in range(len(available), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("2 — Diff feature sign sanity (home rows vs away rows)", fontsize=13,
                 color=STYLE["text"], y=1.01)
    fig.tight_layout()
    b64 = fig_to_b64(fig)

    header = ["Feature", "Home row mean", "Away row mean", "Status"]
    return section_img(b64) + html_table(header, rows)


def eda_weather_symmetry(df_team: pd.DataFrame) -> str:
    print("  [3/7] Weather feature symmetry...")
    available = [c for c in WEATHER_FEATURES if c in df_team.columns]
    if not available:
        return "<p>No weather features found in team dataset.</p>"

    home = df_team[df_team["is_home"] == 1][available]
    away = df_team[df_team["is_home"] == 0][available]

    rows = []
    for col in available:
        hm = home[col].dropna().mean()
        am = away[col].dropna().mean()
        diff_pct = abs(hm - am) / (abs(hm) + 1e-9) * 100
        status = "✓ identical" if diff_pct < 0.1 else f"⚠ differ by {diff_pct:.2f}%"
        rows.append((col, f"{hm:.4f}", f"{am:.4f}", status))

    fig, axes = plt.subplots(1, min(len(available), 3), figsize=(14, 4))
    if len(available) == 1:
        axes = [axes]
    for i, col in enumerate(available[:3]):
        ax = axes[i]
        ax.hist(home[col].dropna(), bins=40, alpha=0.7, color=STYLE["accent"],  density=True, label="Home row")
        ax.hist(away[col].dropna(), bins=40, alpha=0.7, color=STYLE["accent2"], density=True, label="Away row")
        ax.set_title(col, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y")

    fig.suptitle("3 — Weather features: home vs away rows (should be identical)",
                 fontsize=13, color=STYLE["text"], y=1.02)
    fig.tight_layout()
    b64 = fig_to_b64(fig)

    header = ["Feature", "Home row mean", "Away row mean", "Status"]
    return section_img(b64) + html_table(header, rows)


def eda_missing_data(df_team: pd.DataFrame) -> str:
    print("  [4/7] Missing data map...")
    avail = [c for c in LOCKED_FEATURES if c in df_team.columns]
    miss  = df_team[avail].isna().mean().sort_values(ascending=False) * 100

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = [STYLE["red"] if v > 10 else STYLE["accent2"] if v > 2 else STYLE["green"]
              for v in miss.values]
    bars = ax.barh(miss.index, miss.values, color=colors)
    ax.set_xlabel("Missing %")
    ax.set_title("Missing data by locked feature")
    ax.axvline(2,  color=STYLE["accent2"], lw=1, ls="--", label="2% threshold")
    ax.axvline(10, color=STYLE["red"],     lw=1, ls="--", label="10% threshold")
    ax.legend(fontsize=8)
    ax.grid(True, axis="x")
    ax.invert_yaxis()
    for bar, val in zip(bars, miss.values):
        if val > 0.1:
            ax.text(val + 0.2, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}%", va="center", fontsize=8, color=STYLE["text"])
    fig.tight_layout()
    b64 = fig_to_b64(fig)

    # By season breakdown
    rows = []
    if "season" in df_team.columns:
        for feat in avail:
            by_season = df_team.groupby("season")[feat].apply(lambda x: x.isna().mean() * 100)
            worst_season = by_season.idxmax()
            rows.append((feat,
                         f"{miss[feat]:.1f}%",
                         f"{worst_season} ({by_season[worst_season]:.1f}%)" if not pd.isna(by_season[worst_season]) else "—"))

    header = ["Feature", "Overall missing", "Worst season"]
    return section_img(b64) + (html_table(header, rows) if rows else "")


def eda_feature_correlations(df_team: pd.DataFrame) -> str:
    print("  [5/7] Feature correlations with target...")
    avail = [c for c in LOCKED_FEATURES if c in df_team.columns and c != "is_home"]
    target = df_team["target_runs"].astype(float)

    pearson_rows = []
    for col in avail:
        s = df_team[col].astype(float)
        valid = s.notna() & target.notna()
        if valid.sum() < 50:
            continue
        r, p = stats.pearsonr(s[valid], target[valid])
        pearson_rows.append((col, r, p))

    pearson_rows.sort(key=lambda x: abs(x[1]), reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart of correlations
    ax = axes[0]
    feat_names = [r[0] for r in pearson_rows]
    corrs      = [r[1] for r in pearson_rows]
    bar_colors = [STYLE["green"] if c > 0 else STYLE["red"] for c in corrs]
    y_pos = np.arange(len(feat_names))
    ax.barh(y_pos, corrs, color=bar_colors, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_names, fontsize=8)
    ax.axvline(0, color=STYLE["muted"], lw=1)
    ax.set_xlabel("Pearson r with target_runs")
    ax.set_title("Feature-target correlation")
    ax.grid(True, axis="x")
    ax.invert_yaxis()

    # Scatter for top feature
    ax2 = axes[1]
    if pearson_rows:
        top_feat = pearson_rows[0][0]
        x = df_team[top_feat].astype(float)
        y = target
        valid = x.notna() & y.notna()
        ax2.scatter(x[valid], y[valid], alpha=0.15, s=8, color=STYLE["accent"])
        m, b = np.polyfit(x[valid], y[valid], 1)
        xs = np.linspace(x[valid].min(), x[valid].max(), 100)
        ax2.plot(xs, m * xs + b, color=STYLE["accent2"], lw=2,
                 label=f"slope={m:.3f}")
        ax2.set_xlabel(top_feat)
        ax2.set_ylabel("target_runs")
        ax2.set_title(f"Top feature: {top_feat}")
        ax2.legend(fontsize=9)
        ax2.grid(True)

    fig.suptitle("5 — Feature correlations with target runs", fontsize=13,
                 color=STYLE["text"], y=1.01)
    fig.tight_layout()
    b64 = fig_to_b64(fig)

    header = ["Feature", "Pearson r", "p-value", "Signal"]
    table_rows = [
        (r[0], f"{r[1]:.4f}", f"{r[2]:.2e}",
         "✓ significant" if r[2] < 0.05 else "— not significant")
        for r in pearson_rows
    ]
    return section_img(b64) + html_table(header, table_rows)


def eda_home_field_advantage(df_game: pd.DataFrame) -> str:
    print("  [6/7] Home field advantage...")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Win rate by season
    ax = axes[0]
    if "season" in df_game.columns:
        df_game["home_win"] = (df_game["home_runs"] > df_game["away_runs"]).astype(int)
        hfa = df_game.groupby("season")["home_win"].agg(["mean","count"])
        # Wilson confidence intervals
        n, p = hfa["count"].values, hfa["mean"].values
        z = 1.96
        ci = z * np.sqrt(p * (1 - p) / n)
        x = np.arange(len(hfa))
        ax.bar(x, hfa["mean"] * 100, color=STYLE["accent"], alpha=0.85)
        ax.errorbar(x, hfa["mean"] * 100, yerr=ci * 100, fmt="none",
                    color=STYLE["text"], capsize=3, lw=1.5)
        ax.axhline(50, color=STYLE["muted"], lw=1, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(hfa.index.astype(str), rotation=45, fontsize=8)
        ax.set_ylabel("Home win rate (%)")
        ax.set_title("Home win rate by season")
        ax.set_ylim(40, 65)
        ax.grid(True, axis="y")

    # Run differential distribution
    ax2 = axes[1]
    diff = (df_game["home_runs"] - df_game["away_runs"]).dropna()
    ax2.hist(diff, bins=np.arange(-15.5, 16.5, 1), color=STYLE["accent"], alpha=0.85)
    ax2.axvline(0,           color=STYLE["muted"],  lw=1, ls="--")
    ax2.axvline(diff.mean(), color=STYLE["accent2"], lw=2,
                label=f"mean={diff.mean():.3f}")
    ax2.set_xlabel("Home runs − Away runs")
    ax2.set_title("Score differential distribution")
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="y")

    # Monthly HFA
    ax3 = axes[2]
    df_copy = df_game.copy()
    df_copy["month"] = df_copy["game_date"].dt.month
    monthly = df_copy.groupby("month")["home_win"].mean() * 100
    months  = ["Apr","May","Jun","Jul","Aug","Sep","Oct"]
    m_vals  = [monthly.get(m, np.nan) for m in range(4, 11)]
    ax3.plot(months, m_vals, "o-", color=STYLE["green"], lw=2, ms=7)
    ax3.axhline(50, color=STYLE["muted"], lw=1, ls="--")
    ax3.set_ylabel("Home win rate (%)")
    ax3.set_title("Home win rate by month")
    ax3.set_ylim(40, 65)
    ax3.grid(True)

    fig.suptitle("6 — Home field advantage", fontsize=13, color=STYLE["text"], y=1.01)
    fig.tight_layout()
    b64 = fig_to_b64(fig)

    overall_hfa   = df_game["home_win"].mean() * 100
    mean_diff     = diff.mean()
    _, ttest_pval = stats.ttest_1samp(diff.dropna(), 0)
    rows = [
        ("Overall home win rate", f"{overall_hfa:.2f}%"),
        ("Mean score differential (home − away)", f"{mean_diff:.3f}"),
        ("t-test p-value (diff ≠ 0)", f"{ttest_pval:.4e}"),
        ("is_home feature guidance",
         "Keep as binary 0/1. Do NOT dampen with * 0.3 — HFA is real and meaningful."),
    ]
    return section_img(b64) + stat_table(rows)


def eda_leakage_check(df_team: pd.DataFrame) -> str:
    print("  [7/7] Leakage check...")
    avail = [c for c in LOCKED_FEATURES if c in df_team.columns]

    if "target_runs" not in df_team.columns:
        return "<p>target_runs not in dataset.</p>"

    # Compute correlation with runs AND with win outcome
    df_team = df_team.copy()
    # We can reconstruct win from paired rows (home target > away target for same game)
    # Simpler: just flag if corr(feature, target_runs) is suspiciously high
    target = df_team["target_runs"].astype(float)

    rows = []
    high_corr_features = []
    for col in avail:
        s = df_team[col].astype(float)
        valid = s.notna() & target.notna()
        if valid.sum() < 50:
            rows.append((col, "—", "—", "insufficient data"))
            continue
        r, _ = stats.pearsonr(s[valid], target[valid])
        flag = abs(r) > 0.5
        if flag:
            high_corr_features.append(col)
        rows.append((col,
                     f"{r:.4f}",
                     f"{abs(r):.4f}",
                     f"⚠ HIGH — check for leakage ({abs(r):.2f} > 0.5)" if flag else "✓ ok"))

    # Visualise
    feat_names = [r[0] for r in rows if r[1] != "—"]
    abs_corrs  = [float(r[2]) for r in rows if r[1] != "—"]
    colors     = [STYLE["red"] if v > 0.5 else STYLE["accent2"] if v > 0.3 else STYLE["green"]
                  for v in abs_corrs]

    fig, ax = plt.subplots(figsize=(12, 5))
    y = np.arange(len(feat_names))
    ax.barh(y, abs_corrs, color=colors, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(feat_names, fontsize=8)
    ax.axvline(0.5, color=STYLE["red"],     lw=1.5, ls="--", label="|r| > 0.5  suspect")
    ax.axvline(0.3, color=STYLE["accent2"], lw=1,   ls=":",  label="|r| > 0.3  monitor")
    ax.set_xlabel("|Pearson r| with target_runs")
    ax.set_title("Leakage check — absolute correlation with target")
    ax.legend(fontsize=8)
    ax.grid(True, axis="x")
    ax.invert_yaxis()
    fig.tight_layout()
    b64 = fig_to_b64(fig)

    note = ""
    if high_corr_features:
        note = f'<p class="warn">⚠ High-correlation features: {", ".join(high_corr_features)}. Investigate before training.</p>'

    header = ["Feature", "Pearson r", "|r|", "Status"]
    return section_img(b64) + note + html_table(header, rows)


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def stat_table(rows: list[tuple]) -> str:
    inner = "".join(
        f"<tr><td>{r[0]}</td><td>{r[1]}</td></tr>" for r in rows
    )
    return f'<table class="stat">{inner}</table>'


def html_table(header: list[str], rows: list[tuple]) -> str:
    th = "".join(f"<th>{h}</th>" for h in header)
    trs = ""
    for r in rows:
        warn = "warn-row" if any("⚠" in str(c) for c in r) else ""
        trs += f'<tr class="{warn}">' + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
    return f'<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>'


def build_report(sections: list[tuple[str, str, str]]) -> str:
    """sections: list of (anchor, title, html_content)"""
    nav_items = "".join(
        f'<a href="#{a}">{t}</a>' for a, t, _ in sections
    )
    body = ""
    for anchor, title, content in sections:
        body += f"""
        <section id="{anchor}">
            <h2>{title}</h2>
            {content}
        </section>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>MLB Runs Model — EDA Report</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

  :root {{
    --bg:      #0f1117;
    --surface: #1a1d27;
    --border:  #2a2d3a;
    --accent:  #4f8ef7;
    --accent2: #f7914f;
    --green:   #4fcf8e;
    --red:     #f74f4f;
    --text:    #e8eaf0;
    --muted:   #7b7f94;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 14px;
    line-height: 1.7;
  }}

  header {{
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 2rem 2.5rem 1.5rem;
    position: sticky;
    top: 0;
    z-index: 100;
  }}

  header h1 {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--accent);
    letter-spacing: 0.05em;
    margin-bottom: 0.75rem;
  }}

  header .subtitle {{
    font-size: 0.8rem;
    color: var(--muted);
    margin-bottom: 1rem;
  }}

  nav {{
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
  }}

  nav a {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    text-decoration: none;
    padding: 0.25rem 0.6rem;
    border: 1px solid var(--border);
    border-radius: 3px;
    transition: all 0.15s;
  }}

  nav a:hover {{
    color: var(--accent);
    border-color: var(--accent);
  }}

  main {{
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem 2.5rem 4rem;
  }}

  section {{
    margin-bottom: 4rem;
    padding-top: 1rem;
  }}

  h2 {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 1.5rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
  }}

  .chart-wrap {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1rem;
    margin-bottom: 1.25rem;
    overflow: hidden;
  }}

  .chart-wrap img {{
    width: 100%;
    height: auto;
    display: block;
  }}

  .caption {{
    font-size: 0.75rem;
    color: var(--muted);
    margin-top: 0.5rem;
    font-style: italic;
  }}

  table {{
    width: 100%;
    border-collapse: collapse;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    margin-bottom: 1rem;
  }}

  table.stat td:first-child {{
    color: var(--muted);
    width: 55%;
    padding: 0.35rem 0.75rem 0.35rem 0;
  }}

  table.stat td:last-child {{
    color: var(--text);
    font-weight: 500;
  }}

  table.stat tr {{
    border-bottom: 1px solid var(--border);
  }}

  thead th {{
    text-align: left;
    color: var(--muted);
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
  }}

  tbody td {{
    padding: 0.4rem 0.75rem;
    border-bottom: 1px solid var(--border);
    color: var(--text);
  }}

  tbody tr:hover {{ background: rgba(79,142,247,0.04); }}

  .warn-row td {{ color: var(--accent2); }}

  p.warn {{
    background: rgba(247,145,79,0.1);
    border-left: 3px solid var(--accent2);
    padding: 0.6rem 1rem;
    margin-bottom: 1rem;
    border-radius: 0 4px 4px 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: var(--accent2);
  }}
</style>
</head>
<body>
<header>
  <h1>⚾ MLB RUNS MODEL — EDA REPORT</h1>
  <p class="subtitle">Pre-training data validation &amp; feature audit</p>
  <nav>{nav_items}</nav>
</header>
<main>
{body}
</main>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", default="public")
    ap.add_argument("--out",    default="eda_report.html")
    args = ap.parse_args()

    pg_dsn = os.getenv("PG_DSN")
    if not pg_dsn:
        raise RuntimeError("PG_DSN env var is required (postgresql+psycopg2://...).")

    engine = create_engine(pg_dsn, pool_pre_ping=True)
    df_game, df_team = load_data(engine, args.schema)

    print("Running EDA checks...")
    sections = [
        ("target",   "1 — Target distribution",       eda_target_distribution(df_game, df_team)),
        ("diffsign", "2 — Diff feature sign sanity",  eda_diff_sign_sanity(df_team)),
        ("weather",  "3 — Weather feature symmetry",  eda_weather_symmetry(df_team)),
        ("missing",  "4 — Missing data map",           eda_missing_data(df_team)),
        ("corr",     "5 — Feature correlations",       eda_feature_correlations(df_team)),
        ("hfa",      "6 — Home field advantage",       eda_home_field_advantage(df_game)),
        ("leakage",  "7 — Leakage check",              eda_leakage_check(df_team)),
    ]

    print(f"Writing report to {args.out} ...")
    Path(args.out).write_text(build_report(sections), encoding="utf-8")
    print(f"Done. Open {args.out} in a browser.")


if __name__ == "__main__":
    main()