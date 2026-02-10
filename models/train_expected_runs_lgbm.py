import os
import joblib
import numpy as np
import pandas as pd
import psycopg2

from sklearn.metrics import mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor
from lightgbm import early_stopping, log_evaluation


# Optional: Skellam win prob if scipy exists; else simulation
try:
    from scipy.stats import skellam
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


FEATURE_COLS = [
    # Team form
    "home_win_pct_30", "away_win_pct_30",
    "home_runs_for_30", "away_runs_for_30",
    "home_runs_against_30", "away_runs_against_30",
    # Starters
    "home_sp_ra9_last5", "away_sp_ra9_last5",
    "home_sp_ip_per_start_last5", "away_sp_ip_per_start_last5",
    # Bullpen (3d was most meaningful)
    "home_bp_outs_3d", "away_bp_outs_3d",

    # Context (helps a lot)
    "season",
    "venue_id",
    # Team baselines (season-only last 60 games)
    "home_avg_runs_scored_60",
    "home_avg_runs_allowed_60",
    "away_avg_runs_scored_60",
    "away_avg_runs_allowed_60",
    "home_games_in_window_60",
    "away_games_in_window_60",
    "home_bp_hlev_outs_1d", "away_bp_hlev_outs_1d",
    "home_bp_hlev_outs_3d", "away_bp_hlev_outs_3d",
    "home_bp_hlev_outs_5d", "away_bp_hlev_outs_5d"

]



def load_data():
    conn = psycopg2.connect(os.environ["PG_DSN"])
    q = f"""
    SELECT
        fg.game_id,
        fg.season,
        g.venue_id,

        fg.home_win_pct_30,
        fg.away_win_pct_30,
        fg.home_runs_for_30,
        fg.away_runs_for_30,
        fg.home_runs_against_30,
        fg.away_runs_against_30,
        fg.home_sp_ra9_last5,
        fg.away_sp_ra9_last5,
        fg.home_sp_ip_per_start_last5,
        fg.away_sp_ip_per_start_last5,
        fg.home_avg_runs_scored_60,
        fg.home_avg_runs_allowed_60,
        fg.away_avg_runs_scored_60,
        fg.away_avg_runs_allowed_60,
        fg.home_games_in_window_60,
        fg.away_games_in_window_60,
        COALESCE(fg.home_bp_outs_3d,0) AS home_bp_outs_3d,
        COALESCE(fg.away_bp_outs_3d,0) AS away_bp_outs_3d,

        -- ✅ add these:
        COALESCE(fg.home_bp_hlev_outs_1d,0) AS home_bp_hlev_outs_1d,
        COALESCE(fg.away_bp_hlev_outs_1d,0) AS away_bp_hlev_outs_1d,
        COALESCE(fg.home_bp_hlev_outs_3d,0) AS home_bp_hlev_outs_3d,
        COALESCE(fg.away_bp_hlev_outs_3d,0) AS away_bp_hlev_outs_3d,
        COALESCE(fg.home_bp_hlev_outs_5d,0) AS home_bp_hlev_outs_5d,
        COALESCE(fg.away_bp_hlev_outs_5d,0) AS away_bp_hlev_outs_5d,


        g.home_runs,
        g.away_runs
    FROM features_game fg
    JOIN games g ON g.game_id = fg.game_id
    WHERE
        g.game_type = 'R'
        AND g.venue_id IS NOT NULL
        AND g.home_runs IS NOT NULL
        AND g.away_runs IS NOT NULL

    -- base features not null
        AND fg.home_win_pct_30 IS NOT NULL
        AND fg.away_win_pct_30 IS NOT NULL
        AND fg.home_runs_for_30 IS NOT NULL
        AND fg.away_runs_for_30 IS NOT NULL
        AND fg.home_runs_against_30 IS NOT NULL
        AND fg.away_runs_against_30 IS NOT NULL
        AND fg.home_sp_ra9_last5 IS NOT NULL
        AND fg.away_sp_ra9_last5 IS NOT NULL
        AND fg.home_sp_ip_per_start_last5 IS NOT NULL
        AND fg.away_sp_ip_per_start_last5 IS NOT NULL
        AND fg.home_bp_outs_3d IS NOT NULL
        AND fg.away_bp_outs_3d IS NOT NULL

    -- new rolling 60 constraints
        AND fg.home_games_in_window_60 >= 20
        AND fg.away_games_in_window_60 >= 20
        AND fg.home_avg_runs_scored_60 IS NOT NULL
        AND fg.away_avg_runs_scored_60 IS NOT NULL
        AND fg.home_avg_runs_allowed_60 IS NOT NULL
        AND fg.away_avg_runs_allowed_60 IS NOT NULL

    ORDER BY fg.season, fg.game_id;
    """

    df = pd.read_sql(q, conn)
    conn.close()
    return df


def split(df):
    # Time split
    train = df[df["season"] <= 2021].copy()
    val   = df[df["season"] == 2022].copy()
    test  = df[df["season"] >= 2023].copy()
    return train, val, test


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def win_prob_from_lambdas(lam_home, lam_away, sims=30000, seed=7):
    lam_home = np.asarray(lam_home, dtype=float)
    lam_away = np.asarray(lam_away, dtype=float)

    lam_home = np.clip(lam_home, 0.05, None)
    lam_away = np.clip(lam_away, 0.05, None)

    if HAVE_SCIPY:
        p_tie = skellam.pmf(0, lam_home, lam_away)
        p_le0 = skellam.cdf(0, lam_home, lam_away)
        p_win = 1.0 - p_le0 + 0.5 * p_tie
        return np.clip(p_win, 1e-6, 1 - 1e-6)

    rng = np.random.default_rng(seed)
    H = rng.poisson(lam_home[:, None], size=(lam_home.shape[0], sims))
    A = rng.poisson(lam_away[:, None], size=(lam_away.shape[0], sims))
    p = (H > A).mean(axis=1) + 0.5 * (H == A).mean(axis=1)
    return np.clip(p, 1e-6, 1 - 1e-6)


def eval_runs(name, y_true, y_pred):
    return {
        "split": name,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "mean_true": float(np.mean(y_true)),
        "mean_pred": float(np.mean(y_pred)),
    }


def main():
    df = load_data()
    print(f"Total usable games: {len(df)}")

    train, val, test = split(df)
    print(f"Train {len(train)} | Val {len(val)} | Test {len(test)}")

    # Prepare X/y
    X_train = train[FEATURE_COLS].copy()
    X_val   = val[FEATURE_COLS].copy()
    X_test  = test[FEATURE_COLS].copy()

    yH_train, yA_train = train["home_runs"].astype(int), train["away_runs"].astype(int)
    yH_val,   yA_val   = val["home_runs"].astype(int),   val["away_runs"].astype(int)
    yH_test,  yA_test  = test["home_runs"].astype(int),  test["away_runs"].astype(int)

    # Tell LightGBM which columns are categorical
    cat_cols = ["venue_id"]

    for c in cat_cols:
        X_train[c] = X_train[c].astype("category")
        X_val[c]   = X_val[c].astype("category")
        X_test[c]  = X_test[c].astype("category")


    # Season can be numeric (captures run environment trend)
    # If you want season categorical, you can set it to category too.

    common_params = dict(
        objective="tweedie",
        tweedie_variance_power=1.4,
        learning_rate=0.03,
        n_estimators=20000,          # allow many, but we will early-stop
        num_leaves=16,               # MUCH smaller
        max_depth=6,                 # cap complexity
        min_child_samples=200,       # force broader splits
        min_child_weight=1.0,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=1.0,               # L1
        reg_lambda=5.0,              # L2
        random_state=7,
        n_jobs=-1,
    )


    home_model = LGBMRegressor(**common_params)
    away_model = LGBMRegressor(**common_params)

    print("Training home runs LGBM...")
    home_model.fit(
        X_train, yH_train,
        eval_set=[(X_val, yH_val)],
        eval_metric="tweedie",
        categorical_feature=cat_cols,
        callbacks=[early_stopping(stopping_rounds=200), log_evaluation(200)],
    )

    print("Training away runs LGBM...")
    away_model.fit(
        X_train, yA_train,
        eval_set=[(X_val, yA_val)],
        eval_metric="tweedie",
        categorical_feature=cat_cols,
        callbacks=[early_stopping(stopping_rounds=200), log_evaluation(200)],
    )   

    # Predict lambdas
    predH_train = home_model.predict(X_train)
    predA_train = away_model.predict(X_train)
    predH_val   = home_model.predict(X_val)
    predA_val   = away_model.predict(X_val)
    predH_test  = home_model.predict(X_test)
    predA_test  = away_model.predict(X_test)

    # Evaluate run predictions
    runs_perf = []
    runs_perf.append({"target":"home_runs", **eval_runs("train", yH_train, predH_train)})
    runs_perf.append({"target":"home_runs", **eval_runs("val",   yH_val,   predH_val)})
    runs_perf.append({"target":"home_runs", **eval_runs("test",  yH_test,  predH_test)})
    runs_perf.append({"target":"away_runs", **eval_runs("train", yA_train, predA_train)})
    runs_perf.append({"target":"away_runs", **eval_runs("val",   yA_val,   predA_val)})
    runs_perf.append({"target":"away_runs", **eval_runs("test",  yA_test,  predA_test)})

    print("\nRuns-model performance (LightGBM Tweedie):")
    print(pd.DataFrame(runs_perf).to_string(index=False))

    # Derived win probability
    from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
    ywin_train = (yH_train.values > yA_train.values).astype(int)
    ywin_val   = (yH_val.values > yA_val.values).astype(int)
    ywin_test  = (yH_test.values > yA_test.values).astype(int)

    p_train = win_prob_from_lambdas(predH_train, predA_train)
    p_val   = win_prob_from_lambdas(predH_val,   predA_val)
    p_test  = win_prob_from_lambdas(predH_test,  predA_test)

    win_perf = pd.DataFrame([
        {"split":"train", "log_loss": log_loss(ywin_train, p_train),
         "brier": brier_score_loss(ywin_train, p_train),
         "auc": roc_auc_score(ywin_train, p_train),
         "mean_p_homewin": float(np.mean(p_train))},
        {"split":"val", "log_loss": log_loss(ywin_val, p_val),
         "brier": brier_score_loss(ywin_val, p_val),
         "auc": roc_auc_score(ywin_val, p_val),
         "mean_p_homewin": float(np.mean(p_val))},
        {"split":"test", "log_loss": log_loss(ywin_test, p_test),
         "brier": brier_score_loss(ywin_test, p_test),
         "auc": roc_auc_score(ywin_test, p_test),
         "mean_p_homewin": float(np.mean(p_test))},
    ])

    print("\nWin-prob performance (derived from runs):")
    print(win_perf.to_string(index=False))

    # Save artifacts
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(home_model, "artifacts/runs_model_home_lgbm_poisson.joblib")
    joblib.dump(away_model, "artifacts/runs_model_away_lgbm_poisson.joblib")
    print("\nSaved:")
    print(" - artifacts/runs_model_home_lgbm_poisson.joblib")
    print(" - artifacts/runs_model_away_lgbm_poisson.joblib")

    # Feature importance (top 15)
    imp_home = pd.Series(home_model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    imp_away = pd.Series(away_model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nTop feature importances (home runs):")
    print(imp_home.head(15).to_string())
    print("\nTop feature importances (away runs):")
    print(imp_away.head(15).to_string())


if __name__ == "__main__":
    main()
