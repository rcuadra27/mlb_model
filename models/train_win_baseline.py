import os
import joblib
import psycopg2
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


RAW_COLS = [
    "home_win_pct_30", "away_win_pct_30",
    "home_runs_for_30", "away_runs_for_30",
    "home_runs_against_30", "away_runs_against_30",
    "home_sp_ra9_last5", "away_sp_ra9_last5",
    "home_sp_ip_per_start_last5", "away_sp_ip_per_start_last5",
    "home_bp_outs_1d", "away_bp_outs_1d",
    "home_bp_outs_3d", "away_bp_outs_3d",
    "home_bp_outs_5d", "away_bp_outs_5d",
]

DIFF_FEATURES = [
    "diff_win_pct_30",
    "diff_runs_for_30",
    "diff_runs_against_30",
    "diff_sp_ra9_last5",     # note sign below
    "diff_sp_ip_last5",
    "diff_bp_outs_1d",
    "diff_bp_outs_3d",
    "diff_bp_outs_5d",
]


TARGET_COL = "home_win"


def load_data():
    conn = psycopg2.connect(os.environ["PG_DSN"])
    q = f"""
    SELECT
        game_id,
        season,
        {",".join(RAW_COLS)},
        home_win
    FROM features_game
    WHERE
        home_win_pct_30 IS NOT NULL
        AND away_win_pct_30 IS NOT NULL
        AND home_runs_for_30 IS NOT NULL
        AND away_runs_for_30 IS NOT NULL
        AND home_runs_against_30 IS NOT NULL
        AND away_runs_against_30 IS NOT NULL
        AND home_sp_ra9_last5 IS NOT NULL
        AND away_sp_ra9_last5 IS NOT NULL
        AND home_sp_ip_per_start_last5 IS NOT NULL
        AND away_sp_ip_per_start_last5 IS NOT NULL
        AND home_bp_outs_1d IS NOT NULL AND away_bp_outs_1d IS NOT NULL
        AND home_bp_outs_3d IS NOT NULL AND away_bp_outs_3d IS NOT NULL
        AND home_bp_outs_5d IS NOT NULL AND away_bp_outs_5d IS NOT NULL
    ORDER BY season, game_id;
    """

    df = pd.read_sql(q, conn)
    conn.close()
    return df


def split_data(df):
    train = df[df.season <= 2021]
    val = df[df.season == 2022]
    test = df[df.season >= 2023]

    X_train = train[DIFF_FEATURES]
    y_train = train[TARGET_COL].astype(int)

    X_val = val[DIFF_FEATURES]
    y_val = val[TARGET_COL].astype(int)

    X_test = test[DIFF_FEATURES]
    y_test = test[TARGET_COL].astype(int)

    return X_train, y_train, X_val, y_val, X_test, y_test


def evaluate(name, y_true, y_prob):
    return {
        "split": name,
        "log_loss": log_loss(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "auc": roc_auc_score(y_true, y_prob),
    }


def main():
    print("Loading data...")
    df = load_data()
    df["diff_win_pct_30"] = df["home_win_pct_30"] - df["away_win_pct_30"]
    df["diff_runs_for_30"] = df["home_runs_for_30"] - df["away_runs_for_30"]
    df["diff_runs_against_30"] = df["home_runs_against_30"] - df["away_runs_against_30"]

    df["diff_bp_outs_1d"] = df["away_bp_outs_1d"] - df["home_bp_outs_1d"]
    df["diff_bp_outs_3d"] = df["away_bp_outs_3d"] - df["home_bp_outs_3d"]
    df["diff_bp_outs_5d"] = df["away_bp_outs_5d"] - df["home_bp_outs_5d"]

    # Pitching: lower RA/9 is better, so make it "advantage" as away - home
    df["diff_sp_ra9_last5"] = df["away_sp_ra9_last5"] - df["home_sp_ra9_last5"]

    # Higher IP/start is better for the starter, so home - away is advantage
    df["diff_sp_ip_last5"] = df["home_sp_ip_per_start_last5"] - df["away_sp_ip_per_start_last5"]

    print(f"Total usable games: {len(df)}")

    X_train, y_train, X_val, y_val, X_test, y_test = split_data(df)

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=200,
        ))
    ])

    print("Training model...")
    model.fit(X_train, y_train)

    results = []
    results.append(evaluate("train", y_train, model.predict_proba(X_train)[:, 1]))
    results.append(evaluate("val", y_val, model.predict_proba(X_val)[:, 1]))
    results.append(evaluate("test", y_test, model.predict_proba(X_test)[:, 1]))

    results_df = pd.DataFrame(results)
    print("\nModel performance:")
    print(results_df.to_string(index=False))

    # Save model
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "artifacts/win_model_diff_plus_bullpen.joblib")


    print("\nModel saved to artifacts/win_model_diff_plus_bullpen.joblib")

    # Inspect coefficients
    coef = model.named_steps["logreg"].coef_[0]
    coef_df = pd.DataFrame({
        "feature": DIFF_FEATURES,
        "coefficient": coef
    }).sort_values("coefficient", ascending=False)

    print("\nModel coefficients:")
    print(coef_df.to_string(index=False))


if __name__ == "__main__":
    main()
