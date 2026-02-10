import os
import joblib
import numpy as np
import pandas as pd
import psycopg2

import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data():
    conn = psycopg2.connect(os.environ["PG_DSN"])
    q = """
    SELECT
      fg.game_id,
      fg.season,
      g.game_date,
      fg.home_team_id,
      fg.away_team_id,

      fg.home_win_pct_30, fg.away_win_pct_30,
      fg.home_runs_for_30, fg.away_runs_for_30,
      fg.home_runs_against_30, fg.away_runs_against_30,
      fg.home_sp_ra9_last5, fg.away_sp_ra9_last5,
      fg.home_sp_ip_per_start_last5, fg.away_sp_ip_per_start_last5,
      fg.home_bp_outs_3d, fg.away_bp_outs_3d,

      g.home_runs,
      g.away_runs
    FROM features_game fg
    JOIN games g ON g.game_id = fg.game_id
    WHERE
      fg.home_win_pct_30 IS NOT NULL
      AND fg.away_win_pct_30 IS NOT NULL
      AND fg.home_sp_ra9_last5 IS NOT NULL
      AND fg.away_sp_ra9_last5 IS NOT NULL
      AND fg.home_bp_outs_3d IS NOT NULL
      AND fg.away_bp_outs_3d IS NOT NULL
      AND g.home_runs IS NOT NULL
      AND g.away_runs IS NOT NULL
    ORDER BY fg.season, fg.game_id;
    """
    df = pd.read_sql(q, conn)

    conn.close()
    return df



def split(df):
    train = df[df["season_int"] <= 2021].copy()
    val   = df[df["season_int"] == 2022].copy()
    test  = df[df["season_int"] >= 2023].copy()
    return train, val, test



def eval_basic(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return mae, rmse, float(y_true.mean()), float(y_pred.mean())


def main():
    df = load_data()
    df["season_int"] = df["season"].astype(int)

    df["season"] = pd.Categorical(df["season"], categories=sorted(df["season"].unique()))
    df["home_team_id"] = pd.Categorical(df["home_team_id"], categories=sorted(df["home_team_id"].unique()))
    df["away_team_id"] = pd.Categorical(df["away_team_id"], categories=sorted(df["away_team_id"].unique()))
    print("Total usable games:", len(df))
    train, val, test = split(df)
    print(f"Train {len(train)} | Val {len(val)} | Test {len(test)}")

    # HOME RUNS formula:
    # - season as categorical (run environment changes)
    # - team and opponent as categorical baselines (captures offense/defense/pitching context)
    # - plus numeric rolling form/pitching/bullpen features
    home_formula = """
    home_runs ~
      C(season) + C(home_team_id) + C(away_team_id)
      + home_win_pct_30 + away_win_pct_30
      + home_runs_for_30 + away_runs_for_30
      + home_runs_against_30 + away_runs_against_30
      + home_sp_ra9_last5 + away_sp_ra9_last5
      + home_sp_ip_per_start_last5 + away_sp_ip_per_start_last5
      + home_bp_outs_3d + away_bp_outs_3d
    """

    away_formula = """
    away_runs ~
      C(season) + C(away_team_id) + C(home_team_id)
      + away_win_pct_30 + home_win_pct_30
      + away_runs_for_30 + home_runs_for_30
      + away_runs_against_30 + home_runs_against_30
      + away_sp_ra9_last5 + home_sp_ra9_last5
      + away_sp_ip_per_start_last5 + home_sp_ip_per_start_last5
      + away_bp_outs_3d + home_bp_outs_3d
    """

    print("\nFitting Negative Binomial GLM for home runs...")
    home_model = smf.glm(
        formula=home_formula,
        data=train,
        family=sm.families.NegativeBinomial()
    ).fit()

    print("Fitting Negative Binomial GLM for away runs...")
    away_model = smf.glm(
        formula=away_formula,
        data=train,
        family=sm.families.NegativeBinomial()
    ).fit()

    # Predict expected runs (mu)
    for name, d in [("train", train), ("val", val), ("test", test)]:
        pred_home = home_model.predict(d)
        pred_away = away_model.predict(d)

        mae_h, rmse_h, mt_h, mp_h = eval_basic(d["home_runs"], pred_home)
        mae_a, rmse_a, mt_a, mp_a = eval_basic(d["away_runs"], pred_away)

        print(f"\n{name.upper()} results:")
        print(f"  HOME runs: MAE {mae_h:.3f} | RMSE {rmse_h:.3f} | mean_true {mt_h:.3f} | mean_pred {mp_h:.3f}")
        print(f"  AWAY runs: MAE {mae_a:.3f} | RMSE {rmse_a:.3f} | mean_true {mt_a:.3f} | mean_pred {mp_a:.3f}")

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(home_model, "artifacts/runs_model_home_negbin_sm.joblib")
    joblib.dump(away_model, "artifacts/runs_model_away_negbin_sm.joblib")
    print("\nSaved:")
    print(" - artifacts/runs_model_home_negbin_sm.joblib")
    print(" - artifacts/runs_model_away_negbin_sm.joblib")


if __name__ == "__main__":
    main()

