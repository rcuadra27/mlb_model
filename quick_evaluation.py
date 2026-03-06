import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

pg_dsn = os.getenv("PG_DSN")
engine = create_engine(pg_dsn)

df = pd.read_sql("""
    SELECT
      f.p_home_win_poisson,
      CASE WHEN g.home_runs > g.away_runs THEN 1 ELSE 0 END AS home_win
    FROM public.features_game f
    JOIN public.games g USING (game_id)
    WHERE f.game_date BETWEEN '2024-01-01' AND '2024-12-31'
      AND f.p_home_win_poisson IS NOT NULL
""", engine)

y = df["home_win"].values
p = df["p_home_win_poisson"].values

print("Poisson model (2024 test)")
print("log_loss:", log_loss(y, p))
print("brier   :", brier_score_loss(y, p))
print("auc     :", roc_auc_score(y, p))
print("mean_p  :", np.mean(p))
print("mean_y  :", np.mean(y))
