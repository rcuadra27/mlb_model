# MLB Game Prediction System

A production ML system that predicts MLB game outcomes using LightGBM, deployed on Google Cloud Platform with a React dashboard at [mlbpredictor.com](https://mlbpredictor.com).

## Architecture
```
MLB Stats API / Statcast / Odds API
        ↓
   Cloud SQL (PostgreSQL) ← source of truth
        ↓
   Cloud Run Jobs (daily pipeline)
        ↓
   BigQuery (serving layer)
        ↓
   Cloud Function → React Dashboard (mlbpredictor.com)
```

## Model

- **Algorithm**: LightGBM (`runs_model_v8`)
- **Approach**: Residual regression — predicts deviation from league average runs (4.50)
- **Target**: `actual_runs - league_avg_runs_60d` per team per game
- **49 features**: SP ERA/xwoba/K-rate, lineup xwoba/barrel rate, team rolling stats (7d/15d/30d/60d), park factors, umpire tendencies, weather, market odds
- **Win probability**: Skellam distribution from predicted run totals (0.30–0.70 cap)
- **No calibration**: Raw Skellam probabilities (`--no_calibrate`)

## Project Structure
```
mlb_model/
├── artifacts/
│   ├── runs_model_v8.joblib          # Production model (local)
│   ├── runs_model_v8.txt             # Production model (container)
│   └── runs_model_v8_features.txt    # 49 feature names
│
├── ingest/                           # Data ingestion
│   ├── backfill_games.py             # Game schedule + results
│   ├── backfill_startingpitchers.py  # Starting pitcher assignments
│   ├── backfill_pitcher_starts.py    # Pitcher start stats (ERA, IP)
│   ├── backfill_pitcher_appearances.py # Bullpen appearance data
│   ├── backfill_games_stadiums.py    # Stadium/venue data
│   ├── backfill_reliever_entry_context.py # Reliever context
│   ├── backfill_weather.py           # Historical weather
│   ├── ingest_lineups.py             # Confirmed batting orders + triggers inference
│   ├── ingest_statcast.py            # Baseball Savant pitch data
│   ├── ingest_teams.py               # Team metadata
│   └── pull_odds_moneyline.py        # Odds ingestion
│
├── features/                         # Feature engineering
│   ├── build_features1.py            # Main feature pipeline (all 49 features)
│   ├── closing_odds_scheduler.py     # Pre-game odds collection + inference trigger
│   ├── umpire_features.py            # Umpire run tendency features
│   ├── weather_forecast.py           # Open-Meteo weather forecasts
│   ├── market_movement.py            # Morning vs closing line movement
│   ├── build_lineup_matchups.py      # Lineup vs SP matchup scores
│   ├── add_market_median_ml.py       # Market median ML odds
│   ├── add_poisson_winprob.py        # Poisson win probability
│   ├── add_run_preds_to_features.py  # Run predictions to features
│   └── build_blended_prob.py         # Blended probability model
│
├── inference/
│   └── inference.py                  # Main inference engine
│
├── models/
│   ├── train_runs_team_lgbm.py       # Model training script
│   └── train_expected_runs.py        # Expected runs model
│
├── calibration/
│   └── calibration.py                # Probability calibration
│
├── cloud_function/
│   ├── main.py                       # BigQuery serving endpoint
│   └── requirements.txt
│
├── dashboard/
│   └── src/
│       └── App.jsx                   # React dashboard
│
├── backtest/
│   └── backtest_moneyline.py         # Historical backtesting
│
├── evaluation/
│   └── evaluate_model_quality.py     # Model evaluation metrics
│
├── odds/
│   └── the_odds_api.py               # The Odds API wrapper
│
├── pricing/
│   └── build_ev_board.py             # Expected value calculations
│
├── export_to_bigquery.py             # PostgreSQL → BigQuery export
├── run_daily.sh                      # Daily pipeline entrypoint (Cloud Run)
├── Dockerfile.pipeline               # Pipeline container
├── Dockerfile.dashboard              # Dashboard container
└── requirements.txt                  # Python dependencies
```

## Daily Pipeline (Cloud Run Jobs, PT timezone)

| Time PT | Job | Script |
|---------|-----|--------|
| 6:00am | Backfill yesterday results + today schedule | `backfill_games.py --start YESTERDAY --end TODAY` |
| 6:05am | Starting pitcher assignments | `backfill_startingpitchers.py` |
| 6:15am | Pitcher start stats | `backfill_pitcher_starts.py` |
| 6:25am | Bullpen appearances | `backfill_pitcher_appearances.py` |
| 6:35am | Build features (no statcast) | `build_features1.py --date TODAY` |
| 6:50am | Morning odds + closing odds scheduler | `closing_odds_scheduler.py --date TODAY` |
| 7:45am | Umpire features | `umpire_features.py --date TODAY` |
| 8:00am | Weather forecasts | `weather_forecast.py --date TODAY` |
| 8:15am | Rebuild features with weather | `build_features1.py --date TODAY` |
| 8:30am | Lineup confirmation + inference trigger | `ingest_lineups.py --date TODAY` |

## Inference Chain (triggered by lineups or closing odds)
```
lineup confirmed OR closing odds stored 75min before first pitch
    → build_features1.py --date TODAY  (full statcast)
    → inference.py --date TODAY --no_calibrate --fill_missing
    → export_to_bigquery.py --date TODAY
    → game appears on dashboard
```

## Key Design Decisions

- **Closing odds frozen pre-game**: `store_closing_odds` uses `AND closing_p_home IS NULL` — never overwrites
- **Live odds never enter system**: `fetch_all_odds` skips `commence_time <= now_utc`
- **Morning odds stored once**: `store_morning_odds` uses `AND morning_p_home IS NULL`
- **BigQuery explicit schema**: No `autodetect=True` — prevents type conflicts
- **PT timezone throughout**: All date logic uses `America/Los_Angeles`

## GCP Resources

- **Project**: `mlb-model-491223`
- **Region**: `us-central1`
- **Cloud SQL**: `mlb-postgres` (PostgreSQL 15)
- **Pipeline image**: `gcr.io/mlb-model-491223/mlb-pipeline:v1`
- **Dashboard image**: `gcr.io/mlb-model-491223/mlb-dashboard:v1`
- **Cloud Function**: `get-daily-predictions`
- **BigQuery**: `mlb_model_logs.daily_games`

## Environment Variables
```bash
PG_DSN=postgresql+psycopg2://user:password@host/dbname
ODDS_API_KEY=your_odds_api_key
```

## Local Development
```bash
# Start Cloud SQL proxy
./cloud-sql-proxy mlb-model-491223:us-central1:mlb-postgres --port=5434

# Run pipeline manually
PG_DSN="..." python features/build_features1.py --date 2026-03-31
PG_DSN="..." ODDS_API_KEY="..." python inference/inference.py --date 2026-03-31 --no_calibrate --fill_missing
PG_DSN="..." python export_to_bigquery.py --date 2026-03-31
```
