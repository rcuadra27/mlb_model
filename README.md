# MLB Game Predictor

A machine learning system that predicts expected runs for each team in MLB games and converts these predictions into win probabilities. The system integrates with betting odds APIs to identify value betting opportunities.

## Overview

This project uses LightGBM regression models to predict expected runs scored by home and away teams. The predictions are then converted to win probabilities using Poisson distribution simulation, which can be compared against sportsbook odds to calculate expected value (EV) for betting opportunities.

## Features

- **Expected Runs Prediction**: Separate LightGBM models for home and away team runs using Tweedie loss function
- **Win Probability Calculation**: Converts expected runs to win probabilities using Poisson simulation
- **Betting Odds Integration**: Pulls moneyline odds from The Odds API
- **Expected Value Analysis**: Calculates EV and identifies profitable betting opportunities
- **Comprehensive Feature Engineering**: Includes team form, starting pitcher stats, bullpen usage, venue effects, and more

## Project Structure

```
MLB_model/
├── artifacts/              # Trained model files (.joblib)
├── features/              # Feature engineering pipeline
│   └── build_features.py
├── ingest/                # Data ingestion scripts
│   ├── backfill_games.py
│   ├── backfill_games_stadiums.py
│   ├── backfill_pitcher_appearances.py
│   ├── backfill_pitcher_starts.py
│   ├── backfill_reliever_entry_context.py
│   ├── backfill_startingpitchers.py
│   └── pull_odds_moneyline.py
├── models/                # Model training scripts
│   ├── train_expected_runs_lgbm.py
│   ├── train_expected_runs.py
│   └── train_win_baseline.py
├── odds/                  # Odds API integration
│   └── the_odds_api.py
└── pricing/               # Expected value calculations
    └── build_ev_board.py
```

## Key Components

### Model Training (`models/train_expected_runs_lgbm.py`)

Trains separate LightGBM models for home and away team runs using:
- **Objective**: Tweedie loss (variance power 1.4) - suitable for count data with many zeros
- **Features**: Team form (30-game rolling stats), starting pitcher performance (last 5 starts), bullpen usage, venue effects, season trends
- **Evaluation**: Time-based splits (train: ≤2021, val: 2022, test: ≥2023)
- **Output**: Expected runs (lambda) for each team

### Feature Engineering (`features/build_features.py`)

Builds rolling window features including:
- Team win percentage (30 games)
- Runs scored/allowed (30 games)
- Starting pitcher RA9 and innings per start (last 5 starts)
- Bullpen usage (1d, 3d, 5d windows)
- Team baselines (60-game rolling averages)

### Odds Integration (`ingest/pull_odds_moneyline.py`)

Pulls moneyline odds from The Odds API and stores in PostgreSQL database with:
- Multiple sportsbook support
- Game-to-odds mapping
- Timestamp tracking

### Expected Value Calculation (`pricing/build_ev_board.py`)

Generates betting recommendations by:
1. Loading trained models and predicting expected runs
2. Converting runs to win probabilities via Poisson simulation
3. Pulling latest odds from database
4. Calculating expected value for each side
5. Ranking opportunities by best EV

## Model Features

The models use the following features:

- **Team Form**:
  - `home_win_pct_30`, `away_win_pct_30`
  - `home_runs_for_30`, `away_runs_for_30`
  - `home_runs_against_30`, `away_runs_against_30`

- **Starting Pitchers**:
  - `home_sp_ra9_last5`, `away_sp_ra9_last5`
  - `home_sp_ip_per_start_last5`, `away_sp_ip_per_start_last5`

- **Bullpen**:
  - `home_bp_outs_3d`, `away_bp_outs_3d`
  - `home_bp_hlev_outs_1d/3d/5d`, `away_bp_hlev_outs_1d/3d/5d`

- **Context**:
  - `season` (captures run environment trends)
  - `venue_id` (categorical, captures park effects)

- **Team Baselines**:
  - `home_avg_runs_scored_60`, `home_avg_runs_allowed_60`
  - `away_avg_runs_scored_60`, `away_avg_runs_allowed_60`

## Requirements

### Environment Variables

- `PG_DSN`: PostgreSQL connection string (e.g., `postgresql://user:pass@host:port/dbname`)
- `ODDS_API_KEY`: API key for The Odds API

### Python Dependencies

- `lightgbm` - Gradient boosting models
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `psycopg2` - PostgreSQL adapter
- `scikit-learn` - Model evaluation metrics
- `scipy` - Statistical functions (optional, for Skellam distribution)
- `joblib` - Model serialization
- `requests` - API calls

## Usage

### 1. Data Ingestion

Ingest historical game data:
```bash
python ingest/backfill_games.py
python ingest/backfill_startingpitchers.py
python ingest/backfill_reliever_entry_context.py
```

### 2. Feature Engineering

Build features for all games:
```bash
python features/build_features.py
```

### 3. Model Training

Train the expected runs models:
```bash
python models/train_expected_runs_lgbm.py
```

This will:
- Load features from the database
- Train home and away run prediction models
- Evaluate on validation and test sets
- Save models to `artifacts/` directory
- Print performance metrics and feature importances

### 4. Pull Odds

Pull current odds for a specific date:
```bash
python ingest/pull_odds_moneyline.py --game_date 2024-06-15
```

### 5. Generate EV Board

Generate expected value analysis for a specific date:
```bash
python pricing/build_ev_board.py --game_date 2024-06-15
```

## Model Performance

The models are evaluated on:
- **Runs Prediction**: MAE and RMSE for home/away runs
- **Win Probability**: Log loss, Brier score, and AUC-ROC

Performance metrics are printed during training and can be used to track model quality over time.

## Database Schema

The project uses PostgreSQL with tables including:
- `games` - Game results and metadata
- `features_game` - Engineered features per game
- `odds_ml` - Betting odds data
- Additional tables for pitchers, teams, venues, etc.

## Notes

- Models use time-based splits to prevent data leakage
- Poisson simulation handles ties (extra innings) by assigning 0.5 win probability
- The system supports filtering by sportsbook for odds comparison
- Feature engineering handles missing data with appropriate defaults

## License

[Add your license here]

## Author

[Add your name/contact info here]
