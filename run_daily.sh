#!/bin/bash
YESTERDAY=$(TZ="America/Los_Angeles" date -d "yesterday" +%Y-%m-%d 2>/dev/null || TZ="America/Los_Angeles" date -v-1d +%Y-%m-%d)
TODAY=$(TZ="America/Los_Angeles" date +%Y-%m-%d)
SCRIPT=$1
shift

echo "Running $SCRIPT for $YESTERDAY → $TODAY"

case $SCRIPT in
  backfill_games)
    python ingest/backfill_games.py --start $YESTERDAY --end $TODAY
    ;;
  backfill_sp)
    python ingest/backfill_startingpitchers.py
    ;;
  backfill_pitcher_starts)
    python ingest/backfill_pitcher_starts.py
    ;;
  backfill_appearances)
    python ingest/backfill_pitcher_appearances.py
    ;;
  build_features)
    python features/build_features1.py --date $TODAY --skip-statcast
    ;;
  umpire_features)
    python features/umpire_features.py --date $TODAY
    ;;
  weather_forecast)
    python features/weather_forecast.py --date $TODAY
    ;;
  ingest_lineups)
    python ingest/ingest_lineups.py --date $TODAY
    ;;
  closing_odds)
    python features/closing_odds_scheduler.py --date $TODAY
    ;;
  market_movement)
    python features/market_movement.py --date $TODAY --pull morning
    ;;
  update_scores)
    # Pull finals from MLB Stats API for today and re-sync the past few days
    # so late-arriving finals (extra innings, suspended games) land in BigQuery.
    python ingest/backfill_games.py --start $TODAY --end $TODAY
    for OFFSET in 0 1 2 3; do
      D=$(TZ="America/Los_Angeles" date -d "$TODAY -$OFFSET day" +%Y-%m-%d 2>/dev/null \
          || TZ="America/Los_Angeles" date -v-${OFFSET}d -j -f %Y-%m-%d "$TODAY" +%Y-%m-%d)
      python export_to_bigquery.py --date $D || true
    done
    ;;
  morning_inference)
    python ingest/backfill_games.py --start $YESTERDAY --end $TODAY
    python ingest/backfill_startingpitchers.py
    python ingest/backfill_pitcher_starts.py
    python ingest/backfill_pitcher_appearances.py
    python ingest/backfill_reliever_entry_context.py
    python ingest/ingest_lineups.py --date $TODAY
    python features/umpire_features.py --date $TODAY
    python features/weather_forecast.py --date $TODAY
    python features/market_movement.py --date $TODAY --pull morning
    python features/build_pitchmix_rolling.py --date $TODAY
    python features/build_lineup_matchups.py --start $TODAY --end $TODAY
    python features/build_features1.py --date $TODAY --skip-statcast
    python inference/inference.py --date $TODAY --team_model artifacts/runs_model_v9.joblib --team_features artifacts/runs_model_v9_features.txt --no_calibrate --fill_missing
    python export_to_bigquery.py --date $TODAY
    ;;
  *)
    echo "Unknown script: $SCRIPT"
    exit 1
    ;;
esac
