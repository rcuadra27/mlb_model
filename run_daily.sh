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
    python features/market_movement.py --date $TODAY
    ;;
  update_scores)
    python ingest/backfill_games.py --start $TODAY --end $TODAY
    python export_to_bigquery.py --date $TODAY
    ;;
  *)
    echo "Unknown script: $SCRIPT"
    exit 1
    ;;
esac
