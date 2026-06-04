#!/bin/bash
set -e
YESTERDAY=$(TZ="America/Los_Angeles" date -d "yesterday" +%Y-%m-%d 2>/dev/null || TZ="America/Los_Angeles" date -v-1d +%Y-%m-%d)
TODAY=$(TZ="America/Los_Angeles" date +%Y-%m-%d)
SCRIPT=$1
shift

PIPELINE_FAILURES=()

run_step_continue() {
  local step_name="$1"
  shift
  echo ""
  echo "=== START: $step_name ==="
  set +e
  "$@"
  local rc=$?
  set -e
  if [ $rc -ne 0 ]; then
    echo "=== FAILED: $step_name (exit $rc) ===" >&2
    PIPELINE_FAILURES+=("$step_name")
  else
    echo "=== OK: $step_name ==="
  fi
}

send_pipeline_alert() {
  local job_name="$1"
  local msg="$2"
  python features/pipeline_alert.py \
    --job "$job_name" \
    --date "$TODAY" \
    --message "$msg" \
    --failures "$(IFS=,; echo "${PIPELINE_FAILURES[*]}")" \
    || true
}

finish_pipeline() {
  echo ""
  echo "=== Pipeline smoke test ==="
  set +e
  python features/pipeline_smoke_test.py --date "$TODAY" --fail-on-empty --notify
  local smoke_rc=$?
  set -e
  if [ $smoke_rc -ne 0 ]; then
    PIPELINE_FAILURES+=("smoke_test")
  fi
  if [ ${#PIPELINE_FAILURES[@]} -gt 0 ]; then
    echo ""
    echo "PIPELINE FINISHED WITH ${#PIPELINE_FAILURES[@]} STEP FAILURE(S):" >&2
    for f in "${PIPELINE_FAILURES[@]}"; do
      echo "  - $f" >&2
    done
    return 1
  fi
  return 0
}

run_morning_inference_steps() {
  run_step_continue ingest_standings python ingest/ingest_standings.py --date $TODAY
  run_step_continue ingest_transactions python ingest/ingest_transactions.py --date $TODAY --days 14
  run_step_continue ingest_active_rosters python ingest/ingest_active_rosters.py --date $TODAY
  run_step_continue backfill_sp python ingest/backfill_startingpitchers.py
  run_step_continue build_pitch_mix python features/build_pitch_mix_rolling.py --date $TODAY
  run_step_continue ingest_lineups python ingest/ingest_lineups.py --date $TODAY --once
  run_step_continue umpire_features python features/umpire_features.py --date $TODAY
  run_step_continue weather_forecast python features/weather_forecast.py --date $TODAY
  run_step_continue market_movement python features/market_movement.py --date $TODAY --pull morning
  run_step_continue build_lineup_matchups python features/build_lineup_matchups.py --start $TODAY --end $TODAY
  run_step_continue build_features_skip python features/build_features1.py --date $TODAY --skip-statcast
  run_step_continue build_trends python features/build_trends.py --date $TODAY
  run_step_continue build_features_statcast python features/build_features1.py --date $TODAY --statcast-only
  run_step_continue inference_v10 python inference/inference_v10.py --date $TODAY --model artifacts/baseline_v10_production.joblib --fill_missing
  run_step_continue inference_v10_total python inference/inference_v10_total.py --date $TODAY --model artifacts/totals_v10_umpire_runs_boost_sp_xwoba_total.joblib
  run_step_continue inference_props python inference/inference_props_v1.py --date $TODAY \
      --batter-model artifacts/props_v1_expanded.joblib \
      --pitcher-model artifacts/pitcher_props_v1.joblib \
      --pitcher-walks-model artifacts/pitcher_walks_v1.joblib \
      --pitcher-hits-model artifacts/pitcher_hits_v1.joblib \
      --pitcher-er-model artifacts/pitcher_er_v1.joblib
  run_step_continue build_model_performance python features/build_model_performance.py --date $TODAY
  run_step_continue project_standings python features/project_standings.py --date $TODAY
  run_step_continue build_edges python features/build_edges.py --date $TODAY
  run_step_continue export_bq python export_to_bigquery.py --date $TODAY
}

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
  lineup_chain_refresh)
    # Test/recovery: rerun lineup-dependent chain for GAME_IDS (comma-separated).
    : "${GAME_IDS:?Set GAME_IDS=123,456}"
    python -c "from ingest.ingest_lineups import run_inference_chain; import os; g=[int(x) for x in os.environ['GAME_IDS'].split(',') if x.strip()]; run_inference_chain('${TODAY}', 'public', g)"
    ;;
  ingest_statcast)
    python ingest/ingest_statcast.py --date $YESTERDAY
    python features/build_pitch_mix_rolling.py --date $TODAY
    ;;
  closing_odds)
    python features/closing_odds_scheduler.py --date $TODAY
    ;;
  market_movement)
    python features/market_movement.py --date $TODAY --pull morning
    ;;
  odds_quota_check)
    python features/odds_quota_alert.py
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
    MORNING_MAX_ATTEMPTS="${MORNING_MAX_ATTEMPTS:-3}"
    MORNING_RETRY_WAIT_SEC="${MORNING_RETRY_WAIT_SEC:-180}"
    attempt=1
    while [ "$attempt" -le "$MORNING_MAX_ATTEMPTS" ]; do
      PIPELINE_FAILURES=()
      echo ""
      echo "=== Morning inference attempt ${attempt}/${MORNING_MAX_ATTEMPTS} (${TODAY}) ==="
      run_morning_inference_steps
      if finish_pipeline; then
        exit 0
      fi
      if [ "$attempt" -lt "$MORNING_MAX_ATTEMPTS" ]; then
        echo "Attempt ${attempt} failed; retrying in ${MORNING_RETRY_WAIT_SEC}s..." >&2
        sleep "$MORNING_RETRY_WAIT_SEC"
      fi
      attempt=$((attempt + 1))
    done
    send_pipeline_alert "morning_inference" \
      "All ${MORNING_MAX_ATTEMPTS} morning inference attempts failed for ${TODAY}."
    exit 1
    ;;
  standings_refresh)
    python ingest/ingest_standings.py --date $TODAY
    python features/project_standings.py --date $TODAY
    python export_to_bigquery.py --date $TODAY --only standings
    ;;
  model_performance_refresh)
    python features/build_model_performance.py --date $TODAY
    python export_to_bigquery.py --date $TODAY --only model_performance
    ;;
  train_pitcher_extras)
    python ingest/backfill_pitcher_starts.py --start 2015-01-01
    python models/train_pitcher_extras_v1.py --require-calibration --max-calib-gap-pp 8
    ;;
  transactions_refresh)
    python ingest/ingest_transactions.py --date $TODAY --days 60
    python export_to_bigquery.py --date $TODAY --only transactions
    ;;
  trends_refresh)
    python features/build_trends.py --date $TODAY
    python export_to_bigquery.py --date $TODAY --only trends
    ;;
  edges_refresh)
    python features/build_edges.py --date $TODAY
    python export_to_bigquery.py --date $TODAY --only edges
    ;;
  totals_edges_refresh)
    PIPELINE_FAILURES=()
    run_step_continue inference_v10_total python inference/inference_v10_total.py --date $TODAY \
        --model artifacts/totals_v10_umpire_runs_boost_sp_xwoba_total.joblib
    run_step_continue build_edges python features/build_edges.py --date $TODAY
    run_step_continue export_bq python export_to_bigquery.py --date $TODAY
    finish_pipeline || exit 1
    ;;
  sp_export_refresh)
    PIPELINE_FAILURES=()
    run_step_continue backfill_sp python ingest/backfill_startingpitchers.py
    run_step_continue export_bq python export_to_bigquery.py --date $TODAY
    finish_pipeline || exit 1
    ;;
  export_bq_only)
    python export_to_bigquery.py --date $TODAY
    ;;
  inference_export_refresh)
    PIPELINE_FAILURES=()
    run_step_continue inference_v10 python inference/inference_v10.py --date $TODAY \
        --model artifacts/baseline_v10_production.joblib --fill_missing
    run_step_continue inference_v10_total python inference/inference_v10_total.py --date $TODAY \
        --model artifacts/totals_v10_umpire_runs_boost_sp_xwoba_total.joblib
    run_step_continue export_bq python export_to_bigquery.py --date $TODAY
    finish_pipeline || exit 1
    ;;
  roster_props_edges_refresh)
    PIPELINE_FAILURES=()
    run_step_continue ingest_active_rosters python ingest/ingest_active_rosters.py --date $TODAY
    run_step_continue inference_props python inference/inference_props_v1.py --date $TODAY \
        --batter-model artifacts/props_v1_expanded.joblib \
        --pitcher-model artifacts/pitcher_props_v1.joblib \
        --pitcher-walks-model artifacts/pitcher_walks_v1.joblib \
        --pitcher-hits-model artifacts/pitcher_hits_v1.joblib \
        --pitcher-er-model artifacts/pitcher_er_v1.joblib
    run_step_continue build_edges python features/build_edges.py --date $TODAY
    run_step_continue export_props python export_to_bigquery.py --date $TODAY --only props
    run_step_continue export_edges python export_to_bigquery.py --date $TODAY --only edges
    finish_pipeline || exit 1
    ;;
  trends_era_refresh)
    python ingest/backfill_pitcher_starts.py --start 2026-01-01
    python ingest/backfill_pitcher_appearances.py --start 2026-01-01
    python features/build_trends.py --date $TODAY
    python export_to_bigquery.py --date $TODAY --only trends
    ;;
  diagnose_pitcher_starts)
    DIAGNOSE_NAME="${1:-Ohtani}" python - <<'PY'
import os
from sqlalchemy import create_engine, text

name = os.environ["DIAGNOSE_NAME"]
engine = create_engine(os.environ["PG_DSN"], pool_pre_ping=True)
query = text("""
WITH ids AS (
    SELECT DISTINCT home_sp_id AS pitcher_id
    FROM public.game_starting_pitchers
    WHERE home_sp_name ILIKE :name AND home_sp_id IS NOT NULL
    UNION
    SELECT DISTINCT away_sp_id
    FROM public.game_starting_pitchers
    WHERE away_sp_name ILIKE :name AND away_sp_id IS NOT NULL
    UNION
    SELECT DISTINCT pitcher_id
    FROM public.pitcher_prop_predictions
    WHERE pitcher_name ILIKE :name AND pitcher_id IS NOT NULL
)
SELECT
    ps.game_date,
    ps.pitcher_id,
    COALESCE(gsp.home_sp_name, gsp.away_sp_name, ppp.pitcher_name, 'Pitcher ' || ps.pitcher_id::text) AS pitcher_name,
    t.team_name,
    ps.innings_pitched,
    ps.runs_allowed,
    ps.earned_runs
FROM public.pitcher_starts ps
JOIN ids ON ids.pitcher_id = ps.pitcher_id
LEFT JOIN public.game_starting_pitchers gsp ON gsp.game_id = ps.game_id
LEFT JOIN public.pitcher_prop_predictions ppp ON ppp.pitcher_id = ps.pitcher_id
LEFT JOIN public.teams t ON t.mlb_team_id = ps.team_id
WHERE EXTRACT(YEAR FROM ps.game_date) = 2026
ORDER BY ps.game_date DESC
LIMIT 20
""")
with engine.connect() as conn:
    rows = conn.execute(query, {"name": f"%{name}%"}).fetchall()
print(f"Pitcher start rows for {name}: {len(rows)}")
for row in rows:
    print(tuple(row))
PY
    ;;
  project_standings)
    python features/project_standings.py --date $TODAY "$@"
    ;;
  props_inference)
    python ingest/ingest_lineups.py --date $TODAY --once
    python ingest/ingest_active_rosters.py --date $TODAY
    python features/build_lineup_matchups.py --start $TODAY --end $TODAY
    python features/build_features1.py --date $TODAY --statcast-only
    python inference/inference_props_v1.py --date $TODAY \
        --batter-model artifacts/props_v1_expanded.joblib \
        --pitcher-model artifacts/pitcher_props_v1.joblib \
        --pitcher-walks-model artifacts/pitcher_walks_v1.joblib \
        --pitcher-hits-model artifacts/pitcher_hits_v1.joblib \
        --pitcher-er-model artifacts/pitcher_er_v1.joblib
    python export_to_bigquery.py --date $TODAY
    ;;
  early_inference)
    PIPELINE_FAILURES=()
    run_step_continue backfill_games python ingest/backfill_games.py --start $TODAY --end $TODAY
    run_step_continue backfill_sp python ingest/backfill_startingpitchers.py
    run_step_continue ingest_active_rosters python ingest/ingest_active_rosters.py --date $TODAY
    run_step_continue build_features_skip python features/build_features1.py --date $TODAY --skip-statcast
    run_step_continue inference_v10 python inference/inference_v10.py --date $TODAY \
        --model artifacts/baseline_v10_production.joblib \
        --fill_missing --no-lineup
    run_step_continue inference_v10_total python inference/inference_v10_total.py --date $TODAY \
        --model artifacts/totals_v10_umpire_runs_boost_sp_xwoba_total.joblib
    run_step_continue inference_props python inference/inference_props_v1.py --date $TODAY \
        --batter-model artifacts/props_v1_expanded.joblib \
        --pitcher-model artifacts/pitcher_props_v1.joblib \
        --pitcher-walks-model artifacts/pitcher_walks_v1.joblib \
        --pitcher-hits-model artifacts/pitcher_hits_v1.joblib \
        --pitcher-er-model artifacts/pitcher_er_v1.joblib \
        --all-roster
    run_step_continue export_bq python export_to_bigquery.py --date $TODAY
    finish_pipeline || exit 1
    ;;
  *)
    echo "Unknown script: $SCRIPT"
    exit 1
    ;;
esac
