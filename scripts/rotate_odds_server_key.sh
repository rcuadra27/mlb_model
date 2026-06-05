#!/usr/bin/env bash
# Rotate The Odds API key on SERVER jobs only (never dashboard / Cloud Functions).
#
# 1. Regenerate the key at https://the-odds-api.com/account (invalidates the old key).
# 2. Run:
#      NEW_ODDS_API_KEY='your-new-key' ./scripts/rotate_odds_server_key.sh
#
# Removes ODDS_API_KEY from jobs that do not call the API.

set -euo pipefail

PROJECT="${GCP_PROJECT:-mlb-model-491223}"
REGION="${GCP_REGION:-us-central1}"
NEW_KEY="${NEW_ODDS_API_KEY:-}"

if [[ -z "$NEW_KEY" ]]; then
  echo "Set NEW_ODDS_API_KEY to the regenerated key from the-odds-api.com" >&2
  exit 1
fi

SERVER_JOBS=(
  mlb-morning-inference
  mlb-market-movement
  mlb-closing-odds
)

STRIP_JOBS=(
  mlb-update-scores
  mlb-ingest-lineups
  mlb-early-inference
)

echo "Storing key in Secret Manager (odds-api-key)..."
if gcloud secrets describe odds-api-key --project="$PROJECT" &>/dev/null; then
  printf '%s' "$NEW_KEY" | gcloud secrets versions add odds-api-key --project="$PROJECT" --data-file=-
else
  printf '%s' "$NEW_KEY" | gcloud secrets create odds-api-key --project="$PROJECT" --replication-policy=automatic --data-file=-
fi

for job in "${SERVER_JOBS[@]}"; do
  echo "Updating $job (ODDS_API_KEY from secret)..."
  gcloud run jobs update "$job" \
    --project="$PROJECT" \
    --region="$REGION" \
    --remove-env-vars=ODDS_API_KEY \
    --update-secrets=ODDS_API_KEY=odds-api-key:latest
done

for job in "${STRIP_JOBS[@]}"; do
  echo "Removing unused ODDS_API_KEY from $job..."
  gcloud run jobs update "$job" \
    --project="$PROJECT" \
    --region="$REGION" \
    --remove-env-vars=ODDS_API_KEY \
    2>/dev/null || true
done

echo "Ensuring get-daily-predictions has NO ODDS_API_KEY..."
gcloud run services update get-daily-predictions \
  --project="$PROJECT" \
  --region="$REGION" \
  --remove-env-vars=ODDS_API_KEY \
  2>/dev/null || true

echo "Done. Old key is dead after regeneration; server jobs use Secret Manager odds-api-key."
