#!/usr/bin/env bash
# Hourly Odds API quota check → Pushover (50% of monthly quota by default).
#
# Prerequisites:
#   - Pipeline image on Cloud Run (same as mlb-morning-inference)
#   - PUSHOVER_* + PIPELINE_ALERT_WEBHOOK_URL on mlb-morning-inference (copied below)
#   - NEW_ODDS_API_KEY rotated via scripts/rotate_odds_server_key.sh
#
# Usage:
#   ODDS_MONTHLY_QUOTA=20000 ODDS_ALERT_PCT=25 ./scripts/setup_odds_quota_pushover.sh

set -euo pipefail

PROJECT="${GCP_PROJECT:-mlb-model-491223}"
REGION="${GCP_REGION:-us-central1}"
QUOTA="${ODDS_MONTHLY_QUOTA:-20000}"
ALERT_PCT="${ODDS_ALERT_PCT:-25}"
JOB_NAME="mlb-odds-quota-check"
IMAGE="$(gcloud run jobs describe mlb-morning-inference --project="$PROJECT" --region="$REGION" --format='value(spec.template.spec.template.spec.containers[0].image)')"

echo "Pipeline image: $IMAGE"

# Copy Pushover env from morning job
ENV_JSON="$(gcloud run jobs describe mlb-morning-inference --project="$PROJECT" --region="$REGION" --format=json)"
copy_env() {
  python3 - <<'PY' "$ENV_JSON" "$1"
import json, sys
data = json.loads(sys.argv[1])
name = sys.argv[2]
for c in data["spec"]["template"]["spec"]["template"]["spec"]["containers"][0]["env"]:
    if c["name"] == name and c.get("value"):
        print(c["value"])
        break
PY
}

WEBHOOK="$(copy_env PIPELINE_ALERT_WEBHOOK_URL || true)"
PUSH_USER="$(copy_env PUSHOVER_USER_KEY || true)"
PUSH_TOKEN="$(copy_env PUSHOVER_APP_TOKEN || true)"

if [[ -z "$WEBHOOK" ]]; then
  echo "PIPELINE_ALERT_WEBHOOK_URL missing on mlb-morning-inference" >&2
  exit 1
fi

if gcloud run jobs describe "$JOB_NAME" --project="$PROJECT" --region="$REGION" &>/dev/null; then
  echo "Updating $JOB_NAME..."
  gcloud run jobs update "$JOB_NAME" \
    --project="$PROJECT" \
    --region="$REGION" \
    --image="$IMAGE" \
    --command=bash \
    --args=run_daily.sh,odds_quota_check \
    --set-secrets=ODDS_API_KEY=odds-api-key:latest \
    --set-env-vars="PIPELINE_ALERT_WEBHOOK_URL=${WEBHOOK},PUSHOVER_USER_KEY=${PUSH_USER},PUSHOVER_APP_TOKEN=${PUSH_TOKEN},ODDS_MONTHLY_QUOTA=${QUOTA},ODDS_ALERT_PCT=${ALERT_PCT}"
else
  echo "Creating $JOB_NAME..."
  gcloud run jobs create "$JOB_NAME" \
    --project="$PROJECT" \
    --region="$REGION" \
    --image="$IMAGE" \
    --command=bash \
    --args=run_daily.sh,odds_quota_check \
    --set-secrets=ODDS_API_KEY=odds-api-key:latest \
    --set-env-vars="PIPELINE_ALERT_WEBHOOK_URL=${WEBHOOK},PUSHOVER_USER_KEY=${PUSH_USER},PUSHOVER_APP_TOKEN=${PUSH_TOKEN},ODDS_MONTHLY_QUOTA=${QUOTA},ODDS_ALERT_PCT=${ALERT_PCT}" \
    --max-retries=0 \
    --task-timeout=120s \
    --memory=512Mi \
    --cpu=1
fi

SCHED="mlb-schedule-odds-quota-check"
if gcloud scheduler jobs describe "$SCHED" --project="$PROJECT" --location="$REGION" &>/dev/null; then
  echo "Scheduler $SCHED already exists"
else
  gcloud scheduler jobs create http "$SCHED" \
    --project="$PROJECT" \
    --location="$REGION" \
    --schedule="15 * * * *" \
    --time-zone="America/Los_Angeles" \
    --uri="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT}/jobs/${JOB_NAME}:run" \
    --http-method=POST \
    --oauth-service-account-email="683742314445-compute@developer.gserviceaccount.com" \
    --oauth-token-scope="https://www.googleapis.com/auth/cloud-platform"
fi

THRESHOLD="$(python3 -c "print(int(float('${QUOTA}') * float('${ALERT_PCT}') / 100))")"
echo "Done: $JOB_NAME runs hourly; Pushover when usage >= ${ALERT_PCT}% (${THRESHOLD} of ${QUOTA} credits)."
