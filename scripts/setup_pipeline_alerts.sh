#!/usr/bin/env bash
# Create Cloud Monitoring notification channel + alert policies for pipeline failures.
#
# Usage:
#   PIPELINE_ALERT_EMAIL=you@example.com ./scripts/setup_pipeline_alerts.sh
#
# Optional: also set PIPELINE_ALERT_WEBHOOK_URL on the Cloud Run job for instant
# push/SMS via Pushover, Slack, etc. (see features/pipeline_alert.py).

set -euo pipefail

PROJECT="${GCP_PROJECT:-mlb-model-491223}"
EMAIL="${PIPELINE_ALERT_EMAIL:-}"

if [[ -z "$EMAIL" ]]; then
  echo "Set PIPELINE_ALERT_EMAIL (your address for Monitoring emails)." >&2
  exit 1
fi

TOKEN="$(gcloud auth print-access-token)"
BASE="https://monitoring.googleapis.com/v3/projects/${PROJECT}"

echo "Creating email notification channel for ${EMAIL}..."
CHANNEL_JSON="$(curl -sS -X POST \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  "${BASE}/notificationChannels" \
  -d "{
    \"type\": \"email\",
    \"displayName\": \"MLB pipeline alerts\",
    \"labels\": {\"email_address\": \"${EMAIL}\"},
    \"enabled\": true
  }")"

CHANNEL_NAME="$(echo "$CHANNEL_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin).get('name',''))" 2>/dev/null || true)"
if [[ -z "$CHANNEL_NAME" ]]; then
  echo "Channel response: $CHANNEL_JSON" >&2
  echo "If channel already exists, list channels in Cloud Console → Monitoring → Alerting → Edit channels" >&2
  exit 1
fi
echo "Channel: $CHANNEL_NAME"
echo "(Check your inbox and click Verify on the Monitoring email channel.)"

create_policy() {
  local display_name="$1"
  local filter="$2"
  local doc="$3"
  curl -sS -X POST \
    -H "Authorization: Bearer ${TOKEN}" \
    -H "Content-Type: application/json" \
    "${BASE}/alertPolicies" \
    -d "{
      \"displayName\": \"${display_name}\",
      \"documentation\": {\"content\": \"${doc}\", \"mimeType\": \"text/markdown\"},
      \"combiner\": \"OR\",
      \"enabled\": true,
      \"notificationChannels\": [\"${CHANNEL_NAME}\"],
      \"alertStrategy\": {\"autoClose\": \"1800s\"},
      \"conditions\": [{
        \"displayName\": \"${display_name}\",
        \"conditionThreshold\": {
          \"filter\": \"${filter}\",
          \"comparison\": \"COMPARISON_GT\",
          \"thresholdValue\": 0,
          \"duration\": \"0s\",
          \"trigger\": {\"count\": 1},
          \"aggregations\": [{
            \"alignmentPeriod\": \"300s\",
            \"perSeriesAligner\": \"ALIGN_SUM\",
            \"crossSeriesReducer\": \"REDUCE_SUM\"
          }]
        }
      }]
    }" | python3 -c "import json,sys; r=json.load(sys.stdin); print('  policy:', r.get('name', r))"
}

echo ""
echo "Creating log-based metric for in-pipeline PIPELINE_ALERT lines..."
curl -sS -X POST \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  "${BASE}/metrics" \
  -d '{
    "name": "pipeline_alert",
    "displayName": "Pipeline alert log lines",
    "filter": "resource.type=\"cloud_run_job\" AND textPayload=~\"PIPELINE_ALERT severity=critical\"",
    "metricDescriptor": {
      "metricKind": "DELTA",
      "valueType": "INT64",
      "unit": "1"
    }
  }' 2>/dev/null | python3 -c "
import json,sys
try:
  r=json.load(sys.stdin)
  print('  metric:', r.get('name', r))
except Exception:
  print('  (metric may already exist — OK)')
" || true

echo ""
echo "Creating alert policies..."
create_policy \
  "MLB morning job failed (Cloud Run)" \
  'resource.type="cloud_run_job" AND resource.labels.job_name="mlb-morning-inference" AND metric.type="run.googleapis.com/job/completed_execution_count" AND metric.labels.result="failed"' \
  "mlb-morning-inference execution failed. Check Cloud Run job logs. Site may be partial until manual rerun."

create_policy \
  "MLB pipeline smoke / alert log" \
  'resource.type="cloud_run_job" AND resource.labels.job_name="mlb-morning-inference" AND textPayload=~"PIPELINE_ALERT severity=critical"' \
  "Morning pipeline reported critical failure (step error or empty BQ tables). Check logs for PIPELINE_ALERT."

echo ""
echo "Done. Verify the email channel, then optionally add to mlb-morning-inference:"
echo "  PIPELINE_ALERT_WEBHOOK_URL  (Pushover/Slack webhook for faster push)"
echo "  PIPELINE_ALERT_EMAIL + SENDGRID_API_KEY  (instant email from inside the job)"
