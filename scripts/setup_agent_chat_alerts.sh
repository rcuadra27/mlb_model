#!/usr/bin/env bash
# Cloud Monitoring: log-based metrics + daily spend alert for mlb-agent-chat.
#
# Logs: JSON lines with event=AGENT_CHAT_USAGE and estimated_cost_usd (agent_chat/main.py).
# Alert: request count in 24h > budget/0.01 (~$10/day at ~$0.01/msg on Haiku).
#
# Usage:
#   PIPELINE_ALERT_EMAIL=you@example.com ./scripts/setup_agent_chat_alerts.sh
#   AGENT_CHAT_DAILY_BUDGET_USD=10 ./scripts/setup_agent_chat_alerts.sh
#
# Deferred (not launch-critical): global chat rate limit (Redis), MLB client polling cleanup.

set -euo pipefail

PROJECT="${GCP_PROJECT:-mlb-model-491223}"
EMAIL="${PIPELINE_ALERT_EMAIL:-contact@the-hot-corner.com}"
BUDGET_USD="${AGENT_CHAT_DAILY_BUDGET_USD:-10}"
SERVICE="mlb-agent-chat"
COST_METRIC="agent_chat_estimated_cost_usd"
REQ_METRIC="agent_chat_requests"
# Typical Haiku chat ~$0.01; threshold is a spend proxy until MQL sum on DISTRIBUTION is wired.
COST_PER_REQUEST="${AGENT_CHAT_COST_PER_REQUEST_USD:-0.01}"
REQ_BUDGET="$(python3 -c "import math; print(max(50, int(math.ceil(float('${BUDGET_USD}') / float('${COST_PER_REQUEST}')))))")"

TOKEN="$(gcloud auth print-access-token)"
BASE="https://monitoring.googleapis.com/v3/projects/${PROJECT}"
LOGGING_BASE="https://logging.googleapis.com/v2/projects/${PROJECT}/metrics"
FILTER_LOG='resource.type="cloud_run_revision" AND resource.labels.service_name="'"${SERVICE}"'" AND jsonPayload.event="AGENT_CHAT_USAGE"'

echo "Creating email notification channel for ${EMAIL}..."
CHANNEL_JSON="$(curl -sS -X POST \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  "${BASE}/notificationChannels" \
  -d "{
    \"type\": \"email\",
    \"displayName\": \"MLB agent chat spend alerts\",
    \"labels\": {\"email_address\": \"${EMAIL}\"},
    \"enabled\": true
  }")"

CHANNEL_NAME="$(echo "$CHANNEL_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin).get('name',''))" 2>/dev/null || true)"
if [[ -z "$CHANNEL_NAME" ]]; then
  CHANNEL_NAME="$(gcloud beta monitoring channels list --project="${PROJECT}" \
    --filter="labels.email_address=${EMAIL}" --format='value(name)' 2>/dev/null | head -1 || true)"
fi
if [[ -z "$CHANNEL_NAME" ]]; then
  echo "Could not create or find notification channel. Response: $CHANNEL_JSON" >&2
  exit 1
fi
echo "Channel: $CHANNEL_NAME"
echo "(Verify the Monitoring email channel in your inbox if this is new.)"

upsert_metric() {
  local name="$1"
  local body="$2"
  if gcloud logging metrics describe "${name}" --project="${PROJECT}" &>/dev/null; then
    curl -sS -X PATCH -H "Authorization: Bearer ${TOKEN}" -H "Content-Type: application/json" \
      "${LOGGING_BASE}/${name}" -d "${body}" >/dev/null
    echo "  updated ${name}"
  else
    curl -sS -X POST -H "Authorization: Bearer ${TOKEN}" -H "Content-Type: application/json" \
      "${LOGGING_BASE}" -d "${body}" | python3 -c "import json,sys; r=json.load(sys.stdin); print('  created', r.get('name', r.get('error', r)))"
  fi
}

echo ""
echo "Creating log-based metrics..."
upsert_metric "${COST_METRIC}" "$(FILTER_LOG="${FILTER_LOG}" COST_METRIC="${COST_METRIC}" python3 - <<'PY'
import json, os
print(json.dumps({
    "name": os.environ["COST_METRIC"],
    "description": "Estimated Anthropic USD per chat (jsonPayload.estimated_cost_usd)",
    "filter": os.environ["FILTER_LOG"],
    "valueExtractor": "EXTRACT(jsonPayload.estimated_cost_usd)",
    "metricDescriptor": {"metricKind": "DELTA", "valueType": "DISTRIBUTION", "unit": "USD"},
    "bucketOptions": {"linearBuckets": {"numFiniteBuckets": 100, "width": 0.01, "offset": 0}},
}))
PY
)"

upsert_metric "${REQ_METRIC}" "$(FILTER_LOG="${FILTER_LOG}" REQ_METRIC="${REQ_METRIC}" python3 - <<'PY'
import json, os
print(json.dumps({
    "name": os.environ["REQ_METRIC"],
    "description": "Chat requests (AGENT_CHAT_USAGE); used for daily spend proxy alerts",
    "filter": os.environ["FILTER_LOG"],
    "metricDescriptor": {"metricKind": "DELTA", "valueType": "INT64", "unit": "1"},
}))
PY
)"

DOC="mlb-agent-chat exceeded ~\$${BUDGET_USD}/day proxy: ${REQ_BUDGET}+ requests in 24h (~\$${COST_PER_REQUEST}/msg). Logs include estimated_cost_usd per request. Check Anthropic console."

echo ""
echo "Creating alert policy (24h requests > ${REQ_BUDGET}, ~\$${BUDGET_USD} spend)..."
POLICY_BODY="$(CHANNEL_NAME="${CHANNEL_NAME}" REQ_BUDGET="${REQ_BUDGET}" REQ_METRIC="${REQ_METRIC}" DOC="${DOC}" python3 - <<'PY'
import json, os
channel = os.environ["CHANNEL_NAME"]
req_budget = int(os.environ["REQ_BUDGET"])
req_metric = os.environ["REQ_METRIC"]
doc = os.environ["DOC"]
filt = f'resource.type="cloud_run_revision" AND metric.type="logging.googleapis.com/user/{req_metric}"'
print(json.dumps({
    "displayName": "MLB agent chat daily Anthropic spend",
    "documentation": {"content": doc, "mimeType": "text/markdown"},
    "combiner": "OR",
    "enabled": True,
    "notificationChannels": [channel],
    "alertStrategy": {"autoClose": "3600s"},
    "conditions": [{
        "displayName": f"Agent chat requests 24h > {req_budget}",
        "conditionThreshold": {
            "filter": filt,
            "comparison": "COMPARISON_GT",
            "thresholdValue": req_budget,
            "duration": "0s",
            "trigger": {"count": 1},
            "aggregations": [{
                "alignmentPeriod": "86400s",
                "perSeriesAligner": "ALIGN_SUM",
                "crossSeriesReducer": "REDUCE_SUM",
            }],
        },
    }],
}))
PY
)"

create_policy() {
  curl -sS -X POST -H "Authorization: Bearer ${TOKEN}" -H "Content-Type: application/json" \
    "${BASE}/alertPolicies" -d "$POLICY_BODY"
}

POLICY_JSON="$(create_policy)"
if echo "$POLICY_JSON" | python3 -c "import json,sys; r=json.load(sys.stdin); sys.exit(0 if r.get('name') else 1)" 2>/dev/null; then
  echo "$POLICY_JSON" | python3 -c "import json,sys; r=json.load(sys.stdin); print('  policy:', r.get('name'))"
else
  echo "  metric may still be propagating; retrying in 45s..."
  sleep 45
  POLICY_JSON="$(create_policy)"
  echo "$POLICY_JSON" | python3 -c "import json,sys; r=json.load(sys.stdin); print('  policy:', r.get('name', r))"
fi

echo ""
echo "Done. Alert: ${REQ_BUDGET} requests / 24h (~\$${BUDGET_USD}) → ${EMAIL}"
echo "Metrics: ${COST_METRIC} (USD distribution), ${REQ_METRIC} (counter)."
echo "Tune: AGENT_CHAT_DAILY_BUDGET_USD, AGENT_CHAT_COST_PER_REQUEST_USD"
