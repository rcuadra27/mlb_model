#!/usr/bin/env bash
# Postgres connection safety valve until Cloud SQL managed pooling is enabled.
# Restore GET_DAILY_PREDICTIONS_MAX_INSTANCES after pooling is in production.

set -euo pipefail

PROJECT="${GCP_PROJECT:-mlb-model-491223}"
REGION="${GCP_REGION:-us-central1}"
MAX="${GET_DAILY_PREDICTIONS_MAX_INSTANCES:-6}"

echo "Setting get-daily-predictions max instances to ${MAX}..."
gcloud run services update get-daily-predictions \
  --project="${PROJECT}" \
  --region="${REGION}" \
  --max-instances="${MAX}"

gcloud run services describe get-daily-predictions \
  --project="${PROJECT}" \
  --region="${REGION}" \
  --format=json | python3 -c "
import json,sys
d=json.load(sys.stdin)
print('maxScale', d['spec']['template']['metadata']['annotations'].get('autoscaling.knative.dev/maxScale'))
"
