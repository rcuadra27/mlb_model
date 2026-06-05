#!/usr/bin/env bash
# Pin Cloud Run jobs to a single production image digest (never :latest at runtime).
#
# Usage:
#   ./scripts/pin_pipeline_jobs.sh                    # pin to digest of gcr.io/.../mlb-model:latest
#   PRODUCTION_IMAGE_DIGEST=sha256:abc... ./scripts/pin_pipeline_jobs.sh
#
# After cloud build, run this so ingest-lineups cannot drift to an older digest.

set -euo pipefail

PROJECT="${GCP_PROJECT:-mlb-model-491223}"
REGION="${GCP_REGION:-us-central1}"
IMAGE_REPO="gcr.io/${PROJECT}/mlb-model"

if [[ -n "${PRODUCTION_IMAGE_DIGEST:-}" ]]; then
  DIGEST="${PRODUCTION_IMAGE_DIGEST#sha256:}"
  DIGEST="sha256:${DIGEST}"
else
  DIGEST="$(gcloud container images describe "${IMAGE_REPO}:latest" \
    --format='value(image_summary.digest)' 2>/dev/null || true)"
fi

if [[ -z "${DIGEST}" ]]; then
  echo "Could not resolve production image digest. Set PRODUCTION_IMAGE_DIGEST or build :latest first." >&2
  exit 1
fi

IMAGE="${IMAGE_REPO}@${DIGEST}"
echo "Production image: ${IMAGE}"

# Jobs that run inference and/or export daily_games (must match morning inference).
JOBS=(
  mlb-morning-inference
  mlb-ingest-lineups
  mlb-early-inference
  mlb-update-scores
)

for job in "${JOBS[@]}"; do
  echo "Updating ${job}..."
  gcloud run jobs update "${job}" \
    --image="${IMAGE}" \
    --region="${REGION}" \
    --project="${PROJECT}"
done

mkdir -p config
cat > config/production_image.env <<EOF
# Auto-written by scripts/pin_pipeline_jobs.sh — do not hand-edit digest after pin.
PRODUCTION_IMAGE=${IMAGE}
PRODUCTION_IMAGE_DIGEST=${DIGEST}
PINNED_AT=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF

echo "Wrote config/production_image.env"
echo "Done. All pipeline jobs pinned to ${DIGEST}"
