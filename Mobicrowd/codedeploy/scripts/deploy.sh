#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Config (edit if needed)
# -----------------------------
REGION="us-east-1"
ACCOUNT_ID="405894846876"
REPO="takleef"
CONTAINER_NAME="takleef-back"
PORT_MAP="8000:8000"

# If you need a fixed docker network, set it here; otherwise leave empty.
DOCKER_NETWORK=""

# If you want to fetch a secret on the VPS (optional)
FIREBASE_SECRET_ID="FirebaseServiceAccountB64"   # set "" to disable
FIREBASE_ENV_NAME="FIREBASE_B64"                 # env var name passed to container

# -----------------------------
# Resolve paths (CodeDeploy runs scripts from a deployment dir)
# -----------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"  # one level above scripts/
TAG_FILE="${DEPLOY_ROOT}/image_tag.txt"

if [[ ! -f "${TAG_FILE}" ]]; then
  echo "ERROR: image_tag.txt not found at ${TAG_FILE}"
  exit 1
fi

IMAGE_TAG="$(cat "${TAG_FILE}" | tr -d '[:space:]')"
if [[ -z "${IMAGE_TAG}" ]]; then
  echo "ERROR: IMAGE_TAG is empty (image_tag.txt)"
  exit 1
fi

ECR="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE="${ECR}/${REPO}:${IMAGE_TAG}"

echo "==> Deploying ${IMAGE}"

# -----------------------------
# Docker must be available
# -----------------------------
if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker not found. Install Docker on the VPS."
  exit 1
fi

# -----------------------------
# Login to ECR (uses AWS CLI creds on the VPS or instance config)
# -----------------------------
echo "==> Logging in to ECR"
aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${ECR}"

# -----------------------------
# Pull new image
# -----------------------------
echo "==> Pulling image ${IMAGE}"
docker pull "${IMAGE}"

# -----------------------------
# Stop/remove old container
# -----------------------------
echo "==> Stopping old container (if exists)"
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

# -----------------------------
# Optional cleanup: remove older tagged images for this repo
# Keeps the newly deployed tag; removes other takleef_back_V* tags
# -----------------------------
echo "==> Cleaning old images (optional)"
docker images "${ECR}/${REPO}" --format "{{.Repository}}:{{.Tag}} {{.ID}}" | \
  awk -v new="${IMAGE_TAG}" '$1 ~ /:takleef_back_V/ && $1 !~ (":" new "$") {print $2}' | \
  xargs -r docker rmi -f || true

# -----------------------------
# Optional: fetch secret and pass to container
# -----------------------------
EXTRA_ENV_ARGS=()
if [[ -n "${FIREBASE_SECRET_ID}" ]]; then
  echo "==> Fetching secret ${FIREBASE_SECRET_ID} from Secrets Manager"
  FIREBASE_B64="$(aws secretsmanager get-secret-value \
    --secret-id "${FIREBASE_SECRET_ID}" \
    --region "${REGION}" \
    --query SecretString --output text)"
  EXTRA_ENV_ARGS+=("-e" "${FIREBASE_ENV_NAME}=${FIREBASE_B64}")
fi

# -----------------------------
# Run container
# -----------------------------
echo "==> Starting new container ${CONTAINER_NAME}"
RUN_ARGS=( -d --restart unless-stopped --name "${CONTAINER_NAME}" -p "${PORT_MAP}" )

if [[ -n "${DOCKER_NETWORK}" ]]; then
  RUN_ARGS+=( --network "${DOCKER_NETWORK}" )
fi

docker run "${RUN_ARGS[@]}" "${EXTRA_ENV_ARGS[@]}" "${IMAGE}"

# -----------------------------
# Basic health check (optional)
# -----------------------------
echo "==> Health check (best-effort)"
sleep 2
docker ps --filter "name=${CONTAINER_NAME}" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo "✅ Deploy finished: ${IMAGE}"
