#!/usr/bin/env bash
# run_training.sh
#
# Object-store training test — data generation + training via the mlpstorage CLI.
#
# Runs a complete cycle:
#   1. Data generation  — writes NPZ files to the object store
#   2. Training         — reads the dataset across 5 epochs
#
# All runtime parameters are supplied via environment variables (or .env):
#
#   BUCKET           — S3/MinIO bucket name           (REQUIRED — no default)
#   STORAGE_LIBRARY  — storage library: s3dlio | minio  (default: s3dlio)
#   MODEL            — mlpstorage model name            (default: unet3d)
#   NP               — MPI process count for datagen    (default: 1)
#   SKIP_DATAGEN     — set to 1 to skip data generation (default: 0)
#   SKIP_TRAINING    — set to 1 to skip training run    (default: 0)
#   DATA_DIR         — object prefix for the dataset    (default: test-run/)
#
# Credentials are read from:
#   .env file at the repo root  OR  shell environment variables
#   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_ENDPOINT_URL, AWS_REGION
#
# Usage:
#   cd /path/to/mlp-storage
#
#   # Quick sanity check (1 MPI process, s3dlio)
#   BUCKET=my-test-bucket bash tests/object-store/run_training.sh
#
#   # Use minio instead
#   BUCKET=my-test-bucket STORAGE_LIBRARY=minio bash tests/object-store/run_training.sh
#
#   # 8-process parallel datagen + training
#   BUCKET=my-test-bucket NP=8 bash tests/object-store/run_training.sh
#
#   # Skip datagen (data already present)
#   BUCKET=my-test-bucket SKIP_DATAGEN=1 bash tests/object-store/run_training.sh

set -euo pipefail

# ── Locate repo root ─────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ── Credentials / environment ────────────────────────────────────────────────
if [[ -f .env ]]; then
    echo "[env] Loading from .env"
    set -o allexport
    # shellcheck disable=SC1091
    source .env
    set +o allexport
fi

: "${AWS_ACCESS_KEY_ID:?ERROR: AWS_ACCESS_KEY_ID not set — add it to .env}"
: "${AWS_SECRET_ACCESS_KEY:?ERROR: AWS_SECRET_ACCESS_KEY not set — add it to .env}"
: "${AWS_ENDPOINT_URL:?ERROR: AWS_ENDPOINT_URL not set — add it to .env}"
: "${AWS_REGION:=us-east-1}"
: "${BUCKET:?ERROR: BUCKET not set — pass it as: BUCKET=my-bucket bash $0}"

# ── Tunables ──────────────────────────────────────────────────────────────────
STORAGE_LIBRARY="${STORAGE_LIBRARY:-s3dlio}"
MODEL="${MODEL:-unet3d}"
NP="${NP:-1}"
SKIP_DATAGEN="${SKIP_DATAGEN:-0}"
SKIP_TRAINING="${SKIP_TRAINING:-0}"
DATA_DIR="${DATA_DIR:-test-run/}"

# ── Virtual environment ───────────────────────────────────────────────────────
if [[ ! -f .venv/bin/activate ]]; then
    echo "ERROR: .venv not found — run: uv sync" >&2
    exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate  # .venv managed by uv (run "uv sync" to set up)

if ! command -v mlpstorage &>/dev/null; then
    echo "ERROR: mlpstorage not found in venv. Run: uv sync" >&2
    exit 1
fi

# ── Storage params (passed to mlpstorage via --param) ────────────────────────
# All runtime storage details come from environment — nothing hardcoded here.
STORAGE_PARAMS=(
    "storage.storage_type=s3"
    "storage.storage_root=${BUCKET}"
    "storage.storage_options.storage_library=${STORAGE_LIBRARY}"
    "storage.storage_options.endpoint_url=${AWS_ENDPOINT_URL}"
    "storage.storage_options.access_key_id=${AWS_ACCESS_KEY_ID}"
    "storage.storage_options.secret_access_key=${AWS_SECRET_ACCESS_KEY}"
    "storage.s3_force_path_style=true"
)

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Object-Store Training Test"
echo "════════════════════════════════════════════════════════"
echo "  Model   : ${MODEL}"
echo "  Library : ${STORAGE_LIBRARY}"
echo "  Bucket  : ${BUCKET}"
echo "  Endpoint: ${AWS_ENDPOINT_URL}"
echo "  Data    : s3://${BUCKET}/${DATA_DIR}${MODEL}/train/"
echo "  NP      : ${NP}"
echo "════════════════════════════════════════════════════════"
echo ""

# ── Phase 1: Data generation ─────────────────────────────────────────────────
if [[ "$SKIP_DATAGEN" == "1" ]]; then
    echo "── Skipping datagen (SKIP_DATAGEN=1) ──────────────────────"
else
    echo "── Phase 1: Data generation ────────────────────────────────"
    DLIO_S3_IMPLEMENTATION=mlp mlpstorage training datagen \
        --model "${MODEL}" \
        -np "${NP}" \
        -dd "${DATA_DIR}" \
        --param "${STORAGE_PARAMS[@]}"
    echo ""
    echo "── Datagen complete ─────────────────────────────────────────"
fi
echo ""

# ── Phase 2: Training ─────────────────────────────────────────────────────────
if [[ "$SKIP_TRAINING" == "1" ]]; then
    echo "── Skipping training (SKIP_TRAINING=1) ─────────────────────"
else
    echo "── Phase 2: Training ───────────────────────────────────────"
    DLIO_S3_IMPLEMENTATION=mlp mlpstorage training run \
        --model "${MODEL}" \
        --allow-run-as-root \
        --skip-validation \
        --num-accelerators "${NP}" \
        --accelerator-type h100 \
        --client-host-memory-in-gb 512 \
        --param "${STORAGE_PARAMS[@]}" \
            "dataset.data_folder=${DATA_DIR}${MODEL}"
    echo ""
    echo "── Training complete ────────────────────────────────────────"
fi
echo ""

echo "════════════════════════════════════════════════════════"
echo "  ✅  run_training.sh complete"
echo "════════════════════════════════════════════════════════"
