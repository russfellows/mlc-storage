#!/usr/bin/env bash
# run_checkpointing.sh
#
# Object-store checkpoint test — write + read checkpoints via dlio_benchmark.
#
# Uses the llama3_8b_checkpoint.yaml workload config with all runtime storage
# parameters injected as Hydra overrides at run time — no credentials or
# site-specific values are embedded in config files.
#
# All runtime parameters are supplied via environment variables (or .env):
#
#   BUCKET           — S3/MinIO bucket name           (REQUIRED — no default)
#   STORAGE_LIBRARY  — storage library: s3dlio | minio  (default: s3dlio)
#   NP               — MPI rank count (each rank = 1 GPU shard of llama3-8b)
#                      NP=1: single-rank sanity check (~13.1 GB I/O)
#                      NP=8: full llama3-8b ZeRO-3 (~105 GB I/O)  (default: 1)
#   CHECKPOINTS      — number of checkpoint write + read cycles  (default: 2)
#   MODEL            — DLIO workload name  (default: llama3_8b_checkpoint)
#
# Credentials are read from:
#   .env file at the repo root  OR  shell environment variables
#   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_ENDPOINT_URL, AWS_REGION
#
# Note on NP and s3torchconnector:
#   At NP=1 the entire ~105 GB checkpoint is written as ONE object. The AWS CRT
#   library used by s3torchconnector has a ~78 GB single-object limit, so NP=1
#   WILL FAIL with s3torchconnector.  Use NP≥2 for that library.
#
# Usage:
#   cd /path/to/mlp-storage
#
#   # Quick sanity check (NP=1 rank, s3dlio, 2 checkpoints)
#   BUCKET=my-test-bucket bash tests/object-store/run_checkpointing.sh
#
#   # Full llama3-8b run (8 MPI ranks)
#   BUCKET=my-test-bucket NP=8 bash tests/object-store/run_checkpointing.sh
#
#   # Use minio, 4 ranks, 1 checkpoint
#   BUCKET=my-test-bucket STORAGE_LIBRARY=minio NP=4 CHECKPOINTS=1 \
#       bash tests/object-store/run_checkpointing.sh

# Performance tuning (override as needed via env):
export S3DLIO_ENABLE_RANGE_OPTIMIZATION="${S3DLIO_ENABLE_RANGE_OPTIMIZATION:-0}"
export S3DLIO_RT_THREADS="${S3DLIO_RT_THREADS:-8}"

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
NP="${NP:-1}"
CHECKPOINTS="${CHECKPOINTS:-2}"
MODEL="${MODEL:-llama3_8b_checkpoint}"

# Object prefix under the bucket — library name keeps runs from different
# libraries separated so they can run against the same bucket.
S3_PREFIX="${STORAGE_LIBRARY}/llama3-8b"
CHECKPOINT_FOLDER="s3://${BUCKET}/${S3_PREFIX}"

# ── Virtual environment ───────────────────────────────────────────────────────
if [[ ! -f .venv/bin/activate ]]; then
    echo "ERROR: .venv not found — run: uv sync" >&2
    exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate  # .venv managed by uv (run "uv sync" to set up)

DLIO_BIN=".venv/bin/dlio_benchmark"
if [[ ! -x "$DLIO_BIN" ]]; then
    echo "ERROR: $DLIO_BIN not found. Run: uv sync" >&2
    exit 1
fi

# ── Pre-flight: verify bucket reachability ────────────────────────────────────
echo ""
echo "Checking bucket: s3://${BUCKET}/ ..."
python3 - <<PYEOF
import os, sys
try:
    import s3dlio
    files = s3dlio.list("s3://${BUCKET}/", recursive=False)
    print(f"  Bucket accessible — {len(files)} top-level entries")
except ImportError:
    print("  s3dlio not available — skipping bucket pre-check")
except Exception as e:
    print(f"  ERROR: Could not access s3://${BUCKET}/: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF

RUN_DIR="/tmp/dlio-checkpoint-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Object-Store Checkpoint Test"
echo "════════════════════════════════════════════════════════"
echo "  Model    : ${MODEL}"
echo "  Library  : ${STORAGE_LIBRARY}"
echo "  Bucket   : ${BUCKET}"
echo "  Objects  : ${CHECKPOINT_FOLDER}/"
echo "  Endpoint : ${AWS_ENDPOINT_URL}"
echo "  MPI ranks: ${NP}  (full llama3-8b: NP=8)"
echo "  Checkpts : ${CHECKPOINTS} write + ${CHECKPOINTS} read"
echo "  Run dir  : ${RUN_DIR}"
echo "════════════════════════════════════════════════════════"
echo ""

DLIO_S3_IMPLEMENTATION=mlp \
mpirun -np "${NP}" --allow-run-as-root \
    --mca btl ^vader \
    "${DLIO_BIN}" \
    "workload=${MODEL}" \
    "++hydra.run.dir=${RUN_DIR}" \
    ++hydra.output_subdir=null \
    "++workload.storage.storage_root=${BUCKET}" \
    "++workload.storage.storage_library=${STORAGE_LIBRARY}" \
    "++workload.storage.storage_options.endpoint_url=${AWS_ENDPOINT_URL}" \
    "++workload.checkpoint.checkpoint_folder=${CHECKPOINT_FOLDER}" \
    "++workload.checkpoint.num_checkpoints_write=${CHECKPOINTS}" \
    "++workload.checkpoint.num_checkpoints_read=${CHECKPOINTS}" \
    --config-dir="${REPO_ROOT}/configs/dlio"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  ✅  run_checkpointing.sh complete — results in ${RUN_DIR}"
echo "════════════════════════════════════════════════════════"
