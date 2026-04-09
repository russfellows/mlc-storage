# Object-Store Tests

Tests for S3-compatible object storage backends used by `mlpstorage` and `dlio_benchmark`.

All tests read credentials and runtime configuration from a `.env` file at the
**project root** (`mlp-storage/.env`) — no credentials or site-specific values are
embedded in any test script or config file.

---

## Prerequisites

### 1 — Install dependencies

```bash
cd /path/to/mlp-storage
uv sync
```

### 2 — Create `.env`

Copy the example and fill in your values:

```bash
cp .env.example .env
# edit .env — never commit this file
```

`.env` must contain (at minimum):

```bash
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_ENDPOINT_URL=https://your-s3-host:9000   # or http:// for plain HTTP
AWS_REGION=us-east-1
BUCKET=your-test-bucket                       # used by run_training.sh
STORAGE_LIBRARY=s3dlio                        # s3dlio | minio (default: s3dlio)
```

For HTTPS endpoints with a self-signed certificate, also set:

```bash
AWS_CA_BUNDLE=/path/to/your-cert.crt
```

Shell environment variables already set take precedence over the `.env` file.

### 3 — Ensure the bucket exists

Create your bucket in MinIO (or your S3-compatible store) before running tests:

```bash
# Verify bucket is reachable
uv run python -c "import s3dlio; print(s3dlio.list('s3://your-bucket/', recursive=False))"
```

---

## Tests

There are four tests. All runtime parameters come from `.env` (or environment
variables / CLI flags) — no editing of scripts or config files is needed.

### `run_training.sh` — Data generation + training

Runs a full MLPerf Storage training cycle:

1. **Datagen** — generates synthetic training data and writes it to the object store
2. **Training** — reads the dataset via the mlpstorage CLI

```bash
cd /path/to/mlp-storage

# Default: unet3d model, s3dlio library, 1 MPI process
BUCKET=my-test-bucket bash tests/object-store/run_training.sh

# Use minio instead
BUCKET=my-test-bucket STORAGE_LIBRARY=minio bash tests/object-store/run_training.sh

# 8 parallel MPI processes for datagen + training
BUCKET=my-test-bucket NP=8 bash tests/object-store/run_training.sh

# Skip datagen (data already in bucket)
BUCKET=my-test-bucket SKIP_DATAGEN=1 bash tests/object-store/run_training.sh

# Different model
BUCKET=my-test-bucket MODEL=bert bash tests/object-store/run_training.sh
```

**Runtime parameters** (all optional except BUCKET):

| Variable | Default | Description |
|---|---|---|
| `BUCKET` | *(required)* | S3 bucket for training data |
| `STORAGE_LIBRARY` | `s3dlio` | `s3dlio` or `minio` |
| `MODEL` | `unet3d` | mlpstorage model name |
| `NP` | `1` | MPI process count |
| `SKIP_DATAGEN` | `0` | Set to `1` to skip data generation |
| `SKIP_TRAINING` | `0` | Set to `1` to skip training run |
| `DATA_DIR` | `test-run/` | Object prefix for the dataset |

---

### `run_checkpointing.sh` — Checkpoint write + read

Runs a LLaMA 3 8B checkpoint cycle via `dlio_benchmark`:

1. **Write** — saves `CHECKPOINTS` checkpoint(s) to the object store
2. **Read** — restores each checkpoint back

Uses the `llama3_8b_checkpoint` workload config. All storage runtime parameters
are injected as Hydra overrides — the YAML file contains only model/workload sizing.

```bash
cd /path/to/mlp-storage

# Quick sanity check (1 MPI rank = ~13.1 GB I/O)
BUCKET=my-test-bucket bash tests/object-store/run_checkpointing.sh

# Full llama3-8b run (8 MPI ranks = ~105 GB I/O)
BUCKET=my-test-bucket NP=8 bash tests/object-store/run_checkpointing.sh

# Use minio, 4 ranks, 1 checkpoint only
BUCKET=my-test-bucket STORAGE_LIBRARY=minio NP=4 CHECKPOINTS=1 \
    bash tests/object-store/run_checkpointing.sh
```

**Runtime parameters** (all optional except BUCKET):

| Variable | Default | Description |
|---|---|---|
| `BUCKET` | *(required)* | S3 bucket for checkpoints |
| `STORAGE_LIBRARY` | `s3dlio` | `s3dlio` or `minio` |
| `NP` | `1` | MPI rank count (use `8` for full llama3-8b) |
| `CHECKPOINTS` | `2` | Number of write + read cycles |
| `MODEL` | `llama3_8b_checkpoint` | DLIO workload config name |

> **Note on s3torchconnector and NP=1:** At NP=1 the full ~105 GB checkpoint is a single
> object, which exceeds the AWS CRT library's ~78 GB object limit. Use `NP>=2` with
> s3torchconnector. s3dlio and minio are not affected.

---

### `test_s3lib_get_bench.py` — GET throughput benchmark

Benchmarks raw S3 GET throughput across s3dlio, minio, and s3torchconnector.
All three libraries read from the **same bucket and same objects** for a fair comparison.

```bash
cd /path/to/mlp-storage

# Benchmark existing training objects (bucket from BUCKET env var)
uv run python tests/object-store/test_s3lib_get_bench.py

# Write 20 x 128 MB test objects first, then benchmark
uv run python tests/object-store/test_s3lib_get_bench.py \
    --write --write-num-files 20 --write-size-mb 128

# Serial mode only (per-request latency: p50/p95/p99/max)
uv run python tests/object-store/test_s3lib_get_bench.py --mode serial

# Parallel sweep at custom worker counts
uv run python tests/object-store/test_s3lib_get_bench.py \
    --mode parallel --workers 1 4 8 16 32

# Override bucket and prefix
uv run python tests/object-store/test_s3lib_get_bench.py \
    --bucket my-bucket --prefix data/train/

# Test only s3dlio and minio
uv run python tests/object-store/test_s3lib_get_bench.py --libraries s3dlio minio

uv run python tests/object-store/test_s3lib_get_bench.py --help
```

The `BUCKET` environment variable sets the default bucket; `--bucket` overrides it.

**Test modes:**

| Mode | What it measures |
|---|---|
| `serial` | Per-request latency (p50/p95/p99/max) + single-stream MB/s |
| `parallel` | Aggregate MB/s using `ThreadPoolExecutor` at matched concurrency |
| `native` | s3dlio `get_many()` Rust Tokio async vs Python threads |
| `all` | All three modes (default) |

---

### `test_direct_write_comparison.py` — Native write + read benchmark

Benchmarks raw write and read throughput via each library's native API (no DLIO
overhead). Each library can use its own dedicated bucket, or all can share one.

```bash
cd /path/to/mlp-storage

# Default: all libraries, 100 x 128 MB objects, 8 write + 8 read workers
# Uses BUCKET env var for all libraries (or set BUCKET_S3DLIO etc. individually)
uv run python tests/object-store/test_direct_write_comparison.py

# Per-library buckets
BUCKET_S3DLIO=bucket-a BUCKET_MINIO=bucket-b \
    uv run python tests/object-store/test_direct_write_comparison.py

# 12 workers
uv run python tests/object-store/test_direct_write_comparison.py \
    --num-files 100 --size-mb 128 --write-workers 12 --read-workers 12

# Single library
uv run python tests/object-store/test_direct_write_comparison.py --library s3dlio

uv run python tests/object-store/test_direct_write_comparison.py --help
```

Bucket precedence (highest wins):

1. `--bucket-s3dlio` / `--bucket-minio` / `--bucket-s3torch` CLI flag
2. `BUCKET_S3DLIO` / `BUCKET_MINIO` / `BUCKET_S3TORCH` env var
3. `BUCKET` env var (shared default for all libraries)

---

## Credential Setup

Create `mlp-storage/.env` (never commit — it is already in `.gitignore`):

```bash
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_ENDPOINT_URL=https://your-minio-host:9000
AWS_REGION=us-east-1
BUCKET=your-test-bucket
STORAGE_LIBRARY=s3dlio
```

See `.env.example` at the repo root for a fully annotated template.

---

## TLS / HTTPS Setup

If your endpoint uses a self-signed certificate:

1. Generate the cert with `basicConstraints=CA:FALSE`  
   (Rust-based libraries use **rustls** and enforce RFC 5280 — CA:TRUE is rejected)
2. The cert must include a `subjectAltName` (SAN) matching the server IP or hostname
3. Run `sudo update-ca-certificates` (s3torchconnector uses the system store)
4. Set `AWS_CA_BUNDLE=/path/to/cert.crt` in `.env` (used by s3dlio)

Verify TLS is working:

```bash
# Should return HTTP 403 (AccessDenied) — means TLS handshake succeeded
curl -v https://your-minio-host:9000/
```

---

## Adding More Libraries

Runtime parameters — library, bucket, endpoint, credentials — all flow from
environment variables. To test a new storage library:

1. Add it to `mlpstorage_py/storage/` and register it in `obj_store_lib.py`
2. Set `STORAGE_LIBRARY=<new-library>` in `.env`
3. Run `run_training.sh` or `run_checkpointing.sh` without changing any test script

---

## Archived Tests

Older per-library scripts (dlio\_s3dlio\_\*.sh, dlio\_minio\_\*.sh, etc.),
per-library Python tests, and historical result documents are preserved in
`tests/object-store/old-archive/` for reference. They are **not maintained**.
