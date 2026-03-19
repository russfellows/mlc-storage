# Object Store Tests

Performance tests and benchmarks for object storage backends (s3dlio, minio,
s3torchconnector) used by `mlpstorage`.

All tests load credentials from a `.env` file at the **project root** (`mlp-storage/.env`):

```
AWS_ACCESS_KEY_ID=<key>
AWS_SECRET_ACCESS_KEY=<secret>
AWS_ENDPOINT_URL=http://<host>:<port>
AWS_REGION=us-east-1
```

Environment variables already set in the shell take precedence over the `.env` file.
No credentials are hard-coded in any test.

---

## Results

**[Object_Perf_Results.md](Object_Perf_Results.md)** — Full benchmark results including:
- Direct native-API write + read throughput (all three libraries, 12 parallel workers)
- DLIO streaming checkpoint write + read throughput (16 GB and 100 GB)
- DLIO training MPI sweep (N=1, 2, 4 processes × all three libraries)
- Analysis of DLIO overhead vs native API performance

---

## Test Files

### Cross-Library Comparisons

#### `test_direct_write_comparison.py`
Measures **native API write + read throughput** across all three libraries side-by-side,
without any DLIO involvement. Each library gets its own dedicated bucket.

```bash
cd mlp-storage && source .venv/bin/activate

# Default: 100 × 128 MiB objects, 8 write + 8 read workers, all three libraries
python tests/object-store/test_direct_write_comparison.py

# Reproduce the 12-worker results in Object_Perf_Results.md
python tests/object-store/test_direct_write_comparison.py \
    --num-files 100 --size-mb 128 --write-workers 12 --read-workers 12

# Single library
python tests/object-store/test_direct_write_comparison.py --library s3dlio

# CLI reference
python tests/object-store/test_direct_write_comparison.py --help
```

#### `test_dlio_multilib_demo.py`
Runs **DLIO-driven training and checkpoint workloads** across all three libraries.
I/O goes through DLIO's MPI data generation and PyTorch DataLoader — this is the
realistic DLIO performance as seen by a training job, not direct API throughput.

```bash
cd mlp-storage && source .venv/bin/activate

# Training workload (100 × 128 MiB NPZ, 2 epochs)
python tests/object-store/test_dlio_multilib_demo.py --workload training

# Checkpoint workload (~105 GB streaming checkpoint, llama3-8b profile)
python tests/object-store/test_dlio_multilib_demo.py --workload checkpoint

# Single library
python tests/object-store/test_dlio_multilib_demo.py --workload training --library s3dlio
```

#### `test_training_mpi_sweep.py`
Sweeps MPI **process count (N = 1, 2, 4)** for both datagen and training across all
three libraries. Each (library, N) combination runs as an independent clean cycle:
`clean → datagen(N) → train(N) → clean`. Both write (datagen) and read (training)
throughput are measured at each N.

```bash
cd mlp-storage && source .venv/bin/activate

# Full sweep: all libraries, N = 1, 2, 4
python tests/object-store/test_training_mpi_sweep.py

# Custom process counts
python tests/object-store/test_training_mpi_sweep.py --process-counts 1 2 4 8

# Single library
python tests/object-store/test_training_mpi_sweep.py --library s3dlio

# Skip datagen (use data already in bucket)
python tests/object-store/test_training_mpi_sweep.py --skip-datagen

# Keep objects after the run (skip cleanup)
python tests/object-store/test_training_mpi_sweep.py --skip-cleanup
```

---

### Per-Library Checkpoint Tests

Each of these tests the `StreamingCheckpointing` pipeline for a single library:
a fixed-RAM streaming producer-consumer pipeline where dgen-py generates data
concurrently while the library uploads it. Memory usage is constant at ~128 MB
regardless of checkpoint size.

#### `test_s3dlio_checkpoint.py`
StreamingCheckpointing with the **s3dlio** backend.

```bash
cd mlp-storage && source .venv/bin/activate
python tests/object-store/test_s3dlio_checkpoint.py --size-gb 16
python tests/object-store/test_s3dlio_checkpoint.py --size-gb 100
python tests/object-store/test_s3dlio_checkpoint.py --help
```

#### `test_s3torch_checkpoint.py`
StreamingCheckpointing with the **s3torchconnector** backend.

```bash
cd mlp-storage && source .venv/bin/activate
python tests/object-store/test_s3torch_checkpoint.py --size-gb 16
python tests/object-store/test_s3torch_checkpoint.py --help
```

#### `test_minio_checkpoint.py`
StreamingCheckpointing with the **minio** backend.

```bash
cd mlp-storage && source .venv/bin/activate
python tests/object-store/test_minio_checkpoint.py --size-gb 16
python tests/object-store/test_minio_checkpoint.py --help
```

---

### Direct s3dlio API Tests

#### `test_s3dlio_direct.py`
Tests the two s3dlio write APIs directly (no DLIO, no mlpstorage wrapper):
- `PyObjectWriter` — streaming writer (`write_chunk` + `finalize`)
- `MultipartUploadWriter` — multipart upload (`write` + `close`)

```bash
cd mlp-storage && source .venv/bin/activate
python tests/object-store/test_s3dlio_direct.py --help
```

---

### Shell Script Tests

These shell scripts exercise the full `mlpstorage` CLI for each library — datagen,
training, and checkpoint — and are useful for quick end-to-end smoke tests.

#### `test_mlp_s3dlio.sh`
Full mlpstorage smoke test with **s3dlio** as the storage backend.

```bash
cd mlp-storage
bash tests/object-store/test_mlp_s3dlio.sh
```

#### `test_mlp_minio.sh`
Full mlpstorage smoke test with **minio** as the storage backend.

```bash
cd mlp-storage
bash tests/object-store/test_mlp_minio.sh
```

#### `test_mlp_s3torch.sh`
Full mlpstorage smoke test with **s3torchconnector** as the storage backend.

```bash
cd mlp-storage
bash tests/object-store/test_mlp_s3torch.sh
```

#### `test_s3dlio_multilib.sh`
Shell-based multi-library comparison using s3dlio directly (not via mlpstorage).

```bash
cd mlp-storage
bash tests/object-store/test_s3dlio_multilib.sh
```

#### `demo_streaming_checkpoint.sh`
Quickstart demo showing the two major optimisations: dgen-py integration (155×
faster data generation) and StreamingCheckpointing (192× memory reduction).
Compares old vs new method for both file and object storage.

```bash
TEST_SIZE_GB=1 TEST_CHECKPOINT_DIR=/tmp/ckpt-demo \
    bash tests/object-store/demo_streaming_checkpoint.sh
```

---

## Credential Setup

Create `mlp-storage/.env` (never commit this file):

```bash
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_ENDPOINT_URL=http://your-minio-host:9000
AWS_REGION=us-east-1
```

`.env` is already listed in `.gitignore`. All scripts and Python tests read it
automatically at startup; shell environment variables always take precedence.
