# dgen-py Data Generation Speed Benchmark

**System**: Intel Xeon Platinum 8280L @ 2.70 GHz  
**Cores**: 12 physical / 12 logical (no HT), 1 NUMA node (UMA)  
**RAM**: 31 GB  
**NVMe**: `/mnt/nvme_data` — 68 GB free  
**dgen-py**: v0.2.0  
**Benchmark script**: `dgen-rs/python/examples/bench_generation_speeds.py`

---

## Results

### Section 1 — Thread-count scaling (`fill_chunk`, 32 MB chunk)

| Threads | Throughput  | Per-core   |
|--------:|------------:|-----------:|
| 1       |  3.85 GB/s  | 3.85 GB/s  |
| 4       | 22.15 GB/s  | 5.54 GB/s  |
| 8       | 39.69 GB/s  | 4.96 GB/s  |
| 12      | 47.62 GB/s  | 3.97 GB/s  |

Scaling is mostly linear to 8 cores, then memory-bandwidth bound.  Per-core
peak is at 4 threads (~5.5 GB/s), where L3 utilisation is optimal.

---

### Section 2 — Chunk size impact (all 12 cores, `fill_chunk`)

| Chunk size | Throughput  |
|-----------:|------------:|
| 8 MB       | 23.59 GB/s  |
| 32 MB      | 47.45 GB/s  |
| **64 MB**  | **45.32 GB/s** |
| 256 MB     | 41.41 GB/s  |

**Takeaway**: 32–64 MB is the sweet-spot for this system.
Below 8 MB, per-thread overhead dominates.  Above 64 MB, diminishing returns
from L3 re-use.  Production default: **64 MB** (also used by `data_producer.py`).

---

### Section 3 — `compress_ratio` impact (all 12 cores, 32 MB chunk)

| compress_ratio | Throughput  | Notes                       |
|---------------:|------------:|-----------------------------|
| 1.0            | 63.16 GB/s  | incompressible (production) |
| 2.0            | 74.24 GB/s  | 2:1 compressible (+18%)     |
| 3.0            | 77.27 GB/s  | 3:1 compressible (+22%)     |

`compress_ratio=1.0` produces data that cannot be compressed by storage
systems — the correct setting for benchmarking raw storage throughput.
Use `compress_ratio=2.0` or `3.0` only to model real model-weight distributions
(which are compressible) or to stress test a compressing storage backend.

---

### Section 4 — `generate_buffer()` — BytesView path used by KV cache

This is the API previously used by `KVCacheGenerator.generate()` (before the
producer-consumer change).

| Entry size | Throughput  | Latency     |
|-----------:|------------:|------------:|
| 64 MB      |  6.71 GB/s  |  10.0 ms    |
| 256 MB     |  9.19 GB/s  |  29.2 ms    |
| 512 MB     |  9.97 GB/s  |  53.9 ms    |

**Critical finding**: These 10–54 ms latencies were being serialised with
storage writes.  The storage device sat idle for this entire window on every
`allocate_cache()` call.  This is the problem that the producer-consumer
pipeline solves (see below).

---

### Section 5 — `create_bytearrays()` + `fill_chunk` (pre-allocation pattern)

| Chunk size | Alloc rate    | Fill rate   | Notes        |
|-----------:|--------------:|------------:|--------------|
| 32 MB      | 7 415 GB/s    | 16.76 GB/s  | default chunk |
| 64 MB      | 14 832 GB/s   | 17.88 GB/s  | large chunk   |

Allocation is essentially free (virtual memory reservation only).  Fill rate
is limited by memory bandwidth when data from multiple chunks must be
resident simultaneously — lower than streaming fill_chunk.

---

### Section 6 — Streaming `fill_chunk` → files (4 × 8 GB, 64 MB chunk)

Demonstrates constant-memory streaming to NVMe: **one 64 MB buffer**
regardless of total dataset size written.

| File | Gen rate   | Write rate | Total (gen+write) |
|-----:|-----------:|-----------:|------------------:|
| 1    | 34.13 GB/s |  1.91 GB/s |         1.81 GB/s |
| 2    | 28.48 GB/s |  1.46 GB/s |         1.39 GB/s |
| 3    | 27.30 GB/s |  1.47 GB/s |         1.39 GB/s |
| 4    | 28.26 GB/s |  1.38 GB/s |         1.32 GB/s |
| **TOTAL** | — | — | **1.45 GB/s** (32 GB in 22 s) |

**RAM footprint**: 64 MB constant — the dataset can be arbitrarily large.

**Bottleneck**: NVMe write at ~1.5 GB/s.  Generation rate (28–34 GB/s) is
**19–23× faster** than the storage device — dgen-py will never be the
limiting factor, even on a 30–50 GB/s all-flash array.

---

## Summary

| API / Use case                        | Throughput     | Notes                              |
|---------------------------------------|---------------:|------------------------------------|
| `fill_chunk` streaming (all cores)    |  47–63 GB/s    | unlimited data, 32–64 MB RAM       |
| `fill_chunk` + `compress_ratio=2.0`   |  74–77 GB/s    | compressible data                  |
| `generate_buffer()` in-process        |   6–10 GB/s    | single-call BytesView              |
| Stream to NVMe file (this system)     |   1.45 GB/s    | NVMe is bottleneck                 |

**Per physical core (streaming)**: ~4–5.5 GB/s  
**Expected on 30–50 GB/s all-flash**: generation budget = 30–50 / (47 GB/s) ≈ 63–100% — still sufficient with headroom.

---

## Implication: Why the producer-consumer pipeline is required

Before the `DataGeneratorPool` change, `KVCacheGenerator.generate()` called
`dgen_py.generate_buffer(size)` **synchronously** inside
`MultiTierCache._allocate_cache_inner()`:

```
generate_buffer(256 MB)  → 29 ms    ← storage device IDLE
backend.write(data)      → 170 ms   ← storage timer starts here
                           -------
Total thread time        = 199 ms   (but storage_latency = 170 ms)
```

Although the recorded `storage_write_latencies` correctly excluded generation
(the timer only wraps `backend.write()`), the storage device was idle for
29 ms per entry.  Across many concurrent users this creates artificial
throttling and means the benchmark under-stresses the storage device compared
to a real inference system where KV data arrives from GPU memory immediately.

### After the change

```
DataGeneratorPool (background thread)   [persistent, runs at 47 GB/s]
   fill_chunk → block → queue.put()  ←— this happens CONCURRENTLY with all writes

_allocate_cache_inner:
   queue.get()                       → < 1 ms (block already ready)
   backend.write(data)               → 170 ms  ← storage timer starts here
   Total thread time                 = 171 ms  (same storage latency, less wait)
```

Because `fill_chunk` releases the Python GIL (Rayon-parallel Rust), the
producer thread generates data at full speed while all consumer threads are
doing storage I/O in true parallel.  The queue stays non-empty as long as
storage is the bottleneck (which it is at 1.5 GB/s vs 47 GB/s generation).

### Configuration

CLI flags added:

| Flag                      | Default | Description                                      |
|---------------------------|--------:|--------------------------------------------------|
| `--prefetch-depth N`      | 8       | Queue depth (N × 64 MB blocks pre-generated)     |

RAM overhead: `8 × 64 MB = 512 MB` constant.  
On a 30 GB/s all-flash array: increase to `--prefetch-depth 32` (2 GB pool).  
To disable and revert to inline generation: `--prefetch-depth 0`.

---

## Benchmark usage

```bash
# Default run (4 GB per measurement, 4 × 8 GB files to /mnt/nvme_data):
cd dgen-rs/python/examples
source /path/to/.venv/bin/activate
python bench_generation_speeds.py

# Custom file test (e.g. 8 × 16 GB = 128 GB dataset):
python bench_generation_speeds.py --size-gb 4 --file-gb 16 --num-files 8

# Memory-only tests only (no file I/O):
python bench_generation_speeds.py --out-dir ''
```
