# Fill-Rate Comparison: numpy vs dgen-py

**Script**: `tests/bench_fill_comparison.py`  
**Context**: Follow-on to the zero-copy data generation PR. This benchmark
isolates the single variable that matters for producer throughput: the
in-place buffer fill function.

---

## What this benchmark measures

The old single-buffer-reuse design (`LegacyKVCacheGenerator`) is intentionally
excluded — that approach produces 100% deduplicatable data and is not valid for
storage benchmarks (see `docs/datagen_dedup_analysis.md`).

Both implementations under test use the **same producer-consumer pool
architecture**: identical queue sizes, identical pre-allocated 256 MB
`bytearray` buffers, identical zero-copy `get_view()` consumer path.
The only variable is the function that fills each buffer:

| Backend | Fill call | GIL | Extra allocation |
|---------|-----------|-----|-----------------|
| numpy | `rng.integers(0,256,...) → arr[:]` | **held** | 1× 256 MB temp array per fill |
| dgen-py | `gen.fill_chunk(buf)` | **released** | none — writes directly in-place |

---

## How to run

```bash
cd kv_cache_benchmark
pip install dgen-py   # if not already installed

# Default run (cpu_count//2 producers, 20s per section)
python tests/bench_fill_comparison.py

# Same settings used for the results below
python tests/bench_fill_comparison.py --duration 30 --producers 4 --prefetch 4

# With deduplication check (hashes 16 × 256 MB blocks)
python tests/bench_fill_comparison.py --duration 30 --producers 4 --check-dedup

# Single backend only
python tests/bench_fill_comparison.py --skip-numpy
python tests/bench_fill_comparison.py --skip-dgen

# Multi-consumer (simulates concurrent storage writers)
python tests/bench_fill_comparison.py --consumer-threads 4
```

### Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--duration` | 20 | Seconds per section |
| `--producers` | cpu_count//2 | Fill threads per pool |
| `--buffer-mb` | 256 | Buffer size in MB |
| `--prefetch` | 8 | Ready-queue depth |
| `--entry-mb` | 16 | `get_view()` call size (Section 3) |
| `--consumer-threads` | 1 | Concurrent consumers (Section 3) |
| `--check-dedup` | off | Hash blocks, report collision rate |

---

## Results

**System**: Intel Xeon Platinum 8280L @ 2.70 GHz, 12 logical cores, 31 GB RAM  
**Config**: `--duration 30 --producers 4 --prefetch 4`

### Section 1 — Single fill (1 thread, 10 iterations)

One producer thread, one 256 MB buffer, no concurrency.
The irreducible cost of the fill function itself.

| Backend | Time / fill | Throughput |
|---------|-------------|------------|
| numpy   | 397 ms      | 0.63 GB/s  |
| dgen-py | 6.6 ms      | **37.80 GB/s** |
| **speedup** | | **60×** |

The 397 ms numpy cost comes from two operations:
1. `rng.integers()` — allocates a new 256 MB `uint8` array (GIL held)
2. `arr[:] = data` — copies it into the pre-allocated `bytearray` (GIL held)

dgen-py's `fill_chunk()` does neither: it releases the GIL and writes directly
into the `bytearray` via Rayon-parallel Xoshiro256++. No temporary allocation.

### Section 2 — Pure fill throughput (N threads, no queues, no consumer)

Each of N threads owns one 256 MB buffer and fills it in a tight loop for
the full duration. No queues, no blocking — this is the maximum achievable
fill rate for each backend at N threads.

| Backend | Fills | Throughput | Notes |
|---------|-------|------------|-------|
| numpy   | 52 / 30s | 2.63 GB/s | GIL serializes threads; ~4× single-thread because they take turns |
| dgen-py | 753 / 30s | **39.66 GB/s** | GIL released; Rayon uses all 12 cores per fill |
| **speedup** | | **15×** | |

numpy's GIL means N threads don't give N× throughput. With 4 threads you get
~4× the single-thread rate (0.63 → 2.63 GB/s) because they serialize on the
GIL — adding a 5th or 6th numpy producer adds nearly nothing beyond this.

dgen-py's `fill_chunk()` is already saturating all 12 cores on the first call.
Additional Python producer threads add near-zero benefit, but also add no harm
since the GIL is released.

### Section 3 — End-to-end consumer throughput (pool + get_view)

Consumer calls `get_view(16 MB)` in a tight loop for 30 seconds.
This measures the full pipeline: fill latency + queue transfer + pointer slice.

| Backend | Calls | Total data | Throughput |
|---------|-------|------------|------------|
| numpy   | 4,832 | 81 GB | 2.70 GB/s |
| dgen-py | 71,120 | 1,193 GB | **39.76 GB/s** |
| **speedup** | | | **14.75×** |

Note that Section 2 and Section 3 rates are nearly identical for both backends.
This confirms the pool adds essentially zero overhead: the fill rate IS the
consumer-visible throughput once the queue is warm.

---

## Summary table

| Metric | numpy | dgen-py | Speedup |
|--------|------:|--------:|--------:|
| Single fill (256 MB, 1 thread) | 0.63 GB/s | 37.80 GB/s | **60×** |
| Pure fill (4 threads, sustained) | 2.63 GB/s | 39.66 GB/s | **15×** |
| Consumer get_view (1 thread) | 2.70 GB/s | 39.76 GB/s | **15×** |

---

## Why this matters for storage benchmarks

A storage benchmark at 10 GB/s NVMe write speed needs the data generator to
run **faster** than storage, otherwise benchmark throughput is capped by the
generator — not the storage.

- numpy pool at 2.70 GB/s: **bottleneck** for anything above ~2.7 GB/s storage
- dgen-py pool at 39.76 GB/s: headroom for storage up to ~40 GB/s on this system

On faster hardware (e.g. 32-core server), adding more dgen-py producers scales
linearly because each thread's `fill_chunk()` runs independently; numpy
producers plateau quickly due to GIL serialization.

---

## Rebuttal to numpy-baseline comparisons

Benchmarks that show numpy faster than dgen-py are typically measuring the
**old single-buffer-reuse design**: generate one 256 MB buffer at startup, then
return `memoryview` pointer slices of that same buffer for every subsequent
call. This is O(1) pointer arithmetic with zero generation work — all calls
after the first return _identical data_. Any storage target with deduplication
will report inflated throughput because it is not doing real I/O after the
first 256 MB.

The correct comparison (this benchmark) uses a continuously-regenerating pool
for both backends, producing fresh unique data on every buffer fill.
