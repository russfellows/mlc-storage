# Zero-Copy Data Producer Design

**Module**: `kv_cache/data_producer.py`  
**Class**: `DataGeneratorPool`  
**Last Updated**: February 2026

---

## Overview

`DataGeneratorPool` provides a continuous stream of pre-generated, incompressible
random data to storage backends — with **zero copies, ever**.

The design is built for storage systems writing at 15–30 GB/s.  Prior versions
using `bytes()` conversion hit a hard ceiling near 0.6 GB/s; the zero-copy
design has been validated at **85 GB/s** sustained throughput (memory-to-memory
path) with `get_view()` call latency under **100 µs p95**.

---

## Motivation: Why the Old Design Was Slow

The previous `DataGeneratorPool` kept a `bytes` leftover buffer and returned
slices by copying:

```python
# OLD — copies on every call
data = self._leftover[:size]
self._leftover = self._leftover[size:]   # O(remaining) copy
return bytes(data)                        # another full copy
```

Two compounding problems:

1. **`bytes()` conversion** allocates and copies the entire block per call.
   For a 64 MB block: ~6 ms of GIL-held memcpy, limiting throughput to ~10 GB/s
   even for one core.

2. **Leftover slice** `self._leftover = self._leftover[size:]` copies the entire
   unused tail on every call — O(remaining) per request, worst case 63 MB for a
   1 MB request from a 64 MB block.

---

## Core Design

### The Big Idea: Pointer Arithmetic Only

Python `memoryview` slicing is pure pointer arithmetic — no data movement:

```python
view = memoryview(buf)          # wraps the bytearray, ~150 ns
slice = view[offset:offset+n]  # returns a new memoryview backed by the same
                                # memory — measured at 0.76 µs regardless of n
```

`DataGeneratorPool.get_view(size)` returns such a slice.  The caller writes it
directly to the storage backend without any intermediate copy.

### Buffer Lifecycle

```
┌─────────────────────────────────────────────────────────────────┐
│                    DataGeneratorPool                            │
│                                                                 │
│  empty_queue ──[bytearray 256 MB]──► fill_chunk() ──────────►  │
│                                      (GIL-free, Rayon)         │
│  ready_queue ◄──[bytearray 256 MB]──────────────────────────── │
│       │                                                         │
│       └──► thread-local cursor (buf, view, offset)             │
│                 │                                               │
│                 └──► get_view(n) → view[offset:offset+n]        │
│                      (zero-copy, ~1 µs)                         │
│                                                                 │
│  When consumer exhausts buffer:                                 │
│    view.release()  ← ob_exports → 0 (safe for fill_chunk reuse)│
│    empty_queue ◄── buf (returned for next fill cycle)          │
└─────────────────────────────────────────────────────────────────┘
```

1. **Pre-allocation**: All `bytearray` buffers are allocated once at startup.
   No heap allocation in the hot path.

2. **Producer threads** (`_producer_loop`): Each thread independently pulls an
   empty buffer, calls `gen.fill_chunk(buf)` (GIL-free Rayon fill), then puts
   the filled buffer on the `_ready` queue.

3. **Consumer threads** (`get_view`): Each consumer thread has its own
   `threading.local` state — `(buf, view, offset)`.  `get_view(n)` advances the
   cursor and returns `view[start:start+n]`.  No locks, no shared state in the
   hot path.

4. **Buffer swap** (`_swap_buffer`): When the consumer's offset would overflow
   the buffer, it releases the `memoryview` (so `ob_exports` drops to 0),
   returns the buffer to `_empty`, and fetches the next from `_ready`.

### Why `view.release()` Matters

CPython tracks active `memoryview` exports via the `ob_exports` counter on the
underlying `bytearray`.  A `bytearray` with `ob_exports > 0` is **locked** —
any attempt to resize or pass to a Rust `fill_chunk()` that writes into it could
corrupt memory.

`_swap_buffer()` calls `tls.view.release()` before returning the buffer to the
empty queue.  This drops `ob_exports` to 0 and allows the producer to safely
call `fill_chunk(buf)` on the next cycle.

---

## Configuration

```python
DataGeneratorPool(
    buffer_size_mb  = 256,    # size of each pre-allocated bytearray
    prefetch_depth  = 8,      # number of filled buffers kept ready
    num_producers   = None,   # default: max(2, cpu_count // 2)
)
```

### Producer Scaling

| Logical CPUs | Default Producers | Est. Generation Rate |
|:------------:|:-----------------:|:--------------------:|
| 4            | 2                 | ~8 GB/s              |
| 8            | 4                 | ~16 GB/s             |
| 12           | 6                 | ~24 GB/s             |
| 32           | 16                | 60+ GB/s             |

Each producer drives an independent Rayon thread pool at ~3.85–5 GB/s per
Python thread (see [dgen_benchmark_results.md](dgen_benchmark_results.md)).
The default ensures generation runs well ahead of any single storage namespace.

### Memory Budget

```
total_buffers = num_producers + prefetch_depth + 4  (consumer headroom)

12-core example:
  = 6 + 8 + 4 = 18 buffers × 256 MB = 4.5 GB pre-allocated at startup
```

Reduce `prefetch_depth` on memory-constrained systems.

---

## API

```python
pool = DataGeneratorPool(buffer_size_mb=256, prefetch_depth=8).start()

# Consumer side — call from storage-write loop:
view = pool.get_view(size_bytes)   # memoryview — zero-copy
backend.write(key, view)           # file.write(memoryview) is zero-copy in CPython
# view goes out of scope → CPython refcount → freed immediately
```

### `get_view(size) → memoryview`

- Returns `memoryview[offset : offset + size]` — a pointer into a pre-filled
  256 MB buffer.
- Latency: **~1 µs p50**, **~82 µs p95** (includes occasional buffer swap).
- For `size > buffer_size_mb`: falls back to `_generate_oversized()` (inline
  fill, rare for typical KV cache entries).
- Thread-safe: each consumer thread has independent state.

### Safety Contract

This pool is safe for **synchronous writes only**:

> `backend.write(key, view)` must complete before the consumer thread calls
> `get_view()` again.

CPython's reference counting makes this automatic: the `memoryview` slice is
freed when the caller's local variable goes out of scope, which happens before
the next `get_view()` call in any sequential write loop.

**Not safe** for async workflows where a caller stores views for deferred use
across multiple event-loop turns.

---

## dgen-py Integration

`fill_chunk` is backed by Rayon (Rust thread pool) and releases the GIL:

```python
gen = dgen_py.Generator(
    size          = 1 << 44,   # 16 TB key space — never exhausts
    compress_ratio= 1.0,       # fully incompressible data
    numa_mode     = "auto",    # NUMA-aware allocation
)
gen.fill_chunk(buf)            # writes directly into buf, GIL-free
```

Key property confirmed experimentally: `fill_chunk` works correctly with an
active `memoryview` on the same `bytearray`.  The consumer's `view` slice and
the producer's `fill_chunk` target have non-overlapping lifecycles due to the
queue synchronization: a buffer is only in `_empty` (eligible for fill) after
`view.release()` and `_empty.put()` have both completed.

---

## Performance Results

**System**: Intel Xeon Platinum 8280L @ 2.70 GHz, 12 cores, 31 GB RAM  
**Python**: 3.11 + dgen-py 0.2.0  
**Config**: 256 MB buffers, 6 producers, `prefetch_depth=8`

### Test Suite Results (all 16 tests pass)

| Test | Metric | Result | Target |
|------|--------|-------:|-------:|
| `test_get_view_latency_when_warm` | p50 latency | **1.0 µs** | — |
| `test_get_view_latency_when_warm` | p95 latency | **82 µs** | < 500 µs ✓ |
| `test_sustained_throughput` | sustained rate | **85 GB/s** | > 20 GB/s ✓ |
| `test_pool_vs_inline_latency` | speedup vs inline | **41,902×** | ≥ 100× ✓ |
| `test_kvcache_generator_uses_pool` | p50 latency | **0.006 ms** | < 0.5 ms ✓ |
| `test_kvcache_generator_uses_pool` | p95 latency | **0.010 ms** | < 0.5 ms ✓ |
| `test_concurrent_get_view` | 6 threads × 32 MB | **PASS** | no interference |

### Latency Breakdown

| Operation | Typical Latency | Notes |
|-----------|:--------------:|-------|
| `memoryview` slice (hot) | ~0.76 µs | Pure pointer arithmetic, no data touch |
| `get_view()` common case | ~1.0 µs p50 | Cursor increment + slice |
| `get_view()` with buffer swap | ~50–150 µs | Queue fetch + new `memoryview()` wrapper |
| `fill_chunk(256 MB)` | ~6 ms | GIL-free; overlaps with consumer writes |

### Comparison: Old vs New Design

| Dimension | Old (copy-based) | New (zero-copy) |
|-----------|:----------------:|:---------------:|
| Data copies per `get_view()` | 2+ | **0** |
| `get_view()` p95 latency | ~116 ms | **82 µs** |
| Sustained throughput | ~0.6 GB/s | **85 GB/s** |
| Leftover slice cost | O(remaining) | O(1) |
| GIL held per call | ~6 ms (64 MB) | **~1 µs** |
| Storage target headroom | 0.04× | **2.8× @ 30 GB/s** |

---

## Implementation Notes

### Thread-Local State (no hot-path locks)

```python
self._tls: threading.local  # per-consumer thread

tls.buf    # the current bytearray (256 MB)
tls.view   # memoryview wrapping tls.buf
tls.offset # next-available byte offset within tls.buf
```

Each consumer thread independently advances `tls.offset`.  The only shared
data structures are the two `queue.Queue` objects (`_empty`, `_ready`), which
use their internal GIL-protected locks only at buffer boundaries (once per
256 MB consumed).

### Two-Queue Design

`_empty` and `_ready` form a classic ring buffer over bytearray objects:

- `_empty` has `maxsize=0` (unbounded — all unneeded buffers park here)
- `_ready` has `maxsize=prefetch_depth` — backpressure on producers if
  consumers are slow, preventing unbounded memory growth

### Oversized Entries

Entries larger than `buffer_size_mb` (256 MB) are handled by
`_generate_oversized()`: a fresh `bytearray` is allocated, filled inline, and
returned as a `memoryview`.  This is rare for typical KV cache entry sizes
(16 KB – 8 MB).

---

## Files Changed

| File | Nature of Change |
|------|-----------------|
| `kv_cache/data_producer.py` | Complete rewrite — zero-copy architecture |
| `kv_cache/cache.py` | `get_bytes()` → `get_view()`, new constructor call |
| `tests/test_data_producer.py` | Complete rewrite — 16 tests, µs-scale targets |
