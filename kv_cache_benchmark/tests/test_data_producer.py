#!/usr/bin/env python3
"""
Tests for DataGeneratorPool and the timing-isolation guarantee.

Core invariants tested
----------------------
1. Pool starts; producer threads run and keep the ready queue filled.
2. get_view() returns a memoryview of the exact requested size.
3. get_view() is sub-millisecond (pure pointer arithmetic into a pre-filled
   256 MB buffer) — storage timer starts with data already in hand.
4. Pool sustained throughput is >> storage write speed (target >20 GB/s).
5. Pool is ≥ 100× faster than inline generate_buffer() — proving generation
   time is NOT serialised with storage writes.
6. Multiple consumer threads each get independent, correctly-sized views
   (thread-local cursor design).
7. KVCacheGenerator.generate() uses the pool and returns memoryview.

Run with:
    cd kv_cache_benchmark
    pytest tests/test_data_producer.py -v
    # or directly:
    python tests/test_data_producer.py
"""

import sys
import time
import threading
import statistics
import logging
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest

from kv_cache._compat import DGEN_AVAILABLE
from kv_cache.data_producer import DataGeneratorPool, DEFAULT_BUFFER_SIZE_MB

# ── Test parameters ───────────────────────────────────────────────────────────

BUFFER_MB          = DEFAULT_BUFFER_SIZE_MB  # 256 MB
BUFFER_BYTES       = BUFFER_MB * 1024 * 1024
WARMUP_SECONDS     = 1.5   # let producers fill the ready queue fully
MEASUREMENT_ROUNDS = 20    # get_view() calls per throughput measurement

# Acceptance criteria
# get_view() within a warm buffer is pure pointer arithmetic (~0.76 µs measured);
# p95 < 0.5 ms is intentionally generous to allow occasional buffer swaps.
MAX_WARM_GET_MS    = 0.5
MIN_THROUGHPUT_GBS = 20.0  # must sustain > 20 GB/s (targets 15–30 GB/s storage)
MIN_SPEEDUP        = 100.0 # pool must be ≥100× faster than inline generate_buffer()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _ms(seconds: float) -> float:
    return seconds * 1_000.0


def _us(seconds: float) -> float:
    return seconds * 1_000_000.0


def _gbs(bytes_count: int, seconds: float) -> float:
    return bytes_count / seconds / 1e9


def _warm_pool(pool: DataGeneratorPool, n_buffers: int = 2) -> None:
    """Consume n_buffers worth of data to force a buffer swap and verify pool is hot."""
    for _ in range(n_buffers):
        pool.get_view(BUFFER_BYTES)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def pool():
    """Module-scoped pool — start once, reuse across ALL tests, stop at teardown."""
    if not DGEN_AVAILABLE:
        pytest.skip("dgen-py not installed — DataGeneratorPool tests skipped")
    p = DataGeneratorPool(buffer_size_mb=BUFFER_MB, prefetch_depth=8)
    p.start()
    # Let producers fill the ready queue before tests begin
    time.sleep(WARMUP_SECONDS)
    yield p
    p.stop()


# ── Test 1: Pool starts; producers are alive; ready queue has data ────────────

def test_pool_starts(pool):
    """All producer threads must be alive and the ready queue must have data."""
    assert pool.is_alive, "No producer threads are running"
    assert pool._ready.qsize() > 0, (
        f"Ready queue is empty after {WARMUP_SECONDS} s warm-up "
        f"(maxsize={pool._ready.maxsize})"
    )
    print(
        f"\n  {pool._num_producers} producer(s) running; "
        f"ready queue after {WARMUP_SECONDS} s: "
        f"{pool._ready.qsize()}/{pool._ready.maxsize}"
    )


# ── Test 2: Correct length for aligned sizes ──────────────────────────────────

@pytest.mark.parametrize("size_mb", [1, 32, 64, 128, 256])
def test_correct_length_aligned(pool, size_mb):
    """get_view() must return a memoryview of exactly the requested length."""
    expected = size_mb * 1024 * 1024
    view = pool.get_view(expected)
    assert isinstance(view, memoryview), f"Expected memoryview, got {type(view)}"
    assert len(view) == expected, f"Requested {expected} bytes, got {len(view)}"


# ── Test 3: Correct length for non-aligned (KV-entry) sizes ──────────────────

@pytest.mark.parametrize("size_bytes", [
    131_072,           # 128 KiB
    1_048_576,         # 1 MiB
    50_000_000,        # 50 MB — well within one 256 MB buffer
    200_000_001,       # just under one buffer, odd size
])
def test_correct_length_nonaligned(pool, size_bytes):
    view = pool.get_view(size_bytes)
    assert isinstance(view, memoryview)
    assert len(view) == size_bytes, (
        f"Requested {size_bytes} bytes, got {len(view)}"
    )


# ── Test 4: get_view() is sub-millisecond when pool is warm ──────────────────

def test_get_view_latency_when_warm(pool):
    """
    get_view() within a warm buffer must be nearly instant — pure pointer
    arithmetic.  This is the core timing-isolation mechanism: the storage
    write timer starts with data already available, not after generation.
    """
    _warm_pool(pool)          # ensure we have a full buffer loaded in thread-local
    time.sleep(0.3)

    # Use a sub-buffer size so all calls stay within one 256 MB buffer
    # (no buffer swaps during measurement — pure pointer arithmetic path).
    SIZE = 8 * 1024 * 1024    # 8 MB per call

    latencies_us = []
    for _ in range(MEASUREMENT_ROUNDS):
        t0 = time.perf_counter()
        view = pool.get_view(SIZE)
        t1 = time.perf_counter()
        assert len(view) == SIZE
        latencies_us.append(_us(t1 - t0))

    p50   = statistics.median(latencies_us)
    p95   = sorted(latencies_us)[int(0.95 * len(latencies_us))]
    worst = max(latencies_us)

    print(
        f"\n  get_view({SIZE // 1024**2} MB) latency — "
        f"p50={p50:.1f} µs  p95={p95:.1f} µs  worst={worst:.1f} µs"
    )
    print(f"  (target: p95 < {MAX_WARM_GET_MS * 1000:.0f} µs = {MAX_WARM_GET_MS} ms)")

    assert p95 < MAX_WARM_GET_MS * 1000, (
        f"p95 get_view() latency = {p95:.1f} µs — exceeds "
        f"{MAX_WARM_GET_MS * 1000:.0f} µs target"
    )


# ── Test 5: Sustained throughput >> 20 GB/s ───────────────────────────────────

def test_sustained_throughput(pool):
    """
    Pool must sustain > MIN_THROUGHPUT_GBS (20 GB/s).
    Targets 15–30 GB/s all-flash storage systems.
    """
    _warm_pool(pool)
    time.sleep(0.3)

    total_bytes = 0
    t0 = time.perf_counter()
    for _ in range(MEASUREMENT_ROUNDS):
        view = pool.get_view(BUFFER_BYTES)
        total_bytes += len(view)
    elapsed = time.perf_counter() - t0

    gbs = _gbs(total_bytes, elapsed)
    print(
        f"\n  Sustained get_view() throughput: {gbs:.1f} GB/s "
        f"over {total_bytes / 1e9:.1f} GB "
        f"({MEASUREMENT_ROUNDS} × {BUFFER_MB} MB, {elapsed:.2f} s)"
    )
    print(f"  Target: > {MIN_THROUGHPUT_GBS} GB/s")

    assert gbs >= MIN_THROUGHPUT_GBS, (
        f"Pool throughput {gbs:.1f} GB/s < {MIN_THROUGHPUT_GBS} GB/s minimum"
    )


# ── Test 6: Pool is >> faster than inline generate_buffer() ──────────────────

def test_pool_vs_inline_latency():
    """
    Core timing-isolation proof:

    Compares:
      A) Pool path  : get_view()                — memoryview pointer (~µs)
      B) Inline path: dgen_py.generate_buffer() — synchronous generation (~ms)

    The pool MUST be ≥ MIN_SPEEDUP× faster.  If it is, wrapping backend.write()
    in a timer EXCLUDES generation time when using the pool, but INCLUDES it
    when using the inline path — proving the pool eliminates generation from
    the storage I/O critical path.
    """
    if not DGEN_AVAILABLE:
        pytest.skip("dgen-py not installed")

    import dgen_py

    size = 32 * 1024 * 1024   # 32 MB — fits in one buffer, fast to measure inline too

    # ── A: Pool path ──────────────────────────────────────────────────────────
    pool_a = DataGeneratorPool(buffer_size_mb=BUFFER_MB, prefetch_depth=8)
    pool_a.start()
    time.sleep(WARMUP_SECONDS)
    _warm_pool(pool_a)  # load thread-local buffer

    pool_latencies_us = []
    for _ in range(MEASUREMENT_ROUNDS):
        t0 = time.perf_counter()
        view = pool_a.get_view(size)
        t1 = time.perf_counter()
        assert isinstance(view, memoryview)
        assert len(view) == size
        pool_latencies_us.append(_us(t1 - t0))
    pool_a.stop()

    # ── B: Inline path ────────────────────────────────────────────────────────
    inline_latencies_ms = []
    for _ in range(MEASUREMENT_ROUNDS):
        t0 = time.perf_counter()
        data = bytes(dgen_py.generate_buffer(size))
        t1 = time.perf_counter()
        assert len(data) == size
        inline_latencies_ms.append(_ms(t1 - t0))

    pool_p50_us   = statistics.median(pool_latencies_us)
    inline_p50_ms = statistics.median(inline_latencies_ms)
    speedup = (inline_p50_ms * 1000) / pool_p50_us if pool_p50_us > 0 else float("inf")

    print(
        f"\n  Timing isolation comparison ({size // 1024**2} MB, "
        f"{MEASUREMENT_ROUNDS} rounds):"
    )
    print(
        f"  Pool   p50 = {pool_p50_us:.1f} µs  "
        f"(memoryview pointer; storage timer starts immediately)"
    )
    print(
        f"  Inline p50 = {inline_p50_ms:.1f} ms  "
        f"(generation serialised with write)"
    )
    print(f"  Speedup: {speedup:.0f}×  (target ≥ {MIN_SPEEDUP:.0f}×)")
    print()
    print(f"  ✓ Using the pool, storage timing excludes ~{inline_p50_ms:.1f} ms of")
    print(f"    generation overhead per {size // 1024**2} MB entry.")

    assert speedup >= MIN_SPEEDUP, (
        f"Pool is only {speedup:.0f}× faster than inline — "
        f"expected ≥ {MIN_SPEEDUP:.0f}×.  Pool may not be warm."
    )


# ── Test 7: Thread safety — each thread gets its own independent cursor ───────

def test_concurrent_get_view(pool):
    """
    Multiple threads calling get_view() simultaneously must each get a
    correctly-sized memoryview.  The thread-local design means each thread
    draws from its own current buffer with no contention.
    """
    _warm_pool(pool)
    time.sleep(0.3)

    SIZE = 8 * 1024 * 1024
    N_THREADS = min(pool._num_producers, 8)
    errors = []

    def _worker(tid: int):
        try:
            for _ in range(4):
                view = pool.get_view(SIZE)
                if not isinstance(view, memoryview):
                    errors.append(f"thread {tid}: got {type(view)}, not memoryview")
                    return
                if len(view) != SIZE:
                    errors.append(
                        f"thread {tid}: expected {SIZE} bytes, got {len(view)}"
                    )
        except Exception as exc:
            errors.append(f"thread {tid}: exception: {exc}")

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    print(
        f"\n  Concurrent test: {N_THREADS} threads × 4 calls × {SIZE // 1024**2} MB "
        f"= {N_THREADS * 4 * SIZE // 1024**2} MB total"
    )
    if errors:
        pytest.fail("Thread-safety failures:\n" + "\n".join(errors))


# ── Test 8: KVCacheGenerator integration — uses pool, returns memoryview ──────

def test_kvcache_generator_uses_pool():
    """
    KVCacheGenerator with prefetch_depth > 0 must use the pool path and return
    a memoryview in < MAX_WARM_GET_MS ms (p95) after warm-up.
    """
    from kv_cache.cache import KVCacheGenerator
    from kv_cache.models import ModelConfig

    mc = ModelConfig(
        name="test_model",
        num_layers=4,
        hidden_dim=512,
        num_heads=8,
        kv_heads=4,
    )
    gen = KVCacheGenerator(mc, global_seed=42, prefetch_depth=8)

    if gen._producer_pool is None:
        pytest.skip("dgen-py not installed; pool not created")

    time.sleep(WARMUP_SECONDS)
    for _ in range(2):
        gen.generate(sequence_length=256)

    latencies_ms = []
    for _ in range(20):
        t0 = time.perf_counter()
        data = gen.generate(sequence_length=256)
        t1 = time.perf_counter()
        latencies_ms.append(_ms(t1 - t0))

    entry_size = mc.kv_cache_size_per_token * 256
    p50 = statistics.median(latencies_ms)
    p95 = sorted(latencies_ms)[int(0.95 * len(latencies_ms))]

    print(
        f"\n  KVCacheGenerator.generate() [entry={entry_size // 1024} KiB via pool]"
    )
    print(f"  p50={p50:.3f} ms  p95={p95:.3f} ms  (target p95 < {MAX_WARM_GET_MS} ms)")

    assert isinstance(data, memoryview), (
        f"Expected memoryview from pool path, got {type(data)}"
    )
    assert len(data) == entry_size, f"Expected {entry_size} bytes, got {len(data)}"
    assert p95 < MAX_WARM_GET_MS, (
        f"KVCacheGenerator.generate() p95={p95:.3f} ms, expected < {MAX_WARM_GET_MS} ms"
    )

    gen.shutdown()


# ── Test 9: stop / fresh instance is clean ────────────────────────────────────

def test_stop_and_new_instance():
    """Stopping a pool and creating a fresh one must work correctly."""
    if not DGEN_AVAILABLE:
        pytest.skip("dgen-py not installed")

    p = DataGeneratorPool(buffer_size_mb=BUFFER_MB, prefetch_depth=4)
    p.start()
    time.sleep(WARMUP_SECONDS)

    view = p.get_view(BUFFER_BYTES)
    assert isinstance(view, memoryview)
    assert len(view) == BUFFER_BYTES

    p.stop()
    time.sleep(0.2)
    assert not p.is_alive, "Thread(s) still alive after stop()"

    p2 = DataGeneratorPool(buffer_size_mb=BUFFER_MB, prefetch_depth=4)
    p2.start()
    time.sleep(WARMUP_SECONDS)
    view2 = p2.get_view(BUFFER_BYTES)
    assert isinstance(view2, memoryview)
    assert len(view2) == BUFFER_BYTES
    p2.stop()


# ── Standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
    )
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short", "-s"]
        + sys.argv[1:],
        cwd=str(ROOT),
    )
    sys.exit(result.returncode)
