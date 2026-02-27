#!/usr/bin/env python3
"""
bench_fill_comparison.py — Producer-pool fill-rate comparison: numpy vs dgen-py

Measures ONLY the variable that matters: which RNG can fill 256 MB bytearrays
fastest when used inside the same producer-consumer pool architecture.

Both implementations:
  • Pre-allocate N × 256 MB bytearrays (no allocation in the hot path).
  • Run M background producer threads that fill each buffer IN PLACE.
  • Expose get_view(size) → memoryview — zero-copy pointer slice.
  • The single-buffer-reuse approach is intentionally excluded; that design
    produces 100% deduplicatable data and is not suitable for storage benchmarks.

numpy fill:  rng.integers(0, 256, size=n, dtype=np.uint8) → temp ndarray
             copied into buf[:] — one extra alloc + memcpy per buffer.
             numpy.random.Generator.integers() has NO out= parameter, so
             a temporary 256 MB array is always allocated internally.
             GIL is HELD for the entire fill.

dgen-py fill: gen.fill_chunk(buf)
              — in-place, GIL RELEASED, Rayon-parallel Xoshiro256++.

Usage
-----
    pip install dgen-py numpy
    python tests/bench_fill_comparison.py
    python tests/bench_fill_comparison.py --duration 30 --producers 4 --buffer-mb 256
    python tests/bench_fill_comparison.py --duration 60 --check-dedup
"""

import argparse
import hashlib
import os
import queue
import sys
import threading
import time
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BUFFER_MB: int = 256
DEFAULT_DURATION_S: int = 20
DEFAULT_PREFETCH: int = 8


def _default_num_producers() -> int:
    return max(2, (os.cpu_count() or 4) // 2)


# ---------------------------------------------------------------------------
# Shared pool base — identical lifecycle and get_view() for both backends
# ---------------------------------------------------------------------------

class _FillPool:
    """
    Abstract producer-consumer pool.  Subclasses implement _fill(buf: bytearray).
    """

    def __init__(self, buffer_mb: int, prefetch_depth: int, num_producers: int):
        self._buf_size = buffer_mb * 1024 * 1024
        self._buf_mb = buffer_mb
        self._n_producers = num_producers
        self._prefetch = prefetch_depth

        self._empty: queue.Queue = queue.Queue()
        self._ready: queue.Queue = queue.Queue(maxsize=prefetch_depth)
        self._stop = threading.Event()

        total = num_producers + prefetch_depth + 4
        # Pre-allocate ALL buffers at once via Rust — ~1000× faster than a
        # Python loop of bytearray() calls.  Both variants share this path so
        # allocation cost is NOT part of the fill-rate comparison.
        import dgen_py as _dgen
        for buf in _dgen.create_bytearrays(count=total, size=self._buf_size):
            self._empty.put(buf)

        self._tls = threading.local()
        self._threads = [
            threading.Thread(target=self._producer_loop,
                             name=f"{self._name()}-{i}", daemon=True)
            for i in range(num_producers)
        ]
        self._started = False

        # Counters for throughput measurement
        self._fill_lock = threading.Lock()
        self._fill_bytes: int = 0
        self._fill_time_s: float = 0.0
        self._fill_count: int = 0

    def _name(self) -> str:
        raise NotImplementedError

    def _fill(self, buf: bytearray) -> None:
        """Fill buf in-place with fresh random bytes.  Called from producer thread."""
        raise NotImplementedError

    def start(self) -> "_FillPool":
        if not self._started:
            for t in self._threads:
                t.start()
            self._started = True
        return self

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=3.0)
        self._started = False

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()

    # -----------------------------------------------------------------
    # Consumer API — zero-copy (identical to DataGeneratorPool.get_view)
    # -----------------------------------------------------------------

    def get_view(self, size: int) -> memoryview:
        if size <= 0:
            return memoryview(b"")
        if size > self._buf_size:
            return self._get_oversized(size)

        tls = self._tls
        if not hasattr(tls, "buf") or tls.offset + size > self._buf_size:
            self._swap_buffer()

        start = tls.offset
        tls.offset += size
        return tls.view[start: start + size]

    def _swap_buffer(self) -> None:
        tls = self._tls
        if hasattr(tls, "buf"):
            tls.view.release()
            self._empty.put(tls.buf)
        tls.buf = self._ready.get()
        tls.view = memoryview(tls.buf)
        tls.offset = 0

    def _get_oversized(self, size: int) -> memoryview:
        buf = bytearray(size)
        self._fill(buf)
        return memoryview(buf)

    # -----------------------------------------------------------------
    # Producer loop (identical structure, calls self._fill)
    # -----------------------------------------------------------------

    def _producer_loop(self) -> None:
        self._setup_producer()
        while not self._stop.is_set():
            try:
                buf = self._empty.get(timeout=0.1)
            except queue.Empty:
                continue

            t0 = time.perf_counter()
            self._fill(buf)
            elapsed = time.perf_counter() - t0

            with self._fill_lock:
                self._fill_bytes += len(buf)
                self._fill_time_s += elapsed
                self._fill_count += 1

            while not self._stop.is_set():
                try:
                    self._ready.put(buf, timeout=0.1)
                    break
                except queue.Full:
                    continue

    def _setup_producer(self) -> None:
        """Called once per producer thread before the fill loop starts."""
        pass

    # -----------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------

    def fill_stats(self):
        """Return (total_gb, total_fills, avg_fill_time_ms, avg_fill_gb_s)."""
        with self._fill_lock:
            gb = self._fill_bytes / 1e9
            n = self._fill_count
            t = self._fill_time_s
        avg_ms = (t / n * 1000) if n > 0 else 0.0
        avg_gbs = (self._fill_bytes / t / 1e9) if t > 0 else 0.0
        return gb, n, avg_ms, avg_gbs


# ---------------------------------------------------------------------------
# numpy backend — rng.integers(out=arr) — IN-PLACE, GIL-HELD
# ---------------------------------------------------------------------------

class NumpyFillPool(_FillPool):
    """
    Producer pool that fills 256 MB buffers using numpy.random.Generator.

    numpy.random.Generator.integers() does NOT support an 'out=' parameter,
    so the fill requires a temporary uint8 allocation + a copy into the
    pre-allocated bytearray.  The GIL is held for the entire operation.

    This is the numpy equivalent that is fairest to numpy: per-thread
    Generator, no shared state, same buffer size as dgen-py.

    Extra cost vs dgen-py: one temporary 256 MB allocation per fill + a copy.
    dgen-py fill_chunk() writes DIRECTLY into the bytearray — zero extra alloc.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._np_tls = threading.local()  # per-thread RNG storage

    def _name(self) -> str:
        return "numpy"

    def _setup_producer(self) -> None:
        seed = int(threading.current_thread().ident or 0) & 0xFFFFFFFF
        self._np_tls.rng = np.random.default_rng(seed)

    def _fill(self, buf: bytearray) -> None:
        # integers() has no out= support — allocates a new array, then copies.
        # This is an unavoidable extra allocation on every fill call.
        arr = np.frombuffer(buf, dtype=np.uint8)  # writable view into bytearray
        arr[:] = self._np_tls.rng.integers(0, 256, size=len(arr), dtype=np.uint8)


# ---------------------------------------------------------------------------
# dgen-py backend — gen.fill_chunk(buf) — IN-PLACE, GIL-FREE, Rayon-parallel
# ---------------------------------------------------------------------------

class DgenFillPool(_FillPool):
    """
    Producer pool that fills 256 MB buffers using dgen_py.Generator.fill_chunk().

    fill_chunk() releases the Python GIL and uses Rayon thread-pool (all cores)
    to fill the buffer with Xoshiro256++ PRNG bytes.  Each producer thread
    has its own Generator to avoid contention on shared PRNG state.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dgen_tls = threading.local()  # per-thread Generator storage

    def _name(self) -> str:
        return "dgen-py"

    def _setup_producer(self) -> None:
        import dgen_py
        self._dgen_tls.gen = dgen_py.Generator(
            size=1 << 44,        # 16 TB — never exhausts in practice
            compress_ratio=1.0,  # incompressible output
            numa_mode="auto",
        )

    def _fill(self, buf: bytearray) -> None:
        # fill_chunk holds a mutable Rust borrow for its duration;
        # do NOT call any other Generator methods while it is running.
        # With size=1<<44 (16 TB of PRNG state), exhaustion never occurs
        # in practice, so no is_complete() / reset() check is needed.
        self._dgen_tls.gen.fill_chunk(buf)


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def _benchmark_pure_fill_numpy(buffer_mb: int, num_threads: int, duration_s: float):
    """
    Pure numpy fill throughput: N threads each own one pre-allocated bytearray
    and call rng.integers() → arr[:] in a tight loop for duration_s seconds.
    No queues, no blocking, no consumer.  GIL contention is the only limit.
    """
    buf_size = buffer_mb * 1024 * 1024
    results = []
    barrier = threading.Barrier(num_threads + 1)

    def worker():
        rng = np.random.default_rng(
            int(threading.current_thread().ident or 0) & 0xFFFFFFFF
        )
        buf = bytearray(buf_size)
        arr = np.frombuffer(buf, dtype=np.uint8)
        barrier.wait()                          # all threads start together
        deadline = time.perf_counter() + duration_s
        total = 0
        n = 0
        while time.perf_counter() < deadline:
            arr[:] = rng.integers(0, 256, size=len(arr), dtype=np.uint8)
            total += buf_size
            n += 1
        results.append((total, n))

    threads = [threading.Thread(target=worker, daemon=True)
               for _ in range(num_threads)]
    for t in threads:
        t.start()
    barrier.wait()
    t0 = time.perf_counter()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    total_bytes = sum(r[0] for r in results)
    total_fills = sum(r[1] for r in results)
    return total_bytes / wall / 1e9, total_fills, wall


def _benchmark_pure_fill_dgen(buffer_mb: int, num_threads: int, duration_s: float):
    """
    Pure dgen-py fill throughput: N threads each own one pre-allocated bytearray
    and call gen.fill_chunk() in a tight loop for duration_s seconds.
    GIL released inside fill_chunk — all N threads run concurrently via Rayon.
    No queues, no blocking, no consumer.
    """
    import dgen_py
    buf_size = buffer_mb * 1024 * 1024
    results = []
    barrier = threading.Barrier(num_threads + 1)

    def worker():
        gen = dgen_py.Generator(size=1 << 44, compress_ratio=1.0, numa_mode="auto")
        buf = bytearray(buf_size)
        barrier.wait()                          # all threads start together
        deadline = time.perf_counter() + duration_s
        total = 0
        n = 0
        while time.perf_counter() < deadline:
            gen.fill_chunk(buf)
            if gen.is_complete():
                gen.reset()
            total += buf_size
            n += 1
        results.append((total, n))

    threads = [threading.Thread(target=worker, daemon=True)
               for _ in range(num_threads)]
    for t in threads:
        t.start()
    barrier.wait()
    t0 = time.perf_counter()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    total_bytes = sum(r[0] for r in results)
    total_fills = sum(r[1] for r in results)
    return total_bytes / wall / 1e9, total_fills, wall



    """Drain a few views to let producers fill the ready queue."""
    deadline = time.perf_counter() + warmup_s
    consumed = 0
    while time.perf_counter() < deadline:
        v = pool.get_view(4 * 1024 * 1024)
        consumed += len(v)
    return consumed


def _warmup(pool: _FillPool, warmup_s: float = 2.0) -> int:
    """Drain a few views to let producers fill the ready queue."""
    deadline = time.perf_counter() + warmup_s
    consumed = 0
    while time.perf_counter() < deadline:
        v = pool.get_view(4 * 1024 * 1024)
        consumed += len(v)
    return consumed


def _benchmark_consumer(pool: _FillPool, duration_s: float, entry_bytes: int):
    """
    Single-threaded consumer: call get_view(entry_bytes) in a tight loop
    for duration_s seconds.  Returns (total_bytes, elapsed_s, n_calls).
    """
    total = 0
    calls = 0
    deadline = time.perf_counter() + duration_s
    while time.perf_counter() < deadline:
        v = pool.get_view(entry_bytes)
        total += len(v)
        calls += 1
    elapsed = time.perf_counter() - (deadline - duration_s)
    return total, elapsed, calls


def _benchmark_consumer_mt(pool: _FillPool, duration_s: float,
                            entry_bytes: int, num_threads: int):
    """
    Multi-threaded consumer: N threads each call get_view() for duration_s.
    Returns (total_bytes, elapsed_s).
    """
    results = []
    barrier = threading.Barrier(num_threads + 1)

    def worker():
        barrier.wait()
        total, elapsed, calls = _benchmark_consumer(pool, duration_s, entry_bytes)
        results.append((total, elapsed, calls))

    threads = [threading.Thread(target=worker, daemon=True)
               for _ in range(num_threads)]
    for t in threads:
        t.start()

    barrier.wait()
    t0 = time.perf_counter()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    total_bytes = sum(r[0] for r in results)
    return total_bytes, wall


def _dedup_rate(pool: _FillPool, num_blocks: int = 16) -> float:
    """
    Rough dedup estimate: sample num_blocks × 256 MB, hash each, count collisions.
    Returns collision fraction (0.0 = no dedup, 1.0 = 100% dedup).
    """
    hashes = set()
    collisions = 0
    for _ in range(num_blocks):
        v = pool.get_view(pool._buf_size)
        h = hashlib.sha256(bytes(v)).hexdigest()
        if h in hashes:
            collisions += 1
        else:
            hashes.add(h)
    return collisions / num_blocks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Compare numpy vs dgen-py buffer-fill throughput "
                    "inside an identical producer-consumer pool.")
    ap.add_argument("--duration", type=float, default=DEFAULT_DURATION_S,
                    help=f"Consumer benchmark duration in seconds (default {DEFAULT_DURATION_S})")
    ap.add_argument("--producers", type=int, default=_default_num_producers(),
                    help="Producer threads for each pool (default: cpu_count//2)")
    ap.add_argument("--buffer-mb", type=int, default=DEFAULT_BUFFER_MB,
                    help=f"Buffer size in MB (default {DEFAULT_BUFFER_MB})")
    ap.add_argument("--prefetch", type=int, default=DEFAULT_PREFETCH,
                    help=f"Ready-queue depth (default {DEFAULT_PREFETCH})")
    ap.add_argument("--entry-mb", type=float, default=16.0,
                    help="KV cache entry size per get_view() call in MB (default 16)")
    ap.add_argument("--consumer-threads", type=int, default=1,
                    help="Concurrent consumer threads (default 1)")
    ap.add_argument("--check-dedup", action="store_true",
                    help="Hash 16 × 256 MB blocks and report collision rate")
    ap.add_argument("--skip-numpy", action="store_true",
                    help="Skip numpy pool (e.g. to profile dgen-py alone)")
    ap.add_argument("--skip-dgen", action="store_true",
                    help="Skip dgen-py pool (e.g. to profile numpy alone)")
    args = ap.parse_args()

    entry_bytes = int(args.entry_mb * 1024 * 1024)
    cpus = os.cpu_count() or 1

    print("=" * 70)
    print("  Producer-Pool Fill-Rate Comparison: numpy vs dgen-py")
    print("=" * 70)
    print(f"  System CPUs       : {cpus}")
    print(f"  Producers per pool: {args.producers}")
    print(f"  Buffer size       : {args.buffer_mb} MB")
    print(f"  Prefetch depth    : {args.prefetch} buffers "
          f"({args.prefetch * args.buffer_mb} MB ready queue)")
    print(f"  Total RAM / pool  : "
          f"{(args.producers + args.prefetch + 4) * args.buffer_mb} MB")
    print(f"  Consumer entry    : {args.entry_mb:.0f} MB per get_view() call")
    print(f"  Consumer threads  : {args.consumer_threads}")
    print(f"  Benchmark duration: {args.duration:.0f}s per pool")
    print()

    # Check dgen-py availability
    dgen_available = False
    if not args.skip_dgen:
        try:
            import dgen_py  # noqa: F401
            dgen_available = True
        except ImportError:
            print("  WARNING: dgen-py not installed — skipping dgen-py pool")
            print("  Install with: pip install dgen-py")
            print()

    # -------------------------------------------------------------------------
    # Section 1: Single-fill latency (one buffer, one producer thread, in-place)
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("  Section 1 — Single-buffer fill latency (1 producer, 1 fill call)")
    print("-" * 70)

    buf_single = bytearray(args.buffer_mb * 1024 * 1024)

    # numpy single fill
    if not args.skip_numpy:
        rng = np.random.default_rng(42)
        arr_single = np.frombuffer(buf_single, dtype=np.uint8)  # writable view
        # warmup
        arr_single[:] = rng.integers(0, 256, size=len(arr_single), dtype=np.uint8)
        # measure
        N_SINGLE = 10
        t0 = time.perf_counter()
        for _ in range(N_SINGLE):
            arr_single[:] = rng.integers(0, 256, size=len(arr_single), dtype=np.uint8)
        numpy_single_ms = (time.perf_counter() - t0) / N_SINGLE * 1000
        numpy_single_gbs = (args.buffer_mb / 1024) / (numpy_single_ms / 1000)
        print(f"  numpy  fill ({args.buffer_mb} MB, alloc+copy): "
              f"{numpy_single_ms:7.1f} ms   {numpy_single_gbs:.2f} GB/s")

    if dgen_available and not args.skip_dgen:
        import dgen_py
        gen = dgen_py.Generator(size=1 << 44, compress_ratio=1.0, numa_mode="auto")
        # warmup
        gen.fill_chunk(buf_single)
        N_SINGLE = 10
        t0 = time.perf_counter()
        for _ in range(N_SINGLE):
            gen.fill_chunk(buf_single)
        dgen_single_ms = (time.perf_counter() - t0) / N_SINGLE * 1000
        dgen_single_gbs = (args.buffer_mb / 1024) / (dgen_single_ms / 1000)
        print(f"  dgen-py fill_chunk({args.buffer_mb} MB):  "
              f"{dgen_single_ms:7.1f} ms   {dgen_single_gbs:.2f} GB/s")

    print()

    # -------------------------------------------------------------------------
    # Section 2: Pure producer fill throughput (no queues, no consumer)
    # -------------------------------------------------------------------------
    print("-" * 70)
    print(f"  Section 2 — Pure fill throughput ({args.producers} threads, "
          f"{args.duration:.0f}s, no queues, no consumer)")
    print(f"  Each thread owns 1 × {args.buffer_mb} MB buffer and fills it "
          f"in a tight loop")
    print("-" * 70)

    numpy_pool_gbs = None
    dgen_pool_gbs = None

    if not args.skip_numpy:
        print(f"  [numpy] {args.producers} thread(s) filling …", flush=True)
        numpy_pool_gbs, np_fills, np_wall = _benchmark_pure_fill_numpy(
            args.buffer_mb, args.producers, args.duration)
        print(f"  [numpy] {np_fills} fills in {np_wall:.1f}s  "
              f"throughput={numpy_pool_gbs:.2f} GB/s  "
              f"({args.producers} thread(s))")

    if dgen_available and not args.skip_dgen:
        print(f"  [dgen-py] {args.producers} thread(s) filling …", flush=True)
        dgen_pool_gbs, dg_fills, dg_wall = _benchmark_pure_fill_dgen(
            args.buffer_mb, args.producers, args.duration)
        print(f"  [dgen-py] {dg_fills} fills in {dg_wall:.1f}s  "
              f"throughput={dgen_pool_gbs:.2f} GB/s  "
              f"({args.producers} thread(s))")

    print()

    # -------------------------------------------------------------------------
    # Section 3: Consumer throughput — get_view() sustained rate
    # -------------------------------------------------------------------------
    print("-" * 70)
    print(f"  Section 3 — Consumer get_view() throughput "
          f"({args.consumer_threads} thread(s), {args.duration:.0f}s)")
    print(f"  Entry size: {args.entry_mb:.0f} MB per call")
    print("-" * 70)

    numpy_cons_gbs = None
    dgen_cons_gbs = None

    if not args.skip_numpy:
        with NumpyFillPool(
            buffer_mb=args.buffer_mb,
            prefetch_depth=args.prefetch,
            num_producers=args.producers,
        ) as np_pool:
            _warmup(np_pool, warmup_s=2.0)
            if args.consumer_threads == 1:
                total_b, elapsed, ncalls = _benchmark_consumer(
                    np_pool, args.duration, entry_bytes)
                numpy_cons_gbs = total_b / elapsed / 1e9
                print(f"  [numpy]   {ncalls:,} calls  "
                      f"{total_b/1e9:.1f} GB  "
                      f"{numpy_cons_gbs:.2f} GB/s")
            else:
                total_b, wall = _benchmark_consumer_mt(
                    np_pool, args.duration, entry_bytes, args.consumer_threads)
                numpy_cons_gbs = total_b / wall / 1e9
                print(f"  [numpy]   {total_b/1e9:.1f} GB  "
                      f"{numpy_cons_gbs:.2f} GB/s  "
                      f"({args.consumer_threads} consumers)")

    if dgen_available and not args.skip_dgen:
        with DgenFillPool(
            buffer_mb=args.buffer_mb,
            prefetch_depth=args.prefetch,
            num_producers=args.producers,
        ) as dg_pool:
            _warmup(dg_pool, warmup_s=2.0)
            if args.consumer_threads == 1:
                total_b, elapsed, ncalls = _benchmark_consumer(
                    dg_pool, args.duration, entry_bytes)
                dgen_cons_gbs = total_b / elapsed / 1e9
                print(f"  [dgen-py] {ncalls:,} calls  "
                      f"{total_b/1e9:.1f} GB  "
                      f"{dgen_cons_gbs:.2f} GB/s")
            else:
                total_b, wall = _benchmark_consumer_mt(
                    dg_pool, args.duration, entry_bytes, args.consumer_threads)
                dgen_cons_gbs = total_b / wall / 1e9
                print(f"  [dgen-py] {total_b/1e9:.1f} GB  "
                      f"{dgen_cons_gbs:.2f} GB/s  "
                      f"({args.consumer_threads} consumers)")

    print()

    # -------------------------------------------------------------------------
    # Section 4: Deduplication check (optional)
    # -------------------------------------------------------------------------
    if args.check_dedup:
        print("-" * 70)
        print("  Section 4 — Deduplication check (16 × 256 MB SHA-256 hash)")
        print("  0.00% = fully unique data (correct for storage benchmarks)")
        print("  100%  = all blocks identical (the old single-buffer design flaw)")
        print("-" * 70)

        if not args.skip_numpy:
            with NumpyFillPool(
                buffer_mb=args.buffer_mb,
                prefetch_depth=args.prefetch,
                num_producers=args.producers,
            ) as np_pool:
                _warmup(np_pool)
                rate = _dedup_rate(np_pool) * 100
            print(f"  [numpy]   collision rate: {rate:.2f}%")

        if dgen_available and not args.skip_dgen:
            with DgenFillPool(
                buffer_mb=args.buffer_mb,
                prefetch_depth=args.prefetch,
                num_producers=args.producers,
            ) as dg_pool:
                _warmup(dg_pool)
                rate = _dedup_rate(dg_pool) * 100
            print(f"  [dgen-py] collision rate: {rate:.2f}%")

        print()

    # -------------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("  Summary")
    print("=" * 70)
    header = f"  {'Metric':<38}  {'numpy':>10}  {'dgen-py':>10}  {'speedup':>8}"
    print(header)
    print("  " + "-" * 66)

    def row(label, n_val, d_val, unit="GB/s"):
        n_s = f"{n_val:.2f} {unit}" if n_val is not None else "skipped"
        d_s = f"{d_val:.2f} {unit}" if d_val is not None else "skipped"
        if n_val and d_val:
            sp = f"{d_val/n_val:.2f}×"
        else:
            sp = "—"
        print(f"  {label:<38}  {n_s:>10}  {d_s:>10}  {sp:>8}")

    if not args.skip_numpy or (dgen_available and not args.skip_dgen):
        row(f"Single fill ({args.buffer_mb} MB, 1 thread)",
            numpy_single_gbs if not args.skip_numpy else None,
            dgen_single_gbs if (dgen_available and not args.skip_dgen) else None)
        row(f"Pool fill ({args.producers} producers, sustained)",
            numpy_pool_gbs, dgen_pool_gbs)
        row(f"Consumer get_view ({args.consumer_threads} thread(s))",
            numpy_cons_gbs, dgen_cons_gbs)

    print()
    print("  Notes:")
    print("  • Both pools use identical architecture: same queue sizes, same")
    print("    buffer sizes, same get_view() zero-copy path.  The only")
    print("    variable is the fill function.")
    print("  • numpy fill: GIL held, allocates a NEW 256 MB array then copies")
    print("    into the pre-allocated bytearray — 2× memory traffic per fill.")
    print("    dgen-py fill: GIL released, writes DIRECTLY into the bytearray")
    print("    — 1× memory traffic.  No temporary allocation.")
    print(f"  • numpy GIL means all {args.producers} producer threads serialize")
    print(f"    on fills — net parallelism = 1 thread.")
    print(f"    dgen-py Rayon: {cpus} logical core(s) used per fill_chunk call.")
    print()


if __name__ == "__main__":
    main()
