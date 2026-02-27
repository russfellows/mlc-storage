"""
Zero-copy producer-consumer pipeline for KV cache data generation.

Design: pointer-only, NO copies ever
-------------------------------------
Producer threads fill pre-allocated 256 MB bytearrays via
dgen_py.Generator.fill_chunk() — a GIL-free Rayon-parallel Xoshiro256++ fill
that achieves 47+ GB/s across all cores.

get_view(size) returns memoryview[offset : offset+size] — a pointer into the
current pre-filled 256 MB buffer.  No data is EVER copied.

Each consumer thread has its own current buffer and offset cursor (via
threading.local) so there are no locks or contention in the hot path.

Buffer lifecycle
----------------
empty_queue  →  fill_chunk (producer)  →  ready_queue
    →  thread-local cursor (consumer)  →  empty_queue  →  ...

The buffer is returned to empty_queue only when the consumer thread exhausts
it AND the next get_view() call triggers a swap.  By that point all prior
synchronous backend.write() calls from this thread are complete and no live
memoryview slices reference the buffer.  fill_chunk can then safely overwrite
it on the next producer cycle.

For 15–30 GB/s storage systems
-------------------------------
  • 256 MB buffers reduce buffer-swap overhead (one swap per ~256 MB consumed).
  • Default producers = max(2, cpu_count // 2):
      12-core machine → 6 producers → 6 × ~4 GB/s ≈ 24 GB/s generation
      generation runs ahead of any foreseeable storage device.
  • Each producer has its own Generator — no shared PRNG state, no contention.

Memory budget (defaults, 12-core machine)
------------------------------------------
  total_buffers = num_producers + prefetch_depth + 4 (consumer headroom)
               = 6 + 8 + 4 = 18 buffers × 256 MB = 4.5 GB pre-allocated

  Tune with --prefetch-depth (fewer buffers) or a smaller buffer_size_mb.

Safety contract
---------------
This pool is safe for SYNCHRONOUS writes only:
  backend.write(key, view) must complete and view must go out of scope before
  the consumer thread calls get_view() again.

  CPython's reference counting ensures the memoryview slice is freed
  immediately when the caller's local variable goes out of scope.
  The buffer is only returned to empty_queue after ALL such views are freed.

NOT safe for async writes where the caller stores views for deferred use.
"""

import logging
import os
import queue
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# Default 256 MB — large enough to amortise buffer-swap overhead even at
# 30 GB/s storage (one swap per ~8 ms) while keeping per-consumer RAM bounded.
DEFAULT_BUFFER_SIZE_MB: int = 256

# Backward-compat alias for code that imported DEFAULT_BLOCK_SIZE_MB
DEFAULT_BLOCK_SIZE_MB: int = DEFAULT_BUFFER_SIZE_MB

DEFAULT_PREFETCH_DEPTH: int = 8   # 8 × 256 MB = 2 GB in the ready queue


def _default_num_producers() -> int:
    """
    Half the logical CPUs, minimum 2.

    fill_chunk releases the GIL and runs via Rayon.  Each Python thread drives
    an independent Rayon fill at ~3.85–5 GB/s (section 1 of
    bench_generation_speeds.py).  Half the cores gives:
      4-core  → 2 producers → ~8 GB/s
      8-core  → 4 producers → ~16 GB/s
      12-core → 6 producers → ~24 GB/s
      32-core → 16 producers → 60+ GB/s
    well above any single storage namespace at any drive speed.
    """
    return max(2, (os.cpu_count() or 4) // 2)


class DataGeneratorPool:
    """
    Background producer pool that keeps 256 MB bytearrays pre-filled and
    ready.  Consumers receive zero-copy memoryview slices via get_view().

    No data is ever copied: fill_chunk writes directly into pre-allocated
    bytearrays; get_view() returns memoryview[offset:offset+size] — a pointer
    into the current buffer.

    Thread safety
    -------------
    Each consumer thread maintains its own (buf, view, offset) state in
    threading.local.  get_view() holds NO locks in the common case.
    The only shared state is the thread-safe ready/empty queues.
    """

    def __init__(
        self,
        buffer_size_mb: int = DEFAULT_BUFFER_SIZE_MB,
        prefetch_depth: int = DEFAULT_PREFETCH_DEPTH,
        num_producers: Optional[int] = None,
        # Legacy kwarg — mapped to buffer_size_mb for backward compatibility
        block_size_mb: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        buffer_size_mb :
            Size of each pre-allocated buffer in MB.  256 MB is the default.
        prefetch_depth :
            Number of fully-generated buffers to keep in the ready queue.
            RAM: prefetch_depth × buffer_size_mb MB.
        num_producers :
            Background fill threads.  Default: max(2, cpu_count // 2).
            Increase for storage systems above ~20 GB/s.
        block_size_mb :
            Deprecated alias for buffer_size_mb.
        """
        if block_size_mb is not None and buffer_size_mb == DEFAULT_BUFFER_SIZE_MB:
            buffer_size_mb = block_size_mb

        self._buf_size: int = buffer_size_mb * 1024 * 1024
        self._buf_size_mb: int = buffer_size_mb

        n = num_producers if num_producers is not None else _default_num_producers()
        self._num_producers: int = max(1, n)

        # Two queues: empty buffers waiting to be filled, filled buffers ready.
        self._empty: queue.Queue = queue.Queue()
        self._ready: queue.Queue = queue.Queue(maxsize=prefetch_depth)
        self._stop: threading.Event = threading.Event()

        # Pre-allocate ALL buffers once at startup.  No allocation in hot path.
        # +4 headroom covers typical concurrent consumer threads.
        self._total_buffers: int = self._num_producers + prefetch_depth + 4
        for _ in range(self._total_buffers):
            self._empty.put(bytearray(self._buf_size))

        # Thread-local consumer state: each thread has its own buffer + cursor.
        self._tls: threading.local = threading.local()

        self._threads = [
            threading.Thread(
                target=self._producer_loop,
                name=f"dgen-producer-{i}",
                daemon=True,
            )
            for i in range(self._num_producers)
        ]
        self._started: bool = False

        total_ram_mb = self._total_buffers * buffer_size_mb
        logger.info(
            f"DataGeneratorPool: {self._num_producers} producer(s), "
            f"{prefetch_depth}× {buffer_size_mb} MB ready queue, "
            f"{self._total_buffers} buffers = {total_ram_mb} MB RAM pre-allocated"
        )

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def start(self) -> "DataGeneratorPool":
        """Start producer threads.  Returns self for method chaining.  Idempotent."""
        try:
            import dgen_py  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "dgen-py is required for DataGeneratorPool. "
                "Install it with: pip install dgen-py"
            ) from exc

        if not self._started:
            for t in self._threads:
                t.start()
            self._started = True
            logger.info(
                f"DataGeneratorPool: {self._num_producers} producer thread(s) started, "
                f"{self._buf_size_mb} MB buffers"
            )
        return self

    def stop(self) -> None:
        """Signal producer threads to stop and wait briefly for exit."""
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2.0)
        self._started = False

    @property
    def is_alive(self) -> bool:
        """True if at least one producer thread is still running."""
        return self._started and any(t.is_alive() for t in self._threads)

    # -------------------------------------------------------------------------
    # Consumer API — zero-copy
    # -------------------------------------------------------------------------

    def get_view(self, size: int) -> memoryview:
        """
        Return memoryview[offset : offset+size] from the current pre-filled buffer.

        No data is EVER copied.  Pure pointer arithmetic into a pre-allocated
        256 MB bytearray.

        The underlying buffer is NOT recycled until:
          1. This thread exhausts the buffer (offset + next_size > buf_size).
          2. The NEXT call to get_view() triggers _swap_buffer().
          3. By CPython refcounting, all prior slices are freed (write done).

        Parameters
        ----------
        size : Number of bytes required.

        Returns
        -------
        memoryview — zero-copy view of exactly ``size`` bytes.
                     Valid until caller releases it after synchronous write.

        Notes
        -----
        Entries larger than buffer_size_mb are handled by _generate_oversized()
        (fills a fresh bytearray inline).  Rare for typical KV cache entries.
        """
        if size <= 0:
            return memoryview(b"")

        if size > self._buf_size:
            return self._generate_oversized(size)

        tls = self._tls

        if not hasattr(tls, 'buf') or tls.offset + size > self._buf_size:
            self._swap_buffer()

        start = tls.offset
        tls.offset += size
        return tls.view[start : start + size]   # zero-copy pointer slice

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _swap_buffer(self) -> None:
        """
        Return the exhausted buffer to empty_queue and fetch a filled buffer.

        Called only at buffer exhaustion.  At this point all prior synchronous
        writes from this thread are complete; prior memoryview slices are freed
        by CPython refcounting before this thread can call get_view() again.

        We release the parent memoryview before returning the buffer so that
        ob_exports == 0 when the producer calls fill_chunk on the next cycle.
        """
        tls = self._tls

        if hasattr(tls, 'buf'):
            tls.view.release()        # drops ob_exports back to 0
            self._empty.put(tls.buf)  # producer can now safely fill_chunk into it

        buf = self._ready.get()       # blocks until a producer fills one
        tls.buf = buf
        tls.view = memoryview(buf)
        tls.offset = 0

    def _generate_oversized(self, size: int) -> memoryview:
        """Fallback for entries larger than buffer_size_mb.  Rare."""
        try:
            import dgen_py
            buf = bytearray(size)
            gen = dgen_py.Generator(size, compress_ratio=1.0)
            gen.fill_chunk(buf)
            return memoryview(buf)
        except Exception as exc:
            logger.warning(
                f"DataGeneratorPool: oversized fallback failed ({exc}), returning zeros"
            )
            return memoryview(bytearray(size))

    # -------------------------------------------------------------------------
    # Producer loop
    # -------------------------------------------------------------------------

    def _producer_loop(self) -> None:
        """
        Background loop: get empty buffer → fill_chunk (GIL-free) → ready queue.

        Each thread has its own Generator — no shared PRNG state, no contention
        between producers.  Generator size is 16 TB (effectively infinite).

        No data is EVER copied: fill_chunk writes directly into the bytearray.
        """
        try:
            import dgen_py
        except ImportError:
            logger.error(
                f"{threading.current_thread().name}: dgen-py not available, exiting"
            )
            return

        gen = dgen_py.Generator(
            size=1 << 44,        # 16 TB — never exhausts in practice
            compress_ratio=1.0,  # incompressible (correct for storage benchmarks)
            numa_mode="auto",
        )
        blocks = 0

        while not self._stop.is_set():
            # ── get an empty buffer ──────────────────────────────────────────
            try:
                buf = self._empty.get(timeout=0.1)
            except queue.Empty:
                continue

            # ── fill it in-place (GIL-free Rayon, zero copy) ────────────────
            try:
                gen.fill_chunk(buf)
                if gen.is_complete():
                    gen.reset()
            except Exception as exc:
                logger.error(
                    f"{threading.current_thread().name}: fill_chunk failed: {exc}"
                )
                self._empty.put(buf)
                continue

            # ── enqueue for consumer ─────────────────────────────────────────
            while not self._stop.is_set():
                try:
                    self._ready.put(buf, timeout=0.1)
                    break
                except queue.Full:
                    continue

            blocks += 1
            if blocks % 10 == 0:
                logger.debug(
                    f"{threading.current_thread().name}: {blocks} buffers "
                    f"({blocks * self._buf_size / 1e9:.1f} GB), "
                    f"ready={self._ready.qsize()}/{self._ready.maxsize}"
                )

        logger.info(
            f"{threading.current_thread().name}: stopped after {blocks} buffers "
            f"({blocks * self._buf_size / 1e9:.1f} GB total)"
        )
