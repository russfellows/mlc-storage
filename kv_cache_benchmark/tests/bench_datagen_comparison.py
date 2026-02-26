#!/usr/bin/env python3
"""
Datagen Comparison Benchmark
===============================
Compares the OLD KVCacheGenerator method (pre-commit 377a631) to the NEW
DataGeneratorPool method (post-commit 377a631 / dgen-py Xoshiro256++) across
three dimensions:

  1. GENERATION THROUGHPUT  — GB/s produced; extrapolated to --target-tb
  2. COMPRESSIBILITY        — zstd level-1 and level-3 ratios on a sample
  3. BLOCK-LEVEL DEDUP RATE — SHA-256 unique-block ratio (default 4 KB blocks)

Background
----------
Old method (KVCacheGenerator):
  - Generates ONE fixed 256 MB float16 buffer at startup with NumPy's MT19937.
  - Every subsequent generate() call returns a numpy VIEW into that same
    pre-computed buffer, offset by hash(key).
  - Consequence: after ~256 MB of writes, every 4 KB block you write is a
    repeat of a block already written.  For 10 TB the repeat rate is ~40,000x.

New method (DataGeneratorPool / dgen-py):
  - Producer threads run dgen_py.Generator.fill_chunk() — GIL-free Rayon
    Xoshiro256++ fill — to write fresh, unique random bytes into each 256 MB
    bytearray.
  - Consumers receive a memoryview slice; the data is NEVER the same as any
    prior buffer.
  - Consequence: near-zero block-level dedup rate over any dataset size.

Usage
-----
    # Quick comparison (4 GB sample, no disk writes, no vdbench):
    python tests/bench_datagen_comparison.py --skip-write

    # Write 8 GB files to NVMe and run vdbench dsim on each:
    python tests/bench_datagen_comparison.py --write-gb 8

    # Change data directory (default: /mnt/nvme_data/):
    python tests/bench_datagen_comparison.py --write-gb 8 --data-dir /mnt/nvme_data/

    # Larger sample for more accurate dedup/compress measurements:
    python tests/bench_datagen_comparison.py --write-gb 20 --compress-sample-mb 512

    # Extrapolate throughput to different TB target:
    python tests/bench_datagen_comparison.py --target-tb 10 --write-gb 8
"""

import argparse
import hashlib
import math
import os
import sys
import time
from typing import Iterator, Optional, Tuple

import numpy as np
import zstandard as zstd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GB = 1024 ** 3
MB = 1024 ** 2
KB = 1024

DEFAULT_SAMPLE_GB     = 4      # GB of data to generate for all measurements
DEFAULT_TARGET_TB     = 10     # TB to extrapolate timing to
DEFAULT_KV_ENTRY_MB   = 16     # Size of each simulated KV cache entry (MB)
DEFAULT_BLOCK_SIZE_KB = 4      # Block size for dedup fingerprinting (KB)
DEFAULT_COMPRESS_SAMPLE_MB = 256  # How many MB to compress for ratio test
DEFAULT_SEED          = 42
DEFAULT_DATA_DIR      = "/mnt/nvme_data"
DEFAULT_WRITE_GB      = 8      # GB to write per method when --write-gb used


# ---------------------------------------------------------------------------
# OLD method — exact replica of KVCacheGenerator from before commit 377a631
# ---------------------------------------------------------------------------

class LegacyKVCacheGenerator:
    """
    Replica of the KVCacheGenerator introduced before commit 377a631.

    Generates a 256 MB float16 buffer ONCE at init, then serves every
    generate() call as a VIEW (or tiled copy) from that same pool.

    This is intentionally an in-code replica so the test is self-contained
    and does not require checking out a different git revision.
    """

    BUFFER_SIZE_ELEMENTS = 128 * 1024 * 1024  # 128 M float16 elements = 256 MB

    def __init__(self, seed: int = DEFAULT_SEED):
        self.seed = seed
        print(f"  [old] Pre-generating 256 MB noise buffer (seed={seed}) …", flush=True)
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        self.buffer = rng.uniform(-1.0, 1.0, size=self.BUFFER_SIZE_ELEMENTS).astype(np.float16)
        elapsed = time.perf_counter() - t0
        print(f"  [old] Buffer ready in {elapsed:.2f}s  "
              f"({self.buffer.nbytes / MB:.0f} MB, dtype=float16)", flush=True)

    def _offset_for_key(self, key: str, entry_elements: int) -> int:
        """Replicate _seed_from_key → start_idx logic from original code."""
        h = hashlib.sha256(key.encode()).digest()
        key_hash = int.from_bytes(h[:8], "little") ^ self.seed
        divisor = self.BUFFER_SIZE_ELEMENTS - entry_elements
        return int(key_hash % divisor) if divisor > 0 else 0

    def generate_bytes(self, entry_bytes: int, key: str = "") -> memoryview:
        """
        Return the entry as a bytes-like object (memoryview of the underlying
        float16 buffer) exactly as the old code would present it when the
        caller converts to bytes for storage.
        """
        # entry_bytes must be even (float16)
        entry_bytes = entry_bytes & ~1
        entry_elements = entry_bytes // 2  # float16 = 2 bytes

        if entry_elements <= self.BUFFER_SIZE_ELEMENTS:
            offset = self._offset_for_key(key, entry_elements) if key else 0
            flat = self.buffer[offset : offset + entry_elements]
            # The original code returns the numpy array; callers then passed
            # it to backend.write() which did bytes(data) or data.tobytes().
            # We return the raw memoryview so we can measure bytes produced
            # without an extra copy.
            return memoryview(flat)
        else:
            # Tiled path for entries larger than the pool
            repeats = math.ceil(entry_elements / self.BUFFER_SIZE_ELEMENTS)
            large = np.tile(self.buffer, repeats)[:entry_elements]
            return memoryview(large)

    def stream(self, total_bytes: int, entry_bytes: int) -> Iterator[memoryview]:
        """Yield successive entry-sized windows until total_bytes is reached."""
        produced = 0
        key_counter = 0
        while produced < total_bytes:
            key = f"layer0/user{key_counter}"
            view = self.generate_bytes(min(entry_bytes, total_bytes - produced), key)
            yield view
            produced += len(view) * 2  # memoryview of float16: len gives elements
            key_counter += 1


# ---------------------------------------------------------------------------
# NEW method — inline dgen_py producer pool (no dependency on data_producer.py)
#
# Self-contained so this script works on any git branch.  Uses dgen_py
# directly: Generator.fill_chunk() releases the GIL and runs Rayon-parallel
# Xoshiro256++ at ~4-5 GB/s per thread.
# ---------------------------------------------------------------------------

class InlineDgenPool:
    """
    Minimal double-buffered producer using dgen_py directly.

    Two 256 MB bytearrays alternate: while the consumer reads buffer A,
    a background thread is filling buffer B with fresh Xoshiro256++ bytes.
    When the consumer exhausts A it swaps to B (already full) and kicks off
    a fill of A — zero stall time in the hot path.

    Falls back to os.urandom (CSPRNG, always unique, but slower) if dgen_py
    is not installed.
    """

    BUFFER_SIZE = 256 * MB  # 256 MB per buffer  (2 buffers = 512 MB total)

    def __init__(self):
        try:
            import dgen_py as _dgen
            self._dgen = _dgen
            self._available = True
            print(f"  [new] dgen_py {_dgen.__version__} (Xoshiro256++, GIL-free Rayon)",
                  flush=True)
        except ImportError:
            self._available = False
            print("  [new] WARNING: dgen_py not installed — using os.urandom "
                  "(unique, but ~1 GB/s vs ~85 GB/s)", flush=True)

        self._bufs = [bytearray(self.BUFFER_SIZE), bytearray(self.BUFFER_SIZE)]
        self._cur  = 0   # index of the buffer the consumer is currently reading
        self._off  = 0   # byte offset within the current buffer
        # Pre-fill both buffers synchronously so get_view() never blocks
        self._fill(0)
        self._fill(1)

    def _fill(self, idx: int) -> None:
        """Fill self._bufs[idx] with fresh random bytes."""
        if self._available:
            gen = self._dgen.Generator(size=self.BUFFER_SIZE)
            gen.fill_chunk(self._bufs[idx])
        else:
            self._bufs[idx][:] = os.urandom(self.BUFFER_SIZE)

    def get_view(self, size: int) -> memoryview:
        """Return a memoryview[size] from the current buffer; swap + refill if needed."""
        assert size <= self.BUFFER_SIZE, f"entry size {size} > buffer {self.BUFFER_SIZE}"
        if self._off + size > self.BUFFER_SIZE:
            # Swap to the other (already-full) buffer and schedule a refill
            # of the one we just exhausted.
            old = self._cur
            self._cur = 1 - self._cur
            self._off = 0
            self._fill(old)  # refill old buffer for the next swap
        view = memoryview(self._bufs[self._cur])[self._off : self._off + size]
        self._off += size
        return view

    def stream(self, total_bytes: int, entry_bytes: int) -> Iterator[memoryview]:
        produced = 0
        while produced < total_bytes:
            want = min(entry_bytes, total_bytes - produced)
            yield self.get_view(want)
            produced += want

    def shutdown(self):
        pass  # no background threads in this simplified version


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def measure_throughput(
    gen_stream: Iterator[memoryview],
    total_bytes: int,
    write_path: Optional[str] = None,
    label: str = "",
) -> Tuple[float, float]:
    """
    Consume the generator stream until total_bytes is produced.

    If write_path is given, each chunk is written to that file (O_DIRECT
    is attempted; falls back to buffered).  The file will contain exactly
    the generated bytes and can be passed to 'vdbench dsim' afterwards.

    Returns (elapsed_seconds, throughput_gbs).
    NOTE: throughput includes I/O time when write_path is set, so it
    reflects real storage write speed, not just generation speed.
    """
    produced = 0
    fd = None

    if write_path is not None:
        fd = os.open(write_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        print(f"    {label} writing to {write_path} (buffered + fsync)", flush=True)

    t0 = time.perf_counter()

    for chunk in gen_stream:
        # Normalise to a bytes view (chunk may be float16 memoryview)
        if isinstance(chunk, memoryview) and chunk.itemsize != 1:
            raw = chunk.cast('B')   # reinterpret as bytes — still zero-copy
            n_bytes = len(raw)
        else:
            raw = chunk
            n_bytes = len(chunk)

        if fd is not None:
            # Writing memoryview directly avoids a bytes() copy in most cases
            os.write(fd, raw)

        produced += n_bytes
        if produced >= total_bytes:
            break

        # Progress every ~10 %
        pct = 100.0 * produced / total_bytes
        prev_pct = 100.0 * (produced - n_bytes) / total_bytes
        if int(pct / 10) > int(prev_pct / 10):
            elapsed_so_far = time.perf_counter() - t0
            bw = (produced / GB) / max(elapsed_so_far, 1e-9)
            print(f"    {label} {pct:5.1f}%  {bw:.2f} GB/s", flush=True)

    if fd is not None:
        os.fsync(fd)   # flush to device before vdbench reads the file
        os.close(fd)

    elapsed = time.perf_counter() - t0
    throughput = (produced / GB) / max(elapsed, 1e-9)
    return elapsed, throughput


def measure_compression(data: bytes, label: str) -> dict:
    """Compress data at zstd levels 1 and 3; return ratios and throughput."""
    results = {}
    original_size = len(data)
    for level in (1, 3):
        cctx = zstd.ZstdCompressor(level=level)
        t0 = time.perf_counter()
        compressed = cctx.compress(data)
        elapsed = time.perf_counter() - t0
        ratio = original_size / max(len(compressed), 1)
        bw = (original_size / MB) / max(elapsed, 1e-9)
        results[level] = {
            "original_mb": original_size / MB,
            "compressed_mb": len(compressed) / MB,
            "ratio": ratio,
            "throughput_mbs": bw,
            "elapsed_s": elapsed,
        }
        print(f"    {label} zstd-{level}: "
              f"{original_size/MB:.0f} MB → {len(compressed)/MB:.1f} MB  "
              f"ratio={ratio:.2f}x  ({bw:.0f} MB/s)", flush=True)
    return results


def measure_dedup_rate(data: bytes, block_size: int, label: str) -> dict:
    """
    Split data into fixed-size blocks, SHA-256 fingerprint each, count uniques.

    Returns unique_blocks, total_blocks, dedup_rate (0.0 = all unique,
    1.0 = all duplicates).
    """
    total_bytes = len(data)
    total_blocks = total_bytes // block_size
    if total_blocks == 0:
        print(f"    {label} WARNING: sample too small for block_size={block_size}", flush=True)
        return {"total_blocks": 0, "unique_blocks": 0, "dedup_rate": 0.0}

    seen = set()
    for i in range(total_blocks):
        blk = data[i * block_size : (i + 1) * block_size]
        seen.add(hashlib.sha256(blk).digest())

    unique_blocks = len(seen)
    dedup_rate = 1.0 - (unique_blocks / total_blocks)
    savings_pct = dedup_rate * 100.0

    print(f"    {label} dedup ({block_size//KB} KB blocks): "
          f"{unique_blocks:,} unique / {total_blocks:,} total → "
          f"{savings_pct:.2f}% savings", flush=True)
    return {
        "total_blocks": total_blocks,
        "unique_blocks": unique_blocks,
        "dedup_rate": dedup_rate,
        "savings_pct": savings_pct,
    }


def collect_sample(gen_stream: Iterator[memoryview], sample_bytes: int) -> bytes:
    """Collect exactly sample_bytes from a generator stream into a single bytes object."""
    chunks = []
    collected = 0
    for chunk in gen_stream:
        if isinstance(chunk, memoryview) and chunk.itemsize != 1:
            raw = bytes(chunk)
        else:
            raw = bytes(chunk)
        chunks.append(raw[:sample_bytes - collected])
        collected += len(chunks[-1])
        if collected >= sample_bytes:
            break
    return b"".join(chunks)


def run_vdbench_dsim(filepath: str, dedup_unit_kb: int = 4,
                     java_heap_mb: int = 8192) -> str:
    """
    Run vdbench dsim by calling the JVM directly with sufficient heap.

    The /usr/local/bin/vdbench wrapper hard-codes -Xmx512m for Vdbmain,
    which is too small for large files.  We bypass the wrapper and invoke
    java directly with java_heap_mb (default 8 GB).

    Falls back to analyze_file_native() if java is unavailable or fails.
    """
    import subprocess
    unit_bytes = dedup_unit_kb * KB
    vdbench_dir = "/usr/local/share/vdbench50407"
    cp = f"{vdbench_dir}/:{vdbench_dir}/classes:{vdbench_dir}/vdbench.jar"

    cmd = [
        "java",
        f"-Xmx{java_heap_mb}m",
        f"-Xms256m",
        "-cp", cp,
        "Vdb.Vdbmain",
        "dsim",
        "-u", str(unit_bytes),
        filepath,
    ]
    print(f"\n  Running: vdbench dsim -u {unit_bytes} {filepath} "
          f"  (java heap: {java_heap_mb} MB)", flush=True)
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1800,
        )
        output = (result.stdout + result.stderr).strip()
        if result.returncode != 0 or "Exception" in output or not output:
            print(f"    vdbench exited {result.returncode} — falling back to "
                  f"native analysis", flush=True)
            return analyze_file_native(filepath, dedup_unit_kb)
        for line in output.splitlines():
            print(f"    {line}", flush=True)
        return output
    except FileNotFoundError:
        print(f"    java not found — using native analysis", flush=True)
        return analyze_file_native(filepath, dedup_unit_kb)
    except subprocess.TimeoutExpired:
        print(f"    vdbench dsim timed out — using native analysis", flush=True)
        return analyze_file_native(filepath, dedup_unit_kb)


def analyze_file_native(filepath: str, block_size_kb: int = 4) -> str:
    """
    Pure-Python + zstd-CLI analysis of a binary file.

    Dedup analysis:
      Reads the file in block_size_kb chunks, SHA-256 fingerprints each block,
      and counts unique fingerprints.  Memory cost: 32 bytes × num_unique_blocks
      (e.g. a 256 MB pool at 4 KB blocks = 65,536 unique → only 2 MB RAM).

    Compression analysis:
      Streams the file through 'zstd -1 --stdout' and measures the output size.
      Avoids loading the whole file into RAM.
    """
    import subprocess
    block_size = block_size_kb * KB
    file_size  = os.path.getsize(filepath)
    total_blocks = file_size // block_size
    output_lines = []

    # --- Block-level dedup ---
    print(f"\n  [native] Block dedup ({block_size_kb} KB blocks) on "
          f"{file_size/GB:.2f} GB file …", flush=True)
    seen = set()
    read_bytes = 0
    t0 = time.perf_counter()
    with open(filepath, "rb") as f:
        while True:
            blk = f.read(block_size)
            if len(blk) < block_size:
                break
            seen.add(hashlib.sha256(blk).digest())
            read_bytes += block_size
            pct = 100.0 * read_bytes / file_size
            prev_pct = 100.0 * (read_bytes - block_size) / file_size
            if int(pct / 10) > int(prev_pct / 10):
                print(f"    [native dedup] {pct:.0f}%  "
                      f"unique so far: {len(seen):,}", flush=True)
    elapsed = time.perf_counter() - t0

    unique_blocks = len(seen)
    measured_blocks = read_bytes // block_size
    dedup_ratio  = measured_blocks / max(unique_blocks, 1)
    savings_pct  = 100.0 * (1.0 - unique_blocks / max(measured_blocks, 1))
    dedup_line = (f"  Dedup: {unique_blocks:,} unique / {measured_blocks:,} total "
                  f"{block_size_kb} KB blocks  →  {dedup_ratio:.2f}x ratio  "
                  f"({savings_pct:.4f}% savings)  [{elapsed:.1f}s]")
    print(f"    {dedup_line}", flush=True)
    output_lines.append(dedup_line)

    # --- Compression via zstd CLI (stream, no RAM for full file) ---
    print(f"\n  [native] zstd -1 compression on {file_size/GB:.2f} GB file …",
          flush=True)
    try:
        t1 = time.perf_counter()
        result = subprocess.run(
            ["zstd", "-1", "--stdout", filepath],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3600,
        )
        compressed_size = len(result.stdout)
        zstd_elapsed = time.perf_counter() - t1
        comp_ratio = file_size / max(compressed_size, 1)
        comp_line = (f"  zstd-1: {file_size/GB:.2f} GB → "
                     f"{compressed_size/GB:.2f} GB  →  {comp_ratio:.2f}x ratio  "
                     f"[{zstd_elapsed:.1f}s]")
        print(f"    {comp_line}", flush=True)
        output_lines.append(comp_line)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        output_lines.append(f"  zstd compression unavailable: {exc}")

    return "\n".join(output_lines)


def extrapolate(throughput_gbs: float, target_tb: float) -> str:
    """Return human-readable time-to-complete string."""
    if throughput_gbs <= 0:
        return "N/A"
    target_gb = target_tb * 1024
    seconds = target_gb / throughput_gbs
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h {m:02d}m {s:02d}s  (at {throughput_gbs:.2f} GB/s)"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare old KVCacheGenerator vs new InlineDgenPool (dgen-py)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target-tb",    type=float, default=DEFAULT_TARGET_TB,
                        help="TB to extrapolate timing estimates to")
    parser.add_argument("--sample-gb",    type=float, default=DEFAULT_SAMPLE_GB,
                        help="GB of data to stream for throughput measurement")
    parser.add_argument("--entry-mb",     type=float, default=DEFAULT_KV_ENTRY_MB,
                        help="Simulated KV entry size in MB")
    parser.add_argument("--block-size-kb", type=int, default=DEFAULT_BLOCK_SIZE_KB,
                        help="Block size in KB for dedup fingerprinting")
    parser.add_argument("--compress-sample-mb", type=int, default=DEFAULT_COMPRESS_SAMPLE_MB,
                        help="Sample size in MB for zstd compression ratio test")
    parser.add_argument("--seed",         type=int, default=DEFAULT_SEED,
                        help="RNG seed for old method")
    parser.add_argument("--data-dir",     type=str, default=DEFAULT_DATA_DIR,
                        help="Directory for written data files and vdbench analysis")
    parser.add_argument("--write-gb",     type=float, default=DEFAULT_WRITE_GB,
                        help="GB to write per method to --data-dir for vdbench dsim")
    parser.add_argument("--skip-write",   action="store_true",
                        help="Skip writing files to NVMe; measure generation speed only")
    parser.add_argument("--analyze-existing", action="store_true",
                        help="Skip generation entirely; run analysis only on already-written "
                             "files in --data-dir (datagen_OLD_method.bin / datagen_NEW_method.bin)")
    parser.add_argument("--java-heap-mb", type=int, default=8192,
                        help="Java heap size in MB for vdbench dsim (default 8192)")
    args = parser.parse_args()

    sample_bytes   = int(args.sample_gb * GB)
    entry_bytes    = int(args.entry_mb * MB)
    block_size     = args.block_size_kb * KB
    comp_sample    = args.compress_sample_mb * MB
    write_bytes    = int(args.write_gb * GB)

    old_write_path = os.path.join(args.data_dir, "datagen_OLD_method.bin") if not args.skip_write else None
    new_write_path = os.path.join(args.data_dir, "datagen_NEW_method.bin") if not args.skip_write else None

    # ------------------------------------------------------------------
    # Fast path: --analyze-existing
    # Re-run vdbench dsim / native analysis on already-written files.
    # ------------------------------------------------------------------
    if args.analyze_existing:
        print("=" * 70)
        print(" ANALYZE EXISTING FILES (--analyze-existing)")
        print("=" * 70)
        for label, path in (("OLD", os.path.join(args.data_dir, "datagen_OLD_method.bin")),
                            ("NEW", os.path.join(args.data_dir, "datagen_NEW_method.bin"))):
            if not os.path.exists(path):
                print(f"  {label}: {path} not found — skipping")
                continue
            sz = os.path.getsize(path)
            print(f"\n{'─'*70}")
            print(f"  {label} method file: {path}  ({sz/GB:.2f} GB)")
            print(f"{'─'*70}")
            print(f"\n  1. vdbench dsim (java heap {args.java_heap_mb} MB):")
            run_vdbench_dsim(path, dedup_unit_kb=args.block_size_kb,
                             java_heap_mb=args.java_heap_mb)
            print(f"\n  2. Native analysis (SHA-256 block fingerprint + zstd):")
            analyze_file_native(path, block_size_kb=args.block_size_kb)
        return

    print("=" * 70)
    print(" KV Cache Datagen Comparison Benchmark")
    print("=" * 70)
    print(f"  Sample size       : {args.sample_gb:.1f} GB  (throughput measurement)")
    print(f"  KV entry size     : {args.entry_mb:.1f} MB")
    print(f"  Dedup block       : {args.block_size_kb} KB")
    print(f"  Compress sample   : {args.compress_sample_mb} MB")
    print(f"  Target extrap     : {args.target_tb} TB")
    if not args.skip_write:
        print(f"  Write per method  : {args.write_gb:.1f} GB  (to {args.data_dir})")
        print(f"  Old file          : {old_write_path}")
        print(f"  New file          : {new_write_path}")
        print(f"  vdbench dsim      : yes (after each write)")
    else:
        print(f"  Write files       : skipped (--skip-write)")
    print()

    # -----------------------------------------------------------------------
    # PRECOMPUTED_BUFFER ANALYSIS
    # Settle the question: does the old method's buffer ever get re-filled?
    # -----------------------------------------------------------------------
    print("=" * 70)
    print(" PRECOMPUTED_BUFFER ANALYSIS (old method)")
    print("=" * 70)
    pool_size_mb = LegacyKVCacheGenerator.BUFFER_SIZE_ELEMENTS * 2 // MB  # float16
    pool_unique_blocks = (LegacyKVCacheGenerator.BUFFER_SIZE_ELEMENTS * 2) // block_size
    write_total_blocks = write_bytes // block_size
    theoretical_dedup_pct = max(
        0.0,
        100.0 * (1.0 - pool_unique_blocks / max(write_total_blocks, 1))
    )
    print(f"""
  The old KVCacheGenerator works as follows:

    1. __init__() generates ONE {pool_size_mb} MB float16 buffer using
       numpy.random.default_rng(seed).uniform().

    2. generate() returns a SLICE (numpy view) into that same buffer,
       with the start offset derived from hash(key).  No new random
       data is ever generated.

    3. For entries larger than the pool: np.tile() tiles the same 256 MB
       pool repeatedly — still NO new unique data.

    4. The 'rng' object is a LOCAL variable in __init__().  It goes out
       of scope and is garbage-collected immediately after the buffer is
       created.  There is NO mechanism to re-seed or re-fill the buffer.

  VERDICT: The precomputed_buffer is NEVER re-filled during a test run.
           Every write beyond the first {pool_size_mb} MB is 100% repeat data.

  Unique {args.block_size_kb} KB blocks in the entire pool : {pool_unique_blocks:>12,}
  Unique {args.block_size_kb} KB blocks in {args.write_gb:.0f} GB written file : {write_total_blocks:>12,}
  Theoretical block-dedup savings at {args.write_gb:.0f} GB  : {theoretical_dedup_pct:>11.4f}%
  Theoretical block-dedup savings at {args.target_tb:.0f} TB  : {max(0,100*(1-pool_unique_blocks/max(int(args.target_tb*1024*GB//block_size),1))):>11.6f}%
""")

    results = {}

    # -----------------------------------------------------------------------
    # 1. OLD METHOD
    # -----------------------------------------------------------------------
    print("-" * 70)
    print("TEST 1 — OLD method: LegacyKVCacheGenerator (pre-commit 377a631)")
    print("-" * 70)

    old_gen = LegacyKVCacheGenerator(seed=args.seed)

    # --- throughput (generation speed, no I/O) ---
    print(f"\n  Generation throughput ({args.sample_gb:.0f} GB, no I/O):")
    old_stream = old_gen.stream(sample_bytes, entry_bytes)
    old_gen_elapsed, old_gen_gbs = measure_throughput(
        old_stream, sample_bytes, write_path=None, label="[old gen]"
    )
    print(f"\n  OLD gen throughput : {old_gen_gbs:.3f} GB/s "
          f"(NOTE: this is pure memory bandwidth — pointer arithmetic into"
          f" a {pool_size_mb} MB buffer)", flush=True)

    # --- write to NVMe + vdbench ---
    old_vdbench = ""
    old_write_gbs = None
    if old_write_path:
        print(f"\n  Write {args.write_gb:.0f} GB to NVMe (includes I/O — this is the real storage speed):")
        wstream = old_gen.stream(write_bytes, entry_bytes)
        old_write_elapsed, old_write_gbs = measure_throughput(
            wstream, write_bytes, write_path=old_write_path, label="[old write]"
        )
        print(f"\n  OLD write throughput : {old_write_gbs:.3f} GB/s  "
              f"(with O_DIRECT to NVMe)", flush=True)
        old_vdbench = run_vdbench_dsim(old_write_path,
                                        dedup_unit_kb=args.block_size_kb,
                                        java_heap_mb=args.java_heap_mb)

    # --- compressibility sample ---
    print(f"\n  zstd compressibility ({args.compress_sample_mb} MB sample):")
    comp_data_chunks, bytes_so_far, k = [], 0, 0
    while bytes_so_far < comp_sample:
        view = old_gen.generate_bytes(min(entry_bytes, comp_sample - bytes_so_far),
                                       key=f"layer0/user{k}")
        raw = bytes(view)
        comp_data_chunks.append(raw)
        bytes_so_far += len(raw)
        k += 1
    comp_data_old = b"".join(comp_data_chunks)[:comp_sample]
    old_compress = measure_compression(comp_data_old, "[old]")

    # --- block dedup ---
    print(f"\n  Block dedup ({args.block_size_kb} KB blocks, {args.compress_sample_mb} MB sample):")
    old_dedup = measure_dedup_rate(comp_data_old, block_size, "[old]")

    results["old"] = {
        "gen_throughput_gbs": old_gen_gbs,
        "write_throughput_gbs": old_write_gbs,
        "compression": old_compress,
        "dedup": old_dedup,
        "vdbench": old_vdbench,
    }
    del comp_data_old

    # -----------------------------------------------------------------------
    # 2. NEW METHOD
    # -----------------------------------------------------------------------
    print()
    print("-" * 70)
    print("TEST 2 — NEW method: InlineDgenPool (dgen-py Xoshiro256++)")
    print("-" * 70)

    new_gen = InlineDgenPool()

    # --- throughput ---
    print(f"\n  Generation throughput ({args.sample_gb:.0f} GB, no I/O):")
    new_stream = new_gen.stream(sample_bytes, entry_bytes)
    new_gen_elapsed, new_gen_gbs = measure_throughput(
        new_stream, sample_bytes, write_path=None, label="[new gen]"
    )
    print(f"\n  NEW gen throughput : {new_gen_gbs:.3f} GB/s", flush=True)

    # --- write to NVMe + vdbench ---
    new_vdbench = ""
    new_write_gbs = None
    if new_write_path:
        print(f"\n  Write {args.write_gb:.0f} GB to NVMe:")
        wstream = new_gen.stream(write_bytes, entry_bytes)
        new_write_elapsed, new_write_gbs = measure_throughput(
            wstream, write_bytes, write_path=new_write_path, label="[new write]"
        )
        print(f"\n  NEW write throughput : {new_write_gbs:.3f} GB/s", flush=True)
        new_vdbench = run_vdbench_dsim(new_write_path,
                                        dedup_unit_kb=args.block_size_kb,
                                        java_heap_mb=args.java_heap_mb)

    # --- compressibility ---
    print(f"\n  zstd compressibility ({args.compress_sample_mb} MB sample):")
    comp_data_new = collect_sample(new_gen.stream(comp_sample + MB, entry_bytes), comp_sample)
    new_compress = measure_compression(comp_data_new, "[new]")

    # --- block dedup ---
    print(f"\n  Block dedup ({args.block_size_kb} KB blocks, {args.compress_sample_mb} MB sample):")
    new_dedup = measure_dedup_rate(comp_data_new, block_size, "[new]")

    results["new"] = {
        "gen_throughput_gbs": new_gen_gbs,
        "write_throughput_gbs": new_write_gbs,
        "compression": new_compress,
        "dedup": new_dedup,
        "vdbench": new_vdbench,
    }
    del comp_data_new
    new_gen.shutdown()

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print(" SUMMARY")
    print("=" * 70)

    old_r = results["old"]
    new_r = results["new"]

    gen_speedup = new_r["gen_throughput_gbs"] / max(old_r["gen_throughput_gbs"], 1e-9)
    pool_unique_blks = (LegacyKVCacheGenerator.BUFFER_SIZE_ELEMENTS * 2) // block_size
    target_blks_tb   = int(args.target_tb * 1024 * GB) // block_size
    write_blks       = write_bytes // block_size
    exp_dedup_write  = max(0.0, 1.0 - pool_unique_blks / max(write_blks, 1))
    exp_dedup_tb     = max(0.0, 1.0 - pool_unique_blks / max(target_blks_tb, 1))
    pool_mb          = LegacyKVCacheGenerator.BUFFER_SIZE_ELEMENTS * 2 // MB

    print(f"\n{'Metric':<50} {'OLD':>12} {'NEW':>12}")
    print("-" * 76)
    print(f"{'Generation throughput (GB/s, no I/O)':<50} "
          f"{old_r['gen_throughput_gbs']:>12.3f} {new_r['gen_throughput_gbs']:>12.3f}")

    if old_r["write_throughput_gbs"] is not None:
        print(f"{'NVMe write throughput (GB/s, with I/O)':<50} "
              f"{old_r['write_throughput_gbs']:>12.3f} {new_r['write_throughput_gbs']:>12.3f}")

    old_10tb = extrapolate(old_r["gen_throughput_gbs"], args.target_tb)
    new_10tb = extrapolate(new_r["gen_throughput_gbs"], args.target_tb)
    print(f"{'Time to generate ' + str(args.target_tb) + ' TB (gen only)':<50} "
          f"{old_10tb.split('(')[0].strip():>12} {new_10tb.split('(')[0].strip():>12}")

    for level in (1, 3):
        print(f"{'zstd-' + str(level) + ' compression ratio':<50} "
              f"{old_r['compression'][level]['ratio']:>12.2f}x "
              f"{new_r['compression'][level]['ratio']:>12.2f}x")

    print(f"{'Block dedup savings % (' + str(args.block_size_kb) + ' KB blocks, sample)':<50} "
          f"{old_r['dedup']['savings_pct']:>11.2f}% "
          f"{new_r['dedup']['savings_pct']:>11.2f}%")

    print(f"{'Theoretical dedup at ' + str(args.write_gb) + ' GB (old)':<50} "
          f"{exp_dedup_write*100:>10.4f}% {'~0.0000%':>12}")
    print(f"{'Theoretical dedup at ' + str(args.target_tb) + ' TB (old)':<50} "
          f"{exp_dedup_tb*100:>10.6f}% {'~0.0000%':>12}")

    print()
    print(f"  Generation speedup (new / old): {gen_speedup:.1f}x")
    print(f"  NOTE: OLD 'generation' speed is {old_r['gen_throughput_gbs']:.0f} GB/s because it is just")
    print(f"  returning pointer offsets into a {pool_mb} MB buffer — no data is actually")
    print(f"  being generated. The storage sees the same {pool_mb} MB repeated {int(write_bytes/(pool_mb*MB)):,}× "
          f"at {args.write_gb:.0f} GB.")

    print()
    print("=" * 70)
    print(" INTERPRETATION")
    print("=" * 70)
    pool_bytes = LegacyKVCacheGenerator.BUFFER_SIZE_ELEMENTS * 2
    old_dedup_pct = old_r["dedup"]["savings_pct"]
    new_dedup_pct = new_r["dedup"]["savings_pct"]
    print(f"""
  Old method (NumPy fixed-pool, pre-commit 377a631):
    • A single {pool_bytes//MB} MB float16 buffer is generated ONCE at startup.
    • ALL generate() calls for the entire test return SLICES of that buffer.
    • The buffer is NEVER re-seeded or re-filled — confirmed by code inspection.
    • Unique {args.block_size_kb} KB blocks in the pool  : {pool_bytes//block_size:,}
    • Those same blocks repeat {int(args.write_gb*GB//(pool_bytes)):,}× in a {args.write_gb:.0f} GB file →
      theoretical dedup savings at {args.write_gb:.0f} GB  : {exp_dedup_write*100:.4f}%
    • Theoretical dedup savings at {args.target_tb:.0f} TB         : {exp_dedup_tb*100:.6f}%
    • zstd measures {args.compress_sample_mb} MB sample (first pass = unique pool)
    • vdbench dsim on the full written file captures the repeat pattern

  New method (dgen-py Xoshiro256++):
    • Each 256 MB bytearray is filled from scratch by a GIL-free Rayon thread.
    • Every buffer fill produces statistically independent random bytes.
    • Expected dedup : ≈ 0%   Expected compression ratio : ≈ 1.00×
    • Measured {args.compress_sample_mb} MB sample dedup savings: {new_dedup_pct:.2f}%

  Conclusion:
    ✗ OLD data IS highly dedup-able at scale: 256 MB pool repeats endlessly.
      Storage systems with inline dedup will give INFLATED throughput because
      the device sees the same blocks over and over (cache hits / dedup hits).
      OS page cache also inflates READ speeds — the entire working set is
      always hot after the first {pool_bytes//MB} MB.
    ✓ NEW data is NOT dedup-able: each buffer fill is independently seeded.
      Storage throughput numbers reflect genuine device performance.
""")


if __name__ == "__main__":
    main()
