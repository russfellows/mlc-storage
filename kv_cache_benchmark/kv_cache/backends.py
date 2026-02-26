"""
Storage backend classes for KV Cache Benchmark.

Provides the abstract StorageBackend interface and concrete implementations
for GPU VRAM, CPU RAM, and NVMe/SSD storage tiers.
"""

import os
import gc
import time
import logging
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from kv_cache._compat import (
    HAS_TORCH, TORCH_AVAILABLE,
    HAS_CUPY, CUPY_AVAILABLE,
)
from kv_cache.config import cfg

if HAS_TORCH:
    import torch
if HAS_CUPY:
    import cupy as cp

logger = logging.getLogger(__name__)


# ============================================================================
# STORAGE BACKEND CLASSES
# ============================================================================

class StorageBackend:
    """Abstract base class for all storage backends (GPU, CPU, NVMe)."""

    from dataclasses import dataclass

    @dataclass
    class IOTiming:
        """Captures total latency along with host and device components."""
        total: float
        device: float
        host: float

    def write(self, key: str, data) -> 'StorageBackend.IOTiming':
        """Writes data (bytes-like) to the backend and returns latency breakdown."""
        raise NotImplementedError

    def read(self, key: str) -> Tuple[bytes, 'StorageBackend.IOTiming']:
        """Reads data from the backend; returns raw bytes and latency."""
        raise NotImplementedError

    def delete(self, key: str):
        """Deletes data from the backend."""
        raise NotImplementedError

    def clear(self):
        """Clears all data from the backend."""
        raise NotImplementedError


class GPUMemoryBackend(StorageBackend):
    """
    GPU VRAM storage backend.
    Uses PyTorch or CuPy for GPU operations. This is the fastest tier.
    """

    def __init__(self, use_torch=True, on_eviction_callback=None):
        self.on_eviction_callback = on_eviction_callback

        if use_torch and TORCH_AVAILABLE:
            self.backend = 'torch'
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if self.device.type == 'cpu':
                raise RuntimeError("No GPU available for PyTorch backend")
            memory_fraction = cfg('gpu_backend', 'memory_fraction', default=0.8)
            torch.cuda.set_per_process_memory_fraction(memory_fraction, 0)
            torch.cuda.empty_cache()
        elif CUPY_AVAILABLE:
            self.backend = 'cupy'
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
        else:
            raise RuntimeError("No GPU backend (PyTorch or CuPy) available.")

        self.cache = {}
        self.pinned_memory = {}

    def write(self, key: str, data: np.ndarray) -> StorageBackend.IOTiming:
        """Writes a NumPy array from CPU to GPU VRAM."""
        if self.backend == 'torch' and torch.cuda.is_available():
            required_bytes = data.nbytes
            max_eviction_attempts = cfg('gpu_backend', 'max_eviction_attempts', default=100)
            eviction_count = 0
            free_memory_threshold = cfg('gpu_backend', 'free_memory_threshold', default=0.1)
            usable_fraction = 1.0 - free_memory_threshold

            while eviction_count < max_eviction_attempts:
                free_memory = torch.cuda.mem_get_info()[0]
                if required_bytes <= free_memory * usable_fraction:
                    break

                torch.cuda.empty_cache()
                free_memory = torch.cuda.mem_get_info()[0]
                if required_bytes <= free_memory * usable_fraction:
                    break

                if len(self.cache) == 0:
                    logger.warning(
                        f"GPU OOM: Need {required_bytes / 1024**2:.1f}MB, "
                        f"have {free_memory / 1024**2:.1f}MB, no entries to evict"
                    )
                    break

                oldest_key = next(iter(self.cache))
                evicted_tensor = self.cache.pop(oldest_key)
                evicted_size = evicted_tensor.element_size() * evicted_tensor.nelement()
                del evicted_tensor

                if oldest_key in self.pinned_memory:
                    del self.pinned_memory[oldest_key]

                if self.on_eviction_callback:
                    try:
                        self.on_eviction_callback(oldest_key, 'gpu', evicted_size)
                    except Exception as e:
                        logger.warning(f"GPU eviction callback failed for {oldest_key}: {e}")

                eviction_count += 1
                logger.debug(
                    f"GPU eviction #{eviction_count}: evicted {oldest_key} "
                    f"({evicted_size / 1024**2:.1f}MB)"
                )

            if eviction_count > 0:
                torch.cuda.empty_cache()
                logger.debug(f"GPU: evicted {eviction_count} entries to make room for {key}")

        start = time.perf_counter()

        if self.backend == 'torch':
            if key not in self.pinned_memory:
                self.pinned_memory[key] = torch.from_numpy(data).pin_memory()
            gpu_tensor = self.pinned_memory[key].to(self.device, non_blocking=True)
            torch.cuda.synchronize()
            self.cache[key] = gpu_tensor
            del self.pinned_memory[key]
        else:
            self.cache[key] = cp.asarray(data)
            cp.cuda.Stream.null.synchronize()

        total = time.perf_counter() - start
        return StorageBackend.IOTiming(total=total, device=total, host=total)

    def read(self, key: str) -> Tuple[np.ndarray, StorageBackend.IOTiming]:
        """Reads a tensor from GPU VRAM back to a NumPy array on the CPU."""
        if key not in self.cache:
            raise KeyError(f"Key {key} not found in GPU cache")

        start = time.perf_counter()

        if self.backend == 'torch':
            gpu_tensor = self.cache[key]
            cpu_tensor = gpu_tensor.to('cpu', non_blocking=True)
            torch.cuda.synchronize()
            data = cpu_tensor.numpy()
        else:
            data = cp.asnumpy(self.cache[key])
            cp.cuda.Stream.null.synchronize()

        total = time.perf_counter() - start
        return data, StorageBackend.IOTiming(total=total, device=total, host=total)

    def delete(self, key: str):
        if key in self.cache:
            del self.cache[key]
        if key in self.pinned_memory:
            del self.pinned_memory[key]

    def clear(self):
        """Clears all tensors from the GPU cache and frees memory."""
        for key in list(self.cache.keys()):
            del self.cache[key]
        self.cache.clear()
        for key in list(self.pinned_memory.keys()):
            del self.pinned_memory[key]
        self.pinned_memory.clear()

        if self.backend == 'torch' and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif self.backend == 'cupy':
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()


class SimulatedGPUBackend(StorageBackend):
    """
    Simulated GPU VRAM tier — no GPU hardware, PyTorch, or CuPy required.

    This is an in-memory metadata tracker: it stores only ``{key → size_bytes}``
    in a plain dict and models read/write latency by dividing ``size_bytes`` by a
    configurable bandwidth (default: PCIe 5.0 x16, 64 GB/s).

    Read operations regenerate fresh random bytes via dgen-py so that entries
    demoted to the CPU or NVMe tier receive correctly-sized data — consistent
    with the rest of the pipeline where bytes are synthetic.

    Because this backend is always available (no hardware dependency), the GPU
    tier exists in every run and the benchmark produces a realistic three-tier
    workload distribution regardless of the host machine.
    """

    _BYTES_PER_GB: int = 1024 ** 3

    def __init__(self, bandwidth_gb_s: float = 64.0, on_eviction_callback=None):
        """
        Parameters
        ----------
        bandwidth_gb_s :
            Simulated host↔GPU transfer bandwidth in GB/s.
            Default 64.0 models a PCIe 5.0 x16 link.
            Use ~3350.0 to model intra-GPU HBM3 bandwidth (H100/H200).
        on_eviction_callback :
            Optional callback kept for API compatibility with GPUMemoryBackend.
            Not triggered by this backend (eviction is handled by MultiTierCache).
        """
        self._bandwidth_b_per_s: float = bandwidth_gb_s * self._BYTES_PER_GB
        self.on_eviction_callback = on_eviction_callback
        self._sizes: Dict[str, int] = {}

    def write(self, key: str, data) -> StorageBackend.IOTiming:
        """Record size and return simulated PCIe transfer latency."""
        size = len(data)
        self._sizes[key] = size
        simulated = size / self._bandwidth_b_per_s
        return StorageBackend.IOTiming(total=simulated, device=simulated, host=0.0)

    def write_size(self, key: str, size_bytes: int) -> StorageBackend.IOTiming:
        """Trace-mode shortcut: record size without passing data."""
        self._sizes[key] = size_bytes
        simulated = size_bytes / self._bandwidth_b_per_s
        return StorageBackend.IOTiming(total=simulated, device=simulated, host=0.0)

    def read(self, key: str) -> Tuple[bytes, StorageBackend.IOTiming]:
        """
        Return fresh random bytes of the correct size with simulated latency.

        The actual byte content is regenerated (not stored) — only the size is
        tracked.  This is correct for simulation: the bytes are synthetic anyway.
        """
        if key not in self._sizes:
            raise KeyError(f"Key {key} not found in SimulatedGPUBackend")
        size = self._sizes[key]
        simulated = size / self._bandwidth_b_per_s
        try:
            import dgen_py
            data = dgen_py.generate_buffer(size)
        except ImportError:
            data = bytes(size)  # zero-byte fallback
        return data, StorageBackend.IOTiming(total=simulated, device=simulated, host=0.0)

    def delete(self, key: str):
        self._sizes.pop(key, None)

    def clear(self):
        self._sizes.clear()


class CPUMemoryBackend(StorageBackend):
    """
    CPU RAM storage backend.  Second tier in the cache hierarchy.

    Data layout
    -----------
    Each entry is stored as a ``bytes`` object — immutable, no GC overhead,
    no numpy dtype, no shape metadata.  This is a byte-level cache.

    Write path — ``bytes(data)`` converts any buffer-protocol object (BytesView,
                 memoryview, bytes) into an owned ``bytes`` object: one copy.
    Read path  — returns the stored ``bytes`` reference directly: zero-copy.
    """

    def __init__(self):
        self._raw: Dict[str, bytes] = {}

    def write(self, key: str, data) -> StorageBackend.IOTiming:
        """Store data as owned bytes (one copy from caller's buffer)."""
        start = time.perf_counter()
        self._raw[key] = bytes(data)
        total = time.perf_counter() - start
        return StorageBackend.IOTiming(total=total, device=total, host=total)

    def read(self, key: str) -> Tuple[bytes, StorageBackend.IOTiming]:
        """Return the stored bytes — zero-copy reference."""
        if key not in self._raw:
            raise KeyError(f"Key {key} not found in CPU cache")
        start = time.perf_counter()
        data  = self._raw[key]
        total = time.perf_counter() - start
        return data, StorageBackend.IOTiming(total=total, device=total, host=total)

    def delete(self, key: str):
        self._raw.pop(key, None)

    def clear(self):
        self._raw.clear()
        gc.collect()


class NVMeBackend(StorageBackend):
    """
    NVMe/SSD storage backend using raw binary files.

    Data layout
    -----------
    Each cache entry is stored as a flat ``<key>.bin`` file containing the raw
    little-endian bytes of the array payload (no numpy format header).
    Shape and dtype are kept in the ``metadata`` dict in memory.

    Write path — stores ``data.tobytes()`` (one copy, unavoidable for durability).
    Read path  — ``path.read_bytes()`` + ``np.frombuffer(...).reshape(...).copy()``.
                 The final ``.copy()`` is necessary to:
                   1. Free the transient ``bytes`` object returned by ``read_bytes()``.
                   2. Return a writeable, owned array to callers.
    """

    def __init__(self, base_path: str = None):
        self.temp_dir = None
        if base_path is None:
            self.temp_dir = tempfile.TemporaryDirectory(prefix="kv_cache_")
            self.base_path = Path(self.temp_dir.name)
        else:
            self.base_path = Path(base_path)
            if self.base_path.exists():
                if not self.base_path.is_dir():
                    raise NotADirectoryError(f"Cache path {self.base_path} exists but is not a directory.")
                for entry in self.base_path.glob("*.bin"):
                    try:
                        entry.unlink()
                    except OSError:
                        pass
            else:
                self.base_path.mkdir(parents=True, exist_ok=True)

        if not self.base_path.exists():
            raise OSError(f"Cache directory {self.base_path} does not exist and could not be created.")

        self.metadata = {}

    def _get_path(self, key: str) -> Path:
        """Constructs the file path for a given cache key."""
        return self.base_path / f"{key}.bin"

    def write(self, key: str, data) -> StorageBackend.IOTiming:
        """Write raw bytes to disk — accepts any buffer-protocol object."""
        start = time.perf_counter()
        path  = self._get_path(key)
        size  = len(data)

        with open(path, 'wb') as f:
            f.write(data)
            post_save = time.perf_counter()
            f.flush()
            os.fsync(f.fileno())
            post_fsync = time.perf_counter()

        self.metadata[key] = {'size': size}

        host_time   = post_save  - start
        device_time = post_fsync - post_save
        total       = post_fsync - start
        return StorageBackend.IOTiming(total=total, device=device_time, host=host_time)

    def read(self, key: str) -> Tuple[bytes, StorageBackend.IOTiming]:
        """Read raw bytes from disk — returns bytes, no numpy conversion."""
        start = time.perf_counter()
        path  = self._get_path(key)

        if not path.exists():
            raise KeyError(f"Key {key} not found in NVMe cache")
        if key not in self.metadata:
            raise KeyError(f"Metadata for {key} not found in NVMe cache")

        try:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, 4)  # POSIX_FADV_DONTNEED — drop page cache
            except AttributeError:
                pass
            finally:
                os.close(fd)
        except Exception:
            pass

        pre_load    = time.perf_counter()
        data        = path.read_bytes()
        load_done   = time.perf_counter()

        device_time = load_done - pre_load
        host_time   = pre_load - start
        total       = load_done - start
        return data, StorageBackend.IOTiming(total=total, device=device_time, host=host_time)

    def delete(self, key: str):
        path = self._get_path(key)
        if path.exists():
            path.unlink()
        if key in self.metadata:
            del self.metadata[key]

    def clear(self):
        """Delete all binary cache files from the cache directory."""
        for file in self.base_path.glob("*.bin"):
            file.unlink()
        self.metadata.clear()

    def __del__(self):
        """Clean up the temporary directory when the object is destroyed."""
        if self.temp_dir:
            self.temp_dir.cleanup()


class NullBackend(StorageBackend):
    """
    No-op storage backend used exclusively in trace mode (--io-trace-log).

    All operations are instant and consume no real GPU VRAM, CPU RAM, or
    disk space. The backend tracks object sizes so that reads can return
    a correctly-sized dummy buffer for any downstream .nbytes checks.

    Data is never actually stored — this backend exists solely to let the
    tier-selection and eviction logic run normally while eliminating all
    hardware I/O, enabling the benchmark to act as a pure logical engine
    that characterises I/O patterns without performing them.
    """

    _ZERO_TIMING = StorageBackend.IOTiming(total=0.0, device=0.0, host=0.0)

    def __init__(self):
        # Maps key → byte size of the stored object
        self._sizes: dict = {}

    def write(self, key: str, data: np.ndarray) -> StorageBackend.IOTiming:
        self._sizes[key] = data.nbytes
        return self._ZERO_TIMING

    def write_size(self, key: str, size_bytes: int) -> StorageBackend.IOTiming:
        """Trace-mode shortcut: record size without requiring a numpy array."""
        self._sizes[key] = size_bytes
        return self._ZERO_TIMING

    def read(self, key: str) -> Tuple[np.ndarray, StorageBackend.IOTiming]:
        if key not in self._sizes:
            raise KeyError(f"Key {key} not found in NullBackend")
        dummy = np.zeros(self._sizes[key], dtype=np.uint8)
        return dummy, self._ZERO_TIMING

    def delete(self, key: str):
        self._sizes.pop(key, None)

    def clear(self):
        self._sizes.clear()
