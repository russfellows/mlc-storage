"""Storage reader backends for streaming checkpoint load.

Mirrors storage_writers/ — each backend issues byte-range GETs and
discards each chunk immediately, so peak RAM = chunk_size bytes regardless
of total checkpoint size.

Use StorageReaderFactory.create() to select the appropriate backend.
"""

from .base import StorageReader
from .s3dlio_reader import S3DLIOStorageReader

from typing import Optional, Any


class StorageReaderFactory:
    """Factory for creating storage reader instances."""

    @staticmethod
    def create(
        uri: str,
        backend: Optional[str] = None,
        **kwargs: Any,
    ) -> StorageReader:
        """Create a storage reader instance.

        Args:
            uri:     Full URI (s3://, file://, etc.)
            backend: Explicit backend name: 's3dlio', 'minio', 's3torchconnector'.
                     If None, auto-detects from URI scheme (s3:// → s3dlio).
            **kwargs: Passed to the reader constructor (e.g. chunk_size).

        Returns:
            StorageReader configured for the requested backend.
        """
        if backend:
            if backend == 's3dlio':
                return S3DLIOStorageReader(uri, **kwargs)

            elif backend == 'minio':
                from .minio_reader import MinIOStorageReader
                return MinIOStorageReader(uri, **kwargs)

            elif backend == 's3torchconnector':
                from .s3torch_reader import S3TorchStorageReader
                return S3TorchStorageReader(uri, **kwargs)

            else:
                raise ValueError(
                    f"Unknown backend: {backend!r}. "
                    f"Supported: s3dlio, minio, s3torchconnector"
                )

        # Auto-detect from URI scheme — default to s3dlio for S3 URIs
        if uri.startswith('s3://') or uri.startswith('gs://') or uri.startswith('az://'):
            return S3DLIOStorageReader(uri, **kwargs)

        raise ValueError(
            f"Cannot auto-detect reader backend for URI: {uri!r}. "
            f"Specify backend= explicitly."
        )
