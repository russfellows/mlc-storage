# Storage Libraries Guide

Complete guide to all 3 supported storage libraries for MLPerf Storage benchmarks.

---

## Overview

MLPerf Storage supports **3 storage libraries** for maximum flexibility:

1. **s3dlio** - High-performance multi-protocol library (Rust + Python, zero-copy)
2. **s3torchconnector** - AWS official S3 connector for PyTorch
3. **minio** - MinIO Python SDK (S3-compatible)

---

## Quick Comparison

| Library | Protocols | Zero-Copy | Performance | Best For |
|---------|-----------|-----------|-------------|----------|
| **s3dlio** | S3/Azure/GCS/file/direct | ✅ Yes | ⭐⭐⭐⭐⭐ Highest | Maximum performance, multi-cloud |
| **s3torchconnector** | S3 only | ❌ No | ⭐⭐⭐ Good | AWS S3, standard PyTorch |
| **minio** | S3-compatible | ❌ No | ⭐⭐⭐⭐ Very Good | MinIO servers, native SDK |

---

## Installation

### s3dlio
```bash
cd ~/Documents/Code/s3dlio
pip install -e .
```

### s3torchconnector
```bash
pip install s3torchconnector
```

### minio
```bash
pip install minio
```

---

## Configuration

### Option 1: DLIO Config (MLPerf Storage)

```yaml
reader:
  storage_library: s3dlio  # or s3torchconnector
  data_loader_root: s3://my-bucket/data
  storage_options:
    endpoint_url: http://localhost:9000
    access_key_id: minioadmin
    secret_access_key: minioadmin
```

**Note:** Only `s3dlio` and `s3torchconnector` are supported via DLIO config. For MinIO and Azure, use benchmark scripts directly.

### Option 2: Benchmark Scripts (All Libraries)

```bash
# Compare all installed libraries
python benchmark_write_comparison.py --compare-all

# Compare specific libraries
python benchmark_write_comparison.py --compare s3dlio minio

# Test single library
python benchmark_write_comparison.py --library s3dlio
```

---

## Library-Specific Usage

### s3dlio

**Advantages:**
- Zero-copy architecture (5-30 GB/s throughput)
- Multi-protocol support (S3/Azure/GCS/file/direct)
- Multi-endpoint load balancing
- Drop-in replacement for s3torchconnector

**API:**
```python
import s3dlio

# Write
data = s3dlio.generate_data(100 * 1024 * 1024)  # BytesView (zero-copy)
s3dlio.put_bytes('s3://bucket/key', data)

# Read
data = s3dlio.get('s3://bucket/key')

# Read range (byte-range)
chunk = s3dlio.get_range('s3://bucket/key', offset=1000, length=999)
```

**Multi-Protocol:**
```python
# S3
s3dlio.put_bytes('s3://bucket/file', data)

# Azure
s3dlio.put_bytes('az://container/file', data)

# GCS
s3dlio.put_bytes('gs://bucket/file', data)

# Local file
s3dlio.put_bytes('file:///tmp/file', data)
```

---

### s3torchconnector

**Advantages:**
- Official AWS library
- PyTorch integration
- Standard S3 API

**API:**
```python
from s3torchconnector import S3Client, S3ClientConfig

config = S3ClientConfig(region='us-east-1')
client = S3Client(config)

# Write
writer = client.put_object('bucket', 'key')
writer.write(data_bytes)
writer.close()

# Read
reader = client.get_object('bucket', 'key')
data = reader.read()
```

---

### minio

**Advantages:**
- Native MinIO SDK
- S3-compatible API
- Optimized for MinIO servers

**API:**
```python
from minio import Minio
from io import BytesIO

client = Minio('localhost:9000',
               access_key='minioadmin',
               secret_key='minioadmin',
               secure=False)

# Write
data_io = BytesIO(data_bytes)
client.put_object('bucket', 'file.bin', data_io, len(data_bytes))

# Read
response = client.get_object('bucket', 'file.bin')
data = response.read()
response.close()
response.release_conn()
```

**Byte-Range Read:**
```python
# Read specific byte range
response = client.get_object('bucket', 'file.bin', 
                             offset=1000,  # Start byte
                             length=999)    # Number of bytes
data = response.read()
```

---


### S3-Compatible (s3dlio, s3torchconnector, minio)

**Environment Variables:**
```bash
export AWS_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
```

**Or via Config:**
```python
# s3dlio
s3dlio.configure(endpoint_url='http://localhost:9000',
                 access_key_id='minioadmin',
                 secret_access_key='minioadmin')

# s3torchconnector
from s3torchconnector import S3ClientConfig
config = S3ClientConfig(endpoint=endpoint, region='us-east-1')

# minio
client = Minio('localhost:9000',
               access_key='minioadmin',
               secret_key='minioadmin')
```

### Azure Storage (s3dlio only)

For Azure Blob Storage, use s3dlio with the `az://` protocol:

```python
import s3dlio

# Azure authentication via environment variables
# export AZURE_STORAGE_ACCOUNT=myaccount
# export AZURE_STORAGE_KEY=mykey

# Or use Azure CLI authentication (az login)
s3dlio.put_bytes('az://container/file', data)
data = s3dlio.get('az://container/file')
```

---

## Multi-Endpoint Load Balancing (s3dlio only)

s3dlio supports multi-endpoint configuration for load balancing across multiple servers:

```yaml
reader:
  storage_library: s3dlio
  endpoint_uris:
    - http://minio1:9000
    - http://minio2:9000
    - http://minio3:9000
  load_balance_strategy: round_robin  # or 'least_connections'
```

**See:** [MULTI_ENDPOINT.md](MULTI_ENDPOINT.md) for complete guide

---

## Troubleshooting

### s3dlio: Low performance

**Check zero-copy:**
```python
import s3dlio
data = s3dlio.generate_data(1024)
print(type(data))  # Must be: <class 's3dlio._pymod.BytesView'>

# BAD: bytes(data) creates copy
# GOOD: Use data directly with torch.frombuffer()
```

### minio: Connection refused

**Check MinIO is running:**
```bash
curl http://localhost:9000/minio/health/live
```

**Check credentials:**
```bash
mc alias set local http://localhost:9000 minioadmin minioadmin
mc ls local/
```

---

## Migration Guide

### From s3torchconnector to s3dlio

**Step 1:** Change DLIO config
```yaml
# OLD
reader:
  storage_library: s3torchconnector

# NEW
reader:
  storage_library: s3dlio
```

**Step 2:** That's it! (API compatible)

### From boto3 to s3dlio

**Step 1:** Replace imports
```python
# OLD
import boto3
s3 = boto3.client('s3')
s3.put_object(Bucket='bucket', Key='key', Body=data)

# NEW
import s3dlio
s3dlio.put_bytes('s3://bucket/key', data)
```

---

## Advanced Features

### Byte-Range Reads (All Libraries)

Efficient columnar format support (Parquet, HDF5):

```python
# s3dlio
chunk = s3dlio.get_range('s3://bucket/file.parquet', offset=1000, length=999)

# minio
response = client.get_object('bucket', 'file.parquet', offset=1000, length=999)

# s3torchconnector
reader = client.get_object('bucket', 'file.parquet', start=1000, end=1998)
```

**See:** [PARQUET_FORMATS.md](PARQUET_FORMATS.md) for Parquet integration

---

## Related Documentation

- **[Quick Start](QUICK_START.md)** - Get running in 5 minutes
- **[Performance Testing](PERFORMANCE_TESTING.md)** - Comprehensive benchmarks
- **[S3DLIO Integration](S3DLIO_INTEGRATION.md)** - Deep dive on s3dlio
- **[Multi-Endpoint Guide](MULTI_ENDPOINT.md)** - Load balancing configuration
- **[Parquet Formats](PARQUET_FORMATS.md)** - Byte-range reads for columnar formats

---

## Summary

- **s3dlio**: Best performance, multi-protocol, zero-copy (RECOMMENDED)
- **minio**: Good for MinIO servers, S3-compatible API  
- **s3torchconnector**: Standard AWS S3, PyTorch integration

**For maximum performance:** Use s3dlio with zero-copy verification.  
**For cloud compatibility:** Use s3dlio (works with S3/Azure/GCS).  
**For MinIO servers:** Use minio or s3dlio.
