# Performance Testing Guide

Comprehensive guide to benchmarking storage libraries for MLPerf Storage.

---

## Quick Start

### 1. Compare All Libraries (RECOMMENDED)

```bash
python benchmark_write_comparison.py \
  --compare-all \
  --endpoint http://localhost:9000 \
  --bucket benchmark \
  --files 2000 \
  --size 100 \
  --threads 32
```

**What this does:**
- Tests ALL installed libraries (s3dlio, minio, s3torchconnector)
- Writes 2,000 files × 100 MB = 200 GB per library
- Uses 32 threads for data generation
- Shows side-by-side comparison with speedup factors

---

## Comparison Modes

### Mode 1: Compare All Installed Libraries

```bash
python benchmark_write_comparison.py --compare-all
```

**Output shows:**
- Throughput (GB/s) for each library
- Total time and files per second
- Relative performance comparison
- Winner highlighted with speedup factors

### Mode 2: Compare Specific Libraries

```bash
# s3dlio vs MinIO
python benchmark_write_comparison.py --compare s3dlio minio

# s3dlio vs s3torchconnector (legacy mode)
python benchmark_write_comparison.py --compare-libraries
```

### Mode 3: Single Library Test

```bash
python benchmark_write_comparison.py --library s3dlio
python benchmark_write_comparison.py --library minio
python benchmark_write_comparison.py --library s3torchconnector
```

---

## Tuning for Maximum Performance

### Default Test (Quick)
```bash
# 10 GB test, 8 threads (1-2 minutes)
python benchmark_write_comparison.py \
  --compare-all \
  --files 100 \
  --size 100 \
  --threads 8
```

### Medium Test (Recommended)
```bash
# 200 GB test, 32 threads (3-5 minutes)
python benchmark_write_comparison.py \
  --compare-all \
  --files 2000 \
  --size 100 \
  --threads 32
```

### Large Test (Maximum Performance)
```bash
# 1 TB test, 64 threads (10-30 minutes)
python benchmark_write_comparison.py \
  --compare-all \
  --files 2000 \
  --size 500 \
  --threads 64 \
  --endpoint http://your-server:9000
```

---

## Performance Tuning Parameters

| Parameter | Small | Medium | Large | Notes |
|-----------|-------|--------|-------|-------|
| --files | 100 | 2000 | 5000 | Total file count |
| --size (MB) | 100 | 100-500 | 500-1000 | Per-file size |
| --threads | 8 | 16-32 | 32-64 | Data generation |
| Network | 10 Gbps | 100 Gbps | 200+ Gbps | Bandwidth |
| Storage | SATA SSD | NVMe RAID | Multi-server | Backend |

**Rule of thumb:**
- File size × File count = Total data (per library)
- Threads = 2× CPU cores (for data generation)
- Network must support 3-4× peak throughput (for network overhead)

---

## Read Performance Testing

### Read Comparison

```bash
python benchmark_read_comparison.py \
  --compare-all \
  --endpoint http://localhost:9000 \
  --bucket benchmark \
  --files 2000 \
  --size 100
```

### Single Library Read Test

```bash
python benchmark_s3dlio_read.py \
  --endpoint http://localhost:9000 \
  --bucket benchmark \
  --files 100 \
  --size 100
```

---

## Zero-Copy Verification (s3dlio)

### Quick Verification (No S3 Required)

```bash
python benchmark_s3dlio_write.py --skip-write-test
```

**Expected Output:**
```
================================================================================
ZERO-COPY VERIFICATION
================================================================================

✅ memoryview() works - buffer protocol supported
✅ torch.frombuffer() works
✅ np.frombuffer() works
✅ Zero-copy verified throughout the stack!
```

### Data Generation Speed Test

```bash
python benchmark_s3dlio_write.py \
  --skip-write-test \
  --skip-zerocopy-test \
  --threads 16
```

**Note:** s3dlio provides high-performance data generation for testing.

---

## Benchmark Scripts Overview

### Write Benchmarks

| Script | Purpose | Libraries |
|--------|---------|-----------|
| `benchmark_write_comparison.py` | Compare multiple libraries | All 4 |
| `benchmark_s3dlio_write.py` | s3dlio detailed test | s3dlio only |

### Read Benchmarks

| Script | Purpose | Libraries |
|--------|---------|-----------|
| `benchmark_read_comparison.py` | Compare read performance | All 4 |
| `benchmark_s3dlio_read.py` | s3dlio read test | s3dlio only |

---

## Performance Characteristics

### Relative Performance (General Observations)

Based on testing across various configurations:

**Write Operations:**
- **s3dlio**: Fastest throughput due to zero-copy architecture
- **minio**: Moderate to good performance with native MinIO SDK
- **s3torchconnector**: Standard performance with AWS SDK

**Read Operations:**
- **s3dlio**: Highest throughput with zero-copy reads
- **minio**: Good performance for S3-compatible storage
- **s3torchconnector**: Standard AWS S3 read performance

**Note:** Actual performance varies significantly based on:
- Network bandwidth (10 Gbps vs 100+ Gbps)
- Storage backend (SATA SSD vs NVMe RAID)
- CPU cores and memory
- File size and count
- Server configuration

Run your own benchmarks to determine performance for your specific environment.

---

## Performance Validation Checklist

Before running benchmarks:

- [ ] **Network:** Run `iperf3 -c server` to verify network throughput
- [ ] **Storage:** Run `fio` test to check storage backend performance
- [ ] **CPU:** Check `lscpu` - more cores enable higher thread counts
- [ ] **Memory:** Check `free -h` - sufficient RAM prevents swapping during tests
- [ ] **Zero-copy:** Run `benchmark_s3dlio_write.py --skip-write-test` (s3dlio only)

---

## Troubleshooting

### Problem: Lower than expected throughput

**Network bottleneck check:**
```bash
iperf3 -c your-server
# Verify network bandwidth meets or exceeds storage throughput needs
```

**Storage bottleneck check:**
```bash
fio --name=seq --rw=write --bs=4M --size=10G --numjobs=8 --group_reporting
# Verify storage backend can sustain high throughput
```

**CPU bottleneck check:**
```bash
python benchmark_s3dlio_write.py --skip-write-test --threads 32
# Verify data generation is faster than storage throughput
```

### Problem: Zero-copy not working (s3dlio)

**Type check:**
```python
import s3dlio
data = s3dlio.generate_data(1024)
print(type(data))
# Must be: <class 's3dlio._pymod.BytesView'>
```

**Search for bad conversions:**
```bash
grep -r "bytes(s3dlio" .
grep -r "bytes(data)" .
# Should find ZERO results in hot path
```

### Problem: MinIO connection refused

**Check MinIO status:**
```bash
curl http://localhost:9000/minio/health/live
```

**Verify credentials:**
```bash
mc alias set local http://localhost:9000 minioadmin minioadmin
mc ls local/
```

---

## Advanced Testing

### Multi-Endpoint Testing (s3dlio only)

**Config:**
```yaml
reader:
  storage_library: s3dlio
  endpoint_uris:
    - http://minio1:9000
    - http://minio2:9000
    - http://minio3:9000
  load_balance_strategy: round_robin
```

**Run:**
```bash
mlpstorage training run --model resnet50 --config multi_endpoint.yaml
```

**See:** [MULTI_ENDPOINT.md](MULTI_ENDPOINT.md) for complete guide

### Parquet Byte-Range Testing

Test columnar format efficiency:

**See:** [PARQUET_FORMATS.md](PARQUET_FORMATS.md) for Parquet benchmarks

---

## Performance Analysis

### Analyze Benchmark Logs

```bash
# Extract throughput numbers
grep "Throughput:" benchmark_output.log

# Plot over time (requires matplotlib)
python analyze_benchmark_results.py --log benchmark_output.log
```

### Compare Across Runs

```bash
# Save results
python benchmark_write_comparison.py --compare-all > run1.txt
# ... make changes ...
python benchmark_write_comparison.py --compare-all > run2.txt

# Compare
diff run1.txt run2.txt
```

---

## Continuous Performance Monitoring

### Daily Performance Test

```bash
#!/bin/bash
# daily_perf_test.sh

cd ~/Documents/Code/mlp-storage
source .venv/bin/activate

DATE=$(date +%Y%m%d)

python benchmark_write_comparison.py \
  --compare-all \
  --files 2000 \
  --size 100 \
  --threads 32 > perf_results_${DATE}.log

# Review results and compare against baseline
echo "Performance test complete. Results: perf_results_${DATE}.log"
```

---

## Related Documentation

- **[Storage Libraries](STORAGE_LIBRARIES.md)** - Learn about all 4 libraries
- **[Quick Start](QUICK_START.md)** - Setup and first benchmark
- **[S3DLIO Integration](S3DLIO_INTEGRATION.md)** - Deep dive on s3dlio
- **[Multi-Endpoint](MULTI_ENDPOINT.md)** - Load balancing

---

## Summary

**Quick comparison:**
```bash
python benchmark_write_comparison.py --compare-all
```

**Maximum performance:**
```bash
python benchmark_write_comparison.py \
  --compare-all \
  --files 2000 \
  --size 500 \
  --threads 64
```

**Zero-copy check:**
```bash
python benchmark_s3dlio_write.py --skip-write-test
```

**Note:** Performance varies by environment. s3dlio typically shows the highest throughput due to zero-copy architecture.
