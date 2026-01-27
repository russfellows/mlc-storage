# MLPerf Storage KV Cache Benchmark

A storage benchmarking tool for Large Language Model inference systems. This benchmark measures the performance of your storage subsystem under realistic KV cache offloading workloads, helping you answer critical questions about hardware capacity and configuration.

**Author:** Hazem Awadallah, Kingston Digital
**License:** Apache 2.0
**Version:** MLPerf Storage v3.0 (Enhanced)
**Updated:** January 27, 2026

---

## Table of Contents

1. [What This Benchmark Does](#what-this-benchmark-does)
2. [Architecture Overview](#architecture-overview)
3. [System Requirements](#system-requirements)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Quick Start](#quick-start)
7. [Running the Benchmark](#running-the-benchmark)
8. [ShareGPT Replay Workloads](#sharegpt-replay-workloads)
9. [Using the Wrapper Script](#using-the-wrapper-script)
10. [Understanding Results](#understanding-results)
11. [Unit Testing](#unit-testing)
12. [Excel Export](#excel-export)
13. [MLPerf Submission Guidelines](#mlperf-submission-guidelines)
14. [Troubleshooting](#troubleshooting)

---

## What This Benchmark Does

During LLM inference, models store intermediate attention data in a structure called the KV (Key-Value) cache. This cache grows with conversation length and can consume enormous amounts of memory. Production systems offload this cache from expensive GPU VRAM to cheaper CPU RAM or NVMe storage.

This benchmark simulates that offloading behavior. It generates realistic multi-user inference workloads and measures how your storage performs under pressure. It measures these components:

- How many concurrent users your hardware can support
- Whether your NVMe drive is fast enough to handle cache spillover
- The real latency impact of each storage tier
- Where the bottleneck sits in your system

This is not a pass/fail test. It is a diagnostic tool for system architects and performance engineers.

---

## Architecture Overview

The benchmark implements a three-tier memory hierarchy that mirrors production LLM serving systems.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         KV Cache Benchmark Architecture                      │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────────┐
                              │   User Requests  │
                              │  (Multi-tenant)  │
                              └────────┬─────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │         Request Queue                │
                    │   (Priority-based: QoS levels)       │
                    │   Interactive > Responsive > Batch   │
                    └──────────────────┬───────────────────┘
                                       │
                                       ▼
          ┌────────────────────────────────────────────────────────┐
          │                  IntegratedBenchmark                   │
          │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
          │  │   Prefill   │  │   Decode    │  │  Conversation   │ │
          │  │   (Write)   │  │   (Read)    │  │    Manager      │ │
          │  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘ │
          └─────────┼────────────────┼─────────────────┼───────────┘
                    │                │                 │
                    └────────────────┼─────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MultiTierCache                                     │
│                     (Waterfall LRU Eviction)                                 │
│                                                                              │
│    New Data ─────► Always targets fastest available tier                     │
│                    If full, LRU entry cascades down                          │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                                                                     │    │
│  │   ┌───────────────┐      ┌───────────────┐      ┌───────────────┐  │    │
│  │   │   GPU VRAM    │      │   CPU RAM     │      │    NVMe       │  │    │
│  │   │   (Tier 1)    │─────►│   (Tier 2)    │─────►│   (Tier 3)    │  │    │
│  │   │               │ LRU  │               │ LRU  │               │  │    │
│  │   │  Sub-ms       │evict │  Tens of ms   │evict │  Hundreds     │  │    │
│  │   │  latency      │      │  latency      │      │  of ms        │  │    │
│  │   │               │      │               │      │               │  │    │
│  │   │  PyTorch/CuPy │      │  NumPy arrays │      │  .npy files   │  │    │
│  │   │  tensors      │      │  in memory    │      │  on disk      │  │    │
│  │   └───────────────┘      └───────────────┘      └───────────────┘  │    │
│  │                                                                     │    │
│  │   ◄──── HOT DATA ────────────────────────────── COLD DATA ────►    │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Components

**MultiTierCache**: The core engine. It decides where to place data based on available space and access patterns. New data always targets the fastest tier. When that tier fills up, the least recently used entry gets pushed down to the next tier.

**Inference Phases**: The benchmark models two distinct I/O patterns:
- **Prefill**: Write-heavy. Processing the user prompt generates new KV cache entries.
- **Decode**: Read-heavy. Generating each output token requires reading the existing cache.

**User Simulation**: Creates realistic traffic from multiple concurrent users with different behaviors (chatbot, coding assistant, document analysis) and priority levels.

**Autoscaler**: Automatically adjusts user load to find either the maximum users your system can handle (QoS mode) or the peak throughput of your storage (capacity mode).

---

## System Requirements

### Minimum

- CPU: 8+ cores (AMD EPYC, Intel Xeon)
- RAM: 32 GB
- Storage: 256 GB free space on SSD
- OS: Linux (Ubuntu 22.04, RHEL 9, or similar) or Windows
- Python: 3.8 or higher
- No GPU required (runs in CPU-only mode)

### Recommended

- CPU: 32+ cores
- RAM: 128 GB or more
- GPU: NVIDIA A100/H100 with 40+ GB VRAM (optional but enables full three-tier testing)
- Storage: 1 TB+ on NVMe (PCIe Gen4 or Gen5)
- Tools: `bc`, `jq` for the wrapper script (Linux)

---

## Installation

1. Clone or download this repository.

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

Or install core dependencies manually:

```bash
pip install numpy pyyaml
```

3. For GPU support (optional):

```bash
pip install torch  # or cupy-cuda12x for CuPy
```

4. For ShareGPT replay workloads (optional):

```bash
pip install tiktoken
```

5. For Excel export (optional):

```bash
pip install pandas openpyxl
```

6. Verify the installation:

```bash
python3 kv-cache.py --help
```

---

## Configuration

The benchmark supports a YAML configuration file (`config.yaml`) for tuning internal parameters without modifying the source code. This is the **recommended approach** for MLPerf submissions to ensure reproducibility.

### Using the Configuration File

```bash
python3 kv-cache.py --config config.yaml [other CLI arguments]
```

**Note:** CLI arguments always take precedence over config file values for overlapping settings.

### Configuration File Parameters (config.yaml)

The configuration file controls internal benchmark behavior that affects workload realism and cache dynamics. These settings are **not** exposed as CLI arguments to prevent accidental misconfigurations in MLPerf submissions.

> **Tip:** For most benchmarking scenarios, the defaults are carefully tuned. Only modify these if you understand the impact on your results.

---

#### User Templates

Controls the three simulated user personas. Each persona has distinct characteristics that model real-world usage patterns.

| Persona | Behavior | Use Case |
|---------|----------|----------|
| **Chatbot** | Short prompts, quick responses, fast iteration | Customer service bots, casual conversation |
| **Coding** | Medium prompts with code context, moderate responses | IDE assistants, code completion |
| **Document** | Long prompts with full documents, lengthy analysis | Document summarization, legal/medical analysis |

| Parameter | Type | Default | Impact |
|-----------|------|---------|--------|
| `user_templates.chatbot.context_range` | [min, max] | [256, 1024] | **KV cache write size per request.** Smaller values reduce storage pressure; larger values stress NVMe throughput. |
| `user_templates.chatbot.generation_range` | [min, max] | [50, 150] | **Decode phase duration.** More tokens = more cache reads per request. Affects read/write ratio. |
| `user_templates.chatbot.think_time_range` | [min, max] | [0.1, 0.5] | **Request inter-arrival time.** Shorter = higher request rate, more concurrent cache operations. |
| `user_templates.coding.context_range` | [min, max] | [1024, 4096] | Medium-length contexts typical of code completion scenarios. 4× larger than chatbot. |
| `user_templates.coding.generation_range` | [min, max] | [100, 500] | Code generation often produces longer outputs than conversational AI. |
| `user_templates.coding.think_time_range` | [min, max] | [0.2, 1.0] | Developers pause to review generated code before next request. |
| `user_templates.document.context_range` | [min, max] | [2048, 8192] | **Stress test scenarios.** 8K tokens creates ~1 GB of total KV cache data for 8B models (128 KB/token × 8,192 tokens). |
| `user_templates.document.generation_range` | [min, max] | [200, 800] | Long-form analysis outputs (summaries, reports). |
| `user_templates.document.think_time_range` | [min, max] | [0.3, 1.5] | Users read lengthy outputs before continuing. |

---

#### Token Generation Timing

Simulates GPU compute time per generated token. This controls the backpressure on the storage system.

| Mode | Default (sec/token) | When to Use |
|------|---------------------|-------------|
| `none` | 0.0 | **Pure storage benchmarking.** 100% of measured latency is I/O. Use for MLPerf storage submissions. |
| `fast` | 0.002 (2ms) | Simulates high-end GPU (H100) with optimized inference. Creates light backpressure. |
| `realistic` | 0.030 (30ms) | Simulates typical production GPU throughput. Balances compute/storage for end-to-end analysis. |

**Why it matters:** With `generation_mode=none`, the benchmark hammers storage as fast as possible. With `realistic`, storage has time to absorb writes between decode steps, showing how your system performs under sustained (not burst) load.

---

#### QoS Profiles (Quality of Service)

Defines SLA targets for multi-tenant request prioritization. The benchmark tracks violations against these thresholds.

| Profile | Typical Use Case | Priority |
|---------|------------------|----------|
| **Interactive** | Live chat UIs, real-time assistants | Highest (3) |
| **Responsive** | API calls, near-real-time processing | Medium (2) |
| **Batch** | Overnight jobs, bulk processing | Lowest (1) |

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `qos_profiles.interactive.target_latency_p95_ms` | 50 | 95% of interactive requests must complete within 50ms. Aggressive target for premium users. |
| `qos_profiles.interactive.target_latency_p99_ms` | 100 | 99% within 100ms. Allows some slack for tail latency. |
| `qos_profiles.interactive.target_latency_p999_ms` | 150 | 99.9% (3 nines) within 150ms. Production SLOs often specify this level. |
| `qos_profiles.interactive.target_latency_p9999_ms` | 200 | 99.99% (4 nines) within 200ms. Critical for detecting storage-induced tail latency. |
| `qos_profiles.interactive.priority` | 3 | Highest priority. These requests are dequeued first. |
| `qos_profiles.responsive.target_latency_p95_ms` | 100 | 2× the interactive target. Acceptable for API consumers. |
| `qos_profiles.responsive.target_latency_p99_ms` | 200 | 99% within 200ms. |
| `qos_profiles.responsive.target_latency_p999_ms` | 350 | 99.9% within 350ms. |
| `qos_profiles.responsive.target_latency_p9999_ms` | 500 | 99.99% within 500ms. |
| `qos_profiles.responsive.priority` | 2 | Medium priority. |
| `qos_profiles.batch.target_latency_p95_ms` | 1000 | 1 second. Batch jobs are latency-tolerant. |
| `qos_profiles.batch.target_latency_p99_ms` | 5000 | 5 seconds. Acceptable for offline processing. |
| `qos_profiles.batch.target_latency_p999_ms` | 7500 | 7.5 seconds. |
| `qos_profiles.batch.target_latency_p9999_ms` | 10000 | 10 seconds. Even worst-case should complete eventually. |
| `qos_profiles.batch.priority` | 1 | Lowest priority. Processed when interactive/responsive queues are empty. |

> **Research Basis for QoS Targets** (see [sources.md](sources.md) for full citations):
> - **Interactive (50ms P95, 100ms P99)**: Based on Nielsen Norman Group's 0.1s "instant" threshold, Google RAIL <100ms response target, and observed production LLM APIs (Anthropic Claude TTFT: 50–150ms).
> - **Responsive (100ms P95, 200ms P99)**: Based on Google Core Web Vitals FID <100ms "good" threshold, INP ≤200ms target, and Vercel Edge Functions P99 <200ms.
> - **Batch (1000ms P95, 5000ms P99)**: Based on AWS ALB healthy target <1s, and research showing batch workloads tolerate >1s latency ([Splitwise paper](https://arxiv.org/abs/2401.07935): 80% of production requests need <200ms).
>
> **Note:** MLPerf Inference v4.0–v5.0 defines Server/Offline scenarios but does **not** prescribe specific P95/P99 latency SLAs. These targets represent industry best practices, not MLPerf requirements.

---

#### QoS Distribution

Controls the probability mix of request priorities in the simulated workload.

| Parameter | Default | Effect |
|-----------|---------|--------|
| `interactive_probability` | 0.15 | 15% of requests are INTERACTIVE. Increase to stress-test low-latency paths. |
| `responsive_threshold` | 0.50 | If not INTERACTIVE, 35% of remaining requests (50% - 15%) are RESPONSIVE. The rest are BATCH. |

**Example distribution with defaults:** 15% Interactive, 35% Responsive, 50% Batch.

---

#### Eviction Settings

Controls the waterfall LRU eviction algorithm that moves cold data down the tier hierarchy (GPU → CPU → NVMe).

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `max_recursion_depth` | 10 | **Safety limit.** Prevents infinite cascading evictions. If you hit this limit, your tiers are severely undersized. |
| `target_usage_ratio` | 0.8 | **Tier headroom.** Keeps each tier at 80% capacity, leaving 20% buffer for burst writes. Lower values = more headroom, fewer evictions. |
| `large_entry_limit_ratio` | 0.95 | **Skip-tier threshold.** If a single entry exceeds 95% of tier capacity, skip directly to the next tier. Prevents tier thrashing with huge entries. |
| `max_evictions_hard_cap` | 5000 | **Absolute safety limit.** Stops eviction loop after 5000 entries regardless of space needs. Prevents runaway eviction under pathological conditions. |
| `max_evictions_min` | 1000 | **Minimum eviction budget.** Ensures the algorithm tries at least 1000 evictions before giving up. Helps with large-model scenarios where many small entries must be evicted. |

**Tuning guidance:** If you see "Hit recursion limit" warnings, increase `max_recursion_depth`. If evictions dominate your latency, reduce `target_usage_ratio` to provide more headroom.

---

#### GPU Backend Settings

Controls GPU VRAM allocation and out-of-memory (OOM) recovery behavior.

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `memory_fraction` | 0.9 | **VRAM budget.** Uses 90% of GPU memory, reserving 10% for framework overhead and other processes. |
| `max_eviction_attempts` | 100 | **OOM recovery limit.** On CUDA OOM, attempts up to 100 evictions to free space before failing the write. |
| `free_memory_threshold` | 0.1 | **Proactive eviction trigger.** When free GPU memory drops below 10%, begin evicting to CPU before OOM occurs. |

**Note:** These settings only apply when `--gpu-mem-gb > 0` and PyTorch/CuPy is available.

---

#### Prefix Cache Settings

Controls hierarchical prefix caching for system prompts (e.g., "You are a helpful assistant").

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `min_prefix_length` | 50 | **Minimum tokens for caching.** Prefixes shorter than 50 tokens aren't worth the overhead of caching. |
| `max_prefix_entries` | 1000 | **Prefix cache capacity.** LRU eviction kicks in when this limit is reached. Higher values consume more memory but improve hit rates. |
| `system_prompt_hit_probability` | 0.2 | **Simulation realism.** 20% of requests share a common system prompt. Increase to model deployments with standardized prompts (e.g., corporate assistants). |

**Impact:** Higher `system_prompt_hit_probability` → higher cache hit rates → lower storage throughput (because prefixes are reused). Use 0.0 for pure storage stress testing.

---

#### RAG Settings

Controls Retrieval-Augmented Generation workload simulation, where external documents are injected into the context.

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `chunk_size_tokens` | 512 | **Document chunk granularity.** Each document is split into 512-token chunks for independent caching. Smaller chunks = more cache entries, higher metadata overhead. |
| `top_k_chunks` | 5 | **Retrieval depth.** Number of chunks retrieved per RAG query. More chunks = larger context window = more KV cache I/O. |
| `max_chunk_bytes` | 268435456 | **256 MB per chunk.** Safety limit to prevent single chunks from consuming entire tiers. Particularly important for 70B models where 512 tokens ≈ 160 MB of KV cache (320 KB/token). |

**When to enable RAG:** Use `--enable-rag` when benchmarking systems designed for document-heavy workloads (legal, medical, enterprise search).

---

#### Conversation Settings

Controls multi-turn conversation simulation, modeling how chatbot context accumulates across turns.

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `max_conversations` | 1000 | **Concurrent conversation limit.** LRU eviction removes oldest conversations when this limit is hit. Higher values = more memory for conversation metadata. |
| `max_turns_per_conv` | 50 | **Conversation depth limit.** After 50 turns, the conversation resets. Prevents unbounded context growth in long-running benchmarks. |
| `end_conversation_probability` | 0.2 | **Conversation turnover rate.** 20% chance each turn ends the conversation. Lower values = longer conversations = more cache reuse. |

**Impact on metrics:** Higher `max_turns_per_conv` and lower `end_conversation_probability` increase cache hit rates (context reuse). Use low values for stress testing (force cache misses).

---

#### Autoscaler Settings

Controls the workload autoscaler that discovers system saturation points.

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `min_users` | 1 | **Lower bound.** Autoscaler won't go below 1 user. |
| `max_users` | 10000 | **Upper bound.** Autoscaler stops scaling up at 10,000 users. Prevents runaway resource consumption. |
| `scale_up_factor` | 1.2 | **Growth rate.** Increases users by 20% each scaling action (e.g., 100 → 120 → 144). |
| `scale_down_factor` | 0.8 | **Decay rate.** Decreases users by 20% when SLAs are violated (e.g., 100 → 80 → 64). |
| `consecutive_samples_required` | 2 | **Stability requirement.** Requires 2 consecutive samples agreeing on direction before scaling. Prevents oscillation from transient spikes. |

**QoS mode vs Capacity mode:** In QoS mode, the autoscaler maximizes users while maintaining latency SLAs. In Capacity mode, it maximizes throughput regardless of latency.

---

#### Decode Phase Settings

Controls token generation batching during the decode (read-heavy) phase.

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `batch_size` | 32 | **Decode batch granularity.** Reads 32 tokens worth of KV cache per decode operation. Larger batches amortize I/O overhead but require more memory. |

---

#### ShareGPT Dataset Settings

Controls loading and processing of real ShareGPT conversation data.

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `max_context_tokens` | 8192 | **Context truncation.** Conversations longer than 8192 tokens are truncated. Prevents OOM with very long conversations. |
| `max_generation_tokens` | 2048 | **Generation truncation.** Caps simulated generation at 2048 tokens per turn. |
| `chars_per_token_estimate` | 4 | **Tokenization heuristic.** Used when tiktoken is unavailable. 4 chars/token is typical for English text. |

---

#### Saturation Detection Thresholds

Controls when the StorageMonitor considers the storage subsystem saturated.

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `read_latency_p95_threshold_ms` | 100 | **Read saturation signal.** If P95 read latency exceeds 100ms, storage is considered stressed. |
| `write_latency_p95_threshold_ms` | 50 | **Write saturation signal.** Writes are more sensitive; 50ms threshold triggers concern earlier. |
| `queue_depth_threshold` | 100 | **Queue pressure signal.** More than 100 pending requests indicates backlog is building. |
| `history_window_size` | 10 | **Trend analysis window.** Uses last 10 samples to detect latency trends (increasing = saturation). |

**Used by:** The autoscaler uses these thresholds to decide when to scale down (in QoS mode) or when peak throughput is reached (in capacity mode).

---

#### Validation Limits

Safety limits enforced by `validate_args()` to prevent accidental misconfigurations.

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `max_users` | 100000 | Reasonable upper bound for simulated users. Prevents accidental `--num-users 1000000`. |
| `max_duration_seconds` | 86400 | 24 hours maximum. Prevents runaway benchmarks that run forever. |
| `max_gpu_memory_gb` | 1024 | 1 TB. Covers even the largest GPU clusters (8× H100 80GB = 640GB). |
| `max_cpu_memory_gb` | 16384 | 16 TB. Covers high-memory server configurations. |

---

## Quick Start

Run a basic storage test with 50 users for 2 minutes:

```bash
python3 kv-cache.py \
    --config config.yaml \
    --model llama3.1-8b \
    --num-users 50 \
    --duration 120 \
    --gpu-mem-gb 0 \
    --cpu-mem-gb 4 \
    --generation-mode realistic \
    --cache-dir /mnt/nvme \
    --seed 42 \
    --output results.json
```

This forces all cache operations to hit your NVMe drive, giving you a baseline measurement of storage performance.

---

## Running the Benchmark

### CLI-Only Arguments

These arguments **must** be passed via command line (not configurable in config.yaml):

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--config` | str | None | No | Path to YAML configuration file |
| `--log-level` | str | INFO | No | Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL |
| `--model` | str | llama3.1-8b | Yes | Model config: tiny-1b, mistral-7b, llama2-7b, llama3.1-8b, llama3.1-70b-instruct |
| `--num-users` | int | 100 | Yes | Number of concurrent users to simulate |
| `--duration` | int | 60 | Yes | Benchmark duration in seconds |
| `--gpu-mem-gb` | float | 16 | Yes | GPU VRAM budget in GB (0 to disable) |
| `--cpu-mem-gb` | float | 32 | Yes | CPU RAM budget in GB |
| `--cache-dir` | str | temp | No | Directory for NVMe cache files |
| `--generation-mode` | str | realistic | No | Token generation: none, fast, realistic |
| `--performance-profile` | str | latency | No | Pass/fail criteria: latency, throughput |
| `--disable-multi-turn` | flag | False | No | Disable multi-turn conversation caching |
| `--disable-prefix-caching` | flag | False | No | Disable prefix caching |
| `--enable-rag` | flag | False | No | Enable RAG workload simulation |
| `--rag-num-docs` | int | 10 | No | Number of RAG documents to ingest |
| `--enable-autoscaling` | flag | False | No | Enable workload autoscaling |
| `--autoscaler-mode` | str | qos | No | Autoscaling strategy: qos, capacity |
| `--target-saturation` | float | 0.8 | No | Target storage saturation (0.0-1.0) |
| `--use-burst-trace` | flag | False | No | Use BurstGPT trace for workload |
| `--burst-trace-path` | str | BurstGPT/... | No | Path to BurstGPT trace file |
| `--validation-trace` | str | None | No | Path to validation trace file |
| `--dataset-path` | str | None | No | Path to ShareGPT dataset JSON |
| `--max-conversations` | int | 500 | No | Max conversations from dataset |
| `--output` | str | auto | No | Output JSON file path |
| `--seed` | int | None | **MLPerf** | Random seed for reproducibility |
| `--max-concurrent-allocs` | int | 0 | No | Limit concurrent allocations (0=unlimited) |
| `--request-rate` | float | 0 | No | Target request rate (req/sec, 0=unlimited) |
| `--max-requests` | int | 0 | No | Stop after N requests (0=use duration) |
| `--xlsx-output` | str | None | No | Excel/CSV output file path |

### Test Scenarios

#### Scenario 1: Storage-Only Baseline

Isolate your NVMe drive by setting GPU memory to zero. This tells you the raw performance of your storage.

```bash
python3 kv-cache.py \
    --config config.yaml \
    --model llama3.1-8b \
    --num-users 50 \
    --duration 180 \
    --gpu-mem-gb 0 \
    --cpu-mem-gb 4 \
    --generation-mode realistic \
    --cache-dir /mnt/nvme \
    --seed 42 \
    --output results_storage_only.json
```

#### Scenario 2: Realistic Production Setup

Test a balanced three-tier configuration that mirrors production deployment.

```bash
python3 kv-cache.py \
    --config config.yaml \
    --model llama3.1-8b \
    --num-users 100 \
    --duration 300 \
    --gpu-mem-gb 16 \
    --cpu-mem-gb 32 \
    --generation-mode realistic \
    --cache-dir /mnt/nvme \
    --seed 42 \
    --output results_production.json
```

#### Scenario 3: Find Maximum User Count (QoS Mode)

Let the autoscaler discover how many users your system can handle while maintaining acceptable latency.

```bash
python3 kv-cache.py \
    --config config.yaml \
    --model llama3.1-8b \
    --num-users 20 \
    --duration 300 \
    --gpu-mem-gb 16 \
    --cpu-mem-gb 32 \
    --enable-autoscaling \
    --autoscaler-mode qos \
    --generation-mode realistic \
    --cache-dir /mnt/nvme \
    --seed 42 \
    --output results_autoscale_qos.json
```

#### Scenario 4: Find Peak Storage Throughput (Capacity Mode)

Discover the absolute maximum I/O your storage can deliver by ignoring latency constraints.

```bash
python3 kv-cache.py \
    --config config.yaml \
    --model llama3.1-70b-instruct \
    --num-users 10 \
    --duration 180 \
    --gpu-mem-gb 0 \
    --cpu-mem-gb 4 \
    --enable-autoscaling \
    --autoscaler-mode capacity \
    --generation-mode none \
    --cache-dir /mnt/nvme \
    --seed 42 \
    --output results_capacity.json
```

#### Scenario 5: Low Cache Hit Rate (Maximum Storage Stress)

Force cache misses to maximize NVMe I/O pressure. This is useful for stress testing storage subsystems and measuring worst-case performance.

**Key flags to lower cache hit rate:**
- `--disable-multi-turn`: Each request is independent (no conversation context reuse)
- `--disable-prefix-caching`: No system prompt caching (every request generates fresh KV cache)
- `--cpu-mem-gb 0`: No CPU tier buffer (all evictions go directly to NVMe)
- High user count with synthetic workload: More unique cache entries

```bash
# Minimal caching - forces nearly all operations to hit NVMe
python3 kv-cache.py \
    --config config.yaml \
    --model llama3.1-8b \
    --num-users 200 \
    --duration 180 \
    --gpu-mem-gb 0 \
    --cpu-mem-gb 0 \
    --disable-multi-turn \
    --disable-prefix-caching \
    --generation-mode none \
    --cache-dir /mnt/nvme \
    --seed 42 \
    --output results_low_hit_rate.json
```

**Expected results:** Cache hit rate drops to 10-30% (vs 50-70% with defaults, or 85-97% with ShareGPT).

For even more aggressive stress testing with the 70B model (2.5× larger KV cache per token):

```bash
# Maximum NVMe stress - 70B model with no caching
python3 kv-cache.py \
    --config config.yaml \
    --model llama3.1-70b-instruct \
    --num-users 50 \
    --duration 180 \
    --gpu-mem-gb 0 \
    --cpu-mem-gb 0 \
    --disable-multi-turn \
    --disable-prefix-caching \
    --generation-mode none \
    --cache-dir /mnt/nvme \
    --seed 42 \
    --output results_70b_low_hit_rate.json
```

| Configuration | Typical Cache Hit Rate | Use Case |
|---------------|------------------------|----------|
| ShareGPT + defaults | 85-97% | Realistic production simulation |
| Synthetic + defaults | 50-70% | Balanced stress testing |
| `--disable-multi-turn` only | 30-50% | Moderate stress |
| `--disable-multi-turn --disable-prefix-caching` | 10-30% | Maximum NVMe stress |
| Above + `--cpu-mem-gb 0` | 5-15% | Worst-case storage scenario |

---

## ShareGPT Replay Workloads

While synthetic workloads are excellent for controlled stress testing, they may not capture the nuances of real human-AI interaction. The **ShareGPT Replay** feature addresses this by loading actual conversation data.

### Why Use ShareGPT?

Real conversations exhibit different patterns than synthetic workloads:
- **Higher cache locality**: Users ask follow-up questions, reusing context
- **Variable context sizes**: Real queries vary wildly (10-16,000 tokens)
- **Multi-turn structure**: Conversation flows are preserved

### Downloading the ShareGPT Dataset

Download the full dataset from Hugging Face (~1.2 GB):

```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

**Alternative: Smaller subset for quick testing (~40 MB):**

```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json
```

### Basic ShareGPT Invocation

```bash
python3 kv-cache.py \
    --config config.yaml \
    --model llama3.1-8b \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
    --max-conversations 500 \
    --num-users 50 \
    --duration 300 \
    --gpu-mem-gb 0 \
    --cpu-mem-gb 4 \
    --generation-mode realistic \
    --cache-dir /mnt/nvme \
    --seed 42 \
    --output results_sharegpt.json
```

### ShareGPT with Rate Limiting

Control the request arrival rate for steady-state testing:

```bash
python3 kv-cache.py \
    --config config.yaml \
    --model llama3.1-70b-instruct \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
    --max-conversations 1000 \
    --request-rate 10.0 \
    --num-users 100 \
    --duration 600 \
    --gpu-mem-gb 0 \
    --cpu-mem-gb 8 \
    --generation-mode none \
    --cache-dir /mnt/nvme \
    --seed 42 \
    --output results_sharegpt_rate_limited.json
```

### ShareGPT with Fixed Request Count

Run exactly N requests for reproducible benchmarks:

```bash
python3 kv-cache.py \
    --config config.yaml \
    --model llama3.1-8b \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
    --max-requests 5000 \
    --num-users 50 \
    --gpu-mem-gb 0 \
    --cpu-mem-gb 4 \
    --generation-mode realistic \
    --cache-dir /mnt/nvme \
    --seed 42 \
    --output results_sharegpt_fixed.json
```

### Comparing Real vs Synthetic Workloads

| Metric | ShareGPT (Real) | Synthetic (Random) |
| :--- | :--- | :--- |
| Mean Context Size | ~133 tokens | ~2,676 tokens |
| Cache Hit Rate | 85-97% | 50-70% |
| Multi-turn Locality | High | Medium |
| Throughput | Higher | Lower |
| NVMe Stress | Moderate | Extreme |

**Use ShareGPT** when you want to model real chatbot/assistant usage.
**Use Synthetic** when you want worst-case stress testing or controlled experiments.

---

## Using the Wrapper Script

The `kv-cache-wrapper.sh` script automates a complete benchmark suite. It detects your hardware, calculates appropriate parameters, and runs multiple test scenarios.

### Basic Usage

```bash
./kv-cache-wrapper.sh
```

This runs all test scenarios with default settings. Expect roughly 30 minutes for the full suite.

### Options

```
./kv-cache-wrapper.sh [options]

  -m MODEL     Model to benchmark (default: llama3.1-8b)
  -t SECONDS   Duration for tier comparison tests (default: 120)
  -s SECONDS   Duration for storage saturation test (default: 180)
  -r SECONDS   Duration for production test (default: 180)
  -a SECONDS   Duration for autoscaling tests (default: 300)
  -w LIST      Comma-separated list of workloads to run
  -u USERS     Override baseline user count
  -U USERS     Override high-load user count
  -R           Enable RAG workload
  -D DOCS      Number of RAG documents (default: 10)
  -h           Show help
```

### Available Workloads

```bash
# Run only the storage isolation test
./kv-cache-wrapper.sh -w storage-only

# Run production and autoscaling tests
./kv-cache-wrapper.sh -w production,autoscale

# Run MLPerf submission tests
./kv-cache-wrapper.sh -w mlperf_submission
```

---

## Understanding Results

### Key Metrics

**Throughput (tokens/sec)**: How many tokens the system processes per second. Higher is better.

**Storage Throughput (tokens/sec)**: Raw I/O performance calculated from storage latency, not wall-clock time. This is the fairer metric for comparing storage tiers.

**End-to-End Latency**: Total time from request submission to completion. This is what users experience.

**Storage I/O Latency**: Time spent reading from and writing to storage tiers. This measures your hardware.

**Queue Wait Time**: Time requests spend waiting before processing begins. If this dominates, your system is overloaded.

**Cache Hit Rate**: Percentage of reads served from cache. Higher rates mean less storage pressure.

### Reading the Output

```
### STORAGE PERFORMANCE ASSESSMENT: PASS ###
  Criteria Passed: 4/4
  [PASS] NVMe Write P95 < 500ms: 45.20ms
  [PASS] NVMe Read P95 < 200ms: 123.45ms
  [PASS] CPU RAM P95 < 150ms: 12.30ms
  [PASS] Cache Hit Rate > 30%: 67.5%

### OVERALL PERFORMANCE ###
  Total Requests: 2847
  Total Tokens Generated: 489,231
  Avg Throughput: 1,630.77 tok/s
  Storage Throughput: 2,105.32 tok/s

### LATENCY BREAKDOWN ###
  End-to-End: mean 89.3ms, P50 45.2ms, P95 312.4ms
  Storage I/O: mean 23.1ms, P50 12.4ms, P95 89.2ms
```

---

## Understanding Excel Performance Metrics

The `--xlsx-output` option exports detailed performance metrics to Excel for analysis. This section provides a comprehensive reference for every metric in the export.

### Run Parameters (Configuration)

These columns record the benchmark configuration used for the run:

| Column | Description |
|--------|-------------|
| **Timestamp** | When the benchmark was executed (YYYY-MM-DD HH:MM:SS) |
| **Model** | Model configuration key (e.g., `llama3.1-8b`, `llama3.1-70b-instruct`) |
| **Num Users** | Number of concurrent simulated users |
| **Duration (s)** | Benchmark duration in seconds |
| **GPU Memory (GB)** | GPU VRAM budget allocated |
| **CPU Memory (GB)** | CPU RAM budget allocated |
| **Generation Mode** | Token generation simulation: `none`, `fast`, or `realistic` |
| **Performance Profile** | Pass/fail criteria: `latency` or `throughput` |
| **Multi-turn** | Whether multi-turn conversation caching was enabled |
| **Prefix Caching** | Whether system prompt prefix caching was enabled |
| **RAG Enabled** | Whether RAG workload simulation was enabled |
| **Autoscaling** | Whether workload autoscaling was enabled |
| **Seed** | Random seed for reproducibility |
| **Max Concurrent Allocs** | Limit on parallel cache allocations (0 = unlimited) |
| **Request Rate** | Target request rate in req/sec (0 = unlimited) |
| **Max Requests** | Stop after N requests (0 = use duration) |
| **Dataset Path** | Path to ShareGPT dataset if used |
| **Cache Dir** | Directory used for NVMe cache files |

---

### Throughput Metrics

| Metric | Unit | What It Measures | Interpretation |
|--------|------|------------------|----------------|
| **Total Requests** | count | Total inference requests completed | Higher = more work done. Compare across runs with same duration. |
| **Total Tokens** | count | Total tokens generated across all requests | Primary workload volume indicator. |
| **Elapsed Time (s)** | seconds | Actual wall-clock benchmark duration | May differ slightly from configured duration. |
| **Avg Throughput (tok/s)** | tokens/sec | `Total Tokens / Elapsed Time` | **Wall-clock throughput.** Includes all overheads (queue wait, generation simulation). **Primary metric when `gpu_mem=0` and `cpu_mem=0`.** |
| **Storage Throughput (tok/s)** | tokens/sec | `Total Tokens / Total Storage I/O Time` | **Pure storage throughput.** Excludes generation simulation time. Useful when `cpu_mem > 0` to isolate storage I/O. |
| **Requests/sec** | req/sec | `Total Requests / Elapsed Time` | Request processing rate. Higher = system handling more concurrent users efficiently. |

> **Which throughput metric to use?**
> - **When `gpu_mem=0` and `cpu_mem=0`**: Use **Avg Throughput (tok/s)** — all I/O hits the storage tier, so wall-clock throughput directly reflects storage performance.
> - **When `cpu_mem > 0`**: Use **Storage Throughput (tok/s)** to isolate storage I/O from CPU cache hits.
> - **For MLPerf submissions**: Use **Tier Storage Read/Write Bandwidth (GB/s)** as the primary comparison metric (see below).

---

### End-to-End Latency Metrics

End-to-end (E2E) latency measures the total time from request submission to completion, including queue wait, cache operations, and simulated generation time. **This is what users experience.**

| Metric | What It Measures |
|--------|------------------|
| **E2E Latency Mean (ms)** | Average latency across all requests. Sensitive to outliers. |
| **E2E Latency P50 (ms)** | Median latency. 50% of requests complete within this time. |
| **E2E Latency P95 (ms)** | 95th percentile. 95% of requests complete within this time. **Standard SLA metric.** |
| **E2E Latency P99 (ms)** | 99th percentile. 99% of requests complete within this time. **Tail latency indicator.** |
| **E2E Latency P99.9 (ms)** | 99.9th percentile (3 nines). Captures rare slow requests. |
| **E2E Latency P99.99 (ms)** | 99.99th percentile (4 nines). Extreme tail latency for SLA compliance. |

> **Interpreting percentiles:**
> - **P50** tells you the typical user experience.
> - **P95** is the standard for SLA definitions ("95% of requests under X ms").
> - **P99–P99.99** reveal tail latency issues that affect a small but real fraction of users.
> - Large gaps between P95 and P99 indicate inconsistent performance (investigate queue buildup or storage saturation).

---

### Storage I/O Latency Metrics

Storage latency measures only the time spent on cache read/write operations, excluding queue wait and generation simulation. **This isolates storage subsystem performance.**

| Metric | What It Measures |
|--------|------------------|
| **Storage Latency Mean (ms)** | Average storage I/O time across all operations. |
| **Storage Latency P50 (ms)** | Median storage I/O time. |
| **Storage Latency P95 (ms)** | 95th percentile storage I/O time. **Key metric for storage evaluation.** |
| **Storage Latency P99 (ms)** | 99th percentile storage I/O time. |
| **Storage Latency P99.9 (ms)** | 99.9th percentile storage I/O time. |
| **Storage Latency P99.99 (ms)** | 99.99th percentile storage I/O time. |

---

### Generation Latency Metrics

Generation latency measures the simulated GPU token generation time. Only meaningful when `--generation-mode` is `fast` or `realistic`.

| Metric | What It Measures |
|--------|------------------|
| **Gen Latency Mean (ms)** | Average simulated generation time per request. |
| **Gen Latency P50 (ms)** | Median generation time. |
| **Gen Latency P95 (ms)** | 95th percentile generation time. |
| **Gen Latency P99 (ms)** | 99th percentile generation time. |

> **Note:** With `--generation-mode none`, these values are all 0 (pure storage benchmark).

---

### Storage Tier Latency Breakdown (PRIMARY METRICS)

These metrics provide granular visibility into storage tier operations. The "storage" tier is device-agnostic—it could be NVMe, SATA SSD, CXL memory, or any block storage device. Each operation is decomposed into:

- **Total**: Complete operation time (Host + Device)
- **Device**: Actual storage I/O time (`np.save`/`np.load` with fsync) — **PRIMARY LATENCY METRIC**
- **Host**: CPU serialization/deserialization time

> **⭐ PRIMARY METRICS for MLPerf Storage Comparison:**
> - **Storage Tier Read Device P95 (ms)** — Raw storage read latency
> - **Storage Tier Write Device P95 (ms)** — Raw storage write latency
> - **Tier Storage Read Bandwidth (GB/s)** — Storage read throughput
> - **Tier Storage Write Bandwidth (GB/s)** — Storage write throughput
>
> **What Device Latency Measures:**
> ```
> Device Latency = [ OS/FS Queue ] + [ Block Layer ] + [ Driver ] + [ Physical I/O ]
> ```
> The **Storage Tier Read Device P95 (ms)** is the 95th percentile latency of reading one `.npy` file containing the KV cache data for a single cache entry (one request's token sequence). This captures tail latency—95% of reads complete faster than this value, so it reveals worst-case storage behavior under load.

#### Read Operations (Decode Phase)

| Metric | Component | What It Measures |
|--------|-----------|------------------|
| **Storage Tier Read Total P50–P99.99 (ms)** | Total | Complete read time including deserialization |
| **Storage Tier Read Device P50–P99.99 (ms)** | Device | **⭐ Raw storage read time (`np.load`) — PRIMARY** |
| **Storage Tier Read Host P50–P99.99 (ms)** | Host | NumPy array deserialization CPU time |

#### Write Operations (Prefill Phase)

| Metric | Component | What It Measures |
|--------|-----------|------------------|
| **Storage Tier Write Total P50–P99.99 (ms)** | Total | Complete write time including serialization |
| **Storage Tier Write Device P50–P99.99 (ms)** | Device | **⭐ Raw storage write time (`np.save` + fsync) — PRIMARY** |
| **Storage Tier Write Host P50–P99.99 (ms)** | Host | NumPy array serialization CPU time |

> **Diagnosing storage bottlenecks:**
> - If **Device >> Host**: Your storage device is the bottleneck. Consider faster storage (NVMe Gen5, CXL).
> - If **Host >> Device**: CPU serialization is the bottleneck. Consider faster CPU or memory bandwidth.
> - Typical ratio: Device should be 60-80% of Total for well-balanced systems.

---

### Cache Statistics

| Metric | Unit | What It Measures | Good Values |
|--------|------|------------------|-------------|
| **Cache Hit Rate** | ratio (0–1) | Fraction of reads served from cache vs. storage | Higher is better. 0.7+ with multi-turn enabled. |
| **Read/Write Ratio** | ratio | Total reads / Total writes | Higher indicates read-heavy workload (typical for decode phase). |
| **Total Read (GB)** | GB | Total data read from all tiers | Workload volume indicator. |
| **Total Write (GB)** | GB | Total data written to all tiers | Workload volume indicator. |

---

### Per-Tier I/O Volume

These metrics show data movement through each tier of the cache hierarchy:

| Metric | What It Measures |
|--------|------------------|
| **Tier GPU KV Bytes Written (GB)** | Data written to GPU VRAM tier |
| **Tier GPU KV Bytes Read (GB)** | Data read from GPU VRAM tier |
| **Tier CPU KV Bytes Written (GB)** | Data written to CPU RAM tier |
| **Tier CPU KV Bytes Read (GB)** | Data read from CPU RAM tier |
| **Tier Storage KV Bytes Written (GB)** | Data written to storage tier (NVMe, SATA, CXL, etc.) |
| **Tier Storage KV Bytes Read (GB)** | Data read from storage tier (NVMe, SATA, CXL, etc.) |

> **Analyzing tier distribution:**
> - High GPU/CPU reads with low storage reads = hot data fits in fast tiers (good!)
> - High storage reads = working set exceeds fast tier capacity (consider adding memory)
> - **Tier Storage KV Bytes Read** is a key MLPerf differentiation metric (100% win rate in discovery testing)

---

### Per-Tier Bandwidth (PRIMARY METRICS)

These metrics measure the actual throughput achieved on each tier. **Tier Storage Bandwidth is the primary metric for comparing storage devices.**

| Metric | Unit | What It Measures |
|--------|------|------------------|
| **Tier GPU Read Bandwidth (GB/s)** | GB/s | GPU VRAM read throughput |
| **Tier GPU Write Bandwidth (GB/s)** | GB/s | GPU VRAM write throughput |
| **Tier CPU Read Bandwidth (GB/s)** | GB/s | CPU RAM read throughput |
| **Tier CPU Write Bandwidth (GB/s)** | GB/s | CPU RAM write throughput |
| **Tier Storage Read Bandwidth (GB/s)** | GB/s | **⭐ Storage tier read throughput — PRIMARY** |
| **Tier Storage Write Bandwidth (GB/s)** | GB/s | **⭐ Storage tier write throughput — PRIMARY** |

> **Expected bandwidth ranges:**
> - **GPU**: 500–2000 GB/s (HBM2e/HBM3)
> - **CPU**: 50–200 GB/s (DDR4/DDR5)
> - **Storage (NVMe Gen4)**: 3–7 GB/s
> - **Storage (NVMe Gen5)**: 10–14 GB/s
> - **Storage (SATA SSD)**: 0.4–0.6 GB/s
> - **Storage (CXL Memory)**: 30–50 GB/s

---

### Tier Entry Distribution

| Metric | What It Measures |
|--------|------------------|
| **GPU Entries** | Number of KV cache entries currently in GPU VRAM |
| **CPU Entries** | Number of KV cache entries currently in CPU RAM |
| **Storage Entries** | Number of KV cache entries currently on storage tier |

> **Interpreting entry counts:**
> - Most entries should be in the fastest available tier for optimal performance.
> - High **Storage Entries** with low **GPU/CPU Entries** indicates memory pressure.
> - When `gpu_mem=0` and `cpu_mem=0`, all entries will be in **Storage Entries**.

---

### Multi-turn Statistics

| Metric | What It Measures |
|--------|------------------|
| **Multi-turn Hit Rate** | Fraction of requests that reused context from previous conversation turns |

> **Interpreting Multi-turn Hit Rate:**
> - **High (0.6+)**: Effective conversation context caching. Most requests are follow-ups that reuse existing KV cache entries, reducing redundant computation. Typical for chatbot/assistant workloads.
> - **Low (<0.3)**: Indicates one or more of the following:
>   - `--disable-multi-turn` is enabled (expected: 0.0)
>   - Workload has high conversation turnover (users start new conversations frequently)
>   - Single-shot API usage pattern (each request is independent)
>   - Memory pressure causing cache eviction before context reuse
>   - Short benchmark duration (not enough time for multi-turn patterns to emerge)
>
> **Note:** A low multi-turn hit rate is **not inherently bad**—it depends on your use case. For storage stress testing, low hit rates force more I/O which is often the goal.

---

### Using Excel Metrics for Analysis

**⭐ Primary Metrics for MLPerf Storage Comparison:**

| Metric | When to Use | Why |
|--------|-------------|-----|
| **Tier Storage Read Bandwidth (GB/s)** | Always | Direct measure of storage read throughput |
| **Tier Storage Write Bandwidth (GB/s)** | Always | Direct measure of storage write throughput |
| **Storage Tier Read Device P95 (ms)** | Always | Raw storage read latency (excludes CPU overhead) |
| **Storage Tier Write Device P95 (ms)** | Always | Raw storage write latency (excludes CPU overhead) |
| **Avg Throughput (tok/s)** | When `gpu_mem=0, cpu_mem=0` | Wall-clock throughput equals storage throughput |

**Comparing storage devices:**
1. Run identical benchmarks on each device with `--gpu-mem-gb 0 --cpu-mem-gb 0`
2. Compare **primary metrics**: Tier Storage Read/Write Bandwidth, Storage Tier Device P95 latencies
3. Use **Avg Throughput (tok/s)** as the overall performance score

**Diagnosing performance issues:**
1. Check **Storage Tier Device P95** vs **Storage Tier Host P95**
2. If Device >> Host: Storage device is the bottleneck
3. If Host >> Device: CPU serialization is the bottleneck

**Validating cache configuration:**
1. Check **Cache Hit Rate** and **Multi-turn Hit Rate**
2. Low hit rates with enabled caching: Working set too large for memory budget
3. Compare **Tier Storage KV Bytes Read** across configurations

---

## Unit Testing

This package includes a comprehensive pytest-based test suite to verify core functionality without running the full benchmark.

### Running Tests

```bash
# Run all tests with verbose output
pytest test_kv_cache.py -v

# Run with shorter traceback
pytest test_kv_cache.py -v --tb=short

# Run specific test class
pytest test_kv_cache.py -k "TestModelConfig" -v

# Run only CPU tests (skip GPU tests if no CUDA)
pytest test_kv_cache.py -v -m "not skipif"
```

### Test Coverage

The test suite covers 23 component categories with ~170+ individual tests:

| Test Class | Tests | Coverage |
|------------|-------|----------|
| `TestConfigLoader` | 5 | YAML loading, strict schema validation, error on unknown keys, nested key access |
| `TestCfgHelper` | 4 | Global `cfg()` helper, defaults when config not loaded, list value extraction |
| `TestModelConfig` | 4 | Model configurations, KV cache size per token calculations, dtype handling |
| `TestInferenceRequest` | 5 | Request dataclass, automatic cache key generation, phase handling, QoS assignment |
| `TestQoSProfiles` | 5 | QoS levels (interactive/responsive/batch), SLA targets, priority ordering, p999/p9999 extended metrics |
| `TestKVCacheGenerator` | 4 | Reproducible generation with seeds, correct tensor shapes, dtype consistency, precomputed buffers |
| `TestCPUMemoryBackend` | 4 | Write/read/delete/clear operations, timing metadata, data integrity |
| `TestNVMeBackend` | 5 | File I/O operations, .npy format handling, metadata persistence, temp directory cleanup |
| `TestGPUMemoryBackend` | 4 | CUDA tensor placement, device memory management (skipped without GPU) |
| `TestConversationManager` | 4 | Multi-turn conversation tracking, cache key management, LRU eviction |
| `TestUserSimulator` | 3 | User profile generation from templates, QoS distribution validation |
| `TestMultiTierCache` | 5 | CPU-only allocation paths, cache access patterns, tier selection logic |
| `TestMultiTierCacheWithGPU` | 4 | GPU tier allocation, waterfall eviction GPU→CPU→NVMe (skipped without GPU) |
| `TestXLSXExport` | 4 | CSV fallback, Excel export, run parameters embedding (skipped without pandas) |
| `TestEnums` | 3 | InferencePhase, GenerationMode, QoSLevel enum values |
| `TestTierLogic` | 3 | Tier ordering (GPU→CPU→NVMe), usage tracking, limit validation |
| `TestConfigDrivenConversationManager` | 2 | ConversationManager respects config.yaml settings |
| `TestConfigDrivenUserSimulator` | 3 | UserSimulator reads user_templates from config |
| `TestStatsNamingConvention` | 2 | `storage_*` naming convention validation for metrics keys |
| `TestGPUMemoryBackendEvictionCallback` | 2 | GPU eviction callback invocation and data passing (skipped without GPU) |
| `TestValidateArgs` | 24 | CLI argument validation: positive integers, ranges, memory limits, cache directory safety, forbidden prefixes |
| `TestPerTierPhaseMetrics` | 7 | Per-tier (GPU/CPU/Storage) KV bytes read/written tracking during prefill/decode phases |
| `TestPerTierPhaseMetricsWithGPU` | 4 | GPU tier metrics tracking, phase-aware read/write separation (skipped without GPU) |

### Expected Runtime

- **Without GPU**: ~5-10 seconds
- **With GPU**: ~10-15 seconds

GPU tests are automatically skipped if CUDA is not available.

---

## Excel Export

The benchmark can export results directly to Excel or CSV format for analysis.

### Basic Usage

```bash
python3 kv-cache.py \
    --config config.yaml \
    --model llama3.1-8b \
    --num-users 50 \
    --duration 120 \
    --gpu-mem-gb 0 \
    --cpu-mem-gb 4 \
    --seed 42 \
    --output results.json \
    --xlsx-output results.xlsx
```

### Output Format

The Excel file contains a single row with all key metrics:

| Column | Description |
|--------|-------------|
| Model | Model configuration used |
| Num Users | Concurrent user count |
| Duration (s) | Benchmark duration |
| GPU Mem (GB) | GPU memory budget |
| CPU Mem (GB) | CPU memory budget |
| Total Requests | Requests completed |
| Total Tokens | Tokens processed |
| Avg Throughput (tok/s) | Wall-clock throughput |
| Storage Throughput (tok/s) | Storage I/O throughput |
| Cache Hit Rate | Percentage of cache hits |
| E2E Latency P95 (ms) | End-to-end 95th percentile |
| Storage IO P95 (ms) | Storage I/O 95th percentile |

### Fallback Behavior

- **With openpyxl**: Exports to `.xlsx` format
- **Without openpyxl**: Falls back to `.csv` format
- **Without pandas**: Export is skipped with a warning

---

## MLPerf Submission Guidelines

For official MLPerf v3.0 storage submissions, use these standardized commands. **These invocations have been validated through extensive discovery testing** (1,411 Fast system tests, 268 Slow system tests comparing 14,000 MB/s vs 3,000 MB/s storage).

### Discovery Test Key Findings

| Finding | Impact |
|---------|--------|
| **Metric selection depends on cpu_mem** | Storage Throughput shows only 1.1x at cpu_mem=0GB but 2.2x at cpu_mem=4GB |
| **Best models for differentiation** | llama3.1-8b and mistral-7b show 2.31x ratio |
| **High variance observed** | CV 50-125%, requires 3-5 trials minimum |
| **100% win rate metrics** | Decode Bytes Read and Wall-Clock Throughput at cpu_mem=0GB |

### Option 1: Maximum Storage Stress (cpu_mem=0GB)

Use when you want to stress test NVMe and measure I/O volume differentiation.

**Primary Metrics:** Decode Bytes Read (2.62x differentiation), Wall-Clock Throughput (2.43x differentiation)

```bash
# MLPerf v3.0: Maximum Storage Stress Test (8B Model)
# Run 3-5 trials for statistical significance
for trial in 1 2 3 4 5; do
    python3 kv-cache.py \
        --config config.yaml \
        --model llama3.1-8b \
        --num-users 200 \
        --duration 300 \
        --gpu-mem-gb 0 \
        --cpu-mem-gb 0 \
        --max-concurrent-allocs 16 \
        --generation-mode none \
        --cache-dir /mnt/nvme \
        --seed 42 \
        --output mlperf_v3_stress_8b_trial${trial}.json
done
```

**⚠️ Important:** At cpu_mem=0GB, do NOT use Storage Throughput as your primary metric—use Decode Bytes Read or Wall-Clock Throughput instead.

### Option 2: Storage Throughput Focus (cpu_mem=4GB)

Use when you want Storage Throughput (tok/s) as your primary metric.

**Primary Metric:** Storage Throughput (2.2x differentiation, 97% win rate)

```bash
# MLPerf v3.0: Storage Throughput Test (8B Model)
for trial in 1 2 3 4 5; do
    python3 kv-cache.py \
        --config config.yaml \
        --model llama3.1-8b \
        --num-users 100 \
        --duration 300 \
        --gpu-mem-gb 0 \
        --cpu-mem-gb 4 \
        --max-concurrent-allocs 0 \
        --generation-mode none \
        --cache-dir /mnt/nvme \
        --seed 42 \
        --output mlperf_v3_throughput_8b_trial${trial}.json
done
```

### Option 3: Large Model Submission (70B)

For maximum per-request storage stress (2.5× larger KV cache per token: 320 KB vs 128 KB):

```bash
# MLPerf v3.0: Large Model Storage Stress
for trial in 1 2 3; do
    python3 kv-cache.py \
        --config config.yaml \
        --model llama3.1-70b-instruct \
        --num-users 70 \
        --duration 300 \
        --gpu-mem-gb 0 \
        --cpu-mem-gb 0 \
        --max-concurrent-allocs 4 \
        --generation-mode none \
        --cache-dir /mnt/nvme \
        --seed 42 \
        --output mlperf_v3_stress_70b_trial${trial}.json
done
```

### Critical Parameters (Discovery-Validated)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **--config config.yaml** | Required | Ensures consistent internal settings |
| **--seed 42** | Required | Reproducibility across systems |
| **--gpu-mem-gb 0** | Required | Isolates storage performance |
| **--cpu-mem-gb** | 0 or 4 | 0GB for max stress (use I/O volume metrics), 4GB for Storage Throughput metric |
| **--max-concurrent-allocs** | 0, 4, or 16 | 0 for throughput, 16 for stress testing |
| **--generation-mode** | none or realistic | none for pure I/O, realistic for production simulation |
| **--num-users** | 100-200 | Differentiation stable across range; higher = more throughput |
| **--duration** | 300-600 | 5-10 minutes for stable metrics |

### Trial Requirements

| User Count | Variance (CV) | Minimum Trials |
|------------|---------------|----------------|
| 10 users | ~52% | 3 |
| 50-100 users | ~115-125% | 3-5 |
| 200 users | ~110-120% | 3-5 |

Report **median** rather than mean for publication-quality results.

---

## Troubleshooting

### Out of Memory Errors

Reduce the number of concurrent users or limit parallel allocations:

```bash
python3 kv-cache.py --config config.yaml ... --max-concurrent-allocs 50
```

### Benchmark Hangs

The system may be thrashing. Reduce users or increase memory budgets.

### Poor Cache Hit Rates

Low hit rates indicate your working set exceeds available fast memory. Either:
- Increase GPU/CPU memory budgets
- Reduce user count
- Accept that cold data will hit storage

### Results Vary Between Runs

Use the `--seed` flag for reproducible results.

### Configuration Validation Errors

If you see "Unknown configuration key" errors, check your `config.yaml` for typos. The benchmark uses strict schema validation to prevent silent misconfigurations.

---

## Files in This Package

- `kv-cache.py`: Main benchmark implementation with ShareGPT support
- `config.yaml`: YAML configuration file for internal parameters
- `test_kv_cache.py`: Pytest unit test suite
- `requirements.txt`: Python dependencies
- `README.md`: This documentation
- `MLperf v3 KV cache proposal.md`: Detailed technical documentation

---

## License

Apache License 2.0

---

## Contact

For questions or feedback, open an issue on the repository or contact the MLPerf Storage Working Group.
