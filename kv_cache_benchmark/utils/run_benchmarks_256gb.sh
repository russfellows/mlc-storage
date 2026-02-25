#!/usr/bin/env bash
# =============================================================================
# MLPerf v3.0 KV Cache Benchmark Runner (256GB RAM Safe)
# Kingston Digital, 2025 — Licensed under Apache 2.0
#
# Memory-safe version for systems with 256GB RAM.
# Optimized for STORAGE BENCHMARKING: cpu_mem=0, gpu_mem=0 (NVMe-only)
#
# Includes: stress, throughput, prefill-only, decode-only, and RAG suites.
#
# Usage:
#   ./run_benchmarks_256gb.sh                              # defaults: 3 trials, /mnt/nvme
#   ./run_benchmarks_256gb.sh --trials 1 --cache-dir /mnt/ssd
#   ./run_benchmarks_256gb.sh --suites "prefill decode"    # only run prefill and decode suites
#   ./run_benchmarks_256gb.sh --suites rag                 # only run RAG suite
#   ./run_benchmarks_256gb.sh --models "llama3.1-8b"       # single model
#
# Available suites: stress, throughput, prefill, decode, rag
# =============================================================================
set -euo pipefail

# ─── Defaults (tuned for 256GB RAM, NVMe-only storage testing) ───────────────
TRIALS=3
CACHE_DIR="/mnt/nvme"
DURATION=300
SEED=42
SUITES="stress throughput prefill decode rag"
MODELS=""  # empty = all models
KV_CACHE_CMD="kv-cache"
RESULTS_DIR="results_256gb"

# =============================================================================
# MEMORY BUDGET CALCULATION (256GB system, ~200GB usable for benchmark)
# =============================================================================
# KV cache bytes per token (from config.yaml, verified against HuggingFace):
#   - llama2-7b:           524,288 bytes (500 KB)   ← MHA, largest per-token cache
#   - llama3.1-70b:        327,680 bytes (313 KB)   ← GQA, efficient
#   - qwen3-32b:           262,144 bytes (250 KB)   ← GQA (head_dim=128 explicit)
#   - llama3.1-8b:         131,072 bytes (125 KB)   ← GQA, efficient
#   - mistral-7b:          131,072 bytes (125 KB)   ← GQA
#   - gpt-oss-120b:         73,728 bytes (70 KB)    ← MoE (head_dim=64 explicit)
#   - deepseek-v3:          70,272 bytes (67 KB)    ← MLA compressed (kv_lora_rank=512 + rope=64)
#   - gpt-oss-20b:          49,152 bytes (47 KB)    ← MoE (head_dim=64 explicit)
#
# Peak RAM ≈ num_users × avg_context_tokens × bytes_per_token × in_flight_factor
# With max_concurrent_allocs=N, in_flight_factor ≈ min(N, num_users)
#
# Safe configurations for 256GB (targeting ~150GB peak to leave headroom):
#   - llama2-7b:    30 users × 4K × 500KB × 8 allocs  = 49 GB peak  ✓
#   - llama3.1-70b: 40 users × 4K × 313KB × 8 allocs  = 41 GB peak  ✓
#   - qwen3-32b:    80 users × 4K × 250KB × 8 allocs  = 64 GB peak  ✓
#   - llama3.1-8b:  100 users × 4K × 125KB × 16 allocs = 82 GB peak ✓
#   - mistral-7b:   100 users × 4K × 125KB × 16 allocs = 82 GB peak ✓
#   - deepseek-v3:  150 users × 4K × 67KB × 16 allocs = 66 GB peak  ✓  (MLA compressed)
# =============================================================================

# ─── Parse arguments ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --trials)       TRIALS="$2";      shift 2 ;;
        --cache-dir)    CACHE_DIR="$2";   shift 2 ;;
        --duration)     DURATION="$2";    shift 2 ;;
        --seed)         SEED="$2";        shift 2 ;;
        --suites)       SUITES="$2";      shift 2 ;;
        --models)       MODELS="$2";      shift 2 ;;
        --results-dir)  RESULTS_DIR="$2"; shift 2 ;;
        --help|-h)
            head -16 "$0" | tail -10
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ─── All models from config.yaml (storage benchmark selection) ───────────────
# Ordered by KV cache size (largest first) for progressive storage stress
ALL_MODELS=(
    llama2-7b             # 500 KB/token  - MHA baseline (no GQA), largest per-token
    llama3.1-70b-instruct # 313 KB/token  - Large GQA model
    qwen3-32b             # 250 KB/token  - Medium GQA model (head_dim=128 explicit)
    llama3.1-8b           # 125 KB/token  - Standard GQA model
    mistral-7b            # 125 KB/token  - Standard GQA model
    gpt-oss-120b          # 70 KB/token   - MoE (head_dim=64 explicit)
    deepseek-v3           # 67 KB/token   - MLA compressed (kv_lora_rank=512+rope=64)
    gpt-oss-20b           # 47 KB/token   - MoE (head_dim=64 explicit)
)

# Use user-specified models or full suite
if [[ -n "$MODELS" ]]; then
    read -ra MODEL_LIST <<< "$MODELS"
else
    MODEL_LIST=("${ALL_MODELS[@]}")
fi

# ─── Model classification and RAM-safe parameters ────────────────────────────
# Returns: users max_allocs cpu_mem gpu_mem
# ALL configurations use cpu_mem=0 gpu_mem=0 for pure storage benchmarking
get_model_params() {
    local model="$1"
    local suite="$2"
    
    # Model-specific safe parameters for 256GB RAM
    # Format: users max_allocs cpu_mem gpu_mem
    case "$model" in
        deepseek-v3)
            # 67 KB/token (MLA compressed: kv_lora_rank=512 + qk_rope_head_dim=64)
            # 150 users × 4K × 67KB = 40GB (with allocs=16)
            case "$suite" in
                stress)     echo "150 16 0 0" ;;
                throughput) echo "120 16 0 0" ;;
                prefill)    echo "180 16 0 0" ;;
                decode)     echo "120 16 0 0" ;;
                rag)        echo "100 8 0 0" ;;
            esac
            ;;
        llama2-7b)
            # 512 KB/token - MHA (no GQA), larger than 8B GQA models
            # 30 users × 4K × 512KB = 61GB (with allocs=8)
            case "$suite" in
                stress)     echo "30 8 0 0" ;;
                throughput) echo "25 8 0 0" ;;
                prefill)    echo "35 8 0 0" ;;
                decode)     echo "25 8 0 0" ;;
                rag)        echo "20 4 0 0" ;;
            esac
            ;;
        llama3.1-70b-instruct)
            # 320 KB/token - Large but GQA-efficient
            # 40 users × 4K × 320KB = 51GB (with allocs=8)
            case "$suite" in
                stress)     echo "40 8 0 0" ;;
                throughput) echo "35 8 0 0" ;;
                prefill)    echo "50 8 0 0" ;;
                decode)     echo "35 8 0 0" ;;
                rag)        echo "25 4 0 0" ;;
            esac
            ;;
        qwen3-32b)
            # 250 KB/token - Medium GQA model (head_dim=128 explicit in HF config)
            # 50 users × 4K × 250KB = 50GB (with allocs=8)
            case "$suite" in
                stress)     echo "50 8 0 0" ;;
                throughput) echo "40 8 0 0" ;;
                prefill)    echo "60 8 0 0" ;;
                decode)     echo "40 8 0 0" ;;
                rag)        echo "30 4 0 0" ;;
            esac
            ;;
        llama3.1-8b|mistral-7b)
            # 128 KB/token - Efficient GQA models
            # 100 users × 4K × 128KB = 51GB (with allocs=16)
            case "$suite" in
                stress)     echo "100 16 0 0" ;;
                throughput) echo "80 16 0 0" ;;
                prefill)    echo "120 16 0 0" ;;
                decode)     echo "80 16 0 0" ;;
                rag)        echo "60 8 0 0" ;;
            esac
            ;;
        gpt-oss-120b|gpt-oss-20b|tiny-1b)
            # 48-73 KB/token - MoE models, very efficient KV cache
            # 150 users × 4K × 73KB = 44GB (with allocs=16)
            case "$suite" in
                stress)     echo "150 16 0 0" ;;
                throughput) echo "120 16 0 0" ;;
                prefill)    echo "180 16 0 0" ;;
                decode)     echo "120 16 0 0" ;;
                rag)        echo "100 8 0 0" ;;
            esac
            ;;
        *)
            # Unknown model - use conservative defaults
            echo "30 8 0 0"
            ;;
    esac
}

mkdir -p "${RESULTS_DIR}"

# ─── Detect block device under cache dir ──────────────────────────────────────
# Returns the whole-disk block device path (e.g., /dev/nvme0n1) for iostat.
# Handles both partitioned (nvme0n1p1 → nvme0n1) and whole-device mounts.
detect_block_device() {
    local dir="$1"
    local dev

    # Method 1: df-based detection
    dev=$(df "$dir" 2>/dev/null | tail -1 | awk '{print $1}')

    # Method 2: fallback to findmnt (more reliable for NVMe)
    if [[ -z "$dev" ]] || [[ ! -b "$dev" ]]; then
        dev=$(findmnt -no SOURCE "$dir" 2>/dev/null | head -1)
    fi

    if [[ -n "$dev" ]] && [[ -b "$dev" ]]; then
        # Try to resolve to parent (partition → whole disk)
        local base
        base=$(lsblk -no PKNAME "$dev" 2>/dev/null | head -1)
        if [[ -n "$base" ]]; then
            echo "/dev/${base}"
        else
            # No parent = already a whole-disk device (common for NVMe)
            echo "$dev"
        fi
    else
        echo ""
    fi
}

BLOCK_DEV=$(detect_block_device "${CACHE_DIR}")
# iostat needs just the device name (e.g., "nvme0n1"), not the full path
IOSTAT_DEV=""
if [[ -n "$BLOCK_DEV" ]]; then
    IOSTAT_DEV=$(basename "$BLOCK_DEV")
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${RESULTS_DIR}/benchmark_run_${TIMESTAMP}.log"

echo "================================================================" | tee "$LOG_FILE"
echo "MLPerf v3.0 KV Cache Benchmark (256GB RAM Safe)" | tee -a "$LOG_FILE"
echo "$(date)" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "Trials: ${TRIALS}  Cache Dir: ${CACHE_DIR}  Duration: ${DURATION}s" | tee -a "$LOG_FILE"
echo "Models: ${MODEL_LIST[*]}" | tee -a "$LOG_FILE"
echo "Suites: ${SUITES}" | tee -a "$LOG_FILE"
echo "System RAM: 256GB (parameters tuned for memory safety)" | tee -a "$LOG_FILE"
if [[ -n "$BLOCK_DEV" ]]; then
    echo "Block Device: ${BLOCK_DEV} (iostat target: ${IOSTAT_DEV})" | tee -a "$LOG_FILE"
else
    echo "Block Device: (not detected — iostat monitoring disabled)" | tee -a "$LOG_FILE"
    echo "  Tip: verify mount with 'findmnt ${CACHE_DIR}' or 'df ${CACHE_DIR}'" | tee -a "$LOG_FILE"
fi
echo "================================================================" | tee -a "$LOG_FILE"

run_trial() {
    local suite="$1" model="$2" trial="$3"
    local users="$4" max_allocs="$5" cpu_mem="$6" gpu_mem="$7"
    local extra_args="${8:-}"
    
    local tag="${suite}_${model}_trial${trial}"
    local json_out="${RESULTS_DIR}/mlperf_v3_${tag}.json"
    local xlsx_out="${RESULTS_DIR}/mlperf_v3_${tag}.xlsx"
    local iostat_out="${RESULTS_DIR}/mlperf_v3_${tag}_iostat.log"
    local iostat_pid=""

    echo "" | tee -a "$LOG_FILE"
    echo ">>> [${suite}] ${model} — trial ${trial}/${TRIALS}" | tee -a "$LOG_FILE"
    echo "    users=${users} cpu_mem=${cpu_mem}GB gpu_mem=${gpu_mem}GB max_allocs=${max_allocs}" | tee -a "$LOG_FILE"
    if [[ -n "$extra_args" ]]; then
        echo "    extra: ${extra_args}" | tee -a "$LOG_FILE"
    fi

    # Start iostat background monitor (use short device name for compatibility)
    if [[ -n "$IOSTAT_DEV" ]] && command -v iostat &>/dev/null; then
        iostat -mx "$IOSTAT_DEV" 1 > "$iostat_out" 2>&1 &
        iostat_pid=$!
        echo "    iostat PID ${iostat_pid} monitoring ${IOSTAT_DEV} -> ${iostat_out}" | tee -a "$LOG_FILE"
    elif [[ -z "$IOSTAT_DEV" ]]; then
        echo "    WARNING: No block device detected for ${CACHE_DIR} — iostat disabled" | tee -a "$LOG_FILE"
    fi

    # shellcheck disable=SC2086
    ${KV_CACHE_CMD} \
        --config config.yaml \
        --model "${model}" \
        --num-users "${users}" \
        --duration "${DURATION}" \
        --gpu-mem-gb "${gpu_mem}" \
        --cpu-mem-gb "${cpu_mem}" \
        --max-concurrent-allocs "${max_allocs}" \
        --generation-mode none \
        --cache-dir "${CACHE_DIR}" \
        --seed "${SEED}" \
        --output "${json_out}" \
        --xlsx-output "${xlsx_out}" \
        ${extra_args} \
        2>&1 | tee -a "$LOG_FILE"

    # Stop iostat
    if [[ -n "$iostat_pid" ]]; then
        kill "$iostat_pid" 2>/dev/null || true
        wait "$iostat_pid" 2>/dev/null || true
        echo "    ✓ iostat: ${iostat_out}" | tee -a "$LOG_FILE"
    fi

    echo "    ✓ JSON: ${json_out}" | tee -a "$LOG_FILE"
    echo "    ✓ XLSX: ${xlsx_out}" | tee -a "$LOG_FILE"
}

# ─── Suite 1: Storage Stress (cpu_mem=0, gpu_mem=0, NVMe-only) ───────────────
if [[ "$SUITES" == *"stress"* ]]; then
    echo "" | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"
    echo "SUITE: STORAGE STRESS (cpu=0GB, gpu=0GB, NVMe-only)" | tee -a "$LOG_FILE"
    echo "  Scenario: ALL KV cache I/O goes directly to NVMe" | tee -a "$LOG_FILE"
    echo "  Primary metrics: Read/Write Bandwidth, Device Latency" | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"

    for model in "${MODEL_LIST[@]}"; do
        read -r users max_allocs cpu_mem gpu_mem <<< "$(get_model_params "$model" stress)"
        for trial in $(seq 1 "$TRIALS"); do
            run_trial "stress" "$model" "$trial" "$users" "$max_allocs" "$cpu_mem" "$gpu_mem"
        done
    done
fi

# ─── Suite 2: Storage Throughput (cpu_mem=0, gpu_mem=0 for pure storage) ──────
if [[ "$SUITES" == *"throughput"* ]]; then
    echo "" | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"
    echo "SUITE: STORAGE THROUGHPUT (cpu=0GB, gpu=0GB)" | tee -a "$LOG_FILE"
    echo "  Scenario: Sustained storage throughput measurement" | tee -a "$LOG_FILE"
    echo "  Primary metric: Storage Throughput (GB/s)" | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"

    for model in "${MODEL_LIST[@]}"; do
        read -r users max_allocs cpu_mem gpu_mem <<< "$(get_model_params "$model" throughput)"
        for trial in $(seq 1 "$TRIALS"); do
            run_trial "throughput" "$model" "$trial" "$users" "$max_allocs" "$cpu_mem" "$gpu_mem"
        done
    done
fi

# ─── Suite 3: Prefill-Only (write-heavy, simulates prefill workers) ───────────
if [[ "$SUITES" == *"prefill"* ]]; then
    echo "" | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"
    echo "SUITE: PREFILL-ONLY (write-heavy, cpu=0GB, gpu=0GB)" | tee -a "$LOG_FILE"
    echo "  Scenario: Disaggregated inference — prefill worker" | tee -a "$LOG_FILE"
    echo "  Real-world: Prefill server computes KV, writes to storage" | tee -a "$LOG_FILE"
    echo "  I/O pattern: ~95% writes, minimal reads" | tee -a "$LOG_FILE"
    echo "  Primary metric: Write Bandwidth (GB/s)" | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"

    for model in "${MODEL_LIST[@]}"; do
        read -r users max_allocs cpu_mem gpu_mem <<< "$(get_model_params "$model" prefill)"
        for trial in $(seq 1 "$TRIALS"); do
            run_trial "prefill" "$model" "$trial" "$users" "$max_allocs" "$cpu_mem" "$gpu_mem" \
                "--prefill-only --disable-multi-turn --disable-prefix-caching"
        done
    done
fi

# ─── Suite 4: Decode-Only (read-heavy, simulates decode workers) ──────────────
if [[ "$SUITES" == *"decode"* ]]; then
    echo "" | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"
    echo "SUITE: DECODE-ONLY (read-heavy, cpu=0GB, gpu=0GB)" | tee -a "$LOG_FILE"
    echo "  Scenario: Disaggregated inference — decode worker" | tee -a "$LOG_FILE"
    echo "  Real-world: Decode server reads pre-computed KV from storage" | tee -a "$LOG_FILE"
    echo "  I/O pattern: ~100% reads from pre-populated cache" | tee -a "$LOG_FILE"
    echo "  Primary metric: Read Bandwidth (GB/s), Read Latency P99" | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"

    for model in "${MODEL_LIST[@]}"; do
        read -r users max_allocs cpu_mem gpu_mem <<< "$(get_model_params "$model" decode)"
        for trial in $(seq 1 "$TRIALS"); do
            run_trial "decode" "$model" "$trial" "$users" "$max_allocs" "$cpu_mem" "$gpu_mem" \
                "--decode-only"
        done
    done
fi

# ─── Suite 5: RAG Workload (mixed reads from document cache) ──────────────────
if [[ "$SUITES" == *"rag"* ]]; then
    echo "" | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"
    echo "SUITE: RAG WORKLOAD (cpu=0GB, gpu=0GB)" | tee -a "$LOG_FILE"
    echo "  Scenario: Retrieval-Augmented Generation" | tee -a "$LOG_FILE"
    echo "  Real-world: Each request retrieves 3-5 document chunks" | tee -a "$LOG_FILE"
    echo "  I/O pattern: Write doc embeddings once, read many times" | tee -a "$LOG_FILE"
    echo "  Primary metric: Read Bandwidth, Cache Hit Rate" | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"

    for model in "${MODEL_LIST[@]}"; do
        read -r users max_allocs cpu_mem gpu_mem <<< "$(get_model_params "$model" rag)"
        for trial in $(seq 1 "$TRIALS"); do
            run_trial "rag" "$model" "$trial" "$users" "$max_allocs" "$cpu_mem" "$gpu_mem" \
                "--enable-rag --rag-num-docs 50"
        done
    done
fi

echo "" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "All benchmarks complete — $(date)" | tee -a "$LOG_FILE"
echo "Results in: ${RESULTS_DIR}/" | tee -a "$LOG_FILE"
echo "Log: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Memory usage summary (256GB safe, cpu=0 gpu=0 storage-only):" | tee -a "$LOG_FILE"
echo "  Model              | KB/tok | Users | max_allocs | Peak RAM (est)" | tee -a "$LOG_FILE"
echo "  -------------------|--------|-------|------------|----------------" | tee -a "$LOG_FILE"
echo "  llama2-7b          |    500 |    30 |          8 | ~49 GB" | tee -a "$LOG_FILE"
echo "  llama3.1-70b       |    313 |    40 |          8 | ~41 GB" | tee -a "$LOG_FILE"
echo "  qwen3-32b          |    250 |    50 |          8 | ~50 GB" | tee -a "$LOG_FILE"
echo "  llama3.1-8b        |    125 |   100 |         16 | ~82 GB" | tee -a "$LOG_FILE"
echo "  mistral-7b         |    125 |   100 |         16 | ~82 GB" | tee -a "$LOG_FILE"
echo "  deepseek-v3 (MLA)  |     67 |   150 |         16 | ~66 GB" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "All tests use cpu_mem=0 gpu_mem=0 for pure NVMe storage benchmarking" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
