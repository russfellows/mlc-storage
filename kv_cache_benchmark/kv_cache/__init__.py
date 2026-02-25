"""
KV Cache Benchmark v3.0 — modular package.

Re-exports all public symbols so existing code can do:
    from kv_cache import MultiTierCache, IntegratedBenchmark, ...
"""

# Compatibility flags
from kv_cache._compat import (
    HAS_CUPY, HAS_YAML, HAS_TORCH, HAS_TIKTOKEN,
    CUPY_AVAILABLE, YAML_AVAILABLE, TORCH_AVAILABLE, TIKTOKEN_AVAILABLE,
    HAS_PANDAS, PANDAS_AVAILABLE,
    HAS_OPENPYXL, OPENPYXL_AVAILABLE,
    cp,
)

# Configuration
from kv_cache.config import (
    ConfigLoader,
    cfg,
    get_config,
    set_config,
)

# Core data models
from kv_cache.models import (
    ModelConfig,
    MODEL_CONFIGS,
    InferencePhase,
    GenerationMode,
    GENERATION_TIMING,
    QoSLevel,
    QoSSLA,
    QOS_PROFILES,
    get_qos_profiles,
    UserProfile,
    InferenceRequest,
)

# Conversation management
from kv_cache.conversation import (
    ConversationState,
    ConversationManager,
)

# Prefix caching
from kv_cache.prefix_cache import (
    PrefixType,
    PrefixCacheEntry,
    PrefixMatcher,
    PrefixCacheManager,
)

# RAG workload
from kv_cache.rag import (
    RAGChunk,
    RAGDocument,
    RAGQuery,
    RAGDocumentManager,
)

# Storage backends
from kv_cache.backends import (
    StorageBackend,
    CPUMemoryBackend,
    NVMeBackend,
)

# GPU backend is optional (requires CUDA)
try:
    from kv_cache.backends import GPUMemoryBackend
except Exception:
    pass

# Core cache engine
from kv_cache.cache import (
    KVCacheGenerator,
    MultiTierCache,
)

# Monitoring and autoscaling
from kv_cache.monitoring import (
    StorageMetrics,
    StorageMonitor,
    WorkloadAutoscaler,
    QoSMonitor,
)

# Workload generation and validation
from kv_cache.workload import (
    RealTraceEntry,
    ValidationEngine,
    UserSimulator,
    ShareGPTDatasetLoader,
    validate_args,
    MAX_USERS,
    MAX_DURATION_SECONDS,
    MAX_GPU_MEMORY_GB,
    MAX_CPU_MEMORY_GB,
    FORBIDDEN_CACHE_PREFIXES,
)

# Benchmark orchestrator
from kv_cache.benchmark import IntegratedBenchmark

# CLI
from kv_cache.cli import (
    export_results_to_xlsx,
    main,
)

__all__ = [
    # Compat flags
    'HAS_CUPY', 'HAS_YAML', 'HAS_TORCH', 'HAS_TIKTOKEN',
    'CUPY_AVAILABLE', 'YAML_AVAILABLE', 'TORCH_AVAILABLE', 'TIKTOKEN_AVAILABLE',
    'HAS_PANDAS', 'PANDAS_AVAILABLE', 'HAS_OPENPYXL', 'OPENPYXL_AVAILABLE',
    'cp',
    # Config
    'ConfigLoader', 'cfg', 'get_config', 'set_config',
    # Models
    'ModelConfig', 'MODEL_CONFIGS',
    'InferencePhase', 'GenerationMode', 'GENERATION_TIMING',
    'QoSLevel', 'QoSSLA', 'QOS_PROFILES', 'get_qos_profiles',
    'UserProfile', 'InferenceRequest',
    # Conversation
    'ConversationState', 'ConversationManager',
    # Prefix cache
    'PrefixType', 'PrefixCacheEntry', 'PrefixMatcher', 'PrefixCacheManager',
    # RAG
    'RAGChunk', 'RAGDocument', 'RAGQuery', 'RAGDocumentManager',
    # Backends
    'StorageBackend', 'GPUMemoryBackend', 'CPUMemoryBackend', 'NVMeBackend',
    # Cache engine
    'KVCacheGenerator', 'MultiTierCache',
    # Monitoring
    'StorageMetrics', 'StorageMonitor', 'WorkloadAutoscaler', 'QoSMonitor',
    # Workload
    'RealTraceEntry', 'ValidationEngine', 'UserSimulator', 'ShareGPTDatasetLoader',
    'validate_args', 'MAX_USERS', 'MAX_DURATION_SECONDS',
    'MAX_GPU_MEMORY_GB', 'MAX_CPU_MEMORY_GB', 'FORBIDDEN_CACHE_PREFIXES',
    # Benchmark
    'IntegratedBenchmark',
    # CLI
    'export_results_to_xlsx', 'main',
]
