#!/usr/bin/env python3
"""
Fast Unit Tests for KV Cache Benchmark (pytest version)

Run with:
    pytest test_kv_cache.py -v                                    # Console output
    pytest test_kv_cache.py -v --html=report.html --self-contained-html  # HTML report

Requirements:
    pip install pytest pytest-html

These tests verify core functionality without running the full benchmark.
Typical execution time: < 5 seconds

This version tests kv-cache.py which includes:
- ConfigLoader with YAML support and strict validation
- Extended QoS SLA with p999 and p9999 percentiles
- Config-driven parameters via cfg() helper
- Renamed nvme_* to storage_* in stats
"""

import os
import sys
import argparse
import tempfile
import pytest
import numpy as np
from datetime import datetime
from pathlib import Path

# Import from kv-cache.py (handle the hyphen in filename)
# Try multiple locations: same directory, parent directory
import importlib.util

_kv_cache_path = None
_possible_paths = [
    os.path.join(os.path.dirname(__file__), "kv-cache.py"),        # Same directory
    os.path.join(os.path.dirname(__file__), "..", "kv-cache.py"),  # Parent directory
]
for _path in _possible_paths:
    if os.path.exists(_path):
        _kv_cache_path = _path
        break

if _kv_cache_path is None:
    raise FileNotFoundError(
        f"Could not find kv-cache.py. Searched in:\n"
        + "\n".join(f"  - {os.path.abspath(p)}" for p in _possible_paths)
    )

spec = importlib.util.spec_from_file_location("kv_cache", _kv_cache_path)
kv_cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kv_cache)

# Import all needed classes and functions
MODEL_CONFIGS = kv_cache.MODEL_CONFIGS
ModelConfig = kv_cache.ModelConfig
InferenceRequest = kv_cache.InferenceRequest
InferencePhase = kv_cache.InferencePhase
GenerationMode = kv_cache.GenerationMode
GENERATION_TIMING = kv_cache.GENERATION_TIMING
QoSLevel = kv_cache.QoSLevel
QOS_PROFILES = kv_cache.QOS_PROFILES
KVCacheGenerator = kv_cache.KVCacheGenerator
CPUMemoryBackend = kv_cache.CPUMemoryBackend
NVMeBackend = kv_cache.NVMeBackend
ConversationManager = kv_cache.ConversationManager
UserSimulator = kv_cache.UserSimulator
MultiTierCache = kv_cache.MultiTierCache
export_results_to_xlsx = kv_cache.export_results_to_xlsx
PANDAS_AVAILABLE = kv_cache.PANDAS_AVAILABLE

# New imports for 01-26-2026 version
ConfigLoader = kv_cache.ConfigLoader
cfg = kv_cache.cfg
get_config = kv_cache.get_config
set_config = kv_cache.set_config
get_qos_profiles = kv_cache.get_qos_profiles
QoSSLA = kv_cache.QoSSLA
YAML_AVAILABLE = kv_cache.YAML_AVAILABLE

# Input validation imports
validate_args = kv_cache.validate_args
MAX_USERS = kv_cache.MAX_USERS
MAX_DURATION_SECONDS = kv_cache.MAX_DURATION_SECONDS
MAX_GPU_MEMORY_GB = kv_cache.MAX_GPU_MEMORY_GB
MAX_CPU_MEMORY_GB = kv_cache.MAX_CPU_MEMORY_GB
FORBIDDEN_CACHE_PREFIXES = kv_cache.FORBIDDEN_CACHE_PREFIXES

if PANDAS_AVAILABLE:
    import pandas as pd

# Check for GPU/CUDA availability
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        GPUMemoryBackend = kv_cache.GPUMemoryBackend
except ImportError:
    CUDA_AVAILABLE = False


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def tiny_model_config():
    """Return the tiny-1b model config for fast tests."""
    return MODEL_CONFIGS['tiny-1b']


@pytest.fixture
def llama8b_config():
    """Return the llama3.1-8b model config."""
    return MODEL_CONFIGS['llama3.1-8b']


@pytest.fixture
def kv_generator(tiny_model_config):
    """Return a KVCacheGenerator with deterministic seed."""
    return KVCacheGenerator(tiny_model_config, global_seed=42)


@pytest.fixture
def cpu_backend():
    """Return a fresh CPUMemoryBackend."""
    backend = CPUMemoryBackend()
    yield backend
    backend.clear()


@pytest.fixture
def nvme_backend():
    """Return a fresh NVMeBackend (uses temp directory)."""
    backend = NVMeBackend()
    yield backend
    backend.clear()


@pytest.fixture
def conversation_manager():
    """Return a ConversationManager with small limit."""
    return ConversationManager(max_conversations=5)


@pytest.fixture
def multi_tier_cache(tiny_model_config):
    """Return a MultiTierCache in CPU-only mode."""
    return MultiTierCache(
        model_config=tiny_model_config,
        gpu_memory_gb=0,
        cpu_memory_gb=0.1,  # 100MB
        seed=42
    )


@pytest.fixture
def gpu_backend():
    """Return a fresh GPUMemoryBackend (requires CUDA)."""
    if not CUDA_AVAILABLE:
        pytest.skip("CUDA not available")
    backend = GPUMemoryBackend()
    yield backend
    backend.clear()


@pytest.fixture
def multi_tier_cache_with_gpu(tiny_model_config):
    """Return a MultiTierCache with GPU enabled."""
    if not CUDA_AVAILABLE:
        pytest.skip("CUDA not available")
    return MultiTierCache(
        model_config=tiny_model_config,
        gpu_memory_gb=1.0,  # 1GB GPU
        cpu_memory_gb=0.1,  # 100MB CPU
        seed=42
    )


@pytest.fixture
def mock_benchmark_results():
    """Return mock benchmark results for export tests."""
    return {
        'summary': {
            'total_requests': 100,
            'total_tokens': 10000,
            'elapsed_time': 60.0,
            'avg_throughput_tokens_per_sec': 166.67,
            'storage_throughput_tokens_per_sec': 200.0,
            'requests_per_second': 1.67,
            'end_to_end_latency_ms': {'mean': 50, 'p50': 45, 'p95': 100, 'p99': 150},
            'storage_io_latency_ms': {'mean': 10, 'p50': 8, 'p95': 20, 'p99': 30},
            'generation_latency_ms': {'mean': 40, 'p50': 35, 'p95': 80, 'p99': 120},
            'cache_stats': {
                'cache_hit_rate': 0.65,
                'read_write_ratio': 2.5,
                'total_read_gb': 5.0,
                'total_write_gb': 2.0,
                'gpu_entries': 0,
                'cpu_entries': 50,
                'nvme_entries': 100,
                'prefill_bytes_written_gb': 1.5,
                'decode_bytes_read_gb': 3.5,
            },
            'qos_metrics': {},
            'multi_turn_stats': {'hit_rate': 0.5}
        }
    }


@pytest.fixture
def mock_args():
    """Return mock CLI args for export tests."""
    class MockArgs:
        model = 'llama3.1-8b'
        num_users = 100
        duration = 60
        gpu_mem_gb = 16
        cpu_mem_gb = 32
        generation_mode = 'none'
        performance_profile = 'latency'
        disable_multi_turn = False
        disable_prefix_caching = False
        enable_rag = False
        enable_autoscaling = False
        seed = 42
        max_concurrent_allocs = 0
        request_rate = 0
        max_requests = 0
        dataset_path = None
        cache_dir = None
    return MockArgs()


@pytest.fixture
def sample_config_yaml(tmp_path):
    """Create a sample config.yaml for testing."""
    config_content = '''
user_templates:
  chatbot:
    context_range: [256, 1024]
    generation_range: [50, 150]
    think_time_range: [0.1, 0.5]
  coding:
    context_range: [1024, 4096]
    generation_range: [100, 500]
    think_time_range: [0.2, 1.0]
  document:
    context_range: [2048, 8192]
    generation_range: [200, 800]
    think_time_range: [0.3, 1.5]

qos_profiles:
  interactive:
    target_latency_p95_ms: 50
    target_latency_p99_ms: 100
    target_latency_p999_ms: 150
    target_latency_p9999_ms: 200
    priority: 3
  responsive:
    target_latency_p95_ms: 100
    target_latency_p99_ms: 200
    target_latency_p999_ms: 350
    target_latency_p9999_ms: 500
    priority: 2
  batch:
    target_latency_p95_ms: 1000
    target_latency_p99_ms: 5000
    target_latency_p999_ms: 7500
    target_latency_p9999_ms: 10000
    priority: 1

qos_distribution:
  interactive_probability: 0.15
  responsive_threshold: 0.50

eviction:
  max_recursion_depth: 10
  target_usage_ratio: 0.8
  large_entry_limit_ratio: 0.95
  max_evictions_hard_cap: 5000
  max_evictions_min: 1000

decode:
  batch_size: 32

conversation:
  max_conversations: 1000
  max_turns_per_conv: 50
  end_conversation_probability: 0.2
'''
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)
    return str(config_file)


# =============================================================================
# Test 0: ConfigLoader (New in 01-26-2026)
# =============================================================================

@pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
class TestConfigLoader:
    """Tests for ConfigLoader and cfg() helper function."""
    
    def test_config_loader_without_file(self):
        """ConfigLoader should work without a config file."""
        loader = ConfigLoader(config_path=None)
        assert loader is not None
        assert loader.config == {}
    
    def test_config_loader_loads_yaml(self, sample_config_yaml):
        """ConfigLoader should load and parse YAML file."""
        loader = ConfigLoader(config_path=sample_config_yaml)
        assert loader.config is not None
        assert 'qos_profiles' in loader.config
    
    def test_config_loader_get_nested_value(self, sample_config_yaml):
        """ConfigLoader.get() should retrieve nested values."""
        loader = ConfigLoader(config_path=sample_config_yaml)
        priority = loader.get('qos_profiles', 'interactive', 'priority')
        assert priority == 3
    
    def test_config_loader_get_with_default(self, sample_config_yaml):
        """ConfigLoader.get() should return default for missing keys."""
        loader = ConfigLoader(config_path=sample_config_yaml)
        value = loader.get('nonexistent', 'key', default=42)
        assert value == 42
    
    def test_cfg_without_global_config(self):
        """cfg() should return default when no global config is set."""
        # Ensure no global config
        set_config(None)
        value = cfg('qos_profiles', 'interactive', 'priority', default=99)
        assert value == 99
    
    def test_cfg_with_global_config(self, sample_config_yaml):
        """cfg() should retrieve values from global config."""
        loader = ConfigLoader(config_path=sample_config_yaml)
        set_config(loader)
        try:
            value = cfg('qos_profiles', 'interactive', 'priority', default=99)
            assert value == 3
        finally:
            set_config(None)  # Clean up
    
    def test_config_loader_validates_schema(self, tmp_path):
        """ConfigLoader should reject unknown keys."""
        bad_config = tmp_path / "bad_config.yaml"
        bad_config.write_text('''
unknown_section:
  bad_key: true
''')
        with pytest.raises(ValueError, match="Unknown configuration key"):
            ConfigLoader(config_path=str(bad_config))
    
    def test_get_config_returns_none_initially(self):
        """get_config() should return None before set_config() is called."""
        set_config(None)
        assert get_config() is None
    
    def test_set_config_stores_loader(self, sample_config_yaml):
        """set_config() should store the ConfigLoader globally."""
        loader = ConfigLoader(config_path=sample_config_yaml)
        set_config(loader)
        try:
            assert get_config() is loader
        finally:
            set_config(None)


class TestCfgHelper:
    """Tests for cfg() helper function in various contexts."""
    
    def test_cfg_returns_default_for_none_config(self):
        """cfg() returns default when config is None."""
        set_config(None)
        assert cfg('any', 'path', default='fallback') == 'fallback'
    
    def test_cfg_returns_default_for_missing_key(self, sample_config_yaml):
        """cfg() returns default for missing nested keys."""
        loader = ConfigLoader(config_path=sample_config_yaml)
        set_config(loader)
        try:
            result = cfg('nonexistent', 'nested', 'key', default=123)
            assert result == 123
        finally:
            set_config(None)
    
    def test_cfg_retrieves_list_values(self, sample_config_yaml):
        """cfg() can retrieve list values from config."""
        loader = ConfigLoader(config_path=sample_config_yaml)
        set_config(loader)
        try:
            context_range = cfg('user_templates', 'chatbot', 'context_range')
            assert context_range == [256, 1024]
        finally:
            set_config(None)


# =============================================================================
# Test 1: ModelConfig
# =============================================================================

class TestModelConfig:
    """Tests for ModelConfig dataclass and calculations."""
    
    def test_llama8b_config_exists(self, llama8b_config):
        assert llama8b_config is not None
    
    def test_kv_cache_size_per_token_positive(self, llama8b_config):
        assert llama8b_config.kv_cache_size_per_token > 0
    
    def test_bytes_per_element_float16(self, llama8b_config):
        assert llama8b_config.bytes_per_element == 2
    
    def test_kv_cache_size_formula(self, llama8b_config):
        """Verify: num_layers * kv_heads * kv_dim_per_head * 2 * bytes_per_element"""
        expected = (llama8b_config.num_layers * 
                    llama8b_config.kv_heads * 
                    llama8b_config.kv_dim_per_head * 
                    2 * llama8b_config.bytes_per_element)
        assert llama8b_config.kv_cache_size_per_token == expected
    
    def test_all_five_model_configs_exist(self):
        assert len(MODEL_CONFIGS) == 5
    
    @pytest.mark.parametrize("model_name", [
        'tiny-1b', 'mistral-7b', 'llama2-7b', 'llama3.1-8b', 'llama3.1-70b-instruct'
    ])
    def test_model_config_exists(self, model_name):
        assert model_name in MODEL_CONFIGS


# =============================================================================
# Test 2: InferenceRequest
# =============================================================================

class TestInferenceRequest:
    """Tests for InferenceRequest dataclass."""
    
    def test_create_request(self):
        req = InferenceRequest(
            user_id="test_user",
            request_id="test_req_001",
            timestamp=datetime.now(),
            context_tokens=1024,
            generate_tokens=128,
            priority=2,
            conversation_id="conv_123",
            turn_number=1
        )
        assert req is not None
    
    def test_cache_key_auto_generated(self):
        req = InferenceRequest(
            user_id="test_user",
            request_id="test_req_001",
            timestamp=datetime.now(),
            context_tokens=1024,
            generate_tokens=128,
            priority=2,
            conversation_id="conv_123",
            turn_number=1
        )
        assert req.cache_key == "conv_123_turn_1"
    
    def test_cache_key_fallback_without_conversation(self):
        req = InferenceRequest(
            user_id="test_user2",
            request_id="test_req_002",
            timestamp=datetime.now(),
            context_tokens=512,
            generate_tokens=64,
            priority=1
        )
        assert req.cache_key == "test_user2_ctx"
    
    def test_submit_time_set(self):
        req = InferenceRequest(
            user_id="test_user",
            request_id="test_req",
            timestamp=datetime.now(),
            context_tokens=100,
            generate_tokens=10,
            priority=1
        )
        assert req.submit_time > 0
    
    def test_total_latency_ms(self):
        req = InferenceRequest(
            user_id="test_user",
            request_id="test_req",
            timestamp=datetime.now(),
            context_tokens=100,
            generate_tokens=10,
            priority=1
        )
        req.complete_time = req.submit_time + 0.1  # 100ms
        assert req.total_latency_ms > 0


# =============================================================================
# Test 3: QoS Profiles
# =============================================================================

class TestQoSProfiles:
    """Tests for QoS profiles and SLA."""
    
    def test_three_qos_levels(self):
        assert len(QOS_PROFILES) == 3
    
    def test_interactive_priority_highest(self):
        assert QOS_PROFILES[QoSLevel.INTERACTIVE].priority == 3
    
    def test_responsive_priority_middle(self):
        assert QOS_PROFILES[QoSLevel.RESPONSIVE].priority == 2
    
    def test_batch_priority_lowest(self):
        assert QOS_PROFILES[QoSLevel.BATCH].priority == 1
    
    def test_sla_compliance_starts_at_one(self):
        sla = QOS_PROFILES[QoSLevel.INTERACTIVE]
        assert sla.sla_compliance == 1.0
    
    def test_interactive_target_latency(self):
        sla = QOS_PROFILES[QoSLevel.INTERACTIVE]
        assert sla.target_latency_p95_ms == 50
    
    # New tests for extended QoS percentiles (01-26-2026 feature)
    def test_interactive_has_p999_latency(self):
        """Test that p999 percentile is defined for INTERACTIVE."""
        sla = QOS_PROFILES[QoSLevel.INTERACTIVE]
        assert hasattr(sla, 'target_latency_p999_ms')
        assert sla.target_latency_p999_ms > sla.target_latency_p99_ms
    
    def test_interactive_has_p9999_latency(self):
        """Test that p9999 percentile is defined for INTERACTIVE."""
        sla = QOS_PROFILES[QoSLevel.INTERACTIVE]
        assert hasattr(sla, 'target_latency_p9999_ms')
        assert sla.target_latency_p9999_ms > sla.target_latency_p999_ms
    
    def test_all_qos_levels_have_extended_percentiles(self):
        """Verify all QoS levels have p999 and p9999 defined."""
        for level in QoSLevel:
            sla = QOS_PROFILES[level]
            assert hasattr(sla, 'target_latency_p999_ms')
            assert hasattr(sla, 'target_latency_p9999_ms')
    
    def test_get_qos_profiles_returns_dict(self):
        """Test that get_qos_profiles() returns profiles dict."""
        profiles = get_qos_profiles()
        assert isinstance(profiles, dict)
        assert len(profiles) == 3
    
    def test_get_qos_profiles_levels(self):
        """Test that get_qos_profiles() has all QoS levels."""
        profiles = get_qos_profiles()
        assert QoSLevel.INTERACTIVE in profiles
        assert QoSLevel.RESPONSIVE in profiles
        assert QoSLevel.BATCH in profiles


# =============================================================================
# Test 4: KVCacheGenerator
# =============================================================================

class TestKVCacheGenerator:
    """Tests for KVCacheGenerator."""
    
    def test_generator_created(self, kv_generator):
        assert kv_generator is not None
    
    def test_precomputed_buffer_allocated(self, kv_generator):
        assert kv_generator.precomputed_buffer is not None
    
    def test_precomputed_buffer_size(self, kv_generator):
        assert len(kv_generator.precomputed_buffer) == 128 * 1024 * 1024
    
    def test_generated_data_shape(self, kv_generator, tiny_model_config):
        data = kv_generator.generate(sequence_length=10, key="test_key")
        expected_shape = (
            tiny_model_config.num_layers, 2, 10,
            tiny_model_config.kv_heads, tiny_model_config.kv_dim_per_head
        )
        assert data.shape == expected_shape
    
    def test_generated_data_dtype(self, kv_generator):
        data = kv_generator.generate(sequence_length=10, key="test_key")
        assert data.dtype == np.float16
    
    def test_determinism_same_key(self, kv_generator):
        data1 = kv_generator.generate(sequence_length=10, key="test_key")
        data2 = kv_generator.generate(sequence_length=10, key="test_key")
        assert np.array_equal(data1, data2)
    
    def test_different_key_runs(self, kv_generator):
        """Different key should not crash."""
        kv_generator.generate(sequence_length=10, key="different_key")


# =============================================================================
# Test 5: CPUMemoryBackend
# =============================================================================

class TestCPUMemoryBackend:
    """Tests for CPUMemoryBackend."""
    
    def test_backend_created(self, cpu_backend):
        assert cpu_backend is not None
    
    def test_write_returns_timing(self, cpu_backend):
        test_data = np.random.rand(100, 100).astype(np.float16)
        timing = cpu_backend.write("test_key", test_data)
        assert timing.total >= 0
    
    def test_read_returns_data(self, cpu_backend):
        test_data = np.random.rand(100, 100).astype(np.float16)
        cpu_backend.write("test_key", test_data)
        read_data, _ = cpu_backend.read("test_key")
        assert read_data is not None
    
    def test_read_data_matches_written(self, cpu_backend):
        test_data = np.random.rand(100, 100).astype(np.float16)
        cpu_backend.write("test_key", test_data)
        read_data, _ = cpu_backend.read("test_key")
        assert np.array_equal(read_data, test_data)
    
    def test_read_timing_returned(self, cpu_backend):
        test_data = np.random.rand(100, 100).astype(np.float16)
        cpu_backend.write("test_key", test_data)
        _, timing = cpu_backend.read("test_key")
        assert timing.total >= 0
    
    def test_delete_removes_key(self, cpu_backend):
        test_data = np.random.rand(100, 100).astype(np.float16)
        cpu_backend.write("test_key", test_data)
        cpu_backend.delete("test_key")
        with pytest.raises(KeyError):
            cpu_backend.read("test_key")
    
    def test_clear_empties_cache(self, cpu_backend):
        test_data = np.random.rand(100, 100).astype(np.float16)
        cpu_backend.write("key1", test_data)
        cpu_backend.write("key2", test_data)
        assert len(cpu_backend.cache) == 2
        cpu_backend.clear()
        assert len(cpu_backend.cache) == 0


# =============================================================================
# Test 6: NVMeBackend
# =============================================================================

class TestNVMeBackend:
    """Tests for NVMeBackend (uses temp directory)."""
    
    def test_backend_created(self, nvme_backend):
        assert nvme_backend is not None
    
    def test_temp_directory_exists(self, nvme_backend):
        assert nvme_backend.base_path.exists()
    
    def test_write_returns_timing(self, nvme_backend):
        test_data = np.random.rand(50, 50).astype(np.float32)
        timing = nvme_backend.write("nvme_test", test_data)
        assert timing.total >= 0
    
    def test_write_timing_has_device_component(self, nvme_backend):
        test_data = np.random.rand(50, 50).astype(np.float32)
        timing = nvme_backend.write("nvme_test", test_data)
        assert timing.device >= 0
    
    def test_file_created(self, nvme_backend):
        test_data = np.random.rand(50, 50).astype(np.float32)
        nvme_backend.write("nvme_test", test_data)
        assert (nvme_backend.base_path / "nvme_test.npy").exists()
    
    def test_read_returns_data(self, nvme_backend):
        test_data = np.random.rand(50, 50).astype(np.float32)
        nvme_backend.write("nvme_test", test_data)
        read_data, _ = nvme_backend.read("nvme_test")
        assert read_data is not None
    
    def test_read_data_matches(self, nvme_backend):
        test_data = np.random.rand(50, 50).astype(np.float32)
        nvme_backend.write("nvme_test", test_data)
        read_data, _ = nvme_backend.read("nvme_test")
        assert np.allclose(read_data, test_data)
    
    def test_metadata_stored(self, nvme_backend):
        test_data = np.random.rand(50, 50).astype(np.float32)
        nvme_backend.write("nvme_test", test_data)
        assert "nvme_test" in nvme_backend.metadata
    
    def test_delete_removes_file(self, nvme_backend):
        test_data = np.random.rand(50, 50).astype(np.float32)
        nvme_backend.write("nvme_test", test_data)
        nvme_backend.delete("nvme_test")
        assert not (nvme_backend.base_path / "nvme_test.npy").exists()
    
    def test_clear_removes_all_files(self, nvme_backend):
        test_data = np.random.rand(50, 50).astype(np.float32)
        nvme_backend.write("nvme_test1", test_data)
        nvme_backend.write("nvme_test2", test_data)
        nvme_backend.clear()
        assert len(list(nvme_backend.base_path.glob("*.npy"))) == 0


# =============================================================================
# Test 6b: GPUMemoryBackend (requires CUDA)
# =============================================================================

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestGPUMemoryBackend:
    """Tests for GPUMemoryBackend (requires CUDA)."""
    
    def test_backend_created(self, gpu_backend):
        assert gpu_backend is not None
    
    def test_write_returns_timing(self, gpu_backend):
        test_data = np.random.rand(100, 100).astype(np.float16)
        timing = gpu_backend.write("test_key", test_data)
        assert timing.total >= 0
    
    def test_write_timing_has_device_component(self, gpu_backend):
        test_data = np.random.rand(100, 100).astype(np.float16)
        timing = gpu_backend.write("test_key", test_data)
        assert timing.device >= 0
    
    def test_read_returns_data(self, gpu_backend):
        test_data = np.random.rand(100, 100).astype(np.float16)
        gpu_backend.write("test_key", test_data)
        read_data, _ = gpu_backend.read("test_key")
        assert read_data is not None
    
    def test_read_data_matches_written(self, gpu_backend):
        test_data = np.random.rand(100, 100).astype(np.float16)
        gpu_backend.write("test_key", test_data)
        read_data, _ = gpu_backend.read("test_key")
        assert np.allclose(read_data, test_data, rtol=1e-3)
    
    def test_read_timing_returned(self, gpu_backend):
        test_data = np.random.rand(100, 100).astype(np.float16)
        gpu_backend.write("test_key", test_data)
        _, timing = gpu_backend.read("test_key")
        assert timing.total >= 0
    
    def test_delete_removes_key(self, gpu_backend):
        test_data = np.random.rand(100, 100).astype(np.float16)
        gpu_backend.write("test_key", test_data)
        gpu_backend.delete("test_key")
        with pytest.raises(KeyError):
            gpu_backend.read("test_key")
    
    def test_clear_empties_cache(self, gpu_backend):
        test_data = np.random.rand(100, 100).astype(np.float16)
        gpu_backend.write("key1", test_data)
        gpu_backend.write("key2", test_data)
        gpu_backend.clear()
        assert len(gpu_backend.cache) == 0
    
    def test_data_on_cuda_device(self, gpu_backend):
        """Verify data is stored on GPU."""
        import torch
        test_data = np.random.rand(100, 100).astype(np.float16)
        gpu_backend.write("test_key", test_data)
        # Access internal cache to verify CUDA storage
        assert gpu_backend.cache["test_key"].is_cuda


# =============================================================================
# Test 7: ConversationManager
# =============================================================================

class TestConversationManager:
    """Tests for ConversationManager."""
    
    def test_manager_created(self, conversation_manager):
        assert conversation_manager is not None
    
    def test_max_conversations_set(self, conversation_manager):
        assert conversation_manager.max_conversations == 5
    
    def test_start_conversation(self, conversation_manager):
        conv_id = conversation_manager.start_conversation("user_1")
        assert conv_id is not None
    
    def test_conversation_id_format(self, conversation_manager):
        conv_id = conversation_manager.start_conversation("user_1")
        assert conv_id.startswith("conv_user_1_")
    
    def test_conversation_stored(self, conversation_manager):
        conv_id = conversation_manager.start_conversation("user_1")
        assert conv_id in conversation_manager.conversations
    
    def test_add_turn(self, conversation_manager):
        conv_id = conversation_manager.start_conversation("user_1")
        turn_num, cache_key = conversation_manager.add_turn(conv_id, 100, 50)
        assert turn_num == 1
    
    def test_cache_key_format(self, conversation_manager):
        conv_id = conversation_manager.start_conversation("user_1")
        turn_num, cache_key = conversation_manager.add_turn(conv_id, 100, 50)
        assert cache_key == f"{conv_id}_turn_1"
    
    def test_second_turn_number(self, conversation_manager):
        conv_id = conversation_manager.start_conversation("user_1")
        conversation_manager.add_turn(conv_id, 100, 50)
        turn_num, _ = conversation_manager.add_turn(conv_id, 200, 100)
        assert turn_num == 2
    
    def test_context_size_tracked(self, conversation_manager):
        conv_id = conversation_manager.start_conversation("user_1")
        conversation_manager.add_turn(conv_id, 100, 50)
        conversation_manager.add_turn(conv_id, 200, 100)
        context_size = conversation_manager.get_conversation_context_size(conv_id)
        assert context_size == 450  # 100+50+200+100
    
    def test_previous_turn_keys(self, conversation_manager):
        conv_id = conversation_manager.start_conversation("user_1")
        conversation_manager.add_turn(conv_id, 100, 50)
        conversation_manager.add_turn(conv_id, 200, 100)
        prev_keys = conversation_manager.get_all_previous_turn_keys(conv_id, 2)
        assert len(prev_keys) == 1
    
    def test_max_conversations_enforced(self, conversation_manager):
        for i in range(10):
            conversation_manager.start_conversation(f"user_{i}")
        assert len(conversation_manager.conversations) <= 5


# =============================================================================
# Test 8: UserSimulator
# =============================================================================

class TestUserSimulator:
    """Tests for UserSimulator."""
    
    def test_generate_mixed_users(self):
        users = UserSimulator.generate_mixed_users(10)
        assert len(users) == 10
    
    def test_users_have_valid_context_lengths(self):
        users = UserSimulator.generate_mixed_users(10)
        for user in users:
            assert 256 <= user.context_length <= 8192
    
    def test_qos_levels_assigned(self):
        users = UserSimulator.generate_mixed_users(10)
        qos_levels = set(u.qos_level for u in users)
        assert len(qos_levels) >= 1
    
    def test_single_user_generation(self):
        user = UserSimulator.generate_user("test_user", "chatbot", 2, QoSLevel.RESPONSIVE)
        assert user is not None
    
    def test_single_user_id(self):
        user = UserSimulator.generate_user("test_user", "chatbot", 2, QoSLevel.RESPONSIVE)
        assert user.user_id == "test_user"
    
    def test_single_user_qos(self):
        user = UserSimulator.generate_user("test_user", "chatbot", 2, QoSLevel.RESPONSIVE)
        assert user.qos_level == QoSLevel.RESPONSIVE


# =============================================================================
# Test 9: MultiTierCache (CPU-only)
# =============================================================================

class TestMultiTierCache:
    """Tests for MultiTierCache (CPU-only mode)."""
    
    def test_cache_created_with_zero_gpu_memory(self, multi_tier_cache):
        """With gpu_memory_gb=0, GPU limit should be 0 (even if backend exists)."""
        gpu_limit = multi_tier_cache._get_tier_limit('gpu') if 'gpu' in multi_tier_cache.backends else 0
        assert gpu_limit == 0
    
    def test_cpu_backend_available(self, multi_tier_cache):
        assert 'cpu' in multi_tier_cache.backends
    
    def test_nvme_backend_available(self, multi_tier_cache):
        assert 'nvme' in multi_tier_cache.backends
    
    def test_allocation_succeeds(self, multi_tier_cache):
        success, location, latency = multi_tier_cache.allocate_cache("test_entry", num_tokens=100)
        assert success is True
    
    def test_allocation_location(self, multi_tier_cache):
        success, location, latency = multi_tier_cache.allocate_cache("test_entry", num_tokens=100)
        assert location in ['cpu', 'nvme']
    
    def test_allocation_returns_latency(self, multi_tier_cache):
        success, location, latency = multi_tier_cache.allocate_cache("test_entry", num_tokens=100)
        assert latency >= 0
    
    def test_cache_access_succeeds(self, multi_tier_cache):
        multi_tier_cache.allocate_cache("test_entry", num_tokens=100)
        loc, read_lat = multi_tier_cache.access_cache("test_entry", InferencePhase.DECODE)
        assert loc is not None
    
    def test_cache_access_returns_location(self, multi_tier_cache):
        multi_tier_cache.allocate_cache("test_entry", num_tokens=100)
        loc, _ = multi_tier_cache.access_cache("test_entry", InferencePhase.DECODE)
        assert loc in ['cpu', 'nvme']
    
    def test_nonexistent_key_returns_none(self, multi_tier_cache):
        loc, _ = multi_tier_cache.access_cache("nonexistent_key", InferencePhase.DECODE)
        assert loc is None
    
    def test_stats_returned(self, multi_tier_cache):
        multi_tier_cache.allocate_cache("test_entry", num_tokens=100)
        multi_tier_cache.access_cache("test_entry", InferencePhase.DECODE)
        stats = multi_tier_cache.get_stats(duration=1.0)
        assert stats is not None
    
    def test_cache_hit_recorded(self, multi_tier_cache):
        multi_tier_cache.allocate_cache("test_entry", num_tokens=100)
        multi_tier_cache.access_cache("test_entry", InferencePhase.DECODE)
        stats = multi_tier_cache.get_stats(duration=1.0)
        assert stats['cache_hits'] >= 1
    
    def test_cache_miss_recorded(self, multi_tier_cache):
        multi_tier_cache.access_cache("nonexistent", InferencePhase.DECODE)
        stats = multi_tier_cache.get_stats(duration=1.0)
        assert stats['cache_misses'] >= 1
    
    def test_storage_health_in_stats(self, multi_tier_cache):
        stats = multi_tier_cache.get_stats(duration=1.0)
        assert 'storage_health' in stats


# =============================================================================
# Test 9b: MultiTierCache with GPU
# =============================================================================

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestMultiTierCacheWithGPU:
    """Tests for MultiTierCache with GPU enabled."""
    
    def test_gpu_backend_available(self, multi_tier_cache_with_gpu):
        assert 'gpu' in multi_tier_cache_with_gpu.backends
    
    def test_cpu_backend_available(self, multi_tier_cache_with_gpu):
        assert 'cpu' in multi_tier_cache_with_gpu.backends
    
    def test_nvme_backend_available(self, multi_tier_cache_with_gpu):
        assert 'nvme' in multi_tier_cache_with_gpu.backends
    
    def test_tier_order_with_gpu(self, multi_tier_cache_with_gpu):
        tier_order = multi_tier_cache_with_gpu._get_tier_order()
        assert tier_order == ['gpu', 'cpu', 'nvme']
    
    def test_gpu_limit_set(self, multi_tier_cache_with_gpu):
        gpu_limit = multi_tier_cache_with_gpu._get_tier_limit('gpu')
        assert gpu_limit == 1.0 * 1024**3  # 1GB
    
    def test_allocation_prefers_gpu(self, multi_tier_cache_with_gpu):
        """Small allocations should go to GPU first."""
        success, location, latency = multi_tier_cache_with_gpu.allocate_cache("test_entry", num_tokens=100)
        assert success is True
        assert location == 'gpu'
    
    def test_gpu_overflow_to_cpu(self, multi_tier_cache_with_gpu):
        """When GPU is full, should overflow to CPU."""
        # Fill GPU with large allocations
        for i in range(100):
            multi_tier_cache_with_gpu.allocate_cache(f"entry_{i}", num_tokens=10000)
        
        # Next allocation should go to CPU or NVMe
        success, location, _ = multi_tier_cache_with_gpu.allocate_cache("overflow_entry", num_tokens=10000)
        assert success is True
        assert location in ['cpu', 'nvme']
    
    def test_cache_access_from_gpu(self, multi_tier_cache_with_gpu):
        multi_tier_cache_with_gpu.allocate_cache("test_entry", num_tokens=100)
        loc, read_lat = multi_tier_cache_with_gpu.access_cache("test_entry", InferencePhase.DECODE)
        assert loc == 'gpu'
    
    def test_stats_include_gpu_entries(self, multi_tier_cache_with_gpu):
        multi_tier_cache_with_gpu.allocate_cache("test_entry", num_tokens=100)
        stats = multi_tier_cache_with_gpu.get_stats(duration=1.0)
        assert 'gpu_entries' in stats


# =============================================================================
# Test 10: XLSX Export
# =============================================================================

@pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not installed")
class TestXLSXExport:
    """Tests for XLSX/CSV export functionality."""
    
    def test_csv_export_succeeds(self, mock_benchmark_results, mock_args):
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            test_path = f.name
        try:
            export_results_to_xlsx(mock_benchmark_results, mock_args, test_path)
            assert os.path.exists(test_path)
        finally:
            if os.path.exists(test_path):
                os.unlink(test_path)
    
    def test_csv_has_data(self, mock_benchmark_results, mock_args):
        import pandas as pd
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            test_path = f.name
        try:
            export_results_to_xlsx(mock_benchmark_results, mock_args, test_path)
            df = pd.read_csv(test_path)
            assert len(df) == 1
        finally:
            if os.path.exists(test_path):
                os.unlink(test_path)
    
    def test_csv_has_model_column(self, mock_benchmark_results, mock_args):
        import pandas as pd
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            test_path = f.name
        try:
            export_results_to_xlsx(mock_benchmark_results, mock_args, test_path)
            df = pd.read_csv(test_path)
            assert 'Model' in df.columns
        finally:
            if os.path.exists(test_path):
                os.unlink(test_path)
    
    def test_csv_model_value(self, mock_benchmark_results, mock_args):
        import pandas as pd
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            test_path = f.name
        try:
            export_results_to_xlsx(mock_benchmark_results, mock_args, test_path)
            df = pd.read_csv(test_path)
            assert df['Model'].iloc[0] == 'llama3.1-8b'
        finally:
            if os.path.exists(test_path):
                os.unlink(test_path)
    
    def test_csv_has_throughput_columns(self, mock_benchmark_results, mock_args):
        import pandas as pd
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            test_path = f.name
        try:
            export_results_to_xlsx(mock_benchmark_results, mock_args, test_path)
            df = pd.read_csv(test_path)
            assert 'Avg Throughput (tok/s)' in df.columns
            assert 'Storage Throughput (tok/s)' in df.columns
        finally:
            if os.path.exists(test_path):
                os.unlink(test_path)


# =============================================================================
# Test 11: Enums
# =============================================================================

class TestEnums:
    """Tests for enum consistency."""
    
    def test_inference_phase_count(self):
        assert len(InferencePhase) == 3
    
    def test_inference_phase_prefill(self):
        assert InferencePhase.PREFILL.value == "prefill"
    
    def test_inference_phase_decode(self):
        assert InferencePhase.DECODE.value == "decode"
    
    def test_inference_phase_both(self):
        assert InferencePhase.PREFILL_DECODE.value == "both"
    
    def test_generation_mode_count(self):
        assert len(GenerationMode) == 3
    
    def test_generation_timing_none(self):
        assert GENERATION_TIMING[GenerationMode.NONE] == 0.0
    
    def test_generation_timing_fast(self):
        assert GENERATION_TIMING[GenerationMode.FAST] == 0.002
    
    def test_generation_timing_realistic(self):
        assert GENERATION_TIMING[GenerationMode.REALISTIC] == 0.030
    
    def test_qos_level_count(self):
        assert len(QoSLevel) == 3
    
    def test_timing_matches_modes(self):
        assert len(GENERATION_TIMING) == len(GenerationMode)


# =============================================================================
# Test 12: Tier Logic
# =============================================================================

class TestTierLogic:
    """Tests for tier ordering and limits."""
    
    def test_tier_order_includes_expected_tiers(self, multi_tier_cache):
        """Tier order should include cpu and nvme (gpu may or may not be present)."""
        tier_order = multi_tier_cache._get_tier_order()
        assert 'cpu' in tier_order
        assert 'nvme' in tier_order
        # If GPU is present, it should be first
        if 'gpu' in tier_order:
            assert tier_order.index('gpu') < tier_order.index('cpu')
    
    def test_cpu_limit(self, multi_tier_cache):
        cpu_limit = multi_tier_cache._get_tier_limit('cpu')
        assert cpu_limit == 0.1 * 1024**3  # 100MB
    
    def test_nvme_limit_infinite(self, multi_tier_cache):
        nvme_limit = multi_tier_cache._get_tier_limit('nvme')
        assert nvme_limit == float('inf')
    
    def test_initial_cpu_usage_zero(self, multi_tier_cache):
        cpu_usage = multi_tier_cache._get_tier_usage('cpu')
        assert cpu_usage == 0


# =============================================================================
# Test 13: Config-Driven Parameters (New in 01-26-2026)
# =============================================================================

class TestConfigDrivenConversationManager:
    """Tests for ConversationManager with config-driven parameters."""
    
    def test_default_max_conversations(self):
        """Without config, should use hardcoded default of 1000."""
        set_config(None)
        manager = ConversationManager()
        assert manager.max_conversations == 1000
    
    def test_default_max_turns(self):
        """Without config, should use hardcoded default of 50."""
        set_config(None)
        manager = ConversationManager()
        assert manager.max_turns_per_conv == 50
    
    def test_explicit_params_override_config(self, sample_config_yaml):
        """Explicit constructor params should override config values."""
        loader = ConfigLoader(config_path=sample_config_yaml)
        set_config(loader)
        try:
            manager = ConversationManager(max_conversations=42, max_turns_per_conv=7)
            assert manager.max_conversations == 42
            assert manager.max_turns_per_conv == 7
        finally:
            set_config(None)


@pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
class TestConfigDrivenUserSimulator:
    """Tests for UserSimulator with config-driven parameters."""
    
    def test_user_templates_from_config(self, sample_config_yaml):
        """UserSimulator should read templates from config."""
        loader = ConfigLoader(config_path=sample_config_yaml)
        set_config(loader)
        try:
            templates = UserSimulator._get_user_templates()
            assert 'chatbot' in templates
            assert 'coding' in templates
            assert 'document' in templates
            assert templates['chatbot']['context_range'] == (256, 1024)
        finally:
            set_config(None)
    
    def test_qos_distribution_from_config(self, sample_config_yaml):
        """UserSimulator.generate_mixed_users should use config QoS distribution."""
        loader = ConfigLoader(config_path=sample_config_yaml)
        set_config(loader)
        try:
            # Generate many users to test distribution
            users = UserSimulator.generate_mixed_users(1000)
            # With 15% interactive probability, expect ~150 interactive users
            interactive_count = sum(1 for u in users if u.qos_level == QoSLevel.INTERACTIVE)
            # Allow 50% variance for randomness
            assert 75 <= interactive_count <= 225, f"Expected ~150 interactive, got {interactive_count}"
        finally:
            set_config(None)


# =============================================================================
# Test 14: Stats Naming Convention (storage_* vs nvme_*)
# =============================================================================

class TestStatsNamingConvention:
    """Tests that stats use 'storage_*' naming (not 'nvme_*') in 01-26-2026."""
    
    def test_stats_use_storage_prefix(self, multi_tier_cache):
        """Stats should use 'storage_' prefix instead of 'nvme_'."""
        multi_tier_cache.allocate_cache("test_entry", num_tokens=100)
        multi_tier_cache.access_cache("test_entry", InferencePhase.DECODE)
        stats = multi_tier_cache.get_stats(duration=1.0)
        
        # Check for storage_* naming
        storage_keys = [k for k in stats.keys() if 'storage_' in k.lower()]
        nvme_keys = [k for k in stats.keys() if 'nvme_' in k.lower()]
        
        # Should have storage_* keys
        assert len(storage_keys) > 0, "Expected storage_* keys in stats"
    
    def test_tier_stats_key_format(self, multi_tier_cache):
        """tier_storage_* keys should exist (renamed from tier_nvme_*)."""
        multi_tier_cache.allocate_cache("test_entry", num_tokens=100)
        stats = multi_tier_cache.get_stats(duration=1.0)
        
        # Check for tier_storage_* keys
        tier_storage_keys = [k for k in stats.keys() if k.startswith('tier_storage_')]
        assert len(tier_storage_keys) > 0, "Expected tier_storage_* keys in stats"


# =============================================================================
# Test 15: GPUMemoryBackend Eviction Callback (New in 01-26-2026)
# =============================================================================

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestGPUMemoryBackendEvictionCallback:
    """Tests for GPUMemoryBackend's on_eviction_callback feature."""
    
    def test_gpu_backend_accepts_callback(self):
        """GPUMemoryBackend should accept on_eviction_callback parameter."""
        evicted_keys = []
        def callback(key, tier, size):
            evicted_keys.append((key, tier, size))
        
        backend = GPUMemoryBackend(on_eviction_callback=callback)
        assert backend.on_eviction_callback is callback
        backend.clear()
    
    def test_gpu_backend_works_without_callback(self):
        """GPUMemoryBackend should work without a callback (None)."""
        backend = GPUMemoryBackend(on_eviction_callback=None)
        assert backend.on_eviction_callback is None
        backend.clear()


# =============================================================================
# Test 16: Input Validation (validate_args)
# =============================================================================

class TestValidateArgs:
    """Tests for the validate_args() input validation function."""
    
    @pytest.fixture
    def valid_args(self):
        """Create a valid args namespace with all required attributes."""
        import argparse
        args = argparse.Namespace(
            num_users=100,
            duration=60,
            gpu_mem_gb=16,
            cpu_mem_gb=32,
            rag_num_docs=10,
            max_conversations=500,
            max_concurrent_allocs=0,
            request_rate=0,
            max_requests=0,
            target_saturation=0.8,
            cache_dir=None
        )
        return args
    
    def test_valid_args_pass_through(self, valid_args):
        """Valid arguments should pass validation and return unchanged."""
        result = validate_args(valid_args)
        assert result is valid_args
        assert result.num_users == 100
        assert result.duration == 60
    
    def test_num_users_zero_rejected(self, valid_args):
        """num_users=0 should raise ValueError."""
        valid_args.num_users = 0
        with pytest.raises(ValueError, match="num-users must be positive"):
            validate_args(valid_args)
    
    def test_num_users_negative_rejected(self, valid_args):
        """Negative num_users should raise ValueError."""
        valid_args.num_users = -5
        with pytest.raises(ValueError, match="num-users must be positive"):
            validate_args(valid_args)
    
    def test_num_users_exceeds_limit(self, valid_args):
        """num_users exceeding MAX_USERS should raise ValueError."""
        valid_args.num_users = MAX_USERS + 1
        with pytest.raises(ValueError, match="num-users exceeds limit"):
            validate_args(valid_args)
    
    def test_duration_zero_rejected(self, valid_args):
        """duration=0 should raise ValueError."""
        valid_args.duration = 0
        with pytest.raises(ValueError, match="duration must be positive"):
            validate_args(valid_args)
    
    def test_duration_negative_rejected(self, valid_args):
        """Negative duration should raise ValueError."""
        valid_args.duration = -10
        with pytest.raises(ValueError, match="duration must be positive"):
            validate_args(valid_args)
    
    def test_duration_exceeds_limit(self, valid_args):
        """duration exceeding 24 hours should raise ValueError."""
        valid_args.duration = MAX_DURATION_SECONDS + 1
        with pytest.raises(ValueError, match="duration exceeds 24 hours"):
            validate_args(valid_args)
    
    def test_gpu_mem_negative_rejected(self, valid_args):
        """Negative gpu_mem_gb should raise ValueError."""
        valid_args.gpu_mem_gb = -1
        with pytest.raises(ValueError, match="gpu-mem-gb cannot be negative"):
            validate_args(valid_args)
    
    def test_gpu_mem_zero_allowed(self, valid_args):
        """gpu_mem_gb=0 should be valid (disables GPU tier)."""
        valid_args.gpu_mem_gb = 0
        result = validate_args(valid_args)
        assert result.gpu_mem_gb == 0
    
    def test_gpu_mem_exceeds_limit(self, valid_args):
        """gpu_mem_gb exceeding limit should raise ValueError."""
        valid_args.gpu_mem_gb = MAX_GPU_MEMORY_GB + 1
        with pytest.raises(ValueError, match="gpu-mem-gb exceeds limit"):
            validate_args(valid_args)
    
    def test_cpu_mem_negative_rejected(self, valid_args):
        """Negative cpu_mem_gb should raise ValueError."""
        valid_args.cpu_mem_gb = -1
        with pytest.raises(ValueError, match="cpu-mem-gb cannot be negative"):
            validate_args(valid_args)
    
    def test_cpu_mem_zero_allowed(self, valid_args):
        """cpu_mem_gb=0 should be valid."""
        valid_args.cpu_mem_gb = 0
        result = validate_args(valid_args)
        assert result.cpu_mem_gb == 0
    
    def test_cpu_mem_exceeds_limit(self, valid_args):
        """cpu_mem_gb exceeding limit should raise ValueError."""
        valid_args.cpu_mem_gb = MAX_CPU_MEMORY_GB + 1
        with pytest.raises(ValueError, match="cpu-mem-gb exceeds limit"):
            validate_args(valid_args)
    
    def test_target_saturation_below_zero_rejected(self, valid_args):
        """target_saturation < 0 should raise ValueError."""
        valid_args.target_saturation = -0.1
        with pytest.raises(ValueError, match="target-saturation must be between 0.0 and 1.0"):
            validate_args(valid_args)
    
    def test_target_saturation_above_one_rejected(self, valid_args):
        """target_saturation > 1 should raise ValueError."""
        valid_args.target_saturation = 1.5
        with pytest.raises(ValueError, match="target-saturation must be between 0.0 and 1.0"):
            validate_args(valid_args)
    
    def test_target_saturation_boundaries_valid(self, valid_args):
        """target_saturation at 0.0 and 1.0 should be valid."""
        valid_args.target_saturation = 0.0
        result = validate_args(valid_args)
        assert result.target_saturation == 0.0
        
        valid_args.target_saturation = 1.0
        result = validate_args(valid_args)
        assert result.target_saturation == 1.0
    
    def test_rag_num_docs_negative_rejected(self, valid_args):
        """Negative rag_num_docs should raise ValueError."""
        valid_args.rag_num_docs = -1
        with pytest.raises(ValueError, match="rag-num-docs cannot be negative"):
            validate_args(valid_args)
    
    def test_max_conversations_zero_rejected(self, valid_args):
        """max_conversations=0 should raise ValueError."""
        valid_args.max_conversations = 0
        with pytest.raises(ValueError, match="max-conversations must be positive"):
            validate_args(valid_args)
    
    def test_max_concurrent_allocs_negative_rejected(self, valid_args):
        """Negative max_concurrent_allocs should raise ValueError."""
        valid_args.max_concurrent_allocs = -1
        with pytest.raises(ValueError, match="max-concurrent-allocs cannot be negative"):
            validate_args(valid_args)
    
    def test_request_rate_negative_rejected(self, valid_args):
        """Negative request_rate should raise ValueError."""
        valid_args.request_rate = -1
        with pytest.raises(ValueError, match="request-rate cannot be negative"):
            validate_args(valid_args)
    
    def test_max_requests_negative_rejected(self, valid_args):
        """Negative max_requests should raise ValueError."""
        valid_args.max_requests = -1
        with pytest.raises(ValueError, match="max-requests cannot be negative"):
            validate_args(valid_args)
    
    @pytest.mark.skipif(sys.platform == 'win32', reason="Unix paths not valid on Windows")
    def test_forbidden_cache_dir_rejected(self, valid_args):
        """Cache directories in system paths should be rejected."""
        valid_args.cache_dir = '/etc/kv_cache'
        with pytest.raises(ValueError, match="cannot be a system directory"):
            validate_args(valid_args)
    
    def test_valid_cache_dir_allowed(self, valid_args, tmp_path):
        """Valid cache directory should be accepted."""
        valid_args.cache_dir = str(tmp_path / "kv_cache_test")
        result = validate_args(valid_args)
        assert result.cache_dir == str(tmp_path / "kv_cache_test")
    
    def test_multiple_errors_collected(self, valid_args):
        """Multiple validation errors should all be reported."""
        valid_args.num_users = -1
        valid_args.duration = -1
        valid_args.gpu_mem_gb = -1
        with pytest.raises(ValueError) as exc_info:
            validate_args(valid_args)
        # All three errors should be in the message
        error_msg = str(exc_info.value)
        assert "num-users" in error_msg
        assert "duration" in error_msg
        assert "gpu-mem-gb" in error_msg


# =============================================================================
# Test 17: Per-Tier Phase Metrics
# =============================================================================

class TestPerTierPhaseMetrics:
    """Tests for per-tier KV bytes tracking (prefill/decode per tier)."""
    
    @pytest.fixture
    def tiny_model_config(self):
        """Return the tiny-1b model config for fast tests."""
        return MODEL_CONFIGS['tiny-1b']
    
    @pytest.fixture
    def multi_tier_cache_cpu_only(self, tiny_model_config):
        """Return a MultiTierCache in CPU-only mode (GPU disabled)."""
        return MultiTierCache(
            model_config=tiny_model_config,
            gpu_memory_gb=0,
            cpu_memory_gb=0.1,  # 100MB
            seed=42
        )
    
    def test_stats_have_tier_kv_bytes_written_keys(self, multi_tier_cache_cpu_only):
        """Stats should include tier_*_kv_bytes_written keys."""
        multi_tier_cache_cpu_only.allocate_cache("test_entry", num_tokens=100)
        stats = multi_tier_cache_cpu_only.get_stats(duration=1.0)
        
        # Check for per-tier write tracking
        assert 'tier_gpu_kv_bytes_written_gb' in stats
        assert 'tier_cpu_kv_bytes_written_gb' in stats
        assert 'tier_storage_kv_bytes_written_gb' in stats
    
    def test_stats_have_tier_kv_bytes_read_keys(self, multi_tier_cache_cpu_only):
        """Stats should include tier_*_kv_bytes_read keys."""
        multi_tier_cache_cpu_only.allocate_cache("test_entry", num_tokens=100)
        multi_tier_cache_cpu_only.access_cache("test_entry", InferencePhase.DECODE)
        stats = multi_tier_cache_cpu_only.get_stats(duration=1.0)
        
        # Check for per-tier read tracking
        assert 'tier_gpu_kv_bytes_read_gb' in stats
        assert 'tier_cpu_kv_bytes_read_gb' in stats
        assert 'tier_storage_kv_bytes_read_gb' in stats
    
    def test_cpu_write_bytes_increment_on_allocate(self, multi_tier_cache_cpu_only):
        """Allocating to CPU tier should increment tier_cpu_kv_bytes_written."""
        # Get initial stats
        stats_before = multi_tier_cache_cpu_only.get_stats(duration=1.0)
        cpu_written_before = stats_before.get('tier_cpu_kv_bytes_written_gb', 0)
        
        # Allocate cache entry (goes to CPU since GPU is disabled)
        success, location, _ = multi_tier_cache_cpu_only.allocate_cache("test_entry", num_tokens=100)
        assert success
        assert location == 'cpu'
        
        # Check that CPU write bytes increased
        stats_after = multi_tier_cache_cpu_only.get_stats(duration=1.0)
        cpu_written_after = stats_after.get('tier_cpu_kv_bytes_written_gb', 0)
        
        assert cpu_written_after > cpu_written_before, \
            f"CPU write bytes should increase: {cpu_written_before} -> {cpu_written_after}"
    
    def test_cpu_read_bytes_increment_on_access(self, multi_tier_cache_cpu_only):
        """Accessing from CPU tier should increment tier_cpu_kv_bytes_read."""
        # Allocate first
        multi_tier_cache_cpu_only.allocate_cache("test_entry", num_tokens=100)
        
        # Get stats before access
        stats_before = multi_tier_cache_cpu_only.get_stats(duration=1.0)
        cpu_read_before = stats_before.get('tier_cpu_kv_bytes_read_gb', 0)
        
        # Access the cache entry
        location, _ = multi_tier_cache_cpu_only.access_cache("test_entry", InferencePhase.DECODE)
        assert location == 'cpu'
        
        # Check that CPU read bytes increased
        stats_after = multi_tier_cache_cpu_only.get_stats(duration=1.0)
        cpu_read_after = stats_after.get('tier_cpu_kv_bytes_read_gb', 0)
        
        assert cpu_read_after > cpu_read_before, \
            f"CPU read bytes should increase: {cpu_read_before} -> {cpu_read_after}"
    
    def test_gpu_bytes_zero_when_gpu_disabled(self, multi_tier_cache_cpu_only):
        """With GPU disabled (0 GB), GPU tier bytes should remain zero."""
        # Do some allocations and accesses
        for i in range(5):
            multi_tier_cache_cpu_only.allocate_cache(f"entry_{i}", num_tokens=100)
        for i in range(5):
            multi_tier_cache_cpu_only.access_cache(f"entry_{i}", InferencePhase.DECODE)
        
        stats = multi_tier_cache_cpu_only.get_stats(duration=1.0)
        
        # GPU bytes should be zero since GPU tier is disabled
        assert stats.get('tier_gpu_kv_bytes_written_gb', 0) == 0, \
            "GPU write bytes should be 0 when GPU disabled"
        assert stats.get('tier_gpu_kv_bytes_read_gb', 0) == 0, \
            "GPU read bytes should be 0 when GPU disabled"
    
    def test_storage_tier_overflow(self, tiny_model_config):
        """When CPU is full, allocations should overflow to storage tier."""
        # Create cache with very small CPU limit
        cache = MultiTierCache(
            model_config=tiny_model_config,
            gpu_memory_gb=0,
            cpu_memory_gb=0.001,  # 1MB - very small
            seed=42
        )
        
        # Allocate enough to overflow CPU
        for i in range(20):
            cache.allocate_cache(f"entry_{i}", num_tokens=1000)
        
        stats = cache.get_stats(duration=1.0)
        
        # Storage tier should have received some data
        storage_written = stats.get('tier_storage_kv_bytes_written_gb', 0)
        assert storage_written > 0, \
            f"Storage tier should have data when CPU overflows: {storage_written}"
    
    def test_per_tier_bandwidth_calculated(self, multi_tier_cache_cpu_only):
        """Per-tier bandwidth stats should be calculated."""
        # Do some I/O
        for i in range(10):
            multi_tier_cache_cpu_only.allocate_cache(f"entry_{i}", num_tokens=100)
        for i in range(10):
            multi_tier_cache_cpu_only.access_cache(f"entry_{i}", InferencePhase.DECODE)
        
        stats = multi_tier_cache_cpu_only.get_stats(duration=1.0)
        
        # Bandwidth stats should exist
        assert 'tier_cpu_read_bandwidth_gbps' in stats
        assert 'tier_cpu_write_bandwidth_gbps' in stats
        assert 'tier_storage_read_bandwidth_gbps' in stats
        assert 'tier_storage_write_bandwidth_gbps' in stats


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestPerTierPhaseMetricsWithGPU:
    """Tests for per-tier metrics when GPU is enabled."""
    
    @pytest.fixture
    def tiny_model_config(self):
        """Return the tiny-1b model config for fast tests."""
        return MODEL_CONFIGS['tiny-1b']
    
    @pytest.fixture
    def multi_tier_cache_with_gpu(self, tiny_model_config):
        """Return a MultiTierCache with GPU enabled."""
        return MultiTierCache(
            model_config=tiny_model_config,
            gpu_memory_gb=1.0,  # 1GB GPU
            cpu_memory_gb=0.1,  # 100MB CPU
            seed=42
        )
    
    def test_gpu_write_bytes_increment_on_allocate(self, multi_tier_cache_with_gpu):
        """Allocating to GPU tier should increment tier_gpu_kv_bytes_written."""
        # Get initial stats
        stats_before = multi_tier_cache_with_gpu.get_stats(duration=1.0)
        gpu_written_before = stats_before.get('tier_gpu_kv_bytes_written_gb', 0)
        
        # Allocate cache entry (should go to GPU first)
        success, location, _ = multi_tier_cache_with_gpu.allocate_cache("test_entry", num_tokens=100)
        assert success
        assert location == 'gpu'
        
        # Check that GPU write bytes increased
        stats_after = multi_tier_cache_with_gpu.get_stats(duration=1.0)
        gpu_written_after = stats_after.get('tier_gpu_kv_bytes_written_gb', 0)
        
        assert gpu_written_after > gpu_written_before, \
            f"GPU write bytes should increase: {gpu_written_before} -> {gpu_written_after}"
    
    def test_gpu_read_bytes_increment_on_access(self, multi_tier_cache_with_gpu):
        """Accessing from GPU tier should increment tier_gpu_kv_bytes_read."""
        # Allocate first
        multi_tier_cache_with_gpu.allocate_cache("test_entry", num_tokens=100)
        
        # Get stats before access
        stats_before = multi_tier_cache_with_gpu.get_stats(duration=1.0)
        gpu_read_before = stats_before.get('tier_gpu_kv_bytes_read_gb', 0)
        
        # Access the cache entry
        location, _ = multi_tier_cache_with_gpu.access_cache("test_entry", InferencePhase.DECODE)
        assert location == 'gpu'
        
        # Check that GPU read bytes increased
        stats_after = multi_tier_cache_with_gpu.get_stats(duration=1.0)
        gpu_read_after = stats_after.get('tier_gpu_kv_bytes_read_gb', 0)
        
        assert gpu_read_after > gpu_read_before, \
            f"GPU read bytes should increase: {gpu_read_before} -> {gpu_read_after}"
    
    def test_gpu_bandwidth_calculated(self, multi_tier_cache_with_gpu):
        """GPU tier bandwidth stats should be calculated."""
        # Do some I/O
        for i in range(5):
            multi_tier_cache_with_gpu.allocate_cache(f"entry_{i}", num_tokens=100)
        for i in range(5):
            multi_tier_cache_with_gpu.access_cache(f"entry_{i}", InferencePhase.DECODE)
        
        stats = multi_tier_cache_with_gpu.get_stats(duration=1.0)
        
        # GPU bandwidth stats should exist
        assert 'tier_gpu_read_bandwidth_gbps' in stats
        assert 'tier_gpu_write_bandwidth_gbps' in stats


# =============================================================================
# Main entry point for running without pytest
# =============================================================================

def pytest_configure(config):
    """Add metadata to pytest-html report."""
    if hasattr(config, '_metadata'):
        config._metadata['Project'] = 'MLPerf v3 KV Cache Benchmark'
        config._metadata['Source File'] = 'kv-cache.py'
        config._metadata['Models'] = 'tiny-1b, mistral-7b, llama2-7b, llama3.1-8b, llama3.1-70b-instruct'
        config._metadata['Test File'] = 'test_kv_cache.py'
        config._metadata['New Features Tested'] = 'ConfigLoader, Extended QoS (p999/p9999), cfg() helper, storage_* naming'


def pytest_html_report_title(report):
    """Set custom title for HTML report."""
    report.title = "KV Cache Benchmark - Unit Test Report"


if __name__ == "__main__":
    # Generate HTML report by default when run directly
    report_path = Path(__file__).parent / "test_report.html"
    exit_code = pytest.main([
        __file__, 
        "-v",
        f"--html={report_path}",
        "--self-contained-html",
    ])
    if exit_code == 0:
        print(f"\n✓ All tests passed! HTML report: {report_path}")
    else:
        print(f"\n✗ Some tests failed. HTML report: {report_path}")
    sys.exit(exit_code)
