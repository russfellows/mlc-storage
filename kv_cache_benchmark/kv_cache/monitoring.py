"""
Monitoring, autoscaling, and QoS tracking for KV Cache Benchmark.

Contains StorageMetrics, StorageMonitor, WorkloadAutoscaler, and QoSMonitor.
"""

import time
import logging
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from kv_cache.config import cfg
from kv_cache.models import QoSLevel, QoSSLA, QOS_PROFILES, InferenceRequest

logger = logging.getLogger(__name__)


# ============================================================================
# ADAPTIVE AUTOSCALING
# ============================================================================

@dataclass
class StorageMetrics:
    """A snapshot of storage performance metrics at a point in time."""
    timestamp: float
    read_throughput_gbps: float
    write_throughput_gbps: float
    read_iops: int
    write_iops: int
    read_latency_p95_ms: float
    write_latency_p95_ms: float
    queue_depth: int
    is_saturated: bool = False
    saturation_level: float = 0.0


class StorageMonitor:
    """Monitors storage performance in real-time to feed the autoscaler."""

    def __init__(self, benchmark_instance, sampling_interval_ms: float = 100):
        self.benchmark_instance = benchmark_instance
        self.sampling_interval = sampling_interval_ms / 1000.0
        self.last_collection_time = None
        self.last_total_read = 0
        self.last_total_write = 0
        self.metrics_history = []
        self.lock = threading.Lock()

    def collect_metrics(self, cache, queue_size):
        """Collects all relevant performance metrics."""
        now = time.time()
        if self.last_collection_time is None:
            self.last_collection_time = now
            self.last_total_read = cache.stats.get('total_read_bytes', 0)
            self.last_total_write = cache.stats.get('total_write_bytes', 0)
            return {}

        elapsed = now - self.last_collection_time
        if elapsed == 0:
            return {}

        stats = cache.get_stats(duration=self.benchmark_instance.duration)
        current_total_read = stats.get('total_read_bytes', 0)
        current_total_write = stats.get('total_write_bytes', 0)

        read_delta = max(current_total_read - self.last_total_read, 0)
        write_delta = max(current_total_write - self.last_total_write, 0)

        read_throughput = (read_delta / 1024**3) / elapsed
        write_throughput = (write_delta / 1024**3) / elapsed

        queue_depth = queue_size

        read_iops = int((read_delta / 4096) / elapsed) if elapsed > 0 else 0
        write_iops = int((write_delta / (16 * 1024)) / elapsed) if elapsed > 0 else 0

        read_latency_p95_ms = stats.get('storage_read_p95_ms', 0.0)
        write_latency_p95_ms = stats.get('storage_write_p95_ms', 0.0)

        # Saturation Detection Logic
        read_lat_threshold = cfg('saturation_detection', 'read_latency_p95_threshold_ms', default=100)
        write_lat_threshold = cfg('saturation_detection', 'write_latency_p95_threshold_ms', default=50)
        queue_depth_threshold = cfg('saturation_detection', 'queue_depth_threshold', default=100)

        is_saturated = False
        if len(self.metrics_history) >= 2:
            prev_metric = self.metrics_history[-2]
            if (prev_metric.read_latency_p95_ms < read_lat_threshold and
                prev_metric.write_latency_p95_ms < write_lat_threshold and
                prev_metric.queue_depth < queue_depth_threshold):
                if (abs(prev_metric.read_latency_p95_ms - read_latency_p95_ms) > 20 or
                    abs(prev_metric.write_latency_p95_ms - write_latency_p95_ms) > 10 or
                    abs(prev_metric.queue_depth - queue_depth) > 10):
                    is_saturated = True
            else:
                if (read_latency_p95_ms > read_lat_threshold * 1.2 or
                    write_latency_p95_ms > write_lat_threshold * 1.2 or
                    queue_depth > queue_depth_threshold * 1.2):
                    is_saturated = True

        metrics = StorageMetrics(
            timestamp=now,
            read_throughput_gbps=read_throughput,
            write_throughput_gbps=write_throughput,
            read_iops=read_iops,
            write_iops=write_iops,
            read_latency_p95_ms=read_latency_p95_ms,
            write_latency_p95_ms=write_latency_p95_ms,
            queue_depth=queue_depth,
            is_saturated=is_saturated
        )

        with self.lock:
            self.metrics_history.append(metrics)
            saturation_level = self._compute_saturation_from_history(self.metrics_history)

        metrics.saturation_level = saturation_level

        self.last_collection_time = now
        self.last_total_read = current_total_read
        self.last_total_write = current_total_write
        return metrics

    def get_saturation_level(self) -> float:
        """Calculates the storage saturation level (0.0 = idle, 1.0 = saturated)."""
        with self.lock:
            history_snapshot = list(self.metrics_history)

        return self._compute_saturation_from_history(history_snapshot)

    def _compute_saturation_from_history(self, history: List[StorageMetrics]) -> float:
        if len(history) < 10:
            return 0.0

        recent_metrics = history[-10:]

        latencies = [m.read_latency_p95_ms for m in recent_metrics]
        if len(latencies) > 1:
            latency_trend = np.polyfit(range(len(latencies)), latencies, 1)[0]
        else:
            latency_trend = 0

        throughputs = [m.read_throughput_gbps + m.write_throughput_gbps for m in recent_metrics]
        throughput_variance = np.std(throughputs) / (np.mean(throughputs) + 0.01)

        latency_factor = min(max(latencies) / 100, 1.0)
        plateau_factor = 1.0 if throughput_variance < 0.1 and latency_trend > 0 else 0.5

        saturation = latency_factor * plateau_factor
        return min(saturation, 1.0)


class WorkloadAutoscaler:
    """Automatically scales the number of simulated users to find a performance limit."""

    def __init__(self,
                 mode: str = 'qos',
                 initial_users: int = 10,
                 target_saturation: float = 0.8,
                 scale_interval_seconds: int = 10):
        self.mode = mode
        self.current_users = initial_users
        self.target_saturation = target_saturation
        self.scale_interval = scale_interval_seconds
        self.min_users = cfg('autoscaler', 'min_users', default=1)
        self.max_users = cfg('autoscaler', 'max_users', default=10000)
        self.scale_up_factor = cfg('autoscaler', 'scale_up_factor', default=1.2)
        self.scale_down_factor = cfg('autoscaler', 'scale_down_factor', default=0.8)
        self.consecutive_samples_required = cfg('autoscaler', 'consecutive_samples_required', default=2)
        self.scaling_history = []
        self.lock = threading.Lock()

        self.cooldown_counter = 0
        self.cooldown_period = 3
        self.downward_trend_count = 0

        self.capacity_stage = 0
        self.last_throughput = 0.0
        self.peak_throughput = 0.0
        self.peak_user_count = 0
        self.capacity_test_finished = False
        self.throughput_history: List[float] = []
        self.capacity_initial_fraction = 0.4
        self.capacity_scale_fraction = 0.2
        self.capacity_min_step = 5
        self.capacity_max_step = 100

    def calculate_scale_action(
        self,
        metrics: Optional[StorageMetrics],
        current_throughput: float,
        saturation_level: Optional[float] = None
    ) -> Tuple[str, int]:
        """Decides the next scaling action based on the selected mode."""
        if self.mode == 'qos':
            if not metrics: return 'stable', self.current_users
            return self._calculate_qos_action(metrics, saturation_level)
        elif self.mode == 'capacity':
            return self._calculate_capacity_action(current_throughput)
        return 'stable', self.current_users

    def _calculate_qos_action(self, metrics: StorageMetrics, saturation_level: Optional[float]) -> Tuple[str, int]:
        """Determines the scaling action for 'qos' mode."""
        with self.lock:
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
                return 'hold', self.current_users

            saturation = saturation_level
            if saturation is None:
                saturation = 1.0 if metrics.is_saturated else 0.0

            action = 'hold'
            target_users = self.current_users

            if saturation > self.target_saturation * 1.1:
                self.downward_trend_count += 1
                if self.downward_trend_count >= 2:
                    target_users = max(int(self.current_users * 0.8), self.min_users)
                    if target_users < self.current_users:
                        self.current_users = target_users
                        self.cooldown_counter = self.cooldown_period
                        action = 'scale_down'
            elif saturation < self.target_saturation * 0.9:
                self.downward_trend_count = 0
                target_users = min(int(self.current_users * 1.2), self.max_users)
                if target_users > self.current_users:
                    self.current_users = target_users
                    action = 'scale_up'
            else:
                self.downward_trend_count = 0

            return action, self.current_users
        return 'hold', self.current_users

    def _calculate_capacity_action(self, current_throughput: float) -> Tuple[str, int]:
        """Determines the scaling action for 'capacity' mode."""
        with self.lock:
            self.throughput_history.append(current_throughput)

            if not self.throughput_history or len(self.throughput_history) == 1:
                self.peak_throughput = current_throughput
                self.peak_user_count = self.current_users
                step = self._compute_capacity_step(self.capacity_initial_fraction)
                new_users = min(self.current_users + step, self.max_users)
                if new_users > self.current_users:
                    self.current_users = new_users
                    return 'scale_up', self.current_users
                return 'hold', self.current_users

            if current_throughput > self.peak_throughput * 1.01:
                self.peak_throughput = current_throughput
                self.peak_user_count = self.current_users
                self.downward_trend_count = 0
                step = self._compute_capacity_step(self.capacity_scale_fraction)
                new_users = min(self.current_users + step, self.max_users)
                if new_users > self.current_users:
                    self.current_users = new_users
                    return 'scale_up', self.current_users
                return 'hold', self.current_users

            self.downward_trend_count += 1
            if self.downward_trend_count >= 2:
                self.capacity_test_finished = True
                logger.info(f"Peak capacity found at {self.peak_throughput:.2f} tok/s. Stopping test.")
                return 'stop', self.current_users

            return 'hold', self.current_users
        return 'hold', self.current_users

    def _compute_capacity_step(self, fraction: float) -> int:
        """Calculate a bounded capacity-mode step for smoother scaling."""
        raw_step = max(int(self.current_users * fraction), self.capacity_min_step)
        return min(raw_step, self.capacity_max_step)


# ============================================================================
# QOS MONITORING
# ============================================================================

class QoSMonitor:
    """Monitors and reports on QoS compliance in real-time."""

    def __init__(self):
        self.requests_by_qos: Dict[QoSLevel, List[InferenceRequest]] = {level: [] for level in QoSLevel}
        self.lock = threading.Lock()
        self.violations_by_qos: Dict[QoSLevel, int] = {level: 0 for level in QoSLevel}

    def record_request(self, request: InferenceRequest):
        """Records a completed request and checks if it violated its SLA."""
        with self.lock:
            self.requests_by_qos[request.qos_level].append(request)

            sla = QOS_PROFILES[request.qos_level]
            if request.total_latency_ms > sla.target_latency_p95_ms:
                self.violations_by_qos[request.qos_level] += 1
                sla.violations += 1
            sla.total_requests += 1

    def get_qos_metrics(self, qos_level: QoSLevel) -> Dict:
        """Gets performance metrics for a specific QoS level."""
        with self.lock:
            requests = self.requests_by_qos[qos_level]
            if not requests: return {'no_data': True}

            latencies = [r.total_latency_ms for r in requests]
            sla = QOS_PROFILES[qos_level]

            return {
                'total_requests': len(requests),
                'latency_ms': {
                    'mean': np.mean(latencies), 'p50': np.percentile(latencies, 50),
                    'p95': np.percentile(latencies, 95), 'p99': np.percentile(latencies, 99),
                    'max': np.max(latencies),
                },
                'sla': {
                    'target_p95_ms': sla.target_latency_p95_ms,
                    'actual_p95_ms': np.percentile(latencies, 95),
                    'compliance': sla.sla_compliance,
                    'met': sla.sla_compliance >= 0.95
                }
            }

    def get_all_qos_metrics(self) -> Dict:
        """Gets metrics for all QoS levels."""
        return {level.value: self.get_qos_metrics(level) for level in QoSLevel}
