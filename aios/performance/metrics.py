"""
Prometheus Metrics for AIOS

Exposes all pipeline metrics on http://localhost:9090/metrics.
The metrics server runs in a separate thread to avoid blocking the asyncio event loop.

Key metrics from the debugging spec:
- aios_turn_latency_ms (Histogram, per stage)
- aios_state_transitions_total (Counter, from/to/trigger)
- aios_queue_depth (Gauge, per queue)
- aios_queue_drops_total (Counter, per queue)
- aios_vad_speech_probability (Histogram)
- aios_audio_underrun_total (Counter, per stream)
- aios_audio_overflow_total (Counter, per stream)
- aios_gpu_vram_bytes (Gauge)
- aios_degradation_level (Gauge)
- aios_correction_accuracy (Histogram)
- aios_barge_in_latency_ms (Histogram)
"""

import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# Default metrics port
DEFAULT_METRICS_PORT = 9090

# Try to import prometheus_client; gracefully degrade if unavailable
try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        start_http_server,
        REGISTRY,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed. Metrics will be disabled.")


class MetricsCollector:
    """
    Centralized Prometheus metrics collector for AIOS.

    All metrics are registered once at initialization. Components report
    values via the collector's methods.
    """

    def __init__(self, port: Optional[int] = None):
        self._port = port or int(os.environ.get("AIOS_METRICS_PORT", DEFAULT_METRICS_PORT))
        self._server_started = False
        self._enabled = PROMETHEUS_AVAILABLE

        if not self._enabled:
            return

        # Turn latency (per stage)
        self.turn_latency = Histogram(
            "aios_turn_latency_ms",
            "Per-stage and end-to-end latency in milliseconds",
            labelnames=["stage"],
            buckets=[5, 10, 20, 50, 80, 100, 150, 200, 300, 400, 500, 800, 1000],
        )

        # State machine transitions
        self.state_transitions = Counter(
            "aios_state_transitions_total",
            "State machine transition counts",
            labelnames=["from_state", "to_state", "trigger"],
        )

        # Queue depth
        self.queue_depth = Gauge(
            "aios_queue_depth",
            "Current queue depth",
            labelnames=["queue"],
        )

        # Queue drops
        self.queue_drops = Counter(
            "aios_queue_drops_total",
            "Items dropped due to backpressure",
            labelnames=["queue"],
        )

        # VAD speech probability
        self.vad_speech_probability = Histogram(
            "aios_vad_speech_probability",
            "Per-frame VAD speech probability",
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )

        # VAD event counters
        self.vad_speech_start = Counter(
            "aios_vad_speech_start_total",
            "Number of speech onset events",
        )
        self.vad_speech_end = Counter(
            "aios_vad_speech_end_total",
            "Number of speech offset events",
        )
        self.vad_barge_in = Counter(
            "aios_vad_barge_in_total",
            "Number of barge-in events",
        )
        self.vad_false_trigger = Counter(
            "aios_vad_false_trigger_total",
            "Speech starts with no ASR output within 2s",
        )
        self.vad_inference_ms = Histogram(
            "aios_vad_inference_ms",
            "Per-frame VAD inference latency in milliseconds",
            buckets=[0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0],
        )
        self.vad_energy_gate_reject = Counter(
            "aios_vad_energy_gate_reject_total",
            "Frames rejected by energy gate",
        )

        # Audio buffer metrics
        self.audio_underrun = Counter(
            "aios_audio_underrun_total",
            "Capture/playback underruns",
            labelnames=["stream"],
        )
        self.audio_overflow = Counter(
            "aios_audio_overflow_total",
            "Capture/playback overflows",
            labelnames=["stream"],
        )

        # GPU VRAM
        self.gpu_vram_bytes = Gauge(
            "aios_gpu_vram_bytes",
            "Current GPU VRAM usage in bytes",
        )

        # Degradation level
        self.degradation_level = Gauge(
            "aios_degradation_level",
            "Current degradation level (0-4)",
        )

        # Correction accuracy
        self.correction_accuracy = Histogram(
            "aios_correction_accuracy",
            "Edit distance ratio (corrected vs raw)",
            buckets=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 1.0],
        )

        # Barge-in latency
        self.barge_in_latency = Histogram(
            "aios_barge_in_latency_ms",
            "Time from barge-in detection to silence in milliseconds",
            buckets=[10, 20, 50, 80, 100, 150, 200, 300],
        )

        # Error counter
        self.errors = Counter(
            "aios_errors_total",
            "Total error count",
            labelnames=["component"],
        )

        # State duration
        self.state_duration = Gauge(
            "aios_state_duration_seconds",
            "Time in current state in seconds",
        )

    def start_server(self):
        """Start the Prometheus metrics HTTP server in a background thread."""
        if not self._enabled or self._server_started:
            return

        try:
            start_http_server(self._port)
            self._server_started = True
            logger.info("Prometheus metrics server started on port %d", self._port)
        except Exception as e:
            logger.error("Failed to start metrics server: %s", e)

    # ── Convenience methods ──

    def record_turn_latency(self, stage: str, latency_ms: float):
        if self._enabled:
            self.turn_latency.labels(stage=stage).observe(latency_ms)

    def record_state_transition(self, from_state: str, to_state: str, trigger: str):
        if self._enabled:
            self.state_transitions.labels(
                from_state=from_state, to_state=to_state, trigger=trigger
            ).inc()

    def set_queue_depth(self, queue_name: str, depth: int):
        if self._enabled:
            self.queue_depth.labels(queue=queue_name).set(depth)

    def record_queue_drop(self, queue_name: str):
        if self._enabled:
            self.queue_drops.labels(queue=queue_name).inc()

    def record_vad_probability(self, probability: float):
        if self._enabled:
            self.vad_speech_probability.observe(probability)

    def record_audio_underrun(self, stream: str):
        if self._enabled:
            self.audio_underrun.labels(stream=stream).inc()

    def record_audio_overflow(self, stream: str):
        if self._enabled:
            self.audio_overflow.labels(stream=stream).inc()

    def set_gpu_vram(self, vram_bytes: int):
        if self._enabled:
            self.gpu_vram_bytes.set(vram_bytes)

    def set_degradation_level(self, level: int):
        if self._enabled:
            self.degradation_level.set(level)

    def record_barge_in_latency(self, latency_ms: float):
        if self._enabled:
            self.barge_in_latency.observe(latency_ms)

    def record_error(self, component: str):
        if self._enabled:
            self.errors.labels(component=component).inc()

    def set_state_duration(self, duration_s: float):
        if self._enabled:
            self.state_duration.set(duration_s)


# Singleton instance
_metrics: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """Get the global MetricsCollector singleton."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics
