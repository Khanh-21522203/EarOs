"""
Latency Instrumentation

Per-stage latency tracking using time.perf_counter_ns() for nanosecond precision.
Provides span-based tracing for each pipeline stage within a turn.

Every stage boundary records entry/exit timestamps. Latency is computed as
the difference. All spans are tagged with turn_id for correlation.

Measurement methodology:
- time.perf_counter_ns() at every stage boundary
- Per-turn summary with stage-by-stage breakdown
- Histogram aggregation for p50/p95/p99 reporting
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

# Stage names matching the pipeline
STAGES = [
    "audio_capture",
    "aec",
    "vad",
    "asr",
    "correction",
    "context_injection",
    "slm_response",
    "tts",
    "audio_playback",
]

# Latency targets (ms) from the spec
LATENCY_TARGETS: Dict[str, float] = {
    "audio_capture": 5.0,
    "aec": 5.0,
    "vad": 3.0,
    "asr": 200.0,
    "correction": 80.0,
    "context_injection": 5.0,
    "slm_response": 200.0,
    "tts": 100.0,
    "audio_playback": 5.0,
    "end_to_end": 500.0,
    "barge_in": 100.0,
    "capture_to_asr_token": 50.0,  # requirements.md NFR-1
}

# Hard limits (ms)
LATENCY_HARD_LIMITS: Dict[str, float] = {
    "asr": 350.0,
    "correction": 140.0,
    "slm_response": 400.0,
    "tts": 200.0,
    "end_to_end": 800.0,
    "barge_in": 200.0,
    "capture_to_asr_token": 100.0,  # requirements.md NFR-1
}


@dataclass
class Span:
    """A timing span for a single pipeline stage."""
    stage: str
    turn_id: int
    start_ns: int = 0
    end_ns: int = 0

    @property
    def duration_ms(self) -> float:
        if self.end_ns == 0 or self.start_ns == 0:
            return 0.0
        return (self.end_ns - self.start_ns) / 1_000_000

    @property
    def is_complete(self) -> bool:
        return self.start_ns > 0 and self.end_ns > 0


@dataclass
class TurnTrace:
    """Complete trace for a single conversational turn."""
    turn_id: int
    spans: Dict[str, Span] = field(default_factory=dict)
    turn_start_ns: int = 0
    turn_end_ns: int = 0

    @property
    def end_to_end_ms(self) -> float:
        if self.turn_end_ns == 0 or self.turn_start_ns == 0:
            return 0.0
        return (self.turn_end_ns - self.turn_start_ns) / 1_000_000

    def get_stage_latency(self, stage: str) -> float:
        span = self.spans.get(stage)
        if span and span.is_complete:
            return span.duration_ms
        return 0.0

    def summary(self) -> Dict[str, float]:
        result = {}
        for stage in STAGES:
            result[stage] = self.get_stage_latency(stage)
        result["end_to_end"] = self.end_to_end_ms
        return result


class LatencyHistogram:
    """Simple in-memory histogram for latency percentile calculation."""

    def __init__(self, max_samples: int = 1000):
        self._samples: List[float] = []
        self._max_samples = max_samples

    def record(self, value_ms: float):
        self._samples.append(value_ms)
        if len(self._samples) > self._max_samples:
            self._samples.pop(0)

    def percentile(self, p: float) -> float:
        if not self._samples:
            return 0.0
        sorted_samples = sorted(self._samples)
        idx = int(len(sorted_samples) * p / 100.0)
        idx = min(idx, len(sorted_samples) - 1)
        return sorted_samples[idx]

    @property
    def p50(self) -> float:
        return self.percentile(50)

    @property
    def p95(self) -> float:
        return self.percentile(95)

    @property
    def p99(self) -> float:
        return self.percentile(99)

    @property
    def count(self) -> int:
        return len(self._samples)

    @property
    def max(self) -> float:
        return max(self._samples) if self._samples else 0.0


class LatencyTracer:
    """
    Per-stage latency tracer for the AIOS pipeline.

    Usage:
        tracer.start_turn(turn_id)
        tracer.start_span("asr", turn_id)
        # ... do ASR work ...
        tracer.end_span("asr", turn_id)
        tracer.end_turn(turn_id)
    """

    def __init__(self, max_traces: int = 100):
        self._traces: Dict[int, TurnTrace] = {}
        self._max_traces = max_traces
        self._histograms: Dict[str, LatencyHistogram] = defaultdict(LatencyHistogram)
        self._barge_in_histogram = LatencyHistogram()

    def start_turn(self, turn_id: int):
        """Record the start of a conversational turn."""
        trace = TurnTrace(turn_id=turn_id, turn_start_ns=time.perf_counter_ns())
        self._traces[turn_id] = trace
        self._evict_old_traces()

    def end_turn(self, turn_id: int):
        """Record the end of a conversational turn."""
        trace = self._traces.get(turn_id)
        if trace:
            trace.turn_end_ns = time.perf_counter_ns()
            e2e = trace.end_to_end_ms
            self._histograms["end_to_end"].record(e2e)

            # Record per-stage histograms
            for stage in STAGES:
                latency = trace.get_stage_latency(stage)
                if latency > 0:
                    self._histograms[stage].record(latency)

            # Log turn summary
            summary = trace.summary()
            stages_str = ", ".join(
                f"{k}={v:.0f}ms" for k, v in summary.items() if v > 0
            )
            logger.info(
                "Turn %d latency: %s",
                turn_id, stages_str,
            )

            # Check for threshold violations
            self._check_thresholds(turn_id, summary)

    def start_span(self, stage: str, turn_id: int):
        """Record the start of a pipeline stage span."""
        trace = self._traces.get(turn_id)
        if trace:
            trace.spans[stage] = Span(
                stage=stage,
                turn_id=turn_id,
                start_ns=time.perf_counter_ns(),
            )

    def end_span(self, stage: str, turn_id: int) -> float:
        """
        Record the end of a pipeline stage span.

        Returns:
            Duration in milliseconds.
        """
        trace = self._traces.get(turn_id)
        if trace and stage in trace.spans:
            trace.spans[stage].end_ns = time.perf_counter_ns()
            return trace.spans[stage].duration_ms
        return 0.0

    def record_barge_in(self, latency_ms: float):
        """Record a barge-in latency measurement."""
        self._barge_in_histogram.record(latency_ms)

    def get_trace(self, turn_id: int) -> Optional[TurnTrace]:
        return self._traces.get(turn_id)

    def get_histograms(self) -> Dict[str, dict]:
        """Get histogram summaries for all stages."""
        result = {}
        for name, hist in self._histograms.items():
            if hist.count > 0:
                result[name] = {
                    "p50": hist.p50,
                    "p95": hist.p95,
                    "p99": hist.p99,
                    "max": hist.max,
                    "count": hist.count,
                }
        if self._barge_in_histogram.count > 0:
            result["barge_in"] = {
                "p50": self._barge_in_histogram.p50,
                "p95": self._barge_in_histogram.p95,
                "p99": self._barge_in_histogram.p99,
                "max": self._barge_in_histogram.max,
                "count": self._barge_in_histogram.count,
            }
        return result

    def get_profile(self) -> str:
        """Get a formatted latency profile string."""
        lines = [
            f"{'Stage':<20} {'p50':>8} {'p95':>8} {'p99':>8} {'max':>8}",
        ]
        for name in STAGES + ["end_to_end", "barge_in"]:
            hist = self._histograms.get(name) or (
                self._barge_in_histogram if name == "barge_in" else None
            )
            if hist and hist.count > 0:
                lines.append(
                    f"{name:<20} {hist.p50:>7.0f}ms {hist.p95:>7.0f}ms "
                    f"{hist.p99:>7.0f}ms {hist.max:>7.0f}ms"
                )
        return "\n".join(lines)

    def _check_thresholds(self, turn_id: int, summary: Dict[str, float]):
        """Check if any stage exceeds its latency target or hard limit."""
        for stage, latency in summary.items():
            if latency <= 0:
                continue
            hard_limit = LATENCY_HARD_LIMITS.get(stage)
            target = LATENCY_TARGETS.get(stage)

            if hard_limit and latency > hard_limit:
                logger.error(
                    "Turn %d: %s EXCEEDS HARD LIMIT: %.0fms > %.0fms",
                    turn_id, stage, latency, hard_limit,
                )
            elif target and latency > target:
                logger.warning(
                    "Turn %d: %s exceeds target: %.0fms > %.0fms",
                    turn_id, stage, latency, target,
                )

    def _evict_old_traces(self):
        """Evict oldest traces if over capacity."""
        while len(self._traces) > self._max_traces:
            oldest_id = min(self._traces.keys())
            del self._traces[oldest_id]
