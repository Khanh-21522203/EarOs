"""
Unit tests for Performance modules: LatencyTracer, DegradationManager, MetricsCollector.

Tests from latency-model.md Definition of Done:
- Per-stage latency instrumentation
- Degradation levels L1-L4 activate and recover correctly
- Histogram percentile calculations
"""

import time
import pytest

from aios.performance.latency import (
    LatencyTracer,
    LatencyHistogram,
    Span,
    TurnTrace,
    LATENCY_TARGETS,
)
from aios.performance.degradation import (
    DegradationManager,
    DegradationLevel,
)


# ── LatencyHistogram ──

class TestLatencyHistogram:

    def test_record_and_percentile(self):
        h = LatencyHistogram()
        for i in range(100):
            h.record(float(i))
        assert h.p50 == pytest.approx(50, abs=2)
        assert h.p95 == pytest.approx(95, abs=2)
        assert h.count == 100

    def test_empty_histogram(self):
        h = LatencyHistogram()
        assert h.p50 == 0.0
        assert h.count == 0

    def test_max_samples_eviction(self):
        h = LatencyHistogram(max_samples=10)
        for i in range(20):
            h.record(float(i))
        assert h.count == 10


# ── Span ──

class TestSpan:

    def test_duration_ms(self):
        s = Span(stage="asr", turn_id=1, start_ns=1, end_ns=5_000_001)
        assert s.duration_ms == pytest.approx(5.0)

    def test_incomplete_span(self):
        s = Span(stage="asr", turn_id=1, start_ns=100)
        assert not s.is_complete
        assert s.duration_ms == 0.0


# ── LatencyTracer ──

class TestLatencyTracer:

    def test_start_end_turn(self):
        tracer = LatencyTracer()
        tracer.start_turn(1)
        tracer.start_span("asr", 1)
        tracer.end_span("asr", 1)
        tracer.end_turn(1)

        trace = tracer.get_trace(1)
        assert trace is not None
        assert trace.end_to_end_ms > 0
        assert trace.get_stage_latency("asr") >= 0

    def test_histogram_populated(self):
        tracer = LatencyTracer()
        for i in range(5):
            tracer.start_turn(i)
            tracer.start_span("asr", i)
            tracer.end_span("asr", i)
            tracer.end_turn(i)

        histograms = tracer.get_histograms()
        assert "end_to_end" in histograms
        assert histograms["end_to_end"]["count"] == 5

    def test_barge_in_recording(self):
        tracer = LatencyTracer()
        tracer.record_barge_in(85.0)
        tracer.record_barge_in(95.0)
        histograms = tracer.get_histograms()
        assert "barge_in" in histograms
        assert histograms["barge_in"]["count"] == 2

    def test_profile_output(self):
        tracer = LatencyTracer()
        tracer.start_turn(1)
        tracer.start_span("asr", 1)
        tracer.end_span("asr", 1)
        tracer.end_turn(1)
        profile = tracer.get_profile()
        assert "Stage" in profile

    def test_evict_old_traces(self):
        tracer = LatencyTracer(max_traces=5)
        for i in range(10):
            tracer.start_turn(i)
            tracer.end_turn(i)
        assert tracer.get_trace(0) is None
        assert tracer.get_trace(9) is not None


# ── DegradationManager ──

class TestDegradationManager:

    def test_starts_at_l0(self):
        dm = DegradationManager()
        assert dm.level == DegradationLevel.L0_NORMAL
        assert not dm.is_degraded

    def test_l1_correction_skip(self):
        dm = DegradationManager()
        for _ in range(3):
            dm.report_correction_latency(150.0)  # > 140ms
        assert dm.level == DegradationLevel.L1_CORRECTION_SKIP
        assert dm.skip_slm_correction

    def test_l2_context_reduction(self):
        dm = DegradationManager()
        for _ in range(3):
            dm.report_slm_latency(450.0)  # > 400ms
        assert dm.level == DegradationLevel.L2_CONTEXT_REDUCTION
        assert dm.reduced_context
        assert dm.context_turn_limit == 3

    def test_l3_tts_degradation(self):
        dm = DegradationManager()
        for _ in range(3):
            dm.report_tts_latency(250.0)  # > 200ms
        assert dm.level == DegradationLevel.L3_TTS_DEGRADATION

    def test_l4_multi_stage_failure(self):
        dm = DegradationManager()
        for _ in range(3):
            dm.report_correction_latency(150.0)
            dm.report_slm_latency(450.0)
        dm.check_multi_stage_failure()
        assert dm.level == DegradationLevel.L4_PASS_THROUGH
        assert dm.skip_correction_entirely
        assert dm.context_turn_limit == 1

    def test_recovery_after_timer(self):
        dm = DegradationManager()
        dm._recovery_timer_s = 0.0  # Instant recovery for testing
        for _ in range(3):
            dm.report_correction_latency(150.0)
        assert dm.level == DegradationLevel.L1_CORRECTION_SKIP

        dm.attempt_recovery()
        assert dm.level == DegradationLevel.L0_NORMAL

    def test_no_recovery_before_timer(self):
        dm = DegradationManager()
        dm._recovery_timer_s = 999.0  # Very long timer
        for _ in range(3):
            dm.report_correction_latency(150.0)
        assert dm.level == DegradationLevel.L1_CORRECTION_SKIP

        dm.attempt_recovery()
        assert dm.level == DegradationLevel.L1_CORRECTION_SKIP  # Still degraded

    def test_success_resets_counter(self):
        dm = DegradationManager()
        dm.report_correction_latency(150.0)
        dm.report_correction_latency(150.0)
        dm.report_correction_latency(50.0)  # Success resets counter
        dm.report_correction_latency(150.0)
        dm.report_correction_latency(150.0)
        assert dm.level == DegradationLevel.L0_NORMAL  # Only 2 consecutive

    def test_status(self):
        dm = DegradationManager()
        status = dm.get_status()
        assert status["level"] == 0
        assert status["is_degraded"] is False
