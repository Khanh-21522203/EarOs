"""
Graceful Degradation Manager

When latency exceeds targets, AIOS degrades in a defined order:

Level | Trigger                                          | Action
L0    | All stages within budget                         | Full pipeline
L1    | Correction exceeds 140ms for 3 consecutive       | Disable SLM correction, rule-based only
L2    | SLM first token exceeds 400ms                    | Reduce context window from 10 to 3 turns
L3    | TTS exceeds 200ms                                | Switch to lower-quality faster TTS
L4    | Multiple stages exceeding limits                 | Skip correction entirely, minimal context

Each level has a fixed 10s recovery timer. After 10s, the system attempts
to restore the previous level. If it immediately exceeds again, it degrades back.
"""

import logging
import time
from enum import IntEnum
from typing import Optional

logger = logging.getLogger(__name__)

RECOVERY_TIMER_S = 10.0


class DegradationLevel(IntEnum):
    L0_NORMAL = 0
    L1_CORRECTION_SKIP = 1
    L2_CONTEXT_REDUCTION = 2
    L3_TTS_DEGRADATION = 3
    L4_PASS_THROUGH = 4


class DegradationManager:
    """
    Manages graceful degradation of the AIOS pipeline.

    Tracks per-stage latency violations and escalates/recovers
    degradation levels accordingly.
    """

    def __init__(self):
        self._level = DegradationLevel.L0_NORMAL
        self._level_entered_at = time.monotonic()

        # Consecutive overshoot counters
        self._correction_overshoots = 0
        self._slm_overshoots = 0
        self._tts_overshoots = 0

        # Thresholds
        self._correction_hard_limit_ms = 140.0
        self._slm_hard_limit_ms = 400.0
        self._tts_hard_limit_ms = 200.0
        self._consecutive_limit = 3

        # Recovery
        self._recovery_timer_s = RECOVERY_TIMER_S
        self._total_degradations = 0
        self._total_recoveries = 0

    @property
    def level(self) -> DegradationLevel:
        return self._level

    @property
    def is_degraded(self) -> bool:
        return self._level > DegradationLevel.L0_NORMAL

    @property
    def skip_slm_correction(self) -> bool:
        return self._level >= DegradationLevel.L1_CORRECTION_SKIP

    @property
    def reduced_context(self) -> bool:
        return self._level >= DegradationLevel.L2_CONTEXT_REDUCTION

    @property
    def context_turn_limit(self) -> int:
        if self._level >= DegradationLevel.L4_PASS_THROUGH:
            return 1
        if self._level >= DegradationLevel.L2_CONTEXT_REDUCTION:
            return 3
        return 10

    @property
    def skip_correction_entirely(self) -> bool:
        return self._level >= DegradationLevel.L4_PASS_THROUGH

    def report_correction_latency(self, latency_ms: float):
        """Report a correction stage latency measurement."""
        if latency_ms > self._correction_hard_limit_ms:
            self._correction_overshoots += 1
            if self._correction_overshoots >= self._consecutive_limit:
                self._escalate_to(DegradationLevel.L1_CORRECTION_SKIP)
        else:
            self._correction_overshoots = 0

    def report_slm_latency(self, latency_ms: float):
        """Report an SLM first-token latency measurement."""
        if latency_ms > self._slm_hard_limit_ms:
            self._slm_overshoots += 1
            if self._slm_overshoots >= self._consecutive_limit:
                self._escalate_to(DegradationLevel.L2_CONTEXT_REDUCTION)
        else:
            self._slm_overshoots = 0

    def report_tts_latency(self, latency_ms: float):
        """Report a TTS first-frame latency measurement."""
        if latency_ms > self._tts_hard_limit_ms:
            self._tts_overshoots += 1
            if self._tts_overshoots >= self._consecutive_limit:
                self._escalate_to(DegradationLevel.L3_TTS_DEGRADATION)
        else:
            self._tts_overshoots = 0

    def check_multi_stage_failure(self):
        """Check if multiple stages are failing simultaneously."""
        failing_count = sum([
            self._correction_overshoots >= self._consecutive_limit,
            self._slm_overshoots >= self._consecutive_limit,
            self._tts_overshoots >= self._consecutive_limit,
        ])
        if failing_count >= 2:
            self._escalate_to(DegradationLevel.L4_PASS_THROUGH)

    def attempt_recovery(self):
        """
        Attempt to recover to a lower degradation level.

        Called periodically (e.g., every turn or every few seconds).
        """
        if self._level == DegradationLevel.L0_NORMAL:
            return

        elapsed = time.monotonic() - self._level_entered_at
        if elapsed < self._recovery_timer_s:
            return

        previous_level = self._level
        new_level = DegradationLevel(max(0, self._level - 1))
        self._level = new_level
        self._level_entered_at = time.monotonic()
        self._total_recoveries += 1

        # Reset the overshoot counter for the recovered stage
        if previous_level == DegradationLevel.L1_CORRECTION_SKIP:
            self._correction_overshoots = 0
        elif previous_level == DegradationLevel.L2_CONTEXT_REDUCTION:
            self._slm_overshoots = 0
        elif previous_level == DegradationLevel.L3_TTS_DEGRADATION:
            self._tts_overshoots = 0

        logger.info(
            "Degradation recovery: L%d -> L%d (after %.0fs)",
            previous_level, new_level, elapsed,
        )

    def _escalate_to(self, target: DegradationLevel):
        """Escalate to a higher degradation level."""
        if target <= self._level:
            return

        previous = self._level
        self._level = target
        self._level_entered_at = time.monotonic()
        self._total_degradations += 1

        logger.warning(
            "Degradation escalation: L%d -> L%d",
            previous, target,
        )

    def get_status(self) -> dict:
        return {
            "level": int(self._level),
            "level_name": self._level.name,
            "is_degraded": self.is_degraded,
            "time_in_level_s": time.monotonic() - self._level_entered_at,
            "correction_overshoots": self._correction_overshoots,
            "slm_overshoots": self._slm_overshoots,
            "tts_overshoots": self._tts_overshoots,
            "total_degradations": self._total_degradations,
            "total_recoveries": self._total_recoveries,
        }
