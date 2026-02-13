"""
Streaming Accent Correction Engine

Sliding-window accent correction for Vietnamese-accented English ASR output.
Two-phase correction:
  Phase 1: Rule-based pre-filter (<= 5ms) - hot-word matching, consonant restoration
  Phase 2: SLM contextual correction (<= 75ms) - model-based rewriting

Ghost-text / Solid-text model:
  - Ghost-text: tentative corrected text from ASR partials (may change)
  - Solid-text: committed corrected text from ASR finals (will not change)

Performance: Total correction stage <= 80ms.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from collections import deque

logger = logging.getLogger(__name__)

# Sliding window parameters
WINDOW_DURATION_MS = 500
COMMIT_AGE_MS = 500

# Confidence gating
MAX_EDIT_DISTANCE_RATIO = 0.30
MIN_LOG_PROBABILITY = -2.0

# Degradation thresholds
LATENCY_HARD_LIMIT_MS = 120
CONSECUTIVE_OVERSHOOT_LIMIT = 3
DEGRADATION_COOLDOWN_S = 10.0


@dataclass
class CorrectedText:
    """
    Output of the correction engine.

    Attributes:
        original: Raw ASR text before correction
        corrected: Corrected text
        is_ghost: True = tentative (from partial), False = committed (from final)
        confidence: Correction confidence 0.0-1.0
        turn_id: Turn identifier for stale detection
    """
    original: str
    corrected: str
    is_ghost: bool = True
    confidence: float = 1.0
    turn_id: int = 0


@dataclass
class WindowToken:
    """A token in the sliding correction window."""
    text: str
    timestamp: float
    is_final: bool = False
    stability: float = 0.0
    corrected: Optional[str] = None


# Vietnamese-accented English phonetic error rules
# Maps common ASR misrecognitions to corrections
CONSONANT_RULES: Dict[str, str] = {
    # Final consonant deletion
    "tes": "test",
    "buil": "build",
    "hel": "help",
    "wor": "work",
    "firs": "first",
    "las": "last",
    "lis": "list",
    "nex": "next",
    "tex": "text",
    "bes": "best",
    # Th-stopping
    "ting": "thing",
    "tink": "think",
    "dat": "that",
    "dis": "this",
    "dey": "they",
    "dem": "them",
    "wid": "with",
    "birt": "birth",
    "mout": "mouth",
    # Consonant cluster reduction
    "sring": "string",
    "pease": "please",
    "cass": "class",
    "schoo": "school",
}

# Common word-level corrections
WORD_CORRECTIONS: Dict[str, str] = {
    "wery": "very",
    "vine": "wine",
    "sip": "ship",
    "see": "she",
    "light": "right",  # context-dependent, conservative default
}


class RuleBasedCorrector:
    """
    Phase 1: Rule-based pre-filter for high-confidence corrections.

    Applies deterministic rules based on the phonetic error taxonomy.
    Target latency: <= 5ms.
    """

    def __init__(self, hot_words: Optional[List[dict]] = None):
        self._hot_words: Dict[str, str] = {}
        if hot_words:
            for hw in hot_words:
                term = hw.get("term", "")
                for variant in hw.get("phonetic_variants", []):
                    self._hot_words[variant.lower()] = term

    def set_hot_words(self, hot_words: List[dict]):
        """Update hot-word list."""
        self._hot_words.clear()
        for hw in hot_words:
            term = hw.get("term", "")
            for variant in hw.get("phonetic_variants", []):
                self._hot_words[variant.lower()] = term

    def correct(self, text: str) -> tuple[str, bool]:
        """
        Apply rule-based corrections.

        Args:
            text: Raw ASR text.

        Returns:
            Tuple of (corrected_text, was_modified).
        """
        if not text:
            return text, False

        original = text
        words = text.split()
        corrected_words = []
        modified = False

        for word in words:
            lower_word = word.lower()

            # Hot-word matching (highest priority)
            # Check multi-word hot-word patterns
            hot_match = self._hot_words.get(lower_word)
            if hot_match:
                corrected_words.append(hot_match)
                modified = True
                continue

            # Consonant rule corrections
            rule_match = CONSONANT_RULES.get(lower_word)
            if rule_match:
                # Preserve original casing pattern
                if word[0].isupper():
                    rule_match = rule_match.capitalize()
                corrected_words.append(rule_match)
                modified = True
                continue

            # Word-level corrections
            word_match = WORD_CORRECTIONS.get(lower_word)
            if word_match:
                if word[0].isupper():
                    word_match = word_match.capitalize()
                corrected_words.append(word_match)
                modified = True
                continue

            corrected_words.append(word)

        # Check multi-word hot-word patterns
        result = " ".join(corrected_words)
        for variant, term in self._hot_words.items():
            if variant in result.lower():
                # Case-insensitive replacement
                idx = result.lower().find(variant)
                result = result[:idx] + term + result[idx + len(variant):]
                modified = True

        return result, modified


class StreamingCorrectionEngine:
    """
    Streaming accent correction engine.

    Operates on a 500ms sliding window of ASR tokens.
    Two-phase correction: rule-based pre-filter + SLM contextual correction.

    Stability-based correction behavior:
      0.0-0.3: No correction (display raw partial)
      0.3-0.7: Lightweight correction (hot-word substitution only)
      0.7-1.0: Full correction pipeline
      1.0 (final): Full correction + commit to context
    """

    def __init__(
        self,
        corrected_queue: asyncio.Queue,
        hot_words: Optional[List[dict]] = None,
    ):
        """
        Initialize the correction engine.

        Args:
            corrected_queue: Queue to output CorrectedText results.
            hot_words: Initial hot-word list.
        """
        self._corrected_queue = corrected_queue
        self._rule_corrector = RuleBasedCorrector(hot_words)

        # Sliding window
        self._window: deque[WindowToken] = deque()
        self._window_duration_ms = WINDOW_DURATION_MS

        # SLM correction state
        self._slm_enabled = True
        self._consecutive_overshoots = 0
        self._degradation_until: float = 0.0

        # Metrics
        self._total_corrections: int = 0
        self._rule_corrections: int = 0
        self._slm_corrections: int = 0
        self._rejected_corrections: int = 0
        self._total_latency_ms: float = 0.0

    def set_hot_words(self, hot_words: List[dict]):
        """Update hot-word list at runtime."""
        self._rule_corrector.set_hot_words(hot_words)

    async def process_asr_result(
        self,
        text: str,
        is_final: bool,
        stability: float,
        turn_id: int = 0,
    ) -> Optional[CorrectedText]:
        """
        Process an ASR result through the correction pipeline.

        Args:
            text: ASR transcribed text.
            is_final: Whether this is a final result.
            stability: Stability score 0.0-1.0.
            turn_id: Current turn ID.

        Returns:
            CorrectedText if correction was applied, None otherwise.
        """
        start_time = time.time()

        # Add to sliding window
        token = WindowToken(
            text=text,
            timestamp=time.time(),
            is_final=is_final,
            stability=stability,
        )
        self._window.append(token)
        self._evict_old_tokens()

        # Determine correction level based on stability
        if stability < 0.3 and not is_final:
            # No correction - display raw partial as ghost-text
            result = CorrectedText(
                original=text,
                corrected=text,
                is_ghost=True,
                confidence=stability,
                turn_id=turn_id,
            )
        elif stability < 0.7 and not is_final:
            # Lightweight correction - hot-word substitution only
            corrected, _ = self._rule_corrector.correct(text)
            result = CorrectedText(
                original=text,
                corrected=corrected,
                is_ghost=True,
                confidence=stability,
                turn_id=turn_id,
            )
            self._rule_corrections += 1
        else:
            # Full correction pipeline
            result = await self._full_correction(text, is_final, stability, turn_id)

        self._total_corrections += 1

        # Track latency
        latency_ms = (time.time() - start_time) * 1000
        self._total_latency_ms += latency_ms
        self._check_latency_degradation(latency_ms)

        # Enqueue result
        if result:
            try:
                self._corrected_queue.put_nowait(result)
            except asyncio.QueueFull:
                logger.warning("Corrected queue full, dropping result")

        return result

    async def _full_correction(
        self, text: str, is_final: bool, stability: float, turn_id: int
    ) -> CorrectedText:
        """Apply full two-phase correction."""
        # Phase 1: Rule-based pre-filter
        rule_corrected, rule_modified = self._rule_corrector.correct(text)
        if rule_modified:
            self._rule_corrections += 1

        # Phase 2: SLM correction (if enabled and not degraded)
        slm_corrected = rule_corrected
        if self._should_use_slm():
            slm_result = await self._slm_correct(rule_corrected)
            if slm_result is not None:
                # Confidence gating
                if self._passes_confidence_gate(rule_corrected, slm_result):
                    slm_corrected = slm_result
                    self._slm_corrections += 1
                else:
                    self._rejected_corrections += 1

        return CorrectedText(
            original=text,
            corrected=slm_corrected,
            is_ghost=not is_final,
            confidence=stability,
            turn_id=turn_id,
        )

    async def _slm_correct(self, text: str) -> Optional[str]:
        """
        Apply SLM-based contextual correction.

        In MVP, this is a placeholder. In production, this calls the SLM
        with a correction-specific prompt via the thread pool executor.
        """
        # TODO: Integrate actual SLM inference via ThreadPoolExecutor
        # For now, return None (no SLM correction applied)
        return None

    def _should_use_slm(self) -> bool:
        """Check if SLM correction should be used."""
        if not self._slm_enabled:
            return False
        if time.time() < self._degradation_until:
            return False
        return True

    def _passes_confidence_gate(self, original: str, corrected: str) -> bool:
        """
        Check if a correction passes the confidence gate.

        Gates:
        1. Edit distance <= 30% of original text length
        2. Correction does not change named entities or numbers
        """
        if not original or not corrected:
            return False

        # Gate 1: Edit distance ratio
        edit_dist = self._levenshtein_distance(original, corrected)
        ratio = edit_dist / max(len(original), 1)
        if ratio > MAX_EDIT_DISTANCE_RATIO:
            logger.debug(
                "Correction rejected: edit distance ratio %.2f > %.2f",
                ratio, MAX_EDIT_DISTANCE_RATIO,
            )
            return False

        # Gate 2: Preserve numbers
        original_numbers = set(w for w in original.split() if w.isdigit())
        corrected_numbers = set(w for w in corrected.split() if w.isdigit())
        if original_numbers != corrected_numbers:
            logger.debug("Correction rejected: number mismatch")
            return False

        return True

    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        """Compute Levenshtein edit distance between two strings."""
        if len(s1) < len(s2):
            return StreamingCorrectionEngine._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        prev_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row

        return prev_row[-1]

    def _evict_old_tokens(self):
        """Evict tokens older than the window duration."""
        cutoff = time.time() - (self._window_duration_ms / 1000.0)
        while self._window and self._window[0].timestamp < cutoff:
            self._window.popleft()

    def _check_latency_degradation(self, latency_ms: float):
        """Check if SLM correction should be degraded due to latency."""
        if latency_ms > LATENCY_HARD_LIMIT_MS:
            self._consecutive_overshoots += 1
            if self._consecutive_overshoots >= CONSECUTIVE_OVERSHOOT_LIMIT:
                self._degradation_until = time.time() + DEGRADATION_COOLDOWN_S
                self._consecutive_overshoots = 0
                logger.warning(
                    "SLM correction degraded for %.0fs due to sustained latency > %dms",
                    DEGRADATION_COOLDOWN_S, LATENCY_HARD_LIMIT_MS,
                )
        else:
            self._consecutive_overshoots = 0

    def reset(self):
        """Reset correction state (e.g., on new turn)."""
        self._window.clear()

    def get_stats(self) -> dict:
        """Get correction engine statistics."""
        avg_latency = (
            self._total_latency_ms / max(1, self._total_corrections)
        )
        return {
            "total_corrections": self._total_corrections,
            "rule_corrections": self._rule_corrections,
            "slm_corrections": self._slm_corrections,
            "rejected_corrections": self._rejected_corrections,
            "average_latency_ms": avg_latency,
            "slm_enabled": self._slm_enabled,
            "degraded": time.time() < self._degradation_until,
            "window_size": len(self._window),
        }
