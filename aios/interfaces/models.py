"""
Model Abstractions for AIOS

Defines Protocol classes for ASR, SLM, and TTS models, enabling model swapping,
testing with mocks, and future provider changes without modifying pipeline code.

All models implement a common pattern:
- async lifecycle (load, is_ready, close)
- async streaming inference
- cancellation support via asyncio.Task.cancel()

MVP models:
- ASR: PersonaPlex (gRPC streaming)
- SLM: Phi-3-mini-4k-instruct (GPTQ 4-bit, local inference)
- TTS: NVIDIA Riva TTS (gRPC streaming) or VITS (local fallback)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import (
    AsyncIterator,
    Optional,
    List,
    Protocol,
    runtime_checkable,
)

logger = logging.getLogger(__name__)


# ── Data types ──

@dataclass
class ASRResult:
    """Result from ASR model inference."""
    text: str
    is_final: bool = False
    stability: float = 0.0
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.monotonic)
    turn_id: int = 0


@dataclass
class ModelInfo:
    """Metadata about a loaded model."""
    name: str
    version: str
    provider: str
    vram_bytes: int = 0
    is_fallback: bool = False


# ── Protocol classes ──

@runtime_checkable
class ASRModel(Protocol):
    """
    Abstract interface for ASR models.

    Contract:
    - stream() accepts an async iterator of 20ms PCM frames and yields ASRResult.
    - close() sends half-close and waits for final results (timeout: 3s).
    - is_ready() returns True when model is loaded and connection established.
    - Implementation must handle reconnection internally.
    """

    async def stream(self, audio: AsyncIterator[bytes]) -> AsyncIterator[ASRResult]: ...
    async def close(self) -> None: ...
    def is_ready(self) -> bool: ...
    def info(self) -> ModelInfo: ...


@runtime_checkable
class SLMModel(Protocol):
    """
    Abstract interface for SLM (Small Language Model).

    Contract:
    - generate() accepts a prompt and yields response tokens one at a time.
    - Must support asyncio.Task.cancel() -- stops generation, yields no further tokens.
    - load() loads model weights into GPU VRAM. Blocks until complete.
    - vram_usage_bytes() returns current GPU memory allocated by this model.
    """

    async def generate(self, prompt: str, max_tokens: int = 512) -> AsyncIterator[str]: ...
    async def load(self, model_path: str) -> None: ...
    def is_ready(self) -> bool: ...
    def vram_usage_bytes(self) -> int: ...
    def info(self) -> ModelInfo: ...


@runtime_checkable
class TTSModel(Protocol):
    """
    Abstract interface for TTS models.

    Contract:
    - synthesize() accepts text and yields PCM audio frames (20ms each).
    - First audio frame should be yielded within 100ms.
    - sample_rate() returns native output sample rate (e.g., 24000).
    """

    async def synthesize(self, text: str) -> AsyncIterator[bytes]: ...
    async def load(self, model_path: str) -> None: ...
    def is_ready(self) -> bool: ...
    def sample_rate(self) -> int: ...
    def info(self) -> ModelInfo: ...


# ── Mock implementations (for testing) ──

class MockASRModel:
    """Mock ASR model that returns pre-configured results."""

    def __init__(
        self,
        results: Optional[List[ASRResult]] = None,
        latency_ms: float = 10.0,
    ):
        self._results = results or []
        self._latency_ms = latency_ms
        self._ready = True
        self._closed = False

    async def stream(self, audio: AsyncIterator[bytes]) -> AsyncIterator[ASRResult]:
        frame_count = 0
        async for frame in audio:
            frame_count += 1
            await asyncio.sleep(self._latency_ms / 1000.0)

            # Yield pre-configured results at appropriate intervals
            if self._results and frame_count % 10 == 0:
                idx = min(frame_count // 10 - 1, len(self._results) - 1)
                yield self._results[idx]

    async def close(self) -> None:
        self._closed = True

    def is_ready(self) -> bool:
        return self._ready

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="mock-asr",
            version="1.0",
            provider="mock",
        )


class MockSLMModel:
    """Mock SLM that returns pre-configured response tokens."""

    def __init__(
        self,
        response_tokens: Optional[List[str]] = None,
        latency_per_token_ms: float = 5.0,
    ):
        self._response_tokens = response_tokens or ["I ", "understand. ", "How ", "can ", "I ", "help?"]
        self._latency_per_token_ms = latency_per_token_ms
        self._ready = True
        self._vram = 0

    async def generate(self, prompt: str, max_tokens: int = 512) -> AsyncIterator[str]:
        for token in self._response_tokens[:max_tokens]:
            await asyncio.sleep(self._latency_per_token_ms / 1000.0)
            yield token

    async def load(self, model_path: str) -> None:
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    def vram_usage_bytes(self) -> int:
        return self._vram

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="mock-slm",
            version="1.0",
            provider="mock",
        )


class MockTTSModel:
    """Mock TTS that returns synthetic silence frames."""

    def __init__(
        self,
        frame_duration_ms: int = 20,
        output_sample_rate: int = 24000,
        latency_ms: float = 10.0,
    ):
        self._frame_duration_ms = frame_duration_ms
        self._sample_rate_val = output_sample_rate
        self._latency_ms = latency_ms
        self._ready = True
        # Pre-compute a silence frame
        samples_per_frame = output_sample_rate * frame_duration_ms // 1000
        self._silence_frame = bytes(samples_per_frame * 2)  # 16-bit silence

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        # Estimate number of frames: ~100ms of audio per word
        word_count = max(1, len(text.split()))
        frames_per_word = max(1, (100 * self._sample_rate_val) // (self._frame_duration_ms * self._sample_rate_val))
        total_frames = word_count * frames_per_word

        await asyncio.sleep(self._latency_ms / 1000.0)  # Initial latency

        for _ in range(total_frames):
            yield self._silence_frame

    async def load(self, model_path: str) -> None:
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    def sample_rate(self) -> int:
        return self._sample_rate_val

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="mock-tts",
            version="1.0",
            provider="mock",
        )


# ── Model Manager ──

class ModelManager:
    """
    Manages model lifecycle: loading, health checks, fallback activation.

    Startup sequence:
    1. Load primary models (ASR connection, SLM weights, TTS connection)
    2. Load fallback models into CPU memory (dormant)
    3. Run health checks on all primary models
    4. Emit all_models_ready

    Fallback strategy:
    - ASR: PersonaPlex -> Whisper-tiny (local CPU)
    - SLM (correction): Phi-3-mini -> rule-based only
    - SLM (response): Phi-3-mini -> canned response
    - TTS: Riva -> VITS (local)
    """

    def __init__(self):
        self._asr: Optional[ASRModel] = None
        self._slm: Optional[SLMModel] = None
        self._tts: Optional[TTSModel] = None

        self._asr_fallback: Optional[ASRModel] = None
        self._slm_fallback: Optional[SLMModel] = None
        self._tts_fallback: Optional[TTSModel] = None

        self._using_asr_fallback = False
        self._using_slm_fallback = False
        self._using_tts_fallback = False

        # Failure tracking for fallback triggers
        self._asr_consecutive_failures = 0
        self._slm_consecutive_failures = 0
        self._tts_consecutive_failures = 0
        self._max_consecutive_failures = 3

        # Recovery timers
        self._asr_recovery_interval = 30.0
        self._slm_recovery_interval = 10.0
        self._tts_recovery_interval = 30.0
        self._last_asr_recovery_attempt = 0.0
        self._last_slm_recovery_attempt = 0.0
        self._last_tts_recovery_attempt = 0.0

        self._ready = False

    def set_primary_models(
        self,
        asr: ASRModel,
        slm: SLMModel,
        tts: TTSModel,
    ):
        """Set primary model instances."""
        self._asr = asr
        self._slm = slm
        self._tts = tts

    def set_fallback_models(
        self,
        asr: Optional[ASRModel] = None,
        slm: Optional[SLMModel] = None,
        tts: Optional[TTSModel] = None,
    ):
        """Set fallback model instances (loaded dormant in CPU memory)."""
        self._asr_fallback = asr
        self._slm_fallback = slm
        self._tts_fallback = tts

    @property
    def asr(self) -> Optional[ASRModel]:
        if self._using_asr_fallback and self._asr_fallback:
            return self._asr_fallback
        return self._asr

    @property
    def slm(self) -> Optional[SLMModel]:
        if self._using_slm_fallback and self._slm_fallback:
            return self._slm_fallback
        return self._slm

    @property
    def tts(self) -> Optional[TTSModel]:
        if self._using_tts_fallback and self._tts_fallback:
            return self._tts_fallback
        return self._tts

    def report_asr_failure(self):
        """Report an ASR inference failure."""
        self._asr_consecutive_failures += 1
        if self._asr_consecutive_failures >= self._max_consecutive_failures:
            self._activate_asr_fallback()

    def report_asr_success(self):
        """Report a successful ASR inference."""
        self._asr_consecutive_failures = 0

    def report_slm_failure(self):
        """Report an SLM inference failure."""
        self._slm_consecutive_failures += 1
        if self._slm_consecutive_failures >= self._max_consecutive_failures:
            self._activate_slm_fallback()

    def report_slm_success(self):
        """Report a successful SLM inference."""
        self._slm_consecutive_failures = 0

    def report_tts_failure(self):
        """Report a TTS inference failure."""
        self._tts_consecutive_failures += 1
        if self._tts_consecutive_failures >= self._max_consecutive_failures:
            self._activate_tts_fallback()

    def report_tts_success(self):
        """Report a successful TTS inference."""
        self._tts_consecutive_failures = 0

    def _activate_asr_fallback(self):
        if self._asr_fallback and not self._using_asr_fallback:
            self._using_asr_fallback = True
            self._last_asr_recovery_attempt = time.monotonic()
            logger.warning("ASR fallback activated after %d failures", self._asr_consecutive_failures)

    def _activate_slm_fallback(self):
        if self._slm_fallback and not self._using_slm_fallback:
            self._using_slm_fallback = True
            self._last_slm_recovery_attempt = time.monotonic()
            logger.warning("SLM fallback activated after %d failures", self._slm_consecutive_failures)

    def _activate_tts_fallback(self):
        if self._tts_fallback and not self._using_tts_fallback:
            self._using_tts_fallback = True
            self._last_tts_recovery_attempt = time.monotonic()
            logger.warning("TTS fallback activated after %d failures", self._tts_consecutive_failures)

    async def attempt_recovery(self):
        """Attempt to recover primary models from fallback state."""
        now = time.monotonic()

        if self._using_asr_fallback and self._asr:
            if now - self._last_asr_recovery_attempt >= self._asr_recovery_interval:
                self._last_asr_recovery_attempt = now
                if self._asr.is_ready():
                    self._using_asr_fallback = False
                    self._asr_consecutive_failures = 0
                    logger.info("ASR recovered to primary model")

        if self._using_slm_fallback and self._slm:
            if now - self._last_slm_recovery_attempt >= self._slm_recovery_interval:
                self._last_slm_recovery_attempt = now
                if self._slm.is_ready():
                    self._using_slm_fallback = False
                    self._slm_consecutive_failures = 0
                    logger.info("SLM recovered to primary model")

        if self._using_tts_fallback and self._tts:
            if now - self._last_tts_recovery_attempt >= self._tts_recovery_interval:
                self._last_tts_recovery_attempt = now
                if self._tts.is_ready():
                    self._using_tts_fallback = False
                    self._tts_consecutive_failures = 0
                    logger.info("TTS recovered to primary model")

    def is_ready(self) -> bool:
        """Check if all primary models are ready."""
        asr_ok = self._asr is not None and self._asr.is_ready()
        slm_ok = self._slm is not None and self._slm.is_ready()
        tts_ok = self._tts is not None and self._tts.is_ready()
        return asr_ok and slm_ok and tts_ok

    def get_status(self) -> dict:
        return {
            "asr_ready": self._asr.is_ready() if self._asr else False,
            "slm_ready": self._slm.is_ready() if self._slm else False,
            "tts_ready": self._tts.is_ready() if self._tts else False,
            "asr_fallback_active": self._using_asr_fallback,
            "slm_fallback_active": self._using_slm_fallback,
            "tts_fallback_active": self._using_tts_fallback,
            "asr_failures": self._asr_consecutive_failures,
            "slm_failures": self._slm_consecutive_failures,
            "tts_failures": self._tts_consecutive_failures,
        }
