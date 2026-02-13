"""
Data models for PersonaPlex ASR integration.

Defines the data structures for streaming requests, responses, and results.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum
from datetime import datetime
import time


class ResultType(Enum):
    """Type of ASR result."""
    PARTIAL = "partial"  # Interim, unstable result
    FINAL = "final"      # Final, stable result


class CorrectionLevel(Enum):
    """Correction level based on stability score."""
    NONE = "none"                    # 0.0-0.3: No correction
    LIGHTWEIGHT = "lightweight"      # 0.3-0.7: Hot-word substitution only
    FULL = "full"                    # 0.7-1.0: Full correction pipeline
    COMMITTED = "committed"          # 1.0: Final result, commit to context


@dataclass
class ASRResult:
    """
    Represents a single ASR result from PersonaPlex.

    Attributes:
        text: The transcribed text
        is_final: Whether this is a final (stable) result
        stability: Stability score 0.0-1.0 (1.0 = final)
        confidence: Confidence score 0.0-1.0
        alternatives: List of alternative transcriptions
        timestamp: When this result was received
        audio_offset_ms: Offset from start of audio in milliseconds
    """
    text: str
    is_final: bool
    stability: float = 0.0
    confidence: float = 0.0
    alternatives: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    audio_offset_ms: int = 0

    @property
    def result_type(self) -> ResultType:
        """Get the result type enum."""
        return ResultType.FINAL if self.is_final else ResultType.PARTIAL

    @property
    def correction_level(self) -> CorrectionLevel:
        """Determine correction level based on stability."""
        if self.is_final:
            return CorrectionLevel.COMMITTED
        if self.stability >= 0.7:
            return CorrectionLevel.FULL
        if self.stability >= 0.3:
            return CorrectionLevel.LIGHTWEIGHT
        return CorrectionLevel.NONE

    def should_correct(self) -> bool:
        """Check if this result should be corrected."""
        return self.correction_level in (
            CorrectionLevel.FULL,
            CorrectionLevel.COMMITTED
        )


@dataclass
class StreamingRecognitionConfig:
    """
    Configuration for streaming recognition request.

    Corresponds to PersonaPlex StreamingRecognitionConfig proto.
    """
    interim_results: bool = True
    single_utterance: bool = False
    model: str = "personaplex-en-us"
    sample_rate_hertz: int = 16000
    encoding: str = "LINEAR16"
    language_code: str = "en-US"
    enable_speaker_diarization: bool = False
    max_alternatives: int = 1
    profanity_filter: bool = False
    enable_automatic_punctuation: bool = True
    speech_contexts: List[str] = field(default_factory=list)  # Hot-word boosts


@dataclass
class LiveRecognitionRequest:
    """
    A request in the streaming recognition call.

    Can contain audio data or config.
    """
    # Audio content (raw PCM bytes)
    audio_content: Optional[bytes] = None

    # Streaming config (sent in first message)
    config: Optional[StreamingRecognitionConfig] = None

    @classmethod
    def create_config(cls, config: StreamingRecognitionConfig) -> "LiveRecognitionRequest":
        """Create the initial config message."""
        return cls(config=config)

    @classmethod
    def create_audio(cls, audio: bytes) -> "LiveRecognitionRequest":
        """Create an audio chunk message."""
        return cls(audio_content=audio)


@dataclass
class StreamingMetadata:
    """
    Metadata for the streaming session.
    """
    request_id: str
    start_time: float
    audio_bytes_received: int = 0
    results_received: int = 0
    final_results_received: int = 0
    last_result_time: float = 0.0

    @property
    def duration_seconds(self) -> float:
        """Get session duration in seconds."""
        return time.time() - self.start_time

    @property
    def audio_duration_seconds(self) -> float:
        """Get audio duration based on bytes received (16kHz, 16-bit, mono)."""
        # 1 sample = 2 bytes (16-bit), 16000 samples/sec = 32000 bytes/sec
        return self.audio_bytes_received / 32000.0


@dataclass
class SessionState:
    """
    Tracks the state of a streaming session.
    """
    is_active: bool = False
    is_connected: bool = False
    reconnect_count: int = 0
    last_error: Optional[str] = None
    last_error_time: Optional[float] = None


@dataclass
class LatencyMetrics:
    """
    Tracks latency metrics for the streaming session.
    """
    first_chunk_sent: float = 0.0
    first_result_received: float = 0.0
    first_partial_latency_ms: float = 0.0

    chunk_latencies: List[float] = field(default_factory=list)

    def record_first_result(self):
        """Record latency to first result."""
        if self.first_chunk_sent > 0 and self.first_result_received == 0.0:
            self.first_result_received = time.time()
            self.first_partial_latency_ms = (
                self.first_result_received - self.first_chunk_sent
            ) * 1000

    def record_chunk_latency(self, chunk_send_time: float, result_recv_time: float):
        """Record latency for a single chunk->result cycle."""
        latency_ms = (result_recv_time - chunk_send_time) * 1000
        self.chunk_latencies.append(latency_ms)

    @property
    def average_chunk_latency_ms(self) -> float:
        """Get average chunk latency."""
        if not self.chunk_latencies:
            return 0.0
        return sum(self.chunk_latencies) / len(self.chunk_latencies)
