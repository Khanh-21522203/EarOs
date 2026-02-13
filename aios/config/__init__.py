"""
AIOS Configuration Module

Manages all configuration settings for the AIOS runtime including:
- PersonaPlex ASR configuration
- SLM configuration
- TTS configuration
- Audio settings
- GPU/CPU settings
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class DeviceType(Enum):
    """Available device types for inference."""
    CPU = "cpu"
    CUDA = "cuda"


@dataclass(frozen=True)
class AudioConfig:
    """Audio format configuration for capture and playback."""
    # Capture settings (microphone -> ASR)
    capture_sample_rate: int = 16000  # Hz
    capture_bit_depth: int = 16  # signed integer
    capture_channels: int = 1  # mono

    # Playback settings (TTS -> speaker)
    playback_sample_rate: int = 24000  # Hz
    playback_bit_depth: int = 16  # signed integer
    playback_channels: int = 1  # mono

    # Frame settings
    frame_duration_ms: int = 20  # milliseconds
    frame_samples: int = field(init=False)

    # Ring buffer settings
    ring_buffer_duration_ms: int = 200  # jitter absorption

    def __post_init__(self):
        object.__setattr__(
            self,
            'frame_samples',
            self.capture_sample_rate * self.frame_duration_ms // 1000
        )
        object.__setattr__(
            self,
            'ring_buffer_samples',
            self.capture_sample_rate * self.ring_buffer_duration_ms // 1000
        )


@dataclass(frozen=True)
class PersonaPlexConfig:
    """Configuration for NVIDIA PersonaPlex ASR integration."""
    # Connection settings
    server_host: str = "localhost"
    server_port: int = 50051
    use_ssl: bool = False
    use_mock: bool = False  # Use mock service for testing without actual server

    # Model settings
    model_name: str = "personaplex-en-us"
    language_code: str = "en-US"

    # Streaming settings
    interim_results: bool = True  # Enable partial results
    single_utterance: bool = False  # AIOS manages VAD
    enable_speaker_diarization: bool = False  # Single-speaker MVP
    max_alternatives: int = 1
    profanity_filter: bool = False
    enable_automatic_punctuation: bool = True

    # Audio encoding (must match AudioConfig)
    sample_rate_hertz: int = 16000
    encoding: str = "LINEAR16"  # Raw PCM

    # Reconnection settings
    enable_reconnect: bool = True
    reconnect_max_attempts: int = 5
    reconnect_base_delay_ms: int = 100
    reconnect_max_delay_ms: int = 2000

    # Timeout settings
    request_timeout_seconds: int = 300
    connection_timeout_seconds: int = 10

    # Stability thresholds for correction
    stability_no_correction: float = 0.3
    stability_lightweight_correction: float = 0.7


@dataclass(frozen=True)
class GPUConfig:
    """GPU/CPU configuration for inference."""
    # Device selection
    primary_device: DeviceType = DeviceType.CPU  # Default to CPU for testing
    fallback_device: Optional[DeviceType] = None

    # CUDA settings
    cuda_device_id: int = 0
    enable_cuda_stream_isolation: bool = True

    # Memory management
    enable_memory_monitoring: bool = True
    memory_warning_threshold_gb: float = 7.0  # Warning at 7GB on 8GB GPU
    clear_cache_between_turns: bool = True

    # Thread pool settings
    thread_pool_workers: int = 3  # One per model: ASR, SLM, TTS

    # OOM recovery
    enable_oom_recovery: bool = True
    oom_retry_count: int = 1


@dataclass(frozen=True)
class StreamingConfig:
    """Configuration for streaming result processing."""
    # Chunk settings
    audio_chunk_size: int = 320  # 20ms at 16kHz (16-bit = 640 bytes)

    # Queue settings
    result_queue_size: int = 100
    audio_queue_size: int = 50

    # Latency monitoring
    enable_latency_tracking: bool = True
    latency_warning_threshold_ms: float = 500.0
    latency_spike_threshold_seconds: float = 5.0

    # Result buffering
    max_result_buffer_seconds: float = 3.0


@dataclass(frozen=True)
class ASRConfig:
    """Complete ASR configuration."""
    personaplex: PersonaPlexConfig = field(default_factory=PersonaPlexConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)


# Default configuration instance
default_asr_config = ASRConfig()


def load_config_from_env():
    """Load configuration from environment variables."""
    from .settings import load_config_from_env as _load
    return _load()
