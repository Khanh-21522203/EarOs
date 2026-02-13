"""
AIOS - Speech-Native AI Operating System

A speech-native AI operating layer built on NVIDIA PersonaPlex.
Delivers real-time speech-to-speech interaction with conversational continuity.

5-Layer Architecture:
  Layer 1: Audio I/O Kernel (capture, playback, AEC, ring buffers)
  Layer 2: Context & Memory Manager (conversation history, token counting)
  Layer 3: Pipeline Orchestrator (stage routing, queue management, barge-in)
  Layer 4: Context Injection Engine (prompt assembly, hot-word injection)
  Layer 5: State Machine & Event Bus (states, transitions, watchdog)
"""

from .config import (
    ASRConfig,
    AudioConfig,
    PersonaPlexConfig,
    GPUConfig,
    StreamingConfig,
    DeviceType,
    load_config_from_env,
    default_asr_config,
)

from .core.models import (
    ASRResult,
    ResultType,
    CorrectionLevel,
    StreamingRecognitionConfig,
    StreamingMetadata,
    SessionState,
)

from .core.result_processor import (
    ResultProcessor,
    ProcessedResult,
    create_result_processor,
)

from .core.state_machine import (
    StateMachine,
    EventBus,
    Event,
    EventType,
    ConversationState,
    create_state_machine,
)

from .core.ring_buffer import (
    SPSCRingBuffer,
    create_capture_ring_buffer,
    create_playback_ring_buffer,
)

from .core.audio_io import (
    AudioIOKernel,
    AcousticEchoCanceller,
)

from .core.vad import (
    VoiceActivityDetector,
    VADState,
)

from .core.context_manager import (
    ContextManager,
    TurnRecord,
    ContextWindow,
    ContextQuery,
)

from .core.context_injection import (
    ContextInjectionEngine,
    AssembledPrompt,
    HotWord,
)

from .core.streaming_correction import (
    StreamingCorrectionEngine,
    CorrectedText,
    RuleBasedCorrector,
)

from .core.pipeline import (
    PipelineOrchestrator,
    PipelineQueues,
    create_pipeline,
)

from .infrastructure.streaming_client import (
    PersonaPlexStreamingClient,
    streaming_client,
)

from .infrastructure.gpu_manager import (
    GPUManager,
    MemoryState,
    create_gpu_manager,
)

from .infrastructure.asr_service import (
    ASRService,
    create_asr_service,
    transcribe_stream,
)

__version__ = "0.1.0-mvp"
__author__ = "AIOS Team"

__all__ = [
    # Config
    "ASRConfig",
    "AudioConfig",
    "PersonaPlexConfig",
    "GPUConfig",
    "StreamingConfig",
    "DeviceType",
    "load_config_from_env",
    "default_asr_config",

    # Core models
    "ASRResult",
    "ResultType",
    "CorrectionLevel",
    "StreamingRecognitionConfig",
    "StreamingMetadata",
    "SessionState",

    # Result processor
    "ResultProcessor",
    "ProcessedResult",
    "create_result_processor",

    # Layer 5: State Machine & Event Bus
    "StateMachine",
    "EventBus",
    "Event",
    "EventType",
    "ConversationState",
    "create_state_machine",

    # Layer 1: Audio I/O Kernel
    "SPSCRingBuffer",
    "create_capture_ring_buffer",
    "create_playback_ring_buffer",
    "AudioIOKernel",
    "AcousticEchoCanceller",
    "VoiceActivityDetector",
    "VADState",

    # Layer 2: Context & Memory Manager
    "ContextManager",
    "TurnRecord",
    "ContextWindow",
    "ContextQuery",

    # Layer 4: Context Injection Engine
    "ContextInjectionEngine",
    "AssembledPrompt",
    "HotWord",

    # Streaming Correction
    "StreamingCorrectionEngine",
    "CorrectedText",
    "RuleBasedCorrector",

    # Layer 3: Pipeline Orchestrator
    "PipelineOrchestrator",
    "PipelineQueues",
    "create_pipeline",

    # Infrastructure
    "PersonaPlexStreamingClient",
    "streaming_client",
    "GPUManager",
    "MemoryState",
    "create_gpu_manager",
    "ASRService",
    "create_asr_service",
    "transcribe_stream",
]
