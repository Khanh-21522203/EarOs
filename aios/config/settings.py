"""
Environment-based configuration loader.

Loads configuration from environment variables with defaults from config module.
"""

import os
from dataclasses import replace
from . import (
    ASRConfig,
    AudioConfig,
    PersonaPlexConfig,
    GPUConfig,
    StreamingConfig,
    DeviceType
)


def load_config_from_env() -> ASRConfig:
    """
    Load ASR configuration from environment variables.

    Environment variables:
        PERSONAPLEX_HOST        - PersonaPlex server host (default: localhost)
        PERSONAPLEX_PORT        - PersonaPlex server port (default: 50051)
        DEVICE                  - Primary device: 'cpu' or 'cuda' (default: cpu)
        CUDA_DEVICE_ID          - CUDA device ID (default: 0)
        CAPTURE_SAMPLE_RATE     - Audio capture sample rate (default: 16000)
        FRAME_DURATION_MS       - Audio frame duration in ms (default: 20)
    """
    # Audio configuration
    audio = AudioConfig(
        capture_sample_rate=int(os.getenv("CAPTURE_SAMPLE_RATE", 16000)),
        frame_duration_ms=int(os.getenv("FRAME_DURATION_MS", 20)),
    )

    # PersonaPlex configuration
    personaplex = PersonaPlexConfig(
        server_host=os.getenv("PERSONAPLEX_HOST", "localhost"),
        server_port=int(os.getenv("PERSONAPLEX_PORT", 50051)),
        model_name=os.getenv("PERSONAPLEX_MODEL", "personaplex-en-us"),
        language_code=os.getenv("PERSONAPLEX_LANGUAGE", "en-US"),
        reconnect_max_attempts=int(os.getenv("RECONNECT_MAX_ATTEMPTS", 5)),
    )

    # GPU configuration
    device_str = os.getenv("DEVICE", "cpu").lower()
    primary_device = DeviceType.CUDA if device_str == "cuda" else DeviceType.CPU

    gpu = GPUConfig(
        primary_device=primary_device,
        cuda_device_id=int(os.getenv("CUDA_DEVICE_ID", 0)),
        thread_pool_workers=int(os.getenv("THREAD_POOL_WORKERS", 3)),
    )

    # Streaming configuration
    streaming = StreamingConfig(
        audio_queue_size=int(os.getenv("AUDIO_QUEUE_SIZE", 50)),
        result_queue_size=int(os.getenv("RESULT_QUEUE_SIZE", 100)),
    )

    return ASRConfig(
        audio=audio,
        personaplex=personaplex,
        gpu=gpu,
        streaming=streaming,
    )


# Convenience: load default config
current_config = load_config_from_env()
