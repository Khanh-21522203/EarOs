#!/usr/bin/env python3
"""
Test script for PersonaPlex ASR integration.

This script demonstrates the usage of the AIOS ASR service
with simulated audio input (mock mode).
"""

import asyncio
import logging
import numpy as np
from pathlib import Path

# Add the project root to the path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from aios import (
    ASRService,
    create_asr_service,
    ASRConfig,
    DeviceType,
    ProcessedResult,
    PersonaPlexConfig,
    GPUConfig,
    StreamingConfig,
    AudioConfig,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_silence_frame(
    sample_rate: int = 16000,
    duration_ms: int = 20,
    bit_depth: int = 16
) -> bytes:
    """Generate a silent audio frame (all zeros)."""
    num_samples = sample_rate * duration_ms // 1000
    # 16-bit signed integer = 2 bytes per sample
    return bytes(num_samples * (bit_depth // 8))


def generate_tone_frame(
    frequency: int = 440,
    sample_rate: int = 16000,
    duration_ms: int = 20,
    volume: float = 0.5
) -> bytes:
    """Generate an audio frame with a sine wave tone."""
    num_samples = sample_rate * duration_ms // 1000
    t = np.linspace(0, duration_ms / 1000.0, num_samples, False)
    wave = np.sin(2 * np.pi * frequency * t) * volume

    # Convert to 16-bit signed integer
    wave_int16 = (wave * 32767).astype(np.int16)
    return wave_int16.tobytes()


async def test_mock_streaming():
    """Test the ASR service with mock PersonaPlex server."""
    logger.info("Starting mock streaming test")

    # Create config with CPU device (for testing without GPU) and mock mode
    config = ASRConfig(
        personaplex=PersonaPlexConfig(
            use_mock=True,  # Enable mock mode
        ),
        audio=AudioConfig(),
        gpu=GPUConfig(
            primary_device=DeviceType.CPU,
            enable_memory_monitoring=False,
        ),
        streaming=StreamingConfig(),
    )

    # Create service
    service = create_asr_service(config)

    # Setup callbacks
    async def on_partial(result: ProcessedResult):
        logger.info(
            f"[PARTIAL] Ghost text: '{result.text}' "
            f"(correction: {result.correction_level.value})"
        )

    async def on_final(result: ProcessedResult):
        logger.info(
            f"[FINAL] Solid text: '{result.text}' "
            f"(confidence: {result.result.confidence:.2f})"
        )

    service.set_callbacks(on_partial=on_partial, on_final=on_final)

    # Initialize service
    await service.initialize()

    # Check status
    status = service.get_status()
    logger.info(f"Service status: {status}")

    # Start streaming session and collect results
    logger.info("Starting streaming session...")

    results_received = []

    # Create audio sender task
    async def send_audio_frames():
        """Send simulated audio frames."""
        await asyncio.sleep(0.1)  # Wait for session to start

        logger.info("Sending audio frames...")

        # Send 2 seconds of "audio" (100 frames * 20ms = 2 seconds)
        for i in range(100):
            frame = generate_silence_frame()
            await service.send_audio(frame)

            # Small delay to simulate real-time audio
            await asyncio.sleep(0.015)

        logger.info("Finished sending audio frames")

    # Start streaming session with timeout
    streaming_task = asyncio.create_task(service.session().__anext__())

    # Wait a moment then start sending audio
    await asyncio.sleep(0.1)
    audio_task = asyncio.create_task(send_audio_frames())

    # Collect results for a limited time
    try:
        # Since session() is an async iterator that never ends in mock mode,
        # we'll wait for results from callbacks
        await asyncio.wait_for(audio_task, timeout=5.0)
        logger.info("Audio sending complete")

        # Wait a bit more for any final results
        await asyncio.sleep(1.0)

    except asyncio.TimeoutError:
        logger.warning("Audio sending timed out")
    finally:
        # Stop streaming
        service._is_streaming = False
        streaming_task.cancel()
        audio_task.cancel()

        try:
            await streaming_task
        except asyncio.CancelledError:
            pass

        try:
            await audio_task
        except asyncio.CancelledError:
            pass

    # Print final status
    logger.info("Session ended")
    logger.info(f"Final display text: '{service.get_current_display_text()}'")

    # Shutdown
    await service.shutdown()
    logger.info("Service shut down")


async def test_config_from_env():
    """Test loading configuration from environment variables."""
    from aios.config import load_config_from_env

    config = load_config_from_env()

    logger.info("Configuration loaded from environment:")
    logger.info(f"  Device: {config.gpu.primary_device.value}")
    logger.info(f"  PersonaPlex server: {config.personaplex.server_host}:{config.personaplex.server_port}")
    logger.info(f"  Sample rate: {config.audio.capture_sample_rate} Hz")
    logger.info(f"  Frame size: {config.audio.frame_samples} samples")


async def test_gpu_manager():
    """Test GPU manager functionality."""
    from aios.infrastructure.gpu_manager import GPUManager, create_gpu_manager
    from aios.config import GPUConfig, DeviceType

    logger.info("Testing GPU manager")

    # Test CPU mode
    cpu_config = GPUConfig(primary_device=DeviceType.CPU)
    cpu_manager = create_gpu_manager(cpu_config)

    stats = await cpu_manager.update_memory_stats()
    logger.info(f"CPU mode - Device: {cpu_manager.current_device.value}")
    logger.info(f"CPU mode - State: {stats.state.value}")

    # Test CUDA mode (will fall back to CPU if no CUDA)
    cuda_config = GPUConfig(primary_device=DeviceType.CUDA)
    cuda_manager = create_gpu_manager(cuda_config)

    stats = await cuda_manager.update_memory_stats()
    logger.info(f"CUDA mode - Available: {cuda_manager.is_cuda_available}")
    logger.info(f"CUDA mode - Device: {cuda_manager.current_device.value}")
    logger.info(f"CUDA mode - State: {stats.state.value}")

    if cuda_manager.is_cuda_available:
        logger.info(cuda_manager.get_vram_usage_summary())


async def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("AIOS PersonaPlex Integration Tests")
    logger.info("=" * 60)

    # Test 1: Config loading
    logger.info("\n[Test 1] Configuration Loading")
    await test_config_from_env()

    # Test 2: GPU Manager
    logger.info("\n[Test 2] GPU Manager")
    await test_gpu_manager()

    # Test 3: Mock Streaming
    logger.info("\n[Test 3] Mock Streaming")
    await test_mock_streaming()

    logger.info("\n" + "=" * 60)
    logger.info("All tests completed")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
