"""
ASR Service - Main entry point for PersonaPlex streaming ASR.

Integrates the streaming client, result processor, and GPU manager
to provide a complete ASR solution for AIOS.
"""

import asyncio
import logging
from typing import Optional, AsyncIterator, Callable, Awaitable
from dataclasses import dataclass

from ..config import ASRConfig, DeviceType
from ..core.models import StreamingRecognitionConfig
from ..core.result_processor import ResultProcessor, ProcessedResult, create_result_processor
from .streaming_client import PersonaPlexStreamingClient, streaming_client
from .gpu_manager import GPUManager, create_gpu_manager

logger = logging.getLogger(__name__)


@dataclass
class ASRServiceConfig:
    """Configuration for the ASR service."""
    # Core config
    config: ASRConfig

    # Callbacks
    on_partial: Optional[Callable[[ProcessedResult], Awaitable[None]]] = None
    on_final: Optional[Callable[[ProcessedResult], Awaitable[None]]] = None

    # Enable/disable features
    enable_result_processor: bool = True
    enable_gpu_manager: bool = True
    enable_latency_tracking: bool = True


class ASRService:
    """
    Main ASR service for AIOS.

    Provides a high-level interface for streaming ASR with:
    - Automatic connection management
    - Result processing with stability-based correction routing
    - GPU resource management with OOM recovery
    - Latency tracking

    Usage:
        service = ASRService(config)

        async with service.session():
            async for result in service.results():
                print(f"Result: {result.text}")
    """

    def __init__(self, config: ASRConfig):
        """
        Initialize the ASR service.

        Args:
            config: Complete ASR configuration
        """
        self.config = config
        self.plex_config = config.personaplex
        self.audio_config = config.audio
        self.gpu_config = config.gpu
        self.streaming_config = config.streaming

        # Components
        self._gpu_manager: Optional[GPUManager] = None
        self._result_processor: Optional[ResultProcessor] = None
        self._client: Optional[PersonaPlexStreamingClient] = None
        self._result_queue: Optional[asyncio.Queue[ProcessedResult]] = None

        # Session state
        self._is_running: bool = False
        self._is_streaming: bool = False

        logger.info("ASRService initialized")

    async def initialize(self):
        """
        Initialize the ASR service components.

        Must be called before starting a session.
        """
        # Initialize GPU manager
        if self.gpu_config.enable_memory_monitoring:
            self._gpu_manager = create_gpu_manager(self.gpu_config)
            await self._gpu_manager.update_memory_stats()

        # Initialize result processor
        self._result_processor, self._result_queue = create_result_processor(
            queue_size=self.streaming_config.result_queue_size,
            enable_latency_tracking=self.streaming_config.enable_latency_tracking
        )

        # Initialize streaming client
        device = self.gpu_config.primary_device
        if self._gpu_manager:
            device = DeviceType.CUDA if self._gpu_manager.should_use_cuda() else DeviceType.CPU

        self._client = PersonaPlexStreamingClient(
            config=self.plex_config,
            device=device,
            result_processor=self._result_processor
        )

        self._is_running = True
        logger.info("ASRService initialized components")

    async def shutdown(self):
        """Shutdown the ASR service and cleanup resources."""
        self._is_running = False

        if self._client:
            await self._client.disconnect()
            self._client = None

        self._result_processor = None
        self._result_queue = None
        self._gpu_manager = None

        logger.info("ASRService shut down")

    async def connect(self) -> bool:
        """
        Connect to the PersonaPlex server.

        Returns:
            True if connected successfully, False otherwise
        """
        if not self._client:
            await self.initialize()

        return await self._client.connect()

    async def disconnect(self):
        """Disconnect from the PersonaPlex server."""
        if self._client:
            await self._client.disconnect()

    def set_callbacks(
        self,
        on_partial: Optional[Callable[[ProcessedResult], Awaitable[None]]] = None,
        on_final: Optional[Callable[[ProcessedResult], Awaitable[None]]] = None
    ):
        """Set callbacks for partial and final results."""
        if self._result_processor:
            self._result_processor.set_callbacks(
                on_partial=on_partial,
                on_final=on_final
            )

    async def session(
        self,
        streaming_config: Optional[StreamingRecognitionConfig] = None
    ) -> AsyncIterator[ProcessedResult]:
        """
        Context manager for a streaming session.

        Args:
            streaming_config: Optional streaming config

        Yields:
            ProcessedResult objects as they arrive

        Usage:
            async with service.session() as results:
                async for result in results:
                    print(result.text)
        """
        if not self._is_running:
            await self.initialize()

        self._is_streaming = True

        try:
            async for result in self._client.start_streaming(streaming_config):
                if not self._is_streaming:
                    break
                yield result
        finally:
            self._is_streaming = False

    def results(self) -> AsyncIterator[ProcessedResult]:
        """
        Get an async iterator of results from the result queue.

        This is an alternative to session() for more fine-grained control.

        Usage:
            async for result in service.results():
                print(result.text)
        """
        if not self._result_queue:
            raise RuntimeError("Service not initialized. Call initialize() first.")

        return self._iterate_results()

    async def _iterate_results(self) -> AsyncIterator[ProcessedResult]:
        """Internal iterator for results."""
        while self._is_streaming:
            try:
                result = await asyncio.wait_for(
                    self._result_queue.get(),
                    timeout=0.1
                )
                yield result
            except asyncio.TimeoutError:
                continue

    async def send_audio(self, audio: bytes):
        """
        Send audio data to the active streaming session.

        Args:
            audio: Raw PCM audio bytes (16kHz, 16-bit, mono)
        """
        if not self._client:
            raise RuntimeError("Client not initialized. Call initialize() first.")

        # Validate audio size
        expected_bytes = self.audio_config.frame_samples * 2  # 16-bit = 2 bytes
        if len(audio) != expected_bytes:
            logger.warning(
                f"Audio chunk size mismatch: expected {expected_bytes}, got {len(audio)}"
            )

        await self._client.send_audio(audio)

    async def send_audio_frames(self, frames: list[bytes]):
        """
        Send multiple audio frames to the active streaming session.

        Args:
            frames: List of raw PCM audio bytes
        """
        for frame in frames:
            await self.send_audio(frame)

    def get_current_display_text(self) -> str:
        """
        Get the current display text from the result processor.

        Returns the most recent final text, or partial text if no final available.
        """
        if self._result_processor:
            return self._result_processor.get_current_display_text()
        return ""

    @property
    def is_connected(self) -> bool:
        """Check if connected to PersonaPlex."""
        return self._client.is_connected if self._client else False

    @property
    def is_streaming(self) -> bool:
        """Check if a streaming session is active."""
        return self._is_streaming

    @property
    def gpu_manager(self) -> Optional[GPUManager]:
        """Get the GPU manager."""
        return self._gpu_manager

    @property
    def latency_metrics(self):
        """Get latency metrics from the result processor."""
        if self._result_processor:
            return self._result_processor.metrics
        return None

    def get_status(self) -> dict:
        """Get the current status of the ASR service."""
        return {
            "is_running": self._is_running,
            "is_connected": self.is_connected,
            "is_streaming": self._is_streaming,
            "has_gpu_manager": self._gpu_manager is not None,
            "device": self.gpu_config.primary_device.value,
            "vram_summary": self._gpu_manager.get_vram_usage_summary() if self._gpu_manager else None,
        }


# Factory function
def create_asr_service(config: ASRConfig) -> ASRService:
    """
    Factory function to create an ASR service.

    Args:
        config: Complete ASR configuration

    Returns:
        Configured ASRService instance
    """
    return ASRService(config)


# Convenience function for quick usage
async def transcribe_stream(
    audio_iterator: AsyncIterator[bytes],
    config: Optional[ASRConfig] = None
) -> AsyncIterator[ProcessedResult]:
    """
    Convenience function to transcribe an audio stream.

    Args:
        audio_iterator: Async iterator of audio chunks (20ms, 16kHz, 16-bit, mono)
        config: Optional ASR configuration

    Yields:
        ProcessedResult objects
    """
    if config is None:
        from ..config import default_asr_config
        config = default_asr_config

    service = create_asr_service(config)
    await service.initialize()

    async with service.session():
        async for audio_chunk in audio_iterator:
            await service.send_audio(audio_chunk)

        # Yield results from queue
        async for result in service.results():
            yield result

    await service.shutdown()
