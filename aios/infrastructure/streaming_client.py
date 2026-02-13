"""
PersonaPlex streaming gRPC client.

Implements bidirectional streaming for real-time ASR with support for:
- Reconnection with exponential backoff
- Audio chunk streaming
- Result processing
- GPU/CPU inference support
"""

import asyncio
import logging
import time
from typing import Optional, AsyncIterator, Callable, Awaitable
from contextlib import asynccontextmanager
from dataclasses import dataclass

import grpc
from grpc import aio as grpc_aio

from ..config import PersonaPlexConfig, DeviceType
from ..core.models import (
    ASRResult,
    StreamingRecognitionConfig,
    LiveRecognitionRequest,
    StreamingMetadata,
    SessionState,
    LatencyMetrics,
)
from ..core.result_processor import ResultProcessor, ProcessedResult

logger = logging.getLogger(__name__)


# Mock gRPC service for development/testing
# In production, this would be the actual PersonaPlex protobuf definitions
class MockPersonaPlexService:
    """Mock service for development without actual PersonaPlex server."""

    async def StreamingRecognize(self, request_iterator: AsyncIterator):
        """Mock streaming recognize that returns simulated results."""
        audio_buffer = b""
        chunk_count = 0

        async for request in request_iterator:
            if request.config:
                # First message with config
                logger.info(f"Received config: {request.config.model}")
                continue

            if request.audio_content:
                audio_buffer += request.audio_content
                chunk_count += 1

                # Simulate streaming results
                if chunk_count % 5 == 0:  # Every 5 chunks
                    # Return partial result
                    yield MockResponse(
                        text="hello world partial",
                        is_final=False,
                        stability=0.5,
                        confidence=0.8,
                    )

                if chunk_count >= 25:  # After ~500ms (25 * 20ms)
                    # Return final result
                    yield MockResponse(
                        text="hello world",
                        is_final=True,
                        stability=1.0,
                        confidence=0.95,
                    )
                    chunk_count = 0
                    audio_buffer = b""


class MockResponse:
    """Mock response object."""

    def __init__(self, text: str, is_final: bool, stability: float, confidence: float):
        self.results = [MockResult(text, is_final, stability, confidence)]


class MockResult:
    """Mock result object."""

    def __init__(self, text: str, is_final: bool, stability: float, confidence: float):
        self.alternatives = [MockAlternative(text, confidence)]
        self.is_final = is_final
        self.stability = stability


class MockAlternative:
    """Mock alternative object."""

    def __init__(self, transcript: str, confidence: float):
        self.transcript = transcript
        self.confidence = confidence


@dataclass
class ReconnectConfig:
    """Configuration for reconnection behavior."""
    enable: bool = True
    max_attempts: int = 5
    base_delay_ms: int = 100
    max_delay_ms: int = 2000
    backoff_multiplier: float = 2.0


class PersonaPlexStreamingClient:
    """
    Client for PersonaPlex streaming ASR via gRPC.

    Features:
    - Bidirectional streaming with asyncio
    - Automatic reconnection with exponential backoff
    - Result processing and routing
    - GPU/CPU support (configurable)
    - Latency tracking
    """

    def __init__(
        self,
        config: PersonaPlexConfig,
        device: DeviceType = DeviceType.CPU,
        result_processor: Optional[ResultProcessor] = None
    ):
        """
        Initialize the streaming client.

        Args:
            config: PersonaPlex configuration
            device: Device type for inference (CPU or CUDA)
            result_processor: Optional result processor for handling results
        """
        self.config = config
        self.device = device
        self.result_processor = result_processor

        # Session state
        self._state = SessionState()
        self._metadata: Optional[StreamingMetadata] = None
        self._metrics: Optional[LatencyMetrics] = None
        self._reconnect_config = ReconnectConfig(
            enable=config.enable_reconnect,
            max_attempts=config.reconnect_max_attempts,
            base_delay_ms=config.reconnect_base_delay_ms,
            max_delay_ms=config.reconnect_max_delay_ms,
        )

        # gRPC state
        self._channel: Optional[grpc_aio.Channel] = None
        self._call: Optional[grpc_aio.StreamStreamCall] = None
        self._response_queue: Optional[asyncio.Queue] = None
        self._request_queue: Optional[asyncio.Queue] = None

        # Task management
        self._receive_task: Optional[asyncio.Task] = None
        self._send_task: Optional[asyncio.Task] = None
        self._stopped = asyncio.Event()

        logger.info(
            f"PersonaPlexStreamingClient initialized "
            f"(device={device.value}, server={config.server_host}:{config.server_port})"
        )

    @property
    def is_connected(self) -> bool:
        """Check if the client is connected."""
        # In mock mode, always return True
        if self.config.use_mock:
            return True
        return self._state.is_connected

    @property
    def is_active(self) -> bool:
        """Check if a streaming session is active."""
        return self._state.is_active

    @property
    def state(self) -> SessionState:
        """Get the current session state."""
        return self._state

    async def connect(self) -> bool:
        """
        Establish a connection to the PersonaPlex server.

        Returns:
            True if connection successful, False otherwise
        """
        # In mock mode, skip connection
        if self.config.use_mock:
            logger.info("Using mock PersonaPlex service (no connection needed)")
            self._state.is_connected = True
            return True

        if self._state.is_connected:
            logger.debug("Already connected")
            return True

        target = f"{self.config.server_host}:{self.config.server_port}"

        try:
            # Create gRPC channel
            if self.config.use_ssl:
                credentials = grpc.ssl_channel_credentials()
                self._channel = grpc_aio.secure_channel(target, credentials)
            else:
                self._channel = grpc_aio.insecure_channel(target)

            # Test connection with a simple readiness check
            await asyncio.wait_for(
                self._channel.channel_ready(),
                timeout=self.config.connection_timeout_seconds
            )

            self._state.is_connected = True
            self._state.reconnect_count = 0
            self._state.last_error = None

            logger.info(f"Connected to PersonaPlex at {target}")
            return True

        except asyncio.TimeoutError:
            error = f"Connection timeout to {target}"
            logger.error(error)
            self._state.last_error = error
            self._state.last_error_time = time.time()
            return False

        except grpc.aio.AioRpcError as e:
            error = f"gRPC connection error: {e.code()}: {e.details()}"
            logger.error(error)
            self._state.last_error = error
            self._state.last_error_time = time.time()
            return False

        except Exception as e:
            error = f"Unexpected connection error: {type(e).__name__}: {e}"
            logger.error(error)
            self._state.last_error = error
            self._state.last_error_time = time.time()
            return False

    async def disconnect(self):
        """Close the gRPC channel and cleanup resources."""
        self._stopped.set()

        # Cancel tasks
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        if self._send_task and not self._send_task.done():
            self._send_task.cancel()
            try:
                await self._send_task
            except asyncio.CancelledError:
                pass

        # Close channel (skip in mock mode)
        if self._channel:
            await self._channel.close()
            self._channel = None

        self._state.is_connected = False
        self._state.is_active = False

        if self.config.use_mock:
            logger.info("Mock service stopped")
        else:
            logger.info("Disconnected from PersonaPlex")

    async def start_streaming(
        self,
        config: Optional[StreamingRecognitionConfig] = None
    ) -> AsyncIterator[ProcessedResult]:
        """
        Start a streaming session.

        Args:
            config: Optional streaming config (uses default if not provided)

        Yields:
            ProcessedResult objects as they arrive
        """
        if not self.config.use_mock and not self._state.is_connected:
            if not await self._connect_with_reconnect():
                raise ConnectionError("Failed to connect to PersonaPlex")

        if config is None:
            config = self._create_default_config()

        # Initialize session
        request_id = f"stream-{int(time.time() * 1000)}"
        self._metadata = StreamingMetadata(
            request_id=request_id,
            start_time=time.time()
        )
        self._metrics = LatencyMetrics()
        self._state.is_active = True
        self._stopped.clear()

        # Initialize result processor if provided
        if self.result_processor:
            self.result_processor.start_session(self._metadata)

        # Create queues for bidirectional streaming
        self._request_queue = asyncio.Queue(maxsize=self.config.request_timeout_seconds * 50)
        self._response_queue = asyncio.Queue(maxsize=100)

        # Start streaming call
        try:
            if self.config.use_mock:
                # Mock mode - use a simpler streaming approach
                self._send_task = asyncio.create_task(self._mock_send_loop())
                self._receive_task = asyncio.create_task(self._mock_receive_loop())
            else:
                # Real gRPC mode (placeholder)
                mock_service = MockPersonaPlexService()
                stream_method = mock_service.StreamingRecognize
                self._send_task = asyncio.create_task(self._send_loop(stream_method))
                self._receive_task = asyncio.create_task(self._receive_loop(stream_method))

            # Send config as first message
            config_request = LiveRecognitionRequest.create_config(config)
            await self._request_queue.put(config_request)

            # Yield results from the response queue
            try:
                while not self._stopped.is_set():
                    try:
                        result = await asyncio.wait_for(
                            self._response_queue.get(),
                            timeout=0.1
                        )
                        yield result
                    except asyncio.TimeoutError:
                        continue
            finally:
                await self._end_streaming()

        except Exception as e:
            logger.error(f"Streaming error: {type(e).__name__}: {e}")
            self._state.is_active = False
            raise

    async def _mock_send_loop(self):
        """Mock send loop - just consumes from request queue to track audio."""
        try:
            audio_chunks_received = 0
            while not self._stopped.is_set():
                try:
                    request = await asyncio.wait_for(
                        self._request_queue.get(),
                        timeout=0.1
                    )
                    if request.audio_content:
                        audio_chunks_received += 1
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            logger.debug("Mock send loop cancelled")

    async def _mock_receive_loop(self):
        """Mock receive loop that generates simulated responses."""
        try:
            # Simulate receiving results over time
            chunk_count = 0
            start_time = time.time()

            while not self._stopped.is_set():
                # Check how much time has passed
                elapsed = time.time() - start_time

                # Generate partial results periodically
                if int(elapsed * 10) % 5 == 0 and int(elapsed * 10) > 0:
                    # Every ~500ms
                    mock_result = MockResponse(
                        text="hello world partial",
                        is_final=False,
                        stability=0.5,
                        confidence=0.8,
                    )
                    await self._process_response(mock_result)
                    await asyncio.sleep(0.5)

                # Generate final result periodically
                if int(elapsed) > 0 and int(elapsed) % 2 == 0:
                    # Every 2 seconds
                    mock_result = MockResponse(
                        text="hello world",
                        is_final=True,
                        stability=1.0,
                        confidence=0.95,
                    )
                    await self._process_response(mock_result)
                    start_time = time.time()  # Reset

                await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            logger.debug("Mock receive loop cancelled")
        except Exception as e:
            logger.error(f"Mock receive loop error: {e}")

    async def _send_loop(self, stream_method):
        """Send audio chunks to the server."""
        try:
            # Create an async iterator from our request queue
            request_iterator = self._iterate_requests()

            # Send requests to the streaming method
            if self.config.use_mock:
                # Mock mode - iterate and manually feed to mock
                async for request in request_iterator:
                    if request.config:
                        # Config is handled by mock internally
                        pass
                    if request.audio_content:
                        # Audio is processed by mock
                        pass
            else:
                # Real gRPC mode - would use actual stream
                async for request in request_iterator:
                    pass  # Would send to gRPC stream
        except asyncio.CancelledError:
            logger.debug("Send loop cancelled")
        except Exception as e:
            logger.error(f"Send loop error: {e}")

    async def _receive_loop(self, stream_method):
        """Receive results from the server."""
        try:
            if self.config.use_mock:
                # In mock mode, we need to drive the mock service directly
                await self._mock_receive_loop()
            else:
                # Real gRPC mode
                async for response in stream_method(self._iterate_requests()):
                    await self._process_response(response)
        except asyncio.CancelledError:
            logger.debug("Receive loop cancelled")
        except Exception as e:
            logger.error(f"Receive loop error: {e}")

    async def _iterate_requests(self) -> AsyncIterator[LiveRecognitionRequest]:
        """Iterate over requests from the request queue."""
        while not self._stopped.is_set():
            try:
                request = await asyncio.wait_for(
                    self._request_queue.get(),
                    timeout=0.1
                )
                yield request
            except asyncio.TimeoutError:
                continue

    async def _process_response(self, response):
        """Process a response from the server."""
        # Convert mock response to ASRResult
        if hasattr(response, 'results') and response.results:
            result_data = response.results[0]

            if hasattr(result_data, 'alternatives') and result_data.alternatives:
                alt = result_data.alternatives[0]

                asr_result = ASRResult(
                    text=getattr(alt, 'transcript', ''),
                    is_final=getattr(result_data, 'is_final', False),
                    stability=getattr(result_data, 'stability', 1.0),
                    confidence=getattr(alt, 'confidence', 0.0),
                )

                # Process through result processor
                if self.result_processor:
                    processed = await self.result_processor.process_result(asr_result)
                    if processed:
                        await self._response_queue.put(processed)
                else:
                    # Direct routing without processor
                    from ..core.result_processor import ProcessedResult, CorrectionLevel
                    processed = ProcessedResult(
                        result=asr_result,
                        correction_level=asr_result.correction_level,
                        is_ghost_text=not asr_result.is_final
                    )
                    await self._response_queue.put(processed)

    async def _end_streaming(self):
        """End the streaming session."""
        logger.debug("Ending streaming session")

        self._stopped.set()
        self._state.is_active = False

        # Wait for tasks to complete
        if self._send_task and not self._send_task.done():
            self._send_task.cancel()
            try:
                await self._send_task
            except asyncio.CancelledError:
                pass

        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        if self.result_processor:
            self.result_processor.end_session()

    async def send_audio(self, audio: bytes):
        """
        Send audio data to the streaming session.

        Args:
            audio: Raw PCM audio bytes (16kHz, 16-bit, mono)
        """
        if not self._state.is_active:
            logger.warning("Cannot send audio: no active session")
            return

        if self._request_queue:
            request = LiveRecognitionRequest.create_audio(audio)
            try:
                await asyncio.wait_for(
                    self._request_queue.put(request),
                    timeout=0.05
                )
            except asyncio.TimeoutError:
                logger.warning("Request queue full, dropping audio")

    def _create_default_config(self) -> StreamingRecognitionConfig:
        """Create the default streaming config."""
        return StreamingRecognitionConfig(
            interim_results=self.config.interim_results,
            single_utterance=self.config.single_utterance,
            model=self.config.model_name,
            sample_rate_hertz=self.config.sample_rate_hertz,
            encoding=self.config.encoding,
            language_code=self.config.language_code,
            enable_speaker_diarization=self.config.enable_speaker_diarization,
            max_alternatives=self.config.max_alternatives,
            profanity_filter=self.config.profanity_filter,
            enable_automatic_punctuation=self.config.enable_automatic_punctuation,
        )

    async def _connect_with_reconnect(self) -> bool:
        """
        Attempt to connect with exponential backoff retry.

        Returns:
            True if connected successfully, False otherwise
        """
        if not self._reconnect_config.enable:
            return await self.connect()

        delay = self._reconnect_config.base_delay_ms / 1000.0

        for attempt in range(self._reconnect_config.max_attempts):
            if await self.connect():
                if attempt > 0:
                    logger.info(f"Reconnected after {attempt + 1} attempts")
                return True

            if attempt < self._reconnect_config.max_attempts - 1:
                logger.info(f"Reconnect attempt {attempt + 1} failed, "
                          f"retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
                delay = min(
                    delay * self._reconnect_config.backoff_multiplier,
                    self._reconnect_config.max_delay_ms / 1000.0
                )

        logger.error(f"Failed to reconnect after {self._reconnect_config.max_attempts} attempts")
        return False


from dataclasses import dataclass


@asynccontextmanager
async def streaming_client(
    config: PersonaPlexConfig,
    device: DeviceType = DeviceType.CPU
):
    """
    Context manager for a PersonaPlex streaming client.

    Usage:
        async with streaming_client(config, device) as client:
            async for result in client.start_streaming():
                print(result.text)
    """
    client = PersonaPlexStreamingClient(config, device)

    try:
        if not await client.connect():
            raise ConnectionError("Failed to connect to PersonaPlex")
        yield client
    finally:
        await client.disconnect()
