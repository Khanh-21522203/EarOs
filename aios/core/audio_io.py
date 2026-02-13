"""
Layer 1: Audio I/O Kernel

Captures raw audio from the microphone, delivers PCM frames to the pipeline,
receives synthesized audio from the pipeline, and plays it back through the speaker.

Concurrency boundary:
- Dedicated OS thread for audio callbacks (PortAudio/sounddevice)
- Callback thread writes to lock-free ring buffer
- asyncio task reads from ring buffer and pushes frames into pipeline queue

Audio formats:
- Capture: 16 kHz, 16-bit signed int, mono PCM (640 bytes/frame @ 20ms)
- Playback: 24 kHz, 16-bit signed int, mono PCM (960 bytes/frame @ 20ms)
"""

import asyncio
import logging
import os
import time
import numpy as np
from typing import Optional, Callable

from .ring_buffer import SPSCRingBuffer, create_capture_ring_buffer, create_playback_ring_buffer

logger = logging.getLogger(__name__)

# Frame sizes
CAPTURE_SAMPLE_RATE = 16000
CAPTURE_FRAME_DURATION_MS = 20
CAPTURE_FRAME_SAMPLES = CAPTURE_SAMPLE_RATE * CAPTURE_FRAME_DURATION_MS // 1000  # 320
CAPTURE_FRAME_BYTES = CAPTURE_FRAME_SAMPLES * 2  # 640 bytes (16-bit)

PLAYBACK_SAMPLE_RATE = 24000
PLAYBACK_FRAME_DURATION_MS = 20
PLAYBACK_FRAME_SAMPLES = PLAYBACK_SAMPLE_RATE * PLAYBACK_FRAME_DURATION_MS // 1000  # 480
PLAYBACK_FRAME_BYTES = PLAYBACK_FRAME_SAMPLES * 2  # 960 bytes (16-bit)

# Silence frames (pre-allocated to avoid allocations in callback)
CAPTURE_SILENCE = bytes(CAPTURE_FRAME_BYTES)
PLAYBACK_SILENCE = bytes(PLAYBACK_FRAME_BYTES)


class AudioIOKernel:
    """
    Audio I/O Kernel — Layer 1 of the AIOS pipeline.

    Manages microphone capture and speaker playback using sounddevice
    with lock-free ring buffers for thread-safe communication between
    the audio callback thread and the asyncio event loop.
    """

    def __init__(
        self,
        capture_queue: asyncio.Queue,
        playback_queue: asyncio.Queue,
        capture_device: Optional[int] = None,
        playback_device: Optional[int] = None,
        enable_aec: bool = True,
    ):
        """
        Initialize the Audio I/O Kernel.

        Args:
            capture_queue: Queue to push captured PCM frames into (-> pipeline)
            playback_queue: Queue to read TTS PCM frames from (pipeline ->)
            capture_device: Audio input device index (None = system default)
            playback_device: Audio output device index (None = system default)
            enable_aec: Enable acoustic echo cancellation
        """
        self._capture_queue = capture_queue
        self._playback_queue = playback_queue

        # Device selection (overridable via env vars)
        self._capture_device = capture_device or self._get_env_device("AIOS_CAPTURE_DEVICE")
        self._playback_device = playback_device or self._get_env_device("AIOS_PLAYBACK_DEVICE")

        # Ring buffers (1 second capacity each)
        self._capture_ring = create_capture_ring_buffer(CAPTURE_SAMPLE_RATE, 1.0)
        self._playback_ring = create_playback_ring_buffer(PLAYBACK_SAMPLE_RATE, 1.0)

        # AEC state
        self._enable_aec = enable_aec
        self._aec_processor: Optional[AcousticEchoCanceller] = None

        # Streams
        self._capture_stream = None
        self._playback_stream = None

        # Tasks
        self._capture_task: Optional[asyncio.Task] = None
        self._playback_task: Optional[asyncio.Task] = None
        self._running = False

        # Metrics
        self._capture_frames_produced: int = 0
        self._playback_frames_consumed: int = 0
        self._device_error_count: int = 0
        self._retry_count: int = 0
        self._max_retries: int = 5

    @staticmethod
    def _get_env_device(env_var: str) -> Optional[int]:
        """Get device index from environment variable."""
        val = os.environ.get(env_var)
        if val is not None:
            try:
                return int(val)
            except ValueError:
                logger.warning(f"Invalid device index in {env_var}: {val}")
        return None

    async def start(self):
        """Start audio capture and playback streams."""
        self._running = True

        if self._enable_aec:
            self._aec_processor = AcousticEchoCanceller(
                sample_rate=CAPTURE_SAMPLE_RATE,
                frame_size=CAPTURE_FRAME_SAMPLES,
            )

        try:
            self._open_streams()
        except Exception as e:
            logger.error(f"Failed to open audio streams: {e}")
            raise

        # Start asyncio tasks to bridge ring buffers <-> queues
        self._capture_task = asyncio.create_task(self._capture_pump_loop())
        self._playback_task = asyncio.create_task(self._playback_pump_loop())

        logger.info(
            f"Audio I/O Kernel started "
            f"(capture={self._capture_device or 'default'}, "
            f"playback={self._playback_device or 'default'}, "
            f"aec={self._enable_aec})"
        )

    def _open_streams(self):
        """Open sounddevice input and output streams."""
        try:
            import sounddevice as sd
        except ImportError:
            logger.warning("sounddevice not available — running in mock audio mode")
            return

        try:
            self._capture_stream = sd.RawInputStream(
                samplerate=CAPTURE_SAMPLE_RATE,
                blocksize=CAPTURE_FRAME_SAMPLES,
                dtype="int16",
                channels=1,
                device=self._capture_device,
                callback=self._capture_callback,
            )

            self._playback_stream = sd.RawOutputStream(
                samplerate=PLAYBACK_SAMPLE_RATE,
                blocksize=PLAYBACK_FRAME_SAMPLES,
                dtype="int16",
                channels=1,
                device=self._playback_device,
                callback=self._playback_callback,
            )

            self._capture_stream.start()
            self._playback_stream.start()

        except Exception as e:
            logger.error(f"Error opening audio streams: {e}")
            self._device_error_count += 1
            raise

    def _capture_callback(self, indata, frames, time_info, status):
        """
        PortAudio capture callback — runs in dedicated OS thread.

        MUST NOT allocate memory or block. Writes raw PCM to ring buffer.
        """
        if status:
            self._device_error_count += 1

        # Write raw bytes to capture ring buffer
        data = bytes(indata)
        if not self._capture_ring.write(data):
            # Overflow — ring buffer full, frame dropped
            pass

    def _playback_callback(self, outdata, frames, time_info, status):
        """
        PortAudio playback callback — runs in dedicated OS thread.

        MUST NOT allocate memory or block. Reads from ring buffer.
        """
        if status:
            self._device_error_count += 1

        # Read from playback ring buffer
        data = self._playback_ring.read(len(outdata))
        if data is not None:
            outdata[:] = data
        else:
            # Underrun — insert silence
            outdata[:] = PLAYBACK_SILENCE[:len(outdata)]

    async def _capture_pump_loop(self):
        """
        Asyncio task: reads frames from capture ring buffer,
        applies AEC, and pushes to capture_queue.
        """
        try:
            while self._running:
                frame = self._capture_ring.read(CAPTURE_FRAME_BYTES)
                if frame is not None:
                    # Apply echo cancellation if enabled
                    if self._aec_processor and self._enable_aec:
                        frame = self._aec_processor.process(frame)

                    # Push to pipeline queue (drop-oldest on full)
                    if self._capture_queue.full():
                        try:
                            self._capture_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    try:
                        self._capture_queue.put_nowait(frame)
                        self._capture_frames_produced += 1
                    except asyncio.QueueFull:
                        pass
                else:
                    # No data yet — yield to event loop briefly
                    await asyncio.sleep(0.001)
        except asyncio.CancelledError:
            logger.debug("Capture pump loop cancelled")

    async def _playback_pump_loop(self):
        """
        Asyncio task: reads TTS frames from playback_queue
        and writes to playback ring buffer.
        """
        try:
            while self._running:
                try:
                    frame = await asyncio.wait_for(
                        self._playback_queue.get(),
                        timeout=0.02,
                    )
                    # Write to playback ring buffer
                    if not self._playback_ring.write(frame):
                        # Overflow — TTS generating faster than playback
                        logger.warning("Playback ring buffer overflow")
                    self._playback_frames_consumed += 1
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            logger.debug("Playback pump loop cancelled")

    def feed_reference_signal(self, playback_frame: bytes):
        """
        Feed the far-end reference signal to AEC.

        Called when TTS audio is written to the playback ring buffer,
        so AEC knows what audio the speaker is playing.
        """
        if self._aec_processor:
            self._aec_processor.set_reference(playback_frame)

    def flush_playback(self):
        """Flush the playback ring buffer and queue (for barge-in)."""
        self._playback_ring.clear()
        while not self._playback_queue.empty():
            try:
                self._playback_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        logger.info("Playback flushed")

    async def stop(self):
        """Stop audio streams and cleanup."""
        self._running = False

        if self._capture_task and not self._capture_task.done():
            self._capture_task.cancel()
            try:
                await self._capture_task
            except asyncio.CancelledError:
                pass

        if self._playback_task and not self._playback_task.done():
            self._playback_task.cancel()
            try:
                await self._playback_task
            except asyncio.CancelledError:
                pass

        if self._capture_stream:
            try:
                self._capture_stream.stop()
                self._capture_stream.close()
            except Exception as e:
                logger.error(f"Error closing capture stream: {e}")
            self._capture_stream = None

        if self._playback_stream:
            try:
                self._playback_stream.stop()
                self._playback_stream.close()
            except Exception as e:
                logger.error(f"Error closing playback stream: {e}")
            self._playback_stream = None

        logger.info("Audio I/O Kernel stopped")

    async def handle_device_disconnect(self):
        """
        Handle audio device disconnection.

        Retries device enumeration every 2s, max 5 retries.
        """
        for attempt in range(self._max_retries):
            self._retry_count += 1
            logger.warning(f"Audio device retry {attempt + 1}/{self._max_retries}")
            await asyncio.sleep(2.0)

            try:
                self._open_streams()
                logger.info("Audio device reconnected")
                return
            except Exception:
                continue

        logger.error(f"Audio device reconnection failed after {self._max_retries} retries")

    def get_stats(self) -> dict:
        """Get audio I/O statistics."""
        return {
            "capture_frames_produced": self._capture_frames_produced,
            "playback_frames_consumed": self._playback_frames_consumed,
            "device_error_count": self._device_error_count,
            "capture_ring": self._capture_ring.get_stats(),
            "playback_ring": self._playback_ring.get_stats(),
        }


class AcousticEchoCanceller:
    """
    Software-based Acoustic Echo Cancellation (AEC).

    Receives the near-end signal (mic capture with echo) and the
    far-end reference (playback audio), outputs the echo-cancelled signal.

    Uses a simple spectral subtraction approach as a baseline.
    Can be replaced with speexdsp or WebRTC AEC for production.

    Latency budget: <= 5ms per 20ms frame.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_size: int = 320,
        filter_length_ms: int = 128,
    ):
        self._sample_rate = sample_rate
        self._frame_size = frame_size
        self._filter_length = int(sample_rate * filter_length_ms / 1000)

        # Reference signal buffer (far-end / playback)
        self._reference_buffer = np.zeros(self._filter_length, dtype=np.int16)
        self._has_reference = False

        # Adaptive filter state
        self._filter_coeffs = np.zeros(self._filter_length, dtype=np.float64)
        self._step_size = 0.01

    def set_reference(self, playback_frame: bytes):
        """
        Set the far-end reference signal (what the speaker is playing).

        Args:
            playback_frame: Raw PCM bytes of the playback audio.
        """
        ref_samples = np.frombuffer(playback_frame, dtype=np.int16)
        # Shift buffer and append new reference
        shift = len(ref_samples)
        self._reference_buffer = np.roll(self._reference_buffer, -shift)
        self._reference_buffer[-shift:] = ref_samples[:shift] if shift <= len(ref_samples) else ref_samples
        self._has_reference = True

    def process(self, capture_frame: bytes) -> bytes:
        """
        Process a capture frame through echo cancellation.

        Args:
            capture_frame: Raw PCM bytes from microphone.

        Returns:
            Echo-cancelled PCM bytes.
        """
        if not self._has_reference:
            return capture_frame

        near_end = np.frombuffer(capture_frame, dtype=np.int16).astype(np.float64)

        # Simple spectral subtraction AEC
        # In production, replace with speexdsp or WebRTC AEC
        ref_segment = self._reference_buffer[-self._frame_size:].astype(np.float64)

        # Estimate echo component
        echo_estimate = ref_segment * 0.3  # Simplified echo path estimate

        # Subtract echo
        cancelled = near_end - echo_estimate

        # Clip to int16 range
        cancelled = np.clip(cancelled, -32768, 32767).astype(np.int16)

        return cancelled.tobytes()

    def reset(self):
        """Reset the AEC state."""
        self._reference_buffer[:] = 0
        self._filter_coeffs[:] = 0
        self._has_reference = False
