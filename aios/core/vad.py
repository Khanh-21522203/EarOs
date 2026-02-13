"""
Voice Activity Detection (VAD) for AIOS.

Uses Silero VAD (ONNX, CPU-only) for speech/non-speech classification.
Runs on the asyncio event loop (~3ms per frame on CPU).

Publishes speech_start and speech_end events to the event bus.
Forwards speech-classified frames to the ASR pipeline.
"""

import asyncio
import logging
import time
import numpy as np
from typing import Optional
from enum import Enum

logger = logging.getLogger(__name__)

# VAD parameters
VAD_SAMPLE_RATE = 16000
VAD_FRAME_DURATION_MS = 20
VAD_FRAME_SAMPLES = VAD_SAMPLE_RATE * VAD_FRAME_DURATION_MS // 1000  # 320


class VADState(Enum):
    """VAD internal state."""
    SILENCE = "silence"
    SPEECH = "speech"
    SPEECH_ENDING = "speech_ending"  # In silence cooldown after speech


class VoiceActivityDetector:
    """
    Voice Activity Detection using energy-based detection with optional
    Silero VAD model integration.

    Classifies 20ms PCM frames as speech or non-speech.
    Publishes speech_start/speech_end events to the event bus.
    Forwards speech frames to the speech_queue for ASR.

    Performance: ~3ms per frame on CPU.
    """

    def __init__(
        self,
        speech_queue: asyncio.Queue,
        event_bus=None,
        speech_threshold: float = 0.5,
        silence_duration_ms: int = 500,
        min_speech_duration_ms: int = 100,
        use_silero: bool = False,
    ):
        """
        Initialize the VAD.

        Args:
            speech_queue: Queue to forward speech frames to (-> ASR).
            event_bus: Event bus for publishing speech_start/speech_end events.
            speech_threshold: Threshold for speech detection (0.0-1.0).
            silence_duration_ms: Duration of silence before speech_end (ms).
            min_speech_duration_ms: Minimum speech duration to trigger speech_start.
            use_silero: Use Silero VAD model (requires onnxruntime).
        """
        self._speech_queue = speech_queue
        self._event_bus = event_bus

        # Thresholds
        self._speech_threshold = speech_threshold
        self._silence_frames = silence_duration_ms // VAD_FRAME_DURATION_MS
        self._min_speech_frames = min_speech_duration_ms // VAD_FRAME_DURATION_MS

        # State
        self._state = VADState.SILENCE
        self._speech_frame_count: int = 0
        self._silence_frame_count: int = 0
        self._speech_start_time: float = 0.0

        # Silero model
        self._use_silero = use_silero
        self._silero_model = None
        self._silero_h = None
        self._silero_c = None

        # Energy-based fallback parameters
        self._energy_threshold: float = 500.0
        self._energy_smoothing: float = 0.95
        self._smoothed_energy: float = 0.0

        # Metrics
        self._total_frames: int = 0
        self._speech_frames: int = 0
        self._total_speech_events: int = 0

        if use_silero:
            self._load_silero_model()

    def _load_silero_model(self):
        """Load Silero VAD ONNX model."""
        try:
            import torch
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=True,
            )
            self._silero_model = model
            logger.info("Silero VAD model loaded (ONNX)")
        except Exception as e:
            logger.warning(f"Failed to load Silero VAD: {e}. Using energy-based VAD.")
            self._use_silero = False

    def process_frame(self, frame: bytes) -> tuple[bool, float]:
        """
        Process a single 20ms PCM frame.

        Args:
            frame: Raw PCM bytes (16kHz, 16-bit, mono, 640 bytes).

        Returns:
            Tuple of (is_speech, confidence).
        """
        self._total_frames += 1

        if self._use_silero and self._silero_model is not None:
            return self._process_silero(frame)
        else:
            return self._process_energy(frame)

    def _process_energy(self, frame: bytes) -> tuple[bool, float]:
        """Energy-based VAD (fallback when Silero is unavailable)."""
        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
        energy = np.sqrt(np.mean(samples ** 2))

        # Exponential smoothing
        self._smoothed_energy = (
            self._energy_smoothing * self._smoothed_energy +
            (1 - self._energy_smoothing) * energy
        )

        # Normalize to 0-1 range (approximate)
        confidence = min(1.0, self._smoothed_energy / (self._energy_threshold * 2))
        is_speech = self._smoothed_energy > self._energy_threshold

        return is_speech, confidence

    def _process_silero(self, frame: bytes) -> tuple[bool, float]:
        """Silero VAD model inference."""
        try:
            import torch

            samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
            tensor = torch.FloatTensor(samples)

            confidence = self._silero_model(tensor, VAD_SAMPLE_RATE).item()
            is_speech = confidence > self._speech_threshold

            return is_speech, confidence
        except Exception as e:
            logger.error(f"Silero VAD error: {e}")
            return self._process_energy(frame)

    async def process_frame_async(self, frame: bytes, turn_id: int = 0) -> bool:
        """
        Process a frame and handle state transitions.

        Args:
            frame: Raw PCM bytes.
            turn_id: Current turn ID for event tagging.

        Returns:
            True if frame is speech and was forwarded.
        """
        is_speech, confidence = self.process_frame(frame)

        if is_speech:
            self._speech_frame_count += 1
            self._silence_frame_count = 0
            self._speech_frames += 1

            if self._state == VADState.SILENCE:
                if self._speech_frame_count >= self._min_speech_frames:
                    # Transition to SPEECH
                    self._state = VADState.SPEECH
                    self._speech_start_time = time.time()
                    self._total_speech_events += 1
                    await self._publish_speech_start(turn_id)

            elif self._state == VADState.SPEECH_ENDING:
                # Speech resumed during cooldown
                self._state = VADState.SPEECH

            # Forward speech frame to ASR
            if self._state == VADState.SPEECH:
                await self._forward_frame(frame)
                return True

        else:
            self._silence_frame_count += 1

            if self._state == VADState.SPEECH:
                self._state = VADState.SPEECH_ENDING

            elif self._state == VADState.SPEECH_ENDING:
                if self._silence_frame_count >= self._silence_frames:
                    # Transition to SILENCE
                    self._state = VADState.SILENCE
                    self._speech_frame_count = 0
                    await self._publish_speech_end(turn_id)

        return False

    async def _forward_frame(self, frame: bytes):
        """Forward a speech frame to the speech queue."""
        if self._speech_queue.full():
            # Drop oldest (real-time constraint)
            try:
                self._speech_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            self._speech_queue.put_nowait(frame)
        except asyncio.QueueFull:
            pass

    async def _publish_speech_start(self, turn_id: int):
        """Publish speech_start event to the event bus."""
        logger.info("VAD: speech_start detected")
        if self._event_bus:
            from .state_machine import Event, EventType
            event = Event(
                type=EventType.SPEECH_START,
                turn_id=turn_id,
            )
            await self._event_bus.publish(event)

    async def _publish_speech_end(self, turn_id: int):
        """Publish speech_end event to the event bus."""
        duration_ms = (time.time() - self._speech_start_time) * 1000
        logger.info(f"VAD: speech_end detected (duration={duration_ms:.0f}ms)")
        if self._event_bus:
            from .state_machine import Event, EventType
            event = Event(
                type=EventType.SPEECH_END,
                turn_id=turn_id,
                data={"speech_duration_ms": duration_ms},
            )
            await self._event_bus.publish(event)

    def reset(self):
        """Reset VAD state (e.g., on turn boundary)."""
        self._state = VADState.SILENCE
        self._speech_frame_count = 0
        self._silence_frame_count = 0

    def get_stats(self) -> dict:
        """Get VAD statistics."""
        return {
            "state": self._state.value,
            "total_frames": self._total_frames,
            "speech_frames": self._speech_frames,
            "speech_ratio": self._speech_frames / max(1, self._total_frames),
            "total_speech_events": self._total_speech_events,
            "using_silero": self._use_silero,
        }
