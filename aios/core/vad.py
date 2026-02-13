"""
Voice Activity Detection (VAD) for AIOS.

Pipeline (from voice-activity-detection.md):
  Echo-Cancelled Audio (20ms frames)
    -> Energy Gate (< 0.1ms, reject frames below -50 dBFS)
    -> Silero VAD (~1.5ms, neural classification)
    -> Temporal Smoother (asymmetric debounce)
    -> Event Bus (speech_start / speech_end / barge_in)

Asymmetric debounce thresholds:
  - speech_start: 3 consecutive speech frames (60ms)
  - speech_end:  15 consecutive silence frames (300ms)
  - barge_in:     5 consecutive speech frames during SPEAKING (100ms)

Pre-roll buffer: 200ms (10 frames) sent retroactively on speech_start.

Startup calibration: first 2s measures ambient noise floor for energy gate.
"""

import asyncio
import logging
import math
import time
import numpy as np
from collections import deque
from typing import Optional, List
from enum import Enum

logger = logging.getLogger(__name__)

# VAD parameters
VAD_SAMPLE_RATE = 16000
VAD_FRAME_DURATION_MS = 20
VAD_FRAME_SAMPLES = VAD_SAMPLE_RATE * VAD_FRAME_DURATION_MS // 1000  # 320

# Asymmetric debounce thresholds (frames)
ONSET_FRAMES = 3          # 60ms — speech_start
OFFSET_FRAMES = 15         # 300ms — speech_end
BARGE_IN_FRAMES = 5        # 100ms — barge_in during SPEAKING

# Pre-roll buffer size (frames)
PRE_ROLL_FRAMES = 10       # 200ms

# Energy gate defaults
ENERGY_GATE_DEFAULT_DBFS = -50.0
ENERGY_GATE_MARGIN_DB = 10.0

# Startup calibration duration
CALIBRATION_FRAMES = 100   # 2s at 20ms/frame


class VADState(Enum):
    """VAD internal state."""
    SILENCE = "silence"
    SPEECH = "speech"
    SPEECH_ENDING = "speech_ending"  # In silence cooldown after speech


class VoiceActivityDetector:
    """
    Voice Activity Detection with energy gate, Silero VAD, and temporal smoother.

    Classifies 20ms PCM frames as speech or non-speech.
    Publishes speech_start/speech_end/barge_in events to the event bus.
    Forwards speech frames to the speech_queue for ASR.

    Performance: <= 3ms per frame on CPU.
    """

    def __init__(
        self,
        speech_queue: asyncio.Queue,
        event_bus=None,
        speech_threshold: float = 0.5,
        use_silero: bool = False,
        barge_in_mode: bool = False,
    ):
        self._speech_queue = speech_queue
        self._event_bus = event_bus

        # Thresholds
        self._speech_threshold = speech_threshold
        self._barge_in_mode = barge_in_mode

        # State
        self._state = VADState.SILENCE
        self._consecutive_speech: int = 0
        self._consecutive_silence: int = 0
        self._speech_start_time: float = 0.0

        # Pre-roll buffer (circular, last 10 frames = 200ms)
        self._pre_roll: deque[bytes] = deque(maxlen=PRE_ROLL_FRAMES)

        # Energy gate
        self._energy_gate_dbfs: float = ENERGY_GATE_DEFAULT_DBFS
        self._calibrating: bool = True
        self._calibration_energies: List[float] = []

        # Silero model
        self._use_silero = use_silero
        self._silero_model = None

        # Energy-based fallback parameters
        self._energy_threshold: float = 500.0
        self._energy_smoothing: float = 0.95
        self._smoothed_energy: float = 0.0

        # Metrics
        self._total_frames: int = 0
        self._speech_frames: int = 0
        self._total_speech_events: int = 0
        self._total_barge_in_events: int = 0
        self._energy_gate_rejects: int = 0
        self._false_trigger_count: int = 0

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

    def set_barge_in_mode(self, enabled: bool):
        """Enable/disable barge-in mode (during SPEAKING state)."""
        self._barge_in_mode = enabled

    def _compute_energy_dbfs(self, samples: np.ndarray) -> float:
        """Compute RMS energy in dBFS for a frame."""
        rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2))
        if rms < 1.0:
            return -96.0  # Silence floor
        return 20.0 * math.log10(rms / 32768.0)

    def _energy_gate(self, frame: bytes) -> tuple[bool, float]:
        """
        Energy gate pre-filter. Rejects frames below the noise floor.

        Returns:
            (passes_gate, energy_dbfs)
        """
        samples = np.frombuffer(frame, dtype=np.int16)
        energy_dbfs = self._compute_energy_dbfs(samples)

        # Startup calibration: measure ambient noise floor for first 2s
        if self._calibrating:
            self._calibration_energies.append(energy_dbfs)
            if len(self._calibration_energies) >= CALIBRATION_FRAMES:
                ambient_floor = np.median(self._calibration_energies)
                self._energy_gate_dbfs = ambient_floor + ENERGY_GATE_MARGIN_DB
                self._calibrating = False
                logger.info(
                    "VAD energy gate calibrated: ambient=%.1f dBFS, threshold=%.1f dBFS",
                    ambient_floor, self._energy_gate_dbfs,
                )

        passes = energy_dbfs > self._energy_gate_dbfs
        if not passes:
            self._energy_gate_rejects += 1
        return passes, energy_dbfs

    def process_frame(self, frame: bytes) -> tuple[bool, float]:
        """
        Process a single 20ms PCM frame through energy gate + VAD.

        Returns:
            Tuple of (is_speech, confidence).
        """
        self._total_frames += 1

        # Always store in pre-roll buffer
        self._pre_roll.append(frame)

        # Stage 1: Energy gate (< 0.1ms)
        passes_gate, energy_dbfs = self._energy_gate(frame)
        if not passes_gate:
            return False, 0.0

        # Stage 2: VAD classification
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
        Process a frame with temporal smoothing and state transitions.

        Uses asymmetric debounce:
        - Onset: 3 frames (60ms) for speech_start
        - Offset: 15 frames (300ms) for speech_end
        - Barge-in: 5 frames (100ms) during SPEAKING state

        Returns:
            True if frame is speech and was forwarded.
        """
        is_speech, confidence = self.process_frame(frame)

        # Temporal smoother (Stage 3)
        if is_speech:
            self._consecutive_speech += 1
            self._consecutive_silence = 0
            self._speech_frames += 1

            if self._state == VADState.SILENCE:
                onset_threshold = BARGE_IN_FRAMES if self._barge_in_mode else ONSET_FRAMES
                if self._consecutive_speech >= onset_threshold:
                    self._state = VADState.SPEECH
                    self._speech_start_time = time.monotonic()
                    self._total_speech_events += 1

                    if self._barge_in_mode:
                        self._total_barge_in_events += 1
                        await self._publish_barge_in(turn_id)
                    else:
                        await self._publish_speech_start(turn_id)
                        # Send pre-roll buffer (200ms retroactive)
                        await self._send_pre_roll()

            elif self._state == VADState.SPEECH_ENDING:
                # Speech resumed during cooldown
                self._state = VADState.SPEECH

            # Forward speech frame to ASR
            if self._state == VADState.SPEECH:
                await self._forward_frame(frame)
                return True

        else:
            self._consecutive_silence += 1
            self._consecutive_speech = 0

            if self._state == VADState.SPEECH:
                self._state = VADState.SPEECH_ENDING

            elif self._state == VADState.SPEECH_ENDING:
                if self._consecutive_silence >= OFFSET_FRAMES:
                    self._state = VADState.SILENCE
                    await self._publish_speech_end(turn_id)

        return False

    async def _send_pre_roll(self):
        """Send pre-roll buffer (200ms) to ASR on speech_start."""
        for frame in self._pre_roll:
            await self._forward_frame(frame)

    async def _forward_frame(self, frame: bytes):
        """Forward a speech frame to the speech queue."""
        if self._speech_queue.full():
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
            await self._event_bus.publish(Event(
                type=EventType.SPEECH_START,
                turn_id=turn_id,
            ))

    async def _publish_speech_end(self, turn_id: int):
        """Publish speech_end event to the event bus."""
        duration_ms = (time.monotonic() - self._speech_start_time) * 1000
        logger.info(f"VAD: speech_end detected (duration={duration_ms:.0f}ms)")
        if self._event_bus:
            from .state_machine import Event, EventType
            await self._event_bus.publish(Event(
                type=EventType.SPEECH_END,
                turn_id=turn_id,
                metadata={"speech_duration_ms": duration_ms},
            ))

    async def _publish_barge_in(self, turn_id: int):
        """Publish barge_in event to the event bus."""
        logger.info("VAD: barge_in detected")
        if self._event_bus:
            from .state_machine import Event, EventType
            await self._event_bus.publish(Event(
                type=EventType.BARGE_IN,
                turn_id=turn_id,
            ))

    def report_false_trigger(self):
        """Called when speech_start had no ASR output within 2s (false positive)."""
        self._false_trigger_count += 1

    def reset(self):
        """Reset VAD state (e.g., on turn boundary)."""
        self._state = VADState.SILENCE
        self._consecutive_speech = 0
        self._consecutive_silence = 0

    def get_stats(self) -> dict:
        """Get VAD statistics."""
        return {
            "state": self._state.value,
            "total_frames": self._total_frames,
            "speech_frames": self._speech_frames,
            "speech_ratio": self._speech_frames / max(1, self._total_frames),
            "total_speech_events": self._total_speech_events,
            "total_barge_in_events": self._total_barge_in_events,
            "energy_gate_rejects": self._energy_gate_rejects,
            "false_trigger_count": self._false_trigger_count,
            "energy_gate_dbfs": self._energy_gate_dbfs,
            "calibrating": self._calibrating,
            "barge_in_mode": self._barge_in_mode,
            "using_silero": self._use_silero,
        }
