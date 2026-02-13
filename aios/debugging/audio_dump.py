"""
Audio Dump for Debug Mode

When AIOS_DEBUG=1, raw audio is saved to WAV files for offline analysis:
- /tmp/aios_debug/capture_{turn_id}.wav   (16 kHz, 16-bit, mono)
- /tmp/aios_debug/aec_{turn_id}.wav       (16 kHz, 16-bit, mono)
- /tmp/aios_debug/playback_{turn_id}.wav  (24 kHz, 16-bit, mono)
"""

import logging
import os
import struct
import wave
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

AIOS_DEBUG = os.environ.get("AIOS_DEBUG", "0") == "1"
DEBUG_DIR = Path("/tmp/aios_debug")

CAPTURE_SAMPLE_RATE = 16000
PLAYBACK_SAMPLE_RATE = 24000


class AudioDumper:
    """
    Accumulates audio frames per turn and writes WAV files on turn completion.

    Only active when AIOS_DEBUG=1. No-ops otherwise.
    """

    def __init__(self, enabled: Optional[bool] = None):
        self._enabled = enabled if enabled is not None else AIOS_DEBUG
        self._capture_buffers: dict[int, bytearray] = {}
        self._aec_buffers: dict[int, bytearray] = {}
        self._playback_buffers: dict[int, bytearray] = {}

        if self._enabled:
            DEBUG_DIR.mkdir(parents=True, exist_ok=True)
            logger.info("Audio dumper enabled. Output dir: %s", DEBUG_DIR)

    def record_capture(self, turn_id: int, frame: bytes):
        if not self._enabled:
            return
        if turn_id not in self._capture_buffers:
            self._capture_buffers[turn_id] = bytearray()
        self._capture_buffers[turn_id].extend(frame)

    def record_aec(self, turn_id: int, frame: bytes):
        if not self._enabled:
            return
        if turn_id not in self._aec_buffers:
            self._aec_buffers[turn_id] = bytearray()
        self._aec_buffers[turn_id].extend(frame)

    def record_playback(self, turn_id: int, frame: bytes):
        if not self._enabled:
            return
        if turn_id not in self._playback_buffers:
            self._playback_buffers[turn_id] = bytearray()
        self._playback_buffers[turn_id].extend(frame)

    def flush_turn(self, turn_id: int):
        """Write accumulated audio for a turn to WAV files."""
        if not self._enabled:
            return

        if turn_id in self._capture_buffers:
            self._write_wav(
                DEBUG_DIR / f"capture_{turn_id}.wav",
                self._capture_buffers.pop(turn_id),
                CAPTURE_SAMPLE_RATE,
            )

        if turn_id in self._aec_buffers:
            self._write_wav(
                DEBUG_DIR / f"aec_{turn_id}.wav",
                self._aec_buffers.pop(turn_id),
                CAPTURE_SAMPLE_RATE,
            )

        if turn_id in self._playback_buffers:
            self._write_wav(
                DEBUG_DIR / f"playback_{turn_id}.wav",
                self._playback_buffers.pop(turn_id),
                PLAYBACK_SAMPLE_RATE,
            )

    @staticmethod
    def _write_wav(path: Path, data: bytes, sample_rate: int):
        """Write raw PCM data to a WAV file."""
        try:
            with wave.open(str(path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(data)
            logger.debug("Audio dump written: %s (%d bytes)", path, len(data))
        except Exception as e:
            logger.error("Failed to write audio dump %s: %s", path, e)

    def cleanup(self):
        """Clear all buffers."""
        self._capture_buffers.clear()
        self._aec_buffers.clear()
        self._playback_buffers.clear()
