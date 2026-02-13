"""
Synthetic Audio Generator for AIOS Testing

Produces PCM frames with controlled characteristics for deterministic,
reproducible testing without live microphone input.

- speech(): generates sine-wave PCM simulating speech energy
- silence(): generates zero-valued PCM frames
- noise(): generates Gaussian noise at a specified SNR
- barge_in(): generates speech -> silence -> interrupt speech sequence
"""

import numpy as np
from typing import List

CAPTURE_SAMPLE_RATE = 16000
FRAME_DURATION_MS = 20
FRAME_SAMPLES = CAPTURE_SAMPLE_RATE * FRAME_DURATION_MS // 1000  # 320
FRAME_BYTES = FRAME_SAMPLES * 2  # 640 bytes (16-bit)


class SyntheticAudio:
    """
    Synthetic audio generator for testing.

    All methods return lists of 20ms PCM frames (640 bytes each,
    16 kHz, 16-bit signed int, mono).
    """

    def __init__(self, sample_rate: int = CAPTURE_SAMPLE_RATE):
        self._sample_rate = sample_rate
        self._frame_samples = sample_rate * FRAME_DURATION_MS // 1000

    def speech(self, duration_ms: int, frequency_hz: float = 440.0, amplitude: float = 0.5) -> List[bytes]:
        """
        Generate PCM frames simulating speech (sine wave at given frequency).

        Args:
            duration_ms: Duration in milliseconds.
            frequency_hz: Tone frequency.
            amplitude: Amplitude 0.0-1.0 (relative to int16 max).

        Returns:
            List of 20ms PCM frames.
        """
        total_samples = self._sample_rate * duration_ms // 1000
        t = np.arange(total_samples, dtype=np.float64) / self._sample_rate
        signal = (amplitude * 32767 * np.sin(2 * np.pi * frequency_hz * t)).astype(np.int16)
        return self._split_frames(signal)

    def silence(self, duration_ms: int) -> List[bytes]:
        """
        Generate silent PCM frames.

        Args:
            duration_ms: Duration in milliseconds.

        Returns:
            List of 20ms PCM frames.
        """
        total_samples = self._sample_rate * duration_ms // 1000
        signal = np.zeros(total_samples, dtype=np.int16)
        return self._split_frames(signal)

    def noise(self, duration_ms: int, snr_db: float = 20.0, speech_amplitude: float = 0.5) -> List[bytes]:
        """
        Generate Gaussian noise at a specified SNR relative to speech level.

        Args:
            duration_ms: Duration in milliseconds.
            snr_db: Signal-to-noise ratio in dB.
            speech_amplitude: Reference speech amplitude for SNR calculation.

        Returns:
            List of 20ms PCM frames.
        """
        total_samples = self._sample_rate * duration_ms // 1000
        speech_power = (speech_amplitude * 32767) ** 2
        noise_power = speech_power / (10 ** (snr_db / 10))
        noise_std = np.sqrt(noise_power)
        signal = np.random.normal(0, noise_std, total_samples).astype(np.float64)
        signal = np.clip(signal, -32768, 32767).astype(np.int16)
        return self._split_frames(signal)

    def barge_in(
        self,
        speech_ms: int = 2000,
        silence_ms: int = 500,
        interrupt_ms: int = 1000,
    ) -> List[bytes]:
        """
        Generate a barge-in sequence: speech -> silence -> interrupt speech.

        Simulates a user speaking, pausing, then interrupting the AI.

        Args:
            speech_ms: Initial speech duration.
            silence_ms: Silence gap between speech and interrupt.
            interrupt_ms: Interrupt speech duration.

        Returns:
            List of 20ms PCM frames for the full sequence.
        """
        frames = []
        frames.extend(self.speech(speech_ms, frequency_hz=440.0))
        frames.extend(self.silence(silence_ms))
        frames.extend(self.speech(interrupt_ms, frequency_hz=520.0))
        return frames

    def speech_with_pauses(
        self,
        segments: List[int],
        pause_ms: int = 200,
    ) -> List[bytes]:
        """
        Generate speech with natural intra-utterance pauses.

        Args:
            segments: List of speech segment durations in ms.
            pause_ms: Pause duration between segments.

        Returns:
            List of 20ms PCM frames.
        """
        frames = []
        for i, seg_ms in enumerate(segments):
            frames.extend(self.speech(seg_ms))
            if i < len(segments) - 1:
                frames.extend(self.silence(pause_ms))
        return frames

    def _split_frames(self, signal: np.ndarray) -> List[bytes]:
        """Split a signal array into 20ms PCM frames."""
        frames = []
        for i in range(0, len(signal), self._frame_samples):
            chunk = signal[i:i + self._frame_samples]
            if len(chunk) < self._frame_samples:
                # Pad last frame with zeros
                padded = np.zeros(self._frame_samples, dtype=np.int16)
                padded[:len(chunk)] = chunk
                chunk = padded
            frames.append(chunk.tobytes())
        return frames


def generate_speech_frames(duration_ms: int = 1000) -> List[bytes]:
    """Convenience function to generate speech frames."""
    return SyntheticAudio().speech(duration_ms)


def generate_silence_frames(duration_ms: int = 500) -> List[bytes]:
    """Convenience function to generate silence frames."""
    return SyntheticAudio().silence(duration_ms)
