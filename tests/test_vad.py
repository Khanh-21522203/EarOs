"""
Unit tests for Voice Activity Detection.

Tests from voice-activity-detection.md Definition of Done:
- Energy gate calibration runs at startup and sets threshold correctly
- Temporal smoother correctly debounces onset (60ms) and offset (300ms)
- Pre-roll buffer delivers 200ms of pre-speech audio to ASR
- Barge-in detection with 5-frame threshold
"""

import asyncio
import numpy as np
import pytest

from aios.core.vad import (
    VoiceActivityDetector,
    VADState,
    ONSET_FRAMES,
    OFFSET_FRAMES,
    BARGE_IN_FRAMES,
    PRE_ROLL_FRAMES,
    VAD_FRAME_SAMPLES,
)


def make_speech_frame(amplitude: float = 0.5) -> bytes:
    """Generate a loud frame that passes energy gate."""
    samples = (np.sin(np.linspace(0, 2 * np.pi * 440, VAD_FRAME_SAMPLES)) * amplitude * 32767).astype(np.int16)
    return samples.tobytes()


def make_silence_frame() -> bytes:
    """Generate a silent frame."""
    return bytes(VAD_FRAME_SAMPLES * 2)


@pytest.fixture
def vad():
    q = asyncio.Queue(maxsize=500)
    v = VoiceActivityDetector(speech_queue=q, use_silero=False)
    # Skip calibration for tests by setting calibrating=False
    v._calibrating = False
    v._energy_gate_dbfs = -50.0
    return v, q


# ── Energy Gate ──

def test_energy_gate_rejects_silence(vad):
    v, q = vad
    is_speech, conf = v.process_frame(make_silence_frame())
    assert not is_speech
    assert v._energy_gate_rejects > 0


def test_energy_gate_passes_speech(vad):
    v, q = vad
    is_speech, conf = v.process_frame(make_speech_frame())
    assert conf > 0


# ── Startup Calibration ──

def test_startup_calibration():
    q = asyncio.Queue(maxsize=500)
    v = VoiceActivityDetector(speech_queue=q, use_silero=False)
    assert v._calibrating is True

    # Feed 100 silence frames (2s) to complete calibration
    for _ in range(100):
        v.process_frame(make_silence_frame())

    assert v._calibrating is False
    assert v._energy_gate_dbfs > -96.0  # Should be calibrated above silence floor


# ── Temporal Smoother: Onset ──

@pytest.mark.asyncio
async def test_onset_requires_3_frames(vad):
    v, q = vad

    # 2 speech frames should NOT trigger speech_start
    for _ in range(ONSET_FRAMES - 1):
        result = await v.process_frame_async(make_speech_frame())
    assert v._state == VADState.SILENCE

    # 3rd frame triggers speech_start
    result = await v.process_frame_async(make_speech_frame())
    assert v._state == VADState.SPEECH


# ── Temporal Smoother: Offset ──

@pytest.mark.asyncio
async def test_offset_requires_15_frames(vad):
    v, q = vad

    # Enter speech state
    for _ in range(ONSET_FRAMES):
        await v.process_frame_async(make_speech_frame())
    assert v._state == VADState.SPEECH

    # 14 silence frames should NOT trigger speech_end
    for _ in range(OFFSET_FRAMES - 1):
        await v.process_frame_async(make_silence_frame())
    assert v._state == VADState.SPEECH_ENDING

    # 15th silence frame triggers speech_end
    await v.process_frame_async(make_silence_frame())
    assert v._state == VADState.SILENCE


@pytest.mark.asyncio
async def test_speech_resumes_during_cooldown(vad):
    v, q = vad

    # Enter speech
    for _ in range(ONSET_FRAMES):
        await v.process_frame_async(make_speech_frame())
    assert v._state == VADState.SPEECH

    # Start silence cooldown
    for _ in range(5):
        await v.process_frame_async(make_silence_frame())
    assert v._state == VADState.SPEECH_ENDING

    # Resume speech — should go back to SPEECH
    await v.process_frame_async(make_speech_frame())
    assert v._state == VADState.SPEECH


# ── Pre-roll Buffer ──

@pytest.mark.asyncio
async def test_pre_roll_sends_200ms(vad):
    v, q = vad

    # Feed 15 frames (more than pre-roll buffer)
    for _ in range(15):
        v.process_frame(make_silence_frame())

    # Now trigger speech_start
    for _ in range(ONSET_FRAMES):
        await v.process_frame_async(make_speech_frame())

    # Queue should have pre-roll frames + speech frames
    assert q.qsize() >= PRE_ROLL_FRAMES


# ── Barge-in Mode ──

@pytest.mark.asyncio
async def test_barge_in_requires_5_frames():
    q = asyncio.Queue(maxsize=500)
    v = VoiceActivityDetector(speech_queue=q, use_silero=False, barge_in_mode=True)
    v._calibrating = False
    v._energy_gate_dbfs = -50.0

    # 4 speech frames should NOT trigger barge_in
    for _ in range(BARGE_IN_FRAMES - 1):
        await v.process_frame_async(make_speech_frame())
    assert v._state == VADState.SILENCE

    # 5th frame triggers barge_in
    await v.process_frame_async(make_speech_frame())
    assert v._state == VADState.SPEECH
    assert v._total_barge_in_events == 1


# ── Reset ──

@pytest.mark.asyncio
async def test_reset(vad):
    v, q = vad
    for _ in range(ONSET_FRAMES):
        await v.process_frame_async(make_speech_frame())
    assert v._state == VADState.SPEECH

    v.reset()
    assert v._state == VADState.SILENCE
    assert v._consecutive_speech == 0
    assert v._consecutive_silence == 0


# ── Stats ──

def test_stats(vad):
    v, q = vad
    stats = v.get_stats()
    assert "state" in stats
    assert "total_frames" in stats
    assert "energy_gate_rejects" in stats
    assert "barge_in_mode" in stats
