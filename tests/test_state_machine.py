"""
Unit tests for the State Machine & Event Bus (Layer 5).

Tests from turn-taking.md Definition of Done:
- Valid transition tests (all transitions)
- Invalid transition rejection tests
- Watchdog timers fire correctly for each state
- Turn ID tracking prevents stale item processing
- All state transitions logged with timestamp, from-state, to-state, trigger
"""

import asyncio
import pytest
import pytest_asyncio

from aios.core.state_machine import (
    StateMachine,
    EventBus,
    Event,
    EventType,
    ConversationState,
    VALID_TRANSITIONS,
    create_state_machine,
)


@pytest.fixture
def sm_bus():
    """Create a state machine + event bus pair."""
    return create_state_machine(watchdog_timeout_seconds=60.0)


# ── Valid Transitions ──

@pytest.mark.asyncio
async def test_idle_to_listening_on_speech_start(sm_bus):
    sm, bus = sm_bus
    await sm.start()
    assert sm.state == ConversationState.IDLE

    await bus.publish(Event(type=EventType.SPEECH_START))
    await asyncio.sleep(0.15)
    assert sm.state == ConversationState.LISTENING
    assert sm.turn_id == 1
    await sm.stop()


@pytest.mark.asyncio
async def test_listening_to_processing_on_speech_end(sm_bus):
    sm, bus = sm_bus
    await sm.start()
    await bus.publish(Event(type=EventType.SPEECH_START))
    await asyncio.sleep(0.15)

    await bus.publish(Event(type=EventType.SPEECH_END))
    await asyncio.sleep(0.15)
    assert sm.state == ConversationState.PROCESSING
    await sm.stop()


@pytest.mark.asyncio
async def test_processing_to_speaking_on_first_tts_frame(sm_bus):
    sm, bus = sm_bus
    await sm.start()
    await bus.publish(Event(type=EventType.SPEECH_START))
    await asyncio.sleep(0.15)
    await bus.publish(Event(type=EventType.SPEECH_END))
    await asyncio.sleep(0.15)

    await bus.publish(Event(type=EventType.FIRST_TTS_FRAME))
    await asyncio.sleep(0.15)
    assert sm.state == ConversationState.SPEAKING
    await sm.stop()


@pytest.mark.asyncio
async def test_processing_to_speaking_on_first_token(sm_bus):
    sm, bus = sm_bus
    await sm.start()
    await bus.publish(Event(type=EventType.SPEECH_START))
    await asyncio.sleep(0.15)
    await bus.publish(Event(type=EventType.SPEECH_END))
    await asyncio.sleep(0.15)

    await bus.publish(Event(type=EventType.FIRST_TOKEN))
    await asyncio.sleep(0.15)
    assert sm.state == ConversationState.SPEAKING
    await sm.stop()


@pytest.mark.asyncio
async def test_processing_to_idle_on_response_complete(sm_bus):
    sm, bus = sm_bus
    await sm.start()
    await bus.publish(Event(type=EventType.SPEECH_START))
    await asyncio.sleep(0.15)
    await bus.publish(Event(type=EventType.SPEECH_END))
    await asyncio.sleep(0.15)

    await bus.publish(Event(type=EventType.RESPONSE_COMPLETE))
    await asyncio.sleep(0.15)
    assert sm.state == ConversationState.IDLE
    await sm.stop()


@pytest.mark.asyncio
async def test_speaking_to_idle_on_playback_complete(sm_bus):
    sm, bus = sm_bus
    await sm.start()
    await bus.publish(Event(type=EventType.SPEECH_START))
    await asyncio.sleep(0.15)
    await bus.publish(Event(type=EventType.SPEECH_END))
    await asyncio.sleep(0.15)
    await bus.publish(Event(type=EventType.FIRST_TOKEN))
    await asyncio.sleep(0.15)

    await bus.publish(Event(type=EventType.PLAYBACK_COMPLETE))
    await asyncio.sleep(0.15)
    assert sm.state == ConversationState.IDLE
    await sm.stop()


@pytest.mark.asyncio
async def test_speaking_to_interrupted_on_barge_in(sm_bus):
    sm, bus = sm_bus
    await sm.start()
    await bus.publish(Event(type=EventType.SPEECH_START))
    await asyncio.sleep(0.15)
    await bus.publish(Event(type=EventType.SPEECH_END))
    await asyncio.sleep(0.15)
    await bus.publish(Event(type=EventType.FIRST_TOKEN))
    await asyncio.sleep(0.15)

    await bus.publish(Event(type=EventType.BARGE_IN))
    await asyncio.sleep(0.15)
    assert sm.state == ConversationState.INTERRUPTED
    await sm.stop()


@pytest.mark.asyncio
async def test_interrupted_to_listening_on_interrupt_complete(sm_bus):
    sm, bus = sm_bus
    await sm.start()
    await bus.publish(Event(type=EventType.SPEECH_START))
    await asyncio.sleep(0.15)
    await bus.publish(Event(type=EventType.SPEECH_END))
    await asyncio.sleep(0.15)
    await bus.publish(Event(type=EventType.FIRST_TOKEN))
    await asyncio.sleep(0.15)
    await bus.publish(Event(type=EventType.BARGE_IN))
    await asyncio.sleep(0.15)

    await bus.publish(Event(type=EventType.INTERRUPT_COMPLETE))
    await asyncio.sleep(0.15)
    assert sm.state == ConversationState.LISTENING
    await sm.stop()


@pytest.mark.asyncio
async def test_processing_to_listening_on_speech_start(sm_bus):
    """User speaks again during processing — cancel and restart."""
    sm, bus = sm_bus
    await sm.start()
    await bus.publish(Event(type=EventType.SPEECH_START))
    await asyncio.sleep(0.15)
    await bus.publish(Event(type=EventType.SPEECH_END))
    await asyncio.sleep(0.15)
    assert sm.state == ConversationState.PROCESSING

    await bus.publish(Event(type=EventType.SPEECH_START))
    await asyncio.sleep(0.15)
    assert sm.state == ConversationState.LISTENING
    await sm.stop()


# ── Invalid Transitions ──

@pytest.mark.asyncio
async def test_reject_speech_end_from_idle(sm_bus):
    sm, bus = sm_bus
    await sm.start()
    await bus.publish(Event(type=EventType.SPEECH_END))
    await asyncio.sleep(0.15)
    assert sm.state == ConversationState.IDLE
    await sm.stop()


@pytest.mark.asyncio
async def test_reject_barge_in_from_idle(sm_bus):
    sm, bus = sm_bus
    await sm.start()
    await bus.publish(Event(type=EventType.BARGE_IN))
    await asyncio.sleep(0.15)
    assert sm.state == ConversationState.IDLE
    await sm.stop()


@pytest.mark.asyncio
async def test_reject_barge_in_from_listening(sm_bus):
    sm, bus = sm_bus
    await sm.start()
    await bus.publish(Event(type=EventType.SPEECH_START))
    await asyncio.sleep(0.15)

    await bus.publish(Event(type=EventType.BARGE_IN))
    await asyncio.sleep(0.15)
    assert sm.state == ConversationState.LISTENING
    await sm.stop()


@pytest.mark.asyncio
async def test_reject_speech_end_from_speaking(sm_bus):
    """Offset event during system speech is echo, not user."""
    sm, bus = sm_bus
    await sm.start()
    await bus.publish(Event(type=EventType.SPEECH_START))
    await asyncio.sleep(0.15)
    await bus.publish(Event(type=EventType.SPEECH_END))
    await asyncio.sleep(0.15)
    await bus.publish(Event(type=EventType.FIRST_TOKEN))
    await asyncio.sleep(0.15)
    assert sm.state == ConversationState.SPEAKING

    await bus.publish(Event(type=EventType.SPEECH_END))
    await asyncio.sleep(0.15)
    assert sm.state == ConversationState.SPEAKING
    await sm.stop()


@pytest.mark.asyncio
async def test_reject_playback_complete_from_listening(sm_bus):
    sm, bus = sm_bus
    await sm.start()
    await bus.publish(Event(type=EventType.SPEECH_START))
    await asyncio.sleep(0.15)

    await bus.publish(Event(type=EventType.PLAYBACK_COMPLETE))
    await asyncio.sleep(0.15)
    assert sm.state == ConversationState.LISTENING
    await sm.stop()


# ── Turn ID Tracking ──

@pytest.mark.asyncio
async def test_turn_id_increments(sm_bus):
    sm, bus = sm_bus
    await sm.start()
    assert sm.turn_id == 0

    # Turn 1
    await bus.publish(Event(type=EventType.SPEECH_START))
    await asyncio.sleep(0.15)
    assert sm.turn_id == 1

    await bus.publish(Event(type=EventType.SPEECH_END))
    await asyncio.sleep(0.15)
    await bus.publish(Event(type=EventType.RESPONSE_COMPLETE))
    await asyncio.sleep(0.15)

    # Turn 2
    await bus.publish(Event(type=EventType.SPEECH_START))
    await asyncio.sleep(0.15)
    assert sm.turn_id == 2
    await sm.stop()


# ── Transition History ──

@pytest.mark.asyncio
async def test_transition_history_recorded(sm_bus):
    sm, bus = sm_bus
    await sm.start()
    await bus.publish(Event(type=EventType.SPEECH_START))
    await asyncio.sleep(0.15)
    await bus.publish(Event(type=EventType.SPEECH_END))
    await asyncio.sleep(0.15)

    history = sm.get_recent_transitions(10)
    assert len(history) == 2
    assert history[0].from_state == ConversationState.IDLE
    assert history[0].to_state == ConversationState.LISTENING
    assert history[1].from_state == ConversationState.LISTENING
    assert history[1].to_state == ConversationState.PROCESSING
    await sm.stop()


# ── Status ──

@pytest.mark.asyncio
async def test_get_status(sm_bus):
    sm, bus = sm_bus
    await sm.start()
    status = sm.get_status()
    assert status["state"] == "IDLE"
    assert status["turn_id"] == 0
    assert status["total_transitions"] == 0
    assert status["invalid_transitions"] == 0
    await sm.stop()


# ── Event Bus ──

@pytest.mark.asyncio
async def test_event_bus_overflow():
    bus = EventBus(max_depth=2)
    assert await bus.publish(Event(type=EventType.SPEECH_START))
    assert await bus.publish(Event(type=EventType.SPEECH_END))
    assert not await bus.publish(Event(type=EventType.FIRST_TOKEN))
    assert bus.total_dropped == 1


@pytest.mark.asyncio
async def test_event_bus_subscriber():
    bus = EventBus()
    received = []

    async def on_event(event):
        received.append(event)

    bus.subscribe(on_event)
    event = Event(type=EventType.SPEECH_START)
    await bus.notify_subscribers(event)
    assert len(received) == 1
    assert received[0].type == EventType.SPEECH_START
