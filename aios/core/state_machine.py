"""
Layer 5: State Machine & Event Bus

Manages the conversational state machine, dispatches state transition events
to all layers, enforces valid transitions, and logs every state change.

States: IDLE, LISTENING, PROCESSING, SPEAKING, INTERRUPTED
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, Awaitable, Dict, Set, List
from enum import Enum

logger = logging.getLogger(__name__)

# Per-state watchdog timeouts (seconds) from turn-taking.md
STATE_TIMEOUTS: Dict[str, float] = {
    "LISTENING": 10.0,
    "PROCESSING": 5.0,
    "SPEAKING": 30.0,
    "INTERRUPTED": 0.5,
}


class ConversationState(Enum):
    """Conversational states for the AIOS state machine."""
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    SPEAKING = "SPEAKING"
    INTERRUPTED = "INTERRUPTED"


class EventType(Enum):
    """Events that trigger state transitions."""
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"
    FIRST_TOKEN = "first_token"
    FIRST_TTS_FRAME = "first_tts_frame"
    PLAYBACK_COMPLETE = "playback_complete"
    BARGE_IN = "barge_in"
    RESPONSE_COMPLETE = "response_complete"
    INTERRUPT_COMPLETE = "interrupt_complete"
    TURN_TIMEOUT = "turn_timeout"
    PROCESSING_TIMEOUT = "processing_timeout"
    WATCHDOG_TIMEOUT = "watchdog_timeout"
    SHUTDOWN = "shutdown"


@dataclass
class Event:
    """An event published to the event bus."""
    type: EventType
    timestamp: float = field(default_factory=time.monotonic)
    turn_id: int = 0
    metadata: Optional[dict] = None

    def __repr__(self) -> str:
        return f"Event({self.type.value}, turn={self.turn_id})"


# Valid state transitions: (current_state, event) -> next_state
# From turn-taking.md Section 3
VALID_TRANSITIONS: Dict[tuple[ConversationState, EventType], ConversationState] = {
    # Happy path
    (ConversationState.IDLE, EventType.SPEECH_START): ConversationState.LISTENING,
    (ConversationState.LISTENING, EventType.SPEECH_END): ConversationState.PROCESSING,
    (ConversationState.PROCESSING, EventType.FIRST_TTS_FRAME): ConversationState.SPEAKING,
    (ConversationState.PROCESSING, EventType.FIRST_TOKEN): ConversationState.SPEAKING,
    (ConversationState.PROCESSING, EventType.RESPONSE_COMPLETE): ConversationState.IDLE,
    (ConversationState.SPEAKING, EventType.PLAYBACK_COMPLETE): ConversationState.IDLE,
    # Barge-in
    (ConversationState.SPEAKING, EventType.BARGE_IN): ConversationState.INTERRUPTED,
    (ConversationState.INTERRUPTED, EventType.INTERRUPT_COMPLETE): ConversationState.LISTENING,
    # User speaks again during processing (cancel current generation)
    (ConversationState.PROCESSING, EventType.SPEECH_START): ConversationState.LISTENING,
    # Timeouts
    (ConversationState.LISTENING, EventType.TURN_TIMEOUT): ConversationState.IDLE,
    (ConversationState.PROCESSING, EventType.PROCESSING_TIMEOUT): ConversationState.IDLE,
    # Watchdog can force any non-IDLE state to IDLE
    (ConversationState.LISTENING, EventType.WATCHDOG_TIMEOUT): ConversationState.IDLE,
    (ConversationState.PROCESSING, EventType.WATCHDOG_TIMEOUT): ConversationState.IDLE,
    (ConversationState.SPEAKING, EventType.WATCHDOG_TIMEOUT): ConversationState.IDLE,
    (ConversationState.INTERRUPTED, EventType.WATCHDOG_TIMEOUT): ConversationState.IDLE,
}


@dataclass
class StateTransitionRecord:
    """Record of a state transition for logging and debugging."""
    from_state: ConversationState
    to_state: ConversationState
    event: EventType
    turn_id: int
    timestamp: float
    duration_in_previous_state_ms: float


StateChangeCallback = Callable[[ConversationState, ConversationState, Event], Awaitable[None]]


class EventBus:
    """
    Async event bus for inter-layer communication.

    All layers publish events to the bus; the state machine task consumes
    them sequentially. Uses asyncio.Queue with bounded capacity.
    """

    def __init__(self, max_depth: int = 100):
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=max_depth)
        self._subscribers: List[Callable[[Event], Awaitable[None]]] = []
        self._total_published: int = 0
        self._total_dropped: int = 0

    async def publish(self, event: Event) -> bool:
        """
        Publish an event to the bus.

        Returns:
            True if published, False if dropped (bus overflow).
        """
        try:
            self._queue.put_nowait(event)
            self._total_published += 1
            return True
        except asyncio.QueueFull:
            self._total_dropped += 1
            logger.critical(
                f"Event bus overflow! Dropping event {event.type.value}. "
                f"Total dropped: {self._total_dropped}. System may be overloaded."
            )
            return False

    async def consume(self) -> Event:
        """Consume the next event from the bus (blocks until available)."""
        return await self._queue.get()

    def subscribe(self, callback: Callable[[Event], Awaitable[None]]):
        """Subscribe to all events on the bus."""
        self._subscribers.append(callback)

    async def notify_subscribers(self, event: Event):
        """Notify all subscribers of an event."""
        for callback in self._subscribers:
            try:
                await callback(event)
            except Exception as e:
                logger.error(f"Event subscriber error: {e}")

    @property
    def depth(self) -> int:
        return self._queue.qsize()

    @property
    def total_published(self) -> int:
        return self._total_published

    @property
    def total_dropped(self) -> int:
        return self._total_dropped


class StateMachine:
    """
    Conversational state machine for AIOS.

    Manages state transitions, enforces valid transitions, logs every
    state change, and runs a watchdog timer for stuck states.

    Performance constraints:
    - Event dispatch latency: <= 1ms
    - State transition + logging: <= 2ms
    """

    def __init__(
        self,
        event_bus: EventBus,
        watchdog_timeout_seconds: float = 10.0,
    ):
        self._event_bus = event_bus
        self._watchdog_timeout = watchdog_timeout_seconds

        # State
        self._state = ConversationState.IDLE
        self._turn_id: int = 0
        self._state_entered_at: float = time.monotonic()

        # History
        self._transition_history: List[StateTransitionRecord] = []
        self._max_history: int = 100

        # Callbacks
        self._on_state_change: List[StateChangeCallback] = []

        # Tasks
        self._consumer_task: Optional[asyncio.Task] = None
        self._watchdog_task: Optional[asyncio.Task] = None
        self._running: bool = False

        # Metrics
        self._invalid_transition_count: int = 0
        self._total_transitions: int = 0

        # Flush callbacks — called when transitioning to INTERRUPTED
        self._flush_callbacks: List[Callable[[], Awaitable[None]]] = []

    @property
    def state(self) -> ConversationState:
        return self._state

    @property
    def turn_id(self) -> int:
        return self._turn_id

    @property
    def is_running(self) -> bool:
        return self._running

    def on_state_change(self, callback: StateChangeCallback):
        """Register a callback for state changes."""
        self._on_state_change.append(callback)

    def on_flush(self, callback: Callable[[], Awaitable[None]]):
        """Register a callback for queue flushing (barge-in)."""
        self._flush_callbacks.append(callback)

    async def start(self):
        """Start the state machine event consumer and watchdog."""
        self._running = True
        self._consumer_task = asyncio.create_task(self._event_consumer_loop())
        self._watchdog_task = asyncio.create_task(self._watchdog_loop())
        logger.info("State machine started")

    async def stop(self):
        """Stop the state machine."""
        self._running = False

        if self._consumer_task and not self._consumer_task.done():
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass

        if self._watchdog_task and not self._watchdog_task.done():
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                pass

        logger.info("State machine stopped")

    async def _event_consumer_loop(self):
        """Main event consumer loop — processes events sequentially."""
        try:
            while self._running:
                try:
                    event = await asyncio.wait_for(
                        self._event_bus.consume(),
                        timeout=0.1,
                    )
                    await self._handle_event(event)
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            logger.debug("Event consumer loop cancelled")

    async def _handle_event(self, event: Event):
        """Handle a single event — attempt state transition."""
        start_time = time.monotonic()

        if event.type == EventType.SHUTDOWN:
            self._running = False
            return

        transition_key = (self._state, event.type)
        new_state = VALID_TRANSITIONS.get(transition_key)

        if new_state is None:
            self._invalid_transition_count += 1
            logger.error(
                f"Invalid transition: state={self._state.value}, "
                f"event={event.type.value}. Remaining in {self._state.value}. "
                f"Total invalid: {self._invalid_transition_count}"
            )
            return

        # Execute transition
        old_state = self._state
        now = time.monotonic()
        duration_ms = (now - self._state_entered_at) * 1000

        self._state = new_state
        self._state_entered_at = now
        self._total_transitions += 1

        # Increment turn ID on new listening state from IDLE
        if old_state == ConversationState.IDLE and new_state == ConversationState.LISTENING:
            self._turn_id += 1

        # Record transition
        record = StateTransitionRecord(
            from_state=old_state,
            to_state=new_state,
            event=event.type,
            turn_id=self._turn_id,
            timestamp=now,
            duration_in_previous_state_ms=duration_ms,
        )
        self._transition_history.append(record)
        if len(self._transition_history) > self._max_history:
            self._transition_history.pop(0)

        # Log transition
        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            f"State transition: {old_state.value} -> {new_state.value} "
            f"[event={event.type.value}, turn={self._turn_id}, "
            f"prev_duration={duration_ms:.1f}ms, processing={elapsed_ms:.2f}ms]"
        )

        # Handle barge-in flush
        if new_state == ConversationState.INTERRUPTED:
            await self._handle_barge_in()

        # Notify callbacks
        for callback in self._on_state_change:
            try:
                await callback(old_state, new_state, event)
            except Exception as e:
                logger.error(f"State change callback error: {e}")

        # Notify event bus subscribers
        await self._event_bus.notify_subscribers(event)

    async def _handle_barge_in(self):
        """Handle barge-in: flush downstream queues."""
        logger.info("Barge-in detected — flushing downstream queues")
        for callback in self._flush_callbacks:
            try:
                await callback()
            except Exception as e:
                logger.error(f"Flush callback error: {e}")

    async def _watchdog_loop(self):
        """Watchdog timer — per-state timeouts from turn-taking.md Section 7."""
        try:
            while self._running:
                await asyncio.sleep(0.25)

                if self._state == ConversationState.IDLE:
                    continue

                elapsed = time.monotonic() - self._state_entered_at
                state_name = self._state.value
                timeout = STATE_TIMEOUTS.get(state_name, self._watchdog_timeout)

                if elapsed > timeout:
                    # Choose the appropriate timeout event type
                    if self._state == ConversationState.LISTENING:
                        event_type = EventType.TURN_TIMEOUT
                    elif self._state == ConversationState.PROCESSING:
                        event_type = EventType.PROCESSING_TIMEOUT
                    elif self._state == ConversationState.INTERRUPTED:
                        event_type = EventType.INTERRUPT_COMPLETE
                    else:
                        event_type = EventType.WATCHDOG_TIMEOUT

                    logger.error(
                        f"Watchdog timeout! Stuck in {state_name} "
                        f"for {elapsed:.1f}s (limit: {timeout}s). "
                        f"Emitting {event_type.value}."
                    )
                    await self._event_bus.publish(Event(
                        type=event_type,
                        turn_id=self._turn_id,
                    ))
        except asyncio.CancelledError:
            logger.debug("Watchdog loop cancelled")

    def get_status(self) -> dict:
        """Get current state machine status."""
        return {
            "state": self._state.value,
            "turn_id": self._turn_id,
            "total_transitions": self._total_transitions,
            "invalid_transitions": self._invalid_transition_count,
            "time_in_current_state_ms": (time.monotonic() - self._state_entered_at) * 1000,
            "event_bus_depth": self._event_bus.depth,
            "event_bus_total_published": self._event_bus.total_published,
            "event_bus_total_dropped": self._event_bus.total_dropped,
        }

    def get_recent_transitions(self, count: int = 10) -> List[StateTransitionRecord]:
        """Get recent transition history."""
        return self._transition_history[-count:]


def create_state_machine(
    watchdog_timeout_seconds: float = 10.0,
    event_bus_depth: int = 100,
) -> tuple[StateMachine, EventBus]:
    """
    Factory function to create a state machine with its event bus.

    Returns:
        Tuple of (StateMachine, EventBus)
    """
    event_bus = EventBus(max_depth=event_bus_depth)
    state_machine = StateMachine(
        event_bus=event_bus,
        watchdog_timeout_seconds=watchdog_timeout_seconds,
    )
    return state_machine, event_bus
