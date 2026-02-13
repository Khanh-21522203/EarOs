"""
Layer 2: Context & Memory Manager

Maintains short-term conversational history (last 10 turns), provides
context retrieval for prompt assembly, and manages the conversation turn lifecycle.

Concurrency boundary: Single-threaded, accessed only from the asyncio event loop.
No locks required. All mutations are sequential.

Performance constraints:
- Context retrieval: <= 2ms (in-memory list slice + token counting)
- Token counting: pre-computed on turn insertion, not on retrieval
- Memory footprint: <= 50MB for 10 turns with embeddings
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List
from collections import deque

logger = logging.getLogger(__name__)

# Default token budget
DEFAULT_MAX_TURNS = 10
DEFAULT_MAX_CONTEXT_TOKENS = 2048

# Approximate token counting: ~4 chars per token (GPT-style tokenizer estimate)
CHARS_PER_TOKEN_ESTIMATE = 4


@dataclass
class TurnRecord:
    """
    A single conversational turn.

    Attributes:
        role: "user" or "assistant"
        text: The turn text content
        timestamp: When this turn was recorded
        interrupted: Whether this turn was interrupted (barge-in)
        token_count: Pre-computed token count for this turn
        turn_id: Unique turn identifier
    """
    role: str  # "user" | "assistant"
    text: str
    timestamp: float = field(default_factory=time.time)
    interrupted: bool = False
    token_count: int = 0
    turn_id: int = 0

    def __post_init__(self):
        if self.token_count == 0:
            self.token_count = estimate_token_count(self.text)

    def validate(self) -> bool:
        """Validate the turn record schema."""
        if self.role not in ("user", "assistant"):
            return False
        if not isinstance(self.text, str) or len(self.text) == 0:
            return False
        if not isinstance(self.timestamp, (int, float)):
            return False
        return True


@dataclass
class ContextWindow:
    """
    A retrieved context window for prompt assembly.

    Attributes:
        turns: Ordered list of turn records fitting within token budget
        total_tokens: Total token count of all turns in the window
        turn_count: Number of turns in the window
        oldest_turn_timestamp: Timestamp of the oldest turn
    """
    turns: List[TurnRecord]
    total_tokens: int
    turn_count: int
    oldest_turn_timestamp: float


@dataclass
class ContextQuery:
    """Query parameters for context retrieval."""
    max_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS
    recency_bias: float = 1.0  # 1.0 = most recent first, 0.0 = no bias


def estimate_token_count(text: str) -> int:
    """
    Estimate token count for a text string.

    Uses a simple character-based heuristic (~4 chars per token).
    For production, replace with actual tokenizer (tiktoken or model-specific).

    Args:
        text: Input text.

    Returns:
        Estimated token count.
    """
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN_ESTIMATE)


class ContextManager:
    """
    Context & Memory Manager — Layer 2 of the AIOS pipeline.

    Maintains a sliding window of recent conversational turns and provides
    context retrieval for the Context Injection Engine (Layer 4).

    All operations are synchronous (no I/O) and run on the asyncio event loop.
    """

    def __init__(
        self,
        max_turns: int = DEFAULT_MAX_TURNS,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    ):
        """
        Initialize the Context Manager.

        Args:
            max_turns: Maximum number of turns to retain.
            max_context_tokens: Maximum token budget for context window.
        """
        self._max_turns = max_turns
        self._max_context_tokens = max_context_tokens

        # Turn storage — deque for efficient eviction of oldest turns
        self._turns: deque[TurnRecord] = deque(maxlen=max_turns)

        # Pre-computed total token count
        self._total_tokens: int = 0

        # Turn counter
        self._turn_counter: int = 0

        # Metrics
        self._eviction_count: int = 0
        self._rejected_count: int = 0

        logger.info(
            f"ContextManager initialized "
            f"(max_turns={max_turns}, max_tokens={max_context_tokens})"
        )

    def add_turn(self, role: str, text: str, interrupted: bool = False, turn_id: int = 0) -> Optional[TurnRecord]:
        """
        Add a completed turn to the conversation history.

        Token count is pre-computed on insertion.

        Args:
            role: "user" or "assistant"
            text: Turn text content
            interrupted: Whether this turn was interrupted
            turn_id: Turn identifier from the state machine

        Returns:
            The TurnRecord if added, None if rejected.
        """
        record = TurnRecord(
            role=role,
            text=text,
            interrupted=interrupted,
            turn_id=turn_id or self._turn_counter,
        )

        # Validate schema
        if not record.validate():
            self._rejected_count += 1
            logger.error(f"Rejected invalid turn record: role={role}, text_len={len(text)}")
            return None

        # Check if adding this turn exceeds token budget — evict oldest if needed
        while (self._total_tokens + record.token_count > self._max_context_tokens
               and len(self._turns) > 0):
            evicted = self._turns.popleft()
            self._total_tokens -= evicted.token_count
            self._eviction_count += 1
            logger.debug(
                f"Evicted oldest turn (role={evicted.role}, "
                f"tokens={evicted.token_count}) to fit budget"
            )

        # Add the turn
        self._turns.append(record)
        self._total_tokens += record.token_count
        self._turn_counter += 1

        logger.debug(
            f"Added turn: role={role}, tokens={record.token_count}, "
            f"total_turns={len(self._turns)}, total_tokens={self._total_tokens}"
        )

        return record

    def get_context(self, query: Optional[ContextQuery] = None) -> ContextWindow:
        """
        Retrieve the context window for prompt assembly.

        Performance: <= 2ms (in-memory list slice + pre-computed token counts).

        Args:
            query: Optional query parameters.

        Returns:
            ContextWindow with turns fitting within the token budget.
        """
        if query is None:
            query = ContextQuery()

        max_tokens = min(query.max_tokens, self._max_context_tokens)

        # Build context window from most recent turns, respecting token budget
        selected_turns: List[TurnRecord] = []
        token_budget = max_tokens
        oldest_timestamp = time.time()

        # Iterate from most recent to oldest
        for turn in reversed(self._turns):
            if turn.token_count <= token_budget:
                selected_turns.insert(0, turn)  # Maintain chronological order
                token_budget -= turn.token_count
                oldest_timestamp = min(oldest_timestamp, turn.timestamp)
            else:
                break  # No more room

        total_tokens = max_tokens - token_budget

        return ContextWindow(
            turns=selected_turns,
            total_tokens=total_tokens,
            turn_count=len(selected_turns),
            oldest_turn_timestamp=oldest_timestamp if selected_turns else 0.0,
        )

    def get_last_n_turns(self, n: int) -> List[TurnRecord]:
        """Get the last N turns."""
        turns = list(self._turns)
        return turns[-n:] if n < len(turns) else turns

    def get_last_user_text(self) -> Optional[str]:
        """Get the text of the most recent user turn."""
        for turn in reversed(self._turns):
            if turn.role == "user":
                return turn.text
        return None

    def get_last_assistant_text(self) -> Optional[str]:
        """Get the text of the most recent assistant turn."""
        for turn in reversed(self._turns):
            if turn.role == "assistant":
                return turn.text
        return None

    def clear(self):
        """Clear all conversation history."""
        self._turns.clear()
        self._total_tokens = 0
        logger.info("Conversation history cleared")

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    def get_stats(self) -> dict:
        """Get context manager statistics."""
        return {
            "turn_count": len(self._turns),
            "max_turns": self._max_turns,
            "total_tokens": self._total_tokens,
            "max_context_tokens": self._max_context_tokens,
            "eviction_count": self._eviction_count,
            "rejected_count": self._rejected_count,
            "token_utilization_pct": (
                (self._total_tokens / self._max_context_tokens) * 100
                if self._max_context_tokens > 0 else 0
            ),
        }
