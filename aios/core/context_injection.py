"""
Layer 4: Context Injection Engine

Assembles the final prompt for the SLM by combining the current ASR output,
corrected text, conversational context from Layer 2, system instructions,
and hot-word hints.

Concurrency boundary: Runs on the asyncio event loop. Synchronous string
assembly (no I/O). Accesses Layer 2 via direct function call (same thread).

Performance constraints:
- Prompt assembly: <= 5ms
- Token counting: cached per-segment, incremental updates only

Token budget allocation:
- System prompt: <= 512 tokens
- Context window: <= 2048 tokens
- Current turn: <= 512 tokens
- Reserved for response: >= 1024 tokens
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List

from .context_manager import ContextManager, ContextQuery, ContextWindow, estimate_token_count

logger = logging.getLogger(__name__)

# Token budget allocation
SYSTEM_PROMPT_MAX_TOKENS = 512
CONTEXT_WINDOW_MAX_TOKENS = 2048
CURRENT_TURN_MAX_TOKENS = 512
RESPONSE_RESERVED_TOKENS = 1024

DEFAULT_SYSTEM_PROMPT = (
    "You are AIOS, a conversational AI assistant. "
    "Respond concisely and naturally in spoken English. "
    "Keep responses under 3 sentences unless asked for detail. "
    "The user may have a Vietnamese accent. Do not comment on pronunciation.\n"
    "Known terms: {hot_words}"
)

CORRECTION_SYSTEM_PROMPT = (
    "Fix accent-related ASR errors only. "
    "Do not rephrase. Do not add words. Do not remove words. "
    "Only correct phonetic misrecognitions typical of Vietnamese-accented English."
)


@dataclass
class HotWord:
    """A hot-word entry for ASR and correction boosting."""
    term: str
    phonetic_variants: List[str] = field(default_factory=list)


@dataclass
class AssembledPrompt:
    """
    The fully assembled prompt ready for SLM inference.

    Includes token counts per segment and versioning metadata
    for debugging and cache invalidation.
    """
    full_prompt: str
    system_prompt_tokens: int
    context_tokens: int
    current_turn_tokens: int
    total_tokens: int
    assembly_time_ms: float
    hot_words_injected: int = 0
    # Prompt versioning metadata (context-injection.md Section 7)
    turn_id: int = 0
    prompt_version: int = 1
    context_hash: str = ""


class ContextInjectionEngine:
    """
    Context Injection Engine -- Layer 4 of the AIOS pipeline.

    Assembles prompts for the SLM by combining:
    1. System instructions
    2. Conversational context from the Context Manager (Layer 2)
    3. Current turn ASR/corrected text
    4. Hot-word hints

    All operations are synchronous and run on the asyncio event loop.
    """

    def __init__(
        self,
        context_manager: ContextManager,
        system_prompt: Optional[str] = None,
        hot_words: Optional[List[HotWord]] = None,
        model_context_window: int = 4096,
    ):
        self._context_manager = context_manager
        self._model_context_window = model_context_window

        # System prompt
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._system_prompt_tokens = estimate_token_count(self._system_prompt)

        if self._system_prompt_tokens > SYSTEM_PROMPT_MAX_TOKENS:
            logger.warning(
                "System prompt exceeds budget: %d > %d tokens. Truncating.",
                self._system_prompt_tokens, SYSTEM_PROMPT_MAX_TOKENS,
            )
            self._system_prompt = self._truncate_to_tokens(
                self._system_prompt, SYSTEM_PROMPT_MAX_TOKENS
            )
            self._system_prompt_tokens = estimate_token_count(self._system_prompt)

        # Hot-words
        self._hot_words: List[HotWord] = hot_words or []
        self._hot_word_text = self._format_hot_words()

        # Prompt versioning
        self._prompt_version: int = 1

        # Metrics
        self._total_assemblies: int = 0
        self._total_truncations: int = 0

        logger.info(
            "ContextInjectionEngine initialized "
            "(system_tokens=%d, hot_words=%d, model_window=%d)",
            self._system_prompt_tokens, len(self._hot_words), model_context_window,
        )

    def set_system_prompt(self, prompt: str):
        """Update the system prompt. Increments prompt_version."""
        self._system_prompt = prompt
        self._system_prompt_tokens = estimate_token_count(prompt)
        self._prompt_version += 1

    def set_hot_words(self, hot_words: List[HotWord]):
        """Update the hot-word list (takes effect on next assembly)."""
        self._hot_words = hot_words
        self._hot_word_text = self._format_hot_words()
        logger.info("Hot-word list updated: %d entries", len(hot_words))

    def assemble_prompt(
        self,
        current_turn_text: str,
        is_correction: bool = False,
        turn_id: int = 0,
    ) -> AssembledPrompt:
        """
        Assemble the full prompt for SLM inference.

        Args:
            current_turn_text: The current turn's ASR or corrected text.
            is_correction: If True, assemble a correction prompt instead.

        Returns:
            AssembledPrompt with the full prompt and token counts.
        """
        start_time = time.time()
        self._total_assemblies += 1

        if is_correction:
            return self._assemble_correction_prompt(current_turn_text, start_time, turn_id)
        else:
            return self._assemble_response_prompt(current_turn_text, start_time, turn_id)

    def _assemble_response_prompt(
        self, current_turn_text: str, start_time: float, turn_id: int = 0,
    ) -> AssembledPrompt:
        """Assemble a response generation prompt."""
        current_turn_tokens = estimate_token_count(current_turn_text)
        if current_turn_tokens > CURRENT_TURN_MAX_TOKENS:
            current_turn_text = self._truncate_to_tokens(
                current_turn_text, CURRENT_TURN_MAX_TOKENS
            )
            current_turn_tokens = CURRENT_TURN_MAX_TOKENS
            self._total_truncations += 1

        available_for_context = (
            self._model_context_window
            - self._system_prompt_tokens
            - current_turn_tokens
            - RESPONSE_RESERVED_TOKENS
        )
        available_for_context = min(available_for_context, CONTEXT_WINDOW_MAX_TOKENS)
        available_for_context = max(available_for_context, 0)

        # Retrieve context from Layer 2
        context_query = ContextQuery(max_tokens=available_for_context)
        context_window = self._context_manager.get_context(context_query)

        # Build the prompt parts
        parts = []

        # System prompt section (interpolate hot-words per context-injection.md §2)
        parts.append("[SYSTEM]")
        system_prompt_rendered = self._system_prompt.format(
            hot_words=self._hot_word_text or "(none)"
        )
        parts.append(system_prompt_rendered)
        parts.append("[/SYSTEM]")

        # Context turns (formatted per memory-and-persistence.md Section 4)
        for turn in context_window.turns:
            parts.append(turn.format_for_context())

        # Current user turn
        parts.append("[User] " + current_turn_text)

        # Assistant response marker
        parts.append("[Assistant]")

        full_prompt = "\n".join(parts)
        total_tokens = (
            self._system_prompt_tokens
            + context_window.total_tokens
            + current_turn_tokens
        )

        assembly_time_ms = (time.time() - start_time) * 1000

        ctx_hash = self._context_manager.context_hash()

        return AssembledPrompt(
            full_prompt=full_prompt,
            system_prompt_tokens=self._system_prompt_tokens,
            context_tokens=context_window.total_tokens,
            current_turn_tokens=current_turn_tokens,
            total_tokens=total_tokens,
            assembly_time_ms=assembly_time_ms,
            hot_words_injected=len(self._hot_words),
            turn_id=turn_id,
            prompt_version=self._prompt_version,
            context_hash=ctx_hash,
        )

    def _assemble_correction_prompt(
        self, raw_asr_text: str, start_time: float, turn_id: int = 0,
    ) -> AssembledPrompt:
        """Assemble a correction prompt for accent error fixing."""
        correction_tokens = estimate_token_count(CORRECTION_SYSTEM_PROMPT)

        # Get last 2 committed sentences for context
        recent_turns = self._context_manager.get_last_n_turns(2)
        context_text = " ".join(t.text for t in recent_turns)
        context_tokens = estimate_token_count(context_text)

        current_turn_tokens = estimate_token_count(raw_asr_text)

        # Build correction prompt
        parts = []
        parts.append("[SYSTEM]")
        parts.append(CORRECTION_SYSTEM_PROMPT)
        if self._hot_word_text:
            parts.append("Hot words: " + self._hot_word_text)
        parts.append("[/SYSTEM]")

        if context_text:
            parts.append("Context: " + context_text)

        parts.append("Raw ASR: " + raw_asr_text)
        parts.append("Corrected:")

        full_prompt = "\n".join(parts)
        total_tokens = correction_tokens + context_tokens + current_turn_tokens

        assembly_time_ms = (time.time() - start_time) * 1000

        return AssembledPrompt(
            full_prompt=full_prompt,
            system_prompt_tokens=correction_tokens,
            context_tokens=context_tokens,
            current_turn_tokens=current_turn_tokens,
            total_tokens=total_tokens,
            assembly_time_ms=assembly_time_ms,
            hot_words_injected=len(self._hot_words),
            turn_id=turn_id,
            prompt_version=self._prompt_version,
            context_hash=self._context_manager.context_hash(),
        )

    def _format_hot_words(self) -> str:
        """Format hot-words as a comma-separated string."""
        if not self._hot_words:
            return ""
        return ", ".join(hw.term for hw in self._hot_words)

    @staticmethod
    def _truncate_to_tokens(text: str, max_tokens: int) -> str:
        """Truncate text to fit within a token budget."""
        from .context_manager import CHARS_PER_TOKEN_ESTIMATE
        max_chars = max_tokens * CHARS_PER_TOKEN_ESTIMATE
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rsplit(" ", 1)[0] + "..."

    def get_stats(self) -> dict:
        """Get context injection statistics."""
        return {
            "total_assemblies": self._total_assemblies,
            "total_truncations": self._total_truncations,
            "system_prompt_tokens": self._system_prompt_tokens,
            "hot_word_count": len(self._hot_words),
            "model_context_window": self._model_context_window,
        }
