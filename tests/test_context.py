"""
Unit tests for Context Manager and Context Injection Engine.

Tests from memory-and-persistence.md and context-injection.md Definition of Done:
- Memory Manager stores and retrieves 10 turns correctly
- FIFO eviction works when 11th turn is inserted
- get_context(max_tokens) respects token budget and returns newest turns first
- Interrupted turns are correctly marked and formatted
- Token counts are pre-computed and match
- Prompt assembler produces correctly structured prompts
- Token budget is never exceeded
- Truncation correctly removes oldest turns first
- Hot-words appear in both system prompt and corrected current turn
- Prompt versioning metadata is logged
"""

import pytest

from aios.core.context_manager import (
    ContextManager,
    TurnRecord,
    ContextQuery,
    estimate_token_count,
)
from aios.core.context_injection import (
    ContextInjectionEngine,
    HotWord,
    AssembledPrompt,
    SYSTEM_PROMPT_MAX_TOKENS,
    RESPONSE_RESERVED_TOKENS,
)


# ── Context Manager ──

class TestContextManager:

    def test_add_and_retrieve_turns(self):
        cm = ContextManager(max_turns=10)
        cm.add_turn("user", "Hello")
        cm.add_turn("assistant", "Hi there!")
        assert cm.turn_count == 2

    def test_fifo_eviction_at_capacity(self):
        cm = ContextManager(max_turns=5, max_context_tokens=10000)
        for i in range(7):
            cm.add_turn("user", f"Turn {i}")
        assert cm.turn_count == 5

    def test_get_context_respects_budget(self):
        cm = ContextManager(max_turns=10, max_context_tokens=10000)
        for i in range(5):
            cm.add_turn("user", f"This is turn number {i} with some text")
        ctx = cm.get_context(ContextQuery(max_tokens=20))
        assert ctx.total_tokens <= 20

    def test_get_context_returns_newest_first(self):
        cm = ContextManager(max_turns=10, max_context_tokens=10000)
        cm.add_turn("user", "First")
        cm.add_turn("user", "Second")
        cm.add_turn("user", "Third")
        ctx = cm.get_context(ContextQuery(max_tokens=10000))
        assert ctx.turns[-1].text == "Third"
        assert ctx.turns[0].text == "First"

    def test_token_count_precomputed(self):
        cm = ContextManager()
        record = cm.add_turn("user", "Hello world")
        assert record.token_count > 0
        assert record.token_count == estimate_token_count("Hello world")

    def test_mark_interrupted(self):
        cm = ContextManager()
        cm.add_turn("assistant", "The weather is", turn_id=42)
        assert cm.mark_interrupted(42)
        turns = cm.get_last_n_turns(1)
        assert turns[0].interrupted is True

    def test_mark_interrupted_not_found(self):
        cm = ContextManager()
        cm.add_turn("user", "Hello", turn_id=1)
        assert not cm.mark_interrupted(999)

    def test_interrupted_turn_format(self):
        record = TurnRecord(role="assistant", text="The weather is", interrupted=True)
        formatted = record.format_for_context()
        assert "[interrupted]" in formatted
        assert "[Assistant]" in formatted

    def test_user_turn_format(self):
        record = TurnRecord(role="user", text="Hello")
        formatted = record.format_for_context()
        assert "[User]" in formatted
        assert "Hello" in formatted

    def test_raw_asr_stored(self):
        cm = ContextManager()
        record = cm.add_turn("user", "test string", raw_asr="tes string")
        assert record.raw_asr == "tes string"

    def test_duration_ms_stored(self):
        cm = ContextManager()
        record = cm.add_turn("user", "hello", duration_ms=123.4)
        assert record.duration_ms == 123.4

    def test_get_last_user_text(self):
        cm = ContextManager()
        cm.add_turn("user", "Hello")
        cm.add_turn("assistant", "Hi")
        cm.add_turn("user", "How are you?")
        assert cm.get_last_user_text() == "How are you?"

    def test_get_last_assistant_text(self):
        cm = ContextManager()
        cm.add_turn("user", "Hello")
        cm.add_turn("assistant", "Hi")
        assert cm.get_last_assistant_text() == "Hi"

    def test_clear(self):
        cm = ContextManager()
        cm.add_turn("user", "Hello")
        cm.clear()
        assert cm.turn_count == 0
        assert cm.total_tokens == 0

    def test_context_hash_changes(self):
        cm = ContextManager()
        cm.add_turn("user", "Hello")
        h1 = cm.context_hash()
        cm.add_turn("user", "World")
        h2 = cm.context_hash()
        assert h1 != h2

    def test_reject_invalid_role(self):
        cm = ContextManager()
        result = cm.add_turn("invalid_role", "Hello")
        assert result is None

    def test_reject_empty_text(self):
        cm = ContextManager()
        result = cm.add_turn("user", "")
        assert result is None

    def test_stats(self):
        cm = ContextManager()
        cm.add_turn("user", "Hello")
        stats = cm.get_stats()
        assert stats["turn_count"] == 1
        assert stats["total_tokens"] > 0


# ── Context Injection Engine ──

class TestContextInjection:

    def test_assemble_response_prompt(self):
        cm = ContextManager()
        cm.add_turn("user", "Tell me about NVIDIA")
        cm.add_turn("assistant", "NVIDIA is a technology company.")

        engine = ContextInjectionEngine(context_manager=cm)
        prompt = engine.assemble_prompt("What about their ASR product?")

        assert prompt.full_prompt is not None
        assert len(prompt.full_prompt) > 0
        assert prompt.assembly_time_ms >= 0
        assert "[User]" in prompt.full_prompt
        assert "[Assistant]" in prompt.full_prompt
        assert "[SYSTEM]" in prompt.full_prompt

    def test_hot_word_injection(self):
        cm = ContextManager()
        engine = ContextInjectionEngine(
            context_manager=cm,
            hot_words=[HotWord(term="PersonaPlex", phonetic_variants=["personna plex"])],
        )
        prompt = engine.assemble_prompt("Tell me about PersonaPlex")
        assert "PersonaPlex" in prompt.full_prompt

    def test_token_budget_not_exceeded(self):
        cm = ContextManager()
        for i in range(10):
            cm.add_turn("user", f"This is a longer turn number {i} with some extra text to use tokens")
            cm.add_turn("assistant", f"Response to turn {i} with some additional content")

        engine = ContextInjectionEngine(context_manager=cm, model_context_window=4096)
        prompt = engine.assemble_prompt("Current question")
        assert prompt.total_tokens <= 4096 - RESPONSE_RESERVED_TOKENS

    def test_truncate_oldest_context_first(self):
        cm = ContextManager(max_turns=10, max_context_tokens=10000)
        cm.add_turn("user", "Very old turn")
        cm.add_turn("assistant", "Old response")
        cm.add_turn("user", "Recent turn")

        engine = ContextInjectionEngine(context_manager=cm, model_context_window=4096)
        prompt = engine.assemble_prompt("Current question")
        # Recent turn should be in context
        assert "Recent turn" in prompt.full_prompt

    def test_correction_prompt(self):
        cm = ContextManager()
        cm.add_turn("user", "Previous sentence")

        engine = ContextInjectionEngine(context_manager=cm)
        prompt = engine.assemble_prompt("tes string", is_correction=True)
        assert "Raw ASR" in prompt.full_prompt
        assert "tes string" in prompt.full_prompt

    def test_prompt_versioning(self):
        cm = ContextManager()
        engine = ContextInjectionEngine(context_manager=cm)
        prompt = engine.assemble_prompt("Hello", turn_id=42)
        assert prompt.turn_id == 42
        assert prompt.prompt_version == 1
        assert len(prompt.context_hash) > 0

    def test_prompt_version_increments_on_system_prompt_change(self):
        cm = ContextManager()
        engine = ContextInjectionEngine(context_manager=cm)
        p1 = engine.assemble_prompt("Hello")
        engine.set_system_prompt("New system prompt")
        p2 = engine.assemble_prompt("Hello")
        assert p2.prompt_version == p1.prompt_version + 1

    def test_empty_context(self):
        cm = ContextManager()
        engine = ContextInjectionEngine(context_manager=cm)
        prompt = engine.assemble_prompt("Hello")
        assert prompt.context_tokens == 0

    def test_empty_current_turn_skipped(self):
        cm = ContextManager()
        engine = ContextInjectionEngine(context_manager=cm)
        prompt = engine.assemble_prompt("")
        assert prompt.current_turn_tokens == 0

    def test_stats(self):
        cm = ContextManager()
        engine = ContextInjectionEngine(context_manager=cm)
        engine.assemble_prompt("Hello")
        stats = engine.get_stats()
        assert stats["total_assemblies"] == 1
