"""
Test Fixtures for AIOS

Provides pytest fixtures for unit and integration tests:
- Pipeline with mock models (no GPU, no audio)
- Pre-configured state machine
- Synthetic audio sequences
- Mock ASR results
"""

import asyncio
from typing import List, Optional

from ..core.state_machine import EventBus, StateMachine, create_state_machine
from ..core.context_manager import ContextManager
from ..core.context_injection import ContextInjectionEngine, HotWord
from ..core.streaming_correction import StreamingCorrectionEngine
from ..core.pipeline import PipelineOrchestrator, PipelineQueues
from ..interfaces.models import (
    ASRResult,
    MockASRModel,
    MockSLMModel,
    MockTTSModel,
    ModelManager,
)
from ..performance.latency import LatencyTracer
from ..performance.degradation import DegradationManager


def create_test_pipeline(
    system_prompt: Optional[str] = None,
    hot_words: Optional[List[HotWord]] = None,
    slm_response_tokens: Optional[List[str]] = None,
) -> PipelineOrchestrator:
    """
    Create a fully wired pipeline with mock models for testing.

    No GPU, no audio hardware, no gRPC connections required.
    """
    return PipelineOrchestrator(
        system_prompt=system_prompt,
        hot_words=hot_words,
        enable_audio=False,
        enable_aec=False,
    )


def create_test_model_manager(
    asr_results: Optional[List[ASRResult]] = None,
    slm_tokens: Optional[List[str]] = None,
) -> ModelManager:
    """Create a ModelManager with mock models."""
    mm = ModelManager()
    mm.set_primary_models(
        asr=MockASRModel(results=asr_results),
        slm=MockSLMModel(response_tokens=slm_tokens),
        tts=MockTTSModel(),
    )
    return mm


def create_test_context(turns: Optional[List[dict]] = None) -> ContextManager:
    """
    Create a ContextManager pre-populated with test turns.

    Args:
        turns: List of dicts with keys: role, text, interrupted (optional).
    """
    cm = ContextManager()
    if turns:
        for t in turns:
            cm.add_turn(
                role=t["role"],
                text=t["text"],
                interrupted=t.get("interrupted", False),
            )
    return cm


def create_mock_asr_results(texts: List[str]) -> List[ASRResult]:
    """Create a list of mock ASR results from text strings."""
    results = []
    for i, text in enumerate(texts):
        is_final = (i == len(texts) - 1)
        results.append(ASRResult(
            text=text,
            is_final=is_final,
            stability=1.0 if is_final else 0.5,
            confidence=0.95 if is_final else 0.7,
        ))
    return results
