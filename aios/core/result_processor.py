"""
Result processor for PersonaPlex ASR streaming results.

Handles classification, routing, and buffering of partial and final results
with stability-based correction decisions.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional, AsyncIterator, Callable, Awaitable
from collections import deque

from .models import (
    ASRResult,
    ResultType,
    CorrectionLevel,
    StreamingMetadata,
    LatencyMetrics
)

logger = logging.getLogger(__name__)


@dataclass
class ProcessedResult:
    """
    A result that has been processed for downstream consumption.

    Includes the original ASR result plus metadata about how it should be handled.
    """
    result: ASRResult
    correction_level: CorrectionLevel
    should_display: bool = True
    is_ghost_text: bool = False  # True for partial, False for final

    @property
    def text(self) -> str:
        return self.result.text


@dataclass
class ResultBuffer:
    """
    Buffer for managing streaming results and ghost-text updates.

    Handles the replacement of ghost-text (partial) with solid-text (final).
    """
    _buffer: deque[ProcessedResult] = field(default_factory=lambda: deque(maxlen=50))
    _last_final_text: str = ""
    _last_partial_text: str = ""

    def add_partial(self, result: ASRResult) -> Optional[ProcessedResult]:
        """
        Add a partial result to the buffer.

        Returns a ProcessedResult if it should be forwarded downstream.
        """
        processed = ProcessedResult(
            result=result,
            correction_level=result.correction_level,
            is_ghost_text=True
        )

        self._last_partial_text = result.text

        # Only forward if correction level allows it
        if result.should_correct():
            self._buffer.append(processed)
            return processed

        return None

    def add_final(self, result: ASRResult) -> Optional[ProcessedResult]:
        """
        Add a final result to the buffer.

        Returns a ProcessedResult that should be committed to context.
        """
        processed = ProcessedResult(
            result=result,
            correction_level=CorrectionLevel.COMMITTED,
            is_ghost_text=False
        )

        self._last_final_text = result.text
        self._last_partial_text = ""  # Clear partial on final
        self._buffer.append(processed)

        return processed

    def get_current_display_text(self) -> str:
        """
        Get the text that should currently be displayed.
        Prefers final text, falls back to partial.
        """
        return self._last_final_text or self._last_partial_text

    def clear(self):
        """Clear the buffer at the start of a new utterance."""
        self._buffer.clear()
        self._last_final_text = ""
        self._last_partial_text = ""


class ResultProcessor:
    """
    Processes PersonaPlex streaming results and routes them appropriately.

    Responsibilities:
    - Classify results as partial/final
    - Determine correction level based on stability
    - Route results to appropriate queues (ghost-text vs solid-text)
    - Track latency metrics
    """

    def __init__(
        self,
        result_queue: asyncio.Queue[ProcessedResult],
        enable_latency_tracking: bool = True
    ):
        """
        Initialize the result processor.

        Args:
            result_queue: Queue to forward processed results to
            enable_latency_tracking: Whether to track latency metrics
        """
        self.result_queue = result_queue
        self.enable_latency_tracking = enable_latency_tracking

        self._buffer = ResultBuffer()
        self._metadata: Optional[StreamingMetadata] = None
        self._metrics: Optional[LatencyMetrics] = None

        # Callbacks for different result types
        self._on_partial: Optional[Callable[[ProcessedResult], Awaitable[None]]] = None
        self._on_final: Optional[Callable[[ProcessedResult], Awaitable[None]]] = None

        logger.debug("ResultProcessor initialized")

    def set_callbacks(
        self,
        on_partial: Optional[Callable[[ProcessedResult], Awaitable[None]]] = None,
        on_final: Optional[Callable[[ProcessedResult], Awaitable[None]]] = None
    ):
        """Set callbacks for partial and final results."""
        self._on_partial = on_partial
        self._on_final = on_final

    def start_session(self, metadata: StreamingMetadata):
        """Start a new streaming session."""
        self._metadata = metadata
        self._metrics = LatencyMetrics()
        self._buffer.clear()
        logger.debug(f"ResultProcessor started session {metadata.request_id}")

    def end_session(self):
        """End the current streaming session."""
        self._metadata = None
        logger.debug("ResultProcessor ended session")

    async def process_result(self, result: ASRResult) -> Optional[ProcessedResult]:
        """
        Process a single ASR result.

        Routes to appropriate handler based on result type and stability.

        Args:
            result: The ASR result to process

        Returns:
            ProcessedResult if it should be forwarded, None otherwise
        """
        if self._metadata:
            self._metadata.results_received += 1
            if result.is_final:
                self._metadata.final_results_received += 1

        # Route based on result type
        if result.is_final:
            processed = self._handle_final(result)
        else:
            processed = self._handle_partial(result)

        # Forward to queue if processing produced a result
        if processed and not self.result_queue.full():
            try:
                await asyncio.wait_for(
                    self.result_queue.put(processed),
                    timeout=0.1
                )
            except asyncio.TimeoutError:
                logger.warning("Result queue full, dropping result")

        # Trigger callbacks
        if processed:
            if result.is_final and self._on_final:
                await self._on_final(processed)
            elif not result.is_final and self._on_partial:
                await self._on_partial(processed)

        return processed

    def _handle_partial(self, result: ASRResult) -> Optional[ProcessedResult]:
        """Handle a partial (interim) result."""
        processed = self._buffer.add_partial(result)

        if processed:
            logger.debug(
                f"Partial result (stability={result.stability:.2f}): "
                f"'{result.text[:50]}...'"
            )

        return processed

    def _handle_final(self, result: ASRResult) -> Optional[ProcessedResult]:
        """Handle a final result."""
        processed = self._buffer.add_final(result)

        logger.info(
            f"Final result (confidence={result.confidence:.2f}): "
            f"'{result.text}'"
        )

        return processed

    def get_current_display_text(self) -> str:
        """Get the current display text from the buffer."""
        return self._buffer.get_current_display_text()

    @property
    def metadata(self) -> Optional[StreamingMetadata]:
        """Get the current session metadata."""
        return self._metadata

    @property
    def metrics(self) -> Optional[LatencyMetrics]:
        """Get the current latency metrics."""
        return self._metrics


# Factory function for creating result processor with queue
def create_result_processor(
    queue_size: int = 100,
    enable_latency_tracking: bool = True
) -> tuple[ResultProcessor, asyncio.Queue[ProcessedResult]]:
    """
    Create a result processor with its result queue.

    Returns:
        Tuple of (ResultProcessor, result_queue)
    """
    result_queue = asyncio.Queue(maxsize=queue_size)
    processor = ResultProcessor(
        result_queue=result_queue,
        enable_latency_tracking=enable_latency_tracking
    )
    return processor, result_queue
