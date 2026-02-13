"""
Layer 3: Pipeline Orchestrator

Coordinates the end-to-end processing pipeline:
  Mic -> Capture -> VAD -> ASR -> Correction -> Context Injection -> SLM -> TTS -> Speaker

Manages streaming overlaps between stages, queue routing, task lifecycle,
and barge-in handling via task cancellation.

Integrates:
- Model abstractions (ASRModel, SLMModel, TTSModel) via ModelManager
- Per-stage latency tracing via LatencyTracer
- Graceful degradation via DegradationManager
- Prometheus metrics via MetricsCollector
- Audio debug dumps via AudioDumper
- VAD barge-in mode toggling based on state machine state

Concurrency boundary: Runs on the main asyncio event loop. GPU inference calls
(ASR, SLM, TTS) are dispatched via loop.run_in_executor(thread_pool, ...) to a
shared ThreadPoolExecutor(max_workers=3).

Performance constraints:
- Queue depth per stage: 10 items max (backpressure via Queue(maxsize=N))
- Pipeline startup: <= 100ms from IDLE to first ASR frame dispatched
- Streaming overlap: TTS begins before SLM completes (token-by-token forwarding)
"""

import asyncio
import gc
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Callable, Awaitable

from .state_machine import (
    StateMachine,
    EventBus,
    Event,
    EventType,
    ConversationState,
    create_state_machine,
)
from .audio_io import AudioIOKernel
from .vad import VoiceActivityDetector
from .context_manager import ContextManager
from .context_injection import ContextInjectionEngine, HotWord
from .streaming_correction import StreamingCorrectionEngine, CorrectedText

logger = logging.getLogger(__name__)

# Queue depths from spec
CAPTURE_QUEUE_DEPTH = 50
SPEECH_QUEUE_DEPTH = 50
ASR_QUEUE_DEPTH = 10
CORRECTED_QUEUE_DEPTH = 10
PROMPT_QUEUE_DEPTH = 5
RESPONSE_QUEUE_DEPTH = 20
PLAYBACK_QUEUE_DEPTH = 50

# Timeouts
ASR_TIMEOUT_S = 2.0
SLM_TIMEOUT_S = 3.0
TTS_TIMEOUT_S = 2.0
TURN_TIMEOUT_S = 10.0


class PipelineQueues:
    """All inter-stage queues with bounded capacity."""

    def __init__(self):
        self.capture: asyncio.Queue[bytes] = asyncio.Queue(maxsize=CAPTURE_QUEUE_DEPTH)
        self.speech: asyncio.Queue[bytes] = asyncio.Queue(maxsize=SPEECH_QUEUE_DEPTH)
        self.asr: asyncio.Queue[dict] = asyncio.Queue(maxsize=ASR_QUEUE_DEPTH)
        self.corrected: asyncio.Queue[CorrectedText] = asyncio.Queue(maxsize=CORRECTED_QUEUE_DEPTH)
        self.prompt: asyncio.Queue[str] = asyncio.Queue(maxsize=PROMPT_QUEUE_DEPTH)
        self.response: asyncio.Queue[str] = asyncio.Queue(maxsize=RESPONSE_QUEUE_DEPTH)
        self.playback: asyncio.Queue[bytes] = asyncio.Queue(maxsize=PLAYBACK_QUEUE_DEPTH)

    def flush_downstream(self):
        """Flush all downstream queues (for barge-in)."""
        for q in [self.asr, self.corrected, self.prompt, self.response, self.playback]:
            while not q.empty():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    break

    def get_depths(self) -> dict:
        """Get current depth of all queues."""
        return {
            "capture": self.capture.qsize(),
            "speech": self.speech.qsize(),
            "asr": self.asr.qsize(),
            "corrected": self.corrected.qsize(),
            "prompt": self.prompt.qsize(),
            "response": self.response.qsize(),
            "playback": self.playback.qsize(),
        }


class PipelineOrchestrator:
    """
    Pipeline Orchestrator -- Layer 3 of the AIOS architecture.

    Connects all pipeline stages and manages their lifecycle:
    - Audio I/O Kernel (Layer 1) for capture/playback
    - VAD for speech detection
    - ASR (PersonaPlex) for transcription
    - Streaming Correction for accent error fixing
    - Context Injection Engine (Layer 4) for prompt assembly
    - SLM for response generation
    - TTS for speech synthesis
    - State Machine (Layer 5) for coordination

    Integrations:
    - ModelManager for model lifecycle and fallback
    - LatencyTracer for per-stage latency instrumentation
    - DegradationManager for graceful degradation (L0-L4)
    - MetricsCollector for Prometheus metrics
    - AudioDumper for debug audio WAV dumps

    All GPU inference is dispatched to a ThreadPoolExecutor(max_workers=3).
    """

    def __init__(
        self,
        state_machine: Optional[StateMachine] = None,
        event_bus: Optional[EventBus] = None,
        context_manager: Optional[ContextManager] = None,
        hot_words: Optional[List[HotWord]] = None,
        system_prompt: Optional[str] = None,
        enable_audio: bool = True,
        enable_aec: bool = True,
        thread_pool_workers: int = 3,
        model_manager=None,
        latency_tracer=None,
        degradation_manager=None,
        metrics_collector=None,
        audio_dumper=None,
    ):
        # Create state machine and event bus if not provided
        if state_machine is None or event_bus is None:
            self._state_machine, self._event_bus = create_state_machine()
        else:
            self._state_machine = state_machine
            self._event_bus = event_bus

        # Queues
        self._queues = PipelineQueues()

        # Layer 2: Context Manager
        self._context_manager = context_manager or ContextManager()

        # Layer 4: Context Injection Engine
        self._context_injection = ContextInjectionEngine(
            context_manager=self._context_manager,
            system_prompt=system_prompt,
            hot_words=hot_words,
        )

        # Streaming Correction Engine
        self._correction_engine = StreamingCorrectionEngine(
            corrected_queue=self._queues.corrected,
        )

        # Layer 1: Audio I/O Kernel
        self._enable_audio = enable_audio
        self._audio_kernel: Optional[AudioIOKernel] = None
        if enable_audio:
            self._audio_kernel = AudioIOKernel(
                capture_queue=self._queues.capture,
                playback_queue=self._queues.playback,
                enable_aec=enable_aec,
            )

        # VAD
        self._vad = VoiceActivityDetector(
            speech_queue=self._queues.speech,
            event_bus=self._event_bus,
        )

        # Thread pool for GPU inference
        self._thread_pool = ThreadPoolExecutor(
            max_workers=thread_pool_workers,
            thread_name_prefix="aios-gpu",
        )

        # Model Manager (optional — enables real model inference)
        self._model_manager = model_manager

        # Performance instrumentation (optional)
        self._latency_tracer = latency_tracer
        self._degradation_manager = degradation_manager
        self._metrics = metrics_collector
        self._audio_dumper = audio_dumper

        # Pipeline tasks
        self._tasks: List[asyncio.Task] = []
        self._running = False

        # Turn tracking
        self._current_turn_text = ""
        self._current_raw_asr_text = ""
        self._current_response_text = ""
        self._turn_start_time: float = 0.0

        # Register state change callbacks
        self._state_machine.on_state_change(self._on_state_change)
        self._state_machine.on_flush(self._on_flush)

        logger.info("PipelineOrchestrator initialized")

    @property
    def state_machine(self) -> StateMachine:
        return self._state_machine

    @property
    def event_bus(self) -> EventBus:
        return self._event_bus

    @property
    def context_manager(self) -> ContextManager:
        return self._context_manager

    @property
    def queues(self) -> PipelineQueues:
        return self._queues

    @property
    def model_manager(self):
        return self._model_manager

    @property
    def latency_tracer(self):
        return self._latency_tracer

    @property
    def degradation_manager(self):
        return self._degradation_manager

    async def start(self):
        """Start the pipeline orchestrator and all sub-components."""
        self._running = True

        # Start state machine
        await self._state_machine.start()

        # Start audio I/O
        if self._audio_kernel:
            try:
                await self._audio_kernel.start()
            except Exception as e:
                logger.warning("Audio I/O failed to start: %s. Running without audio.", e)
                self._audio_kernel = None

        # Start pipeline stage tasks
        self._tasks = [
            asyncio.create_task(self._vad_loop(), name="vad_loop"),
            asyncio.create_task(self._asr_loop(), name="asr_loop"),
            asyncio.create_task(self._correction_loop(), name="correction_loop"),
            asyncio.create_task(self._response_loop(), name="response_loop"),
            asyncio.create_task(self._tts_loop(), name="tts_loop"),
            asyncio.create_task(self._queue_monitor_loop(), name="queue_monitor"),
        ]

        logger.info("Pipeline started with %d stage tasks", len(self._tasks))

    async def stop(self):
        """Stop the pipeline and all sub-components."""
        self._running = False

        # Cancel all tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to finish
        for task in self._tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._tasks.clear()

        # Stop sub-components
        if self._audio_kernel:
            await self._audio_kernel.stop()

        await self._state_machine.stop()

        # Shutdown thread pool
        self._thread_pool.shutdown(wait=False)

        # Run GC during shutdown
        gc.collect()

        logger.info("Pipeline stopped")

    # ── State change handler ──

    async def _on_state_change(
        self,
        old_state: ConversationState,
        new_state: ConversationState,
        event: Event,
    ):
        """Handle state transitions from the state machine."""
        # Report to metrics
        if self._metrics:
            self._metrics.record_state_transition(
                old_state.value, new_state.value, event.type.value,
            )

        if new_state == ConversationState.IDLE:
            # Turn complete — commit context, end trace, run GC
            if old_state in (ConversationState.SPEAKING, ConversationState.PROCESSING):
                self._commit_turn()
            if self._latency_tracer:
                self._latency_tracer.end_turn(self._state_machine.turn_id)
            if self._audio_dumper:
                self._audio_dumper.flush_turn(self._state_machine.turn_id)
            # Attempt degradation recovery between turns
            if self._degradation_manager:
                self._degradation_manager.attempt_recovery()
            # Attempt model recovery between turns
            if self._model_manager:
                await self._model_manager.attempt_recovery()
            # Disable barge-in mode
            self._vad.set_barge_in_mode(False)
            gc.collect()

        elif new_state == ConversationState.LISTENING:
            # New turn starting
            self._current_turn_text = ""
            self._current_raw_asr_text = ""
            self._current_response_text = ""
            self._turn_start_time = time.monotonic()
            self._correction_engine.reset()
            self._vad.reset()
            self._vad.set_barge_in_mode(False)
            if self._latency_tracer:
                self._latency_tracer.start_turn(self._state_machine.turn_id)

        elif new_state == ConversationState.SPEAKING:
            # Enable barge-in detection during system speech
            self._vad.set_barge_in_mode(True)

        elif new_state == ConversationState.INTERRUPTED:
            # Barge-in — record latency, cancel downstream
            if self._latency_tracer:
                barge_in_ms = (time.monotonic() - self._turn_start_time) * 1000
                self._latency_tracer.record_barge_in(barge_in_ms)
            if self._metrics:
                self._metrics.record_barge_in_latency(
                    (time.monotonic() - self._turn_start_time) * 1000
                )
            await self._cancel_downstream()

    async def _on_flush(self):
        """Flush all downstream queues on barge-in."""
        self._queues.flush_downstream()
        if self._audio_kernel:
            self._audio_kernel.flush_playback()
        logger.info("Pipeline flushed (barge-in)")

    def _commit_turn(self):
        """Commit the current turn to conversation history."""
        turn_duration_ms = (time.monotonic() - self._turn_start_time) * 1000 if self._turn_start_time else 0.0

        if self._current_turn_text:
            self._context_manager.add_turn(
                role="user",
                text=self._current_turn_text,
                raw_asr=self._current_raw_asr_text,
                turn_id=self._state_machine.turn_id,
                duration_ms=turn_duration_ms,
            )
        if self._current_response_text:
            interrupted = (
                self._state_machine.state == ConversationState.INTERRUPTED
            )
            self._context_manager.add_turn(
                role="assistant",
                text=self._current_response_text,
                interrupted=interrupted,
                turn_id=self._state_machine.turn_id,
            )

    async def _cancel_downstream(self):
        """Cancel active SLM and TTS tasks."""
        for task in self._tasks:
            if task.get_name() in ("response_loop", "tts_loop"):
                if not task.done():
                    task.cancel()
                    try:
                        await asyncio.wait_for(asyncio.shield(task), timeout=0.05)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass

    # ── Pipeline stage loops ──

    async def _vad_loop(self):
        """Stage 2: VAD — classify capture frames as speech/non-speech."""
        try:
            while self._running:
                try:
                    frame = await asyncio.wait_for(
                        self._queues.capture.get(), timeout=0.05
                    )
                    turn_id = self._state_machine.turn_id

                    if self._latency_tracer:
                        self._latency_tracer.start_span("vad", turn_id)

                    if self._audio_dumper:
                        self._audio_dumper.record_capture(turn_id, frame)

                    await self._vad.process_frame_async(frame, turn_id=turn_id)

                    if self._latency_tracer:
                        vad_ms = self._latency_tracer.end_span("vad", turn_id)
                        if self._metrics:
                            self._metrics.record_turn_latency("vad", vad_ms)

                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            logger.debug("VAD loop cancelled")

    async def _asr_loop(self):
        """Stage 3: ASR — transcribe speech frames."""
        try:
            while self._running:
                try:
                    frame = await asyncio.wait_for(
                        self._queues.speech.get(), timeout=0.05
                    )
                    turn_id = self._state_machine.turn_id

                    if self._latency_tracer:
                        self._latency_tracer.start_span("asr", turn_id)

                    loop = asyncio.get_event_loop()
                    try:
                        result = await asyncio.wait_for(
                            loop.run_in_executor(
                                self._thread_pool,
                                self._asr_inference,
                                frame,
                            ),
                            timeout=ASR_TIMEOUT_S,
                        )

                        if self._latency_tracer:
                            asr_ms = self._latency_tracer.end_span("asr", turn_id)
                            if self._metrics:
                                self._metrics.record_turn_latency("asr", asr_ms)

                        if result:
                            if self._model_manager:
                                self._model_manager.report_asr_success()
                            await self._queues.asr.put(result)
                        else:
                            if self._latency_tracer:
                                self._latency_tracer.end_span("asr", turn_id)

                    except asyncio.TimeoutError:
                        logger.warning("ASR inference timeout (%.1fs)", ASR_TIMEOUT_S)
                        if self._model_manager:
                            self._model_manager.report_asr_failure()
                        if self._metrics:
                            self._metrics.record_error("asr")

                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            logger.debug("ASR loop cancelled")

    def _asr_inference(self, frame: bytes) -> Optional[dict]:
        """
        ASR inference (runs in thread pool).

        Uses ModelManager.asr if available, otherwise returns None.
        """
        # TODO: Integrate actual PersonaPlex gRPC streaming via ModelManager
        return None

    async def _correction_loop(self):
        """Stage 4: Correction — apply accent correction to ASR results."""
        try:
            while self._running:
                try:
                    asr_result = await asyncio.wait_for(
                        self._queues.asr.get(), timeout=0.05
                    )
                    text = asr_result.get("text", "")
                    is_final = asr_result.get("is_final", False)
                    stability = asr_result.get("stability", 0.0)
                    turn_id = self._state_machine.turn_id

                    # Check degradation: skip correction if L4
                    if self._degradation_manager and self._degradation_manager.skip_correction_entirely:
                        if is_final:
                            self._current_raw_asr_text += " " + text
                            self._current_turn_text += " " + text
                            self._current_turn_text = self._current_turn_text.strip()
                            self._current_raw_asr_text = self._current_raw_asr_text.strip()
                            prompt = self._context_injection.assemble_prompt(
                                current_turn_text=self._current_turn_text,
                                turn_id=turn_id,
                            )
                            try:
                                await self._queues.prompt.put(prompt.full_prompt)
                            except asyncio.QueueFull:
                                logger.warning("Prompt queue full")
                        continue

                    if self._latency_tracer:
                        self._latency_tracer.start_span("correction", turn_id)

                    # Check degradation: rule-based only if L1+
                    skip_slm = (
                        self._degradation_manager
                        and self._degradation_manager.skip_slm_correction
                    )

                    corrected = await self._correction_engine.process_asr_result(
                        text=text,
                        is_final=is_final,
                        stability=stability,
                        turn_id=turn_id,
                    )

                    if self._latency_tracer:
                        corr_ms = self._latency_tracer.end_span("correction", turn_id)
                        if self._degradation_manager:
                            self._degradation_manager.report_correction_latency(corr_ms)
                        if self._metrics:
                            self._metrics.record_turn_latency("correction", corr_ms)

                    if corrected and is_final:
                        self._current_raw_asr_text += " " + text
                        self._current_raw_asr_text = self._current_raw_asr_text.strip()
                        self._current_turn_text += " " + corrected.corrected
                        self._current_turn_text = self._current_turn_text.strip()

                        if self._latency_tracer:
                            self._latency_tracer.start_span("context_injection", turn_id)

                        prompt = self._context_injection.assemble_prompt(
                            current_turn_text=self._current_turn_text,
                            turn_id=turn_id,
                        )

                        if self._latency_tracer:
                            ci_ms = self._latency_tracer.end_span("context_injection", turn_id)
                            if self._metrics:
                                self._metrics.record_turn_latency("context_injection", ci_ms)

                        try:
                            await self._queues.prompt.put(prompt.full_prompt)
                        except asyncio.QueueFull:
                            logger.warning("Prompt queue full")

                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            logger.debug("Correction loop cancelled")

    async def _response_loop(self):
        """Stage 5: SLM Response Generation — generate response tokens."""
        try:
            while self._running:
                try:
                    prompt = await asyncio.wait_for(
                        self._queues.prompt.get(), timeout=0.05
                    )
                    turn_id = self._state_machine.turn_id

                    if self._latency_tracer:
                        self._latency_tracer.start_span("slm_response", turn_id)

                    loop = asyncio.get_event_loop()
                    try:
                        response_tokens = await asyncio.wait_for(
                            loop.run_in_executor(
                                self._thread_pool,
                                self._slm_inference,
                                prompt,
                            ),
                            timeout=SLM_TIMEOUT_S,
                        )

                        if self._latency_tracer:
                            slm_ms = self._latency_tracer.end_span("slm_response", turn_id)
                            if self._degradation_manager:
                                self._degradation_manager.report_slm_latency(slm_ms)
                            if self._metrics:
                                self._metrics.record_turn_latency("slm_response", slm_ms)

                        if response_tokens:
                            if self._model_manager:
                                self._model_manager.report_slm_success()

                            # Emit first_token event
                            await self._event_bus.publish(Event(
                                type=EventType.FIRST_TOKEN,
                                turn_id=turn_id,
                            ))

                            # Stream tokens to TTS
                            for token in response_tokens:
                                self._current_response_text += token
                                try:
                                    await self._queues.response.put(token)
                                except asyncio.QueueFull:
                                    logger.warning("Response queue full, SLM paused")
                                    await self._queues.response.put(token)

                            # Signal response complete
                            await self._event_bus.publish(Event(
                                type=EventType.RESPONSE_COMPLETE,
                                turn_id=turn_id,
                            ))

                    except asyncio.TimeoutError:
                        logger.warning("SLM inference timeout (%.1fs)", SLM_TIMEOUT_S)
                        if self._model_manager:
                            self._model_manager.report_slm_failure()
                        if self._metrics:
                            self._metrics.record_error("slm")
                        # Fallback response
                        fallback = "I didn't catch that."
                        self._current_response_text = fallback
                        await self._queues.response.put(fallback)

                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            logger.debug("Response loop cancelled")

    def _slm_inference(self, prompt: str) -> Optional[List[str]]:
        """
        SLM inference (runs in thread pool).

        Uses ModelManager.slm if available, otherwise returns None.
        """
        # TODO: Integrate actual SLM inference via ModelManager
        return None

    async def _tts_loop(self):
        """Stage 6: TTS — synthesize response tokens into audio."""
        try:
            sentence_buffer = ""
            while self._running:
                try:
                    token = await asyncio.wait_for(
                        self._queues.response.get(), timeout=0.05
                    )

                    sentence_buffer += token

                    # Accumulate tokens into sentence fragments (split on punctuation)
                    if any(p in token for p in ".!?,;:"):
                        turn_id = self._state_machine.turn_id

                        if self._latency_tracer:
                            self._latency_tracer.start_span("tts", turn_id)

                        loop = asyncio.get_event_loop()
                        try:
                            audio_frames = await asyncio.wait_for(
                                loop.run_in_executor(
                                    self._thread_pool,
                                    self._tts_inference,
                                    sentence_buffer,
                                ),
                                timeout=TTS_TIMEOUT_S,
                            )

                            if self._latency_tracer:
                                tts_ms = self._latency_tracer.end_span("tts", turn_id)
                                if self._degradation_manager:
                                    self._degradation_manager.report_tts_latency(tts_ms)
                                if self._metrics:
                                    self._metrics.record_turn_latency("tts", tts_ms)

                            if audio_frames:
                                if self._model_manager:
                                    self._model_manager.report_tts_success()

                                # Emit first_tts_frame event on first frame
                                first_frame = True
                                for frame in audio_frames:
                                    if first_frame:
                                        await self._event_bus.publish(Event(
                                            type=EventType.FIRST_TTS_FRAME,
                                            turn_id=turn_id,
                                        ))
                                        first_frame = False

                                    try:
                                        self._queues.playback.put_nowait(frame)
                                    except asyncio.QueueFull:
                                        try:
                                            self._queues.playback.get_nowait()
                                        except asyncio.QueueEmpty:
                                            pass
                                        self._queues.playback.put_nowait(frame)

                                    if self._audio_kernel:
                                        self._audio_kernel.feed_reference_signal(frame)

                                    if self._audio_dumper:
                                        self._audio_dumper.record_playback(turn_id, frame)

                        except asyncio.TimeoutError:
                            logger.warning("TTS inference timeout (%.1fs)", TTS_TIMEOUT_S)
                            if self._model_manager:
                                self._model_manager.report_tts_failure()
                            if self._metrics:
                                self._metrics.record_error("tts")

                        sentence_buffer = ""

                except asyncio.TimeoutError:
                    # Flush remaining buffer if we have text
                    if sentence_buffer.strip():
                        loop = asyncio.get_event_loop()
                        try:
                            audio_frames = await asyncio.wait_for(
                                loop.run_in_executor(
                                    self._thread_pool,
                                    self._tts_inference,
                                    sentence_buffer,
                                ),
                                timeout=TTS_TIMEOUT_S,
                            )
                            if audio_frames:
                                for frame in audio_frames:
                                    try:
                                        self._queues.playback.put_nowait(frame)
                                    except asyncio.QueueFull:
                                        pass
                        except asyncio.TimeoutError:
                            pass
                        sentence_buffer = ""
                    continue

        except asyncio.CancelledError:
            logger.debug("TTS loop cancelled")

    def _tts_inference(self, text: str) -> Optional[List[bytes]]:
        """
        TTS inference (runs in thread pool).

        Uses ModelManager.tts if available, otherwise returns None.
        """
        # TODO: Integrate actual TTS inference via ModelManager
        return None

    async def _queue_monitor_loop(self):
        """Monitor queue depths and emit metrics every 100ms."""
        try:
            while self._running:
                await asyncio.sleep(0.1)
                depths = self._queues.get_depths()

                # Report to Prometheus metrics
                if self._metrics:
                    for name, depth in depths.items():
                        self._metrics.set_queue_depth(name, depth)
                    if self._degradation_manager:
                        self._metrics.set_degradation_level(int(self._degradation_manager.level))

                # Check for multi-stage degradation
                if self._degradation_manager:
                    self._degradation_manager.check_multi_stage_failure()

                # Log warning if any queue is near capacity
                for name, depth in depths.items():
                    max_depth = getattr(
                        self._queues, name
                    ).maxsize if hasattr(getattr(self._queues, name), 'maxsize') else 0
                    if max_depth > 0 and depth > max_depth * 0.8:
                        logger.warning(
                            "Queue '%s' near capacity: %d/%d",
                            name, depth, max_depth,
                        )
        except asyncio.CancelledError:
            pass

    # ── Public API ──

    async def inject_audio(self, frame: bytes):
        """
        Inject an audio frame into the pipeline (for testing without audio hardware).

        Args:
            frame: Raw PCM bytes (16kHz, 16-bit, mono).
        """
        if self._queues.capture.full():
            try:
                self._queues.capture.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            self._queues.capture.put_nowait(frame)
        except asyncio.QueueFull:
            pass

    async def inject_text(self, text: str):
        """
        Inject text directly into the pipeline (bypasses ASR).

        Useful for testing the downstream pipeline without audio.
        """
        asr_result = {
            "text": text,
            "is_final": True,
            "stability": 1.0,
            "confidence": 1.0,
        }
        try:
            await self._queues.asr.put(asr_result)
        except asyncio.QueueFull:
            logger.warning("ASR queue full, cannot inject text")

    def get_status(self) -> dict:
        """Get pipeline status."""
        status = {
            "running": self._running,
            "state": self._state_machine.state.value,
            "turn_id": self._state_machine.turn_id,
            "queue_depths": self._queues.get_depths(),
            "state_machine": self._state_machine.get_status(),
            "vad": self._vad.get_stats(),
            "correction": self._correction_engine.get_stats(),
            "context": self._context_manager.get_stats(),
            "context_injection": self._context_injection.get_stats(),
            "tasks_alive": sum(1 for t in self._tasks if not t.done()),
            "tasks_total": len(self._tasks),
        }
        if self._model_manager:
            status["models"] = self._model_manager.get_status()
        if self._degradation_manager:
            status["degradation"] = self._degradation_manager.get_status()
        if self._latency_tracer:
            status["latency_histograms"] = self._latency_tracer.get_histograms()
        return status


def create_pipeline(
    system_prompt: Optional[str] = None,
    hot_words: Optional[List[HotWord]] = None,
    enable_audio: bool = True,
    enable_aec: bool = True,
    model_manager=None,
    enable_metrics: bool = False,
    enable_debug: bool = False,
) -> PipelineOrchestrator:
    """
    Factory function to create a fully wired pipeline.

    Args:
        system_prompt: Optional system prompt for the SLM.
        hot_words: Optional hot-word list.
        enable_audio: Enable audio I/O (disable for testing).
        enable_aec: Enable acoustic echo cancellation.
        model_manager: Optional ModelManager with loaded models.
        enable_metrics: Enable Prometheus metrics collection.
        enable_debug: Enable audio debug dumps.

    Returns:
        Configured PipelineOrchestrator.
    """
    from ..performance.latency import LatencyTracer
    from ..performance.degradation import DegradationManager

    latency_tracer = LatencyTracer()
    degradation_manager = DegradationManager()

    metrics_collector = None
    if enable_metrics:
        from ..performance.metrics import get_metrics
        metrics_collector = get_metrics()

    audio_dumper = None
    if enable_debug:
        from ..debugging.audio_dump import AudioDumper
        audio_dumper = AudioDumper(enabled=True)

    return PipelineOrchestrator(
        system_prompt=system_prompt,
        hot_words=hot_words,
        enable_audio=enable_audio,
        enable_aec=enable_aec,
        model_manager=model_manager,
        latency_tracer=latency_tracer,
        degradation_manager=degradation_manager,
        metrics_collector=metrics_collector,
        audio_dumper=audio_dumper,
    )
