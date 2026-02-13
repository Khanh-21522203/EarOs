"""
Structured Logging Configuration for AIOS

All logs use structured JSON format for machine parseability.
Non-blocking log handler ensures audio callback thread is never blocked.

Log levels:
- ERROR: Invalid state transitions, model failures, device errors
- WARN: Queue drops, latency threshold exceeded, degradation events
- INFO: State transitions, turn lifecycle events, degradation changes
- DEBUG: Queue depths, per-frame VAD scores, partial ASR results
- TRACE (=5): Individual audio frame timestamps, ring buffer indices

Component tags: audio_io, vad, asr, correction, context, slm, tts,
                state_machine, pipeline

Debug mode (AIOS_DEBUG=1):
- Full turn text in logs
- Audio frame dumps to /tmp/aios_debug/
- ASR partial/final result logging
"""

import json
import logging
import os
import queue
import sys
import threading
import time
from typing import Optional

# Custom TRACE level (below DEBUG)
TRACE = 5
logging.addLevelName(TRACE, "TRACE")

# Check debug mode
AIOS_DEBUG = os.environ.get("AIOS_DEBUG", "0") == "1"

# Log queue capacity for non-blocking handler
LOG_QUEUE_CAPACITY = 10_000


class StructuredFormatter(logging.Formatter):
    """Formats log records as structured JSON."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "ts": time.time(),
            "level": record.levelname,
            "component": getattr(record, "component", record.name.split(".")[-1]),
            "event": record.getMessage(),
        }

        # Add turn_id if present
        turn_id = getattr(record, "turn_id", None)
        if turn_id is not None:
            log_entry["turn_id"] = turn_id

        # Add extra fields
        if hasattr(record, "extra_fields") and record.extra_fields:
            log_entry.update(record.extra_fields)

        # Add exception info
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = str(record.exc_info[1])

        return json.dumps(log_entry, default=str)


class NonBlockingHandler(logging.Handler):
    """
    Non-blocking log handler using a bounded queue.

    The audio callback thread must never block on log writes.
    Log records are enqueued to a bounded queue. A background thread
    writes records to the target handler. If the queue is full,
    records are dropped (not blocking the producer).
    """

    def __init__(self, target_handler: logging.Handler, capacity: int = LOG_QUEUE_CAPACITY):
        super().__init__()
        self._queue: queue.Queue = queue.Queue(maxsize=capacity)
        self._target = target_handler
        self._dropped_count = 0
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._running = True
        self._thread.start()

    def emit(self, record: logging.LogRecord):
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            self._dropped_count += 1

    def _writer_loop(self):
        while self._running:
            try:
                record = self._queue.get(timeout=0.1)
                self._target.emit(record)
            except queue.Empty:
                continue
            except Exception:
                pass

    def close(self):
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        super().close()

    @property
    def dropped_count(self) -> int:
        return self._dropped_count


def configure_logging(
    level: Optional[str] = None,
    json_format: bool = True,
    non_blocking: bool = True,
) -> logging.Handler:
    """
    Configure AIOS structured logging.

    Args:
        level: Log level string. Defaults to AIOS_LOG_LEVEL env var or INFO.
        json_format: Use structured JSON format.
        non_blocking: Use non-blocking handler (recommended for production).

    Returns:
        The configured root handler.
    """
    log_level_str = level or os.environ.get("AIOS_LOG_LEVEL", "INFO")
    log_level_map = {
        "TRACE": TRACE,
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARN": logging.WARNING,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    log_level = log_level_map.get(log_level_str.upper(), logging.INFO)

    # If debug mode, force DEBUG level
    if AIOS_DEBUG:
        log_level = logging.DEBUG

    # Create stream handler
    stream_handler = logging.StreamHandler(sys.stdout)

    if json_format:
        stream_handler.setFormatter(StructuredFormatter())
    else:
        stream_handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        ))

    # Wrap in non-blocking handler if requested
    if non_blocking:
        handler = NonBlockingHandler(stream_handler)
    else:
        handler = stream_handler

    # Configure root logger
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level)

    # Suppress noisy third-party loggers
    for noisy in ["urllib3", "grpc", "asyncio"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return handler


class ComponentLogger:
    """
    Convenience wrapper for component-tagged logging.

    Usage:
        log = ComponentLogger("vad")
        log.info("speech_start detected", turn_id=42, extra={"confidence": 0.95})
    """

    def __init__(self, component: str):
        self._logger = logging.getLogger(f"aios.{component}")
        self._component = component

    def _log(self, level: int, msg: str, turn_id: Optional[int] = None, extra: Optional[dict] = None):
        record = self._logger.makeRecord(
            name=self._logger.name,
            level=level,
            fn="",
            lno=0,
            msg=msg,
            args=(),
            exc_info=None,
        )
        record.component = self._component
        if turn_id is not None:
            record.turn_id = turn_id
        record.extra_fields = extra or {}
        self._logger.handle(record)

    def trace(self, msg: str, **kwargs):
        self._log(TRACE, msg, **kwargs)

    def debug(self, msg: str, **kwargs):
        self._log(logging.DEBUG, msg, **kwargs)

    def info(self, msg: str, **kwargs):
        self._log(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs):
        self._log(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, **kwargs):
        self._log(logging.ERROR, msg, **kwargs)

    def critical(self, msg: str, **kwargs):
        self._log(logging.CRITICAL, msg, **kwargs)
