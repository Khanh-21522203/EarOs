#!/usr/bin/env python3
"""
AIOS - Speech-Native AI Operating System

Main entry point. Initializes the 5-layer pipeline and runs the
real-time speech-to-speech interaction loop.

Usage:
    python main.py                        # Run with audio hardware
    python main.py --no-audio             # Run without audio (testing mode)
    python main.py --hot-words hw.json    # Load hot-words from file
    python main.py --metrics              # Enable Prometheus metrics on :9090
    python main.py --debug                # Enable debug mode (audio dumps, verbose logs)
"""

import argparse
import asyncio
import gc
import json
import logging
import os
import signal
import sys
from pathlib import Path

from aios.core.pipeline import PipelineOrchestrator, create_pipeline
from aios.core.context_injection import HotWord
from aios.core.state_machine import ConversationState
from aios.debugging.logging_config import configure_logging
from aios.debugging.health import HealthCheckServer, run_startup_probes
from aios.performance.metrics import get_metrics

logger = logging.getLogger("aios.main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AIOS Speech-to-Speech Runtime")
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable audio hardware (testing mode)",
    )
    parser.add_argument(
        "--no-aec",
        action="store_true",
        help="Disable acoustic echo cancellation",
    )
    parser.add_argument(
        "--hot-words",
        type=str,
        default=None,
        help="Path to hot-words JSON file",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Path to system prompt text file or inline prompt",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (audio dumps, verbose logging)",
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Enable Prometheus metrics server on :9090",
    )
    parser.add_argument(
        "--health-port",
        type=int,
        default=8080,
        help="Health check server port (default: 8080)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to aios.yaml config file",
    )
    return parser.parse_args()


def load_hot_words(path: str) -> list[HotWord]:
    """Load hot-words from a JSON file (array of {term, phonetic_variants})."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
        hot_words = []
        # Support both array format and {hot_words: [...]} format
        entries = data if isinstance(data, list) else data.get("hot_words", [])
        for entry in entries:
            hot_words.append(HotWord(
                term=entry["term"],
                phonetic_variants=entry.get("phonetic_variants", []),
            ))
        logger.info("Loaded %d hot-words from %s", len(hot_words), path)
        return hot_words
    except Exception as e:
        logger.error("Failed to load hot-words from %s: %s", path, e)
        return []


def load_system_prompt(path_or_text: str) -> str:
    """Load system prompt from file or return as-is if not a file path."""
    p = Path(path_or_text)
    if p.exists() and p.is_file():
        return p.read_text().strip()
    return path_or_text


def configure_gc():
    """Configure garbage collector for real-time audio performance."""
    gc.set_threshold(700, 10, 0)
    logger.info("GC configured: threshold=(700, 10, 0)")


async def run_pipeline(args: argparse.Namespace):
    """Initialize and run the AIOS pipeline."""
    # Run startup probes
    probes = run_startup_probes(
        require_gpu=not args.no_audio,
        require_audio=not args.no_audio,
    )
    logger.info("Startup probes: %s", probes)

    # Load hot-words
    hot_words = None
    if args.hot_words:
        hot_words = load_hot_words(args.hot_words)

    # Load system prompt
    system_prompt = None
    if args.system_prompt:
        system_prompt = load_system_prompt(args.system_prompt)

    # Set debug env var if --debug flag
    if args.debug:
        os.environ["AIOS_DEBUG"] = "1"

    # Enable metrics
    if args.metrics:
        metrics = get_metrics()
        metrics.start_server()

    # Create the pipeline
    pipeline = create_pipeline(
        system_prompt=system_prompt,
        hot_words=hot_words,
        enable_audio=not args.no_audio,
        enable_aec=not args.no_aec,
        enable_metrics=args.metrics,
        enable_debug=args.debug,
    )

    # Start health check server
    health_server = HealthCheckServer(
        port=args.health_port,
        status_callback=pipeline.get_status,
    )
    await health_server.start()

    # Setup graceful shutdown
    shutdown_event = asyncio.Event()

    def signal_handler():
        logger.info("Shutdown signal received")
        shutdown_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    # Start the pipeline
    logger.info("=" * 60)
    logger.info("AIOS Speech-to-Speech Runtime starting...")
    logger.info("=" * 60)

    try:
        await pipeline.start()
        health_server.set_ready(True)

        logger.info("Pipeline running. State: %s", pipeline.state_machine.state.value)
        logger.info("Health: http://0.0.0.0:%d/health", args.health_port)
        if args.metrics:
            logger.info("Metrics: http://0.0.0.0:9090/metrics")
        logger.info("Press Ctrl+C to stop.")

        # Main loop — wait for shutdown signal
        while not shutdown_event.is_set():
            await asyncio.sleep(1.0)

            # Periodic status log
            status = pipeline.get_status()
            state = status["state"]
            turn = status["turn_id"]
            tasks = status["tasks_alive"]
            logger.debug(
                "Status: state=%s turn=%d tasks=%d/%d",
                state, turn, tasks, status["tasks_total"],
            )

    except Exception as e:
        logger.error("Pipeline error: %s", e, exc_info=True)
    finally:
        logger.info("Shutting down pipeline...")
        health_server.set_ready(False)
        await pipeline.stop()
        await health_server.stop()
        logger.info("AIOS stopped.")


def main():
    args = parse_args()

    # Configure structured logging
    configure_logging(
        level="DEBUG" if args.debug else "INFO",
        json_format=not args.debug,  # Human-readable in debug mode
        non_blocking=True,
    )

    configure_gc()

    logger.info("Python %s", sys.version)
    logger.info("AIOS v0.1.0-MVP")

    try:
        asyncio.run(run_pipeline(args))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")


if __name__ == "__main__":
    main()
