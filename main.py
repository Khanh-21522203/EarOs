#!/usr/bin/env python3
"""
AIOS - Speech-Native AI Operating System

Main entry point. Initializes the 5-layer pipeline and runs the
real-time speech-to-speech interaction loop.

Usage:
    python main.py                    # Run with audio hardware
    python main.py --no-audio         # Run without audio (testing mode)
    python main.py --hot-words hw.json  # Load hot-words from file
"""

import argparse
import asyncio
import gc
import json
import logging
import signal
import sys
from pathlib import Path

from aios.core.pipeline import PipelineOrchestrator, create_pipeline
from aios.core.context_injection import HotWord
from aios.core.state_machine import ConversationState

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
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
        help="Custom system prompt for the SLM",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def load_hot_words(path: str) -> list[HotWord]:
    """Load hot-words from a JSON file."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
        hot_words = []
        for entry in data.get("hot_words", []):
            hot_words.append(HotWord(
                term=entry["term"],
                phonetic_variants=entry.get("phonetic_variants", []),
            ))
        logger.info("Loaded %d hot-words from %s", len(hot_words), path)
        return hot_words
    except Exception as e:
        logger.error("Failed to load hot-words from %s: %s", path, e)
        return []


def configure_gc():
    """Configure garbage collector for real-time audio performance."""
    # Disable generation 2 GC to prevent long pauses during active conversation
    gc.set_threshold(700, 10, 0)
    logger.info("GC configured: threshold=(700, 10, 0)")


async def run_pipeline(args: argparse.Namespace):
    """Initialize and run the AIOS pipeline."""
    # Load hot-words if specified
    hot_words = None
    if args.hot_words:
        hot_words = load_hot_words(args.hot_words)

    # Create the pipeline
    pipeline = create_pipeline(
        system_prompt=args.system_prompt,
        hot_words=hot_words,
        enable_audio=not args.no_audio,
        enable_aec=not args.no_aec,
    )

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

        logger.info("Pipeline running. State: %s", pipeline.state_machine.state.value)
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
        await pipeline.stop()
        logger.info("AIOS stopped.")


def main():
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    configure_gc()

    logger.info("Python %s", sys.version)
    logger.info("AIOS v0.1.0-mvp")

    try:
        asyncio.run(run_pipeline(args))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")


if __name__ == "__main__":
    main()
