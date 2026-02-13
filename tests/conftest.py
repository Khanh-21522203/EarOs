"""
Shared pytest fixtures for AIOS tests.
"""

import asyncio
import pytest


@pytest.fixture
def event_loop():
    """Create a new event loop for each test."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
