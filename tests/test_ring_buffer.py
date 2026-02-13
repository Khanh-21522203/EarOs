"""
Unit tests for SPSC Ring Buffer.

Tests:
- Read/write correctness
- Wrap-around behavior
- Overflow drops new frame (metric incremented)
- Underrun returns None (metric incremented)
"""

import pytest

from aios.core.ring_buffer import SPSCRingBuffer, create_capture_ring_buffer, create_playback_ring_buffer


class TestSPSCRingBuffer:

    def test_write_and_read(self):
        buf = SPSCRingBuffer(1024)
        data = b"hello world"
        assert buf.write(data)
        result = buf.read(len(data))
        assert result == data

    def test_available_read_write(self):
        buf = SPSCRingBuffer(1024)
        assert buf.available_read == 0
        assert buf.available_write == 1023  # capacity - 1

        buf.write(b"test")
        assert buf.available_read == 4
        assert buf.available_write == 1019

    def test_overflow_returns_false(self):
        buf = SPSCRingBuffer(16)
        big_data = bytes(16)
        assert not buf.write(big_data)  # capacity - 1 = 15
        assert buf.overflow_count == 1

    def test_underrun_returns_none(self):
        buf = SPSCRingBuffer(1024)
        result = buf.read(100)
        assert result is None
        assert buf.underrun_count == 1

    def test_wrap_around(self):
        buf = SPSCRingBuffer(32)
        # Fill most of the buffer
        data1 = bytes(range(20))
        assert buf.write(data1)
        # Read it out
        result1 = buf.read(20)
        assert result1 == data1

        # Write again (wraps around)
        data2 = bytes(range(20, 40))
        assert buf.write(data2)
        result2 = buf.read(20)
        assert result2 == data2

    def test_multiple_writes_and_reads(self):
        buf = SPSCRingBuffer(256)
        for i in range(10):
            data = bytes([i] * 10)
            assert buf.write(data)
            result = buf.read(10)
            assert result == data

    def test_factory_capture_buffer(self):
        buf = create_capture_ring_buffer()
        assert buf._capacity > 0

    def test_factory_playback_buffer(self):
        buf = create_playback_ring_buffer()
        assert buf._capacity > 0
