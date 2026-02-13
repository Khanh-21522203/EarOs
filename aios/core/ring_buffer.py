"""
SPSC (Single-Producer, Single-Consumer) Lock-Free Ring Buffer

Used for audio I/O between the PortAudio callback thread and the asyncio
event loop. No mutexes, no condition variables — uses atomic index variables.

Capture ring buffer: 16,000 samples (1 second) = 32,000 bytes
Playback ring buffer: 24,000 samples (1 second) = 48,000 bytes
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SPSCRingBuffer:
    """
    Single-Producer, Single-Consumer lock-free ring buffer.

    Thread-safety: The producer (write) and consumer (read) may operate
    from different threads without locks. Only the producer writes _write_idx;
    only the consumer writes _read_idx.

    Correctness invariant:
        _write_idx - _read_idx (mod capacity) == number of unread bytes.
        Producer never advances _write_idx past _read_idx.
        Consumer never advances _read_idx past _write_idx.
    """

    def __init__(self, capacity_bytes: int):
        """
        Initialize the ring buffer.

        Args:
            capacity_bytes: Total buffer capacity in bytes.
        """
        self._buf = bytearray(capacity_bytes)
        self._capacity = capacity_bytes
        self._write_idx: int = 0  # Written only by producer
        self._read_idx: int = 0   # Written only by consumer

        # Metrics
        self._total_written: int = 0
        self._total_read: int = 0
        self._overflow_count: int = 0
        self._underrun_count: int = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def available_read(self) -> int:
        """Number of bytes available to read."""
        return (self._write_idx - self._read_idx) % self._capacity

    @property
    def available_write(self) -> int:
        """Number of bytes available to write (one slot reserved to distinguish full from empty)."""
        return (self._capacity - 1) - self.available_read

    @property
    def overflow_count(self) -> int:
        return self._overflow_count

    @property
    def underrun_count(self) -> int:
        return self._underrun_count

    def write(self, data: bytes) -> bool:
        """
        Write data to the ring buffer (producer side).

        Args:
            data: Bytes to write.

        Returns:
            True if written successfully, False if overflow (data dropped).
        """
        size = len(data)
        if size > self.available_write:
            self._overflow_count += 1
            return False

        write_pos = self._write_idx % self._capacity
        end_pos = write_pos + size

        if end_pos <= self._capacity:
            # Contiguous write
            self._buf[write_pos:end_pos] = data
        else:
            # Wrap-around write
            first_part = self._capacity - write_pos
            self._buf[write_pos:self._capacity] = data[:first_part]
            self._buf[0:size - first_part] = data[first_part:]

        self._write_idx = (self._write_idx + size) % self._capacity
        self._total_written += size
        return True

    def read(self, size: int) -> Optional[bytes]:
        """
        Read data from the ring buffer (consumer side).

        Args:
            size: Number of bytes to read.

        Returns:
            Bytes read, or None if underrun (not enough data).
        """
        if size > self.available_read:
            self._underrun_count += 1
            return None

        read_pos = self._read_idx % self._capacity
        end_pos = read_pos + size

        if end_pos <= self._capacity:
            # Contiguous read
            data = bytes(self._buf[read_pos:end_pos])
        else:
            # Wrap-around read
            first_part = self._capacity - read_pos
            data = bytes(self._buf[read_pos:self._capacity]) + bytes(self._buf[0:size - first_part])

        self._read_idx = (self._read_idx + size) % self._capacity
        self._total_read += size
        return data

    def peek(self, size: int) -> Optional[bytes]:
        """Peek at data without consuming it."""
        if size > self.available_read:
            return None

        read_pos = self._read_idx % self._capacity
        end_pos = read_pos + size

        if end_pos <= self._capacity:
            return bytes(self._buf[read_pos:end_pos])
        else:
            first_part = self._capacity - read_pos
            return bytes(self._buf[read_pos:self._capacity]) + bytes(self._buf[0:size - first_part])

    def clear(self):
        """Clear the buffer (consumer side)."""
        self._read_idx = self._write_idx

    def get_stats(self) -> dict:
        """Get buffer statistics."""
        return {
            "capacity": self._capacity,
            "available_read": self.available_read,
            "available_write": self.available_write,
            "occupancy_pct": (self.available_read / self._capacity) * 100 if self._capacity > 0 else 0,
            "total_written": self._total_written,
            "total_read": self._total_read,
            "overflow_count": self._overflow_count,
            "underrun_count": self._underrun_count,
        }


def create_capture_ring_buffer(sample_rate: int = 16000, duration_seconds: float = 1.0) -> SPSCRingBuffer:
    """
    Create a ring buffer for audio capture.

    Args:
        sample_rate: Capture sample rate in Hz.
        duration_seconds: Buffer duration in seconds.

    Returns:
        SPSCRingBuffer sized for capture.
    """
    # 16-bit samples = 2 bytes per sample
    capacity = int(sample_rate * duration_seconds * 2)
    return SPSCRingBuffer(capacity)


def create_playback_ring_buffer(sample_rate: int = 24000, duration_seconds: float = 1.0) -> SPSCRingBuffer:
    """
    Create a ring buffer for audio playback.

    Args:
        sample_rate: Playback sample rate in Hz.
        duration_seconds: Buffer duration in seconds.

    Returns:
        SPSCRingBuffer sized for playback.
    """
    capacity = int(sample_rate * duration_seconds * 2)
    return SPSCRingBuffer(capacity)
