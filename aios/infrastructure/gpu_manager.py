"""
GPU resource manager for AIOS runtime.

Manages CUDA device selection, memory monitoring, and OOM recovery.
Supports both CPU and CUDA inference with automatic fallback.
"""

import asyncio
import logging
import gc
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum

from ..config import DeviceType, GPUConfig

logger = logging.getLogger(__name__)


class MemoryState(Enum):
    """GPU memory state."""
    HEALTHY = "healthy"        # Below warning threshold
    WARNING = "warning"        # Above warning but below critical
    CRITICAL = "critical"      # Above critical threshold, near OOM
    OOM = "oom"                # Out of memory occurred


@dataclass
class MemoryStats:
    """GPU memory statistics."""
    allocated_gb: float = 0.0
    reserved_gb: float = 0.0
    free_gb: float = 0.0
    total_gb: float = 0.0
    state: MemoryState = MemoryState.HEALTHY
    timestamp: float = 0.0


@dataclass
class ModelVRAMBudget:
    """VRAM budget for a single model."""
    model_name: str
    estimated_gb: float
    actual_allocated_gb: float = 0.0


# Default VRAM budgets in GB (from spec)
DEFAULT_VRAM_BUDGETS = {
    "personaplex_asr": 2.0,
    "slm_correction": 1.75,
    "slm_response": 1.75,
    "tts": 1.0,
    "cuda_runtime": 1.0,
}


class GPUManager:
    """
    Manages GPU resources for AIOS runtime.

    Responsibilities:
    - Device selection (CPU/CUDA) with fallback
    - Memory monitoring and warning thresholds
    - OOM detection and recovery
    - CUDA stream isolation
    """

    def __init__(self, config: GPUConfig):
        """
        Initialize the GPU manager.

        Args:
            config: GPU configuration
        """
        self.config = config
        self.primary_device = config.primary_device
        self.fallback_device = config.fallback_device

        # Memory tracking
        self._memory_stats: Optional[MemoryStats] = None
        self._model_budgets: Dict[str, ModelVRAMBudget] = {}
        self._oom_count = 0

        # CUDA state
        self._cuda_available: bool = False
        self._cuda_device_id: Optional[int] = None
        self._cuda_streams: Dict[str, Any] = {}

        # Initialize
        self._initialize_cuda()
        self._initialize_budgets()

        logger.info(
            f"GPUManager initialized: device={self.primary_device.value}, "
            f"cuda_available={self._cuda_available}"
        )

    @property
    def is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        return self._cuda_available

    @property
    def current_device(self) -> DeviceType:
        """Get the current active device."""
        if self.primary_device == DeviceType.CUDA and self._cuda_available:
            return DeviceType.CUDA
        if self.fallback_device:
            return self.fallback_device
        return DeviceType.CPU

    @property
    def memory_stats(self) -> Optional[MemoryStats]:
        """Get current memory statistics."""
        return self._memory_stats

    def _initialize_cuda(self):
        """Initialize CUDA device if available."""
        try:
            import torch

            self._cuda_available = torch.cuda.is_available()

            if self._cuda_available:
                self._cuda_device_id = self.config.cuda_device_id

                # Set default device
                torch.cuda.set_device(self._cuda_device_id)

                # Create dedicated CUDA streams for each model
                if self.config.enable_cuda_stream_isolation:
                    self._cuda_streams = {
                        "asr": torch.cuda.Stream(),
                        "slm": torch.cuda.Stream(),
                        "tts": torch.cuda.Stream(),
                    }

                logger.info(
                    f"CUDA initialized: device={self._cuda_device_id}, "
                    f"device_name={torch.cuda.get_device_name(self._cuda_device_id)}"
                )
            else:
                logger.warning("CUDA not available, falling back to CPU")

        except ImportError:
            logger.warning("PyTorch not available, CPU only")
            self._cuda_available = False

    def _initialize_budgets(self):
        """Initialize VRAM budgets for each model."""
        for model_name, budget_gb in DEFAULT_VRAM_BUDGETS.items():
            self._model_budgets[model_name] = ModelVRAMBudget(
                model_name=model_name,
                estimated_gb=budget_gb
            )

        total_budget = sum(b.estimated_gb for b in self._model_budgets.values())
        logger.info(f"Total VRAM budget: {total_budget:.2f} GB")

    def get_stream(self, model_name: str) -> Any:
        """
        Get the CUDA stream for a specific model.

        Args:
            model_name: One of 'asr', 'slm', 'tts'

        Returns:
            CUDA stream or None if not using CUDA
        """
        if not self._cuda_available:
            return None

        return self._cuda_streams.get(model_name)

    async def update_memory_stats(self) -> MemoryStats:
        """
        Update and return current memory statistics.

        Returns:
            Current memory statistics
        """
        if not self._cuda_available:
            self._memory_stats = MemoryStats(
                allocated_gb=0.0,
                reserved_gb=0.0,
                free_gb=0.0,
                total_gb=0.0,
                state=MemoryState.HEALTHY,
                timestamp=asyncio.get_event_loop().time()
            )
            return self._memory_stats

        try:
            import torch

            allocated = torch.cuda.memory_allocated(self._cuda_device_id) / (1024**3)
            reserved = torch.cuda.memory_reserved(self._cuda_device_id) / (1024**3)
            total = torch.cuda.get_device_properties(self._cuda_device_id).total_memory / (1024**3)
            free = total - allocated

            # Determine state
            state = MemoryState.HEALTHY
            if allocated >= self.config.memory_warning_threshold_gb:
                state = MemoryState.CRITICAL
            elif allocated >= self.config.memory_warning_threshold_gb * 0.8:
                state = MemoryState.WARNING

            self._memory_stats = MemoryStats(
                allocated_gb=allocated,
                reserved_gb=reserved,
                free_gb=free,
                total_gb=total,
                state=state,
                timestamp=asyncio.get_event_loop().time()
            )

            if state != MemoryState.HEALTHY:
                logger.warning(
                    f"GPU memory state: {state.value} "
                    f"({allocated:.2f}GB / {total:.2f}GB)"
                )

        except Exception as e:
            logger.error(f"Error updating memory stats: {e}")

        return self._memory_stats

    async def clear_cache(self):
        """Clear CUDA cache to free memory."""
        if not self._cuda_available:
            return

        try:
            import torch

            # Synchronize and clear
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            # Force Python garbage collection
            gc.collect()

            # Update stats after clearing
            await self.update_memory_stats()

            logger.debug("CUDA cache cleared")

        except Exception as e:
            logger.error(f"Error clearing CUDA cache: {e}")

    async def handle_oom(self) -> bool:
        """
        Handle out-of-memory error.

        Recovery strategy:
        1. Clear CUDA cache
        2. Force garbage collection
        3. Check if memory is now available

        Returns:
            True if recovery successful, False otherwise
        """
        self._oom_count += 1
        logger.warning(f"Handling OOM error (count: {self._oom_count})")

        # Clear cache
        await self.clear_cache()

        # Update stats to check if we're still in OOM
        stats = await self.update_memory_stats()

        if stats.state == MemoryState.OOM:
            logger.error("Still in OOM after cache clear")
            return False

        logger.info("OOM recovery successful")
        return True

    def should_use_cuda(self) -> bool:
        """Check if CUDA should be used for inference."""
        return (
            self._cuda_available and
            self.primary_device == DeviceType.CUDA
        )

    def get_device_for_inference(self) -> Any:
        """
        Get the device object for inference (PyTorch).

        Returns:
            torch.device or appropriate device object
        """
        if not self.should_use_cuda():
            return "cpu"

        try:
            import torch
            return torch.device(f"cuda:{self._cuda_device_id}")
        except ImportError:
            return "cpu"

    def get_model_device_context(self, model_name: str):
        """
        Get a context manager for running inference on a specific model's device/stream.

        Args:
            model_name: One of 'asr', 'slm', 'tts'

        Returns:
            Context manager for device/stream
        """
        if not self.should_use_cuda():
            return self._cpu_context()

        try:
            import torch

            stream = self.get_stream(model_name)
            if stream:
                return torch.cuda.stream(stream)

            return self._cuda_context()

        except ImportError:
            return self._cpu_context()

    @staticmethod
    def _cpu_context():
        """No-op CPU context."""
        class CPUContext:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return CPUContext()

    @staticmethod
    def _cuda_context():
        """CUDA device context."""
        try:
            import torch
            return torch.cuda.device(torch.cuda.current_device())
        except ImportError:
            return GPUManager._cpu_context()

    def get_vram_usage_summary(self) -> str:
        """Get a summary of VRAM usage."""
        if not self._cuda_available:
            return "CUDA not available, using CPU"

        stats = self._memory_stats
        if not stats:
            return "No memory stats available"

        lines = [
            f"GPU Memory Usage:",
            f"  Allocated: {stats.allocated_gb:.2f} GB",
            f"  Reserved:  {stats.reserved_gb:.2f} GB",
            f"  Free:      {stats.free_gb:.2f} GB",
            f"  Total:     {stats.total_gb:.2f} GB",
            f"  State:     {stats.state.value}",
        ]

        return "\n".join(lines)


def create_gpu_manager(config: GPUConfig) -> GPUManager:
    """
    Factory function to create a GPU manager.

    Args:
        config: GPU configuration

    Returns:
        Configured GPUManager instance
    """
    return GPUManager(config)


async def monitor_memory_periodically(
    gpu_manager: GPUManager,
    interval_seconds: float = 5.0,
    callback=None
):
    """
    Periodically monitor GPU memory and optionally call a callback.

    Args:
        gpu_manager: GPU manager to monitor
        interval_seconds: How often to check memory
        callback: Optional async callback function to call with stats
    """
    while True:
        try:
            stats = await gpu_manager.update_memory_stats()

            if callback:
                await callback(stats)

            await asyncio.sleep(interval_seconds)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in memory monitor: {e}")
            await asyncio.sleep(interval_seconds)
