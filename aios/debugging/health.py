"""
Health Check HTTP Server for AIOS

Provides /health endpoint for liveness probes, served by asyncio.
Runs on the asyncio event loop (no separate thread needed).

Startup probes:
- GPU available (torch.cuda.is_available())
- VRAM sufficient (torch.cuda.mem_get_info())
- Audio device available (sounddevice.query_devices())

Liveness probes:
- Event loop responsive (HTTP GET /health)
- State machine not stuck (watchdog)

Readiness probes:
- All models ready
- Audio device open
- State is IDLE or LISTENING
"""

import asyncio
import json
import logging
import time
from typing import Optional, Callable

logger = logging.getLogger(__name__)

DEFAULT_HEALTH_PORT = 8080


class HealthCheckServer:
    """
    Lightweight HTTP health check server running on the asyncio event loop.

    Endpoints:
    - GET /health      -> 200 if event loop responsive
    - GET /ready       -> 200 if all models ready and system operational
    - GET /status      -> 200 with JSON status payload
    """

    def __init__(
        self,
        port: int = DEFAULT_HEALTH_PORT,
        status_callback: Optional[Callable[[], dict]] = None,
    ):
        self._port = port
        self._status_callback = status_callback
        self._server: Optional[asyncio.AbstractServer] = None
        self._start_time = time.monotonic()
        self._ready = False

    def set_ready(self, ready: bool):
        self._ready = ready

    async def start(self):
        """Start the health check HTTP server."""
        try:
            self._server = await asyncio.start_server(
                self._handle_connection,
                host="0.0.0.0",
                port=self._port,
            )
            logger.info("Health check server started on port %d", self._port)
        except Exception as e:
            logger.error("Failed to start health check server: %s", e)

    async def stop(self):
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            logger.info("Health check server stopped")

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ):
        try:
            data = await asyncio.wait_for(reader.read(4096), timeout=5.0)
            request = data.decode("utf-8", errors="replace")

            # Parse the request path
            path = "/"
            if request.startswith("GET "):
                parts = request.split(" ")
                if len(parts) >= 2:
                    path = parts[1]

            if path == "/health":
                await self._respond(writer, 200, {"status": "ok", "uptime_s": time.monotonic() - self._start_time})
            elif path == "/ready":
                if self._ready:
                    await self._respond(writer, 200, {"status": "ready"})
                else:
                    await self._respond(writer, 503, {"status": "not_ready"})
            elif path == "/status":
                status = {}
                if self._status_callback:
                    try:
                        status = self._status_callback()
                    except Exception as e:
                        status = {"error": str(e)}
                status["uptime_s"] = time.monotonic() - self._start_time
                await self._respond(writer, 200, status)
            else:
                await self._respond(writer, 404, {"error": "not_found"})

        except asyncio.TimeoutError:
            pass
        except Exception as e:
            logger.debug("Health check connection error: %s", e)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _respond(self, writer: asyncio.StreamWriter, status_code: int, body: dict):
        body_bytes = json.dumps(body).encode("utf-8")
        status_text = {200: "OK", 404: "Not Found", 503: "Service Unavailable"}.get(status_code, "Unknown")
        response = (
            f"HTTP/1.1 {status_code} {status_text}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body_bytes)}\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        ).encode("utf-8") + body_bytes
        writer.write(response)
        await writer.drain()


def run_startup_probes(require_gpu: bool = True, require_audio: bool = True) -> dict:
    """
    Run startup probes and return results.

    Returns:
        Dict with probe results. All values True means startup can proceed.
    """
    results = {}

    # GPU probe
    if require_gpu:
        try:
            import torch
            results["gpu_available"] = torch.cuda.is_available()
            if results["gpu_available"]:
                free, total = torch.cuda.mem_get_info()
                results["gpu_vram_free_gb"] = round(free / (1024**3), 2)
                results["gpu_vram_total_gb"] = round(total / (1024**3), 2)
                results["gpu_vram_sufficient"] = free > 7 * (1024**3)  # 7GB minimum
            else:
                results["gpu_vram_sufficient"] = False
        except ImportError:
            results["gpu_available"] = False
            results["gpu_vram_sufficient"] = False
    else:
        results["gpu_available"] = True
        results["gpu_vram_sufficient"] = True

    # Audio device probe
    if require_audio:
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            results["audio_device_available"] = len(devices) > 0
        except Exception as e:
            results["audio_device_available"] = False
            results["audio_device_error"] = str(e)
    else:
        results["audio_device_available"] = True

    return results
