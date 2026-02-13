# EarOS — Speech-Native AI Operating System

**Real-time speech-to-speech interaction with conversational continuity, accent-aware correction, and sub-500ms latency.**

EarOS (AIOS) is a speech-native AI operating layer built on NVIDIA PersonaPlex. It captures live microphone audio, transcribes with streaming ASR, corrects Vietnamese-accented English errors in real-time, generates contextual responses via a small language model (SLM), and synthesizes speech — all within a single asyncio-driven Python process targeting ≤ 500ms end-to-end turn latency.

---

## Features

- **Full-Duplex Audio** — Simultaneous capture and playback with lock-free SPSC ring buffers and software acoustic echo cancellation (AEC).
- **Streaming ASR** — NVIDIA PersonaPlex gRPC bidirectional streaming with partial and final results, hot-word boosting.
- **Accent-Aware Correction** — Two-phase sliding-window correction engine: rule-based pre-filter (≤ 5ms) + SLM contextual correction (≤ 75ms) for Vietnamese-accented English.
- **Conversational Continuity** — 10-turn sliding context window with FIFO eviction, token-budget-aware prompt assembly, and prompt versioning.
- **Barge-In Support** — User can interrupt AI speech at any time. TTS stops within 100ms, downstream queues flush, and a new ASR stream opens immediately.
- **Deterministic State Machine** — 5-state FSM (IDLE → LISTENING → PROCESSING → SPEAKING → INTERRUPTED) with per-state watchdog timers and a sequential event bus.
- **Graceful Degradation** — 5-level degradation (L0–L4) triggered by sustained latency overshoots, with automatic 10-second recovery timers.
- **Model Abstraction Layer** — Protocol-based interfaces for ASR, SLM, and TTS with automatic fallback (e.g., PersonaPlex → Whisper-tiny, Phi-3 → rule-based, Riva → VITS).
- **Observability** — Structured JSON logging, Prometheus metrics on `:9090`, HTTP health/readiness probes on `:8080`, per-stage latency tracing, and debug audio WAV dumps.
- **Hot-Word Runtime Updates** — JSON-based hot-word list with file-watch reload (≤ 5s delay), boosting both ASR recognition and correction accuracy.

---

## Architecture

EarOS uses a **5-layer architecture** with asyncio-first concurrency and explicit threading boundaries for audio I/O and GPU inference.

```
┌─────────────────────────────────────────────────────────────────┐
│                        EarOS Process                            │
│                                                                 │
│  Layer 5: State Machine & Event Bus                             │
│  ├── 5-state FSM with deterministic transitions                 │
│  ├── Per-state watchdog timers (LISTENING 10s, PROCESSING 5s,   │
│  │   SPEAKING 30s, INTERRUPTED 0.5s)                            │
│  └── Async event bus (bounded queue, sequential consumption)    │
│                                                                 │
│  Layer 4: Context Injection Engine                              │
│  ├── Fixed-slot prompt: System (512t) + Context (2048t)         │
│  │   + Current Turn (512t) + Response Reserve (1024t)           │
│  ├── Hot-word interpolation into system prompt                  │
│  └── Prompt versioning (turn_id, prompt_version, context_hash)  │
│                                                                 │
│  Layer 3: Pipeline Orchestrator                                 │
│  ├── Queue-connected stages: Capture → VAD → ASR → Correction  │
│  │   → Context Injection → SLM → TTS → Playback                │
│  ├── ThreadPoolExecutor(3) for GPU inference                    │
│  └── Barge-in: task cancellation + queue flush                  │
│                                                                 │
│  Layer 2: Context & Memory Manager                              │
│  ├── 10-turn sliding window with FIFO eviction                  │
│  ├── Pre-computed token counts, O(1) retrieval                  │
│  └── Turn records: raw_asr, corrected text, duration, interrupt │
│                                                                 │
│  Layer 1: Audio I/O Kernel                                      │
│  ├── SPSC lock-free ring buffers (capture 32KB, playback 48KB)  │
│  ├── 16 kHz capture / 24 kHz playback, 20ms frames             │
│  └── Software AEC (speexdsp / WebRTC)                           │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Mic (16kHz) → Ring Buffer → AEC → VAD → ASR (PersonaPlex gRPC)
  → Streaming Correction → Context Injection → SLM (Phi-3-mini)
  → TTS (Riva gRPC) → Ring Buffer → Speaker (24kHz)
```

### Thread Model

| Thread | Role | Blocking |
|--------|------|----------|
| Main (asyncio event loop) | VAD, state machine, queue routing, prompt assembly, metrics | No |
| Audio callback (PortAudio) | Mic capture, speaker playback, ring buffer R/W | No (≤ 1ms) |
| GPU workers (ThreadPool × 3) | ASR gRPC, SLM inference, TTS gRPC | Yes (CUDA) |

---

## Project Structure

```
EarOS/
├── main.py                     # Entry point
├── Dockerfile                  # CUDA 11.8 + Python 3.11 container
├── docker-compose.yaml         # AIOS + PersonaPlex + Riva TTS
├── requirements.txt            # Python dependencies
├── config/
│   ├── aios.yaml               # Main configuration
│   ├── models.yaml             # Model registry (versions, endpoints)
│   ├── hot_words.json          # Hot-word list with phonetic variants
│   └── system_prompt.txt       # SLM system prompt template
├── aios/
│   ├── __init__.py             # Package exports
│   ├── config/                 # Configuration dataclasses
│   ├── core/
│   │   ├── state_machine.py    # Layer 5: FSM + Event Bus
│   │   ├── context_manager.py  # Layer 2: Turn history + retrieval
│   │   ├── context_injection.py# Layer 4: Prompt assembly
│   │   ├── pipeline.py         # Layer 3: Pipeline orchestrator
│   │   ├── audio_io.py         # Layer 1: Audio I/O kernel + AEC
│   │   ├── ring_buffer.py      # SPSC lock-free ring buffer
│   │   ├── vad.py              # Voice activity detection (Silero/energy)
│   │   └── streaming_correction.py # Accent correction engine
│   ├── interfaces/
│   │   └── models.py           # ASR/SLM/TTS Protocol classes + mocks
│   ├── performance/
│   │   ├── latency.py          # Per-stage latency tracing
│   │   ├── degradation.py      # Graceful degradation (L0–L4)
│   │   └── metrics.py          # Prometheus metrics collector
│   ├── debugging/
│   │   ├── logging_config.py   # Structured JSON logging
│   │   ├── health.py           # HTTP health/readiness server
│   │   └── audio_dump.py       # Debug WAV file dumper
│   ├── infrastructure/
│   │   ├── streaming_client.py # PersonaPlex gRPC client
│   │   ├── gpu_manager.py      # GPU VRAM management
│   │   └── asr_service.py      # ASR service wrapper
│   └── testing/
│       ├── synthetic_audio.py  # PCM frame generator for tests
│       └── fixtures.py         # Pytest fixtures with mock models
└── tests/
    ├── test_state_machine.py   # FSM transitions, watchdog, event bus
    ├── test_vad.py             # Energy gate, debounce, pre-roll, barge-in
    ├── test_context.py         # Context manager + injection engine
    ├── test_correction.py      # Rule-based correction + confidence gating
    ├── test_ring_buffer.py     # SPSC buffer correctness
    └── test_performance.py     # Latency tracer, degradation, histograms
```

---

## Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA RTX 3080 (10 GB VRAM) | NVIDIA RTX 4090 (24 GB VRAM) |
| CPU | 4 cores, 3.0 GHz | 8 cores, 4.0 GHz |
| RAM | 16 GB | 32 GB |
| Storage | 10 GB SSD | 50 GB NVMe SSD |
| Audio | USB microphone + speakers | Low-latency audio interface |
| OS | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS |

### GPU VRAM Budget (~7.5 GB)

| Model | VRAM |
|-------|------|
| PersonaPlex ASR | ~2.0 GB |
| Phi-3-mini SLM (GPTQ 4-bit) | ~3.5 GB |
| Riva TTS | ~1.0 GB |
| CUDA runtime + buffers | ~1.0 GB |

### Software

- Python 3.11+
- CUDA 11.8+
- PortAudio (`apt install libportaudio2`)

---

## Quick Start

### Local Development

```bash
# Clone and set up
git clone <repo-url> EarOS && cd EarOS
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start model servers (separate terminals)
# personaplex-server --port 50051
# riva-tts-server --port 50052

# Run EarOS
python main.py
```

### Docker Compose (Recommended)

```bash
# Build and start all services (AIOS + PersonaPlex + Riva TTS)
docker compose up -d

# Check health
curl http://localhost:8080/health

# View metrics
curl http://localhost:9090/metrics

# View logs
docker compose logs -f aios
```

---

## Usage

### Command-Line Options

```
python main.py [OPTIONS]

Options:
  --no-audio          Disable audio hardware (testing mode)
  --no-aec            Disable acoustic echo cancellation
  --hot-words PATH    Path to hot-words JSON file
  --system-prompt PATH  Path to system prompt file or inline text
  --debug             Enable debug mode (audio dumps, verbose logging)
  --metrics           Enable Prometheus metrics server on :9090
  --health-port PORT  Health check server port (default: 8080)
  --config PATH       Path to aios.yaml config file
```

### Examples

```bash
# Run with custom hot-words and metrics enabled
python main.py --hot-words config/hot_words.json --metrics

# Run in debug mode (audio dumps to /tmp/aios_debug/, verbose logs)
python main.py --debug

# Run without audio hardware (for testing)
python main.py --no-audio

# Run with custom system prompt
python main.py --system-prompt config/system_prompt.txt
```

---

## Configuration

### `config/aios.yaml` — Main Configuration

Controls audio parameters, VAD thresholds, queue sizes, latency targets, degradation thresholds, and logging settings. Requires restart on change.

### `config/models.yaml` — Model Registry

Specifies model providers, versions, endpoints, and fallback options for ASR, SLM, and TTS. Requires restart on change.

### `config/hot_words.json` — Hot-Word List

```json
{
  "hot_words": [
    {"term": "PersonaPlex", "phonetic_variants": ["personna plex", "persona plex"]},
    {"term": "NVIDIA", "phonetic_variants": ["in vidia", "envidia"]}
  ]
}
```

Supports **runtime hot-reload** via file watch (changes take effect within 5 seconds).

### `config/system_prompt.txt` — System Prompt Template

Supports `{hot_words}` placeholder for automatic hot-word interpolation at prompt assembly time.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AIOS_CONFIG_DIR` | `./config` | Configuration directory path |
| `AIOS_LOG_LEVEL` | `INFO` | Log level: ERROR, WARN, INFO, DEBUG, TRACE |
| `AIOS_DEBUG` | `0` | Enable debug mode (audio dumps, verbose logging) |
| `AIOS_CAPTURE_DEVICE` | System default | Audio capture device name or index |
| `AIOS_PLAYBACK_DEVICE` | System default | Audio playback device name or index |
| `AIOS_METRICS_PORT` | `9090` | Prometheus metrics HTTP port |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device index |

---

## State Machine

```
         speech_start              speech_end           first_tts_frame
IDLE ──────────────→ LISTENING ──────────────→ PROCESSING ──────────────→ SPEAKING
 ↑                       ↑                        │                         │
 │  playback_complete     │  interrupt_complete     │  speech_start          │ barge_in
 └───────────────────── SPEAKING ←──────────────────┘                       │
                          ↑                                                 ↓
                          └──────────────────────────────────────── INTERRUPTED
```

| State | Watchdog Timeout | On Timeout |
|-------|-----------------|------------|
| LISTENING | 10s | → IDLE (turn_timeout) |
| PROCESSING | 5s | → IDLE (processing_timeout) |
| SPEAKING | 30s | → IDLE (watchdog_timeout) |
| INTERRUPTED | 0.5s | → LISTENING (interrupt_complete) |

---

## Degradation Levels

| Level | Trigger | Action | Recovery |
|-------|---------|--------|----------|
| L0 | All stages within budget | Full pipeline | — |
| L1 | Correction > 140ms × 3 | Disable SLM correction, rule-based only | 10s timer |
| L2 | SLM first token > 400ms × 3 | Reduce context window: 10 → 3 turns | 10s timer |
| L3 | TTS > 200ms × 3 | Switch to lower-quality faster TTS | 10s timer |
| L4 | Multiple stages failing | Skip correction entirely, minimal context | 10s timer |

---

## Testing

```bash
# Run all tests (91 tests)
pytest tests/ -v

# Run specific test modules
pytest tests/test_state_machine.py -v    # FSM transitions
pytest tests/test_vad.py -v              # VAD debounce, pre-roll
pytest tests/test_context.py -v          # Context manager + injection
pytest tests/test_correction.py -v       # Accent correction rules
pytest tests/test_ring_buffer.py -v      # SPSC buffer
pytest tests/test_performance.py -v      # Latency + degradation
```

### Test Coverage

| Module | Tests | Focus |
|--------|-------|-------|
| State Machine | 19 | Valid/invalid transitions, watchdog, turn ID, event bus |
| VAD | 10 | Energy gate, calibration, onset/offset debounce, pre-roll, barge-in |
| Context | 29 | Turn storage, eviction, token budget, prompt assembly, versioning |
| Correction | 7 | Consonant restoration, th-fronting, hot-words, hallucination gate |
| Ring Buffer | 8 | Read/write, wrap-around, overflow, underrun |
| Performance | 18 | Latency tracing, degradation levels, recovery, histograms |

---

## Monitoring

### Prometheus Metrics (`:9090/metrics`)

| Metric | Type | Description |
|--------|------|-------------|
| `aios_turn_latency_ms` | Histogram | Per-stage and end-to-end latency |
| `aios_state_transitions_total` | Counter | State transition counts |
| `aios_queue_depth` | Gauge | Current queue depth per stage |
| `aios_queue_drops_total` | Counter | Items dropped due to backpressure |
| `aios_vad_speech_probability` | Histogram | Per-frame VAD confidence |
| `aios_gpu_vram_bytes` | Gauge | Current GPU VRAM usage |
| `aios_degradation_level` | Gauge | Current degradation level (0–4) |
| `aios_barge_in_latency_ms` | Histogram | Barge-in detection to silence latency |
| `aios_errors_total` | Counter | Error count per component |

### Health Endpoints (`:8080`)

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Liveness probe (event loop responsive) |
| `GET /ready` | Readiness probe (all models loaded, audio device open) |

---

## Latency Targets

| Stage | Target | Hard Limit |
|-------|--------|------------|
| Audio capture | 5ms | — |
| AEC | 5ms | — |
| VAD | 3ms | — |
| ASR (first partial) | 200ms | 350ms |
| Correction | 80ms | 140ms |
| Context injection | 5ms | — |
| SLM (first token) | 200ms | 400ms |
| TTS (first frame) | 100ms | 200ms |
| **End-to-end** | **500ms** | **800ms** |
| **Barge-in** | **100ms** | **200ms** |

---

## License

Proprietary. All rights reserved.
