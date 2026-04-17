# Voice Agent Examples — Project Constitution

This repo contains production-ready voice agent examples using Plivo telephony. Every example follows the same structure regardless of AI API or orchestration approach.

## Naming Convention

`{llm-provider+series}-{stt-provider+series}-{tts-provider+series}-{orchestration}[-{variant}]`

**Every component always includes provider name + model series.** The series identifies the API contract; the size variant (mini/nano/pro/flash) is config in `.env`, not part of the folder name.

### LLM component: `{provider}{version}`

Drop the size class (mini/nano/pro/flash) — it's `.env` config. Only include size when two different sizes are used together in the same example.

| Model | Folder component | Notes |
|---|---|---|
| `gpt-5.4-mini` | `gpt5.4` | drop "mini" |
| `gpt-4.1` | `gpt4.1` | |
| `gpt-4.1-mini` (alone) | `gpt4.1` | drop "mini" |
| `gpt-4.1-mini` + `gpt-4.1` (dual) | `gpt4.1mini-gpt4.1` | two sizes → keep both |
| `gpt-4o-mini` | `gpt4o` | drop "mini" |
| `gemini-2.0-flash` | `gemini2` | drop "flash" |
| `gemini-2.5-flash` (live API) | `gemini2.5-live` | drop "flash"; `-live` = S2S API type |
| `gemini-3.1-flash` (live API) | `gemini3.1-live` | drop "flash"; `-live` = S2S API type |
| `gpt-realtime-1.5` (S2S) | `gptrealtime1.5` | "realtime" is the model name |
| `grok-3-fast-voice` (S2S) | `grok3-voice` | |

### Voice AI (STT) component: `{provider}{model-name}{version}`

| Model | Folder component |
|---|---|
| Deepgram `nova-2-phonecall` | `deepgramnova2` |
| Deepgram `nova-3` | `deepgramnova3` |
| Deepgram `flux` | `deepgramflux` |
| AssemblyAI `u3-rt-pro` | `assemblyaiu3` |
| Sarvam STT | `sarvam` (no named model series) |

### Voice AI (TTS) component: `{provider}{model-name}{version}`

| Model | Folder component |
|---|---|
| ElevenLabs `eleven_flash_v2_5` | `elevenflashv2.5` |
| Cartesia `sonic-2` | `cartesiasonic2` |
| Cartesia `sonic-3` | `cartesiasonic3` |
| OpenAI `gpt-4o-mini-tts` | `openaitts4o` |
| Grok `grok-3-fast-voice` (TTS only) | `groktts3` |

### Examples

`gpt5.4-assemblyaiu3-cartesiasonic3-native`, `gemini2.5-live-pipecat`, `gpt4.1-deepgramnova3-elevenflashv2.5-native`

Orchestration types:
- **native** — raw websockets/SDK, custom asyncio task management, client-side Silero VAD (default)
- **pipecat** / **livekit** / **vapi** — framework-based Pipeline, framework-managed VAD

Variants:
- **`-no-vad`** — explicitly opts out of client-side VAD (e.g., `gemini2.5-live-native-no-vad` relies on server-side VAD)
- **`-webrtcvad`** — uses WebRTC VAD instead of Silero (e.g., `gemini2.5-live-native-webrtcvad`)
- All new native examples include Silero VAD by default. These suffixes are the exception, not the rule.

## Canonical File Structure (ALL examples)

```
{example-name}/
├── inbound/
│   ├── __init__.py
│   ├── agent.py              # AI-specific voice agent class (or framework pipeline)
│   ├── server.py             # FastAPI: /answer, /ws, /hangup
│   └── system_prompt.md      # System prompt for inbound calls
├── outbound/
│   ├── __init__.py
│   ├── agent.py              # Same agent class + OutboundCallRecord, CallManager
│   ├── server.py             # FastAPI: /outbound/call, /outbound/ws, etc.
│   └── system_prompt.md      # System prompt for outbound calls
├── utils.py                  # Audio conversion, VAD (if native), phone utils
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # sys.path setup (copy from grok3-voice-native)
│   ├── helpers.py            # ngrok, recording, transcription (copy from grok3-voice-native)
│   ├── test_integration.py   # Unit + local integration tests
│   ├── test_e2e_live.py      # E2E with real API (no phone call)
│   ├── test_live_call.py     # Real inbound call test
│   ├── test_multiturn_voice.py  # Multi-turn conversation test
│   └── test_outbound_call.py # Real outbound call test
├── pyproject.toml
├── .env.example              # Leading dot (industry standard)
├── .gitignore
├── .pre-commit-config.yaml
├── Dockerfile
└── README.md
```

No exceptions. S2S, pipeline, and framework examples all use this structure.

## Config Constant Placement

Constants live where they are consumed:

**`server.py`** owns (duplicated in inbound/outbound — each file is self-contained):
- `SERVER_PORT`, `PLIVO_AUTH_ID`, `PLIVO_AUTH_TOKEN`, `PLIVO_PHONE_NUMBER`, `PUBLIC_URL`

**`agent.py`** owns:
- API keys, model names, voice names, API URLs
- `PLIVO_CHUNK_SIZE = 160` (used in `_send_to_plivo`)
- `SYSTEM_PROMPT` (loaded from `system_prompt.md`)

**`utils.py`** owns only what its functions consume:
- Audio sample rates: `PLIVO_SAMPLE_RATE`, `{API}_SAMPLE_RATE`, `VAD_SAMPLE_RATE`
- VAD params (native only): `VAD_START_THRESHOLD`, `VAD_END_THRESHOLD`, `VAD_MIN_SILENCE_MS`, `VAD_CHUNK_SAMPLES`
- `DEFAULT_COUNTRY_CODE`

## utils.py Requirements

Only utility functions and their internal constants. No server or agent config.

Required functions:
- `ulaw_to_pcm(ulaw_data: bytes) -> bytes` — G.711 decode table
- `pcm_to_ulaw(pcm_data: bytes) -> bytes` — G.711 encode
- `resample_audio(audio_data: bytes, input_rate: int, output_rate: int) -> bytes`
- `plivo_to_{api}(mulaw_8k: bytes) -> bytes` — Plivo audio to API format
- `{api}_to_plivo(pcm: bytes) -> bytes` — API audio to Plivo format
- `normalize_phone_number(phone: str, default_region: str) -> str`

For native examples, also:
- `plivo_to_vad(mulaw_8k: bytes) -> np.ndarray` — float32 16kHz for Silero
- `SileroVADProcessor` class (reference: `grok3-voice-native/utils.py`)

For framework examples: no VAD in utils (framework handles it).

## VAD Strategy

**Native examples**: client-side Silero VAD (`SileroVADProcessor`).
- VAD runs in `plivo_rx` task alongside audio forwarding
- Speech start during AI response triggers barge-in (`response.cancel` or equivalent)
- Speech end triggers turn commit (`input_audio_buffer.commit` + `response.create` or equivalent)
- Reference: `grok3-voice-native/utils.py` (SileroVADProcessor), `grok3-voice-native/inbound/agent.py` (integration)

**Framework examples** (Pipecat/LiveKit): use `vad_enabled=True` in transport params. No separate Silero.

## Audio Pipeline Rules

- `PLIVO_CHUNK_SIZE = 160` — exactly 20ms at 8kHz mono μ-law. Defined in `agent.py._send_to_plivo()`.
- Plivo WebSocket sends/receives base64 μ-law at 8kHz
- playAudio JSON format: `{"event": "playAudio", "media": {"contentType": "audio/x-mulaw", "sampleRate": 8000, "payload": "<base64>"}}`
- Answer webhook returns `<Stream>` XML: `bidirectional=True`, `keepCallAlive=True`, `contentType="audio/x-mulaw;rate=8000"`

## Agent Structure

**Native orchestration**: custom agent class with these methods:
- `__init__`, `run()`, `_run_streaming_tasks()` (3 concurrent tasks)
- `_receive_from_plivo()` — plivo_rx: decode audio, run VAD, forward to API
- `_receive_from_{api}()` — api_rx: receive API events, queue audio for plivo
- `_send_to_plivo()` — plivo_tx: chunk audio to 160 bytes, send playAudio
- Public `run_agent()` function wraps class instantiation

**Framework orchestration**: `run_agent()` function assembles Pipeline. No custom class needed.

**Pipecat PipelineRunner signal handling**:
- Use `PipelineRunner()` (default `handle_sigterm=False`) when running inside uvicorn.
- Do NOT use `PipelineRunner(handle_sigterm=True)` — it calls `loop.add_signal_handler(signal.SIGTERM, ...)` in `__init__`, which **replaces** uvicorn's SIGTERM handler. After the pipeline finishes, uvicorn's handler is never restored, so uvicorn never receives a shutdown signal and the process hangs indefinitely.
- `handle_sigterm=True` is only appropriate for standalone scripts where PipelineRunner owns the process lifecycle.
- PipelineRunner idle timeout is 300s, cancel timeout is 20s — relevant for shutdown timing.

## WebSocket Protocol

1. Plivo sends `{"event": "start", "start": {"callId": "...", "streamId": "..."}}` — handle first
2. Plivo sends `{"event": "media", "media": {"payload": "<base64 μ-law>"}}` — audio data
3. Plivo sends `{"event": "stop"}` — call ended
4. Agent sends `{"event": "playAudio", "media": {...}}` — response audio
5. Agent sends `{"event": "clearAudio"}` — on barge-in to stop playback

## Asyncio Patterns (Native)

```python
# Task management — always use this pattern
tasks = [
    asyncio.create_task(self._receive_from_plivo(), name="plivo_rx"),
    asyncio.create_task(self._receive_from_{api}(ws), name="{api}_rx"),
    asyncio.create_task(self._send_to_plivo(), name="plivo_tx"),
]
try:
    done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in done:
        if task.exception():
            logger.error(f"Task {task.get_name()} failed: {task.exception()}")
finally:
    self._running = False
    for task in tasks:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
```

Note: `_pending` with underscore prefix avoids RUF059 lint warning.

## Package Management

- **Always use `uv`** — never `pip`, `pip install`, or `python -m pip`
- Each example has its own virtualenv (`.venv/` inside the example directory)
- `uv sync` to install deps, `uv add {pkg}` to add new deps, `uv run` to execute commands
- All commands run through `uv run`: `uv run pytest`, `uv run ruff check .`, `uv run python -m inbound.server`
- `uv.lock` is committed to git for reproducible builds

### Dockerfile `uv sync` and optional dependencies

Every example must include `[project.optional-dependencies]` with `observability` and `streaming` extras (reference: `gpt4.1-sarvam-elevenflashv2.5-native/pyproject.toml`). The Dockerfile's `uv sync` command must include `--extra streaming` so Redis is available at runtime. If `pyproject.toml` defines a `streaming` extra but the Dockerfile omits `--extra streaming`, the container will fail at runtime when streaming features are used.

## Git Workflow

- **Never commit directly to `main`**. Always create a feature branch first:
  `git checkout -b {example-name}` (or `git checkout -b fix/{description}` for fixes)
- Push to the `fork` remote (not `origin`, which has IP restrictions):
  `git push -u fork {branch-name}`
- Open a PR from the fork branch to `origin/main` when ready.

## Code Quality

- `from __future__ import annotations` at top of every `.py` file
- `loguru` for logging (not stdlib `logging`) — see "Logging & Observability" below
- No hardcoded API keys — always `os.getenv()`
- `python-dotenv` with `load_dotenv()` at module level
- All imports lazy where heavy (e.g., `import torch` inside methods)

## Logging & Observability

Every native agent exposes three private helpers — `self._log`, `self._logv`, `self._loge` — as the **only** way to emit log lines from agent business logic. Do not call `logger.bind(...).info(...)` / `.debug(...)` / `.error(...)` inline. The helpers centralise the structured fields consumed by hosting apps (Redis/SSE, OTel, file sinks) and keep the console output consistent across examples.

### Helper contract

```python
def _log(self, stage: str, msg: str, **extra: object) -> None:
    if LOG_LEVEL == "quiet":
        return
    elapsed = round(time.monotonic() - self._session_start, 2)
    logger.bind(
        call_id=self.call_id, elapsed_s=elapsed, stage=stage, **extra
    ).info(f"[{self.call_id}] [{elapsed:7.2f}s] [{stage}] {msg}")

def _logv(self, stage: str, msg: str, **extra: object) -> None:   # .debug, guarded by LOG_LEVEL == "verbose"
def _loge(self, stage: str, msg: str, **extra: object) -> None:   # .error, always visible
```

Three non-negotiable rules:

1. **Full `self.call_id` — never truncate.** The console prefix `[{self.call_id}]` and the `call_id=` bound field both carry the complete Plivo CallUUID. Hosting apps correlate logs to a run by exact-UUID match against their database; an 8-char prefix collides and forces downstream workarounds.

2. **`**extra` is for structured event fields.** Pass `event=`, `turn=`, `text=`, or any domain field as a keyword argument. It gets attached to the log record's bound fields (visible to structured sinks) without cluttering the console message. Example:

   ```python
   self._log(
       "turn",
       f"turn {self._turn_count}: '{transcript[:80]}'",
       event="user_text",
       turn=self._turn_count,
       text=transcript,
   )
   ```

3. **No standalone `logger.bind(event=...)` in business code.** A telemetry event and the human-readable log line are the same log call. When you have a log line that should ALSO fire a structured event, add kwargs to it — don't emit twice.

### Required structured events

Every agent must emit these four events. The hosting app (VoxLab) subscribes to them via Redis Streams; missing events break the Conversation tab, metrics dashboard, or both.

| Event | When to fire | Minimum fields | Helper to use |
|---|---|---|---|
| `user_text` | Immediately when the STT layer delivers a final transcript that will be processed as a user turn. One emit per turn. | `event="user_text"`, `turn=<int>`, `text=<str>` | `self._log("turn", …)` or `self._logv("…_rx", …)` |
| `agent_text` | Immediately when the LLM (or S2S model) produces a text response for the caller. Fire in the line where the response first exists — do not wait for TTS. One emit per agent turn. | `event="agent_text"`, `turn=<int>`, `text=<str>` | `self._log("llm", …)` |
| `turn_complete` | At end-of-playback (Plivo `playedStream`) or on barge-in during actual audio playback. Wrapped by the dedicated `_emit_turn_complete()` helper — do not inline. | `event="turn_complete"`, `turn`, per-turn latency metrics (`llm_ms`, `tts_total_ms`, `tts_ttfb_ms`, `playback_ms`), `barge_in=<bool>` | `self._emit_turn_complete(barge_in=…)` |
| `call_summary` / `session_end` | Once at session end with aggregate counters. | `turns`, `barge_ins`, `duration_s`, `ttfs_avg_ms`, `errors` | Inline `logger.bind(event="…", …).info(...)` inside the agent's `run()` finally-block (long-lived aggregation, one shot — justifiable inline) |

For `turn_complete` specifically: the emit gate in the barge-in handler must be `if self._is_playing:`, not `if task_cancelled:`. The greeting and post-TTS-pre-`playedStream` windows are real barge-ins but no asyncio task is in flight — gating on task state drops those emits.

### Log level guidance

- **`.info` (`_log`)** — every meaningful pipeline event a developer wants to see while watching a call live: session start/end, turn commits, LLM responses, TTS completion, barge-in, tool calls. A user on an open terminal should see the whole story of one call in ~30–60 lines at default settings.
- **`.debug` (`_logv`)** — per-packet stats, VAD frame numbers, queue sizes, intermediate probability values, follow-up-LLM latency breakdowns. Enabled only when `LOG_LEVEL=verbose`.
- **`.error` (`_loge`)** — anything that represents a failure the agent is recovering from (API error, websocket disconnect, STT timeout). Always visible regardless of `LOG_LEVEL`.

Do not route exception stack traces through `_log`. Use `logger.exception(...)` or the exception handler's own return path.

### Do NOT

- Call `logger.bind(...).info(...)` inline inside a method body when the same line should appear on the console. Use `self._log(stage, msg, **extra)`.
- Truncate `call_id` anywhere — not in the message prefix, not in the bound field, not in OTel span attributes. Full UUID always.
- Emit a "display event" (`user_text`, `agent_text`) behind the same gate as a "turn event" (`turn_complete`). Display events fire as soon as text is known; turn events fire when the agent turn is logically over. Coupling them delays what the UI can show by seconds.
- Duplicate the same content at two log levels. If `_log("llm", "response …: '{text[:80]}'", …)` already prints the truncated text at INFO, don't also emit a `_logv("llm", f"full: {text}")` a line later. Attach the full text via `text=` on the bound fields.

### Reference files

- `gpt5.4-deepgramnova3-groktts3-native/inbound/agent.py` — canonical pipeline-agent logging shape, including the three helpers, the `_emit_turn_complete` helper, and the folded user/agent_text emissions.
- `gemini2.5-live-native-no-vad/inbound/agent.py` — canonical S2S logging shape, showing text events folded into `_logv` for API-transcribed content.

## Lint

Ruff with: `select = ["E", "W", "F", "I", "B", "UP", "SIM", "RUF"]`, `line-length = 100`, `target-version = "py310"`

Run: `uv run ruff check .`

## Testing

**Unit tests** (`-k "unit"`): offline, no API keys needed
- `TestUnitAudioConversion`: ulaw↔pcm roundtrip, silence detection
- `TestUnitPhoneNormalization`: E.164 formatting

**Local integration** (`-k "local"`): starts server subprocess, tests WebSocket flow with real API
- `TestLocalIntegration`: health check, answer webhook XML, WebSocket audio flow

**E2E live call tests**: real Plivo calls, recording, transcription
- `test_live_call.py`: inbound call → greeting verification
- `test_outbound_call.py`: outbound call → greeting verification
- `test_multiturn_voice.py`: multi-turn + barge-in verification

Test infra: `conftest.py` sets `sys.path`, `helpers.py` has ngrok/recording/transcription utils.

**Server subprocess teardown** in `server_process` fixture — always use SIGTERM with SIGKILL fallback:
```python
os.kill(proc.pid, signal.SIGTERM)
try:
    proc.wait(timeout=5)
except subprocess.TimeoutExpired:
    proc.kill()
    proc.wait()
```
Pipecat servers may not exit on SIGTERM alone when a PipelineRunner has been active (see "Pipecat PipelineRunner signal handling" above). Native servers typically exit cleanly on SIGTERM, but the fallback pattern is safe for all examples.

Run: `uv run pytest tests/test_integration.py -v -k "unit"` (offline)

## Reference Files

- **Primary reference**: `grok3-voice-native/` — complete native example with Silero VAD
- `grok3-voice-native/utils.py` — SileroVADProcessor class, audio conversion
- `grok3-voice-native/inbound/agent.py` — native agent pattern with VAD + barge-in
- `grok3-voice-native/outbound/agent.py` — OutboundCallRecord, CallManager pattern
- `grok3-voice-native/tests/` — full test suite to replicate
- `gemini2.5-live-native-no-vad/` — alternative native pattern (SDK-based, server-side VAD, no client-side VAD)
- `gemini2.5-live-pipecat/inbound/agent.py` — framework Pipeline reference

## README Demo Description (Required)

The text between H1 (`#`) and the first H2 (`##`) in each README is displayed as the demo description in the hosting app. It must be **5 lines or fewer** but pack maximum technical detail. Use `gpt4.1mini-sarvam-elevenlabs-native/README.md` as the reference.

### Format

Text and bullet lists only — **no tables, no diagrams, no code blocks** between H1 and first H2. Write a dense description (≤5 lines) that traces the full pipeline from telephony input to audio output, naming every component along the way. Include:

- Orchestration approach (native/framework)
- Each component: service name, model/engine, protocol (WS/HTTP), audio format, sample rate, region
- VAD: engine, frame size, threshold values with empirical tuning rationale (echo vs speech probability ranges)
- Barge-in: what gets cancelled and what event is sent
- Any notable audio conversions (resample or no-resample)

### Rules

- No vague descriptions ("production-ready", "best-in-class") — every word should be a technical fact
- No tables, diagrams, or code blocks — the hosting app doesn't render them properly. Bullet lists are fine
- Do not include observed latency — that belongs in the detailed sections below
- The rest of the README (after the first H2) can use tables, diagrams, and full detail

## Slash Commands (Phase Workflow)

```
/scaffold-example {name} {description}   # Phase 1: directory structure + boilerplate
/implement-agent {name} {api-docs-url}    # Phase 2: write agent.py + utils.py
/test-example {name}                       # Phase 3: create tests + run them
/review-example {name}                     # Phase 4: quality gate checklist
/document-example {name}                   # Phase 5: README + .env.example validation
```

Each phase gets a fresh context window. Run sequentially.

## CI Validation

```bash
./scripts/validate-example.sh {example-name}
```

Exit 0 = pass, exit 1 = fail. Checks structure, lint, unit tests, config placement.
