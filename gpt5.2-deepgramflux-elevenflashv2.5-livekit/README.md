# GPT-5.2 Mini + Deepgram Flux + ElevenLabs Flash v2.5 — LiveKit Voice Agent

LiveKit Agents framework orchestration with Plivo SIP trunking. Plivo forwards inbound calls via SIP directly to LiveKit — no webhook server needed. LiveKit's dispatch rule routes each caller to a unique room where a `VoicePipelineAgent` runs the STT-LLM-TTS pipeline. STT is Deepgram Flux (`/listen/v2` WebSocket, streaming transcription optimized for conversational audio). LLM is OpenAI `gpt-5.2-mini` via Responses API with 5 function tools. TTS is ElevenLabs `eleven_flash_v2_5` (low-latency streaming synthesis). Turn detection uses Silero VAD (speech presence, barge-in trigger) + LiveKit MultilingualModel (135M transformer, semantic end-of-turn prediction on partial transcripts in a 4-turn sliding window). Noise cancellation is Krisp BVC (background voice cancellation, applied to inbound audio before STT). LiveKit handles all audio transport, format conversion, and interruption management natively — no mu-law conversion or WebSocket handling in application code.

- Inbound calls need only the agent worker — SIP trunk and dispatch rule are auto-created on startup, Plivo routes directly to LiveKit, no `/answer` or `/hangup` webhooks required
- Outbound calls use LiveKit's SIP API (`CreateSIPParticipantRequest`) to dial through a Plivo Zentrunk outbound SIP trunk, with call metadata (system prompt, initial message) passed via room metadata JSON
- Barge-in handled by `VoicePipelineAgent` with `allow_interruptions=True` — Silero VAD detects speech onset, pipeline cancels in-flight TTS

## Pipeline Architecture

```
┌──────────┐       ┌────────────┐       ┌──────────────┐       ┌──────────────────────────────────────┐
│  Phone   │──────▶│   Plivo    │──SIP─▶│   LiveKit    │──────▶│         Agent Worker                 │
│  (PSTN)  │◀──────│  Zentrunk  │◀─SIP──│  SIP Bridge  │◀──────│                                      │
└──────────┘       └────────────┘       └──────────────┘       │  ┌──────────┐  ┌───────────────────┐  │
                                                                │  │ Krisp    │  │ Deepgram Flux     │  │
                                                                │  │ BVC     │──▶│ STT (/listen/v2)  │  │
                                                                │  └──────────┘  └────────┬──────────┘  │
                                                                │                         │             │
                                                                │                ┌────────▼──────────┐  │
                                                                │                │  OpenAI GPT-5.2   │  │
                                                                │                │  mini (Responses) │  │
                                                                │                └────────┬──────────┘  │
                                                                │                         │             │
                                                                │                ┌────────▼──────────┐  │
                                                                │                │  ElevenLabs       │  │
                                                                │                │  Flash v2.5 TTS   │  │
                                                                │                └───────────────────┘  │
                                                                │  ┌──────────┐  ┌───────────────────┐  │
                                                                │  │ Silero   │  │ MultilingualModel │  │
                                                                │  │ VAD      │  │ (turn detection)  │  │
                                                                │  └──────────┘  └───────────────────┘  │
                                                                └──────────────────────────────────────┘
```

### Component Summary

| Component | Service | Model / Engine | Protocol | Notes |
|---|---|---|---|---|
| STT | Deepgram | Flux (STT v2 API) | WebSocket `/listen/v2` | Streaming, conversational audio optimized |
| LLM | OpenAI | `gpt-5.2-mini` | HTTPS (Responses API) | 5 function tools |
| TTS | ElevenLabs | `eleven_flash_v2_5` | WebSocket | Low-latency streaming synthesis |
| VAD | Silero | ONNX v5 | Local CPU | Pre-loaded in prewarm, ~32ms frames |
| Turn Detect | LiveKit | MultilingualModel | Local CPU | 135M transformer, semantic EOT |
| Noise | Krisp | BVC | Local | Background voice cancellation |
| Transport | LiveKit | SIP Bridge + WebRTC | SIP/WebRTC | All audio format conversion handled |
| Telephony | Plivo | Zentrunk SIP | SIP | Inbound + outbound |

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- LiveKit server — [LiveKit Cloud](https://cloud.livekit.io/) (free tier) or [self-hosted](https://docs.livekit.io/home/self-hosting/local/)
- OpenAI, Deepgram, ElevenLabs API keys
- Plivo account with Zentrunk SIP trunking

## Quick Start

### 1. Install dependencies

```bash
cd gpt5.2-deepgramflux-elevenflashv2.5-livekit
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your credentials
```

### 3. Configure Plivo Zentrunk

In the Plivo Console:
1. Go to **Zentrunk → Inbound Trunks** → Create New
2. Set the SIP endpoint to your LiveKit SIP URI: `{trunk-id}.sip.livekit.cloud;transport=tcp`
3. Attach your Plivo phone number to the inbound trunk

### 4. Run the inbound agent

```bash
uv run python -m inbound.agent dev
```

That's it. The agent:
1. Creates a LiveKit inbound SIP trunk (with Krisp enabled)
2. Creates a dispatch rule (individual rooms per call)
3. Starts the worker — auto-joins rooms when callers connect

### 5. Call your Plivo number

The call routes: Phone → Plivo Zentrunk → SIP → LiveKit → Agent.

## Outbound Calls

### Start the outbound agent worker

```bash
uv run python -m outbound.agent dev
```

### Start the outbound management API (separate terminal)

```bash
uv run python -m outbound.server
```

### Initiate a call

```bash
curl -X POST "http://localhost:8000/outbound/call?\
phone_number=+15551234567&\
opening_reason=Follow%20up%20on%20your%20demo%20request&\
objective=Schedule%20a%20product%20demo"
```

### Check status

```bash
curl "http://localhost:8000/outbound/status/{call_id}"
```

## Project Structure

```
gpt5.2-deepgramflux-elevenflashv2.5-livekit/
├── inbound/
│   ├── agent.py              # Agent worker + SIP setup (all you need)
│   ├── server.py             # Optional health check
│   └── system_prompt.md
├── outbound/
│   ├── agent.py              # Agent worker + CallManager + SIP setup
│   ├── server.py             # Call initiation API
│   └── system_prompt.md
├── utils.py                  # Phone normalization
├── tests/
├── pyproject.toml
├── .env.example
├── Dockerfile
└── README.md
```

**Key difference from Pipecat/native examples**: No `/answer`, `/hangup`, or `/ws` webhooks. LiveKit handles all call routing via SIP — the agent worker is the only process needed for inbound.

## Local Development

LiveKit always requires a server. Options:

1. **LiveKit Cloud** (recommended): Free tier at [cloud.livekit.io](https://cloud.livekit.io). Set `LIVEKIT_URL=wss://your-project.livekit.cloud`
2. **Self-hosted**: Run `livekit-server` locally. Set `LIVEKIT_URL=ws://localhost:7880`. See [local setup guide](https://docs.livekit.io/home/self-hosting/local/)

## Testing

```bash
# Unit tests (offline, no API keys)
uv run pytest tests/test_integration.py -v -k "unit"

# Local integration (starts outbound server, tests endpoints)
uv run pytest tests/test_integration.py -v -k "local"

# E2E with LiveKit (requires LiveKit credentials)
uv run pytest tests/test_e2e_live.py -v
```

## License

MIT
