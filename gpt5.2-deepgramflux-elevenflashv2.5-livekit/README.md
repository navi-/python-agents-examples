# GPT-5.2 Mini + Deepgram Flux + ElevenLabs Flash v2.5 — LiveKit Voice Agent

LiveKit Agents framework orchestration with Plivo SIP trunking. Inbound calls hit Plivo, route via SIP to a LiveKit room where a `VoicePipelineAgent` runs the full STT-LLM-TTS pipeline. STT is Deepgram Flux (`/listen/v2` WebSocket, streaming transcription optimized for conversational audio). LLM is OpenAI `gpt-5.2-mini` via Responses API with 5 function tools. TTS is ElevenLabs `eleven_flash_v2_5` (low-latency streaming synthesis). VAD is Silero (pre-loaded at worker startup via `prewarm`, runs locally on CPU). Noise cancellation is Krisp BVC (background voice cancellation, applied to inbound audio before STT). LiveKit handles all audio transport, format conversion, and interruption management natively.

- Outbound calls use LiveKit's SIP API (`CreateSIPParticipantRequest`) to dial through a Plivo SIP trunk, with call metadata (system prompt, initial message) passed via room metadata JSON
- Barge-in is handled by LiveKit's `VoicePipelineAgent` with `allow_interruptions=True` — Silero VAD detects speech onset, pipeline cancels in-flight TTS, and new STT input is processed immediately
- Two-process architecture: FastAPI management server (webhooks, call initiation, status) + LiveKit agent worker (audio processing, AI pipeline)

## Pipeline Architecture

```
┌──────────┐       ┌────────────┐       ┌──────────────┐       ┌──────────────────────────────────────┐
│  Phone   │──────▶│   Plivo    │──────▶│   LiveKit    │──────▶│         Agent Worker                 │
│  (PSTN)  │◀──────│  Gateway   │◀──────│  SIP Bridge  │◀──────│                                      │
└──────────┘       └────────────┘       └──────────────┘       │  ┌──────────┐  ┌───────────────────┐  │
                    SIP Trunk                                   │  │ Krisp    │  │ Deepgram Flux     │  │
                    (bidirectional)                              │  │ BVC     │──▶│ STT (/listen/v2)  │  │
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
                                                                │  ┌──────────┐                         │
                                                                │  │ Silero   │ (VAD for turn-taking)   │
                                                                │  │ VAD      │                         │
                                                                │  └──────────┘                         │
                                                                └──────────────────────────────────────┘
```

### Component Summary

| Component | Service | Model / Engine | Protocol | Notes |
|---|---|---|---|---|
| STT | Deepgram | Flux (via STT v2 API) | WebSocket `/listen/v2` | Streaming, conversational audio optimized |
| LLM | OpenAI | `gpt-5.2-mini` | HTTPS (Responses API) | 5 function tools |
| TTS | ElevenLabs | `eleven_flash_v2_5` | WebSocket | Low-latency streaming synthesis |
| VAD | Silero | ONNX v5 | Local CPU | Pre-loaded in prewarm, ~32ms frames |
| Noise | Krisp | BVC | Local | Background voice cancellation |
| Transport | LiveKit | SIP Bridge + WebRTC | SIP/WebRTC | Handles audio format conversion |
| Telephony | Plivo | SIP Trunk | SIP | Inbound + outbound via SIP |

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- [LiveKit Cloud](https://cloud.livekit.io/) account (or self-hosted LiveKit server)
- OpenAI API key
- Deepgram API key
- ElevenLabs API key
- Plivo account with SIP trunking enabled
- ngrok (for local development webhooks)

## Quick Start

### 1. Install dependencies

```bash
cd gpt5.2-deepgramflux-elevenflashv2.5-livekit
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your credentials (see Configuration section below).

### 3. Set up LiveKit SIP trunk

1. Go to [LiveKit Cloud](https://cloud.livekit.io/) and create a project
2. Create an **inbound SIP trunk** — note the SIP URI
3. Create an **outbound SIP trunk** — configure with Plivo SIP credentials, note the trunk ID
4. Create a **dispatch rule** to route incoming SIP calls to agent rooms
5. Set `LIVEKIT_SIP_URI` and `LIVEKIT_SIP_TRUNK_ID` in `.env`

### 4. Configure Plivo SIP

Configure your Plivo phone number to forward calls via SIP to the LiveKit SIP URI.
Alternatively, start the management server which auto-configures Plivo webhooks:

```bash
# Start ngrok for webhook access
ngrok http 8000
# Set PUBLIC_URL in .env to the ngrok URL
```

### 5. Start the agent worker

```bash
uv run python -m inbound.agent dev
```

### 6. Start the management server (separate terminal)

```bash
uv run python -m inbound.server
```

### 7. Make a test call

Call your Plivo phone number. The call routes via SIP to LiveKit, and the agent responds.

## Outbound Calls

### Start the outbound agent worker

```bash
uv run python -m outbound.agent dev
```

### Start the outbound management server

```bash
uv run python -m outbound.server
```

### Initiate an outbound call

```bash
curl -X POST "http://localhost:8000/outbound/call?\
phone_number=+15551234567&\
opening_reason=Follow%20up%20on%20your%20demo%20request&\
objective=Schedule%20a%20product%20demo"
```

### Check call status

```bash
curl "http://localhost:8000/outbound/status/{call_id}"
```

## Project Structure

```
gpt5.2-deepgramflux-elevenflashv2.5-livekit/
├── inbound/
│   ├── __init__.py
│   ├── agent.py              # LiveKit VoicePipelineAgent (inbound)
│   ├── server.py             # FastAPI: /answer, /hangup (SIP routing)
│   └── system_prompt.md      # System prompt for inbound calls
├── outbound/
│   ├── __init__.py
│   ├── agent.py              # LiveKit agent + CallManager
│   ├── server.py             # FastAPI: /outbound/call, /outbound/status
│   └── system_prompt.md      # System prompt for outbound calls
├── utils.py                  # Phone normalization, audio conversion
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── helpers.py
│   ├── test_integration.py   # Unit + local integration tests
│   ├── test_e2e_live.py      # E2E with real LiveKit (no phone call)
│   ├── test_live_call.py     # Real inbound call test
│   ├── test_multiturn_voice.py
│   └── test_outbound_call.py
├── pyproject.toml
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── Dockerfile
└── README.md
```

## How It Works

### Inbound Call Flow

1. **Incoming Call**: Phone call arrives at Plivo phone number
2. **Plivo Webhook**: Plivo hits `/answer` on the management server
3. **SIP Routing**: Server returns XML that bridges the call via SIP to LiveKit
4. **LiveKit Room**: LiveKit SIP bridge creates a room with the caller as a participant
5. **Agent Joins**: LiveKit agent worker auto-joins the room
6. **Audio Pipeline**: `VoicePipelineAgent` processes: Krisp BVC → Deepgram Flux STT → GPT-5.2-mini → ElevenLabs TTS
7. **Response**: Audio streams back through LiveKit → SIP → Plivo → phone

### Outbound Call Flow

1. **API Call**: POST to `/outbound/call` with phone number and context
2. **LiveKit SIP**: Server creates a SIP participant via LiveKit API
3. **Plivo Dials**: LiveKit sends SIP INVITE through Plivo's SIP trunk
4. **Callee Answers**: Plivo connects the call, SIP participant joins the LiveKit room
5. **Agent Joins**: Outbound agent worker joins, reads metadata, starts greeting

### Two-Process Architecture

Unlike Pipecat examples that run everything in one process, LiveKit requires:

1. **Management Server** (`python -m inbound.server`): FastAPI server for Plivo webhooks and call management APIs
2. **Agent Worker** (`python -m inbound.agent dev`): LiveKit worker that connects to the LiveKit server and auto-joins rooms

Both must be running for the system to work.

## Configuration

| Variable | Description | Default |
|---|---|---|
| `LIVEKIT_URL` | LiveKit server URL | Required |
| `LIVEKIT_API_KEY` | LiveKit API key | Required |
| `LIVEKIT_API_SECRET` | LiveKit API secret | Required |
| `LIVEKIT_SIP_URI` | Inbound SIP trunk URI | Required |
| `LIVEKIT_SIP_TRUNK_ID` | Outbound SIP trunk ID | Required for outbound |
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `DEEPGRAM_API_KEY` | Deepgram API key | Required |
| `ELEVEN_API_KEY` | ElevenLabs API key | Required |
| `PLIVO_AUTH_ID` | Plivo Auth ID | Required |
| `PLIVO_AUTH_TOKEN` | Plivo Auth Token | Required |
| `PLIVO_PHONE_NUMBER` | Plivo phone number | Required |
| `PUBLIC_URL` | Public URL for webhooks | Required |
| `LLM_MODEL` | OpenAI model name | `gpt-5.2-mini` |
| `ELEVEN_VOICE` | ElevenLabs voice name | `jessica` |
| `ELEVEN_MODEL` | ElevenLabs model | `eleven_flash_v2_5` |
| `SERVER_PORT` | Management server port | `8000` |

## Comparison with Pipecat Examples

| Feature | LiveKit | Pipecat |
|---------|---------|---------|
| Audio transport | LiveKit rooms (WebRTC/SIP) | Direct Plivo WebSocket |
| Telephony bridge | SIP trunking | Plivo WebSocket `<Stream>` |
| VAD | Silero via livekit-plugins-silero | Silero via pipecat |
| Noise cancellation | Krisp BVC (built-in) | Not available |
| Process model | Two processes (server + worker) | Single process |
| Scaling | LiveKit handles room routing | Manual scaling |
| Audio format | Handled by LiveKit | Manual μ-law conversion |

## Testing

```bash
# Unit tests (offline, no API keys needed)
uv run pytest tests/test_integration.py -v -k "unit"

# Local integration (starts server, tests endpoints)
uv run pytest tests/test_integration.py -v -k "local"

# E2E with LiveKit (requires LiveKit credentials)
uv run pytest tests/test_e2e_live.py -v
```

## Troubleshooting

### Agent doesn't respond to calls

- Verify the agent worker is running (`python -m inbound.agent dev`)
- Check LiveKit SIP trunk configuration
- Ensure Plivo is routing calls to the correct SIP URI
- Review agent worker logs for connection errors

### No audio heard

- Verify all API keys are set (OpenAI, Deepgram, ElevenLabs)
- Check LiveKit Cloud dashboard for room activity
- Ensure SIP trunk is properly configured with Plivo credentials

### Outbound calls fail

- Verify `LIVEKIT_SIP_TRUNK_ID` is set
- Check that the outbound SIP trunk has Plivo SIP credentials
- Ensure the outbound agent worker is running

### High latency

- Check Deepgram Flux STT latency in agent logs
- Verify LiveKit server region is close to your users
- Consider ElevenLabs voice/model selection for lower latency

## License

MIT
