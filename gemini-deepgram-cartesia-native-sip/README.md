# Gemini + Deepgram + Cartesia SIP Voice Agent

Voice agent that handles SIP calls directly — no WebSocket streaming, no frameworks. The server manages SIP signaling (INVITE, BYE), RTP audio (G.711 μ-law over UDP), and pipes audio through **Deepgram STT → Gemini LLM → Cartesia TTS** with **Silero VAD** for barge-in.

## Architecture

```
Phone Call → Plivo SIP Trunk → SIP INVITE → Our Server
                                                │
                                          SIP Signaling
                                          (INVITE/BYE/ACK)
                                                │
                                          RTP Audio (UDP)
                                          G.711 μ-law 8kHz
                                                │
                            ┌───────────────────┼───────────────────┐
                            │                   │                   │
                       Deepgram STT       Silero VAD          RTP Send
                       (WebSocket)        (barge-in)         (20ms frames)
                            │                   │                   ▲
                            ▼                   │                   │
                       Gemini LLM          interrupts ──────► drain queue
                       (text chat)                                  │
                            │                                       │
                            ▼                                       │
                       Cartesia TTS ─── resample 24k→8k ──────────┘
                       (HTTP API)
```

## Call Flow

1. **SIP INVITE** arrives on UDP port 5060
2. Server parses SDP, allocates RTP port, sends **200 OK** with SDP answer
3. **RTP session** starts: bidirectional G.711 μ-law audio over UDP
4. Incoming audio → PCM16 8kHz → **Deepgram STT** (streaming WebSocket)
5. Transcript → **Gemini LLM** → response text
6. Response → **Cartesia TTS** → PCM16 24kHz → resample to 8kHz → RTP
7. **Silero VAD** runs on incoming audio for barge-in detection
8. On **SIP BYE**: stop RTP, clean up resources

## Quick Start

```bash
# Install dependencies
uv sync

# Copy and edit environment variables
cp .env.example .env
# Edit .env with your API keys

# Run inbound server
uv run python -m inbound.server

# Or run outbound server
uv run python -m outbound.server
```

## Configuration

### SIP Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `SIP_HOST` | `0.0.0.0` | SIP server bind address |
| `SIP_PORT` | `5060` | SIP UDP port |
| `RTP_PORT_START` | `10000` | Start of RTP port range |
| `RTP_PORT_END` | `20000` | End of RTP port range |
| `SERVER_PORT` | `8000` | FastAPI health/status HTTP port |

### API Keys

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Google Gemini API key |
| `DEEPGRAM_API_KEY` | Deepgram API key |
| `CARTESIA_API_KEY` | Cartesia API key |
| `PLIVO_AUTH_ID` | Plivo auth ID (for outbound calls) |
| `PLIVO_AUTH_TOKEN` | Plivo auth token |

### Model Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model for LLM |
| `DEEPGRAM_MODEL` | `nova-2-phonecall` | Deepgram STT model |
| `CARTESIA_MODEL` | `sonic-2` | Cartesia TTS model |
| `CARTESIA_VOICE_ID` | `79a125e8-...` | Cartesia voice ID |

## Plivo SIP Trunk Setup

1. Create a Plivo SIP trunk (or use existing one)
2. Set the **Origination URI** to point to your server: `sip:YOUR_SERVER_IP:5060`
3. Assign a phone number to the trunk
4. Calls to that number will arrive as SIP INVITEs on your server

## Inbound Server

Handles incoming SIP calls:

```bash
uv run python -m inbound.server
```

- **SIP**: Listens on UDP 5060 for INVITE/BYE
- **HTTP**: Health check on port 8000 (`GET /`, `GET /calls`)

## Outbound Server

Places outbound calls via SIP trunk:

```bash
uv run python -m outbound.server
```

API endpoints:
- `POST /outbound/call` — Initiate a call
- `GET /outbound/calls` — List active calls
- `GET /outbound/call/{id}` — Get call status

Example:
```bash
curl -X POST http://localhost:8000/outbound/call \
  -H "Content-Type: application/json" \
  -d '{
    "phone_number": "+15551234567",
    "opening_reason": "your free trial signup",
    "objective": "qualify the lead"
  }'
```

## Testing

```bash
# Unit tests (no external services needed)
uv run pytest tests/test_integration.py -v -k "unit"

# All tests including integration
uv run pytest tests/test_integration.py -v

# End-to-end live tests (requires running server + SIP trunk)
uv run pytest tests/test_e2e_live.py -v
```

## Docker

```bash
docker build -t voice-agent-sip .

# Inbound (default)
docker run -p 8000:8000 -p 5060:5060/udp -p 10000-20000:10000-20000/udp \
  --env-file .env voice-agent-sip

# Outbound
docker run -p 8000:8000 --env-file .env voice-agent-sip \
  uv run python -m outbound.server
```

## Key Design Decisions

- **Direct SIP/RTP**: No WebSocket streaming layer (Plivo, LiveKit). Raw UDP for lowest latency.
- **Absolute-time RTP scheduling**: Uses monotonic clock for precise 20ms frame timing, preventing drift.
- **Silero VAD for barge-in**: Client-side VAD detects speech during agent response → cancels LLM/TTS → drains audio queue → silence.
- **Separate STT/LLM/TTS**: Each component is independently replaceable. No framework lock-in.
- **asyncio throughout**: All I/O is async. SIP, RTP, STT, LLM, TTS all run concurrently.
