# GPT-5.4-mini + AssemblyAI + Cartesia LiveKit Voice Agent

Real-time voice agent using LiveKit Agents framework with AssemblyAI STT, OpenAI GPT-5.4-mini LLM, and Cartesia TTS for Plivo telephony.

## Features

- **LiveKit Agents Framework**: VoicePipelineAgent for modular voice AI pipelines
- **AssemblyAI STT**: Real-time speech-to-text transcription
- **OpenAI GPT-5.4-mini**: Fast, capable language model for conversation
- **Cartesia TTS**: Low-latency text-to-speech synthesis
- **Silero VAD**: Voice activity detection for natural turn-taking
- **Turn Detection**: LiveKit end-of-utterance model for accurate turn boundaries
- **Plivo Telephony**: Inbound and outbound call support via SIP

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- OpenAI API key
- AssemblyAI API key
- Cartesia API key
- LiveKit server (Cloud or self-hosted) with SIP bridge
- Plivo account with a phone number and SIP trunk
- ngrok (for local development)

## Quick Start

### 1. Install dependencies

```bash
cd gpt5.4mini-assemblyai-cartesia-livekit
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```bash
OPENAI_API_KEY=your_openai_api_key
ASSEMBLYAI_API_KEY=your_assemblyai_api_key
CARTESIA_API_KEY=your_cartesia_api_key
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret
LIVEKIT_SIP_URI=sip.livekit.cloud
PLIVO_AUTH_ID=your_plivo_auth_id
PLIVO_AUTH_TOKEN=your_plivo_auth_token
PLIVO_PHONE_NUMBER=+1234567890
PUBLIC_URL=https://your-ngrok-url.ngrok-free.app
```

### 3. Configure LiveKit SIP

Set up a SIP trunk in LiveKit that connects to your Plivo SIP domain. Configure SIP dispatch rules to route incoming calls to LiveKit rooms where the agent will join.

### 4. Start ngrok

```bash
ngrok http 8000
```

Copy the ngrok URL to `PUBLIC_URL` in your `.env` file.

### 5. Run the inbound server

```bash
uv run python -m inbound.server
```

### 6. Run the outbound server

```bash
uv run python -m outbound.server
```

### 7. Make a test call

Call your Plivo phone number to test the inbound agent, or use the outbound API:

```bash
curl -X POST "http://localhost:8000/outbound/call?phone_number=+1234567890&opening_reason=a+quick+demo"
```

## Project Structure

```
gpt5.4mini-assemblyai-cartesia-livekit/
├── utils.py                  # Phone normalization utilities
├── inbound/
│   ├── __init__.py
│   ├── agent.py              # LiveKit VoicePipelineAgent (inbound)
│   ├── server.py             # FastAPI server + LiveKit worker
│   └── system_prompt.md      # System prompt for inbound calls
├── outbound/
│   ├── __init__.py
│   ├── agent.py              # LiveKit agent + CallManager
│   ├── server.py             # FastAPI server with outbound call management
│   └── system_prompt.md      # System prompt for outbound calls
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # sys.path setup
│   ├── helpers.py            # ngrok, recording, transcription utilities
│   ├── test_integration.py   # Unit + local integration tests
│   ├── test_e2e_live.py      # E2E with real APIs (no phone call)
│   ├── test_live_call.py     # Real inbound call test
│   ├── test_multiturn_voice.py  # Multi-turn conversation test
│   └── test_outbound_call.py # Real outbound call test
├── pyproject.toml
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── Dockerfile
└── README.md
```

## How It Works

```
┌─────────┐     ┌─────────────┐     ┌──────────────┐
│  Phone  │────▶│   Plivo     │────▶│   LiveKit    │
│  Call   │◀────│  (SIP)      │◀────│   Server     │
└─────────┘     └─────────────┘     └──────┬───────┘
                                           │
                              WebRTC Room  │
                                           ▼
                                    ┌──────────────┐
                                    │  LiveKit     │
                                    │  Agent       │
                                    │              │
                                    │ ┌──────────┐ │
                                    │ │ Silero   │ │
                                    │ │ VAD      │ │
                                    │ ├──────────┤ │
                                    │ │ Assembly │ │
                                    │ │ AI STT   │ │
                                    │ ├──────────┤ │
                                    │ │ GPT-5.4  │ │
                                    │ │ mini LLM │ │
                                    │ ├──────────┤ │
                                    │ │ Cartesia │ │
                                    │ │ TTS      │ │
                                    │ └──────────┘ │
                                    └──────────────┘
```

1. **Incoming Call**: Plivo receives call and routes via SIP to LiveKit
2. **Room Creation**: LiveKit creates a room with the SIP participant
3. **Agent Dispatch**: LiveKit agent worker joins the room
4. **Voice Pipeline**: Audio flows through VAD, STT, LLM, TTS pipeline
5. **Response**: Synthesized audio streams back through LiveKit to Plivo

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `ASSEMBLYAI_API_KEY` | AssemblyAI API key | Required |
| `CARTESIA_API_KEY` | Cartesia API key | Required |
| `LIVEKIT_URL` | LiveKit server URL | Required |
| `LIVEKIT_API_KEY` | LiveKit API key | Required |
| `LIVEKIT_API_SECRET` | LiveKit API secret | Required |
| `LIVEKIT_SIP_URI` | LiveKit SIP bridge URI | Required |
| `PLIVO_AUTH_ID` | Plivo Auth ID | Required |
| `PLIVO_AUTH_TOKEN` | Plivo Auth Token | Required |
| `PLIVO_PHONE_NUMBER` | Plivo phone number | Required |
| `PUBLIC_URL` | Public URL for webhooks | Required |
| `SERVER_PORT` | Server port | `8000` |
| `OPENAI_MODEL` | OpenAI model name | `gpt-5.4-mini` |
| `CARTESIA_VOICE` | Cartesia voice ID | `79a125e8-cd45-4c13-8a67-188112f4dd22` |

## Testing

```bash
# Unit tests (offline, no API keys needed)
uv run pytest tests/test_integration.py -v -k "unit"

# Local integration tests (starts server subprocess)
uv run pytest tests/test_integration.py -v -k "local"

# E2E with real APIs (requires API keys)
uv run pytest tests/test_e2e_live.py -v

# Live call tests (requires all credentials + LiveKit server)
uv run pytest tests/test_live_call.py -v -s
uv run pytest tests/test_outbound_call.py -v -s
uv run pytest tests/test_multiturn_voice.py -v -s
```

## Comparison with Other Examples

| Feature | This (LiveKit) | Pipecat | Native |
|---------|---------------|---------|--------|
| Framework | LiveKit Agents | Pipecat | None (direct API) |
| Audio Transport | WebRTC via LiveKit | Plivo WebSocket | Plivo WebSocket |
| VAD | Silero (framework) | Silero (framework) | Silero (custom) |
| Turn Detection | EOUModel | Framework VAD | Custom logic |
| SIP Integration | LiveKit SIP bridge | N/A (WebSocket) | N/A (WebSocket) |
| Code Complexity | Low | Low | High |

## Troubleshooting

### No audio on call

- Verify all API keys are correct (OpenAI, AssemblyAI, Cartesia)
- Check LiveKit server is running and accessible
- Verify SIP trunk configuration between Plivo and LiveKit
- Review server logs for connection errors

### Agent does not respond

- Check LiveKit agent worker is running (look for "Starting LiveKit agent worker" in logs)
- Verify LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET are set
- Ensure SIP dispatch rules are configured in LiveKit

### SIP connection fails

- Verify LIVEKIT_SIP_URI is correct
- Check Plivo SIP trunk configuration
- Ensure firewall allows SIP traffic (ports 5060/5061)

## License

MIT
