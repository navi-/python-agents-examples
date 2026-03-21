# GPT + AssemblyAI + Cartesia — Native Voice Agent

Production-ready voice agent using **GPT-5.4-mini** for conversation, **AssemblyAI Universal-3 Pro** for speech-to-text with smart-turn detection, **Cartesia Sonic-3** for text-to-speech, and **Plivo** for telephony. No orchestration framework — raw WebSockets and asyncio.

## Features

- **Speech-to-Text**: AssemblyAI Universal-3 Pro with immutable transcription
- **Smart-Turn Detection**: AssemblyAI's ML-based end-of-turn detection (no client-side VAD needed)
- **LLM**: OpenAI GPT-5.4-mini with streaming and function calling
- **Text-to-Speech**: Cartesia Sonic-3 with WebSocket streaming and context continuations
- **Barge-in / Interruption**: Full interruption support — cancels LLM stream, Cartesia context, and clears Plivo audio
- **Function Calling**: Order status, SMS, callbacks, transfers, call ending
- **Inbound + Outbound**: Separate servers for incoming and outgoing calls
- **Auto-config**: Plivo webhook auto-configuration on startup

## Architecture

```
Plivo (μ-law 8kHz) ──→ AssemblyAI STT (PCM16 16kHz) ──→ GPT-5.4-mini (text)
                                                              │
Plivo (μ-law 8kHz) ←── Cartesia TTS (PCM16 24kHz) ←─────────┘
```

### Interruption Flow

When the user speaks while the agent is responding:

1. AssemblyAI detects new speech (Turn event with transcript)
2. Agent cancels the active GPT streaming request
3. Agent cancels the Cartesia TTS context
4. Agent clears the audio send queue
5. Agent sends `clearAudio` to Plivo to stop playback immediately
6. AssemblyAI's smart-turn model detects end-of-utterance
7. New GPT response is generated with the user's full utterance

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- [OpenAI API key](https://platform.openai.com/)
- [AssemblyAI API key](https://www.assemblyai.com/)
- [Cartesia API key](https://cartesia.ai/)
- [Plivo account](https://www.plivo.com/) with a phone number
- [ngrok](https://ngrok.com/) for local development

## Quick Start

### 1. Install dependencies

```bash
cd gpt-assemblyai-cartesia-native
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Start ngrok

```bash
ngrok http 8000
# Copy the HTTPS URL to PUBLIC_URL in .env
```

### 4. Run the server

```bash
# Inbound (receive calls)
uv run python -m inbound.server

# Outbound (make calls)
uv run python -m outbound.server
```

### 5. Test

```bash
# Call your Plivo phone number, or:
uv run pytest tests/test_integration.py -v -k "unit"
```

## Project Structure

```
gpt-assemblyai-cartesia-native/
├── inbound/
│   ├── agent.py              # VoiceAgent: AssemblyAI + GPT + Cartesia pipeline
│   ├── server.py             # FastAPI: /answer, /ws, /hangup
│   └── system_prompt.md      # System prompt for inbound calls
├── outbound/
│   ├── agent.py              # VoiceAgent + CallManager for outbound
│   ├── server.py             # FastAPI: /outbound/call, /outbound/ws, etc.
│   └── system_prompt.md      # System prompt for outbound calls
├── utils.py                  # Audio conversion (μ-law, PCM, resampling)
├── tests/
│   ├── conftest.py           # sys.path setup
│   ├── helpers.py            # ngrok, recording, transcription utils
│   └── test_integration.py   # Unit + integration tests
├── pyproject.toml
├── .env.example
├── Dockerfile
└── README.md
```

## Audio Pipeline

| Stage | Format | Sample Rate |
|-------|--------|-------------|
| Plivo WebSocket | μ-law (base64) | 8 kHz |
| AssemblyAI input | PCM16 signed LE | 16 kHz |
| AssemblyAI output | Text (JSON) | — |
| GPT input/output | Text | — |
| Cartesia output | PCM16 signed LE | 24 kHz |
| Plivo playback | μ-law (base64) | 8 kHz |

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | — |
| `OPENAI_MODEL` | GPT model ID | `gpt-5.4-mini` |
| `ASSEMBLYAI_API_KEY` | AssemblyAI API key | — |
| `ASSEMBLYAI_MODEL` | Streaming model | `universal-streaming-english` |
| `CARTESIA_API_KEY` | Cartesia API key | — |
| `CARTESIA_MODEL` | TTS model | `sonic-3` |
| `CARTESIA_VOICE_ID` | Voice ID | `6ccbfb76-...` (Tessa) |
| `END_OF_TURN_CONFIDENCE` | Turn detection threshold (0-1) | `0.4` |
| `PLIVO_AUTH_ID` | Plivo auth ID | — |
| `PLIVO_AUTH_TOKEN` | Plivo auth token | — |
| `PLIVO_PHONE_NUMBER` | Your Plivo number | — |
| `PUBLIC_URL` | ngrok HTTPS URL | — |
| `SERVER_PORT` | Server port | `8000` |

## Testing

```bash
# Unit tests (offline, no API keys needed)
uv run pytest tests/test_integration.py -v -k "unit"

# Local integration (starts server, needs API keys)
uv run pytest tests/test_integration.py -v -k "local"

# AssemblyAI integration (needs ASSEMBLYAI_API_KEY)
uv run pytest tests/test_integration.py -v -k "assemblyai"

# Plivo integration (needs Plivo credentials)
uv run pytest tests/test_integration.py -v -k "plivo"
```

## Deployment

```bash
# Docker
docker build -t gpt-assemblyai-cartesia-agent .
docker run -p 8000:8000 --env-file .env gpt-assemblyai-cartesia-agent

# Outbound server
docker run -p 8000:8000 --env-file .env gpt-assemblyai-cartesia-agent \
  uv run python -m outbound.server
```

## Troubleshooting

- **No audio from agent**: Check all three API keys (OpenAI, AssemblyAI, Cartesia)
- **Slow turn detection**: Lower `END_OF_TURN_CONFIDENCE` (e.g., 0.3) for faster responses
- **Interruption not working**: Ensure Plivo WebSocket is sending audio during playback
- **WebSocket disconnects**: Check ngrok is running and PUBLIC_URL matches
- **Cartesia timeout**: WebSocket disconnects after ~5min idle; reconnection is automatic per-call
