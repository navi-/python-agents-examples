# Gemini + Deepgram + Cartesia Voice Agent (Native)

A voice agent that uses Google Gemini for conversation, Deepgram for speech-to-text, Cartesia for text-to-speech, and Silero VAD for voice activity detection. This implementation uses direct API integration without any orchestration frameworks.

## Features

- Native orchestration with asyncio tasks (no frameworks)
- Google Gemini LLM for intelligent conversation
- Deepgram real-time STT for accurate transcription
- Cartesia TTS for natural-sounding speech
- Silero VAD for client-side voice activity detection and barge-in
- Plivo telephony integration (inbound and outbound calls)
- Outbound call management with campaign tracking

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- [ngrok](https://ngrok.com/) for local development
- API keys for:
  - [Plivo](https://www.plivo.com/) - Telephony
  - [Google AI Studio](https://aistudio.google.com/) - Gemini API
  - [Deepgram](https://deepgram.com/) - Speech-to-text
  - [Cartesia](https://cartesia.ai/) - Text-to-speech

## Quick Start

1. **Navigate to the project:**
   ```bash
   cd gemini-deepgram-cartesia-native
   ```

2. **Create environment file:**
   ```bash
   cp .env.example .env
   ```

3. **Configure your `.env` file** with your API keys and Plivo phone number.

4. **Install dependencies:**
   ```bash
   uv sync
   ```

5. **Start ngrok** (in a separate terminal):
   ```bash
   ngrok http 8000
   ```

6. **Update `PUBLIC_URL`** in your `.env` with the ngrok HTTPS URL.

7. **Run the inbound server:**
   ```bash
   uv run python -m inbound.server
   ```

8. **Call your Plivo phone number** to test the voice agent.

For outbound calls:
```bash
uv run python -m outbound.server
```

## Project Structure

```
gemini-deepgram-cartesia-native/
├── inbound/
│   ├── __init__.py
│   ├── agent.py              # Voice agent (Gemini + Deepgram + Cartesia + VAD)
│   ├── server.py             # FastAPI: /answer, /ws, /hangup
│   └── system_prompt.md      # System prompt for inbound calls
├── outbound/
│   ├── __init__.py
│   ├── agent.py              # Voice agent + OutboundCallRecord, CallManager
│   ├── server.py             # FastAPI: /outbound/call, /outbound/ws, etc.
│   └── system_prompt.md      # System prompt for outbound calls
├── utils.py                  # Audio conversion, Silero VAD, phone utils
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── helpers.py
│   ├── test_integration.py   # Unit + local integration tests
│   ├── test_e2e_live.py      # E2E with real APIs
│   ├── test_live_call.py     # Real inbound call test
│   ├── test_multiturn_voice.py
│   └── test_outbound_call.py # Real outbound call test
├── pyproject.toml
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── Dockerfile
└── README.md
```

## How It Works

### Audio Pipeline

```
Caller -> Plivo (u-law 8kHz) -> ulaw_to_pcm -> Deepgram STT (PCM 8kHz)
                                             -> Silero VAD (float32 16kHz)
Gemini LLM (text) -> Cartesia TTS (PCM 24kHz) -> cartesia_to_plivo (u-law 8kHz) -> Plivo -> Caller
```

### Components

- **`utils.py`**: Audio format conversion (u-law, PCM, resampling), SileroVADProcessor, phone normalization
- **`inbound/agent.py`**: VoiceAgent class with concurrent plivo_rx and plivo_tx tasks, DeepgramSTT, CartesiaTTS, GeminiLLM clients
- **`inbound/server.py`**: FastAPI server with /answer webhook, /ws WebSocket endpoint, /hangup webhook
- **`outbound/agent.py`**: Same agent plus OutboundCallRecord and CallManager for call state tracking
- **`outbound/server.py`**: FastAPI server with /outbound/call, /outbound/answer, /outbound/status endpoints

### VAD and Barge-in

The agent uses Silero VAD to detect speech activity on the caller's audio:
- **Speech started** during AI response triggers barge-in (clears audio queue, sends clearAudio)
- **Speech ended** resets VAD state for the next turn
- Deepgram handles turn segmentation via its own endpoint detection

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google AI API key | Required |
| `GEMINI_MODEL` | Gemini model name | `gemini-2.0-flash` |
| `DEEPGRAM_API_KEY` | Deepgram API key | Required |
| `DEEPGRAM_MODEL` | Deepgram model | `nova-2-phonecall` |
| `CARTESIA_API_KEY` | Cartesia API key | Required |
| `CARTESIA_VOICE_ID` | Cartesia voice ID | British Lady |
| `CARTESIA_MODEL` | Cartesia model | `sonic-2` |
| `PLIVO_AUTH_ID` | Plivo account auth ID | Required |
| `PLIVO_AUTH_TOKEN` | Plivo account auth token | Required |
| `PLIVO_PHONE_NUMBER` | Your Plivo phone number | Required |
| `PUBLIC_URL` | Public URL for webhooks | Required |
| `SERVER_PORT` | Server port | `8000` |

## Testing

Run unit tests (offline, no API keys needed):
```bash
uv run pytest tests/test_integration.py -v -k "unit"
```

Run local integration tests (requires API keys):
```bash
uv run pytest tests/test_integration.py -v -k "local"
```

Run E2E live tests (requires API keys, starts server):
```bash
uv run pytest tests/test_e2e_live.py -v -s
```

Run live call tests (requires Plivo + API keys + ngrok):
```bash
uv run pytest tests/test_live_call.py -v -s
```

Run outbound call tests:
```bash
uv run pytest tests/test_outbound_call.py -v -s
```

## Customization

### System Prompt
Edit `inbound/system_prompt.md` or `outbound/system_prompt.md`, or set the `SYSTEM_PROMPT` environment variable.

### Voice Selection
Change `CARTESIA_VOICE_ID` in your `.env` file. Browse available voices at [Cartesia](https://play.cartesia.ai/).

### STT Model
Change `DEEPGRAM_MODEL` for different accuracy/speed tradeoffs:
- `nova-2-phonecall` - Optimized for phone audio
- `nova-2` - General purpose
- `nova-2-meeting` - Optimized for meetings
