# Gemini + Deepgram + ElevenLabs Voice Agent (Native)

A voice agent that uses Google Gemini for LLM reasoning, Deepgram for speech-to-text, ElevenLabs for text-to-speech, and Silero VAD for voice activity detection. This implementation uses direct API integration without any orchestration frameworks, connected via Plivo telephony.

## Features

- Google Gemini LLM for conversational reasoning
- Deepgram real-time STT via WebSocket
- ElevenLabs TTS for natural voice synthesis
- Silero VAD for client-side voice activity detection and barge-in
- Plivo telephony with bidirectional WebSocket audio streaming
- Native asyncio orchestration (no frameworks)
- Inbound and outbound call support
- 3 concurrent tasks: plivo_rx, deepgram_rx, plivo_tx

## Architecture

```
Plivo (u-law 8kHz) --> Deepgram STT (PCM 8kHz) --> Gemini LLM
                                                        |
Plivo (u-law 8kHz) <-- ElevenLabs TTS (PCM 24kHz) <----+

Silero VAD runs on Plivo audio for barge-in detection and turn management.
```

**Audio Flow:**
1. Caller speaks --> Plivo captures audio (u-law 8kHz)
2. Audio converted to PCM --> sent to Deepgram for transcription
3. VAD runs in parallel for speech start/end detection
4. Transcript sent to Gemini for response generation
5. Response text sent to ElevenLabs for speech synthesis
6. TTS audio (PCM 24kHz) converted to u-law 8kHz --> sent back to caller
7. Barge-in: if user speaks during response, audio queue is cleared

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- [ngrok](https://ngrok.com/) for local development
- API keys for:
  - [Plivo](https://www.plivo.com/) - Telephony
  - [Google AI Studio](https://aistudio.google.com/) - Gemini API
  - [Deepgram](https://deepgram.com/) - Speech-to-text
  - [ElevenLabs](https://elevenlabs.io/) - Text-to-speech

## Setup

1. **Navigate to the project:**
   ```bash
   cd gemini-deepgram-elevenlabs-native
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

   Or the outbound server:
   ```bash
   uv run python -m outbound.server
   ```

8. **Call your Plivo phone number** to test the voice agent.

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google AI API key | Required |
| `GEMINI_MODEL` | Gemini model name | `gemini-2.0-flash` |
| `DEEPGRAM_API_KEY` | Deepgram API key | Required |
| `DEEPGRAM_MODEL` | Deepgram model | `nova-2-phonecall` |
| `ELEVENLABS_API_KEY` | ElevenLabs API key | Required |
| `ELEVENLABS_VOICE_ID` | ElevenLabs voice ID | `21m00Tcm4TlvDq8ikWAM` |
| `ELEVENLABS_MODEL` | ElevenLabs model | `eleven_flash_v2_5` |
| `PLIVO_AUTH_ID` | Plivo account auth ID | Required |
| `PLIVO_AUTH_TOKEN` | Plivo account auth token | Required |
| `PLIVO_PHONE_NUMBER` | Your Plivo phone number | Required |
| `PUBLIC_URL` | Public URL for webhooks | Required |
| `SERVER_PORT` | Server port | `8000` |

## File Structure

```
gemini-deepgram-elevenlabs-native/
|-- inbound/
|   |-- __init__.py
|   |-- agent.py              # Voice agent for inbound calls
|   |-- server.py             # FastAPI: /answer, /ws, /hangup
|   +-- system_prompt.md      # System prompt for inbound calls
|-- outbound/
|   |-- __init__.py
|   |-- agent.py              # Voice agent + CallManager for outbound
|   |-- server.py             # FastAPI: /outbound/call, /outbound/ws
|   +-- system_prompt.md      # System prompt for outbound calls
|-- utils.py                  # Audio conversion, VAD, phone utils
|-- tests/
|   |-- conftest.py
|   |-- helpers.py
|   |-- test_integration.py   # Unit + local integration tests
|   |-- test_e2e_live.py      # E2E with real APIs
|   |-- test_live_call.py     # Real inbound call test
|   |-- test_multiturn_voice.py
|   +-- test_outbound_call.py # Real outbound call test
|-- pyproject.toml
|-- .env.example
|-- Dockerfile
+-- README.md
```

## How It Works

### Components

- **utils.py**: Audio format conversion (u-law, PCM, resampling), SileroVADProcessor, phone normalization
- **inbound/agent.py**: VoiceAgent class with 3 concurrent tasks (plivo_rx, deepgram_rx, plivo_tx)
- **inbound/server.py**: FastAPI server for inbound call webhooks and WebSocket handling
- **outbound/agent.py**: VoiceAgent + OutboundCallRecord + CallManager for outbound calls
- **outbound/server.py**: FastAPI server for outbound call management

### Audio Format Conversion

- **Plivo**: u-law 8kHz mono (telephony standard)
- **Deepgram**: Linear PCM 16-bit 8kHz
- **ElevenLabs**: Linear PCM 16-bit 24kHz
- **Silero VAD**: Float32 16kHz

### VAD and Barge-in

Client-side Silero VAD runs on every Plivo audio frame:
- **Speech start** during agent response triggers barge-in (clears audio queue, sends clearAudio)
- **Speech end** signals turn completion for natural conversation flow

## Testing

```bash
# Unit tests (offline, no API keys needed)
uv run pytest tests/test_integration.py -v -k "unit"

# Local integration (starts server, needs API keys)
uv run pytest tests/test_integration.py -v -k "local"

# E2E live tests (needs API keys)
uv run pytest tests/test_e2e_live.py -v -s

# Live call tests (needs Plivo + API keys + ngrok)
uv run pytest tests/test_live_call.py -v -s

# Outbound call tests
uv run pytest tests/test_outbound_call.py -v -s
```

## Troubleshooting

### No audio from agent
- Check ElevenLabs API key and voice ID
- Verify audio format conversion is working
- Check server logs for TTS errors

### Poor transcription quality
- Ensure using `nova-2-phonecall` model for phone audio
- Check audio is being received from Plivo (check logs)

### Webhook not receiving calls
- Verify ngrok is running and URL is correct in `.env`
- Check Plivo console for webhook configuration
- Ensure `PUBLIC_URL` uses HTTPS
