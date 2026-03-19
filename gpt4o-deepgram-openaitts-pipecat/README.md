# GPT-4o Deepgram OpenAI-TTS Pipecat Voice Agent

Pipecat-orchestrated voice agent using GPT-4o-mini for LLM, Deepgram for STT, and GPT-4o-mini-TTS for text to speech, with Plivo telephony.

## Features

- **Pipecat Pipeline**: Framework-managed audio pipeline with built-in VAD
- **GPT-4o-mini LLM**: Fast, cost-effective language model for conversation
- **Deepgram STT**: Real-time speech-to-text with direct mu-law support
- **GPT-4o-mini-TTS**: High-quality text to speech with voice affect control
- **Barge-in Support**: Users can interrupt the agent mid-response
- **Multi-turn Conversations**: Maintains context across conversation turns
- **Auto-Configuration**: Automatically configures Plivo webhooks on startup
- **Inbound + Outbound**: Full support for both incoming and outgoing calls

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- OpenAI API key
- Deepgram API key
- Plivo account with a phone number
- ngrok (for local development)

## Quick Start

### 1. Install dependencies

```bash
cd gpt4o-deepgram-openaitts-pipecat
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```bash
OPENAI_API_KEY=your_openai_api_key
DEEPGRAM_API_KEY=your_deepgram_api_key
PLIVO_AUTH_ID=your_plivo_auth_id
PLIVO_AUTH_TOKEN=your_plivo_auth_token
PLIVO_PHONE_NUMBER=+1234567890
PUBLIC_URL=https://your-ngrok-url.ngrok-free.app
```

### 3. Start ngrok

```bash
ngrok http 8000
```

Copy the ngrok URL to `PUBLIC_URL` in your `.env` file.

### 4. Run the server

```bash
# Inbound (receives calls)
uv run python -m inbound.server

# Outbound (places calls)
uv run python -m outbound.server
```

The inbound server will:
1. Start on port 8000
2. Auto-configure Plivo webhooks for your phone number
3. Display "Ready! Call +1234567890 to test"

### 5. Make a test call

Call your Plivo phone number and start talking to the agent.

## Project Structure

```
gpt4o-deepgram-openaitts-pipecat/
├── utils.py                  # Audio conversion + phone utils
├── inbound/
│   ├── agent.py              # Pipecat pipeline for inbound calls
│   ├── server.py             # Standalone inbound FastAPI app
│   ├── system_prompt.md      # Inbound call system prompt
│   └── tts_instructions.md   # TTS voice affect instructions
├── outbound/
│   ├── agent.py              # Pipecat pipeline + CallManager for outbound
│   ├── server.py             # Standalone outbound FastAPI app
│   ├── system_prompt.md      # Outbound call system prompt (with template variables)
│   └── tts_instructions.md   # TTS voice affect instructions
├── tests/                    # Integration and E2E tests
├── pyproject.toml            # Project dependencies
├── .env.example              # Environment variable template
└── README.md                 # This file
```

## How It Works

```
┌─────────┐     ┌─────────────┐     ┌─────────────┐
│  Phone  │────▶│   Plivo     │────▶│   Server    │
│  Call   │◀────│  (PSTN)     │◀────│  (FastAPI)  │
└─────────┘     └─────────────┘     └──────┬──────┘
                                           │
                     WebSocket (μ-law 8kHz)│
                                           ▼
                                    ┌─────────────┐
                                    │  Pipecat    │
                                    │  Pipeline   │
                                    │             │
                                    │ Transport   │
                                    │   ↓         │
                                    │ Deepgram    │
                                    │  (STT)      │
                                    │   ↓         │
                                    │ GPT-4o-mini │
                                    │  (LLM)      │
                                    │   ↓         │
                                    │ GPT-4o-mini │
                                    │  (TTS)      │
                                    │   ↓         │
                                    │ Transport   │
                                    └─────────────┘
```

1. **Incoming Call**: Plivo receives call and hits `/answer` webhook
2. **WebSocket Setup**: Server returns XML to establish bidirectional stream
3. **Audio Streaming**: Plivo streams μ-law 8kHz audio via WebSocket
4. **Pipecat Pipeline**: Audio flows through STT → LLM → TTS automatically
5. **VAD**: Pipecat's built-in VAD handles turn detection and barge-in
6. **Response Streaming**: TTS output is serialized back to μ-law 8kHz for Plivo

## Audio Formats

| Stage | Format | Sample Rate |
|-------|--------|-------------|
| Plivo → Pipecat | μ-law | 8 kHz |
| Pipecat → Deepgram STT | μ-law | 8 kHz |
| Deepgram STT → GPT-4o-mini | Text | N/A |
| GPT-4o-mini → GPT-4o-mini-TTS | Text | N/A |
| GPT-4o-mini-TTS → Pipecat | PCM16 | 24 kHz |
| Pipecat → Plivo | μ-law | 8 kHz |

## Framework Configuration

Pipecat manages the audio pipeline through its `FastAPIWebsocketTransport` with `PlivoFrameSerializer`:

| Setting | Value | Description |
|---------|-------|-------------|
| `vad_enabled` | `True` | Framework-managed voice activity detection |
| `vad_audio_passthrough` | `True` | Pass audio through during VAD silence |
| `allow_interruptions` | `True` | Enable barge-in (user can interrupt agent) |
| `add_wav_header` | `False` | Raw audio, no WAV container |

### TTS Voice Instructions

Voice affect is controlled via `tts_instructions.md` files in each directory. These are passed to GPT-4o-mini-TTS as the `instructions` parameter, controlling tone, pacing, and speaking style.

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `DEEPGRAM_API_KEY` | Deepgram API key | Required |
| `PLIVO_AUTH_ID` | Plivo Auth ID | Required |
| `PLIVO_AUTH_TOKEN` | Plivo Auth Token | Required |
| `PLIVO_PHONE_NUMBER` | Your Plivo phone number | Required |
| `PUBLIC_URL` | Public URL for webhooks (ngrok) | Required |
| `SERVER_PORT` | Server port | `8000` |
| `LLM_MODEL` | OpenAI LLM model | `gpt-4o-mini` |
| `TTS_MODEL` | OpenAI TTS model | `gpt-4o-mini-tts` |
| `TTS_VOICE` | TTS voice name | `alloy` |
| `DEFAULT_COUNTRY_CODE` | Default country for phone parsing | `US` |

### Available Voices

GPT-4o-mini-TTS supports these voices:

| Name | Description |
|------|-------------|
| alloy | Neutral, balanced (default) |
| ash | Warm, conversational |
| ballad | Soft, gentle |
| coral | Clear, friendly |
| echo | Smooth, authoritative |
| fable | Expressive, dynamic |
| nova | Energetic, bright |
| onyx | Deep, resonant |
| sage | Calm, measured |
| shimmer | Light, cheerful |
| verse | Versatile, natural |

## Testing

The test suite includes unit tests, integration tests, and end-to-end live call tests.

```bash
# Run unit tests (offline, no API keys needed)
uv run pytest tests/test_integration.py -v -k "unit"

# Run local integration tests (needs API keys, starts server)
uv run pytest tests/test_integration.py -v -k "local"

# Run E2E tests with real API (no phone call)
uv run pytest tests/test_e2e_live.py -v -s

# Run real phone call tests
uv run pytest tests/test_live_call.py -v -s
uv run pytest tests/test_outbound_call.py -v -s
uv run pytest tests/test_multiturn_voice.py -v -s
```

**Requirements for live call tests:**
- Valid Plivo credentials and phone numbers in `.env`
- Valid OpenAI and Deepgram API keys in `.env`
- `PLIVO_TEST_NUMBER` — a second Plivo number on the same account
- ngrok binary available on PATH
- `faster-whisper` (dev dependency, for transcription verification)

## Deployment

### Docker

```bash
# Build the image
docker build -t gpt4o-pipecat-voice-agent .

# Run inbound server (default)
docker run -p 8000:8000 --env-file .env gpt4o-pipecat-voice-agent

# Run outbound server
docker run -p 8000:8000 --env-file .env gpt4o-pipecat-voice-agent \
  uv run python -m outbound.server
```

## Troubleshooting

### No audio heard on call

- Verify `OPENAI_API_KEY` and `DEEPGRAM_API_KEY` are correct
- Check server logs for pipeline errors
- Ensure the Pipecat serializer is receiving the `streamId` from Plivo

### Agent doesn't respond after speaking

- Pipecat's built-in VAD handles turn detection automatically
- Check that `vad_enabled=True` is set in transport params
- Review server logs for STT transcription output

### WebSocket disconnects immediately

- Ensure ngrok is running and URL in `.env` matches
- Check Plivo credentials are correct
- Verify the server is accessible from the internet

### TTS voice sounds wrong

- Check `TTS_VOICE` in `.env` matches a valid voice name
- Review `tts_instructions.md` for voice affect settings
- GPT-4o-mini-TTS `instructions` parameter controls tone, pacing, and style

### Audio quality issues

- 8kHz telephony is lower quality than the TTS native 24kHz output
- This is expected — audio is downsampled for phone compatibility
- Deepgram receives μ-law directly at 8kHz, avoiding extra conversion
