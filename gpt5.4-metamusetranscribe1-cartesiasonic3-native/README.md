# Babel: Meta Muse Voice Transcribe + GPT-5.4 mini + Cartesia Sonic-3, Native Speakerphone Concierge

Native orchestration (raw WebSockets + asyncio, no framework). Plivo μ-law 8kHz (160B/20ms chunks) is forwarded untouched as `g711_ulaw` to Meta Muse Voice Transcribe (`muse-voice-transcribe-1.0`) over the Meta Model API realtime transcription WebSocket, which streams partial and final transcripts with per-utterance speaker diarization labels (20+ speakers), server-side endpointing (400ms silence) and keyword/context biasing that the agent updates mid-call with the caller names it learns. Every diarized turn is rendered as `[Speaker N (name)]` lines for OpenAI `gpt-5.4-mini` (streaming chat completions, 5 tools: identify_speaker, add_order_item, get_order_summary, send_sms, end_call); the reply starts with a `<xx>` ISO-639-1 tag that selects the Cartesia `sonic-3` synthesis language, and text is streamed sentence by sentence into one Cartesia WebSocket context (PCM16 24kHz, resampled to 8kHz μ-law). Local Silero VAD (ONNX v5, 512-sample/32ms frames at 16kHz) is the barge-in trigger: start threshold 0.85 rejects Cartesia playback echo (0.5 to 0.75 speech probability) while real speech scores 0.9+, end threshold 0.35, 500ms silence also arms a 1.5s fallback commit if Muse never endpoints. Barge-in cancels the in-flight LLM/TTS turn task, cancels the Cartesia context, drains the send queue and sends `clearAudio` to Plivo.

- Muse endpointing, not VAD, is the primary end-of-turn signal; finals from several speakers within 300ms are merged into one multi-speaker turn
- Mid-sentence code-switching (Hindi/English, Spanish/English) arrives as one transcript; the LLM answers each speaker in their own language with one Cartesia voice
- Turn text comes from Plivo `text` events too, so the whole pipeline is testable without a phone

## The Demo: one phone, three people, three languages

Three friends call Bistro Mundo, put the phone in the middle of the table and order in English, Spanish and Hindi. Babel remembers who said what, answers each person in their language, survives an interruption, reads the order back grouped by person and texts the ticket.

- Script with timestamps and what the pipeline does on every line: [`demo/conversation_script.md`](demo/conversation_script.md)
- Rendered call (88s, distinct voice per speaker): [`demo/demo_conversation.mp3`](demo/demo_conversation.mp3)
- Re-render with different voices or lines: [`demo/generate_demo_audio.py`](demo/generate_demo_audio.py) (Kokoro ONNX, offline)

Try it on a real call: dial your Plivo number, put it on speaker, and have two people order in two languages. Then ask "who ordered what?".

## Pipeline Architecture

```
┌──────────┐       ┌────────────┐       ┌────────────────────────────────────────────────────────────┐
│  Phone   │──────▶│   Plivo    │──────▶│                     Voice Agent (Babel)                    │
│ (table)  │◀──────│  Gateway   │◀──────│                                                            │
└──────────┘       └────────────┘       │  ┌───────────┐   ┌────────────────────┐   ┌─────────────┐  │
                    μ-law 8kHz          │  │  Silero   │   │  Muse Voice        │   │ GPT-5.4     │  │
                    bidirectional       │  │  VAD      │   │  Transcribe (WS)   │   │ mini (SSE)  │  │
                    WebSocket           │  │  (local)  │   │  ASR + diarization │   │ 5 tools     │  │
                                        │  └─────┬─────┘   │  + endpointing     │   └──────▲──────┘  │
                                        │        │         └─────────┬──────────┘          │         │
                                        │  speech_start             │ finals tagged        │ <xx>    │
                                        │  → barge-in               │ [Speaker N]          │ tagged  │
                                        │        │    ┌─────────────▼────────────┐         │ reply   │
                                        │        │    │  Turn assembler          │─────────┘         │
                                        │        │    │  debounce 300ms, merge   │                   │
                                        │        │    │  speakers, VAD fallback  │                   │
                                        │        │    └──────────────────────────┘                   │
                                        │  ┌─────▼──────────────────────────────────────────────┐    │
                                        │  │  Cartesia Sonic-3 (WS): one context per turn,      │    │
                                        │  │  sentence-chunked continuations, language = <xx>   │    │
                                        │  │  PCM16 24kHz → μ-law 8kHz → 20ms playAudio frames  │    │
                                        │  └────────────────────────────────────────────────────┘    │
                                        └────────────────────────────────────────────────────────────┘
```

### Component Summary

| Stage | Service / model | Transport | Audio in → out |
|---|---|---|---|
| Telephony | Plivo Voice AI | bidirectional WS, `audio/x-mulaw;rate=8000` | μ-law 8kHz |
| STT | Meta Muse Voice Transcribe `muse-voice-transcribe-1.0` | WS realtime transcription session | μ-law 8kHz (`g711_ulaw`) → text + speaker label |
| Barge-in | Silero VAD v5 (ONNX, local) | in-process | μ-law 8kHz → float32 16kHz |
| LLM | OpenAI `gpt-5.4-mini` | HTTPS streaming (SSE) | text → text (+ tool calls) |
| TTS | Cartesia `sonic-3` | WS, context continuations | text → PCM16 24kHz |
| Output | resample + G.711 encode | in-process | PCM16 24kHz → μ-law 8kHz |

## Features

- **Diarization-aware conversation**: every user message the LLM sees is `[Speaker 2 (Priya)] ...`; the agent keeps a per-speaker order and can read it back by person
- **Code-switching**: Muse transcribes Hindi/English or Spanish/English within one sentence; the LLM replies in the speaker's language; Cartesia synthesises it with one voice
- **Keyword/context biasing that learns**: caller names discovered via `identify_speaker` are pushed back into the Muse session as keywords mid-call
- **Muse endpointing as the turn signal** with a Silero VAD fallback, plus 300ms debounce so two people finishing together become one turn
- **Streaming end to end**: LLM tokens → sentence chunks → Cartesia continuations → 20ms Plivo frames; first audio typically starts before the LLM finishes
- **Barge-in**: VAD speech start during playback cancels the turn task and Cartesia context, drains the queue, sends `clearAudio`
- **Graceful hang-up**: `end_call` waits for Plivo `playedStream` so the goodbye finishes
- **Inbound + Outbound** servers, Plivo webhook auto-configuration, structured telemetry events (`user_text`, `agent_text`, `turn_complete`, `call_summary`, `speaker_detected`, `speaker_identified`), optional OTel + Redis Streams sinks

## Prerequisites

- Python 3.10+ (3.12 recommended)
- [uv](https://docs.astral.sh/uv/) package manager
- [Meta Model API key](https://developer.meta.com/ai/products/meta-model-api/) (Muse Voice Transcribe)
- [OpenAI API key](https://platform.openai.com/)
- [Cartesia API key](https://cartesia.ai/)
- [Plivo account](https://www.plivo.com/) with a phone number
- [ngrok](https://ngrok.com/) for local development

## Quick Start

### 1. Install dependencies

```bash
cd gpt5.4-metamusetranscribe1-cartesiasonic3-native
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in META_API_KEY, OPENAI_API_KEY, CARTESIA_API_KEY, Plivo credentials
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
# Offline unit tests (no keys needed)
uv run pytest tests/test_integration.py -v -k "unit"

# Then call your Plivo number and put it on speaker
```

## Project Structure

```
gpt5.4-metamusetranscribe1-cartesiasonic3-native/
├── inbound/
│   ├── agent.py              # MuseTranscribeSTT, LanguageTagParser, VoiceAgent (Babel)
│   ├── server.py             # FastAPI: /answer, /ws, /hangup, /hold
│   └── system_prompt.md      # Babel persona, speaker-label + language-tag rules
├── outbound/
│   ├── agent.py              # Same agent + OutboundCallRecord, CallManager
│   ├── server.py             # FastAPI: /outbound/call, /outbound/answer, /outbound/status/{id}
│   └── system_prompt.md      # Outbound template ({{opening_reason}}, {{objective}}, {{context}})
├── utils.py                  # μ-law/PCM, resampling, plivo_to_muse, cartesia_to_plivo, SileroVADProcessor
├── demo/
│   ├── conversation_script.md    # Demo script with timestamps and pipeline notes
│   ├── demo_conversation.mp3     # Rendered demo call
│   └── generate_demo_audio.py    # Kokoro renderer (one voice per speaker)
├── tests/
│   ├── test_integration.py   # Unit (audio, phone, Muse parsing, language tags, turn formatting) + local
│   ├── test_e2e_live.py      # Server + WebSocket + real APIs, Whisper-verified audio
│   ├── test_live_call.py     # Real inbound call via Plivo + ngrok
│   ├── test_multiturn_voice.py  # Multi-turn TTS-driven conversation
│   └── test_outbound_call.py # Real outbound call
├── pyproject.toml
├── .env.example
├── Dockerfile
└── README.md
```

## How It Works

### Turn lifecycle

```
Plivo media ──▶ Muse (g711_ulaw)      Silero VAD (16kHz)
                    │                       │
      partial/final segments          speech_start ──▶ barge-in (cancel turn, cancel Cartesia ctx,
      [speaker label]                                   drain queue, clearAudio)
                    │                 speech_end   ──▶ arm 1.5s fallback (commit Muse partials
      final ──▶ pending_finals                          if no endpoint arrives)
                    │
      300ms debounce (merge speakers) ──▶ "[Speaker 1] ...\n[Speaker 2 (Carlos)] ..."
                    │
      _commit_turn ──▶ user_text event ──▶ GPT-5.4 mini stream
                                              │  <xx> tag → Cartesia language
                                              │  sentence chunk → Cartesia continue=true
                                              │  tool calls → execute → follow-up stream
                                              ▼
                                   continue=false → wait done → checkpoint
                                              │
                          Plivo playedStream ──▶ turn_complete (llm_ms, tts_ttfb_ms, playback_ms)
```

### Muse Voice Transcribe protocol notes

`MuseTranscribeSTT` talks to the Meta Model API realtime transcription session. Meta documents the Model API as a drop-in for OpenAI-SDK-compatible clients, so the client speaks the OpenAI Realtime transcription-session dialect:

| Direction | Message | Purpose |
|---|---|---|
| → | `transcription_session.update` | model, `input_audio_format` (`g711_ulaw` or `pcm16`), language, keyword/context biasing, diarization, server endpointing |
| → | `input_audio_buffer.append` | base64 audio, one Plivo frame at a time |
| → | `input_audio_buffer.commit` | fallback endpoint when VAD hears silence but Muse has not finalised |
| ← | `conversation.item.input_audio_transcription.delta` | partial text (+ `speaker`) |
| ← | `conversation.item.input_audio_transcription.completed` | final text for the utterance (+ `speaker`) |
| ← | `input_audio_buffer.speech_started` / `speech_stopped` | Muse endpointing events |

Diarization labels are read from `speaker`, `speaker_id` or `speaker_label` at the event, item or segment level. Everything protocol-specific lives in `build_session_config()` and `parse_event()` in `inbound/agent.py` (mirrored in `outbound/agent.py`); if Meta's API reference names a field differently, that is the only place to change, and `tests/test_integration.py -k muse` exercises the parser offline. This example was written from Meta's launch material, so verify field names against the Model API reference before production use.

### Reply language tag

The system prompt makes the model begin every reply with `<en>`, `<es>`, `<hi>` and so on. `LanguageTagParser` holds back the first few streamed characters until the tag is resolved, strips it, and the code selects the Cartesia `language`. Unknown codes fall back to `CARTESIA_DEFAULT_LANGUAGE`.

### Audio pipeline

| Stage | Format | Sample rate |
|---|---|---|
| Plivo WebSocket | μ-law (base64) | 8 kHz |
| Muse input (`g711_ulaw`) | μ-law (base64, no conversion) | 8 kHz |
| Muse input (`pcm16`, optional) | PCM16 (resampled) | 24 kHz |
| Silero VAD | float32 | 16 kHz |
| Cartesia output | PCM16 signed LE | 24 kHz |
| Plivo playback | μ-law, 160-byte / 20ms frames | 8 kHz |

## Configuration

| Variable | Description | Default |
|---|---|---|
| `META_API_KEY` | Meta Model API key | — |
| `MUSE_MODEL` | Muse model id | `muse-voice-transcribe-1.0` |
| `MUSE_REALTIME_URL` | Realtime session endpoint | `wss://api.meta.ai/v1/realtime` |
| `MUSE_INPUT_FORMAT` | `g711_ulaw` (passthrough) or `pcm16` (24kHz) | `g711_ulaw` |
| `MUSE_LANGUAGE` | Fixed language, empty = auto + code-switching | `` |
| `MUSE_DIARIZATION` / `MUSE_MAX_SPEAKERS` | Speaker labels | `true` / `20` |
| `MUSE_ENDPOINT_SILENCE_MS` | Server endpointing silence | `400` |
| `MUSE_KEYWORDS` / `MUSE_CONTEXT` | Biasing seed (names are added at runtime) | menu terms |
| `TURN_DEBOUNCE_MS` | Merge window for consecutive finals | `300` |
| `MUSE_ENDPOINT_TIMEOUT_MS` | VAD fallback commit delay | `1500` |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | LLM | — / `gpt-5.4-mini` |
| `CARTESIA_API_KEY` / `CARTESIA_MODEL` / `CARTESIA_VOICE_ID` | TTS | — / `sonic-3` / Tessa |
| `CARTESIA_DEFAULT_LANGUAGE` | Fallback synthesis language | `en` |
| `PLIVO_AUTH_ID` / `PLIVO_AUTH_TOKEN` / `PLIVO_PHONE_NUMBER` | Telephony | — |
| `PUBLIC_URL` / `SERVER_PORT` | Webhook base URL / port | — / `8000` |
| `LOG_LEVEL` | `verbose`, `normal`, `quiet` | `normal` |
| `LOG_FORMAT` / `LOG_FILE` / `REDIS_EVENTS_URL` | Structured sinks | — |

## Testing

```bash
# Unit tests (offline): audio, phone numbers, Muse event parsing, language tags, turn formatting
uv run pytest tests/test_integration.py -v -k "unit"

# Local integration: starts the server, checks health/answer XML/WebSocket audio (needs API keys)
uv run pytest tests/test_integration.py -v -k "local"

# API integration: Muse session, Cartesia synthesis, OpenAI completion
uv run pytest tests/test_integration.py -v -k "muse or cartesia or openai"

# E2E without a phone: WebSocket + text injection, Whisper-verified audio
uv run pytest tests/test_e2e_live.py -v -s

# Real calls (Plivo + ngrok + PLIVO_TEST_NUMBER)
uv run pytest tests/test_live_call.py -v -s
uv run pytest tests/test_outbound_call.py -v -s
```

## Deployment

```bash
docker build -t babel-muse-agent .
docker run -p 8000:8000 --env-file .env babel-muse-agent

# Outbound server
docker run -p 8000:8000 --env-file .env babel-muse-agent uv run python -m outbound.server
```

## Troubleshooting

- **Muse session error on connect**: check `META_API_KEY`, then compare `build_session_config()` with the Meta Model API reference (field names for diarization/biasing)
- **No speaker labels**: set `MUSE_DIARIZATION=true` and check `parse_event` sees a `speaker`-style field; run with `LOG_LEVEL=verbose` to print raw partials
- **Agent answers in the wrong language**: the reply tag is missing; check the system prompt still carries the "Reply Language Tag" section, or set `CARTESIA_DEFAULT_LANGUAGE`
- **Turns commit late**: lower `MUSE_ENDPOINT_SILENCE_MS`; if Muse never endpoints, the VAD fallback fires after `MUSE_ENDPOINT_TIMEOUT_MS`
- **Barge-in triggers on the agent's own voice**: raise `VAD_START_THRESHOLD` in `utils.py` (0.85 default); verbose logs print the probability
- **Two people counted as one**: Muse merges very short overlapping utterances; ask speakers to pause briefly, or lower `TURN_DEBOUNCE_MS`
