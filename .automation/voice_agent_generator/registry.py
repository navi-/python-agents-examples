"""Component registry — defines all available LLMs, STTs, TTSs, and orchestrations.

Each component declares:
- name / short_name: for directory naming
- provider: the company behind it
- api_style: websocket | http | sdk
- api_version: tracks breaking API changes between model generations
- integration_notes: documents API-level differences from other models in same provider
- sample_rate: audio sample rate (for STT/TTS)
- dependencies: Python packages needed
- env_vars: required environment variables
- doc_url: API documentation URL for the generator agent
"""

from __future__ import annotations

from pydantic import BaseModel


class LLMComponent(BaseModel):
    """Large Language Model component."""

    name: str  # e.g. "gpt-4.1-mini"
    short_name: str  # e.g. "gpt4.1mini" — used in directory naming
    provider: str  # e.g. "openai"
    api_style: str  # "http" | "websocket" | "sdk"
    api_version: str = ""  # e.g. "chat-v1", "chat-v2" — tracks breaking API changes
    streaming: bool = True
    env_vars: list[str]  # e.g. ["OPENAI_API_KEY"]
    dependencies: list[str]  # e.g. ["openai>=1.0"]
    model_id: str  # exact model string for API calls
    doc_url: str = ""
    supports_tools: bool = True
    max_tokens_default: int = 300
    max_tokens_param: str = "max_tokens"  # parameter name — "max_tokens" or "max_completion_tokens"
    notes: str = ""
    integration_notes: list[str] = []  # API-specific differences from other versions


class STTComponent(BaseModel):
    """Speech-to-Text component."""

    name: str  # e.g. "deepgram"
    short_name: str  # e.g. "deepgram"
    provider: str
    api_style: str  # "websocket" | "http"
    api_version: str = ""  # tracks breaking changes between model generations
    input_sample_rate: int  # what sample rate it accepts
    input_format: str  # "pcm16" | "mulaw" | "raw"
    needs_resample_from_plivo: bool  # True if input_sample_rate != 8000
    env_vars: list[str]
    dependencies: list[str]
    model_id: str = ""
    doc_url: str = ""
    notes: str = ""
    integration_notes: list[str] = []


class TTSComponent(BaseModel):
    """Text-to-Speech component."""

    name: str  # e.g. "elevenlabs"
    short_name: str  # e.g. "elevenlabs"
    provider: str
    api_style: str  # "websocket" | "http"
    api_version: str = ""
    output_sample_rate: int  # what sample rate it produces
    output_format: str  # "pcm16" | "mulaw" | "mp3"
    needs_resample_to_plivo: bool  # True if output_sample_rate != 8000
    env_vars: list[str]
    dependencies: list[str]
    voice_id_default: str = ""
    doc_url: str = ""
    notes: str = ""
    integration_notes: list[str] = []


class VoiceNativeComponent(BaseModel):
    """S2S / voice-native API (combined STT+LLM+TTS in one API)."""

    name: str  # e.g. "grok-voice"
    short_name: str  # e.g. "grok-voice"
    provider: str
    api_style: str  # "websocket"
    api_version: str = ""
    input_sample_rate: int
    output_sample_rate: int
    input_format: str
    output_format: str
    env_vars: list[str]
    dependencies: list[str]
    model_id: str
    doc_url: str = ""
    notes: str = ""
    integration_notes: list[str] = []


class OrchestrationStyle(BaseModel):
    """How the agent is orchestrated."""

    name: str  # "native" | "pipecat" | "livekit" | "vapi"
    needs_vad_in_utils: bool  # native=True, frameworks=False
    framework_deps: list[str]  # e.g. ["pipecat-ai>=0.1"]
    notes: str = ""


# =============================================================================
# Built-in Registry
# =============================================================================


LLMS: dict[str, LLMComponent] = {
    "gpt4.1mini": LLMComponent(
        name="GPT-4.1 Mini",
        short_name="gpt4.1mini",
        provider="openai",
        api_style="http",
        api_version="chat-v1",
        env_vars=["OPENAI_API_KEY"],
        dependencies=["httpx>=0.27.0", "aiohttp>=3.13.3"],
        model_id="gpt-4.1-mini",
        doc_url="https://platform.openai.com/docs/api-reference/chat/create",
        max_tokens_param="max_tokens",
        integration_notes=[
            "Uses `max_tokens` parameter (deprecated in newer models).",
            "Same /v1/chat/completions endpoint as other OpenAI chat models.",
            "Tool calling uses `tools` array with `type: function` schema.",
        ],
    ),
    "gpt4.1": LLMComponent(
        name="GPT-4.1",
        short_name="gpt4.1",
        provider="openai",
        api_style="http",
        api_version="chat-v1",
        env_vars=["OPENAI_API_KEY"],
        dependencies=["httpx>=0.27.0", "aiohttp>=3.13.3"],
        model_id="gpt-4.1",
        doc_url="https://platform.openai.com/docs/api-reference/chat/create",
        max_tokens_param="max_tokens",
        integration_notes=[
            "Uses `max_tokens` parameter (deprecated in newer models).",
            "Can be used as a reasoning model alongside gpt-4.1-mini for routing.",
            "Same /v1/chat/completions endpoint as gpt-4.1-mini.",
        ],
    ),
    "gpt5.4mini": LLMComponent(
        name="GPT-5.4 Mini",
        short_name="gpt5.4mini",
        provider="openai",
        api_style="http",
        api_version="chat-v2",
        env_vars=["OPENAI_API_KEY"],
        dependencies=["httpx>=0.27.0", "aiohttp>=3.13.3"],
        model_id="gpt-5.4-mini",
        doc_url="https://platform.openai.com/docs/api-reference/chat/create",
        max_tokens_param="max_completion_tokens",
        integration_notes=[
            "BREAKING: Uses `max_completion_tokens` instead of `max_tokens`.",
            "Same /v1/chat/completions endpoint, but `max_tokens` is rejected.",
            "Tool calling format identical to gpt-4.1 series.",
        ],
    ),
    "claude-sonnet": LLMComponent(
        name="Claude Sonnet 4.6",
        short_name="claude-sonnet",
        provider="anthropic",
        api_style="http",
        api_version="messages-v1",
        env_vars=["ANTHROPIC_API_KEY"],
        dependencies=["anthropic>=0.52.0"],
        model_id="claude-sonnet-4-6",
        doc_url="https://docs.anthropic.com/en/api/messages",
        integration_notes=[
            "Uses Anthropic Messages API (not OpenAI-compatible).",
            "Different auth header: `x-api-key` instead of `Authorization: Bearer`.",
            "Tool use via `tools` parameter with `input_schema` (not `parameters`).",
            "Response format: `content` array with `type: text` or `type: tool_use` blocks.",
            "Uses `max_tokens` parameter (required, not optional).",
        ],
    ),
    "claude-haiku": LLMComponent(
        name="Claude Haiku 4.5",
        short_name="claude-haiku",
        provider="anthropic",
        api_style="http",
        api_version="messages-v1",
        env_vars=["ANTHROPIC_API_KEY"],
        dependencies=["anthropic>=0.52.0"],
        model_id="claude-haiku-4-5-20251001",
        doc_url="https://docs.anthropic.com/en/api/messages",
        integration_notes=[
            "Same Messages API as Claude Sonnet — identical integration code.",
            "Lower cost, faster responses, same tool use format.",
        ],
    ),
    "gemini": LLMComponent(
        name="Gemini 2.0 Flash",
        short_name="gemini",
        provider="google",
        api_style="http",
        api_version="genai-v1",
        env_vars=["GOOGLE_API_KEY"],
        dependencies=["httpx>=0.27.0"],
        model_id="gemini-2.0-flash",
        doc_url="https://ai.google.dev/api/generate-content",
        integration_notes=[
            "Uses Google GenAI SDK: `client.aio.models.generate_content()`.",
            "Text-only — NOT the Live API (which is a different integration).",
            "Tool calling via `types.Tool(function_declarations=[...])`.",
            "Different from Gemini Live which uses `client.aio.live.connect()`.",
        ],
    ),
    "grok": LLMComponent(
        name="Grok",
        short_name="grok",
        provider="xai",
        api_style="http",
        api_version="openai-compat-v1",
        env_vars=["XAI_API_KEY"],
        dependencies=["httpx>=0.27.0"],
        model_id="grok-3-fast",
        doc_url="https://docs.x.ai/api",
        integration_notes=[
            "OpenAI-compatible API at `https://api.x.ai/v1/chat/completions`.",
            "Same request/response format as OpenAI chat, different base URL and auth.",
            "Uses `max_tokens` parameter (like OpenAI chat-v1).",
        ],
    ),
}

STTS: dict[str, STTComponent] = {
    "deepgram": STTComponent(
        name="Deepgram",
        short_name="deepgram",
        provider="deepgram",
        api_style="websocket",
        api_version="v1-listen",
        input_sample_rate=8000,
        input_format="pcm16",
        needs_resample_from_plivo=False,
        env_vars=["DEEPGRAM_API_KEY"],
        dependencies=["websockets>=15.0"],
        model_id="nova-3",
        doc_url="https://developers.deepgram.com/docs/getting-started-with-live-streaming-audio",
        integration_notes=[
            "WebSocket to `wss://api.deepgram.com/v1/listen`.",
            "Query params: model, encoding=linear16, sample_rate=8000, channels=1.",
            "nova-2 uses `interim_results=false`; nova-3 uses `interim_results=true`.",
            "Results: `{type: Results, channel: {alternatives: [{transcript}]}}`.",
            "Model bump (nova-2 → nova-3) is config-only, no code changes.",
        ],
    ),
    "sarvam": STTComponent(
        name="Sarvam.ai",
        short_name="sarvam",
        provider="sarvam",
        api_style="websocket",
        api_version="v1-stream",
        input_sample_rate=8000,
        input_format="pcm16",
        needs_resample_from_plivo=False,
        env_vars=["SARVAM_API_KEY"],
        dependencies=["websockets>=15.0"],
        model_id="saaras:v3",
        doc_url="https://docs.sarvam.ai/api-reference-docs/speech-to-text-translate/stt-streaming",
        integration_notes=[
            "WebSocket to `wss://api.sarvam.ai/speech-to-text/stream`.",
            "Query params: language-code, model, sample_rate, input_audio_codec.",
            "Different result format from Deepgram: `{type: data, data: {transcript}}`.",
            "Optimized for Indian English accents, hosted in India.",
        ],
    ),
    "whisper": STTComponent(
        name="OpenAI Whisper",
        short_name="whisper",
        provider="openai",
        api_style="http",
        api_version="v1-audio",
        input_sample_rate=16000,
        input_format="pcm16",
        needs_resample_from_plivo=True,
        env_vars=["OPENAI_API_KEY"],
        dependencies=["httpx>=0.27.0"],
        model_id="whisper-1",
        doc_url="https://platform.openai.com/docs/api-reference/audio/createTranscription",
        integration_notes=[
            "HTTP POST to `/v1/audio/transcriptions` (not streaming).",
            "Requires 16kHz input — must resample from Plivo 8kHz.",
            "Batch-mode: sends accumulated audio, waits for full transcription.",
            "Higher accuracy than streaming STTs but adds latency.",
        ],
    ),
    "google-stt": STTComponent(
        name="Google Cloud STT",
        short_name="googlestt",
        provider="google",
        api_style="websocket",
        api_version="v2-streaming",
        input_sample_rate=8000,
        input_format="mulaw",
        needs_resample_from_plivo=False,
        env_vars=["GOOGLE_API_KEY"],
        dependencies=["websockets>=15.0"],
        doc_url="https://cloud.google.com/speech-to-text/docs/streaming-recognize",
        integration_notes=[
            "Accepts μ-law directly — no decode needed from Plivo.",
            "Different from Google's GenAI STT (which is part of Gemini Live).",
        ],
    ),
}

TTSS: dict[str, TTSComponent] = {
    "elevenlabs": TTSComponent(
        name="ElevenLabs",
        short_name="elevenlabs",
        provider="elevenlabs",
        api_style="websocket",
        api_version="ws-stream-input",
        output_sample_rate=24000,
        output_format="pcm16",
        needs_resample_to_plivo=True,
        env_vars=["ELEVENLABS_API_KEY"],
        dependencies=["websockets>=15.0"],
        voice_id_default="pNInz6obpgDQGcFmaJgB",
        doc_url="https://elevenlabs.io/docs/api-reference/websockets",
        integration_notes=[
            "WebSocket `stream-input` endpoint for low-latency streaming.",
            "Protocol: BOS (config) → text chunks → EOS (flush) → receive audio.",
            "Text split at sentence boundaries for progressive synthesis.",
            "Output PCM16 24kHz — must resample to 8kHz for Plivo.",
            "TTFB 100-200ms (WebSocket) vs 500ms-2s (HTTP variant).",
            "Also has HTTP API (`POST /v1/text-to-speech/{voice_id}`) — different integration, "
            "full text in single request, higher latency but simpler code.",
        ],
    ),
    "openaitts": TTSComponent(
        name="OpenAI TTS",
        short_name="openaitts",
        provider="openai",
        api_style="websocket",
        api_version="v1-audio-speech",
        output_sample_rate=24000,
        output_format="pcm16",
        needs_resample_to_plivo=True,
        env_vars=["OPENAI_API_KEY"],
        dependencies=["websockets>=15.0"],
        voice_id_default="alloy",
        doc_url="https://platform.openai.com/docs/api-reference/audio/createSpeech",
        integration_notes=[
            "WebSocket streaming TTS from OpenAI.",
            "Output PCM16 24kHz — must resample to 8kHz for Plivo.",
        ],
    ),
    "groktts": TTSComponent(
        name="Grok TTS",
        short_name="groktts",
        provider="xai",
        api_style="websocket",
        api_version="v1-realtime-synthesize",
        output_sample_rate=8000,
        output_format="mulaw",
        needs_resample_to_plivo=False,
        env_vars=["XAI_API_KEY"],
        dependencies=["websockets>=15.0"],
        doc_url="https://docs.x.ai/api",
        notes="Outputs u-law 8kHz directly — no resample needed for Plivo.",
        integration_notes=[
            "WebSocket to `wss://api.x.ai/v1/realtime/synthesize`.",
            "Outputs μ-law 8kHz natively — matches Plivo format, NO conversion needed.",
            "Unique among TTS providers: zero audio processing overhead.",
        ],
    ),
    "cartesia": TTSComponent(
        name="Cartesia",
        short_name="cartesia",
        provider="cartesia",
        api_style="websocket",
        api_version="v1-ws",
        output_sample_rate=8000,
        output_format="pcm16",
        needs_resample_to_plivo=False,
        env_vars=["CARTESIA_API_KEY"],
        dependencies=["websockets>=15.0", "cartesia>=1.0.0"],
        doc_url="https://docs.cartesia.ai/api-reference/tts/stream-speech-websocket",
        integration_notes=[
            "Can output at 8kHz directly — no resample needed.",
            "WebSocket with context-based streaming (reuse voice context across turns).",
        ],
    ),
    "google-tts": TTSComponent(
        name="Google Cloud TTS",
        short_name="googletts",
        provider="google",
        api_style="http",
        api_version="v1-synthesize",
        output_sample_rate=8000,
        output_format="mulaw",
        needs_resample_to_plivo=False,
        env_vars=["GOOGLE_API_KEY"],
        dependencies=["httpx>=0.27.0"],
        doc_url="https://cloud.google.com/text-to-speech/docs/reference/rest",
        integration_notes=[
            "HTTP POST — batch synthesis, not streaming.",
            "Can output μ-law directly — matches Plivo, no conversion.",
            "Different from Gemini Live audio output (which is S2S, not standalone TTS).",
        ],
    ),
}

VOICE_NATIVE: dict[str, VoiceNativeComponent] = {
    "grok-voice": VoiceNativeComponent(
        name="Grok Voice",
        short_name="grok-voice",
        provider="xai",
        api_style="websocket",
        api_version="v1-realtime",
        input_sample_rate=24000,
        output_sample_rate=24000,
        input_format="pcm16",
        output_format="pcm16",
        model_id="grok-3-fast-voice",
        env_vars=["XAI_API_KEY"],
        dependencies=["websockets>=15.0"],
        doc_url="https://docs.x.ai/api",
        integration_notes=[
            "S2S over WebSocket — single connection handles STT+LLM+TTS.",
            "Session config via `session.update` message with tools and system prompt.",
            "Audio: bidirectional PCM16 24kHz binary frames.",
            "Function calls arrive as JSON events within the audio stream.",
        ],
    ),
    "gpt-realtime": VoiceNativeComponent(
        name="GPT Realtime",
        short_name="gpt-realtime",
        provider="openai",
        api_style="websocket",
        api_version="v1-realtime",
        input_sample_rate=24000,
        output_sample_rate=24000,
        input_format="pcm16",
        output_format="pcm16",
        model_id="gpt-4o-realtime-preview",
        env_vars=["OPENAI_API_KEY"],
        dependencies=["websockets>=15.0"],
        doc_url="https://platform.openai.com/docs/guides/realtime",
        integration_notes=[
            "COMPLETELY different API from OpenAI Chat Completions.",
            "WebSocket to `wss://api.openai.com/v1/realtime?model=...`.",
            "Session config via `session.update` with audio format, voice, tools.",
            "Audio: bidirectional PCM16 24kHz binary frames.",
            "Tool calls via `conversation.item.create` + `response.create` messages.",
            "NOT compatible with Chat Completions code — full rewrite required.",
        ],
    ),
    "gemini-live": VoiceNativeComponent(
        name="Gemini Live",
        short_name="gemini-live",
        provider="google",
        api_style="websocket",
        api_version="live-v1",
        input_sample_rate=16000,
        output_sample_rate=24000,
        input_format="pcm16",
        output_format="pcm16",
        model_id="gemini-2.0-flash-live-001",
        env_vars=["GOOGLE_API_KEY"],
        dependencies=["google-genai>=1.0"],
        doc_url="https://ai.google.dev/api/multimodal-live",
        integration_notes=[
            "COMPLETELY different API from Gemini HTTP (generate_content).",
            "Uses `client.aio.live.connect()` context manager — NOT `generate_content()`.",
            "Bidirectional audio+text streaming with `send_client_content()`.",
            "Turn management via `turn_complete=True` flag.",
            "NOT compatible with Gemini HTTP LLM code — full rewrite required.",
        ],
    ),
}

ORCHESTRATIONS: dict[str, OrchestrationStyle] = {
    "native": OrchestrationStyle(
        name="native",
        needs_vad_in_utils=True,
        framework_deps=[],
        notes="Raw websockets + asyncio, client-side Silero VAD.",
    ),
    "pipecat": OrchestrationStyle(
        name="pipecat",
        needs_vad_in_utils=False,
        framework_deps=["pipecat-ai[silero]>=0.1"],
        notes="Pipecat pipeline framework. VAD handled by framework.",
    ),
    "livekit": OrchestrationStyle(
        name="livekit",
        needs_vad_in_utils=False,
        framework_deps=["livekit-agents>=0.8"],
        notes="LiveKit agents framework.",
    ),
    "vapi": OrchestrationStyle(
        name="vapi",
        needs_vad_in_utils=False,
        framework_deps=[],
        notes="Vapi hosted platform (HTTP webhooks, no local audio processing).",
    ),
}


# =============================================================================
# Provider catalog — groups components by provider for per-provider docs
# =============================================================================


def get_provider_catalog() -> dict[str, dict]:
    """Build a catalog of all components grouped by provider.

    Returns a dict like:
    {
        "openai": {
            "display_name": "OpenAI",
            "llms": [gpt4.1mini, gpt4.1, gpt5.4mini],
            "stts": [whisper],
            "ttss": [openaitts],
            "voice_native": [gpt-realtime],
            "api_surfaces": ["chat-v1", "chat-v2", "v1-realtime", ...],
        },
        ...
    }
    """
    providers: dict[str, dict] = {}

    def _ensure(provider: str) -> dict:
        if provider not in providers:
            providers[provider] = {
                "display_name": provider.title(),
                "llms": [],
                "stts": [],
                "ttss": [],
                "voice_native": [],
            }
        return providers[provider]

    # Display name overrides
    display_names = {
        "openai": "OpenAI",
        "anthropic": "Anthropic",
        "google": "Google",
        "xai": "xAI",
        "deepgram": "Deepgram",
        "sarvam": "Sarvam.ai",
        "elevenlabs": "ElevenLabs",
        "cartesia": "Cartesia",
    }

    for key, comp in LLMS.items():
        p = _ensure(comp.provider)
        p["display_name"] = display_names.get(comp.provider, comp.provider.title())
        p["llms"].append((key, comp))

    for key, comp in STTS.items():
        p = _ensure(comp.provider)
        p["display_name"] = display_names.get(comp.provider, comp.provider.title())
        p["stts"].append((key, comp))

    for key, comp in TTSS.items():
        p = _ensure(comp.provider)
        p["display_name"] = display_names.get(comp.provider, comp.provider.title())
        p["ttss"].append((key, comp))

    for key, comp in VOICE_NATIVE.items():
        p = _ensure(comp.provider)
        p["display_name"] = display_names.get(comp.provider, comp.provider.title())
        p["voice_native"].append((key, comp))

    return providers
