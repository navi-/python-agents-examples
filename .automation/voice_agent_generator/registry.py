"""Component registry — defines all available LLMs, STTs, TTSs, and orchestrations.

Each component declares:
- name / short_name: for directory naming
- provider: the company behind it
- api_style: websocket | http | sdk
- sample_rate: audio sample rate (for STT/TTS)
- dependencies: Python packages needed
- env_vars: required environment variables
- compatible_with: list of orchestration types this component works with
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
    streaming: bool = True
    env_vars: list[str]  # e.g. ["OPENAI_API_KEY"]
    dependencies: list[str]  # e.g. ["openai>=1.0"]
    model_id: str  # exact model string for API calls
    doc_url: str = ""
    supports_tools: bool = True
    max_tokens_default: int = 300
    notes: str = ""


class STTComponent(BaseModel):
    """Speech-to-Text component."""

    name: str  # e.g. "deepgram"
    short_name: str  # e.g. "deepgram"
    provider: str
    api_style: str  # "websocket" | "http"
    input_sample_rate: int  # what sample rate it accepts
    input_format: str  # "pcm16" | "mulaw" | "raw"
    needs_resample_from_plivo: bool  # True if input_sample_rate != 8000
    env_vars: list[str]
    dependencies: list[str]
    model_id: str = ""
    doc_url: str = ""
    notes: str = ""


class TTSComponent(BaseModel):
    """Text-to-Speech component."""

    name: str  # e.g. "elevenlabs"
    short_name: str  # e.g. "elevenlabs"
    provider: str
    api_style: str  # "websocket" | "http"
    output_sample_rate: int  # what sample rate it produces
    output_format: str  # "pcm16" | "mulaw" | "mp3"
    needs_resample_to_plivo: bool  # True if output_sample_rate != 8000
    env_vars: list[str]
    dependencies: list[str]
    voice_id_default: str = ""
    doc_url: str = ""
    notes: str = ""


class VoiceNativeComponent(BaseModel):
    """S2S / voice-native API (combined STT+LLM+TTS in one API)."""

    name: str  # e.g. "grok-voice"
    short_name: str  # e.g. "grok-voice"
    provider: str
    api_style: str  # "websocket"
    input_sample_rate: int
    output_sample_rate: int
    input_format: str
    output_format: str
    env_vars: list[str]
    dependencies: list[str]
    model_id: str
    doc_url: str = ""
    notes: str = ""


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
        env_vars=["OPENAI_API_KEY"],
        dependencies=["httpx>=0.27.0", "aiohttp>=3.13.3"],
        model_id="gpt-4.1-mini",
        doc_url="https://platform.openai.com/docs/api-reference/chat/create",
    ),
    "gpt4.1": LLMComponent(
        name="GPT-4.1",
        short_name="gpt4.1",
        provider="openai",
        api_style="http",
        env_vars=["OPENAI_API_KEY"],
        dependencies=["httpx>=0.27.0", "aiohttp>=3.13.3"],
        model_id="gpt-4.1",
        doc_url="https://platform.openai.com/docs/api-reference/chat/create",
    ),
    "gpt5.4mini": LLMComponent(
        name="GPT-5.4 Mini",
        short_name="gpt5.4mini",
        provider="openai",
        api_style="http",
        env_vars=["OPENAI_API_KEY"],
        dependencies=["httpx>=0.27.0", "aiohttp>=3.13.3"],
        model_id="gpt-5.4-mini",
        doc_url="https://platform.openai.com/docs/api-reference/chat/create",
    ),
    "claude-sonnet": LLMComponent(
        name="Claude Sonnet 4.6",
        short_name="claude-sonnet",
        provider="anthropic",
        api_style="http",
        env_vars=["ANTHROPIC_API_KEY"],
        dependencies=["anthropic>=0.52.0"],
        model_id="claude-sonnet-4-6",
        doc_url="https://docs.anthropic.com/en/api/messages",
    ),
    "claude-haiku": LLMComponent(
        name="Claude Haiku 4.5",
        short_name="claude-haiku",
        provider="anthropic",
        api_style="http",
        env_vars=["ANTHROPIC_API_KEY"],
        dependencies=["anthropic>=0.52.0"],
        model_id="claude-haiku-4-5-20251001",
        doc_url="https://docs.anthropic.com/en/api/messages",
    ),
    "gemini": LLMComponent(
        name="Gemini 2.0 Flash",
        short_name="gemini",
        provider="google",
        api_style="http",
        env_vars=["GOOGLE_API_KEY"],
        dependencies=["httpx>=0.27.0"],
        model_id="gemini-2.0-flash",
        doc_url="https://ai.google.dev/api/generate-content",
    ),
    "grok": LLMComponent(
        name="Grok",
        short_name="grok",
        provider="xai",
        api_style="http",
        env_vars=["XAI_API_KEY"],
        dependencies=["httpx>=0.27.0"],
        model_id="grok-3-fast",
        doc_url="https://docs.x.ai/api",
    ),
}

STTS: dict[str, STTComponent] = {
    "deepgram": STTComponent(
        name="Deepgram",
        short_name="deepgram",
        provider="deepgram",
        api_style="websocket",
        input_sample_rate=8000,
        input_format="pcm16",
        needs_resample_from_plivo=False,
        env_vars=["DEEPGRAM_API_KEY"],
        dependencies=["websockets>=15.0"],
        model_id="nova-3",
        doc_url="https://developers.deepgram.com/docs/getting-started-with-live-streaming-audio",
    ),
    "sarvam": STTComponent(
        name="Sarvam.ai",
        short_name="sarvam",
        provider="sarvam",
        api_style="websocket",
        input_sample_rate=8000,
        input_format="pcm16",
        needs_resample_from_plivo=False,
        env_vars=["SARVAM_API_KEY"],
        dependencies=["websockets>=15.0"],
        model_id="saaras:v3",
        doc_url="https://docs.sarvam.ai/api-reference-docs/speech-to-text-translate/stt-streaming",
    ),
    "whisper": STTComponent(
        name="OpenAI Whisper",
        short_name="whisper",
        provider="openai",
        api_style="http",
        input_sample_rate=16000,
        input_format="pcm16",
        needs_resample_from_plivo=True,
        env_vars=["OPENAI_API_KEY"],
        dependencies=["httpx>=0.27.0"],
        model_id="whisper-1",
        doc_url="https://platform.openai.com/docs/api-reference/audio/createTranscription",
    ),
    "google-stt": STTComponent(
        name="Google Cloud STT",
        short_name="googlestt",
        provider="google",
        api_style="websocket",
        input_sample_rate=8000,
        input_format="mulaw",
        needs_resample_from_plivo=False,
        env_vars=["GOOGLE_API_KEY"],
        dependencies=["websockets>=15.0"],
        doc_url="https://cloud.google.com/speech-to-text/docs/streaming-recognize",
    ),
}

TTSS: dict[str, TTSComponent] = {
    "elevenlabs": TTSComponent(
        name="ElevenLabs",
        short_name="elevenlabs",
        provider="elevenlabs",
        api_style="websocket",
        output_sample_rate=24000,
        output_format="pcm16",
        needs_resample_to_plivo=True,
        env_vars=["ELEVENLABS_API_KEY"],
        dependencies=["websockets>=15.0"],
        voice_id_default="pNInz6obpgDQGcFmaJgB",
        doc_url="https://elevenlabs.io/docs/api-reference/websockets",
    ),
    "openaitts": TTSComponent(
        name="OpenAI TTS",
        short_name="openaitts",
        provider="openai",
        api_style="websocket",
        output_sample_rate=24000,
        output_format="pcm16",
        needs_resample_to_plivo=True,
        env_vars=["OPENAI_API_KEY"],
        dependencies=["websockets>=15.0"],
        voice_id_default="alloy",
        doc_url="https://platform.openai.com/docs/api-reference/audio/createSpeech",
    ),
    "groktts": TTSComponent(
        name="Grok TTS",
        short_name="groktts",
        provider="xai",
        api_style="websocket",
        output_sample_rate=8000,
        output_format="mulaw",
        needs_resample_to_plivo=False,
        env_vars=["XAI_API_KEY"],
        dependencies=["websockets>=15.0"],
        doc_url="https://docs.x.ai/api",
        notes="Outputs u-law 8kHz directly — no resample needed for Plivo.",
    ),
    "cartesia": TTSComponent(
        name="Cartesia",
        short_name="cartesia",
        provider="cartesia",
        api_style="websocket",
        output_sample_rate=8000,
        output_format="pcm16",
        needs_resample_to_plivo=False,
        env_vars=["CARTESIA_API_KEY"],
        dependencies=["websockets>=15.0", "cartesia>=1.0.0"],
        doc_url="https://docs.cartesia.ai/api-reference/tts/stream-speech-websocket",
    ),
    "google-tts": TTSComponent(
        name="Google Cloud TTS",
        short_name="googletts",
        provider="google",
        api_style="http",
        output_sample_rate=8000,
        output_format="mulaw",
        needs_resample_to_plivo=False,
        env_vars=["GOOGLE_API_KEY"],
        dependencies=["httpx>=0.27.0"],
        doc_url="https://cloud.google.com/text-to-speech/docs/reference/rest",
    ),
}

VOICE_NATIVE: dict[str, VoiceNativeComponent] = {
    "grok-voice": VoiceNativeComponent(
        name="Grok Voice",
        short_name="grok-voice",
        provider="xai",
        api_style="websocket",
        input_sample_rate=24000,
        output_sample_rate=24000,
        input_format="pcm16",
        output_format="pcm16",
        model_id="grok-3-fast-voice",
        env_vars=["XAI_API_KEY"],
        dependencies=["websockets>=15.0"],
        doc_url="https://docs.x.ai/api",
    ),
    "gpt-realtime": VoiceNativeComponent(
        name="GPT Realtime",
        short_name="gpt-realtime",
        provider="openai",
        api_style="websocket",
        input_sample_rate=24000,
        output_sample_rate=24000,
        input_format="pcm16",
        output_format="pcm16",
        model_id="gpt-4o-realtime-preview",
        env_vars=["OPENAI_API_KEY"],
        dependencies=["websockets>=15.0"],
        doc_url="https://platform.openai.com/docs/guides/realtime",
    ),
    "gemini-live": VoiceNativeComponent(
        name="Gemini Live",
        short_name="gemini-live",
        provider="google",
        api_style="websocket",
        input_sample_rate=16000,
        output_sample_rate=24000,
        input_format="pcm16",
        output_format="pcm16",
        model_id="gemini-2.0-flash-live-001",
        env_vars=["GOOGLE_API_KEY"],
        dependencies=["google-genai>=1.0"],
        doc_url="https://ai.google.dev/api/multimodal-live",
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
