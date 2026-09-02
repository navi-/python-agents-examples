"""Outbound voice agent: Meta Muse Voice Transcribe STT + GPT-5.4 mini + Cartesia Sonic-3 TTS.

Loads the outbound system prompt template and provides run_agent() for outbound
call WebSocket sessions, plus CallManager for tracking call lifecycle.

Status state machine:
    initiating -> ringing -> connected -> completed
                         |-> no_answer
                |-> failed

Pipeline architecture:
  Plivo audio -> Muse Voice Transcribe (streaming ASR + diarization + endpointing)
              -> GPT-5.4 mini (streaming, function calling)
              -> Cartesia Sonic-3 (WebSocket, sentence-chunked continuations)
              -> Plivo audio

"Babel" is a speakerphone concierge: several people talk to one phone in
whatever language they like. Muse tags every transcript with a stable speaker
label, so the LLM can answer each person in their own language and keep a
per-person order. Muse's built-in endpointing is the primary end-of-turn
signal; Silero VAD provides barge-in (speech start during playback) and a
fallback commit if an endpoint event never arrives.

Pipeline logging is controlled by the LOG_LEVEL env var:
  verbose : every pipeline event (partials, VAD frames, queue sizes, TTFB)
  normal  : key events (turn lifecycle, STT finals, LLM responses, TTS timing)
  quiet   : errors and session start/end only
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import functools
import json
import os
import random
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from loguru import logger
from openai import AsyncOpenAI

from utils import (
    SileroVADProcessor,
    cartesia_to_plivo,
    plivo_to_muse,
    plivo_to_vad,
)

load_dotenv()

# ---------------------------------------------------------------------------
# OTel tracing (optional: no-op when opentelemetry is not installed)
# ---------------------------------------------------------------------------
try:
    from opentelemetry import trace as _otel_trace

    _tracer = _otel_trace.get_tracer("voice-agent")
except ImportError:
    _otel_trace = None  # type: ignore[assignment]
    _tracer = None  # type: ignore[assignment]


def _traced(span_name: str):
    """Wrap an async method in an OTel span carrying the full call_id."""

    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(self, *args, **kwargs):
            if not _tracer:
                return await fn(self, *args, **kwargs)
            with _tracer.start_as_current_span(span_name, attributes={"call_id": self.call_id}):
                return await fn(self, *args, **kwargs)

        return wrapper

    return decorator


# =============================================================================
# Agent configuration
# =============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")

# Meta Model API: Muse Voice Transcribe over the OpenAI-compatible realtime
# transcription session. Meta documents the Model API as a drop-in for
# OpenAI-SDK-compatible clients, so the wire format below follows the
# OpenAI Realtime "transcription" session contract with Muse-specific
# session fields (diarization, keyword/context biasing). All names are
# centralised in MuseTranscribeSTT.build_session_config() and
# MuseTranscribeSTT.parse_event() so they can be adjusted in one place.
META_API_KEY = os.getenv("META_API_KEY", "")
MUSE_MODEL = os.getenv("MUSE_MODEL", "muse-voice-transcribe-1.0")
MUSE_REALTIME_URL = os.getenv("MUSE_REALTIME_URL", "wss://api.meta.ai/v1/realtime")
MUSE_INPUT_FORMAT = os.getenv("MUSE_INPUT_FORMAT", "g711_ulaw")  # g711_ulaw | pcm16
MUSE_LANGUAGE = os.getenv("MUSE_LANGUAGE", "")  # "" = auto-detect + code-switching
MUSE_DIARIZATION = os.getenv("MUSE_DIARIZATION", "true").lower() in ("1", "true", "yes")
MUSE_MAX_SPEAKERS = int(os.getenv("MUSE_MAX_SPEAKERS", "20"))
MUSE_ENDPOINT_SILENCE_MS = int(os.getenv("MUSE_ENDPOINT_SILENCE_MS", "400"))
MUSE_KEYWORDS = [
    k.strip()
    for k in os.getenv(
        "MUSE_KEYWORDS",
        "Babel,Bistro Mundo,paneer tikka,paella,tacos al pastor,margherita,mango lassi",
    ).split(",")
    if k.strip()
]
MUSE_CONTEXT = os.getenv(
    "MUSE_CONTEXT",
    "Restaurant phone line. Multiple people order food on speakerphone, often "
    "switching between English, Spanish and Hindi mid-sentence.",
)

# Turn assembly: consecutive final segments (possibly from different speakers)
# arriving within this window are merged into one LLM turn.
TURN_DEBOUNCE_MS = int(os.getenv("TURN_DEBOUNCE_MS", "300"))
# If Silero VAD saw speech end but Muse has not emitted a final segment within
# this window, force a commit of whatever partial text Muse has produced.
MUSE_ENDPOINT_TIMEOUT_MS = int(os.getenv("MUSE_ENDPOINT_TIMEOUT_MS", "1500"))

CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", "")
CARTESIA_MODEL = os.getenv("CARTESIA_MODEL", "sonic-3")
CARTESIA_VOICE_ID = os.getenv("CARTESIA_VOICE_ID", "6ccbfb76-1fc6-48f7-b71d-91ac6298247b")
CARTESIA_API_VERSION = os.getenv("CARTESIA_API_VERSION", "2025-04-16")
CARTESIA_WS_URL = os.getenv("CARTESIA_WS_URL", "wss://api.cartesia.ai/tts/websocket")
CARTESIA_DEFAULT_LANGUAGE = os.getenv("CARTESIA_DEFAULT_LANGUAGE", "en")
# Languages Sonic-3 can synthesise; anything else falls back to the default.
CARTESIA_LANGUAGES = {
    "en",
    "es",
    "fr",
    "de",
    "hi",
    "it",
    "ja",
    "ko",
    "nl",
    "pl",
    "pt",
    "ru",
    "sv",
    "tr",
    "zh",
    "ar",
    "bn",
    "cs",
    "da",
    "el",
    "fi",
    "he",
    "hu",
    "id",
    "ms",
    "no",
    "ro",
    "ta",
    "th",
    "uk",
    "vi",
    "bg",
    "hr",
    "sk",
    "tl",
}

# Logging verbosity: "verbose", "normal" (default), "quiet"
LOG_LEVEL = os.getenv("LOG_LEVEL", "normal").lower()

if TYPE_CHECKING:
    from fastapi import WebSocket

# =============================================================================
# System Prompt
# =============================================================================

_OUTBOUND_PROMPT_TEMPLATE = (Path(__file__).parent / "system_prompt.md").read_text().strip()


def build_outbound_prompt(
    opening_reason: str = "",
    objective: str = "",
    context: str = "",
) -> str:
    """Build a concrete outbound system prompt by substituting template variables."""
    prompt = _OUTBOUND_PROMPT_TEMPLATE
    prompt = prompt.replace("{{opening_reason}}", opening_reason)
    prompt = prompt.replace("{{objective}}", objective)
    prompt = prompt.replace("{{context}}", context)
    return prompt


# Default system prompt (no template substitution)
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", _OUTBOUND_PROMPT_TEMPLATE)

# =============================================================================
# Outbound Call Records
# =============================================================================


@dataclass
class OutboundCallRecord:
    """Tracks the state of a single outbound call."""

    call_id: str
    phone_number: str
    status: str = "initiating"  # initiating|ringing|connected|completed|failed|no_answer
    campaign_id: str = ""
    context: str = ""
    system_prompt: str = ""
    initial_message: str = ""
    opening_reason: str = ""
    objective: str = ""
    plivo_request_uuid: str = ""
    plivo_call_uuid: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    connected_at: datetime | None = None
    ended_at: datetime | None = None
    duration: int = 0
    hangup_cause: str = ""
    outcome: str = ""  # success|no_answer|busy|failed


def determine_outcome(hangup_cause: str, duration: int) -> str:
    """Map Plivo hangup cause and duration to a high-level outcome.

    See https://www.plivo.com/docs/voice/troubleshooting/hangup-causes/
    """
    cause = hangup_cause.upper() if hangup_cause else ""

    if cause in ("NO_ANSWER", "ORIGINATOR_CANCEL"):
        return "no_answer"
    if cause in ("USER_BUSY", "CALL_REJECTED"):
        return "busy"
    if cause in (
        "UNALLOCATED_NUMBER",
        "INVALID_NUMBER_FORMAT",
        "NO_ROUTE_DESTINATION",
        "NETWORK_OUT_OF_ORDER",
        "SERVICE_UNAVAILABLE",
        "RECOVERY_ON_TIMER_EXPIRE",
        "BEARERCAPABILITY_NOTAVAIL",
    ):
        return "failed"

    # If the call was answered and had meaningful duration, consider it success
    if duration > 0 or cause in ("NORMAL_CLEARING", ""):
        return "success"

    return "failed"


class CallManager:
    """Thread-safe manager for outbound call records."""

    def __init__(self) -> None:
        self._calls: dict[str, OutboundCallRecord] = {}
        self._lock = threading.Lock()

    def create_call(
        self,
        phone_number: str,
        campaign_id: str = "",
        opening_reason: str = "",
        objective: str = "",
        context: str = "",
    ) -> OutboundCallRecord:
        """Create and register a new outbound call record."""
        call_id = str(uuid.uuid4())
        system_prompt = build_outbound_prompt(opening_reason, objective, context)

        if opening_reason:
            initial_message = (
                "The call has been answered. Begin with your outbound greeting now. "
                "State your name, company, and that you are reaching out regarding: "
                f"{opening_reason}. Then ask if now is a good time."
            )
        else:
            initial_message = (
                "The call has been answered. Begin with your outbound greeting now. "
                "State your name, company, and why you are calling. Then ask if now is a good time."
            )

        record = OutboundCallRecord(
            call_id=call_id,
            phone_number=phone_number,
            campaign_id=campaign_id,
            opening_reason=opening_reason,
            objective=objective,
            context=context,
            system_prompt=system_prompt,
            initial_message=initial_message,
        )

        with self._lock:
            self._calls[call_id] = record

        return record

    def get_call(self, call_id: str) -> OutboundCallRecord | None:
        """Look up a call by its ID."""
        with self._lock:
            return self._calls.get(call_id)

    def update_status(self, call_id: str, status: str, **kwargs: Any) -> OutboundCallRecord | None:
        """Thread-safe status update with optional extra fields."""
        with self._lock:
            record = self._calls.get(call_id)
            if record is None:
                return None
            record.status = status
            for key, value in kwargs.items():
                if hasattr(record, key):
                    setattr(record, key, value)
            return record

    def get_active_calls(self) -> list[OutboundCallRecord]:
        """Return calls with status in (initiating, ringing, connected)."""
        with self._lock:
            return [
                r
                for r in self._calls.values()
                if r.status in ("initiating", "ringing", "connected")
            ]

    def get_calls_by_campaign(self, campaign_id: str) -> list[OutboundCallRecord]:
        """Return all calls for a given campaign."""
        with self._lock:
            return [r for r in self._calls.values() if r.campaign_id == campaign_id]

    def reset(self) -> None:
        """Clear all records (useful for testing)."""
        with self._lock:
            self._calls.clear()


# =============================================================================
# Muse Voice Transcribe STT client
# =============================================================================


@dataclass
class TranscriptSegment:
    """One transcript update from Muse."""

    text: str
    speaker: str = ""  # stable diarization label ("1", "2", ...) or "" if unknown
    is_final: bool = False
    item_id: str = ""


class MuseTranscribeSTT:
    """Streaming client for Meta Muse Voice Transcribe.

    Audio is streamed continuously; Muse returns partial (delta) and final
    (completed) transcript segments, each tagged with a diarization speaker
    label, and emits its own endpointing events. Segments are pushed to
    ``on_segment`` (an asyncio.Queue) for the agent's turn state machine.
    """

    SPEAKER_KEYS = ("speaker", "speaker_id", "speaker_label")

    def __init__(
        self,
        on_segment: asyncio.Queue[TranscriptSegment] | None = None,
        keywords: list[str] | None = None,
        context: str | None = None,
        language: str | None = None,
    ):
        self._ws = None
        self._running = False
        self._receive_task: asyncio.Task | None = None
        self._partials: dict[str, str] = {}
        self._partial_speakers: dict[str, str] = {}
        self.on_segment = on_segment
        self.on_speech_started: asyncio.Event = asyncio.Event()
        self.keywords: list[str] = list(keywords if keywords is not None else MUSE_KEYWORDS)
        self.context: str = context if context is not None else MUSE_CONTEXT
        self.language: str = language if language is not None else MUSE_LANGUAGE
        self.session_id: str = ""
        self.errors: int = 0

    # -- Protocol ---------------------------------------------------------

    def build_session_config(self) -> dict[str, Any]:
        """Session config sent as ``transcription_session.update``.

        Core fields follow the OpenAI Realtime transcription session
        (``input_audio_format``, ``input_audio_transcription``,
        ``turn_detection``). Diarization and keyword/context biasing are the
        Muse-specific extensions; they live under ``input_audio_transcription``
        next to the model they configure.
        """
        transcription: dict[str, Any] = {
            "model": MUSE_MODEL,
            "prompt": self._biasing_prompt(),
            "keywords": list(self.keywords),
            "context": self.context,
            "diarization": {"enabled": MUSE_DIARIZATION, "max_speakers": MUSE_MAX_SPEAKERS},
        }
        if self.language:
            transcription["language"] = self.language
        return {
            "type": "transcription_session.update",
            "session": {
                "input_audio_format": MUSE_INPUT_FORMAT,
                "input_audio_transcription": transcription,
                "turn_detection": {
                    "type": "server_vad",
                    "silence_duration_ms": MUSE_ENDPOINT_SILENCE_MS,
                    "create_response": False,
                },
            },
        }

    def _biasing_prompt(self) -> str:
        parts = []
        if self.context:
            parts.append(self.context)
        if self.keywords:
            parts.append("Vocabulary: " + ", ".join(self.keywords))
        return " ".join(parts)

    @classmethod
    def extract_speaker(cls, event: dict[str, Any]) -> str:
        """Pull a diarization speaker label out of an event, tolerating shapes."""
        for container in (event, event.get("item") or {}, event.get("diarization") or {}):
            for key in cls.SPEAKER_KEYS:
                value = container.get(key) if isinstance(container, dict) else None
                if value not in (None, ""):
                    return str(value)
        segments = event.get("segments")
        if isinstance(segments, list) and segments:
            first = segments[0]
            if isinstance(first, dict):
                for key in cls.SPEAKER_KEYS:
                    if first.get(key) not in (None, ""):
                        return str(first[key])
        return ""

    def parse_event(self, event: dict[str, Any]) -> TranscriptSegment | None:
        """Turn a raw Muse event into a TranscriptSegment (or None).

        Handles ``conversation.item.input_audio_transcription.delta`` /
        ``.completed`` plus any ``*.delta`` / ``*.completed`` transcript event
        carrying ``delta`` or ``transcript`` text.
        """
        etype = str(event.get("type", ""))
        item_id = str(event.get("item_id") or event.get("id") or "current")
        speaker = self.extract_speaker(event)

        if etype.endswith(".delta") and "transcription" in etype:
            delta = str(event.get("delta") or "")
            if not delta:
                return None
            self._partials[item_id] = self._partials.get(item_id, "") + delta
            if speaker:
                self._partial_speakers[item_id] = speaker
            return TranscriptSegment(
                text=self._partials[item_id],
                speaker=speaker or self._partial_speakers.get(item_id, ""),
                is_final=False,
                item_id=item_id,
            )

        if (etype.endswith(".completed") or etype.endswith(".done")) and "transcription" in etype:
            text = str(event.get("transcript") or event.get("text") or "").strip()
            if not text:
                text = self._partials.get(item_id, "").strip()
            self._partials.pop(item_id, None)
            partial_speaker = self._partial_speakers.pop(item_id, "")
            if not text:
                return None
            return TranscriptSegment(
                text=text, speaker=speaker or partial_speaker, is_final=True, item_id=item_id
            )

        return None

    # -- Lifecycle --------------------------------------------------------

    async def connect(self) -> None:
        """Open the realtime session and start the receive loop."""
        import websockets

        self._running = True
        url = f"{MUSE_REALTIME_URL}?intent=transcription"
        headers = {
            "Authorization": f"Bearer {META_API_KEY}",
            "OpenAI-Beta": "realtime=v1",
        }
        self._ws = await websockets.connect(url, additional_headers=headers, max_size=None)
        await self._ws.send(json.dumps(self.build_session_config()))
        self._receive_task = asyncio.create_task(self._receive_loop(), name="muse_rx")

    async def _receive_loop(self) -> None:
        try:
            async for raw in self._ws:
                event = json.loads(raw)
                etype = str(event.get("type", ""))

                if etype in ("transcription_session.created", "transcription_session.updated"):
                    session = event.get("session") or {}
                    self.session_id = str(session.get("id") or self.session_id)
                    logger.debug(f"Muse session {etype}: id={self.session_id}")
                    continue
                if etype == "input_audio_buffer.speech_started":
                    self.on_speech_started.set()
                    continue
                if etype == "error":
                    self.errors += 1
                    logger.error(f"Muse error: {event.get('error') or event}")
                    continue

                segment = self.parse_event(event)
                if segment is not None and self.on_segment is not None:
                    self.on_segment.put_nowait(segment)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self._running:
                self.errors += 1
                logger.error(f"Muse receive error: {e}")

    async def send_audio(self, audio: bytes) -> None:
        """Append raw audio (bytes in the configured input format)."""
        if self._ws is None or not self._running:
            return
        payload = base64.b64encode(audio).decode("ascii")
        await self._ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": payload}))

    async def commit(self) -> None:
        """Ask Muse to finalise the buffered audio (fallback endpoint)."""
        if self._ws is None or not self._running:
            return
        await self._ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

    async def update_biasing(
        self, keywords: list[str] | None = None, context: str | None = None
    ) -> None:
        """Push new keyword/context biasing mid-session (e.g. learned names)."""
        if keywords is not None:
            merged = list(self.keywords)
            for k in keywords:
                if k and k not in merged:
                    merged.append(k)
            self.keywords = merged
        if context is not None:
            self.context = context
        if self._ws is None or not self._running:
            return
        await self._ws.send(json.dumps(self.build_session_config()))

    def pending_text(self) -> str:
        """Partial text Muse has produced but not yet finalised."""
        return " ".join(t.strip() for t in self._partials.values() if t.strip()).strip()

    def pending_speaker(self) -> str:
        for item_id in self._partials:
            if self._partial_speakers.get(item_id):
                return self._partial_speakers[item_id]
        return ""

    def clear_pending(self) -> None:
        self._partials.clear()
        self._partial_speakers.clear()

    async def close(self) -> None:
        self._running = False
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._receive_task
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
        self._ws = None


# =============================================================================
# Reply language tag parser
# =============================================================================

_LANG_TAG_RE = re.compile(r"^\s*<([a-zA-Z]{2,3})(?:-[a-zA-Z]{2,4})?>\s*")


class LanguageTagParser:
    """Strips the leading ``<xx>`` language tag from a streamed LLM reply.

    Feed deltas in order; ``feed`` returns the text safe to forward to TTS.
    The first few characters are held back until the tag (or its absence) is
    known so the tag never leaks into speech.
    """

    HOLD_CHARS = 8

    def __init__(self, default: str = CARTESIA_DEFAULT_LANGUAGE):
        self.language = default
        self.resolved = False
        self._buffer = ""

    def feed(self, delta: str) -> str:
        if self.resolved:
            return delta
        self._buffer += delta
        match = _LANG_TAG_RE.match(self._buffer)
        if match:
            self.resolved = True
            code = match.group(1).lower()
            if code in CARTESIA_LANGUAGES:
                self.language = code
            out = self._buffer[match.end() :]
            self._buffer = ""
            return out
        stripped = self._buffer.lstrip()
        if stripped and not stripped.startswith("<"):
            # No tag coming
            self.resolved = True
            out = self._buffer
            self._buffer = ""
            return out
        if len(self._buffer) > self.HOLD_CHARS and ">" not in self._buffer:
            self.resolved = True
            out = self._buffer
            self._buffer = ""
            return out
        return ""

    def flush(self) -> str:
        self.resolved = True
        out = self._buffer
        self._buffer = ""
        return out


# =============================================================================
# Tool functions (in-memory demo implementations)
# =============================================================================


@dataclass
class TableState:
    """Per-call memory of who is who and who ordered what."""

    speakers: dict[str, dict[str, str]] = field(default_factory=dict)
    order: list[dict[str, Any]] = field(default_factory=list)

    def label(self, speaker: str) -> str:
        name = self.speakers.get(speaker, {}).get("name", "")
        return f"Speaker {speaker} ({name})" if name else f"Speaker {speaker}"


async def send_sms(phone_number: str, message: str) -> dict[str, Any]:
    """Send SMS to the caller. Replace with your actual implementation."""
    if not phone_number:
        return {"status": "error", "message": "Phone number required"}
    return {
        "status": "sent",
        "phone_number": phone_number,
        "message_preview": message[:50] + "..." if len(message) > 50 else message,
        "confirmation_id": f"SMS{random.randint(100000, 999999)}",
    }


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "identify_speaker",
            "description": (
                "Record the name and/or preferred language of a diarized speaker label "
                "so later replies can address them by name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "speaker_label": {
                        "type": "string",
                        "description": "The Muse speaker label, e.g. '1' or '2'",
                    },
                    "name": {"type": "string", "description": "The person's name"},
                    "language": {
                        "type": "string",
                        "description": "ISO 639-1 code of the language they prefer",
                    },
                },
                "required": ["speaker_label"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_order_item",
            "description": "Add one menu item to the table order for a specific speaker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "speaker_label": {"type": "string", "description": "Who ordered it"},
                    "item": {"type": "string", "description": "Menu item name in English"},
                    "quantity": {"type": "integer", "description": "How many", "default": 1},
                    "notes": {"type": "string", "description": "Modifiers, e.g. 'extra spicy'"},
                },
                "required": ["speaker_label", "item"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_summary",
            "description": "Return the current table order grouped by speaker.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_sms",
            "description": "Text the order ticket or a confirmation to a phone number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "phone_number": {"type": "string", "description": "Phone number"},
                    "message": {"type": "string", "description": "Message content"},
                },
                "required": ["phone_number", "message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "end_call",
            "description": "End the call gracefully after saying goodbye.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Reason for ending"},
                },
            },
        },
    },
]

_SPEAKER_PREFIX_RE = re.compile(r"^\s*speaker[\s_:-]*", re.IGNORECASE)


def normalize_speaker_label(value: object) -> str:
    """'Speaker 2' / 'speaker_2' / 2 -> '2' (the raw Muse diarization label)."""
    return _SPEAKER_PREFIX_RE.sub("", str(value or "")).strip()


# Sentence boundaries used to chunk streamed LLM text into Cartesia continuations.
_SENTENCE_END = (".", "!", "?", "।", "。", "！", "？", ",", ";", ":", "，")  # noqa: RUF001


# =============================================================================
# Voice Agent
# =============================================================================


class VoiceAgent:
    """Voice session: Plivo + Muse Voice Transcribe + GPT-5.4 mini + Cartesia Sonic-3.

    Concurrent tasks:
        - plivo_rx    : receive Plivo audio, forward to Muse, run Silero VAD (barge-in)
        - muse_watch  : consume Muse segments, assemble diarized turns, commit to LLM
        - cartesia_rx : receive Cartesia audio, route active context to the send queue
        - plivo_tx    : send queued u-law audio to Plivo in 20ms chunks
    """

    def __init__(
        self,
        websocket: WebSocket,
        call_id: str,
        from_number: str = "",
        to_number: str = "",
        system_prompt: str | None = None,
        initial_message: str = "Hello, I'm calling for help.",
        stream_id: str = "",
        parent_call_id: str = "",
        sip_headers: dict[str, str] | None = None,
    ):
        self.websocket = websocket
        self.call_id = call_id
        self.parent_call_id = parent_call_id or call_id
        self.from_number = from_number
        self.to_number = to_number
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.initial_message = initial_message
        self.sip_headers = sip_headers or {}

        self._running = False
        self._send_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._vad = SileroVADProcessor()
        self._is_playing = False  # True while Plivo is playing audio to caller
        self._barged_in = False  # True after speech_started until speech_ended
        self._segment_queue: asyncio.Queue[TranscriptSegment] = asyncio.Queue()
        self._stt = MuseTranscribeSTT(on_segment=self._segment_queue)
        self._openai = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self._turn_lock = asyncio.Lock()
        self._conversation_history: list[dict[str, Any]] = []
        self._current_turn_task: asyncio.Task | None = None
        self._stream_id = stream_id
        self._checkpoint_counter = 0
        self._checkpoint_sent_time: float | None = None
        self._playback_done = asyncio.Event()
        self._end_call_requested = False

        # Turn assembly (Muse endpointing + debounce, VAD fallback)
        self._pending_finals: list[TranscriptSegment] = []
        self._debounce_task: asyncio.Task | None = None
        self._fallback_task: asyncio.Task | None = None

        # Table memory (speaker names + order) exposed to tools
        self.table = TableState()

        # Cartesia state
        self._cartesia_ws = None
        self._active_context_id: str | None = None
        self._tts_done: asyncio.Event = asyncio.Event()
        self._tts_t0: float | None = None
        self._tts_first_chunk: float | None = None
        self._tts_chunks = 0
        self._tts_bytes = 0

        # Session counters
        self._turn_count = 0
        self._barge_in_count = 0
        self._error_count = 0
        self._session_start = time.monotonic()
        self._plivo_rx_bytes = 0
        self._plivo_tx_chunks = 0
        self._speech_end_time: float | None = None
        self._ttfs_samples: list[float] = []
        self._speakers_seen: set[str] = set()

        # Per-turn metrics (reset at start of each _process_text_turn)
        self._turn_llm_ms: float | None = None
        self._turn_tts_total_ms: float | None = None
        self._turn_tts_ttfb_ms: float | None = None
        self._turn_tts_chunks: int = 0
        self._turn_tts_audio_s: float = 0.0
        self._turn_text: str = ""
        self._turn_agent_text: str = ""
        self._turn_language: str = CARTESIA_DEFAULT_LANGUAGE
        self._turn_start_time: float | None = None

    # -- Structured logging with call ID, elapsed time, and pipeline stage --

    def _log(self, stage: str, msg: str, **extra: object) -> None:
        """Log at 'normal' level: key pipeline events.

        ``extra`` fields are bound to the record for structured sinks
        (Redis/SSE, OTel) without cluttering the console line.
        """
        if LOG_LEVEL == "quiet":
            return
        elapsed = round(time.monotonic() - self._session_start, 2)
        logger.bind(call_id=self.call_id, elapsed_s=elapsed, stage=stage, **extra).info(
            f"[{self.call_id}] [{elapsed:7.2f}s] [{stage}] {msg}"
        )

    def _logv(self, stage: str, msg: str, **extra: object) -> None:
        """Log at 'verbose' level: detailed debugging info."""
        if LOG_LEVEL != "verbose":
            return
        elapsed = round(time.monotonic() - self._session_start, 2)
        logger.bind(call_id=self.call_id, elapsed_s=elapsed, stage=stage, **extra).debug(
            f"[{self.call_id}] [{elapsed:7.2f}s] [{stage}] {msg}"
        )

    def _loge(self, stage: str, msg: str, **extra: object) -> None:
        """Log errors: always visible regardless of LOG_LEVEL."""
        self._error_count += 1
        elapsed = round(time.monotonic() - self._session_start, 2)
        logger.bind(call_id=self.call_id, elapsed_s=elapsed, stage=stage, **extra).error(
            f"[{self.call_id}] [{elapsed:7.2f}s] [{stage}] {msg}"
        )

    def _emit_turn_complete(self, barge_in: bool = False) -> None:
        """Emit a structured turn_complete event with per-turn metrics."""
        playback_ms = None
        if self._checkpoint_sent_time is not None:
            playback_ms = round((time.monotonic() - self._checkpoint_sent_time) * 1000)
            self._checkpoint_sent_time = None
        logger.bind(
            event="turn_complete",
            call_id=self.parent_call_id,
            turn=self._turn_count,
            user_text=self._turn_text or "",
            agent_text=self._turn_agent_text or "",
            language=self._turn_language,
            llm_ms=self._turn_llm_ms,
            tts_total_ms=self._turn_tts_total_ms,
            tts_ttfb_ms=self._turn_tts_ttfb_ms,
            tts_chunks=self._turn_tts_chunks,
            tts_audio_duration_s=self._turn_tts_audio_s,
            playback_ms=playback_ms,
            barge_in=barge_in,
        ).info(
            f"[{self.call_id}] turn {self._turn_count} complete{' (barge-in)' if barge_in else ''}"
        )

    # -- Tools -----------------------------------------------------------------

    async def _handle_function_call(self, name: str, arguments: str) -> dict[str, Any]:
        """Execute a function call and return the result."""
        try:
            args = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            args = {}

        self._log("tool", f"calling {name}({args})")

        try:
            if name == "identify_speaker":
                label = normalize_speaker_label(args.get("speaker_label"))
                if not label:
                    return {"error": "speaker_label required"}
                entry = self.table.speakers.setdefault(label, {})
                if args.get("name"):
                    entry["name"] = str(args["name"]).strip()
                if args.get("language"):
                    entry["language"] = str(args["language"]).strip().lower()
                names = [s["name"] for s in self.table.speakers.values() if s.get("name")]
                # Feed learned names back into Muse's keyword biasing.
                await self._stt.update_biasing(keywords=names)
                self._log(
                    "stt",
                    f"biasing updated with names: {names}",
                    event="speaker_identified",
                    speaker=label,
                    name=entry.get("name", ""),
                    language=entry.get("language", ""),
                )
                result: dict[str, Any] = {"status": "ok", "speaker": self.table.label(label)}
            elif name == "add_order_item":
                label = normalize_speaker_label(args.get("speaker_label"))
                item = {
                    "speaker_label": label,
                    "speaker": self.table.label(label),
                    "item": str(args.get("item", "")).strip(),
                    "quantity": int(args.get("quantity") or 1),
                    "notes": str(args.get("notes") or "").strip(),
                }
                self.table.order.append(item)
                result = {
                    "status": "added",
                    "item": item,
                    "items_on_ticket": len(self.table.order),
                }
            elif name == "get_order_summary":
                grouped: dict[str, list[str]] = {}
                for entry in self.table.order:
                    line = f"{entry['quantity']} x {entry['item']}"
                    if entry["notes"]:
                        line += f" ({entry['notes']})"
                    grouped.setdefault(self.table.label(entry["speaker_label"]), []).append(line)
                result = {"status": "ok", "order_by_speaker": grouped}
            elif name == "send_sms":
                result = await send_sms(
                    phone_number=args.get("phone_number", "") or self.from_number,
                    message=args.get("message", ""),
                )
            elif name == "end_call":
                self._log("tool", f"end_call: {args.get('reason')}")
                self._end_call_requested = True
                result = {"status": "call_ending", "reason": args.get("reason", "")}
            else:
                result = {"error": f"Unknown function: {name}"}

            self._log("tool", f"{name} -> {result.get('status', 'done')}")
            return result

        except Exception as e:
            self._loge("tool", f"{name} ERROR: {e}")
            return {"error": str(e)}

    # -- LLM -------------------------------------------------------------------

    @_traced("llm")
    async def _generate_llm_response(self, user_text: str, context_id: str) -> str:
        """Stream GPT-5.4 mini; forward sentence chunks to Cartesia as they arrive.

        Returns the full assistant text (language tag stripped). Tool calls are
        executed and followed by a second streamed completion.
        """
        self._conversation_history.append({"role": "user", "content": user_text})
        t0 = time.monotonic()
        parser = LanguageTagParser()
        full_text = ""
        text_buffer = ""
        usage: dict[str, Any] = {}

        async def stream_once(with_tools: bool) -> dict[int, dict[str, str]]:
            nonlocal full_text, text_buffer, usage, parser
            if not full_text:
                # Fresh reply (or tool-call round produced no text): expect a new tag.
                parser = LanguageTagParser()
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(self._conversation_history)
            kwargs: dict[str, Any] = {
                "model": OPENAI_MODEL,
                "messages": messages,
                "stream": True,
                "max_completion_tokens": 300,
                "stream_options": {"include_usage": True},
            }
            if with_tools:
                kwargs["tools"] = TOOL_DEFINITIONS
            stream = await self._openai.chat.completions.create(**kwargs)
            tool_calls: dict[int, dict[str, str]] = {}
            async for chunk in stream:
                if chunk.usage:
                    usage = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                    }
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        slot = tool_calls.setdefault(
                            tc.index, {"id": "", "name": "", "arguments": ""}
                        )
                        if tc.id:
                            slot["id"] = tc.id
                        if tc.function and tc.function.name:
                            slot["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            slot["arguments"] += tc.function.arguments
                    continue
                if delta.content:
                    out = parser.feed(delta.content)
                    if not out:
                        continue
                    if not full_text:
                        self._turn_language = parser.language
                        self._logv(
                            "llm",
                            f"first token ({(time.monotonic() - t0) * 1000:.0f}ms, "
                            f"lang={parser.language})",
                        )
                    full_text += out
                    text_buffer += out
                    if text_buffer.rstrip().endswith(_SENTENCE_END) and len(text_buffer) > 12:
                        await self._send_to_cartesia(text_buffer, context_id, parser.language)
                        text_buffer = ""
            tail = parser.flush()
            if tail:
                full_text += tail
                text_buffer += tail
            return tool_calls

        try:
            tool_calls = await stream_once(with_tools=True)
            llm_ms = (time.monotonic() - t0) * 1000

            if tool_calls:
                ordered = [tool_calls[i] for i in sorted(tool_calls)]
                self._log("llm", f"tool calls ({llm_ms:.0f}ms): {[tc['name'] for tc in ordered]}")
                self._conversation_history.append(
                    {
                        "role": "assistant",
                        "content": full_text or None,
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {"name": tc["name"], "arguments": tc["arguments"]},
                            }
                            for tc in ordered
                        ],
                    }
                )
                for tc in ordered:
                    fn_result = await self._handle_function_call(tc["name"], tc["arguments"])
                    self._conversation_history.append(
                        {"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(fn_result)}
                    )
                self._logv("llm", "follow-up request after tool calls")
                t1 = time.monotonic()
                await stream_once(with_tools=False)
                self._logv("llm", f"follow-up response ({(time.monotonic() - t1) * 1000:.0f}ms)")
                llm_ms = (time.monotonic() - t0) * 1000

            self._turn_llm_ms = round(llm_ms)
            if text_buffer.strip():
                await self._send_to_cartesia(text_buffer, context_id, parser.language)
                text_buffer = ""

            assistant_text = full_text.strip()
            self._conversation_history.append({"role": "assistant", "content": assistant_text})

            if _otel_trace:
                span = _otel_trace.get_current_span()
                span.set_attribute("llm.latency_ms", llm_ms)
                span.set_attribute("gen_ai.request.model", OPENAI_MODEL)
                span.set_attribute("gen_ai.usage.prompt_tokens", usage.get("prompt_tokens", 0))
                span.set_attribute(
                    "gen_ai.usage.completion_tokens", usage.get("completion_tokens", 0)
                )

            # This single line doubles as the agent_text event for structured sinks.
            self._log(
                "llm",
                f"response ({llm_ms:.0f}ms, lang={parser.language}, "
                f"{usage.get('prompt_tokens', '?')}->{usage.get('completion_tokens', '?')} tok): "
                f"'{assistant_text[:80]}'",
                event="agent_text",
                turn=self._turn_count,
                text=assistant_text,
                language=parser.language,
            )
            return assistant_text

        except asyncio.CancelledError:
            raise
        except Exception as e:
            llm_ms = (time.monotonic() - t0) * 1000
            self._turn_llm_ms = round(llm_ms)
            self._loge("llm", f"ERROR ({llm_ms:.0f}ms): {e}")
            fallback = "Sorry, I lost that for a second. Could you say it again?"
            await self._send_to_cartesia(fallback, context_id, CARTESIA_DEFAULT_LANGUAGE)
            return fallback

    # -- Cartesia TTS ----------------------------------------------------------

    async def _connect_cartesia(self) -> None:
        import websockets

        url = (
            f"{CARTESIA_WS_URL}?api_key={CARTESIA_API_KEY}&cartesia_version={CARTESIA_API_VERSION}"
        )
        self._cartesia_ws = await websockets.connect(url, max_size=None)
        self._log("tts", f"connected to Cartesia ({CARTESIA_MODEL}, voice={CARTESIA_VOICE_ID[:8]})")

    async def _send_to_cartesia(
        self, text: str, context_id: str, language: str, is_last: bool = False
    ) -> None:
        """Send a text chunk into a Cartesia context (continuation stream)."""
        if self._cartesia_ws is None:
            return
        if not text.strip() and not is_last:
            return
        if self._tts_t0 is None:
            self._tts_t0 = time.monotonic()
            self._is_playing = True
        message = {
            "context_id": context_id,
            "model_id": CARTESIA_MODEL,
            "transcript": text,
            "voice": {"mode": "id", "id": CARTESIA_VOICE_ID},
            "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 24000},
            "language": language if language in CARTESIA_LANGUAGES else CARTESIA_DEFAULT_LANGUAGE,
            "continue": not is_last,
        }
        self._logv("tts", f"send ({len(text)} chars, lang={message['language']}, last={is_last})")
        try:
            await self._cartesia_ws.send(json.dumps(message))
        except Exception as e:
            self._loge("tts", f"send ERROR: {e}")

    async def _cancel_cartesia_context(self) -> None:
        if self._active_context_id and self._cartesia_ws is not None:
            with contextlib.suppress(Exception):
                await self._cartesia_ws.send(
                    json.dumps({"context_id": self._active_context_id, "cancel": True})
                )
        self._active_context_id = None

    @_traced("tts")
    async def _receive_from_cartesia(self) -> None:
        """Route Cartesia audio for the active context to the Plivo send queue."""
        try:
            async for raw in self._cartesia_ws:
                message = json.loads(raw)
                ctx = message.get("context_id")
                if ctx != self._active_context_id:
                    continue  # stale context (barge-in) or unknown
                mtype = message.get("type")
                if mtype == "chunk" and message.get("data"):
                    if not self._is_playing:
                        continue
                    pcm_24k = base64.b64decode(message["data"])
                    if self._tts_first_chunk is None and self._tts_t0 is not None:
                        self._tts_first_chunk = time.monotonic()
                        ttfb_ms = (self._tts_first_chunk - self._tts_t0) * 1000
                        self._logv("tts", f"first chunk (TTFB {ttfb_ms:.0f}ms)")
                    await self._send_queue.put(cartesia_to_plivo(pcm_24k))
                    self._tts_chunks += 1
                    self._tts_bytes += len(pcm_24k)
                elif mtype == "error":
                    self._loge("tts", f"Cartesia error: {message.get('error')}")
                    self._tts_done.set()
                if message.get("done"):
                    self._tts_done.set()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self._running:
                self._loge("tts", f"receive ERROR: {e}")

    def _reset_tts_metrics(self, context_id: str) -> None:
        self._active_context_id = context_id
        self._tts_done.clear()
        self._tts_t0 = None
        self._tts_first_chunk = None
        self._tts_chunks = 0
        self._tts_bytes = 0

    def _record_tts_metrics(self) -> None:
        if self._tts_t0 is None:
            return
        total_ms = (time.monotonic() - self._tts_t0) * 1000
        audio_s = self._tts_bytes / (24000 * 2)
        self._turn_tts_total_ms = round(total_ms)
        self._turn_tts_chunks = self._tts_chunks
        self._turn_tts_audio_s = round(audio_s, 2)
        ttfb_str = ""
        if self._tts_first_chunk is not None:
            ttfb = (self._tts_first_chunk - self._tts_t0) * 1000
            self._turn_tts_ttfb_ms = round(ttfb)
            ttfb_str = f"TTFB={ttfb:.0f}ms, "
        self._log(
            "tts",
            f"done: {self._tts_chunks} chunks, {ttfb_str}{audio_s:.1f}s audio in {total_ms:.0f}ms",
        )

    # -- Session ----------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """Build system prompt with call context."""
        system_prompt = self.system_prompt
        if self.from_number:
            call_time = datetime.now().strftime("%I:%M %p on %A, %B %d")
            system_prompt += f"""

## Current Call Context
- Caller's phone number: {self.from_number}
- Call ID: {self.call_id}
- Time: {call_time}

You can text the caller's phone number without asking for it."""
        return system_prompt

    async def run(self) -> None:
        """Run the voice agent session."""
        self._session_start = time.monotonic()
        self._running = True
        self.system_prompt = self._build_system_prompt()
        logger.info(
            f"[{self.call_id}] [  0.00s] [session] "
            f"started (from={self.from_number}, to={self.to_number}, log={LOG_LEVEL})"
        )
        logger.bind(
            event="call_answered",
            call_id=self.parent_call_id,
            leg_call_id=self.call_id,
            from_number=self.from_number,
            to_number=self.to_number,
            sip_headers=self.sip_headers,
        ).info(
            f"[{self.call_id}] [  0.00s] [session] call answered (sip_headers={self.sip_headers})"
        )

        try:
            await self._stt.connect()
            self._log(
                "stt",
                f"connected to Muse ({MUSE_MODEL}, format={MUSE_INPUT_FORMAT}, "
                f"diarization={MUSE_DIARIZATION}, keywords={len(self._stt.keywords)})",
            )
            await self._connect_cartesia()
        except Exception as e:
            self._loge("session", f"connect ERROR: {e}")
            self._running = False

        if self._running:
            self._log(
                "session", "starting streaming tasks (plivo_rx, muse_watch, cartesia_rx, plivo_tx)"
            )
            try:
                await self._run_streaming_tasks()
            except Exception as e:
                self._loge("session", f"streaming ERROR: {e}")

        self._running = False
        await self._stt.close()
        if self._cartesia_ws is not None:
            with contextlib.suppress(Exception):
                await self._cartesia_ws.close()
        duration = round(time.monotonic() - self._session_start, 1)
        avg_ttfs = (
            round(sum(self._ttfs_samples) / len(self._ttfs_samples)) if self._ttfs_samples else None
        )
        logger.bind(
            event="call_summary",
            call_id=self.parent_call_id,
            duration_s=duration,
            turns=self._turn_count,
            barge_ins=self._barge_in_count,
            speakers=len(self._speakers_seen),
            ttfs_avg_ms=avg_ttfs,
            ttfs_samples=len(self._ttfs_samples),
            errors=self._error_count + self._stt.errors,
            rx_bytes=self._plivo_rx_bytes,
            tx_chunks=self._plivo_tx_chunks,
        ).info(
            f"[{self.call_id}] [{duration:7.1f}s] [session] ended -- {self._turn_count} turns, "
            f"{len(self._speakers_seen)} speakers, {self._barge_in_count} barge-ins, "
            f"TTFS avg={avg_ttfs}ms, rx={self._plivo_rx_bytes}B, tx={self._plivo_tx_chunks} chunks"
        )

    async def _run_streaming_tasks(self) -> None:
        """Run the concurrent streaming tasks."""
        tasks = [
            asyncio.create_task(self._receive_from_plivo(), name="plivo_rx"),
            asyncio.create_task(self._watch_segments(), name="muse_watch"),
            asyncio.create_task(self._receive_from_cartesia(), name="cartesia_rx"),
            asyncio.create_task(self._send_to_plivo(), name="plivo_tx"),
        ]

        # Greeting runs as the first turn once the tasks are live.
        self._turn_count += 1
        self._log("turn", f"turn {self._turn_count}: generating greeting")
        self._current_turn_task = asyncio.create_task(
            self._process_text_turn(self.initial_message), name="turn_greeting"
        )
        self._current_turn_task.add_done_callback(
            lambda t: t.exception() if not t.cancelled() else None
        )

        try:
            done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task.exception():
                    self._loge("session", f"task {task.get_name()} failed: {task.exception()}")
        finally:
            self._running = False
            for task in [*tasks, self._current_turn_task, self._debounce_task, self._fallback_task]:
                if task and not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

    # -- Plivo RX + VAD -------------------------------------------------------

    async def _receive_from_plivo(self) -> None:
        """Receive audio from Plivo, stream to Muse, run VAD for barge-in."""
        media_count = 0
        try:
            while self._running:
                data = await self.websocket.receive_text()
                message = json.loads(data)
                event = message.get("event")

                if event == "media":
                    payload = message.get("media", {}).get("payload", "")
                    if not payload:
                        continue
                    mulaw_audio = base64.b64decode(payload)
                    self._plivo_rx_bytes += len(mulaw_audio)
                    media_count += 1
                    if media_count == 1:
                        self._log("plivo_rx", "first audio packet received")
                    if media_count % 500 == 0:
                        self._logv("plivo_rx", f"{media_count} packets")

                    # Always forward to Muse, even during playback: Muse's own
                    # endpointing needs continuous audio and echo is dropped
                    # on barge-in by clearing pending partials.
                    if MUSE_INPUT_FORMAT == "pcm16":
                        await self._stt.send_audio(plivo_to_muse(mulaw_audio))
                    else:
                        await self._stt.send_audio(mulaw_audio)

                    speech_started, speech_ended = self._vad.process(plivo_to_vad(mulaw_audio))

                    if speech_started and not self._barged_in:
                        self._barged_in = True
                        self._log("vad", "speech START detected")
                        await self._handle_barge_in()

                    if speech_ended:
                        self._speech_end_time = time.monotonic()
                        self._log("vad", "speech END -- waiting for Muse endpoint")
                        self._arm_endpoint_fallback()
                        self._vad.reset()
                        self._barged_in = False

                elif event == "playedStream":
                    name = message.get("name", "")
                    self._is_playing = False
                    self._playback_done.set()
                    playback_ms = None
                    if self._checkpoint_sent_time is not None:
                        playback_ms = round((time.monotonic() - self._checkpoint_sent_time) * 1000)
                    self._log(
                        "plivo_rx",
                        f"playedStream: '{name}' -- playback complete ({playback_ms}ms)",
                    )
                    self._emit_turn_complete(barge_in=False)
                    if self._end_call_requested:
                        self._log("session", "end_call requested -- goodbye played, hanging up")
                        break

                elif event == "clearedAudio":
                    self._is_playing = False
                    self._logv("plivo_rx", "clearedAudio confirmed by Plivo")

                elif event == "text":
                    text = message.get("text", "")
                    if text:
                        await self._commit_turn(text)

                elif event == "stop":
                    self._log("plivo_rx", "received stop event -- call ended")
                    break

        except Exception as e:
            if "1000" not in str(e):
                self._loge("plivo_rx", f"ERROR: {e}")
        finally:
            self._logv("plivo_rx", f"exiting -- received {media_count} media packets")

    async def _handle_barge_in(self) -> None:
        """Caller started talking: cancel LLM/TTS, flush audio, clear Plivo playback."""
        task_cancelled = False
        if self._current_turn_task and not self._current_turn_task.done():
            self._current_turn_task.cancel()
            task_cancelled = True
        await self._cancel_cartesia_context()
        cleared = 0
        while not self._send_queue.empty():
            try:
                self._send_queue.get_nowait()
                cleared += 1
            except asyncio.QueueEmpty:
                break
        clear_event: dict[str, Any] = {"event": "clearAudio"}
        if self._stream_id:
            clear_event["streamId"] = self._stream_id
        with contextlib.suppress(Exception):
            await self.websocket.send_text(json.dumps(clear_event))
        # Drop any echo Muse may have transcribed from our own playback.
        self._stt.clear_pending()
        if self._is_playing:
            self._barge_in_count += 1
            self._log(
                "vad",
                f"barge-in: clearAudio sent, cancelled={task_cancelled}, cleared={cleared} chunks",
            )
            self._emit_turn_complete(barge_in=True)
        self._is_playing = False

    # -- Turn assembly (Muse endpointing + diarization) --------------------------

    async def _watch_segments(self) -> None:
        """Consume Muse segments; finals are debounced into diarized turns."""
        try:
            while self._running:
                try:
                    segment = await asyncio.wait_for(self._segment_queue.get(), timeout=0.2)
                except TimeoutError:
                    continue

                if segment.speaker and segment.speaker not in self._speakers_seen:
                    self._speakers_seen.add(segment.speaker)
                    self._log(
                        "stt",
                        f"new speaker detected: {self.table.label(segment.speaker)}",
                        event="speaker_detected",
                        speaker=segment.speaker,
                    )

                if not segment.is_final:
                    self._logv("stt", f"partial [{segment.speaker or '?'}]: '{segment.text[-60:]}'")
                    continue

                self._logv(
                    "stt",
                    f"final [{segment.speaker or '?'}]: '{segment.text[:80]}'",
                )
                self._pending_finals.append(segment)
                if self._fallback_task and not self._fallback_task.done():
                    self._fallback_task.cancel()
                if self._debounce_task and not self._debounce_task.done():
                    self._debounce_task.cancel()
                self._debounce_task = asyncio.create_task(
                    self._debounce_commit(), name="turn_debounce"
                )
        except asyncio.CancelledError:
            pass

    async def _debounce_commit(self) -> None:
        try:
            await asyncio.sleep(TURN_DEBOUNCE_MS / 1000)
        except asyncio.CancelledError:
            return
        segments, self._pending_finals = self._pending_finals, []
        if not segments:
            return
        await self._commit_turn(self._format_turn(segments))

    def _format_turn(self, segments: list[TranscriptSegment]) -> str:
        """Render diarized segments as speaker-tagged lines for the LLM."""
        lines: list[str] = []
        for seg in segments:
            if seg.speaker:
                lines.append(f"[{self.table.label(seg.speaker)}] {seg.text}")
            else:
                lines.append(seg.text)
        return "\n".join(lines)

    def _arm_endpoint_fallback(self) -> None:
        """VAD saw silence: if Muse never endpoints, commit its partial text."""
        if self._fallback_task and not self._fallback_task.done():
            self._fallback_task.cancel()
        self._fallback_task = asyncio.create_task(
            self._endpoint_fallback(), name="endpoint_fallback"
        )

    async def _endpoint_fallback(self) -> None:
        try:
            await asyncio.sleep(MUSE_ENDPOINT_TIMEOUT_MS / 1000)
            if self._pending_finals or not self._stt.pending_text():
                return
            self._log("stt", "no Muse endpoint after VAD silence -- requesting commit")
            await self._stt.commit()
            await asyncio.sleep(0.5)
            if self._pending_finals:
                return
            text = self._stt.pending_text()
            if not text:
                return
            speaker = self._stt.pending_speaker()
            self._stt.clear_pending()
            self._log("stt", "committing partial transcript (fallback)")
            await self._commit_turn(
                self._format_turn([TranscriptSegment(text=text, speaker=speaker, is_final=True)])
            )
        except asyncio.CancelledError:
            pass

    async def _commit_turn(self, transcript: str) -> None:
        """Commit a user turn for processing."""
        if self._current_turn_task and not self._current_turn_task.done():
            self._current_turn_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._current_turn_task
        self._turn_count += 1
        # The single _log line below doubles as the user_text event.
        self._log(
            "turn",
            f"turn {self._turn_count}: '{transcript[:80]}'",
            event="user_text",
            turn=self._turn_count,
            text=transcript,
        )
        self._current_turn_task = asyncio.create_task(
            self._process_text_turn(transcript), name=f"turn_{self._turn_count}"
        )
        self._current_turn_task.add_done_callback(
            lambda t: t.exception() if not t.cancelled() else None
        )

    # -- Turn processing -------------------------------------------------------

    async def _process_text_turn(self, text: str) -> None:
        """Process one turn: streamed LLM -> Cartesia continuations -> Plivo."""
        async with self._turn_lock:
            self._turn_llm_ms = None
            self._turn_tts_total_ms = None
            self._turn_tts_ttfb_ms = None
            self._turn_tts_chunks = 0
            self._turn_tts_audio_s = 0.0
            self._turn_text = text
            self._turn_agent_text = ""
            self._turn_language = CARTESIA_DEFAULT_LANGUAGE
            self._turn_start_time = time.monotonic()
            context_id = str(uuid.uuid4())
            self._reset_tts_metrics(context_id)
            self._playback_done.clear()
            try:
                response_text = await self._generate_llm_response(text, context_id)
                self._turn_agent_text = response_text
                if not response_text.strip() and self._tts_t0 is None:
                    self._logv("turn", "empty LLM response, skipping TTS")
                    self._active_context_id = None
                    return
                # Close the Cartesia context and wait for the last audio chunk.
                await self._send_to_cartesia("", context_id, self._turn_language, is_last=True)
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._tts_done.wait(), timeout=15.0)
                self._record_tts_metrics()
                await self._send_checkpoint()
                self._log("turn", "TTS done, audio queued for playback")
                if self._end_call_requested:
                    # Give Plivo time to play the goodbye before the session ends.
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(
                            self._playback_done.wait(), timeout=self._turn_tts_audio_s + 3.0
                        )
                    self._running = False
            except asyncio.CancelledError:
                self._is_playing = False
                self._log("turn", "turn cancelled (barge-in)")
            except Exception as e:
                self._is_playing = False
                self._loge("turn", f"text turn ERROR: {e}")
            finally:
                if self._active_context_id == context_id:
                    self._active_context_id = None

    async def _send_checkpoint(self) -> None:
        """Send a Plivo checkpoint so playedStream reports end of playback."""
        if not self._stream_id:
            return
        self._checkpoint_counter += 1
        name = f"turn_{self._turn_count}_{self._checkpoint_counter}"
        checkpoint = {"event": "checkpoint", "streamId": self._stream_id, "name": name}
        await self.websocket.send_text(json.dumps(checkpoint))
        self._checkpoint_sent_time = time.monotonic()
        self._logv("plivo_tx", f"checkpoint sent: {name}")

    # -- Plivo TX ----------------------------------------------------------------

    async def _send_to_plivo(self) -> None:
        """Send queued audio to Plivo WebSocket in 20ms chunks."""
        PLIVO_CHUNK_SIZE = 160
        audio_buffer = bytearray()

        try:
            while self._running:
                try:
                    audio = await asyncio.wait_for(self._send_queue.get(), timeout=0.1)
                except TimeoutError:
                    continue
                audio_buffer.extend(audio)

                while len(audio_buffer) >= PLIVO_CHUNK_SIZE:
                    chunk = bytes(audio_buffer[:PLIVO_CHUNK_SIZE])
                    audio_buffer = audio_buffer[PLIVO_CHUNK_SIZE:]
                    message = {
                        "event": "playAudio",
                        "media": {
                            "contentType": "audio/x-mulaw",
                            "sampleRate": 8000,
                            "payload": base64.b64encode(chunk).decode("utf-8"),
                        },
                    }
                    await self.websocket.send_text(json.dumps(message))
                    self._plivo_tx_chunks += 1
                    if self._plivo_tx_chunks == 1:
                        self._log("plivo_tx", "first audio chunk sent to Plivo")
                    if self._speech_end_time is not None:
                        ttfs = (time.monotonic() - self._speech_end_time) * 1000
                        self._ttfs_samples.append(ttfs)
                        self._log("metrics", f"TTFS: {ttfs:.0f}ms")
                        self._speech_end_time = None
                    if self._plivo_tx_chunks % 500 == 0:
                        self._logv(
                            "plivo_tx",
                            f"{self._plivo_tx_chunks} chunks sent, "
                            f"queue={self._send_queue.qsize()}",
                        )
        except asyncio.CancelledError:
            pass
        finally:
            self._logv("plivo_tx", f"exiting -- total {self._plivo_tx_chunks} chunks sent")


# =============================================================================
# Public API
# =============================================================================


async def run_agent(
    websocket: WebSocket,
    call_id: str,
    from_number: str = "",
    to_number: str = "",
    system_prompt: str | None = None,
    initial_message: str = "Hello, I'm calling for help.",
    stream_id: str = "",
    parent_call_id: str = "",
    sip_headers: dict[str, str] | None = None,
) -> None:
    """Run a voice agent session for an outbound call."""
    agent = VoiceAgent(
        websocket=websocket,
        call_id=call_id,
        from_number=from_number,
        to_number=to_number,
        system_prompt=system_prompt,
        initial_message=initial_message,
        stream_id=stream_id,
        parent_call_id=parent_call_id,
        sip_headers=sip_headers,
    )
    await agent.run()
