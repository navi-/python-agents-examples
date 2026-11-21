"""Inbound voice agent: OpenAI GPT-Live (full-duplex speech-to-speech).

Native orchestration: one raw WebSocket to the GPT-Live API, a full-duplex voice model
that listens while it speaks and owns turn-taking, backchannels and interruptions:

  Plivo μ-law 8kHz --> GPT-Live (wss://api.openai.com/v1/live/sessions)
                         session.start: model gpt-live-1.1, audio/pcmu 8000,
                         delegation "responses" (backend model + function tools)
  Plivo μ-law 8kHz <-- session.output_audio.delta (base64 μ-law 8kHz)

Both directions are μ-law 8kHz (``audio/pcmu``), so audio is forwarded without
transcoding. There is no client-side VAD and no turn commit: the model hears the caller
continuously. GPT-Live sends no interruption event, so the only client-side barge-in
work is discarding audio that is already buffered: when the caller's speech shows up in
``session.input_transcript.delta`` while more than BARGE_IN_FLUSH_BACKLOG_MS of agent
audio is still unplayed, the send queue is drained and Plivo gets ``clearAudio``.

Tools run in GPT-Live's Responses delegation: the voice model hands a task to the
backend model (OPENAI_DELEGATION_MODEL), whose function calls arrive wrapped in
``response.event`` envelopes; results go back as ``response.item.create`` +
``response.create``.

Pipeline logging is controlled by the LOG_LEVEL env var:
  verbose: every pipeline event: GPT-Live events, packet counts, queue sizes
  normal : key events: turn lifecycle, transcripts, tool calls (default)
  quiet  : errors and session start/end only
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import functools
import json
import os
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import websockets
from dotenv import load_dotenv
from loguru import logger
from websockets.exceptions import ConnectionClosed

from utils import openai_to_plivo, plivo_to_openai

if TYPE_CHECKING:
    from fastapi import WebSocket

load_dotenv()

# ---------------------------------------------------------------------------
# OTel tracing (optional: no-op when opentelemetry is not installed)
# ---------------------------------------------------------------------------
try:
    from opentelemetry import trace as _otel_trace
    from opentelemetry.trace import Status as _Status
    from opentelemetry.trace import StatusCode as _StatusCode

    _tracer = _otel_trace.get_tracer("voice-agent")
except ImportError:
    _otel_trace = None  # type: ignore[assignment]
    _Status = _StatusCode = None  # type: ignore[assignment,misc]
    _tracer = None  # type: ignore[assignment]


def _traced(span_name: str):
    """Decorator that wraps an async method in an OTel span.

    Creates a span with call_id, records exceptions (a task cancelled at teardown is
    marked ``cancelled``, not an error), and ends the span on exit. No-op when
    opentelemetry is not installed.
    """

    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(self, *args, **kwargs):
            if not _tracer:
                return await fn(self, *args, **kwargs)
            with _tracer.start_as_current_span(
                span_name,
                attributes={"call_id": self.parent_call_id},
                record_exception=False,
                set_status_on_exception=False,
            ) as span:
                try:
                    return await fn(self, *args, **kwargs)
                except asyncio.CancelledError:
                    span.set_attribute("cancelled", True)
                    raise
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(_Status(_StatusCode.ERROR, str(exc)))
                    raise

        return wrapper

    return decorator


TRACE_TEXT_MAX = 512  # Truncation for tool text in span attributes


def _truncate(value: Any, limit: int = TRACE_TEXT_MAX) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[: limit - 3] + "..."


# =============================================================================
# Configuration
# =============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# GPT-Live primary WebSocket. No query params: the model goes in session.start.
OPENAI_LIVE_URL = os.getenv("OPENAI_LIVE_URL", "wss://api.openai.com/v1/live/sessions")
# Env values are passed through verbatim into session.start; use OpenAI's own ids.
OPENAI_LIVE_MODEL = os.getenv("OPENAI_LIVE_MODEL", "gpt-live-1.1")
OPENAI_LIVE_VOICE = os.getenv("OPENAI_LIVE_VOICE", "marin")
# Backend model GPT-Live delegates tool work to (session.delegation.responses.model)
OPENAI_DELEGATION_MODEL = os.getenv("OPENAI_DELEGATION_MODEL", "gpt-6-luna")
# delegation.responses.reasoning.effort (none|minimal|low|medium|high|xhigh); omitted when empty
OPENAI_DELEGATION_REASONING_EFFORT = os.getenv("OPENAI_DELEGATION_REASONING_EFFORT", "")

PLIVO_CHUNK_SIZE = 160  # 20ms of μ-law 8kHz mono
CHUNK_S = PLIVO_CHUNK_SIZE / 8000  # Playback time of one chunk
SESSION_START_TIMEOUT_S = 10.0  # Max wait for session.started
SESSION_CLOSE_TIMEOUT_S = 3.0  # Max wait for session.closed (final usage) at teardown
PRE_START_BUFFER_MAX = 16000  # 2s of μ-law 8kHz buffered before session.started
HOUSEKEEPING_INTERVAL_S = 0.1
# The agent's transcript arrives as fragments with no end marker: its text is shown
# (agent_text event) once no fragment has arrived for this long.
AGENT_TEXT_QUIET_S = 1.0
# Barge-in: flush only when at least this much agent audio is still unplayed. GPT-Live
# stops talking by itself; this discards what was already buffered here and at Plivo.
BARGE_IN_FLUSH_BACKLOG_MS = 300
# end_call: hang up once the goodbye has gone quiet for END_CALL_QUIET_S. If the agent
# says nothing after end_call (it said goodbye first), stop waiting after
# END_CALL_GOODBYE_WAIT_S; END_CALL_GRACE_S is the hard limit.
END_CALL_QUIET_S = 1.0
END_CALL_GOODBYE_WAIT_S = 4.0
END_CALL_GRACE_S = 15.0
# Function calls of one backend response are answered together when it completes; if no
# completion event arrives this long after the last call, they are answered anyway.
TOOL_FLUSH_FALLBACK_S = 1.0
TOOL_TIMEOUT_S = 10.0
# μ-law bytes whose segment (exponent) is 0 or 1: amplitude below ~1% of full scale.
# A chunk made only of these is silence or line noise, not agent speech.
QUIET_MULAW_BYTES = bytes(b for b in range(256) if b & 0x70 >= 0x60)
EXAMPLE_NAME = "gptlive1.1-native"

# Logging verbosity: "verbose", "normal" (default), "quiet"
LOG_LEVEL = os.getenv("LOG_LEVEL", "normal").lower()

# =============================================================================
# System Prompt + Greeting
# =============================================================================

# The only prompt source: edit the file, or mount another file over it (see the README).
SYSTEM_PROMPT = (Path(__file__).parent / "system_prompt.md").read_text().strip()

# Spoken when the call connects. GPT-Live has no verbatim-speech field: the greeting is a
# developer message in session.input that tells the model to read this text exactly.
DEFAULT_GREETING = os.getenv(
    "AGENT_GREETING",
    "Hi, this is Alex from TechFlow. I'm built with OpenAI GPT-Live on Plivo. "
    "How can I help you today?",
)

# Instructions for the backend (delegation) model. It never talks to the caller: it runs
# the tools and reports back to the voice model, which does the speaking.
DELEGATION_INSTRUCTIONS = (
    "You are the backend for a phone voice agent. The voice model hands you tasks that "
    "need a tool. Call the matching function with the details from the conversation, "
    "then report the result in one or two plain sentences for the voice model to say "
    "aloud. Never invent results: if a required detail is missing, say which one. "
    "No markdown, lists or special characters."
)

# =============================================================================
# Tool Functions: replace these with your actual implementations
# =============================================================================


async def check_order_status(order_number: str | None, email: str | None) -> dict[str, Any]:
    """Look up order status. Replace with your actual implementation."""
    if not order_number and not email:
        return {"status": "error", "message": "Need order number or email"}

    statuses = [
        {
            "status": "shipped",
            "order_number": order_number or f"TF-{random.randint(100000, 999999)}",
            "shipping_carrier": "FedEx",
            "tracking_number": f"FX{random.randint(1000000000, 9999999999)}",
            "estimated_delivery": (datetime.now() + timedelta(days=2)).strftime("%B %d"),
            "items": "TechFlow Pro Annual Subscription",
        },
        {
            "status": "processing",
            "order_number": order_number or f"TF-{random.randint(100000, 999999)}",
            "message": "Order is being prepared and will ship within 24 hours",
            "items": "TechFlow Teams License (5 seats)",
        },
        {
            "status": "delivered",
            "order_number": order_number or f"TF-{random.randint(100000, 999999)}",
            "delivered_date": (datetime.now() - timedelta(days=1)).strftime("%B %d"),
            "signed_by": "Front Desk",
            "items": "TechFlow Enterprise Setup Kit",
        },
    ]
    return random.choice(statuses)


async def send_sms(phone_number: str, message: str) -> dict[str, Any]:
    """Send SMS to customer. Replace with your actual implementation."""
    if not phone_number:
        return {"status": "error", "message": "Phone number required"}

    return {
        "status": "sent",
        "phone_number": phone_number,
        "message_preview": message[:50] + "..." if len(message) > 50 else message,
        "confirmation_id": f"SMS{random.randint(100000, 999999)}",
    }


async def schedule_callback(
    phone_number: str, reason: str, preferred_time: str, department: str
) -> dict[str, Any]:
    """Schedule a callback. Replace with your actual implementation."""
    if not phone_number:
        return {"status": "error", "message": "Phone number required"}

    return {
        "status": "scheduled",
        "callback_id": f"CB{random.randint(100000, 999999)}",
        "phone_number": phone_number,
        "department": department,
        "scheduled_time": preferred_time or "within 2 business hours",
        "reason": reason,
    }


async def transfer_call(department: str, reason: str) -> dict[str, Any]:
    """Transfer call to human agent. Replace with your actual implementation."""
    return {
        "status": "transferring",
        "department": department,
        "reason": reason,
        "estimated_wait": "less than 2 minutes",
    }


# =============================================================================
# Function Definitions (Responses API function tools, run by the delegation model)
# =============================================================================

FUNCTION_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "check_order_status",
        "description": "Look up the status of a customer's order.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_number": {
                    "type": "string",
                    "description": "Order number (usually starts with TF-)",
                },
                "email": {
                    "type": "string",
                    "description": "Customer's email if order number unavailable",
                },
            },
        },
    },
    {
        "type": "function",
        "name": "send_sms",
        "description": "Send a text message to the customer's phone.",
        "parameters": {
            "type": "object",
            "properties": {
                "phone_number": {"type": "string", "description": "Phone number"},
                "message": {"type": "string", "description": "Message content"},
            },
            "required": ["phone_number", "message"],
        },
    },
    {
        "type": "function",
        "name": "schedule_callback",
        "description": "Schedule a callback from a specialist.",
        "parameters": {
            "type": "object",
            "properties": {
                "phone_number": {"type": "string", "description": "Phone number"},
                "reason": {"type": "string", "description": "Why callback is needed"},
                "preferred_time": {"type": "string", "description": "Preferred time"},
                "department": {"type": "string", "description": "Department"},
            },
            "required": ["phone_number", "reason", "department"],
        },
    },
    {
        "type": "function",
        "name": "transfer_call",
        "description": "Transfer call to human agent.",
        "parameters": {
            "type": "object",
            "properties": {
                "department": {"type": "string", "description": "Department"},
                "reason": {"type": "string", "description": "Transfer reason"},
            },
            "required": ["department", "reason"],
        },
    },
    {
        "type": "function",
        "name": "end_call",
        "description": (
            "End the call once the conversation is complete and goodbyes were said. "
            "The call hangs up after the voice agent's goodbye finishes playing."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "Reason for ending"},
                "resolution": {"type": "string", "description": "How issue was resolved"},
            },
        },
    },
]


# =============================================================================
# Helpers
# =============================================================================


def _is_plivo_disconnect(exc: BaseException) -> bool:
    """True if ``exc`` means Plivo closed the stream (caller hung up / REST hangup).

    Starlette raises ``WebSocketDisconnect`` (whose ``str()`` is empty) when sending on
    or receiving from a closed socket; that is a normal end of call, not an error.
    """
    return (
        type(exc).__name__ in ("WebSocketDisconnect", "ClientDisconnected")
        or isinstance(exc, (ConnectionClosed, ConnectionError))
        or "1000" in str(exc)
    )


def _has_speech(mulaw_8k: bytes) -> bool:
    """True if the chunk holds any sample above the quiet band (see QUIET_MULAW_BYTES)."""
    return bool(mulaw_8k.translate(None, QUIET_MULAW_BYTES))


def _error_text(evt: dict[str, Any]) -> str:
    """``code: message`` of a GPT-Live ``error`` event."""
    error = evt.get("error") or {}
    return f"{error.get('code') or error.get('type') or 'error'}: {error.get('message', '')}"


# =============================================================================
# Send-queue sentinel
# =============================================================================


@dataclass(frozen=True)
class _Checkpoint:
    """Marker placed in the send queue behind the last audio chunk before a hangup.

    When ``_send_to_plivo`` dequeues it, all preceding audio has been sent, so
    the Plivo ``checkpoint`` event is guaranteed to follow the audio it marks.
    """

    name: str


# =============================================================================
# Voice Agent
# =============================================================================


class GPTLiveVoiceAgent:
    """Voice conversation session: Plivo <-> OpenAI GPT-Live."""

    def __init__(
        self,
        websocket: WebSocket,
        call_id: str,
        from_number: str = "",
        to_number: str = "",
        system_prompt: str | None = None,
        initial_message: str = DEFAULT_GREETING,
        stream_id: str = "",
        parent_call_id: str = "",
        sip_headers: dict[str, str] | None = None,
        hangup_callback: Callable[[], Awaitable[None]] | None = None,
    ):
        self.websocket = websocket
        self.call_id = call_id
        self.parent_call_id = parent_call_id or call_id
        # Per-call logger: full IDs as structured fields, same keys as the call_answered event
        self._logger = logger.bind(call_id=self.parent_call_id, leg_call_id=self.call_id)
        self.from_number = from_number
        self.to_number = to_number
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.initial_message = initial_message
        self.sip_headers = sip_headers or {}
        self.hangup_callback = hangup_callback
        self._stream_id = stream_id  # Plivo stream ID for checkpoint/clearAudio events

        # Connection + handshake
        self._running = False
        self._send_queue: asyncio.Queue[bytes | _Checkpoint] = asyncio.Queue()
        self._queued_bytes = 0  # Audio bytes waiting in _send_queue
        self._tx_buffer = bytearray()  # Partial (<160B) audio awaiting the next chunk
        self._live_ws: Any = None
        self._session_started = asyncio.Event()
        self._pre_start_audio = bytearray()
        self._session_id: str | None = None
        self._close_reason: str | None = None
        self._usage: dict[str, Any] = {}

        # Playback tracking. Plivo plays 20ms per chunk, so the time everything sent so
        # far finishes playing is known without a round trip.
        self._play_until = 0.0  # monotonic time the audio already sent to Plivo ends
        self._last_agent_speech = 0.0  # monotonic time of the last non-quiet agent audio
        self._pending_checkpoint: _Checkpoint | None = None
        self._checkpoint_counter = 0

        # Delegation (tools): function calls of the backend response in progress
        self._tool_outputs: list[dict[str, Any]] = []
        self._tool_flush_deadline: float | None = None

        # Hangup
        self._hangup_pending = False
        self._hangup_requested_at = 0.0
        self._hangup_deadline: float | None = None
        self._hangup_done = False

        # Session metrics
        self._turn_count = 0
        self._barge_in_count = 0
        self._error_count = 0
        self._session_start = time.monotonic()
        self._plivo_rx_bytes = 0
        self._plivo_tx_chunks = 0
        self._response_gap_samples: list[int] = []

        # Per-turn state. GPT-Live transcripts are fragments with no turn markers, so a
        # turn is: the caller's fragments, then the agent's, until the caller speaks again.
        self._speaker: str | None = None  # whose transcript fragments are arriving
        self._turn_user_text = ""
        self._turn_agent_text = ""
        self._turn_barge_in = False
        self._turn_response_gap_ms: int | None = None
        self._user_end_ms: int | None = None  # session-timeline end of the caller's speech
        self._user_text_emitted = False
        self._agent_text_emitted_len = 0  # chars of _turn_agent_text already shown
        self._last_agent_fragment = 0.0  # monotonic time of the last agent fragment

    # -- Structured logging with call ID, elapsed time, and pipeline stage --

    def _log(self, stage: str, msg: str, **extra: object) -> None:
        """Log at 'normal' level: key pipeline events."""
        if LOG_LEVEL == "quiet":
            return
        elapsed = round(time.monotonic() - self._session_start, 2)
        self._logger.bind(elapsed_s=elapsed, stage=stage, **extra).info(
            f"[{self.call_id}] [{elapsed:7.2f}s] [{stage}] {msg}"
        )

    def _logv(self, stage: str, msg: str, **extra: object) -> None:
        """Log at 'verbose' level: detailed debugging info."""
        if LOG_LEVEL != "verbose":
            return
        elapsed = round(time.monotonic() - self._session_start, 2)
        self._logger.bind(elapsed_s=elapsed, stage=stage, **extra).debug(
            f"[{self.call_id}] [{elapsed:7.2f}s] [{stage}] {msg}"
        )

    def _loge(self, stage: str, msg: str, **extra: object) -> None:
        """Log errors: always visible regardless of LOG_LEVEL."""
        self._error_count += 1
        elapsed = round(time.monotonic() - self._session_start, 2)
        self._logger.bind(elapsed_s=elapsed, stage=stage, **extra).error(
            f"[{self.call_id}] [{elapsed:7.2f}s] [{stage}] {msg}"
        )

    # -- session.start --

    def _build_call_context(self) -> str:
        """Per-call context appended to the prompt ("" when the caller is unknown)."""
        if not self.from_number:
            return ""
        call_time = datetime.now().strftime("%I:%M %p on %A, %B %d")
        return f"""

## Current Call Context
- Caller's phone number: {self.from_number}
- Call ID: {self.call_id}
- Time: {call_time}

You can use the caller's phone number for SMS or callbacks without asking."""

    def _build_system_prompt(self) -> str:
        """System prompt plus the per-call context."""
        return self.system_prompt + self._build_call_context()

    def _build_greeting_input(self) -> list[dict[str, Any]]:
        """session.input: a developer message that makes the model speak first.

        GPT-Live has no "say this" field; OpenAI's guidance is a developer message that
        tells the model to greet now, with the exact line in ``<say>`` tags.
        """
        if not self.initial_message:
            return []
        text = (
            "The call has just connected. Greet the caller now. Read the following "
            "exactly, without an introduction or explanation, then pause and listen: "
            f"<say>{self.initial_message}</say>"
        )
        return [
            {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": text}],
            }
        ]

    def _build_delegation(self) -> dict[str, Any]:
        """session.delegation: Responses mode, the backend model runs FUNCTION_DEFINITIONS."""
        responses: dict[str, Any] = {
            "model": OPENAI_DELEGATION_MODEL,
            "instructions": DELEGATION_INSTRUCTIONS + self._build_call_context(),
            "tools": FUNCTION_DEFINITIONS,
            "tool_choice": "auto",
        }
        if OPENAI_DELEGATION_REASONING_EFFORT:
            responses["reasoning"] = {"effort": OPENAI_DELEGATION_REASONING_EFFORT}
        return {"type": "responses", "responses": responses}

    def _build_session_start(self) -> dict[str, Any]:
        """Build the GPT-Live ``session.start`` message (the first message on the socket).

        ``audio.format`` is one object for both directions: ``audio/pcmu`` at 8000 is
        Plivo's own format. Model, instructions, audio format and voice cannot be changed
        after the session has started.
        """
        session: dict[str, Any] = {
            "model": OPENAI_LIVE_MODEL,
            "instructions": self._build_system_prompt(),
            "audio": {
                "format": {"type": "audio/pcmu", "rate": 8000},
                "output": {"voice": OPENAI_LIVE_VOICE},
            },
            "delegation": self._build_delegation(),
        }
        greeting_input = self._build_greeting_input()
        if greeting_input:
            session["input"] = greeting_input
        return {"type": "session.start", "session": session}

    # -- Session lifecycle --

    @_traced("session")
    async def run(self) -> None:
        """Run the voice agent session."""
        self._session_start = time.monotonic()
        self._running = True
        # Session start always logs (even in quiet mode), so it does not go through _log.
        # One line carries both the console message and the call_answered event.
        self._logger.bind(
            event="call_answered",
            elapsed_s=0.0,
            stage="session",
            from_number=self.from_number,
            to_number=self.to_number,
            sip_headers=self.sip_headers,
            stream_id=self._stream_id,
        ).info(
            f"[{self.call_id}] [  0.00s] [session] "
            f"started (from={self.from_number}, to={self.to_number}, log={LOG_LEVEL}, "
            f"model={OPENAI_LIVE_MODEL}, delegation={OPENAI_DELEGATION_MODEL}, "
            f"sip_headers={self.sip_headers})"
        )
        if _otel_trace is not None:
            _otel_trace.get_current_span().set_attributes(
                {
                    "leg_call_id": self.call_id,
                    "gen_ai.system": "openai",
                    "gen_ai.request.model": OPENAI_LIVE_MODEL,
                    "openai.live.voice": OPENAI_LIVE_VOICE,
                    "openai.live.delegation_model": OPENAI_DELEGATION_MODEL,
                }
            )

        try:
            async with websockets.connect(
                OPENAI_LIVE_URL,
                additional_headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                max_size=None,
                ping_interval=20,
            ) as live_ws:
                self._live_ws = live_ws
                self._log("session", "connected to GPT-Live")
                try:
                    await self._run_streaming_tasks(live_ws)
                finally:
                    await self._close_live_session(live_ws)
        except Exception as e:
            self._loge("session", f"ERROR: {e}")
            if _otel_trace is not None:
                span = _otel_trace.get_current_span()
                span.record_exception(e)
                span.set_status(_Status(_StatusCode.ERROR, str(e)))
        finally:
            self._running = False
            self._finish_turn()
            self._emit_session_end()

    async def _run_streaming_tasks(self, live_ws: Any) -> None:
        """Run the three concurrent streaming tasks."""
        tasks = [
            asyncio.create_task(self._receive_from_plivo(), name="plivo_rx"),
            asyncio.create_task(self._receive_from_openai(live_ws), name="openai_rx"),
            asyncio.create_task(self._send_to_plivo(), name="plivo_tx"),
        ]
        try:
            done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if not task.cancelled() and task.exception():
                    self._loge("session", f"task {task.get_name()} failed: {task.exception()!r}")
        finally:
            self._running = False
            for task in tasks:
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

    async def _handshake(self, live_ws: Any) -> None:
        """session.start -> session.started, then flush buffered caller audio.

        session.start must be the first message; no audio may be sent before
        session.started.
        """
        t0 = time.monotonic()
        await live_ws.send(json.dumps(self._build_session_start()))
        self._logv("openai", "session.start sent")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + SESSION_START_TIMEOUT_S
        try:
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise TimeoutError
                raw = await asyncio.wait_for(live_ws.recv(), timeout=remaining)
                if isinstance(raw, bytes):
                    self._logv("openai", f"ignoring {len(raw)}B binary during handshake")
                    continue
                evt = json.loads(raw)
                etype = evt.get("type")
                if etype == "session.started":
                    self._on_session_started(evt)
                    break
                if etype == "error":
                    raise RuntimeError(f"GPT-Live error during session.start: {_error_text(evt)}")
                if etype == "session.closed":
                    raise RuntimeError(f"GPT-Live closed the session: {evt.get('reason')}")
                self._logv("openai", f"event before session.started: {etype}")
        except (TimeoutError, asyncio.TimeoutError) as e:
            raise RuntimeError(
                f"GPT-Live session.started not received within {SESSION_START_TIMEOUT_S:.0f}s"
            ) from e

        # Flush caller audio buffered while waiting. Loop until empty with no await between
        # the final emptiness check and set(), so live audio can't overtake buffered audio.
        flushed = 0
        while self._pre_start_audio:
            audio = bytes(self._pre_start_audio)
            self._pre_start_audio.clear()
            await self._send_audio_to_openai(audio)
            flushed += len(audio)
        self._session_started.set()
        self._log(
            "openai",
            f"session.started in {(time.monotonic() - t0) * 1000:.0f}ms "
            f"(id={self._session_id}, flushed {flushed}B audio)",
        )

    def _on_session_started(self, evt: dict[str, Any]) -> None:
        session = evt.get("session") or {}
        self._session_id = session.get("id")
        if _otel_trace is not None and self._session_id:
            _otel_trace.get_current_span().set_attribute("openai.live.session_id", self._session_id)
        self._logv("openai", f"session expires_at={session.get('expires_at')}")

    async def _close_live_session(self, live_ws: Any) -> None:
        """Ask GPT-Live to close and wait briefly for session.closed (final usage).

        A socket that just drops leaves the session's finalization unconfirmed.
        """
        if not self._session_started.is_set() or self._close_reason is not None:
            return
        try:
            await live_ws.send(json.dumps({"type": "session.close"}))
            await asyncio.wait_for(self._drain_until_closed(live_ws), SESSION_CLOSE_TIMEOUT_S)
        except (TimeoutError, asyncio.TimeoutError):
            self._logv("openai", "session.closed not received before timeout")
        except Exception as e:
            self._logv("openai", f"session.close failed: {e!r}")

    async def _drain_until_closed(self, live_ws: Any) -> None:
        async for raw in live_ws:
            if isinstance(raw, bytes):
                continue
            evt = json.loads(raw)
            if evt.get("type") == "session.usage.updated":
                self._usage = evt.get("usage") or self._usage
            elif evt.get("type") == "session.closed":
                self._on_session_closed(evt)
                return

    def _on_session_closed(self, evt: dict[str, Any]) -> None:
        self._close_reason = str(evt.get("reason") or "closed")
        usage = evt.get("usage") or (evt.get("session") or {}).get("usage")
        if usage:
            self._usage = usage
        self._log("openai", f"session.closed (reason={self._close_reason}, usage={self._usage})")

    # -- plivo_rx --

    @_traced("plivo_rx")
    async def _receive_from_plivo(self) -> None:
        """Receive Plivo events: forward caller audio to GPT-Live, handle checkpoint acks."""
        media_count = 0
        try:
            while self._running:
                data = await self.websocket.receive_text()
                try:
                    message = json.loads(data)
                except json.JSONDecodeError:
                    self._logv("plivo_rx", f"ignoring non-JSON frame: {data[:60]!r}")
                    continue
                event = message.get("event")

                if event == "media":
                    payload = message.get("media", {}).get("payload", "")
                    if not payload:
                        continue
                    try:
                        mulaw_audio = base64.b64decode(payload, validate=True)
                    except (binascii.Error, ValueError):
                        self._logv("plivo_rx", "ignoring media frame with bad base64")
                        continue
                    if not mulaw_audio:
                        continue
                    self._plivo_rx_bytes += len(mulaw_audio)
                    media_count += 1
                    if media_count == 1:
                        self._log("plivo_rx", "first audio packet received")
                    if media_count % 500 == 0:
                        self._logv("plivo_rx", f"{media_count} packets")

                    live_audio = plivo_to_openai(mulaw_audio)
                    if not self._session_started.is_set():
                        self._pre_start_audio.extend(live_audio)
                        overflow = len(self._pre_start_audio) - PRE_START_BUFFER_MAX
                        if overflow > 0:
                            del self._pre_start_audio[:overflow]
                    else:
                        await self._send_audio_to_openai(live_audio)

                elif event == "playedStream":
                    await self._on_played_stream(message.get("name", ""))

                elif event == "clearedAudio":
                    self._logv("plivo_rx", "clearedAudio confirmed by Plivo")

                elif event == "stop":
                    self._log("plivo_rx", "received stop event -- call ended")
                    break

        except Exception as e:
            if _is_plivo_disconnect(e):
                self._log("plivo_rx", f"Plivo stream closed ({type(e).__name__})")
            elif self._running:
                self._loge("plivo_rx", f"ERROR: {e!r}")
        finally:
            self._logv("plivo_rx", f"exiting -- received {media_count} media packets")

    async def _send_audio_to_openai(self, audio: bytes) -> None:
        """Append caller audio to the GPT-Live session (base64 μ-law 8kHz)."""
        if self._live_ws is None:
            return
        await self._live_ws.send(
            json.dumps(
                {
                    "type": "session.input_audio.append",
                    "audio": base64.b64encode(audio).decode("ascii"),
                }
            )
        )

    # -- openai_rx --

    @_traced("openai_rx")
    async def _receive_from_openai(self, live_ws: Any) -> None:
        """Handshake, then dispatch GPT-Live events and queue agent audio."""
        await self._handshake(live_ws)
        try:
            async for msg in live_ws:
                if isinstance(msg, bytes):
                    self._logv("openai", f"ignoring {len(msg)}B binary frame")
                    continue
                try:
                    evt = json.loads(msg)
                except json.JSONDecodeError:
                    self._logv("openai", f"non-JSON text frame: {msg[:80]!r}")
                    continue
                if not await self._handle_openai_event(evt):
                    return
        except ConnectionClosed as e:
            if self._running:
                self._log("openai", f"connection closed: {e}")

    async def _handle_openai_event(self, evt: dict[str, Any]) -> bool:
        """Handle one GPT-Live server event. Returns False if the session must end."""
        etype = evt.get("type", "")

        if etype == "session.output_audio.delta":
            self._on_agent_audio(evt.get("delta", ""))
        elif etype == "session.input_transcript.delta":
            await self._on_transcript("user", evt)
        elif etype == "session.output_transcript.delta":
            await self._on_transcript("agent", evt)
        elif etype == "response.event":
            await self._on_delegated_event(evt.get("event") or {})
        elif etype == "session.delegation.created":
            delegation = evt.get("delegation") or {}
            self._log(
                "openai",
                f"delegation created (id={delegation.get('id')}, "
                f"target={delegation.get('target')})",
            )
        elif etype == "session.usage.updated":
            self._usage = evt.get("usage") or self._usage  # cumulative snapshot
            self._logv("openai", f"usage: {evt.get('usage')} context: {evt.get('context_window')}")
        elif etype == "session.started":
            self._on_session_started(evt)
        elif etype == "session.closed":
            self._on_session_closed(evt)
            return False
        elif etype == "error":
            # Errors answer one client event; the session stays open unless it then
            # sends session.closed.
            self._loge("openai", f"error: {_error_text(evt)}")
        elif etype == "info":
            self._log("openai", f"info: {evt.get('code', '')} {evt.get('message', '')}")
        else:
            self._logv("openai", f"event {etype}: {str(evt)[:120]}")
        return True

    def _on_agent_audio(self, delta: str) -> None:
        """Queue a base64 agent-audio delta for Plivo."""
        try:
            audio = openai_to_plivo(base64.b64decode(delta, validate=True))
        except (binascii.Error, ValueError):
            self._logv("openai", "ignoring audio delta with bad base64")
            return
        if not audio:
            return
        if _has_speech(audio):
            self._last_agent_speech = time.monotonic()
        self._queued_bytes += len(audio)
        self._send_queue.put_nowait(audio)

    def _backlog_ms(self) -> int:
        """Agent audio not yet heard by the caller: queued here + sent but unplayed at Plivo."""
        queued = (self._queued_bytes + len(self._tx_buffer)) / 8
        unplayed = max(0.0, self._play_until - time.monotonic()) * 1000
        return round(queued + unplayed)

    async def _on_transcript(self, role: str, evt: dict[str, Any]) -> None:
        """Transcript fragment of the caller (``user``) or the agent.

        Fragments carry session-timeline ``start_ms``/``end_ms`` but no turn markers, so
        turns are cut where the speaker changes: caller fragments after agent fragments
        start the next turn.
        """
        delta = evt.get("delta") or ""
        if not delta:
            return
        if role == "user":
            if self._speaker != "user":
                if self._turn_count > 0:
                    # The caller speaks while or after the agent did: flush buffered
                    # agent audio if there is any, then close the agent's turn.
                    await self._maybe_barge_in()
                    self._finish_turn()
                self._turn_count += 1
                self._speaker = "user"
            self._turn_user_text += delta
            if isinstance(evt.get("end_ms"), int):
                self._user_end_ms = evt["end_ms"]
            self._logv("turn", f"user fragment: {delta!r}")
        else:
            if self._turn_count == 0:
                self._turn_count = 1  # greeting
            if self._speaker != "agent":
                self._on_agent_reply_started(evt.get("start_ms"))
                self._speaker = "agent"
            self._turn_agent_text += delta
            self._last_agent_fragment = time.monotonic()
            self._logv("turn", f"agent fragment: {delta!r}")

    def _on_agent_reply_started(self, start_ms: Any) -> None:
        """First agent fragment of a turn: the caller's text is complete, log it."""
        self._emit_user_text()
        if isinstance(start_ms, int) and self._user_end_ms is not None:
            # Gap on GPT-Live's own timeline: end of caller speech -> start of the reply
            self._turn_response_gap_ms = start_ms - self._user_end_ms
            self._response_gap_samples.append(self._turn_response_gap_ms)
            self._log("metrics", f"response gap: {self._turn_response_gap_ms}ms")

    async def _maybe_barge_in(self) -> None:
        """Caller speech arrived: discard agent audio that has not been played yet.

        GPT-Live handles the interruption itself (it hears the caller while speaking) and
        sends no event for it. What it cannot do is take back audio already delivered, so
        a backlog above BARGE_IN_FLUSH_BACKLOG_MS is dropped here and cleared at Plivo.
        """
        backlog_ms = self._backlog_ms()
        if backlog_ms < BARGE_IN_FLUSH_BACKLOG_MS:
            return
        cleared = 0
        while not self._send_queue.empty():
            try:
                item = self._send_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if isinstance(item, _Checkpoint):
                continue
            cleared += 1
        self._queued_bytes = 0
        self._tx_buffer.clear()
        self._pending_checkpoint = None
        clear_event: dict[str, Any] = {"event": "clearAudio"}
        if self._stream_id:
            clear_event["streamId"] = self._stream_id
        with contextlib.suppress(Exception):
            await self.websocket.send_text(json.dumps(clear_event))
        self._play_until = time.monotonic()
        self._barge_in_count += 1
        self._turn_barge_in = True
        self._log("barge_in", f"flushed {backlog_ms}ms of unplayed agent audio (queue={cleared})")

    def _emit_user_text(self) -> None:
        """Show the caller's text of the open turn (user_text event), once per turn."""
        user_text = self._turn_user_text.strip()
        if not user_text or self._user_text_emitted:
            return
        self._user_text_emitted = True
        self._log(
            "turn",
            f"turn {self._turn_count}: user '{user_text[:80]}'",
            event="user_text",
            turn=self._turn_count,
            text=user_text,
        )

    def _emit_agent_text(self) -> None:
        """Show the agent's text of the open turn that was not shown yet (agent_text event).

        Called when the agent's transcript goes quiet, so the text is visible while the
        caller listens instead of only when the turn closes. Text that arrives later in
        the same turn (the agent resumed) is emitted as its own line by the next call.
        """
        agent_text = self._turn_agent_text[self._agent_text_emitted_len :].strip()
        if not agent_text:
            return
        self._agent_text_emitted_len = len(self._turn_agent_text)
        self._log(
            "llm",
            f"turn {self._turn_count}: agent '{agent_text[:80]}'",
            event="agent_text",
            turn=self._turn_count,
            text=agent_text,
        )

    def _finish_turn(self) -> None:
        """Close the open turn: show any text not shown yet and emit turn_complete."""
        if self._turn_count == 0 or not (self._turn_user_text or self._turn_agent_text):
            return
        # Whatever was not shown yet: a caller turn the agent never answered, and agent
        # text that arrived after the quiet-period emit
        self._emit_user_text()
        self._emit_agent_text()
        self._emit_turn_complete(barge_in=self._turn_barge_in)
        self._speaker = None
        self._turn_user_text = ""
        self._turn_agent_text = ""
        self._turn_barge_in = False
        self._turn_response_gap_ms = None
        self._user_end_ms = None
        self._user_text_emitted = False
        self._agent_text_emitted_len = 0

    # -- Delegation (tools) --

    async def _on_delegated_event(self, event: dict[str, Any]) -> None:
        """A Responses API event of the backend model, unwrapped from ``response.event``.

        Function calls are read from ``response.output_item.done`` items (the forwarded
        lifecycle events carry an empty ``output``). Every call of a response must be
        answered before ``response.create`` resumes the backend, so results are held
        until the response completes.
        """
        etype = event.get("type", "")
        if etype == "response.output_item.done":
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                await self._run_function_call(item)
        elif etype in ("response.completed", "response.done", "response.incomplete"):
            await self._flush_tool_outputs()
        elif etype in ("response.failed", "error"):
            self._loge("tool", f"delegated response failed: {str(event)[:200]}")
            await self._flush_tool_outputs()
        else:
            self._logv("openai", f"delegated event {etype}")

    async def _run_function_call(self, item: dict[str, Any]) -> None:
        """Run one function call and hold its result for ``_flush_tool_outputs``."""
        call_id = item.get("call_id", "")
        name = item.get("name", "")
        arguments = item.get("arguments", "")
        span = None
        if _tracer:
            span = _tracer.start_span(
                f"tool.{name}",
                attributes={
                    "call_id": self.parent_call_id,
                    "tool.name": name,
                    "tool.call_id": call_id,
                    "tool.arguments": _truncate(arguments),
                },
            )
        status = "cancelled"
        result: dict[str, Any] | None = None
        try:
            try:
                result = await asyncio.wait_for(
                    self._handle_function_call(name, arguments), timeout=TOOL_TIMEOUT_S
                )
                status = "error" if "error" in result else str(result.get("status", "ok"))
            except (TimeoutError, asyncio.TimeoutError):
                self._loge("tool", f"{name} timed out")
                result = {"error": "function timed out"}
                status = "timeout"
            self._tool_outputs.append(
                {"type": "function_call_output", "call_id": call_id, "output": json.dumps(result)}
            )
            self._tool_flush_deadline = time.monotonic() + TOOL_FLUSH_FALLBACK_S
        finally:
            if span is not None:
                span.set_attribute("tool.status", status)
                if result is not None:
                    span.set_attribute("tool.result", _truncate(result))
                if status in ("error", "timeout", "cancelled"):
                    span.set_status(_Status(_StatusCode.ERROR, status))
                span.end()

    async def _flush_tool_outputs(self) -> None:
        """Send every held function result, then ``response.create`` to resume the backend."""
        outputs, self._tool_outputs = self._tool_outputs, []
        self._tool_flush_deadline = None
        if not outputs or self._live_ws is None:
            return
        for item in outputs:
            await self._live_ws.send(json.dumps({"type": "response.item.create", "item": item}))
        await self._live_ws.send(json.dumps({"type": "response.create"}))
        self._logv("tool", f"returned {len(outputs)} function result(s) + response.create")

    async def _handle_function_call(self, name: str, arguments: Any) -> dict[str, Any]:
        """Execute a function call and return the result."""
        if isinstance(arguments, dict):
            args = arguments
        else:
            try:
                args = json.loads(arguments) if arguments else {}
            except (json.JSONDecodeError, TypeError):
                args = {}
            if not isinstance(args, dict):
                args = {}

        self._log("tool", f"calling {name}({args})")

        try:
            if name == "check_order_status":
                result = await check_order_status(
                    order_number=args.get("order_number"),
                    email=args.get("email"),
                )
            elif name == "send_sms":
                result = await send_sms(
                    phone_number=args.get("phone_number", ""),
                    message=args.get("message", ""),
                )
            elif name == "schedule_callback":
                result = await schedule_callback(
                    phone_number=args.get("phone_number", ""),
                    reason=args.get("reason", ""),
                    preferred_time=args.get("preferred_time", ""),
                    department=args.get("department", "general"),
                )
            elif name == "transfer_call":
                result = await transfer_call(
                    department=args.get("department", "support"),
                    reason=args.get("reason", "Customer requested transfer"),
                )
            elif name == "end_call":
                self._log("tool", f"end_call: {args.get('reason')} -- hangup after goodbye")
                now = time.monotonic()
                self._hangup_pending = True
                self._hangup_requested_at = now
                self._hangup_deadline = now + END_CALL_GRACE_S
                result = {"status": "call_ending", "reason": args.get("reason", "")}
            else:
                result = {"error": f"Unknown function: {name}"}

            self._log("tool", f"{name} -> {result.get('status', 'done')}")
            return result

        except Exception as e:
            self._loge("tool", f"{name} ERROR: {e}")
            return {"error": str(e)}

    # -- Hangup --

    def _goodbye_finished(self, now: float) -> bool:
        """True once the agent's goodbye after end_call has gone quiet.

        GPT-Live has no end-of-speech event, so quiet is measured on the audio itself:
        no chunk with speech for END_CALL_QUIET_S. If nothing was said after end_call
        (the goodbye came first), waiting stops after END_CALL_GOODBYE_WAIT_S.
        """
        spoke_after = self._last_agent_speech > self._hangup_requested_at
        if spoke_after:
            return now - self._last_agent_speech >= END_CALL_QUIET_S
        return now - self._hangup_requested_at >= END_CALL_GOODBYE_WAIT_S

    async def _on_played_stream(self, name: str) -> None:
        """Plivo confirms all audio before the named checkpoint has played."""
        pending = self._pending_checkpoint
        if pending is None or name != pending.name:
            self._logv("plivo_rx", f"ignoring stale playedStream '{name}'")
            return
        self._pending_checkpoint = None
        self._log("plivo_rx", f"playedStream '{name}' -- playback complete")
        if self._hangup_pending:
            await self._finish_hangup()

    async def _finish_hangup(self) -> None:
        """Hang up via the server-provided callback (idempotent), then stop the session."""
        if self._hangup_done:
            return
        self._hangup_done = True
        self._hangup_pending = False
        self._log("session", "goodbye played -- hanging up")
        if self.hangup_callback is not None:
            try:
                await self.hangup_callback()
            except Exception as e:
                self._loge("session", f"hangup callback ERROR: {e}")
        self._running = False

    # -- plivo_tx --

    @_traced("plivo_tx")
    async def _send_to_plivo(self) -> None:
        """Send queued audio to Plivo in 160-byte (20ms) chunks, plus checkpoints."""
        last_housekeeping = time.monotonic()
        try:
            while self._running:
                # Time-based, not only when the queue is idle: GPT-Live may stream audio
                # without gaps, and the hangup and tool deadlines must still be checked.
                now = time.monotonic()
                if now - last_housekeeping >= HOUSEKEEPING_INTERVAL_S:
                    last_housekeeping = now
                    await self._housekeeping(now)
                try:
                    item = await asyncio.wait_for(
                        self._send_queue.get(), timeout=HOUSEKEEPING_INTERVAL_S
                    )
                except (TimeoutError, asyncio.TimeoutError):
                    continue

                if isinstance(item, _Checkpoint):
                    if self._tx_buffer:
                        pad = PLIVO_CHUNK_SIZE - len(self._tx_buffer)
                        self._tx_buffer.extend(b"\xff" * pad)  # μ-law silence
                        await self._send_play_audio(bytes(self._tx_buffer))
                        self._tx_buffer.clear()
                    await self.websocket.send_text(
                        json.dumps(
                            {
                                "event": "checkpoint",
                                "streamId": self._stream_id,
                                "name": item.name,
                            }
                        )
                    )
                    self._logv("plivo_tx", f"checkpoint sent: {item.name}")
                    continue

                self._queued_bytes = max(0, self._queued_bytes - len(item))
                self._tx_buffer.extend(item)
                while len(self._tx_buffer) >= PLIVO_CHUNK_SIZE:
                    chunk = bytes(self._tx_buffer[:PLIVO_CHUNK_SIZE])
                    del self._tx_buffer[:PLIVO_CHUNK_SIZE]
                    await self._send_play_audio(chunk)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            if not _is_plivo_disconnect(e):
                raise
            self._log("plivo_tx", f"Plivo stream closed ({type(e).__name__})")
        finally:
            self._logv("plivo_tx", f"exiting -- total {self._plivo_tx_chunks} chunks sent")

    async def _send_play_audio(self, chunk: bytes) -> None:
        """Send one playAudio message to Plivo and advance the playback clock."""
        message = {
            "event": "playAudio",
            "media": {
                "contentType": "audio/x-mulaw",
                "sampleRate": 8000,
                "payload": base64.b64encode(chunk).decode("utf-8"),
            },
        }
        await self.websocket.send_text(json.dumps(message))
        self._play_until = max(time.monotonic(), self._play_until) + CHUNK_S
        self._plivo_tx_chunks += 1
        if self._plivo_tx_chunks == 1:
            self._log("plivo_tx", "first audio chunk sent to Plivo")
        if self._plivo_tx_chunks % 500 == 0:
            self._logv(
                "plivo_tx",
                f"{self._plivo_tx_chunks} chunks sent, backlog={self._backlog_ms()}ms",
            )

    async def _housekeeping(self, now: float) -> None:
        """Tool-result fallback flush; hang up once the goodbye has been played."""
        if self._tool_flush_deadline is not None and now >= self._tool_flush_deadline:
            self._logv("tool", "no completion event for the backend response -- answering now")
            with contextlib.suppress(Exception):
                await self._flush_tool_outputs()

        if (
            self._speaker == "agent"
            and len(self._turn_agent_text) > self._agent_text_emitted_len
            and now - self._last_agent_fragment >= AGENT_TEXT_QUIET_S
            and not self._tool_outputs
            and self._tool_flush_deadline is None
        ):
            self._emit_agent_text()

        if not self._hangup_pending:
            return
        if self._hangup_deadline is not None and now >= self._hangup_deadline:
            self._log("session", "hangup deadline reached -- forcing hangup")
            await self._finish_hangup()
        elif self._pending_checkpoint is None and self._goodbye_finished(now):
            if not self._stream_id:
                # No Plivo stream (local tests): no checkpoint/playedStream round trip
                await self._finish_hangup()
                return
            # Plivo confirms with playedStream once everything sent so far has played
            self._checkpoint_counter += 1
            checkpoint = _Checkpoint(f"goodbye_{self._checkpoint_counter}")
            self._pending_checkpoint = checkpoint
            self._send_queue.put_nowait(checkpoint)
            self._logv("session", f"goodbye went quiet -> queued checkpoint {checkpoint.name}")

    # -- Structured events --

    def _emit_turn_complete(self, *, barge_in: bool) -> None:
        """Emit a structured turn_complete event with per-turn metrics."""
        self._logger.bind(
            event="turn_complete",
            call_id=self.parent_call_id,
            turn=self._turn_count,
            barge_in=barge_in,
            user_text=self._turn_user_text.strip(),
            agent_text=self._turn_agent_text.strip(),
            plivo_rx_bytes=self._plivo_rx_bytes,
            plivo_tx_chunks=self._plivo_tx_chunks,
            response_gap_ms=self._turn_response_gap_ms,
        ).info(
            f"[{self.call_id}] turn {self._turn_count} complete{' (barge-in)' if barge_in else ''}"
        )

    def _emit_session_end(self) -> None:
        """Emit the session_end summary event (always logged, even in quiet mode)."""
        duration = round(time.monotonic() - self._session_start, 1)
        gaps = self._response_gap_samples
        avg_gap = round(sum(gaps) / len(gaps)) if gaps else None
        self._logger.bind(
            event="session_end",
            call_id=self.parent_call_id,
            duration_s=duration,
            turns=self._turn_count,
            barge_ins=self._barge_in_count,
            errors=self._error_count,
            # Time to first speech: end of the caller's speech to the start of the reply
            ttfs_avg_ms=avg_gap,
            response_gap_avg_ms=avg_gap,
            response_gap_samples=len(gaps),
            rx_bytes=self._plivo_rx_bytes,
            tx_chunks=self._plivo_tx_chunks,
            openai_session_id=self._session_id,
            openai_close_reason=self._close_reason,
            openai_usage=dict(self._usage),
        ).info(
            f"[{self.call_id}] [{duration:7.1f}s] [session] "
            f"ended -- {self._turn_count} turns, "
            f"{self._barge_in_count} barge-ins, "
            f"response gap avg={avg_gap}ms, "
            f"rx={self._plivo_rx_bytes}B, tx={self._plivo_tx_chunks} chunks"
        )


# =============================================================================
# Public API
# =============================================================================


async def run_agent(
    websocket: WebSocket,
    call_id: str,
    from_number: str = "",
    to_number: str = "",
    system_prompt: str | None = None,
    initial_message: str = DEFAULT_GREETING,
    stream_id: str = "",
    parent_call_id: str = "",
    sip_headers: dict[str, str] | None = None,
    hangup_callback: Callable[[], Awaitable[None]] | None = None,
) -> None:
    """Run a voice agent session for an inbound call."""
    agent = GPTLiveVoiceAgent(
        websocket=websocket,
        call_id=call_id,
        from_number=from_number,
        to_number=to_number,
        system_prompt=system_prompt,
        initial_message=initial_message,
        stream_id=stream_id,
        parent_call_id=parent_call_id,
        sip_headers=sip_headers,
        hangup_callback=hangup_callback,
    )
    await agent.run()
