"""
Integration tests for the OpenAI GPT-Live + Plivo example.

Test Levels:
1. Unit Tests (offline) - audio pass-through, phone normalization, session.start builder,
   function dispatch, GPT-Live event handling with fake WebSockets, outbound call
   details (prompt/greeting), server routes via FastAPI TestClient
2. Local Integration (offline) - start the inbound server against a local fake of the
   GPT-Live API (tests/fake_live.py) and drive the Plivo WebSocket protocol
3. GPT-Live Integration - connect directly to the real GPT-Live API (needs OPENAI_API_KEY)
4. Plivo Integration - validate Plivo credentials and phone number

Run tests:
    uv run pytest tests/test_integration.py -v

Run specific test level:
    uv run pytest tests/test_integration.py -v -k "unit"
    uv run pytest tests/test_integration.py -v -k "local or GPTLive or Plivo"
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import struct
import time
import uuid
from pathlib import Path
from typing import Any, ClassVar

import httpx
import plivo
import pytest
import websockets
from dotenv import load_dotenv
from loguru import logger

from tests.fake_live import FakeLiveServer, FakeLiveThread
from tests.helpers import (
    plivo_signature_headers,
    read_log_events,
    rms_of_ulaw,
    server_log_path,
    start_server,
    stop_server,
    stream_body,
    stream_signature_headers,
    stream_url_from_xml,
    tone_ulaw,
)
from utils import normalize_phone_number, openai_to_plivo, plivo_to_openai

load_dotenv()

# Configuration from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID", "")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN", "")
PLIVO_PHONE_NUMBER = os.getenv("PLIVO_PHONE_NUMBER", "")

TEST_PORT = 18001
LOCAL_HTTP_URL = f"http://localhost:{TEST_PORT}"

STREAM_ID = "stream-0001"
# Plivo auth token the unit tests sign webhooks with (never a real credential)
TEST_AUTH_TOKEN = "test-plivo-auth-token"
CALL_ID = "call-1234-5678-90ab-cdef00000001"


# =============================================================================
# Fakes + fixtures for offline unit tests
# =============================================================================


class FakePlivoWS:
    """Stands in for the FastAPI WebSocket connected to Plivo."""

    def __init__(self, auto_played_stream: bool = False) -> None:
        self.incoming: asyncio.Queue[Any] = asyncio.Queue()
        self.sent: list[dict[str, Any]] = []
        self.auto_played_stream = auto_played_stream

    async def receive_text(self) -> str:
        item = await self.incoming.get()
        if isinstance(item, BaseException):
            raise item
        return item

    async def send_text(self, text: str) -> None:
        msg = json.loads(text)
        self.sent.append(msg)
        if self.auto_played_stream and msg.get("event") == "checkpoint":
            # Plivo acks the checkpoint once playback finishes, then the caller hangs up
            self.push({"event": "playedStream", "name": msg["name"], "streamId": STREAM_ID})
            self.push({"event": "stop"})

    def push(self, message: dict[str, Any]) -> None:
        self.incoming.put_nowait(json.dumps(message))

    def events(self, name: str) -> list[dict[str, Any]]:
        return [m for m in self.sent if m.get("event") == name]


class FakeLiveWS:
    """Stands in for the GPT-Live WebSocket (recv, async-iter, send, async with)."""

    def __init__(self, script: list[Any] | None = None) -> None:
        self.sent: list[Any] = []
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        for item in script or []:
            self.feed(item)

    def feed(self, item: Any) -> None:
        """Queue a server message: dict -> JSON text, None -> EOF."""
        if isinstance(item, dict):
            item = json.dumps(item)
        self._queue.put_nowait(item)

    async def send(self, data: Any) -> None:
        self.sent.append(data)

    async def recv(self) -> Any:
        item = await self._queue.get()
        if item is None:
            raise websockets.exceptions.ConnectionClosedOK(None, None)
        return item

    def __aiter__(self) -> FakeLiveWS:
        return self

    async def __anext__(self) -> Any:
        item = await self._queue.get()
        if item is None:
            raise StopAsyncIteration
        return item

    async def __aenter__(self) -> FakeLiveWS:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def sent_json(self) -> list[dict[str, Any]]:
        return [json.loads(s) for s in self.sent if isinstance(s, str)]

    def sent_types(self) -> list[str]:
        return [m.get("type", "") for m in self.sent_json()]


@pytest.fixture(autouse=True)
def plivo_test_auth_token(monkeypatch):
    """Both servers check webhook and /ws stream signatures with TEST_AUTH_TOKEN."""
    from inbound import server as inbound_server
    from outbound import server as outbound_server

    for server in (inbound_server, outbound_server):
        monkeypatch.setattr(server, "PLIVO_AUTH_TOKEN", TEST_AUTH_TOKEN)


def signed(method: str, public_url: str, path: str, data: dict | None = None) -> dict[str, str]:
    """Headers Plivo sends for ``path`` (query included) under ``public_url``."""
    url = public_url.rstrip("/") + path
    return plivo_signature_headers(method, url, TEST_AUTH_TOKEN, data)


def ws_path(xml: str) -> str:
    """Path + query of the <Stream> URL (what Plivo opens on this server)."""
    url = stream_url_from_xml(xml)
    return url[url.index("/ws") :]


def ws_signed(xml: str) -> dict[str, str]:
    """Headers Plivo sends when it opens the <Stream> URL of an answer response."""
    return stream_signature_headers(stream_url_from_xml(xml), TEST_AUTH_TOKEN)


@pytest.fixture
def captured_events():
    """Collect loguru ``extra`` dicts that carry a structured ``event`` key."""
    events: list[dict[str, Any]] = []

    def sink(message):
        extra = message.record["extra"]
        if "event" in extra:
            events.append(dict(extra))

    sink_id = logger.add(sink, level="DEBUG")
    yield events
    logger.remove(sink_id)


@pytest.fixture
def captured_messages():
    """Collect every log message text."""
    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(m.record["message"]), level="DEBUG")
    yield messages
    logger.remove(sink_id)


def make_agent(
    stream_id: str = STREAM_ID,
    started: bool = True,
    hangup_callback: Any = None,
    direction: str = "inbound",
    **kwargs: Any,
):
    """Build an agent wired to fake Plivo + GPT-Live sockets."""
    import importlib

    agent_mod = importlib.import_module(f"{direction}.agent")
    plivo_ws = kwargs.pop("plivo_ws", None) or FakePlivoWS()
    live_ws = kwargs.pop("live_ws", None) or FakeLiveWS()
    agent = agent_mod.GPTLiveVoiceAgent(
        websocket=plivo_ws,
        call_id=CALL_ID,
        from_number=kwargs.pop("from_number", "+15551234567"),
        to_number=kwargs.pop("to_number", "+16572338892"),
        stream_id=stream_id,
        hangup_callback=hangup_callback,
        **kwargs,
    )
    agent._live_ws = live_ws
    agent._running = True
    if started:
        agent._session_started.set()
    return agent, plivo_ws, live_ws


class HangupRecorder:
    """Async hangup callback that counts invocations."""

    def __init__(self, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    async def __call__(self) -> None:
        self.calls += 1
        if self.fail:
            raise RuntimeError("REST hangup failed")


def of_event(events: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [e for e in events if e["event"] == name]


async def run_send_loop_until(agent, predicate, timeout: float = 2.0) -> None:
    """Run _send_to_plivo until predicate() is true, then stop it."""
    task = asyncio.create_task(agent._send_to_plivo())
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    agent._running = False
    await asyncio.wait_for(task, timeout=1.0)


# =============================================================================
# UNIT TESTS - Audio conversion
# =============================================================================


class TestUnitAudioConversion:
    """μ-law 8kHz end to end: the wrappers are pass-throughs; the codec is test-only."""

    def test_helper_ulaw_roundtrip(self):
        from tests.helpers import pcm_to_ulaw, ulaw_to_pcm

        pcm = struct.pack("<6h", 0, 1000, -1000, 8000, -8000, 30000)
        decoded = struct.unpack("<6h", ulaw_to_pcm(pcm_to_ulaw(pcm)))
        for original, got in zip(struct.unpack("<6h", pcm), decoded, strict=True):
            assert abs(original - got) <= max(abs(original) * 0.05, 16)

    def test_silence_detection(self):
        from inbound.agent import _has_speech

        assert not _has_speech(b"\xff" * 160)  # μ-law digital silence
        assert not _has_speech(b"\x7f" * 160)
        assert rms_of_ulaw(b"\xff" * 160) < 10
        tone = tone_ulaw(0.02)
        assert _has_speech(tone)
        assert rms_of_ulaw(tone) > 500

    def test_plivo_to_openai_passthrough(self):
        audio = bytes(range(256))
        assert plivo_to_openai(audio) is audio

    def test_openai_to_plivo_passthrough(self):
        audio = bytes(range(256))
        assert openai_to_plivo(audio) is audio


# =============================================================================
# UNIT TESTS - Phone normalization
# =============================================================================


class TestUnitPhoneNormalization:
    """Unit tests for utils.normalize_phone_number (E.164 digits, no '+')."""

    def test_normalize_e164_format(self):
        assert normalize_phone_number("+16572338892") == "16572338892"

    def test_normalize_with_spaces(self):
        assert normalize_phone_number("+1 657-233-8892") == "16572338892"

    def test_normalize_local_format(self):
        assert normalize_phone_number("(657) 233-8892", "US") == "16572338892"

    def test_normalize_empty(self):
        assert normalize_phone_number("") == ""

    def test_normalize_unparseable_keeps_digits(self):
        assert normalize_phone_number("abc") == ""


# =============================================================================
# UNIT TESTS - Settings builder
# =============================================================================


# =============================================================================
# UNIT TESTS - Function dispatch
# =============================================================================


# =============================================================================
# UNIT TESTS - session.start builder
# =============================================================================


class TestUnitSessionStart:
    """session.start carries the model, μ-law 8kHz, the prompt, greeting and delegation."""

    def test_envelope_and_audio_format(self):
        agent, _, _ = make_agent()
        start = agent._build_session_start()
        assert start["type"] == "session.start"
        session = start["session"]
        assert session["model"] == "gpt-live-1.1"
        # One format object for both directions: Plivo's own μ-law 8kHz
        assert session["audio"]["format"] == {"type": "audio/pcmu", "rate": 8000}
        assert session["audio"]["output"] == {"voice": "marin"}
        assert "input" not in session["audio"]
        json.dumps(start)  # serializable

    def test_instructions_are_prompt_plus_call_context(self):
        from inbound.agent import SYSTEM_PROMPT

        agent, _, _ = make_agent(from_number="+15550001111")
        instructions = agent._build_session_start()["session"]["instructions"]
        assert instructions.startswith(SYSTEM_PROMPT)
        assert "+15550001111" in instructions
        agent, _, _ = make_agent(from_number="")
        assert agent._build_session_start()["session"]["instructions"] == SYSTEM_PROMPT

    def test_greeting_is_a_developer_message_with_say_tags(self):
        agent, _, _ = make_agent(initial_message="Hello from the test.")
        (message,) = agent._build_session_start()["session"]["input"]
        assert message["type"] == "message" and message["role"] == "developer"
        (content,) = message["content"]
        assert content["type"] == "input_text"
        assert "<say>Hello from the test.</say>" in content["text"]

    def test_empty_greeting_sends_no_input(self):
        agent, _, _ = make_agent(initial_message="")
        assert "input" not in agent._build_session_start()["session"]

    def test_delegation_is_responses_mode_with_function_tools(self):
        from inbound.agent import FUNCTION_DEFINITIONS

        agent, _, _ = make_agent()
        delegation = agent._build_session_start()["session"]["delegation"]
        assert delegation["type"] == "responses"
        responses = delegation["responses"]
        assert responses["model"] == "gpt-6-luna"
        assert responses["tool_choice"] == "auto"
        assert responses["tools"] == FUNCTION_DEFINITIONS
        assert "reasoning" not in responses
        names = [tool["name"] for tool in responses["tools"]]
        assert names == [
            "check_order_status",
            "send_sms",
            "schedule_callback",
            "transfer_call",
            "end_call",
        ]
        assert all(tool["type"] == "function" for tool in responses["tools"])

    def test_env_values_passed_verbatim(self, monkeypatch):
        from inbound import agent as agent_mod

        monkeypatch.setattr(agent_mod, "OPENAI_LIVE_MODEL", "gpt-live-1")
        monkeypatch.setattr(agent_mod, "OPENAI_LIVE_VOICE", "cedar")
        monkeypatch.setattr(agent_mod, "OPENAI_DELEGATION_MODEL", "gpt-6-sol")
        monkeypatch.setattr(agent_mod, "OPENAI_DELEGATION_REASONING_EFFORT", "low")
        agent, _, _ = make_agent()
        session = agent._build_session_start()["session"]
        assert session["model"] == "gpt-live-1"
        assert session["audio"]["output"]["voice"] == "cedar"
        assert session["delegation"]["responses"]["model"] == "gpt-6-sol"
        assert session["delegation"]["responses"]["reasoning"] == {"effort": "low"}

    def test_outbound_greeting_and_prompt(self):
        from outbound.agent import DEFAULT_OUTBOUND_GREETING, SYSTEM_PROMPT

        agent, _ = make_outbound_agent()
        session = agent._build_session_start()["session"]
        assert (
            f"<say>{DEFAULT_OUTBOUND_GREETING}</say>" in session["input"][0]["content"][0]["text"]
        )
        assert session["instructions"].startswith(SYSTEM_PROMPT)
        agent, _ = make_outbound_agent(initial_message="Hi, it's Alex.")
        text = agent._build_session_start()["session"]["input"][0]["content"][0]["text"]
        assert "<say>Hi, it's Alex.</say>" in text


# =============================================================================
# UNIT TESTS - Function dispatch
# =============================================================================


def delegated(event: dict[str, Any]) -> dict[str, Any]:
    """Wrap a Responses event the way GPT-Live forwards the backend model's events."""
    return {"type": "response.event", "delegation_id": "dlg_1", "event": event}


def function_call(name: str, arguments: Any, call_id: str = "call_1") -> dict[str, Any]:
    args = arguments if isinstance(arguments, str) else json.dumps(arguments)
    item = {"type": "function_call", "call_id": call_id, "name": name, "arguments": args}
    return delegated({"type": "response.output_item.done", "item": item})


COMPLETED = delegated({"type": "response.completed", "response": {"id": "resp_1", "output": []}})


class TestUnitFunctionDispatch:
    async def test_check_order_status(self):
        agent, _, _ = make_agent()
        result = await agent._handle_function_call("check_order_status", '{"order_number": "TF-1"}')
        assert result["status"] in ("shipped", "processing", "delivered")
        assert result["order_number"] == "TF-1"

    async def test_check_order_status_needs_identifier(self):
        agent, _, _ = make_agent()
        assert (await agent._handle_function_call("check_order_status", "{}"))["status"] == "error"

    async def test_send_sms_schedule_callback_transfer(self):
        agent, _, _ = make_agent()
        sms = await agent._handle_function_call(
            "send_sms", {"phone_number": "+15551234567", "message": "hi"}
        )
        assert sms["status"] == "sent"
        callback = await agent._handle_function_call(
            "schedule_callback",
            {"phone_number": "+15551234567", "reason": "billing", "department": "billing"},
        )
        assert callback["status"] == "scheduled"
        transfer = await agent._handle_function_call("transfer_call", {"department": "sales"})
        assert transfer["status"] == "transferring"

    async def test_unknown_function_returns_error(self):
        agent, _, _ = make_agent()
        assert "error" in await agent._handle_function_call("nope", "{}")

    @pytest.mark.parametrize("arguments", ["", "not json", "[1, 2]", None])
    async def test_bad_arguments_tolerated(self, arguments):
        agent, _, _ = make_agent()
        assert (await agent._handle_function_call("transfer_call", arguments))["status"]

    async def test_result_returned_after_response_completes(self):
        """Results are held until the backend response completes, then response.create."""
        agent, _, live_ws = make_agent()
        await agent._handle_openai_event(function_call("send_sms", {"phone_number": "1"}))
        assert live_ws.sent == []  # held: every call of the response must be answered first
        await agent._handle_openai_event(COMPLETED)
        item_create, response_create = live_ws.sent_json()
        assert item_create["type"] == "response.item.create"
        assert item_create["item"]["type"] == "function_call_output"
        assert item_create["item"]["call_id"] == "call_1"
        assert json.loads(item_create["item"]["output"])["status"] == "sent"
        assert response_create == {"type": "response.create"}

    async def test_parallel_calls_answered_together(self):
        agent, _, live_ws = make_agent()
        await agent._handle_openai_event(function_call("transfer_call", {}, "call_a"))
        await agent._handle_openai_event(function_call("send_sms", {"phone_number": "1"}, "call_b"))
        await agent._handle_openai_event(COMPLETED)
        assert live_ws.sent_types() == [
            "response.item.create",
            "response.item.create",
            "response.create",
        ]
        assert [m["item"]["call_id"] for m in live_ws.sent_json()[:2]] == ["call_a", "call_b"]

    async def test_completion_without_calls_sends_nothing(self):
        agent, _, live_ws = make_agent()
        await agent._handle_openai_event(COMPLETED)
        assert live_ws.sent == []

    async def test_fallback_flush_without_completion_event(self, monkeypatch):
        from inbound import agent as agent_mod

        monkeypatch.setattr(agent_mod, "TOOL_FLUSH_FALLBACK_S", 0.05)
        agent, _, live_ws = make_agent()
        await agent._handle_openai_event(function_call("transfer_call", {}))
        await run_send_loop_until(agent, lambda: len(live_ws.sent) == 2)
        assert live_ws.sent_types() == ["response.item.create", "response.create"]

    async def test_function_timeout_returns_error(self, monkeypatch):
        from inbound import agent as agent_mod

        async def slow(*_args, **_kwargs):
            await asyncio.sleep(1)

        monkeypatch.setattr(agent_mod, "TOOL_TIMEOUT_S", 0.05)
        monkeypatch.setattr(agent_mod, "transfer_call", slow)
        agent, _, live_ws = make_agent()
        await agent._handle_openai_event(function_call("transfer_call", {}))
        await agent._handle_openai_event(COMPLETED)
        output = json.loads(live_ws.sent_json()[0]["item"]["output"])
        assert output == {"error": "function timed out"}

    async def test_end_call_sets_hangup_pending_without_stopping(self):
        agent, _, _ = make_agent()
        result = await agent._handle_function_call("end_call", '{"reason": "done"}')
        assert result["status"] == "call_ending"
        assert agent._hangup_pending and agent._running


# =============================================================================
# UNIT TESTS - GPT-Live event handling (fake Plivo + GPT-Live sockets)
# =============================================================================

STARTED = {"type": "session.started", "session": {"id": "live_1", "status": "active"}}


def audio_delta(audio: bytes) -> dict[str, Any]:
    return {"type": "session.output_audio.delta", "delta": base64.b64encode(audio).decode()}


def transcript(role: str, delta: str, start_ms: int = 0, end_ms: int = 0) -> dict[str, Any]:
    kind = "input" if role == "user" else "output"
    return {
        "type": f"session.{kind}_transcript.delta",
        "delta": delta,
        "start_ms": start_ms,
        "end_ms": end_ms,
    }


def media(audio: bytes) -> dict[str, Any]:
    return {"event": "media", "media": {"payload": base64.b64encode(audio).decode()}}


class TestUnitLiveEventHandling:
    # -- handshake --

    async def test_handshake_sends_session_start_first(self):
        agent, _, live_ws = make_agent(started=False, live_ws=FakeLiveWS([STARTED]))
        await agent._handshake(live_ws)
        assert live_ws.sent_types() == ["session.start"]
        assert agent._session_started.is_set()
        assert agent._session_id == "live_1"

    async def test_handshake_error_raises(self):
        error = {"type": "error", "error": {"code": "invalid_model", "message": "no such model"}}
        agent, _, live_ws = make_agent(started=False, live_ws=FakeLiveWS([error]))
        with pytest.raises(RuntimeError, match="invalid_model: no such model"):
            await agent._handshake(live_ws)

    async def test_handshake_timeout_raises(self, monkeypatch):
        from inbound import agent as agent_mod

        monkeypatch.setattr(agent_mod, "SESSION_START_TIMEOUT_S", 0.05)
        agent, _, live_ws = make_agent(started=False)
        with pytest.raises(RuntimeError, match=r"session\.started not received"):
            await agent._handshake(live_ws)

    async def test_media_buffered_until_started_then_flushed_in_order(self):
        agent, plivo_ws, live_ws = make_agent(started=False)
        plivo_ws.push(media(b"\x01" * 160))
        plivo_ws.push(media(b"\x02" * 160))
        rx = asyncio.create_task(agent._receive_from_plivo())
        await asyncio.sleep(0.05)
        assert live_ws.sent == []  # nothing may precede session.started
        live_ws.feed(STARTED)
        await agent._handshake(live_ws)
        plivo_ws.push(media(b"\x03" * 160))
        plivo_ws.push({"event": "stop"})
        await asyncio.wait_for(rx, 1)
        start, buffered, live = live_ws.sent_json()
        assert start["type"] == "session.start"
        assert buffered["type"] == live["type"] == "session.input_audio.append"
        assert base64.b64decode(buffered["audio"]) == b"\x01" * 160 + b"\x02" * 160
        assert base64.b64decode(live["audio"]) == b"\x03" * 160

    async def test_pre_start_buffer_is_capped(self):
        from inbound.agent import PRE_START_BUFFER_MAX

        agent, plivo_ws, _ = make_agent(started=False)
        for _ in range(PRE_START_BUFFER_MAX // 160 + 10):
            plivo_ws.push(media(b"\x05" * 160))
        plivo_ws.push({"event": "stop"})
        await agent._receive_from_plivo()
        assert len(agent._pre_start_audio) == PRE_START_BUFFER_MAX

    async def test_bad_plivo_frames_ignored(self):
        agent, plivo_ws, live_ws = make_agent()
        plivo_ws.incoming.put_nowait("not json")
        plivo_ws.push({"event": "media", "media": {"payload": "!!!"}})
        plivo_ws.push({"event": "media", "media": {}})
        plivo_ws.push({"event": "stop"})
        await agent._receive_from_plivo()
        assert live_ws.sent == []

    # -- audio out --

    async def test_agent_audio_becomes_160_byte_play_audio(self):
        agent, plivo_ws, _ = make_agent()
        audio = tone_ulaw(0.05)  # 400 bytes -> 2 full chunks + 80 held back
        await agent._handle_openai_event(audio_delta(audio))
        await run_send_loop_until(agent, lambda: len(plivo_ws.events("playAudio")) == 2)
        chunks = plivo_ws.events("playAudio")
        assert len(chunks) == 2
        for chunk in chunks:
            assert chunk["media"]["contentType"] == "audio/x-mulaw"
            assert chunk["media"]["sampleRate"] == 8000
            assert len(base64.b64decode(chunk["media"]["payload"])) == 160
        sent = b"".join(base64.b64decode(c["media"]["payload"]) for c in chunks)
        assert sent == audio[:320]
        assert bytes(agent._tx_buffer) == audio[320:]

    async def test_bad_audio_delta_ignored(self):
        agent, _, _ = make_agent()
        await agent._handle_openai_event({"type": "session.output_audio.delta", "delta": "!!!"})
        await agent._handle_openai_event({"type": "session.output_audio.delta", "delta": ""})
        assert agent._send_queue.empty()

    # -- turns --

    async def test_transcripts_become_turn_events(self, captured_events):
        agent, _, _ = make_agent()
        for evt in (
            transcript("agent", "Hi, this is "),
            transcript("agent", "Alex."),
            transcript("user", "What plans ", 4000, 4600),
            transcript("user", "do you offer?", 4600, 5200),
            transcript("agent", "We have three.", 5900, 6800),
            transcript("user", "Thanks.", 9000, 9400),
        ):
            await agent._handle_openai_event(evt)
        agent._finish_turn()  # session end closes the open turn

        turns = of_event(captured_events, "turn_complete")
        assert [(t["turn"], t["user_text"], t["agent_text"]) for t in turns] == [
            (1, "", "Hi, this is Alex."),
            (2, "What plans do you offer?", "We have three."),
            (3, "Thanks.", ""),
        ]
        assert turns[1]["response_gap_ms"] == 700  # 5900 - 5200 on GPT-Live's timeline
        assert not any(t["barge_in"] for t in turns)
        user_texts = of_event(captured_events, "user_text")
        # A caller turn the agent never answered is still shown, when the turn closes
        assert [(e["turn"], e["text"]) for e in user_texts] == [
            (2, "What plans do you offer?"),
            (3, "Thanks."),
        ]
        agent_texts = of_event(captured_events, "agent_text")
        assert [(e["turn"], e["text"]) for e in agent_texts] == [
            (1, "Hi, this is Alex."),
            (2, "We have three."),
        ]

    async def test_agent_text_shown_when_transcript_goes_quiet(self, captured_events):
        """agent_text does not wait for turn_complete (the caller's next words)."""
        agent, _, _ = make_agent()
        await agent._handle_openai_event(transcript("user", "What plans?", 1000, 1500))
        await agent._handle_openai_event(transcript("agent", "We have three.", 2000, 2800))
        from inbound.agent import AGENT_TEXT_QUIET_S

        now = time.monotonic()
        await agent._housekeeping(now)  # a fragment just arrived: not quiet yet
        assert of_event(captured_events, "agent_text") == []
        (user_text,) = of_event(captured_events, "user_text")  # shown as the reply starts
        assert (user_text["turn"], user_text["text"]) == (1, "What plans?")

        await agent._housekeeping(now + AGENT_TEXT_QUIET_S + 0.1)
        (shown,) = of_event(captured_events, "agent_text")
        assert (shown["turn"], shown["text"]) == (1, "We have three.")
        assert of_event(captured_events, "turn_complete") == []  # the turn is still open

        # The agent resumes in the same turn: only the new text is shown, nothing twice
        await agent._handle_openai_event(transcript("agent", " Want details?", 4000, 4600))
        agent._finish_turn()
        texts = [e["text"] for e in of_event(captured_events, "agent_text")]
        assert texts == ["We have three.", "Want details?"]
        (turn,) = of_event(captured_events, "turn_complete")
        assert turn["agent_text"] == "We have three. Want details?"
        assert len(of_event(captured_events, "user_text")) == 1

    async def test_text_events_are_single_log_lines(self, captured_events):
        """user_text / agent_text ride on the console line: one record each, not two."""
        agent, _, _ = make_agent()
        await agent._handle_openai_event(transcript("user", "Hello?"))
        await agent._handle_openai_event(transcript("agent", "Hi there."))
        agent._finish_turn()
        for name, stage in (("user_text", "turn"), ("agent_text", "llm")):
            (evt,) = of_event(captured_events, name)
            assert evt["stage"] == stage and "elapsed_s" in evt

    async def test_caller_speaking_first_is_turn_one(self, captured_events):
        agent, _, _ = make_agent(initial_message="")
        await agent._handle_openai_event(transcript("user", "Hello?"))
        await agent._handle_openai_event(transcript("agent", "Hi there."))
        agent._finish_turn()
        (turn,) = of_event(captured_events, "turn_complete")
        assert (turn["turn"], turn["user_text"], turn["agent_text"]) == (1, "Hello?", "Hi there.")

    # -- barge-in --

    async def test_barge_in_flushes_unplayed_backlog(self, captured_events):
        agent, plivo_ws, _ = make_agent()
        await agent._handle_openai_event(transcript("agent", "A long answer"))
        await agent._handle_openai_event(audio_delta(tone_ulaw(2.0)))  # delivered in a burst
        assert agent._backlog_ms() == 2000
        await agent._handle_openai_event(transcript("user", "Wait"))
        assert plivo_ws.events("clearAudio") == [{"event": "clearAudio", "streamId": STREAM_ID}]
        assert agent._send_queue.empty() and agent._backlog_ms() == 0
        assert agent._barge_in_count == 1
        (turn,) = of_event(captured_events, "turn_complete")
        assert turn["barge_in"] is True and turn["agent_text"] == "A long answer"

    async def test_barge_in_counts_audio_already_sent_to_plivo(self):
        """Audio sent faster than real time sits in Plivo's buffer: it is cleared too."""
        agent, plivo_ws, _ = make_agent()
        await agent._handle_openai_event(transcript("agent", "A long answer"))
        await agent._handle_openai_event(audio_delta(tone_ulaw(1.0)))
        await run_send_loop_until(agent, lambda: len(plivo_ws.events("playAudio")) == 50)
        agent._running = True
        assert agent._send_queue.empty() and agent._backlog_ms() > 800
        await agent._handle_openai_event(transcript("user", "Wait"))
        assert len(plivo_ws.events("clearAudio")) == 1

    async def test_no_flush_when_backlog_is_small(self):
        """Real-time delivery leaves nothing to discard: GPT-Live stops by itself."""
        agent, plivo_ws, _ = make_agent()
        await agent._handle_openai_event(transcript("agent", "Sure"))
        await agent._handle_openai_event(audio_delta(tone_ulaw(0.1)))
        await agent._handle_openai_event(transcript("user", "mm-hm"))
        assert plivo_ws.events("clearAudio") == []
        assert agent._barge_in_count == 0 and not agent._send_queue.empty()

    async def test_no_flush_while_caller_keeps_talking(self):
        agent, plivo_ws, _ = make_agent()
        await agent._handle_openai_event(transcript("user", "So I was "))
        await agent._handle_openai_event(audio_delta(tone_ulaw(2.0)))
        await agent._handle_openai_event(transcript("user", "wondering"))
        assert plivo_ws.events("clearAudio") == []

    # -- end_call / hangup --

    async def test_end_call_hangs_up_after_goodbye_checkpoint(self, monkeypatch):
        from inbound import agent as agent_mod

        monkeypatch.setattr(agent_mod, "END_CALL_QUIET_S", 0.1)
        hangup = HangupRecorder()
        plivo_ws = FakePlivoWS(auto_played_stream=True)
        agent, _, _ = make_agent(plivo_ws=plivo_ws, hangup_callback=hangup)
        await agent._handle_openai_event(function_call("end_call", {"reason": "done"}))
        await agent._handle_openai_event(audio_delta(tone_ulaw(0.2)))  # the goodbye
        rx = asyncio.create_task(agent._receive_from_plivo())
        await run_send_loop_until(agent, lambda: hangup.calls == 1)
        await asyncio.wait_for(rx, 1)
        assert hangup.calls == 1
        events = [m["event"] for m in plivo_ws.sent]
        assert events.index("checkpoint") > max(
            i for i, e in enumerate(events) if e == "playAudio"
        )  # the checkpoint follows the goodbye audio
        assert not agent._running

    async def test_end_call_waits_while_goodbye_is_still_coming(self, monkeypatch):
        from inbound import agent as agent_mod

        monkeypatch.setattr(agent_mod, "END_CALL_QUIET_S", 0.3)
        hangup = HangupRecorder()
        agent, plivo_ws, _ = make_agent(hangup_callback=hangup)
        await agent._handle_openai_event(function_call("end_call", {}))
        task = asyncio.create_task(agent._send_to_plivo())
        for _ in range(4):  # speech keeps arriving: never quiet for 0.3s
            await agent._handle_openai_event(audio_delta(tone_ulaw(0.1)))
            await asyncio.sleep(0.1)
        assert hangup.calls == 0 and plivo_ws.events("checkpoint") == []
        agent._running = False
        await asyncio.wait_for(task, 1)

    async def test_end_call_without_goodbye_after_it_stops_waiting(self, monkeypatch):
        from inbound import agent as agent_mod

        monkeypatch.setattr(agent_mod, "END_CALL_GOODBYE_WAIT_S", 0.1)
        hangup = HangupRecorder()
        agent, _, _ = make_agent(stream_id="", hangup_callback=hangup)
        await agent._handle_openai_event(function_call("end_call", {}))
        await run_send_loop_until(agent, lambda: hangup.calls == 1)
        assert hangup.calls == 1

    async def test_hangup_deadline_forces_hangup(self, monkeypatch):
        from inbound import agent as agent_mod

        monkeypatch.setattr(agent_mod, "END_CALL_GRACE_S", 0.1)
        hangup = HangupRecorder(fail=True)  # a failing REST hangup must not raise
        agent, _, _ = make_agent(hangup_callback=hangup)
        await agent._handle_openai_event(function_call("end_call", {}))
        await run_send_loop_until(agent, lambda: hangup.calls == 1)
        assert hangup.calls == 1 and not agent._running

    async def test_stale_played_stream_ignored(self):
        hangup = HangupRecorder()
        agent, _, _ = make_agent(hangup_callback=hangup)
        agent._hangup_pending = True
        await agent._on_played_stream("goodbye_99")
        assert hangup.calls == 0

    # -- session events --

    async def test_error_event_is_not_fatal(self):
        agent, _, _ = make_agent()
        error = {"type": "error", "error": {"code": "responses_validation_failed", "message": "x"}}
        assert await agent._handle_openai_event(error) is True
        assert agent._error_count == 1

    async def test_session_closed_ends_the_session(self):
        agent, _, _ = make_agent()
        closed = {"type": "session.closed", "reason": "expired", "usage": {"seconds": 42}}
        assert await agent._handle_openai_event(closed) is False
        assert agent._close_reason == "expired" and agent._usage == {"seconds": 42}

    async def test_usage_and_unknown_events_tolerated(self):
        agent, _, _ = make_agent()
        usage = {"type": "session.usage.updated", "usage": {"seconds": 7, "active_seconds": 7}}
        assert await agent._handle_openai_event(usage) is True
        assert agent._usage["seconds"] == 7
        assert await agent._handle_openai_event({"type": "session.something.new"}) is True
        assert await agent._handle_openai_event(delegated({"type": "response.created"})) is True

    # -- full session against the fake GPT-Live server --

    async def test_full_session_greets_and_closes(self, monkeypatch, captured_events):
        from inbound import agent as agent_mod

        async with FakeLiveServer() as fake:
            monkeypatch.setattr(agent_mod, "OPENAI_LIVE_URL", fake.url)
            monkeypatch.setattr(agent_mod, "OPENAI_API_KEY", "sk-test")
            plivo_ws = FakePlivoWS()
            agent = agent_mod.GPTLiveVoiceAgent(
                websocket=plivo_ws, call_id=CALL_ID, stream_id=STREAM_ID, initial_message="Hi."
            )
            run = asyncio.create_task(agent.run())
            deadline = time.monotonic() + 3
            while len(plivo_ws.events("playAudio")) < 10 and time.monotonic() < deadline:
                plivo_ws.push(media(b"\xff" * 160))
                await asyncio.sleep(0.02)
            plivo_ws.push({"event": "stop"})
            await asyncio.wait_for(run, 5)

        assert len(plivo_ws.events("playAudio")) >= 10
        assert fake.headers["authorization"] == "Bearer sk-test"
        assert fake.types() == ["session.start", "session.close"]
        assert fake.input_audio_bytes > 0
        (end,) = of_event(captured_events, "session_end")
        assert end["openai_session_id"] == "live_fake_123"
        assert end["openai_close_reason"] == "close_requested"
        assert end["turns"] == 1 and end["errors"] == 0
        assert {"barge_ins", "duration_s", "ttfs_avg_ms"} <= set(end)
        (turn,) = of_event(captured_events, "turn_complete")
        assert turn["agent_text"] == "Hi."


class TestUnitFullCallIdInLogs:
    """Every per-call log line carries the complete call ID, never an 8-char prefix."""

    @pytest.fixture(params=["inbound.agent", "outbound.agent"])
    def agent_mod(self, request, monkeypatch):
        import importlib

        module = importlib.import_module(request.param)
        monkeypatch.setattr(module, "LOG_LEVEL", "verbose")  # _logv logs too
        return module

    @pytest.fixture
    def records(self):
        captured: list[tuple[str, dict[str, Any]]] = []
        sink_id = logger.add(
            lambda m: captured.append((m.record["message"], dict(m.record["extra"]))),
            level="DEBUG",
        )
        yield captured
        logger.remove(sink_id)

    async def test_pipeline_logs_use_full_ids(self, agent_mod, records, monkeypatch):
        def refuse_connect(*_args, **_kwargs):
            raise OSError("no network in unit tests")

        monkeypatch.setattr(agent_mod.websockets, "connect", refuse_connect)
        agent = agent_mod.GPTLiveVoiceAgent(
            websocket=FakePlivoWS(), call_id=CALL_ID, parent_call_id="parent-uuid"
        )
        agent._live_ws = FakeLiveWS()
        await agent.run()  # session start line, connect error (_loge), session_end
        agent._log("test", "normal line")
        agent._logv("test", "verbose line")
        for evt in (
            {"type": "info", "code": "notice", "message": "slow"},
            {"type": "error", "error": {"code": "bad_event", "message": "busy"}},
        ):
            await agent._handle_openai_event(evt)

        per_call = [(msg, extra) for msg, extra in records if CALL_ID in msg]
        assert len(per_call) >= 7  # start, answered, error, session_end, log, logv, info, error
        for msg, extra in per_call:
            assert msg.startswith(f"[{CALL_ID}]"), msg
            assert extra.get("call_id") == "parent-uuid", (msg, extra)
            assert extra.get("leg_call_id") == CALL_ID, (msg, extra)
        events = {extra["event"] for _, extra in per_call if "event" in extra}
        assert {"call_answered", "session_end"} <= events
        assert not any(f"[{CALL_ID[:8]}]" in msg for msg, _ in records)


# =============================================================================
# UNIT TESTS - Outbound CallManager
# =============================================================================


class TestUnitOutboundGreeting:
    """The answer_url ``greeting`` is passed to the agent as is; the prompt is the file."""

    GREETING = "Hi, this is Alex from TechFlow. I'm reaching out because you requested a demo."

    def test_greeting_passed_through_verbatim(self):
        from outbound.agent import SYSTEM_PROMPT, GPTLiveVoiceAgent

        agent = GPTLiveVoiceAgent(
            websocket=FakePlivoWS(), call_id="c", initial_message=self.GREETING
        )
        session = agent._build_session_start()["session"]
        assert f"<say>{self.GREETING}</say>" in session["input"][0]["content"][0]["text"]
        assert session["instructions"] == SYSTEM_PROMPT

    def test_default_greeting_when_absent(self):
        from outbound.agent import DEFAULT_OUTBOUND_GREETING, GPTLiveVoiceAgent

        agent = GPTLiveVoiceAgent(websocket=FakePlivoWS(), call_id="c")
        text = agent._build_session_start()["session"]["input"][0]["content"][0]["text"]
        assert f"<say>{DEFAULT_OUTBOUND_GREETING}</say>" in text

    def test_prompt_file_has_no_placeholders(self):
        from outbound.agent import SYSTEM_PROMPT

        file_text = (Path(__file__).parent.parent / "outbound" / "system_prompt.md").read_text()
        assert file_text.strip() == SYSTEM_PROMPT
        assert "{{" not in SYSTEM_PROMPT and "}}" not in SYSTEM_PROMPT

    @pytest.mark.parametrize("greeting", ["", GREETING])
    async def test_run_agent_passes_greeting(self, monkeypatch, greeting):
        from outbound import agent as agent_mod

        seen: dict[str, Any] = {}

        async def fake_run(self):
            seen.update(prompt=self.system_prompt, greeting=self.initial_message)

        monkeypatch.setattr(agent_mod.GPTLiveVoiceAgent, "run", fake_run)
        await agent_mod.run_agent(websocket=FakePlivoWS(), call_id="c", greeting=greeting)
        assert seen["greeting"] == (greeting or agent_mod.DEFAULT_OUTBOUND_GREETING)
        assert seen["prompt"] == agent_mod.SYSTEM_PROMPT


class TestUnitSystemPromptSource:
    """system_prompt.md is the only prompt source; a SYSTEM_PROMPT env var is ignored."""

    @pytest.mark.parametrize(
        ("direction", "attribute"),
        [("inbound", "SYSTEM_PROMPT"), ("outbound", "SYSTEM_PROMPT")],
    )
    def test_env_var_does_not_override_file(self, monkeypatch, direction, attribute):
        import importlib.util
        import sys

        monkeypatch.setenv("SYSTEM_PROMPT", "You are a pirate.")
        path = Path(__file__).parent.parent / direction / "agent.py"
        spec = importlib.util.spec_from_file_location(f"{direction}_agent_env_prompt", path)
        mod = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, mod)  # dataclasses look it up
        spec.loader.exec_module(mod)
        expected = (path.parent / "system_prompt.md").read_text().strip()
        assert getattr(mod, attribute) == expected


class TestUnitCallContextLabels:
    """#1: the call context names each number by its role in the call's direction."""

    def test_inbound_caller_is_from_number(self):
        agent, _, _ = make_agent(from_number="+15550001111", to_number="+16572338892")
        context = agent._build_call_context()
        assert "- Caller's phone number: +15550001111" in context
        assert "+16572338892" not in context
        assert "use the caller's phone number for SMS or callbacks" in context

    def test_outbound_customer_is_to_number_and_ours_is_from_number(self):
        agent, _ = make_outbound_agent(from_number="+14155550100", to_number="+15550002222")
        context = agent._build_call_context()
        assert "- Customer's phone number (the person you called): +15550002222" in context
        assert "- Our business phone number (caller ID the customer sees): +14155550100" in context
        assert "Caller's phone number" not in context
        assert "use the customer's phone number for SMS or callbacks" in context
        assert "give our business phone number" in context
        # The same context reaches the voice model and the delegation model
        session = agent._build_session_start()["session"]
        assert session["instructions"].endswith(context)
        assert session["delegation"]["responses"]["instructions"].endswith(context)

    def test_outbound_without_numbers_has_no_context(self):
        agent, _ = make_outbound_agent(from_number="", to_number="")
        assert agent._build_call_context() == ""


# =============================================================================
# UNIT TESTS - Server routes (FastAPI TestClient, no network)
# =============================================================================


class TestUnitServerRoutes:
    """Offline tests for the inbound/outbound FastAPI routes."""

    def test_inbound_answer_returns_stream_xml(self, monkeypatch):
        from fastapi.testclient import TestClient

        from inbound import server

        monkeypatch.setattr(server, "PUBLIC_URL", "https://example.ngrok.app")
        client = TestClient(server.app)
        form = {"CallUUID": "uuid-1", "From": "+15551234567", "To": "+16572338892"}
        resp = client.post(
            "/answer",
            data=form,
            headers=signed("POST", "https://example.ngrok.app", "/answer", form),
        )
        assert resp.status_code == 200
        assert "application/xml" in resp.headers["content-type"]
        xml = resp.text
        assert "<Stream" in xml
        assert 'bidirectional="true"' in xml
        assert 'keepCallAlive="true"' in xml
        assert "audio/x-mulaw;rate=8000" in xml
        assert "wss://example.ngrok.app/ws?body=" in xml
        meta = stream_body(xml)
        assert meta["call_uuid"] == "uuid-1"
        assert meta["from"] == "+15551234567"

    def test_inbound_health(self):
        from fastapi.testclient import TestClient

        from inbound import server

        assert TestClient(server.app).get("/").json()["status"] == "ok"

    async def test_rest_hangup_skipped_without_credentials(self, monkeypatch):
        from inbound import server

        monkeypatch.setattr(server, "PLIVO_AUTH_ID", "")
        await server._hangup_call("uuid-1")  # must not raise

    async def test_rest_hangup_calls_plivo(self, monkeypatch):
        from inbound import server

        deleted: list[str] = []

        class FakeCalls:
            def delete(self, call_uuid):
                deleted.append(call_uuid)

        class FakeRestClient:
            def __init__(self, auth_id, auth_token):
                self.calls = FakeCalls()

        monkeypatch.setattr(server, "PLIVO_AUTH_ID", "id")
        monkeypatch.setattr(server, "PLIVO_AUTH_TOKEN", "token")
        monkeypatch.setattr(server.plivo, "RestClient", FakeRestClient)
        await server._hangup_call("uuid-42")
        assert deleted == ["uuid-42"]

    def test_outbound_answer_carries_call_details(self, monkeypatch):
        from fastapi.testclient import TestClient

        from outbound import server

        monkeypatch.setattr(server, "PUBLIC_URL", "https://example.ngrok.app/")
        client = TestClient(server.app)
        path = "/outbound/answer?greeting=Hi%2C%20this%20is%20Alex.%20You%20requested%20a%20demo."
        form = {
            "CallUUID": "a-leg-uuid",
            "From": "+14155550100",
            "To": "+15551234567",
            "SIP-H-Account": "acme",
        }
        resp = client.post(
            path, data=form, headers=signed("POST", "https://example.ngrok.app/", path, form)
        )
        assert "<Stream" in resp.text
        assert 'bidirectional="true"' in resp.text
        assert 'keepCallAlive="true"' in resp.text
        assert "audio/x-mulaw;rate=8000" in resp.text
        assert "wss://example.ngrok.app/ws?body=" in resp.text
        meta = stream_body(resp.text)
        assert meta == {
            "call_uuid": "a-leg-uuid",
            "from": "+14155550100",
            "to": "+15551234567",
            "parent_call_uuid": "",
            "sip_headers": {"SIP-H-Account": "acme"},
            "greeting": "Hi, this is Alex. You requested a demo.",
        }

    def test_outbound_answer_without_details(self, monkeypatch):
        from fastapi.testclient import TestClient

        from outbound import server

        monkeypatch.setattr(server, "PUBLIC_URL", "https://example.ngrok.app")
        path = "/outbound/answer?CallUUID=u1&To=%2B15551234567"
        resp = TestClient(server.app).get(
            path, headers=signed("GET", "https://example.ngrok.app", path)
        )
        meta = stream_body(resp.text)
        assert (meta["call_uuid"], meta["to"]) == ("u1", "+15551234567")
        assert meta["greeting"] == ""

    def test_outbound_hangup_webhook_logs(self, monkeypatch, captured_messages):
        from fastapi.testclient import TestClient

        from outbound import server

        monkeypatch.setattr(server, "PUBLIC_URL", "https://example.ngrok.app")
        form = {"CallUUID": "u1", "Duration": "12", "HangupCause": "NORMAL_CLEARING"}
        resp = TestClient(server.app).post(
            "/outbound/hangup",
            data=form,
            headers=signed("POST", "https://example.ngrok.app", "/outbound/hangup", form),
        )
        assert resp.text == "OK"
        assert any(
            "Outbound call ended: CallUUID=u1, Duration=12s, HangupCause=NORMAL_CLEARING" in m
            for m in captured_messages
        )

    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("post", "/outbound/call"),
            ("get", "/outbound/status/x"),
            ("post", "/outbound/hangup/x"),
            ("get", "/outbound/campaign/x"),
            ("get", "/hold"),
        ],
    )
    def test_outbound_campaign_routes_are_gone(self, method, path):
        from fastapi.testclient import TestClient

        from outbound import server

        assert getattr(TestClient(server.app), method)(path).status_code in (404, 405)

    def test_outbound_ready_message(self, monkeypatch):
        from outbound import server

        monkeypatch.setattr(server, "PUBLIC_URL", "https://t-9.trycloudflare.com/")
        monkeypatch.setattr(server, "PLIVO_PHONE_NUMBER", "+1 (415) 555-0100")
        monkeypatch.setattr(server, "PLIVO_AUTH_ID", "MAREALID")
        monkeypatch.setattr(server, "PLIVO_AUTH_TOKEN", "real-secret-token")
        message = server.ready_message()
        assert message.startswith("Ready! ")
        assert message.endswith("(Ctrl+C to stop)")
        assert '"https://api.plivo.com/v1/Account/$PLIVO_AUTH_ID/Call/"' in message
        assert '-u "$PLIVO_AUTH_ID:$PLIVO_AUTH_TOKEN"' in message
        assert "MAREALID" not in message and "real-secret-token" not in message
        assert '"from": "+14155550100"' in message
        assert '"to": "<E.164 number to call>"' in message
        from urllib.parse import quote

        assert (
            '"answer_url": "https://t-9.trycloudflare.com/outbound/answer'
            f'?greeting={quote(server.READY_EXAMPLE_GREETING)}"'
        ) in message
        assert "Optional answer_url query param (URL-encoded): greeting" in message
        assert '"hangup_url": "https://t-9.trycloudflare.com/outbound/hangup"' in message
        assert "Must be a valid url" not in message
        assert "Must be a valid url" in server.ready_message(tunnel=True)

    def test_outbound_ready_message_placeholders(self, monkeypatch):
        from outbound import server

        monkeypatch.setattr(server, "PUBLIC_URL", "")
        monkeypatch.setattr(server, "PLIVO_PHONE_NUMBER", "")
        message = server.ready_message()
        assert '"from": "<your Plivo number>"' in message
        assert '"answer_url": "<PUBLIC_URL>/outbound/answer?' in message


# =============================================================================
# UNIT TESTS - Webhook authentication (Plivo V3 signatures)
# =============================================================================

SERVER_MODULES = ["inbound.server", "outbound.server"]
PUBLIC = "https://agent.example.com"
FORM = {"CallUUID": "c-1", "From": "+15551230000", "To": "+15557654321"}
# Every Plivo webhook route: (module, method, path)
WEBHOOK_ROUTES = [
    ("inbound.server", "POST", "/answer"),
    ("inbound.server", "GET", "/answer?CallUUID=c-1&From=%2B15551230000"),
    ("inbound.server", "POST", "/hangup"),
    ("inbound.server", "POST", "/fallback"),
    ("inbound.server", "GET", "/hold"),
    ("inbound.server", "POST", "/hold"),
    ("outbound.server", "POST", "/outbound/answer?greeting=a%20demo"),
    ("outbound.server", "GET", "/outbound/answer?CallUUID=c-1&greeting=hi"),
    ("outbound.server", "POST", "/outbound/hangup"),
]


def _server(module: str, monkeypatch, public_url: str = PUBLIC):
    import importlib

    server = importlib.import_module(module)
    monkeypatch.setattr(server, "PUBLIC_URL", public_url)
    return server


def _answer_path(module: str) -> str:
    return "/answer" if module == "inbound.server" else "/outbound/answer?greeting=a%20demo"


class _RunAgentRecorder:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


class TestUnitWebhookAuth:
    """Plivo V3 signature checks on webhooks and on the /ws stream."""

    @pytest.mark.parametrize(("module", "method", "path"), WEBHOOK_ROUTES)
    def test_every_webhook_accepts_signed_and_rejects_unsigned(
        self, monkeypatch, module, method, path
    ):
        from fastapi.testclient import TestClient

        client = TestClient(_server(module, monkeypatch).app)
        data = FORM if method == "POST" else None
        ok = client.request(method, path, data=data, headers=signed(method, PUBLIC, path, data))
        assert ok.status_code == 200, ok.text
        assert client.request(method, path, data=data).status_code == 403

    @pytest.mark.parametrize("module", SERVER_MODULES)
    def test_valid_post_logs_verification(self, monkeypatch, captured_messages, module):
        from fastapi.testclient import TestClient

        path = _answer_path(module)
        client = TestClient(_server(module, monkeypatch).app)
        resp = client.post(path, data=FORM, headers=signed("POST", PUBLIC, path, FORM))
        assert resp.status_code == 200
        suffix = " + query string" if "?" in path else ""
        assert f"Plivo signature verified: POST {path.split('?')[0]}{suffix}" in captured_messages

    @pytest.mark.parametrize("drop", ["X-Plivo-Signature-V3", "X-Plivo-Signature-V3-Nonce", "both"])
    def test_missing_headers_rejected(self, monkeypatch, captured_messages, drop):
        from fastapi.testclient import TestClient

        client = TestClient(_server("inbound.server", monkeypatch).app)
        headers = signed("POST", PUBLIC, "/answer", FORM)
        for name in list(headers):
            if drop in (name, "both"):
                del headers[name]
        assert client.post("/answer", data=FORM, headers=headers).status_code == 403
        assert any("Rejected Plivo webhook POST /answer: missing" in m for m in captured_messages)

    def test_wrong_signature_rejected_and_not_logged(self, monkeypatch, captured_messages):
        from fastapi.testclient import TestClient

        client = TestClient(_server("inbound.server", monkeypatch).app)
        headers = plivo_signature_headers("POST", f"{PUBLIC}/answer", "some-other-token", FORM)
        resp = client.post("/answer", data=FORM, headers=headers)
        assert resp.status_code == 403
        rejected = [m for m in captured_messages if "Rejected Plivo webhook" in m]
        assert rejected and "signature mismatch" in rejected[0]
        joined = "\n".join(captured_messages)
        assert headers["X-Plivo-Signature-V3"] not in joined
        assert TEST_AUTH_TOKEN not in joined

    @pytest.mark.parametrize("module", SERVER_MODULES)
    def test_tampered_form_field_rejected(self, monkeypatch, module):
        from fastapi.testclient import TestClient

        path = _answer_path(module)
        client = TestClient(_server(module, monkeypatch).app)
        headers = signed("POST", PUBLIC, path, FORM)
        tampered = {**FORM, "To": "+19995550000"}
        assert client.post(path, data=tampered, headers=headers).status_code == 403
        extra = {**FORM, "Extra": "x"}
        assert client.post(path, data=extra, headers=headers).status_code == 403

    @pytest.mark.parametrize("method", ["POST", "GET"])
    def test_tampered_outbound_query_string_rejected(self, monkeypatch, method):
        """The answer_url greeting is covered by the signature."""
        from fastapi.testclient import TestClient

        client = TestClient(_server("outbound.server", monkeypatch).app)
        data = FORM if method == "POST" else None
        signed_path = "/outbound/answer?greeting=a%20demo&x=1"
        headers = signed(method, PUBLIC, signed_path, data)
        for sent in (
            "/outbound/answer?greeting=free%20money&x=1",
            "/outbound/answer?greeting=a%20demo&x=1&y=injected",
            "/outbound/answer?greeting=a%20demo",
        ):
            assert client.request(method, sent, data=data, headers=headers).status_code == 403
        ok = client.request(method, signed_path, data=data, headers=headers)
        assert ok.status_code == 200

    def test_post_with_query_base_string_rule(self):
        """SDK rule: URL + '?' + sorted decoded query + '.' + sorted form name/value pairs."""
        from plivo.utils.signature_v3 import construct_post_url

        base = construct_post_url(
            f"{PUBLIC}/outbound/answer?x=c&greeting=a%20demo", {"To": "1", "From": "2"}
        )
        assert base.decode() == f"{PUBLIC}/outbound/answer?greeting=a demo&x=c.From2To1"

    @pytest.mark.parametrize("module", SERVER_MODULES)
    def test_signature_checked_against_public_url_not_request_url(self, monkeypatch, module):
        """Behind a tunnel the server sees http://localhost (here http://testserver)."""
        from fastapi.testclient import TestClient

        path = _answer_path(module)
        client = TestClient(_server(module, monkeypatch).app)
        seen_by_server = plivo_signature_headers(
            "POST", f"http://testserver{path}", TEST_AUTH_TOKEN, FORM
        )
        assert client.post(path, data=FORM, headers=seen_by_server).status_code == 403
        as_plivo = signed("POST", PUBLIC, path, FORM)
        assert client.post(path, data=FORM, headers=as_plivo).status_code == 200

    @pytest.mark.parametrize("module", SERVER_MODULES)
    def test_public_request_url_reconstruction(self, monkeypatch, module):
        from starlette.requests import Request

        server = _server(module, monkeypatch, "https://agent.example.com/")
        scope = {
            "type": "http",
            "method": "POST",
            "scheme": "http",
            "server": ("127.0.0.1", 8000),
            "path": "/outbound/answer",
            "query_string": b"greeting=a%20demo&x=1",
            "headers": [(b"host", b"localhost:8000")],
        }
        assert server.public_request_url(Request(scope)) == (
            "https://agent.example.com/outbound/answer?greeting=a%20demo&x=1"
        )
        scope["query_string"] = b""
        assert server.public_request_url(Request(scope)) == (
            "https://agent.example.com/outbound/answer"
        )

    @pytest.mark.parametrize("module", SERVER_MODULES)
    def test_tunnel_public_url_set_at_runtime_is_used(self, monkeypatch, module):
        from fastapi.testclient import TestClient

        server = _server(module, monkeypatch, "https://old.example.com")
        monkeypatch.setattr(
            server, "start_quick_tunnel", lambda port: ("https://t-7.trycloudflare.com", None)
        )
        monkeypatch.setattr(server.atexit, "register", lambda *a: None)
        server._start_tunnel()
        path = _answer_path(module)
        client = TestClient(server.app)
        tunnel = signed("POST", "https://t-7.trycloudflare.com", path, FORM)
        resp = client.post(path, data=FORM, headers=tunnel)
        assert resp.status_code == 200
        assert stream_url_from_xml(resp.text).startswith("wss://t-7.trycloudflare.com/ws?")
        old = signed("POST", "https://old.example.com", path, FORM)
        assert client.post(path, data=FORM, headers=old).status_code == 403

    def test_no_public_url_rejects(self, monkeypatch, captured_messages):
        from fastapi.testclient import TestClient

        client = TestClient(_server("inbound.server", monkeypatch, "").app)
        headers = signed("POST", PUBLIC, "/answer", FORM)
        assert client.post("/answer", data=FORM, headers=headers).status_code == 403
        assert any("PUBLIC_URL not set" in m for m in captured_messages)

    # --- /ws ----------------------------------------------------------------------

    @pytest.mark.parametrize("module", SERVER_MODULES)
    def test_ws_with_issued_stream_url_runs_agent(self, monkeypatch, module):
        from fastapi.testclient import TestClient

        server = _server(module, monkeypatch)
        recorder = _RunAgentRecorder()
        monkeypatch.setattr(server, "run_agent", recorder)
        hung_up: list[str] = []

        async def fake_hangup(call_uuid: str) -> None:
            hung_up.append(call_uuid)

        monkeypatch.setattr(server, "_hangup_call", fake_hangup)
        client = TestClient(server.app)
        path = _answer_path(module)
        answer = client.post(path, data=FORM, headers=signed("POST", PUBLIC, path, FORM))
        with client.websocket_connect(ws_path(answer.text), headers=ws_signed(answer.text)) as ws:
            ws.send_text(json.dumps({"event": "start", "start": {"callId": "c-1"}}))
        assert len(recorder.calls) == 1
        assert recorder.calls[0]["from_number"] == FORM["From"]
        # keepCallAlive holds the call after the stream: the server ends it when the agent does
        assert hung_up == ["c-1"]

    @pytest.mark.parametrize("module", SERVER_MODULES)
    def test_ws_rejects_unsigned_and_wrongly_signed(self, monkeypatch, captured_messages, module):
        """No headers, a bad signature, or one made over the full wss URL: closed with 1008."""
        from fastapi import WebSocketDisconnect
        from fastapi.testclient import TestClient

        server = _server(module, monkeypatch)
        recorder = _RunAgentRecorder()
        monkeypatch.setattr(server, "run_agent", recorder)
        client = TestClient(server.app)
        path = _answer_path(module)
        answer = client.post(path, data=FORM, headers=signed("POST", PUBLIC, path, FORM))
        stream_url = stream_url_from_xml(answer.text)
        good = ws_signed(answer.text)
        attempts = [
            {},
            {**good, "X-Plivo-Signature-V3": "AAAA"},
            # signed over the https URL with its query string: not what Plivo signs
            plivo_signature_headers(
                "GET", stream_url.replace("wss://", "https://"), TEST_AUTH_TOKEN
            ),
        ]
        for headers in attempts:
            with (
                pytest.raises(WebSocketDisconnect) as exc,
                client.websocket_connect(ws_path(answer.text), headers=headers),
            ):
                pass
            assert exc.value.code == 1008
        assert not recorder.calls
        rejected = [m for m in captured_messages if "Rejected Plivo stream /ws" in m]
        assert len(rejected) == len(attempts)
        assert "missing" in rejected[0]
        assert all("signature mismatch" in m for m in rejected[1:])
        assert not any("AAAA" in m or good["X-Plivo-Signature-V3"] in m for m in rejected)

    @pytest.mark.parametrize("module", SERVER_MODULES)
    def test_ws_signed_url_is_http_host_and_path(self, monkeypatch, module):
        """The stream's signed URL: http scheme, PUBLIC_URL host, request path, no query."""
        from fastapi import WebSocket

        server = _server(module, monkeypatch, "https://agent.example.com")
        scope = {
            "type": "websocket",
            "path": "/ws",
            "query_string": b"body=abc%3D",
            "headers": [(b"host", b"localhost:8000")],
            "server": ("localhost", 8000),
            "scheme": "ws",
        }
        ws = WebSocket(scope, receive=None, send=None)
        assert server.public_request_url(ws) == "http://agent.example.com/ws"

    # --- Startup ------------------------------------------------------------------

    @pytest.mark.parametrize("module", SERVER_MODULES)
    def test_refuses_to_start_without_auth_token(self, monkeypatch, captured_messages, module):
        import sys

        server = _server(module, monkeypatch)
        monkeypatch.setattr(server, "PLIVO_AUTH_TOKEN", "")
        monkeypatch.setattr(sys, "argv", [module])
        started = []
        monkeypatch.setattr(server.uvicorn, "run", lambda *a, **k: started.append(True))
        with pytest.raises(SystemExit) as exc:
            server.main()
        assert exc.value.code == 1
        assert started == []
        assert any("PLIVO_AUTH_TOKEN is empty" in m for m in captured_messages)


# =============================================================================
# UNIT TESTS - --tunnel, Plivo auto-config, Ready line, ports
# =============================================================================


def make_outbound_agent(**kwargs: Any):
    """Build an outbound agent wired to fake Plivo + GPT-Live sockets."""
    kwargs.setdefault("from_number", "+14155550100")
    kwargs.setdefault("to_number", "+15551234567")
    agent, _plivo_ws, live_ws = make_agent(direction="outbound", **kwargs)
    return agent, live_ws


class _NoThread:
    def start(self) -> None:
        pass


class TestUnitQuickTunnel:
    """``--tunnel`` helpers (utils.start_quick_tunnel & friends) with a fake cloudflared."""

    SAMPLE_LINE = (
        "2026-09-23T10:00:00Z INF |  https://quiet-river-demo-42.trycloudflare.com"
        "                                  |"
    )

    @staticmethod
    def _fake_cloudflared(tmp_path, body: str):
        script = tmp_path / "cloudflared"
        script.write_text(f"#!/bin/sh\n{body}\n")
        script.chmod(0o755)
        return str(script)

    def test_parse_tunnel_url(self):
        from utils import parse_tunnel_url

        assert parse_tunnel_url(self.SAMPLE_LINE) == (
            "https://quiet-river-demo-42.trycloudflare.com"
        )
        assert parse_tunnel_url("INF Registered tunnel connection connIndex=0") == ""

    def test_missing_cloudflared_raises_with_install_link(self, monkeypatch):
        import utils

        monkeypatch.setattr(utils.shutil, "which", lambda _name: None)
        with pytest.raises(utils.TunnelError, match="cloudflared not found on PATH") as exc:
            utils.start_quick_tunnel(8000)
        assert utils.CLOUDFLARED_INSTALL_URL in str(exc.value)

    def test_start_returns_url_and_stop_terminates(self, monkeypatch, tmp_path):
        import utils

        fake = self._fake_cloudflared(
            tmp_path,
            'echo "starting" >&2\n'
            'echo "INF |  https://abc-def-1.trycloudflare.com  |" >&2\n'
            "exec sleep 30",
        )
        monkeypatch.setattr(utils.shutil, "which", lambda _name: fake)
        url, proc = utils.start_quick_tunnel(8123, timeout_s=5)
        try:
            assert url == "https://abc-def-1.trycloudflare.com"
            assert proc.poll() is None
            assert proc.args[-2:] == ["--url", "http://localhost:8123"]
        finally:
            utils.stop_tunnel(proc)
        assert proc.poll() is not None
        utils.stop_tunnel(proc)  # idempotent

    def test_no_url_times_out_and_cleans_up(self, monkeypatch, tmp_path):
        import utils

        fake = self._fake_cloudflared(tmp_path, "exec sleep 30")
        monkeypatch.setattr(utils.shutil, "which", lambda _name: fake)
        spawned = []
        real_popen = utils.subprocess.Popen

        def tracking_popen(*args, **kwargs):
            proc = real_popen(*args, **kwargs)
            spawned.append(proc)
            return proc

        monkeypatch.setattr(utils.subprocess, "Popen", tracking_popen)
        with pytest.raises(utils.TunnelError, match="did not report a public URL"):
            utils.start_quick_tunnel(8000, timeout_s=0.5)
        assert spawned and spawned[0].poll() is not None

    def test_process_exiting_early_raises(self, monkeypatch, tmp_path):
        import utils

        fake = self._fake_cloudflared(tmp_path, 'echo "ERR failed to request tunnel" >&2')
        monkeypatch.setattr(utils.shutil, "which", lambda _name: fake)
        with pytest.raises(utils.TunnelError):
            utils.start_quick_tunnel(8000, timeout_s=5)

    @pytest.mark.parametrize("module", ["inbound.server", "outbound.server"])
    def test_start_tunnel_sets_public_url(self, monkeypatch, module):
        import importlib

        server = importlib.import_module(module)
        monkeypatch.setattr(server, "PUBLIC_URL", "")
        monkeypatch.setattr(
            server, "start_quick_tunnel", lambda port: ("https://t-1.trycloudflare.com", None)
        )
        registered = []
        monkeypatch.setattr(server.atexit, "register", lambda *a: registered.append(a))
        server._start_tunnel()
        assert server.PUBLIC_URL == "https://t-1.trycloudflare.com"
        assert registered and registered[0][0] is server.stop_tunnel

    @pytest.mark.parametrize("module", ["inbound.server", "outbound.server"])
    def test_start_tunnel_failure_exits(self, monkeypatch, module):
        import importlib

        import utils

        server = importlib.import_module(module)

        def boom(port):
            raise utils.TunnelError("cloudflared not found on PATH")

        monkeypatch.setattr(server, "start_quick_tunnel", boom)
        with pytest.raises(SystemExit):
            server._start_tunnel()

    async def test_answer_uses_tunnel_url_for_stream(self, monkeypatch):
        """After --tunnel sets PUBLIC_URL, /answer streams to the tunnel's wss:// URL."""
        from inbound import server

        monkeypatch.setattr(server, "PUBLIC_URL", "https://t-2.trycloudflare.com")
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            form = {"CallUUID": "c-1", "From": "+15551230000", "To": "+15557654321"}
            resp = await client.post(
                "/answer",
                data=form,
                headers=signed("POST", "https://t-2.trycloudflare.com", "/answer", form),
            )
        assert resp.status_code == 200
        assert "wss://t-2.trycloudflare.com/ws?body=" in resp.text


class FakePlivoClient:
    """Minimal stand-in for plivo.RestClient covering applications/numbers."""

    def __init__(self, apps=None, reject_urls: int = 0):
        self.apps = list(apps or [])
        self.reject_urls = reject_urls  # first N app writes fail with "valid url"
        self.list_params: list = []
        self.updated: list = []
        self.created: list = []
        self.numbers_updated: list = []
        outer = self

        class _Apps:
            def list(self, **params):
                outer.list_params.append(params)
                prefix = params.get("app_name", "")
                return {"objects": [a for a in outer.apps if a["app_name"].startswith(prefix)]}

            def update(self, app_id, **params):
                outer._maybe_reject()
                outer.updated.append((app_id, params))

            def create(self, app_name, **params):
                outer._maybe_reject()
                outer.created.append((app_name, params))
                return {"app_id": "new-app"}

        class _Numbers:
            def update(self, number, app_id):
                outer.numbers_updated.append((number, app_id))

        self.applications, self.numbers = _Apps(), _Numbers()

    def _maybe_reject(self):
        if self.reject_urls > 0:
            self.reject_urls -= 1
            raise plivo.exceptions.ValidationError(
                "{'answer_url': ['Must be a valid url'], 'hangup_url': ['Must be a valid url']}"
            )


class TestUnitPlivoAutoConfig:
    """Plivo number auto-config: app-name prefix filter, --tunnel URL wait, shutdown."""

    @pytest.fixture
    def inbound_server(self, monkeypatch):
        from inbound import server

        monkeypatch.setattr(server, "PLIVO_AUTH_ID", "MAXXXX")
        monkeypatch.setattr(server, "PLIVO_AUTH_TOKEN", "token")
        monkeypatch.setattr(server, "PLIVO_PHONE_NUMBER", "+14155550123")
        monkeypatch.setattr(server, "PUBLIC_URL", "https://t-3.trycloudflare.com")
        monkeypatch.setattr(server, "TUNNEL_URL_RETRY_INTERVAL_S", 0.0)
        return server

    def _install(self, monkeypatch, server, fake):
        monkeypatch.setattr(server.plivo, "RestClient", lambda **_kw: fake)

    def test_lists_apps_by_prefix_and_updates_exact_match(self, monkeypatch, inbound_server):
        fake = FakePlivoClient(
            apps=[
                {"app_name": "GPT_Live_Native_Test", "app_id": "test-app"},
                {"app_name": "GPT_Live_Native", "app_id": "real-app"},
            ]
        )
        self._install(monkeypatch, inbound_server, fake)
        assert inbound_server.configure_plivo_webhooks()
        assert fake.list_params == [{"app_name": "GPT_Live_Native"}]
        assert [u[0] for u in fake.updated] == ["real-app"]
        assert fake.updated[0][1]["answer_url"] == "https://t-3.trycloudflare.com/answer"
        assert fake.numbers_updated == [("14155550123", "real-app")]

    def test_creates_app_when_only_prefix_matches_exist(self, monkeypatch, inbound_server):
        fake = FakePlivoClient(apps=[{"app_name": "GPT_Live_Native_Test", "app_id": "t"}])
        self._install(monkeypatch, inbound_server, fake)
        assert inbound_server.configure_plivo_webhooks()
        assert fake.updated == []
        assert [c[0] for c in fake.created] == ["GPT_Live_Native"]
        assert fake.numbers_updated == [("14155550123", "new-app")]

    def test_invalid_url_without_wait_fails_fast(self, monkeypatch, inbound_server):
        fake = FakePlivoClient(reject_urls=1)
        self._install(monkeypatch, inbound_server, fake)
        assert not inbound_server.configure_plivo_webhooks()
        assert fake.numbers_updated == []

    def test_tunnel_wait_retries_until_plivo_accepts_url(self, monkeypatch, inbound_server):
        fake = FakePlivoClient(reject_urls=3)
        self._install(monkeypatch, inbound_server, fake)
        assert inbound_server.configure_plivo_webhooks(wait_for_url_s=30)
        assert len(fake.list_params) == 4  # 3 rejections + 1 success
        assert fake.numbers_updated == [("14155550123", "new-app")]

    def test_tunnel_wait_gives_up_after_deadline(self, monkeypatch, inbound_server):
        fake = FakePlivoClient(reject_urls=10_000)
        self._install(monkeypatch, inbound_server, fake)
        monkeypatch.setattr(inbound_server, "TUNNEL_URL_RETRY_INTERVAL_S", 0.01)
        assert not inbound_server.configure_plivo_webhooks(wait_for_url_s=0.05)
        assert 1 < len(fake.list_params) < 50  # retried, then stopped at the deadline
        assert fake.numbers_updated == []

    def test_other_validation_errors_are_not_retried(self, monkeypatch, inbound_server):
        fake = FakePlivoClient()

        def bad_update(number, app_id):
            raise plivo.exceptions.ValidationError("number not found")

        fake.numbers.update = bad_update
        self._install(monkeypatch, inbound_server, fake)
        assert not inbound_server.configure_plivo_webhooks(wait_for_url_s=30)
        assert len(fake.list_params) == 1

    @pytest.mark.parametrize("module", ["inbound.server", "outbound.server"])
    async def test_lifespan_shutdown_stops_tunnel(self, monkeypatch, module):
        """Covers SIGTERM too: uvicorn runs lifespan shutdown before re-raising the signal."""
        import importlib

        server = importlib.import_module(module)
        stopped = []
        monkeypatch.setattr(server, "_tunnel_proc", "fake-proc")
        monkeypatch.setattr(server, "stop_tunnel", lambda proc: stopped.append(proc))
        async with server._lifespan(server.app):
            assert stopped == []
        assert stopped == ["fake-proc"]


@contextlib.contextmanager
def _listening_port():
    """A local port that accepts TCP connections (the kernel completes the handshake)."""
    import socket

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(8)
    try:
        yield sock.getsockname()[1]
    finally:
        sock.close()


def _closed_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _ready_lines(messages: list[str]) -> list[str]:
    return [m for m in messages if m.startswith("Ready!")]


INBOUND_READY = "Ready! Call +14155550123 to talk to the agent (Ctrl+C to stop)"


class TestUnitReadyLine:
    """Startup "Ready!" lines: logged once, only when the server accepts connections
    (and, inbound, the Plivo number is configured)."""

    def test_inbound_server_up_first_then_plivo(self, captured_messages):
        from inbound.server import ReadyLine

        ready = ReadyLine()
        ready.serving()
        assert _ready_lines(captured_messages) == []
        ready.plivo_configured("14155550123")
        assert _ready_lines(captured_messages) == [INBOUND_READY]

    def test_inbound_plivo_first_then_server_up(self, captured_messages):
        from inbound.server import ReadyLine

        ready = ReadyLine()
        ready.plivo_configured("14155550123")
        assert _ready_lines(captured_messages) == []
        ready.serving()
        assert _ready_lines(captured_messages) == [INBOUND_READY]

    def test_inbound_logged_exactly_once(self, captured_messages):
        import threading

        from inbound.server import ReadyLine

        ready = ReadyLine()
        threads = [
            threading.Thread(target=ready.serving if i % 2 else ready.plivo_configured, args=a)
            for i, a in enumerate([("14155550123",), (), ("14155550123",), ()] * 10)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        ready.serving()
        ready.plivo_configured("14155550123")
        assert _ready_lines(captured_messages) == [INBOUND_READY]

    def test_inbound_no_line_without_plivo(self, captured_messages):
        from inbound.server import ReadyLine

        ReadyLine().serving()
        assert _ready_lines(captured_messages) == []

    async def test_wait_until_serving(self):
        from inbound import server

        with _listening_port() as port:
            assert await server._wait_until_serving(port, timeout_s=2)
        assert not await server._wait_until_serving(_closed_port(), timeout_s=0.2)

    @staticmethod
    def _run_main(monkeypatch, server, configured: bool, tunnel: bool = False):
        """Run inbound main() with uvicorn replaced by the app's lifespan on a live port."""
        import sys

        monkeypatch.setattr(server, "_ready", server.ReadyLine())
        monkeypatch.setattr(server, "PLIVO_PHONE_NUMBER", "+1 415 555 0123")
        monkeypatch.setattr(server, "PUBLIC_URL", "https://example.invalid")
        monkeypatch.setattr(server, "configure_plivo_webhooks", lambda **_kw: configured)
        monkeypatch.setattr(sys, "argv", ["inbound.server", *(["--tunnel"] if tunnel else [])])
        monkeypatch.setattr(server, "_start_tunnel", lambda: None)

        def fake_uvicorn_run(app, **_kwargs):
            async def serve():
                with _listening_port() as port:
                    monkeypatch.setattr(server, "SERVER_PORT", port)
                    async with server._lifespan(app):
                        await asyncio.sleep(0.3)

            asyncio.run(serve())

        monkeypatch.setattr(server.uvicorn, "run", fake_uvicorn_run)
        server.main()

    def test_inbound_fixed_url_plivo_before_uvicorn(self, monkeypatch, captured_messages):
        from inbound import server

        self._run_main(monkeypatch, server, configured=True)
        assert _ready_lines(captured_messages) == [INBOUND_READY]
        # Plivo was configured before uvicorn started, yet Ready waited for the server
        assert captured_messages.index(INBOUND_READY) > captured_messages.index(
            "Configuring Plivo webhooks..."
        )

    def test_inbound_fixed_url_plivo_failure_no_line(self, monkeypatch, captured_messages):
        from inbound import server

        self._run_main(monkeypatch, server, configured=False)
        assert _ready_lines(captured_messages) == []
        assert "Plivo auto-configuration failed. Configure manually." in captured_messages

    def test_inbound_tunnel_plivo_after_server_up(self, monkeypatch, captured_messages):
        from inbound import server

        threads: list[Any] = []
        monkeypatch.setattr(
            server.threading, "Thread", lambda **kw: threads.append(kw) or _NoThread()
        )
        self._run_main(monkeypatch, server, configured=True, tunnel=True)
        assert _ready_lines(captured_messages) == []  # server up, Plivo not yet accepted
        (thread,) = threads
        thread["target"](*thread["args"])  # the background configure finishes later
        assert _ready_lines(captured_messages) == [INBOUND_READY]

    def test_inbound_tunnel_plivo_failure_no_line(self, monkeypatch, captured_messages):
        from inbound import server

        monkeypatch.setattr(server, "_ready", server.ReadyLine())
        monkeypatch.setattr(server, "configure_plivo_webhooks", lambda **_kw: False)
        server._ready.serving()
        server._configure_plivo_for_tunnel("14155550123")
        assert _ready_lines(captured_messages) == []
        assert any("Plivo did not accept" in m for m in captured_messages)

    async def test_outbound_logged_once_when_serving(self, monkeypatch, captured_messages):
        from outbound import server

        monkeypatch.setattr(server, "PUBLIC_URL", "https://example.invalid")
        with _listening_port() as port:
            monkeypatch.setattr(server, "SERVER_PORT", port)
            async with server._lifespan(server.app):
                await asyncio.sleep(0.3)
        ready = _ready_lines(captured_messages)
        assert ready == [server.ready_message()]
        assert ready[0].endswith("(Ctrl+C to stop)")

    async def test_outbound_no_line_when_not_serving(self, monkeypatch, captured_messages):
        from outbound import server

        monkeypatch.setattr(server, "SERVER_PORT", _closed_port())
        monkeypatch.setattr(server, "READY_PROBE_TIMEOUT_S", 0.2)
        await server._log_ready_when_serving()
        assert _ready_lines(captured_messages) == []
        assert any("no Ready line" in m for m in captured_messages)


class TestUnitServerPorts:
    """Inbound and outbound default to different ports so both can run at once.

    SERVER_PORT is read at import time, so each case imports the module in a fresh
    interpreter (no state leaks into other tests) with ``load_dotenv`` stubbed out so the
    local ``.env`` can't override the env vars under test.
    """

    @staticmethod
    def _port(module: str, **env_vars: str) -> int:
        import subprocess
        import sys

        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("SERVER_PORT", "OUTBOUND_SERVER_PORT", "OTEL_EXPORTER_OTLP_ENDPOINT")
        }
        env.update(env_vars)
        code = (
            "import dotenv; dotenv.load_dotenv = lambda *a, **k: False\n"
            f"import {module} as m; print(m.SERVER_PORT)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).parent.parent,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        return int(result.stdout.strip().splitlines()[-1])

    def test_inbound_default_port(self):
        assert self._port("inbound.server") == 8000

    def test_inbound_reads_server_port(self):
        assert self._port("inbound.server", SERVER_PORT="18123") == 18123

    def test_outbound_default_port(self):
        assert self._port("outbound.server") == 8001

    def test_outbound_reads_outbound_server_port(self):
        assert self._port("outbound.server", OUTBOUND_SERVER_PORT="18124") == 18124

    def test_outbound_ignores_server_port(self):
        assert self._port("outbound.server", SERVER_PORT="18125") == 8001


# =============================================================================
# UNIT TESTS - ngrok helper (subprocess / httpx mocked; nothing is started or killed)
# =============================================================================


class _FakeNgrokProc:
    """Stands in for the Popen handle of the ngrok agent start_ngrok starts."""

    def __init__(self, exits_on_terminate: bool = True) -> None:
        self.exits_on_terminate = exits_on_terminate
        self.returncode: int | None = None
        self.calls: list[str] = []

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.calls.append("terminate")
        if self.exits_on_terminate:
            self.returncode = -15

    def kill(self) -> None:
        self.calls.append("kill")
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        self.calls.append(f"wait({timeout})")
        if self.returncode is None:
            import subprocess

            raise subprocess.TimeoutExpired("ngrok", timeout or 0)
        return self.returncode


# Kill-by-name commands, split so a grep for them finds nothing in this repo
_KILL_BY_NAME = ("p" + "kill", "kill" + "all")


class TestUnitNgrokHelper:
    """start_ngrok/stop_ngrok never kill by name and only stop their own process."""

    @pytest.fixture
    def ngrok_env(self, monkeypatch):
        import subprocess

        from tests import helpers

        env: dict[str, Any] = {"run": [], "popen": [], "procs": []}

        def fake_run(cmd, *args, **kwargs):
            env["run"].append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, stdout="4242\n", stderr="")

        def fake_popen(cmd, *args, **kwargs):
            env["popen"].append(list(cmd))
            proc = _FakeNgrokProc()
            env["procs"].append(proc)
            return proc

        monkeypatch.setattr(helpers.subprocess, "run", fake_run)
        monkeypatch.setattr(helpers.subprocess, "Popen", fake_popen)
        monkeypatch.setattr(helpers.time, "sleep", lambda _s: None)
        return env

    @staticmethod
    def _tunnels(port: int) -> httpx.Response:
        tunnel = {
            "proto": "https",
            "public_url": "https://ours.ngrok.app",
            "config": {"addr": f"http://localhost:{port}"},
        }
        return httpx.Response(200, json={"tunnels": [tunnel]})

    def test_skips_when_ngrok_api_responds(self, monkeypatch, ngrok_env):
        from tests import helpers

        monkeypatch.setattr(helpers.httpx, "get", lambda *_a, **_k: self._tunnels(9999))
        with pytest.raises(pytest.skip.Exception, match=r"already running \(pid 4242\)"):
            helpers.start_ngrok(18000)
        assert ngrok_env["popen"] == [], "must not start a second ngrok agent"
        assert ngrok_env["run"] == [["pgrep", "-f", "ngrok"]]

    def test_starts_own_agent_and_never_kills_by_name(self, monkeypatch, ngrok_env):
        from tests import helpers

        responses = iter([httpx.ConnectError("refused"), self._tunnels(18000)])

        def fake_get(*_a, **_k):
            item = next(responses)
            if isinstance(item, Exception):
                raise item
            return item

        monkeypatch.setattr(helpers.httpx, "get", fake_get)
        proc, url = helpers.start_ngrok(18000)
        assert url == "https://ours.ngrok.app"
        assert ngrok_env["popen"] == [[helpers.NGROK_BIN, "http", "18000"]]
        assert proc is ngrok_env["procs"][0] and proc.calls == []
        for cmd in ngrok_env["run"]:
            assert cmd[0] not in _KILL_BY_NAME, cmd

    def test_failure_path_stops_only_its_own_process(self, monkeypatch, ngrok_env):
        from tests import helpers

        def refused(*_a, **_k):
            raise httpx.ConnectError("refused")

        monkeypatch.setattr(helpers.httpx, "get", refused)
        with pytest.raises(pytest.skip.Exception, match="did not start"):
            helpers.start_ngrok(18000)
        assert ngrok_env["procs"][0].calls[0] == "terminate"
        assert all(cmd[0] not in _KILL_BY_NAME for cmd in ngrok_env["run"])

    def test_teardown_terminates_only_its_own_process(self, ngrok_env):
        from tests import helpers

        proc = _FakeNgrokProc()
        helpers.stop_ngrok(proc)
        assert proc.calls == ["terminate", "wait(5)"]
        assert ngrok_env["run"] == [] and ngrok_env["popen"] == []

    def test_teardown_kills_own_process_if_terminate_times_out(self, ngrok_env):
        from tests import helpers

        proc = _FakeNgrokProc(exits_on_terminate=False)
        helpers.stop_ngrok(proc)
        assert proc.calls == ["terminate", "wait(5)", "kill", "wait(None)"]
        assert ngrok_env["run"] == []

    def test_teardown_of_exited_process_is_a_no_op(self, ngrok_env):
        from tests import helpers

        proc = _FakeNgrokProc()
        proc.returncode = 0
        helpers.stop_ngrok(proc)
        assert proc.calls == []

    def test_no_kill_by_name_anywhere_in_tests(self):
        tests_dir = Path(__file__).parent
        offenders = [
            f"{path.name}:{n}"
            for path in sorted(tests_dir.glob("*.py"))
            for n, line in enumerate(path.read_text().splitlines(), 1)
            if any(needle in line for needle in _KILL_BY_NAME)
        ]
        assert offenders == []


# =============================================================================
# LOCAL INTEGRATION TESTS (real server subprocess + local fake of the GPT-Live API)
# =============================================================================


LOCAL_LOG_PATH = server_log_path("integration_local_server")


class PlivoStream:
    """Plays Plivo's side of /ws against the local server: sends caller audio in real time."""

    def __init__(self, ws: Any) -> None:
        self.ws = ws
        self.stream_id = str(uuid.uuid4())
        self.call_id = str(uuid.uuid4())
        self.audio = bytearray()
        self.events: list[str] = []
        self.closed = asyncio.Event()
        self._caller_audio = bytearray()
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        start = {"event": "start", "start": {"callId": self.call_id, "streamId": self.stream_id}}
        await self.ws.send(json.dumps(start))
        self._tasks = [asyncio.create_task(self._send()), asyncio.create_task(self._recv())]

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    def say(self, ulaw_audio: bytes) -> None:
        """Queue caller audio; silence is sent whenever nothing is queued."""
        self._caller_audio.extend(ulaw_audio)

    async def _send(self) -> None:
        with contextlib.suppress(Exception):
            while not self.closed.is_set():
                chunk = bytes(self._caller_audio[:160]) or b"\xff" * 160
                del self._caller_audio[:160]
                await self.ws.send(json.dumps(media(chunk.ljust(160, b"\xff"))))
                await asyncio.sleep(0.02)

    async def _recv(self) -> None:
        try:
            async for raw in self.ws:
                msg = json.loads(raw)
                self.events.append(msg.get("event"))
                if msg.get("event") == "playAudio":
                    assert msg["media"]["contentType"] == "audio/x-mulaw"
                    assert msg["media"]["sampleRate"] == 8000
                    self.audio.extend(base64.b64decode(msg["media"]["payload"]))
                elif msg.get("event") == "checkpoint":
                    played = {"event": "playedStream", "name": msg["name"]}
                    await self.ws.send(json.dumps({**played, "streamId": self.stream_id}))
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.closed.set()

    async def wait_for_audio(self, seconds: float, timeout: float = 10.0) -> None:
        """Wait until at least ``seconds`` of agent audio has arrived since the last call."""
        start, deadline = len(self.audio), time.monotonic() + timeout
        while len(self.audio) - start < seconds * 8000 and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        assert len(self.audio) - start >= seconds * 8000, "agent audio did not arrive in time"


class TestLocalIntegration:
    """The inbound server as a subprocess, with GPT-Live replaced by tests/fake_live.py."""

    REPLIES: ClassVar[list[Any]] = [
        "We offer three plans.",
        {
            "function": "check_order_status",
            "arguments": {"order_number": "TF-1"},
            "then": "Your order has shipped.",
        },
        {"function": "end_call", "arguments": {"reason": "done"}, "then": "Goodbye and thank you."},
    ]

    @pytest.fixture(scope="class")
    @classmethod
    def fake_live(cls):
        thread = FakeLiveThread(cls.REPLIES)
        thread.start()
        yield thread.server
        thread.stop()

    @pytest.fixture(scope="class")
    @classmethod
    def server_process(cls, fake_live):
        """Start the inbound server as a subprocess (SIGTERM -> wait(5) -> SIGKILL)."""
        # No PLIVO_AUTH_ID: the local server must never touch the Plivo REST API. Webhook
        # auth is keyed with a test token; requests are signed like Plivo's.
        proc = start_server(
            "inbound.server",
            TEST_PORT,
            LOCAL_LOG_PATH,
            {
                "OPENAI_LIVE_URL": fake_live.url,
                "OPENAI_API_KEY": "sk-local-test",
                "PLIVO_AUTH_ID": "",
                "PLIVO_AUTH_TOKEN": TEST_AUTH_TOKEN,
                "PLIVO_PHONE_NUMBER": "",
                "PUBLIC_URL": LOCAL_HTTP_URL,
            },
        )
        yield proc
        stop_server(proc)

    @staticmethod
    async def _answer(call_uuid: str, signed_request: bool = True) -> httpx.Response:
        form = {"CallUUID": call_uuid, "From": "+15551234567", "To": "+16572338892"}
        headers = signed("POST", LOCAL_HTTP_URL, "/answer", form) if signed_request else {}
        async with httpx.AsyncClient() as client:
            return await client.post(f"{LOCAL_HTTP_URL}/answer", data=form, headers=headers)

    @contextlib.asynccontextmanager
    async def _call(self, call_uuid: str):
        """A Plivo-signed answer webhook, then its signed /ws stream."""
        stream_url = stream_url_from_xml((await self._answer(call_uuid)).text)
        headers = stream_signature_headers(stream_url, TEST_AUTH_TOKEN)
        async with websockets.connect(
            stream_url, additional_headers=headers, close_timeout=2
        ) as ws:
            stream = PlivoStream(ws)
            await stream.start()
            try:
                yield stream
            finally:
                await stream.stop()

    async def test_local_health_check(self, server_process):
        async with httpx.AsyncClient() as client:
            response = await client.get(LOCAL_HTTP_URL)
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    async def test_local_unsigned_stream_rejected(self, server_process):
        """An unsigned /ws connection is refused with HTTP 403."""
        stream_url = stream_url_from_xml((await self._answer("test-unsigned-ws")).text)
        with pytest.raises(websockets.exceptions.InvalidStatus) as exc:
            async with websockets.connect(stream_url, close_timeout=2):
                pass
        assert exc.value.response.status_code == 403

    async def test_local_unsigned_answer_rejected(self, server_process):
        assert (await self._answer("test-unsigned", signed_request=False)).status_code == 403

    async def test_local_answer_webhook(self, server_process):
        response = await self._answer("test123")
        assert response.status_code == 200
        assert "application/xml" in response.headers["content-type"]
        assert "<Stream" in response.text
        assert "bidirectional" in response.text
        assert "audio/x-mulaw" in response.text

    async def test_local_greeting_audio(self, server_process, fake_live):
        """A Plivo start event produces the greeting as 160-byte playAudio with speech energy."""
        async with self._call("test-greeting") as call:
            await call.wait_for_audio(1.0)
        assert len(call.audio) % 160 == 0
        rms = rms_of_ulaw(bytes(call.audio))
        assert rms > 500, f"Audio RMS {rms} too low - may be silence"
        start = next(m for m in reversed(fake_live.received) if m["type"] == "session.start")
        assert start["session"]["audio"]["format"] == {"type": "audio/pcmu", "rate": 8000}
        assert "<say>" in start["session"]["input"][0]["content"][0]["text"]
        assert "+15551234567" in start["session"]["instructions"]  # per-call context

    async def test_local_conversation_tools_and_hangup(self, server_process, fake_live):
        """Caller turns reach GPT-Live; a delegated function runs; end_call closes the stream."""
        async with self._call("test-conversation") as call:
            await call.wait_for_audio(1.0)  # greeting
            await asyncio.sleep(3.5)  # let the greeting finish (the fake plays 0.3s per word)
            for seconds in (1.0, 1.0, 1.0):  # reply, check_order_status, end_call
                call.say(tone_ulaw(0.6))
                await call.wait_for_audio(seconds)
                await asyncio.sleep(seconds + 0.5)
            await asyncio.wait_for(call.closed.wait(), timeout=10)

        assert "checkpoint" in call.events  # hangup waited for the goodbye to play
        outputs = [m["item"] for m in fake_live.received if m["type"] == "response.item.create"]
        assert [json.loads(o["output"]).get("status") for o in outputs[-2:]][1] == "call_ending"
        assert json.loads(outputs[-2]["output"])["order_number"] == "TF-1"
        assert fake_live.types().count("response.create") >= 2
        await asyncio.sleep(0.5)  # let the server flush its logs
        turns = [
            e
            for e in read_log_events(LOCAL_LOG_PATH, "turn_complete")
            if e["call_id"] == call.call_id
        ]
        assert [t["turn"] for t in turns] == [1, 2, 3, 4]
        assert turns[1]["agent_text"] == "We offer three plans."
        ends = [
            e
            for e in read_log_events(LOCAL_LOG_PATH, "session_end")
            if e["call_id"] == call.call_id
        ]
        assert ends and ends[0]["errors"] == 0


# =============================================================================
# GPT-LIVE API INTEGRATION (direct connection to the real API)
# =============================================================================


@pytest.mark.skipif(not OPENAI_API_KEY, reason="OPENAI_API_KEY not configured")
class TestGPTLiveIntegration:
    """Send this example's own session.start to the real API and check what comes back."""

    async def test_session_start_is_accepted_and_greeting_is_spoken(self):
        from inbound.agent import OPENAI_LIVE_URL

        agent, _, _ = make_agent(initial_message="Hello, this is a connection test.")
        audio = bytearray()
        said = ""
        silence = base64.b64encode(b"\xff" * 160).decode()
        async with websockets.connect(
            OPENAI_LIVE_URL, additional_headers={"Authorization": f"Bearer {OPENAI_API_KEY}"}
        ) as ws:
            await ws.send(json.dumps(agent._build_session_start()))
            started = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            assert started["type"] == "session.started", started
            assert started["session"]["id"]
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline and len(audio) < 16000:
                # GPT-Live expects a continuous input stream, silence included
                await ws.send(json.dumps({"type": "session.input_audio.append", "audio": silence}))
                try:
                    evt = json.loads(await asyncio.wait_for(ws.recv(), timeout=0.02))
                except (TimeoutError, asyncio.TimeoutError):
                    continue
                assert evt["type"] != "error", evt
                if evt["type"] == "session.output_audio.delta":
                    audio.extend(base64.b64decode(evt["delta"]))
                elif evt["type"] == "session.output_transcript.delta":
                    said += evt["delta"]
            await ws.send(json.dumps({"type": "session.close"}))
        assert len(audio) >= 16000, f"only {len(audio)} bytes of greeting audio"
        assert rms_of_ulaw(bytes(audio)) > 500  # μ-law 8kHz, as requested, not silence
        assert "connection test" in said.lower(), said


# =============================================================================
# PLIVO INTEGRATION TESTS
# =============================================================================


class TestPlivoIntegration:
    """Integration tests for the Plivo API."""

    @pytest.fixture
    def plivo_configured(self):
        if not all([PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, PLIVO_PHONE_NUMBER]):
            pytest.skip("Plivo credentials not configured")

    def test_plivo_credentials_valid(self, plivo_configured):
        client = plivo.RestClient(auth_id=PLIVO_AUTH_ID, auth_token=PLIVO_AUTH_TOKEN)
        assert client.account.get() is not None

    def test_plivo_phone_number_exists(self, plivo_configured):
        client = plivo.RestClient(auth_id=PLIVO_AUTH_ID, auth_token=PLIVO_AUTH_TOKEN)
        try:
            number = client.numbers.get(number=normalize_phone_number(PLIVO_PHONE_NUMBER))
            assert number is not None
        except plivo.exceptions.ResourceNotFoundError:
            pytest.fail(f"Phone number {PLIVO_PHONE_NUMBER} not found")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
