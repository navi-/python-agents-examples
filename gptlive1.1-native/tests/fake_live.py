"""A local stand-in for the GPT-Live WebSocket API (``/v1/live/sessions``).

It speaks the wire protocol the agent uses, as OpenAI documents it: ``session.start`` ->
``session.started``, ``session.input_audio.append`` in, ``session.output_audio.delta`` and
transcript deltas out, Responses delegation in ``response.event`` envelopes, and
``session.close`` -> ``session.closed``. It is not a model: the "caller speech" it hears
is any audio above the quiet band, and its "voice" is a tone.

Behaviour, per connection:
  1. Speaks the greeting found in ``<say>`` tags of the session.input developer message.
  2. Each time the caller speaks (>= 300ms of loud audio, then 500ms of quiet) it plays
     the next entry of ``replies``: text to speak, or ``{"function": name, "arguments":
     {...}}`` to delegate a function call first and speak once the result is returned.

Point the server at it with OPENAI_LIVE_URL=ws://127.0.0.1:<port>.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import re
import threading
import time
from typing import Any

import websockets

from tests.helpers import rms_of_ulaw, tone_ulaw

SPEECH_RMS = 500.0  # caller audio above this is "speech"
SPEECH_MIN_S = 0.3
END_OF_SPEECH_S = 0.5
WORD_S = 0.3  # the fake voice plays this much tone per word


class FakeLiveServer:
    """In-process fake GPT-Live server; use as ``async with FakeLiveServer() as fake``."""

    def __init__(self, replies: list[Any] | None = None, realtime: bool = True) -> None:
        self.replies = list(replies or [])
        self.realtime = realtime  # False: deliver each utterance's audio in one burst
        self.received: list[dict[str, Any]] = []  # every client event except input audio
        self.headers: dict[str, str] = {}
        self.input_audio_bytes = 0
        self.url = ""
        self._server: Any = None

    async def __aenter__(self) -> FakeLiveServer:
        self._server = await websockets.serve(self._handle, "127.0.0.1", 0)
        port = self._server.sockets[0].getsockname()[1]
        self.url = f"ws://127.0.0.1:{port}"
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._server.close()
        await self._server.wait_closed()

    def types(self) -> list[str]:
        return [m.get("type", "") for m in self.received]

    async def _handle(self, ws: Any) -> None:
        self.headers = dict(ws.request.headers)
        session = _Session(self, ws)
        with contextlib.suppress(websockets.exceptions.ConnectionClosed):
            await session.run()


class _Session:
    def __init__(self, server: FakeLiveServer, ws: Any) -> None:
        self.server = server
        self.ws = ws
        self.t0 = time.monotonic()
        self.replies = list(server.replies)
        self.speech_s = 0.0
        self.quiet_s = 0.0
        self.speaking: asyncio.Task | None = None
        self.pending_calls: dict[str, str] = {}  # call_id -> text to speak afterwards
        self.returned: set[str] = set()

    def _ms(self) -> int:
        return round((time.monotonic() - self.t0) * 1000)

    async def _send(self, event: dict[str, Any]) -> None:
        await self.ws.send(json.dumps(event))

    async def run(self) -> None:
        first = json.loads(await self.ws.recv())
        self.server.received.append(first)
        session = first.get("session") or {}
        if first.get("type") != "session.start" or not session.get("model"):
            await self._send(
                {
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "code": "session_not_started",
                        "message": "The first event must be session.start with a model.",
                    },
                }
            )
            return
        await self._send(
            {
                "type": "session.started",
                "session": {
                    "id": "live_fake_123",
                    "status": "active",
                    "expires_at": int(time.time()) + 3600,
                    "model": session["model"],
                },
            }
        )
        for message in session.get("input") or []:
            text = " ".join(c.get("text", "") for c in message.get("content", []))
            said = re.search(r"<say>(.*?)</say>", text, re.S)
            if message.get("role") == "developer" and said:
                self._speak(said.group(1))

        async for raw in self.ws:
            event = json.loads(raw)
            etype = event.get("type")
            if etype == "session.input_audio.append":
                await self._on_audio(base64.b64decode(event["audio"]))
                continue
            self.server.received.append(event)
            if etype == "response.item.create":
                self.returned.add(event["item"]["call_id"])
            elif etype == "response.create":
                for call_id in [c for c in self.pending_calls if c in self.returned]:
                    self._speak(self.pending_calls.pop(call_id))
            elif etype == "session.close":
                if self.speaking:
                    self.speaking.cancel()
                usage = {"seconds": round(time.monotonic() - self.t0)}
                await self._send(
                    {"type": "session.closed", "reason": "close_requested", "usage": usage}
                )
                return

    async def _on_audio(self, audio: bytes) -> None:
        self.server.input_audio_bytes += len(audio)
        duration = len(audio) / 8000
        if rms_of_ulaw(audio) > SPEECH_RMS:
            if self.speech_s == 0.0 and self.speaking and not self.speaking.done():
                self.speaking.cancel()  # full duplex: the caller talks over the agent
            self.speech_s += duration
            self.quiet_s = 0.0
        elif self.speech_s:
            self.quiet_s += duration
            if self.quiet_s >= END_OF_SPEECH_S:
                heard = self.speech_s >= SPEECH_MIN_S
                self.speech_s = self.quiet_s = 0.0
                if heard:
                    await self._on_caller_turn()

    async def _on_caller_turn(self) -> None:
        now = self._ms()
        await self._send(
            {
                "type": "session.input_transcript.delta",
                "delta": "caller speech",
                "start_ms": now - 1000,
                "end_ms": now - 500,
            }
        )
        reply = self.replies.pop(0) if self.replies else "Okay."
        if isinstance(reply, dict):
            await self._delegate(reply)
        else:
            self._speak(reply)

    async def _delegate(self, reply: dict[str, Any]) -> None:
        call_id = f"call_{len(self.pending_calls) + len(self.returned) + 1}"
        self.pending_calls[call_id] = reply.get("then", "Done.")
        delegation = {"id": "dlg_1", "type": "delegation", "target": "responses"}
        await self._send({"type": "session.delegation.created", "delegation": delegation})
        item = {
            "type": "function_call",
            "call_id": call_id,
            "name": reply["function"],
            "arguments": json.dumps(reply.get("arguments", {})),
        }
        for event in (
            {"type": "response.created", "response": {"id": "resp_1", "output": []}},
            {"type": "response.output_item.done", "item": item},
            {"type": "response.completed", "response": {"id": "resp_1", "output": []}},
        ):
            await self._send({"type": "response.event", "delegation_id": "dlg_1", "event": event})

    def _speak(self, text: str) -> None:
        if self.speaking and not self.speaking.done():
            self.speaking.cancel()
        self.speaking = asyncio.get_running_loop().create_task(self._speak_task(text))

    async def _speak_task(self, text: str) -> None:
        with contextlib.suppress(websockets.exceptions.ConnectionClosed):
            for word in text.split():
                start = self._ms()
                audio = tone_ulaw(WORD_S)
                await self._send(
                    {
                        "type": "session.output_transcript.delta",
                        "delta": word + " ",
                        "start_ms": start,
                        "end_ms": start + round(WORD_S * 1000),
                    }
                )
                for i in range(0, len(audio), 800):  # 100ms deltas
                    delta = base64.b64encode(audio[i : i + 800]).decode()
                    await self._send({"type": "session.output_audio.delta", "delta": delta})
                    if self.server.realtime:
                        await asyncio.sleep(0.1)


class FakeLiveThread:
    """Runs a FakeLiveServer on its own event loop in a daemon thread (for subprocess tests)."""

    def __init__(self, replies: list[Any] | None = None, realtime: bool = True) -> None:
        self.server = FakeLiveServer(replies, realtime)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)

    def start(self) -> str:
        self._thread.start()
        asyncio.run_coroutine_threadsafe(self.server.__aenter__(), self._loop).result(5)
        return self.server.url

    def stop(self) -> None:
        asyncio.run_coroutine_threadsafe(self.server.__aexit__(), self._loop).result(5)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(5)
