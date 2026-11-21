"""
End-to-end live tests for the OpenAI GPT-Live example (real API, no phone call).

These tests:
1. Start the inbound server as a subprocess (JSON logs to a file)
2. Connect via WebSocket, simulating Plivo: start event, one 160-byte μ-law frame every
   20ms (silence unless the caller is speaking), a playback buffer that plays 20ms per
   tick and is emptied by clearAudio, and playedStream replies to checkpoints
3. Receive the agent greeting audio
4. Speak caller turns: GPT-Live takes audio only, so the questions are synthesized with
   OpenAI text-to-speech and streamed as μ-law 8kHz in real time
5. Transcribe the agent audio locally with faster-whisper and check its content
6. Say goodbye and verify the agent ends the call and the server closes the WebSocket

Set SAVE_CALL_AUDIO_DIR to keep each simulated call as a stereo WAV (caller left, agent
right); see the README "Voice samples".

Requirements:
    - Valid OPENAI_API_KEY in .env (GPT-Live, the delegation model and text-to-speech)
    - faster-whisper installed (dev dependency)
    - ffmpeg binary available (PATH, FFMPEG_DIR, or a parent directory)
    - Port 18001 available (used by test server)

Usage:
    uv run pytest tests/test_e2e_live.py -v -s
    SAVE_CALL_AUDIO_DIR=samples uv run pytest tests/test_e2e_live.py -v -s
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import json
import os
import tempfile
import time
import uuid
import wave
from pathlib import Path

import pytest
import websockets
from dotenv import load_dotenv

from tests.helpers import (
    ensure_ffmpeg_on_path,
    log_messages,
    read_log_events,
    rms_of_ulaw,
    server_log_path,
    signed_webhook,
    start_server,
    stop_server,
    stream_signature_headers,
    stream_url_from_xml,
    synthesize_caller_ulaw,
    ulaw_to_pcm,
    write_call_wav,
)

load_dotenv()
ensure_ffmpeg_on_path()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SAVE_CALL_AUDIO_DIR = os.getenv("SAVE_CALL_AUDIO_DIR", "")

TEST_PORT = 18001
TEST_HTTP_URL = f"http://localhost:{TEST_PORT}"
# Plivo auth token the local server checks webhook signatures with
TEST_AUTH_TOKEN = "test-plivo-auth-token"
LOG_PATH = server_log_path("e2e_live_server")
SILENCE = b"\xff" * 160

pytestmark = pytest.mark.skipif(not OPENAI_API_KEY, reason="OPENAI_API_KEY not configured")


# =============================================================================
# Helpers
# =============================================================================


def pcm16_to_wav(pcm_data: bytes, sample_rate: int = 8000) -> bytes:
    """Wrap raw PCM16 bytes in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


def transcribe_ulaw(ulaw_audio: bytes) -> str:
    """Transcribe μ-law 8kHz audio locally using faster-whisper."""
    from faster_whisper import WhisperModel

    model = WhisperModel("base", device="cpu", compute_type="int8")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(pcm16_to_wav(ulaw_to_pcm(ulaw_audio)))
        tmp_path = f.name
    try:
        segments, _ = model.transcribe(tmp_path, language="en")
        return " ".join(seg.text.strip() for seg in segments).strip()
    finally:
        os.unlink(tmp_path)


class SimulatedPlivo:
    """Plays Plivo's side of the bidirectional stream against the local server.

    Like Plivo, it sends one caller frame every 20ms and plays one buffered agent frame
    every 20ms, so audio the server sends faster than real time waits in ``_buffer``
    until it is played or cleared. ``caller_track`` and ``agent_track`` are what each
    side would hear, frame for frame.
    """

    def __init__(self, ws) -> None:
        self.ws = ws
        self.stream_id = str(uuid.uuid4())
        self.call_id = str(uuid.uuid4())
        self.caller_track = bytearray()
        self.agent_track = bytearray()
        self.cleared = 0  # clearAudio events received
        self.closed = asyncio.Event()
        self._caller_audio = bytearray()
        self._buffer: list[bytes] = []  # agent frames waiting to be played
        self._checkpoints: list[tuple[int, str]] = []  # (frames still to play, name)
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        start = {
            "event": "start",
            "start": {"callId": self.call_id, "streamId": self.stream_id},
        }
        await self.ws.send(json.dumps(start))
        self._tasks = [asyncio.create_task(self._clock()), asyncio.create_task(self._recv())]

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    async def _clock(self) -> None:
        """Every 20ms: send one caller frame, play one agent frame."""
        next_tick = time.monotonic()
        with contextlib.suppress(Exception):
            while not self.closed.is_set():
                frame = bytes(self._caller_audio[:160]).ljust(160, b"\xff")
                del self._caller_audio[:160]
                payload = base64.b64encode(frame).decode()
                await self.ws.send(json.dumps({"event": "media", "media": {"payload": payload}}))
                self.caller_track.extend(frame)
                self.agent_track.extend(self._buffer.pop(0) if self._buffer else SILENCE)
                await self._ack_checkpoints()
                next_tick += 0.02
                await asyncio.sleep(max(0.0, next_tick - time.monotonic()))

    async def _ack_checkpoints(self) -> None:
        due = [name for remaining, name in self._checkpoints if remaining <= 1]
        self._checkpoints = [(r - 1, n) for r, n in self._checkpoints if r > 1]
        for name in due:
            played = {"event": "playedStream", "name": name, "streamId": self.stream_id}
            await self.ws.send(json.dumps(played))

    async def _recv(self) -> None:
        try:
            async for raw in self.ws:
                msg = json.loads(raw)
                event = msg.get("event")
                if event == "playAudio":
                    assert msg["media"]["contentType"] == "audio/x-mulaw"
                    chunk = base64.b64decode(msg["media"]["payload"])
                    assert len(chunk) == 160
                    self._buffer.append(chunk)
                elif event == "checkpoint":
                    self._checkpoints.append((len(self._buffer), msg["name"]))
                elif event == "clearAudio":
                    self._buffer.clear()
                    self.cleared += 1
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.closed.set()

    @property
    def buffered_ms(self) -> int:
        """Agent audio received but not played yet (how far ahead of real time it arrives)."""
        return len(self._buffer) * 20

    async def say(self, ulaw_audio: bytes) -> None:
        """Speak caller audio into the call and wait until all of it has been sent."""
        self._caller_audio.extend(ulaw_audio)
        while self._caller_audio and not self.closed.is_set():
            await asyncio.sleep(0.05)

    async def listen(self, timeout: float = 30.0, quiet_s: float = 1.5) -> bytes:
        """Return what the agent says next: from now until it has been quiet for ``quiet_s``."""
        start = len(self.agent_track)
        deadline = time.monotonic() + timeout
        last_speech = None
        while time.monotonic() < deadline and not self.closed.is_set():
            await asyncio.sleep(0.1)
            if rms_of_ulaw(bytes(self.agent_track[-800:])) > 300:
                last_speech = time.monotonic()
            elif last_speech is not None and time.monotonic() - last_speech > quiet_s:
                break
        return bytes(self.agent_track[start:])

    def save(self, name: str) -> None:
        if SAVE_CALL_AUDIO_DIR:
            path = Path(SAVE_CALL_AUDIO_DIR) / f"{name}.wav"
            write_call_wav(path, bytes(self.caller_track), bytes(self.agent_track))
            print(f"\n[audio] saved {path} ({len(self.agent_track) / 8000:.1f}s)")


@contextlib.asynccontextmanager
async def plivo_call(call_uuid: str):
    # Answer webhook and /ws connection signed as Plivo signs them; /ws is its <Stream> URL
    answer = await asyncio.to_thread(
        signed_webhook,
        "POST",
        f"{TEST_HTTP_URL}/answer",
        TEST_AUTH_TOKEN,
        {"CallUUID": call_uuid, "From": "+15551234567", "To": "+16572338892"},
    )
    assert answer.status_code == 200, answer.text
    stream_url = stream_url_from_xml(answer.text)
    async with websockets.connect(
        stream_url,
        additional_headers=stream_signature_headers(stream_url, TEST_AUTH_TOKEN),
        close_timeout=3,
    ) as ws:
        call = SimulatedPlivo(ws)
        await call.start()
        try:
            yield call
        finally:
            await call.stop()
            call.save(call_uuid)


async def caller_audio(text: str) -> bytes:
    return await asyncio.to_thread(synthesize_caller_ulaw, text, OPENAI_API_KEY)


def events_for(call: SimulatedPlivo, event: str) -> list[dict]:
    return [e for e in read_log_events(LOG_PATH, event) if e["call_id"] == call.call_id]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def server_process():
    """Start the inbound server on TEST_PORT (SIGTERM -> wait(5) -> SIGKILL on teardown)."""
    # No PLIVO_AUTH_ID: end_call must not try a REST hangup of a fake call. Webhook auth
    # uses a test token (answer webhooks are signed like Plivo's).
    env = {
        "PLIVO_AUTH_ID": "",
        "PLIVO_AUTH_TOKEN": TEST_AUTH_TOKEN,
        "PLIVO_PHONE_NUMBER": "",
        "PUBLIC_URL": TEST_HTTP_URL,
    }
    proc = start_server("inbound.server", TEST_PORT, LOG_PATH, env)
    print(f"\n[server] logs: {LOG_PATH}")
    yield proc
    stop_server(proc)


# =============================================================================
# Tests
# =============================================================================


class TestE2ELive:
    """End-to-end tests that start the server and simulate a Plivo call."""

    async def test_agent_greeting(self, server_process):
        """Agent speaks the greeting when the call connects."""
        async with plivo_call("e2e-greeting") as call:
            greeting = await call.listen(timeout=20)

        assert len(greeting) > 8000, f"Greeting too short: {len(greeting)} bytes"
        rms = rms_of_ulaw(greeting)
        print(f"\n[Greeting] {len(greeting)} bytes ({len(greeting) / 8000:.1f}s), RMS {rms:.0f}")
        assert rms > 300, f"Audio RMS {rms:.1f} too low: likely silence"

        transcript = transcribe_ulaw(greeting)
        print(f"[Greeting transcript]: {transcript}")
        words = ["alex", "techflow", "live", "plivo", "help"]
        assert any(w in transcript.lower() for w in words), transcript
        await asyncio.sleep(0.5)  # let the server flush its logs
        (turn,) = events_for(call, "turn_complete")
        assert turn["turn"] == 1 and "techflow" in turn["agent_text"].lower(), turn

    async def test_conversation_with_tool_and_goodbye(self, server_process):
        """Plans question -> order status (delegated tool) -> goodbye (end_call hangs up)."""
        plans, order, goodbye = await asyncio.gather(
            caller_audio("What plans do you offer and how much do they cost?"),
            caller_audio("Can you check the status of my order? The number is T F 1 2 3 4 5 6."),
            caller_audio("No, that's everything. Thanks, goodbye."),
        )
        async with plivo_call("e2e-conversation") as call:
            greeting = await call.listen(timeout=20)
            assert greeting, "No greeting received"

            await call.say(plans)
            answer = await call.listen(timeout=30)
            print(f"\n[buffered ahead of real time] {call.buffered_ms}ms")
            await call.say(order)
            status = await call.listen(timeout=40, quiet_s=4.0)  # the tool call takes a moment
            await call.say(goodbye)
            try:
                await asyncio.wait_for(call.closed.wait(), timeout=40)
            except (TimeoutError, asyncio.TimeoutError):
                pytest.fail("Server did not close the WebSocket within 40s of goodbye")

        answer_text = transcribe_ulaw(answer)
        print(f"[Plans answer]: {answer_text}")
        product_words = ["pro", "team", "enterprise", "twelve", "twenty", "dollar", "month", "plan"]
        matches = [w for w in product_words if w in answer_text.lower()]
        assert len(matches) >= 2, f"Not about products. Matches: {matches}, got: {answer_text}"

        status_text = transcribe_ulaw(status)
        print(f"[Order answer]: {status_text}")
        order_words = ["order", "shipped", "processing", "delivered", "tracking", "ship"]
        assert any(w in status_text.lower() for w in order_words), status_text

        await asyncio.sleep(0.5)  # let the server flush its logs
        messages = log_messages(LOG_PATH)
        assert any("calling check_order_status" in m for m in messages), "order tool not used"
        assert any("calling end_call" in m for m in messages), "end_call was never invoked"
        assert any("goodbye played -- hanging up" in m for m in messages)
        turns = events_for(call, "turn_complete")
        print(
            f"[turn_complete] {[(t['turn'], t['barge_in'], t['response_gap_ms']) for t in turns]}"
        )
        assert len(turns) >= 3, turns
        assert "plan" in turns[1]["user_text"].lower(), turns[1]
        (end,) = events_for(call, "session_end")
        assert end["errors"] == 0, end

    async def test_barge_in(self, server_process):
        """Talking over a long answer: the agent stops and answers the new question."""
        question, interruption = await asyncio.gather(
            caller_audio("Please describe all three of your plans in as much detail as you can."),
            caller_audio("Sorry, stop. How much is the Pro plan?"),
        )
        async with plivo_call("e2e-barge-in") as call:
            await call.listen(timeout=20)  # greeting
            await call.say(question)
            deadline = time.monotonic() + 20
            while rms_of_ulaw(bytes(call.agent_track[-1600:])) < 300:  # answer has started
                assert time.monotonic() < deadline, "The agent never started answering"
                await asyncio.sleep(0.1)
            await asyncio.sleep(3)
            interrupted_at = len(call.agent_track)
            await call.say(interruption)
            overlap = bytes(call.agent_track[interrupted_at:])
            reply = await call.listen(timeout=30)

        # The agent must fall silent while the caller is still speaking
        tail_rms = rms_of_ulaw(overlap[-4000:])
        print(f"\n[Barge-in] agent RMS in the last 0.5s of the interruption: {tail_rms:.0f}")
        print(f"[Barge-in] clearAudio events: {call.cleared}")
        assert tail_rms < 300, "The agent kept talking over the caller"
        reply_text = transcribe_ulaw(reply)
        print(f"[Barge-in reply]: {reply_text}")
        assert any(w in reply_text.lower() for w in ("twelve", "12", "pro")), reply_text


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
