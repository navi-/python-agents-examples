"""
Multi-turn voice conversation + barge-in tests over a real Plivo call.

Setup is the same as test_live_call.py (ngrok -> inbound server on port 18004, the
calling leg answers with /hold, PLIVO_PHONE_NUMBER's leg runs the GPT-Live agent).
User turns are spoken into the call with Plivo's Speak API, so they reach GPT-Live
as real telephone audio.

Tests:
1. Multi-turn: plans question -> order status (check_order_status tool) -> goodbye.
   The agent must call end_call and hang up over REST after its goodbye has played.
2. Barge-in: speak over a long answer. GPT-Live hears the caller while it speaks and
   stops by itself, so the server (LOG_FORMAT=json) must log the interruption as a new
   user turn that cut the answer short, and the answer to the second question. A
   turn_complete with barge_in=true is logged only if unplayed audio had to be flushed.

Requirements:
    - PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, PLIVO_PHONE_NUMBER, PLIVO_TEST_NUMBER and
      OPENAI_API_KEY in .env
    - ngrok binary available on PATH
    - faster-whisper installed (dev dependency), ffmpeg available
    - Port 18004 available

Usage:
    uv run pytest tests/test_multiturn_voice.py -v -s
"""

from __future__ import annotations

import os
import time

import plivo
import pytest
from dotenv import load_dotenv

from tests.helpers import (
    best_transcript,
    ensure_ffmpeg_on_path,
    get_app_id_for_number,
    hangup_quietly,
    list_live_call_ids,
    place_call_and_wait,
    read_log_records,
    server_log_path,
    start_server,
    start_tunnel,
    stop_server,
    stop_tunnel,
    upsert_application,
)
from utils import normalize_phone_number

load_dotenv()
ensure_ffmpeg_on_path()

PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID", "")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN", "")
PLIVO_PHONE_NUMBER = os.getenv("PLIVO_PHONE_NUMBER", "")
PLIVO_TEST_NUMBER = os.getenv("PLIVO_TEST_NUMBER", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

TEST_PORT = 18004
LOG_PATH = server_log_path("multiturn_server")
APP_NAME = "GPT_Live_Native_Test"

pytestmark = pytest.mark.skipif(
    not all(
        [PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, PLIVO_PHONE_NUMBER, PLIVO_TEST_NUMBER, OPENAI_API_KEY]
    ),
    reason="Plivo credentials, PLIVO_TEST_NUMBER, or OPENAI_API_KEY not configured",
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def tunnel_url():
    proc, public_url = start_tunnel(TEST_PORT)
    print(f"\n[tunnel] URL: {public_url}")
    yield public_url
    stop_tunnel(proc)


@pytest.fixture(scope="module")
def server_process(tunnel_url):
    """Inbound server with JSON logs (SIGTERM -> wait(5) -> SIGKILL on teardown)."""
    proc = start_server(
        "inbound.server",
        TEST_PORT,
        LOG_PATH,
        {"PUBLIC_URL": tunnel_url, "PLIVO_PHONE_NUMBER": "", "LOG_FORMAT": "json"},
    )
    print(f"[server] logs: {LOG_PATH}")
    yield proc
    stop_server(proc)


@pytest.fixture(scope="module")
def plivo_setup(server_process, tunnel_url):
    client = plivo.RestClient(auth_id=PLIVO_AUTH_ID, auth_token=PLIVO_AUTH_TOKEN)
    phone_digits = normalize_phone_number(PLIVO_PHONE_NUMBER)
    original_app_id = get_app_id_for_number(client, phone_digits)
    app_id = upsert_application(
        client, APP_NAME, f"{tunnel_url}/answer", hangup_url=f"{tunnel_url}/hangup"
    )
    client.numbers.update(number=phone_digits, app_id=app_id)
    print(f"[Plivo] Assigned {phone_digits} to {APP_NAME} ({app_id})")

    yield {"client": client, "public_url": tunnel_url}

    if original_app_id and original_app_id != app_id:
        client.numbers.update(number=phone_digits, app_id=original_app_id)
        print(f"\n[Plivo] Restored {phone_digits} to app {original_app_id}")


# =============================================================================
# Helpers
# =============================================================================


def _place_call(client, public_url):
    outbound_uuid, agent_uuid = place_call_and_wait(
        client,
        from_digits=normalize_phone_number(PLIVO_TEST_NUMBER),
        to_digits=normalize_phone_number(PLIVO_PHONE_NUMBER),
        answer_url=f"{public_url}/hold",
    )
    assert outbound_uuid, "Call did not go live within 30s"
    for call_uuid in (outbound_uuid, agent_uuid):
        if call_uuid:
            try:
                client.calls.start_recording(call_uuid, file_format="mp3")
            except Exception as e:
                print(f"[Call] recording failed on {call_uuid}: {e}")
    return outbound_uuid, agent_uuid


def _say(client, call_uuid: str, text: str) -> None:
    print(f"[Call] user says: '{text}'")
    client.calls.speak(call_uuid, text=text, language="en-US", legs="aleg")


def _new_records(offset: int) -> list[dict]:
    return read_log_records(LOG_PATH)[offset:]


def _wait_for_message(offset: int, text: str, after: str, timeout: float) -> bool:
    """Wait until a log message containing ``text`` follows the first ``after`` event."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        records = _new_records(offset)
        start = next(
            (i for i, r in enumerate(records) if r.get("extra", {}).get("event") == after), None
        )
        if start is not None and any(text in r.get("message", "") for r in records[start:]):
            return True
        time.sleep(0.2)
    return False


def _events(records: list[dict], name: str) -> list[dict]:
    return [r["extra"] for r in records if r.get("extra", {}).get("event") == name]


def _print_turns(records: list[dict]) -> None:
    for r in records:
        msg = r.get("message", "")
        if "response gap" in msg or "[barge_in]" in msg or "[tool]" in msg or "hanging up" in msg:
            print(f"[Log] {msg}")
    for t in _events(records, "turn_complete"):
        print(
            f"[Log] turn_complete turn={t['turn']} barge_in={t['barge_in']} "
            f"response_gap_ms={t['response_gap_ms']} "
            f"user='{t['user_text'][:50]}'"
        )


# =============================================================================
# Tests
# =============================================================================


class TestMultiturnVoice:
    """Multi-turn conversation and barge-in over a real Plivo call."""

    def test_multiturn_conversation_with_end_call(self, plivo_setup):
        """Plans -> order status -> goodbye; the agent hangs up after its goodbye plays."""
        client = plivo_setup["client"]
        offset = len(read_log_records(LOG_PATH))
        outbound_uuid, agent_uuid = _place_call(client, plivo_setup["public_url"])
        agent_hung_up = False

        try:
            time.sleep(16)  # greeting
            _say(client, outbound_uuid, "What plans do you offer?")
            time.sleep(28)
            _say(client, outbound_uuid, "Can you check the status of order T F 1 2 3 4 5 6?")
            time.sleep(28)
            _say(client, outbound_uuid, "No, that's everything. Thanks, goodbye.")
            deadline = time.time() + 40
            while time.time() < deadline:
                time.sleep(2)
                live = set(list_live_call_ids(client))
                if outbound_uuid not in live and (not agent_uuid or agent_uuid not in live):
                    agent_hung_up = True
                    break
        finally:
            hangup_quietly(client, outbound_uuid, agent_uuid)

        time.sleep(2)
        records = _new_records(offset)
        _print_turns(records)
        messages = [r.get("message", "") for r in records]

        assert agent_hung_up, "Call was still live 40s after the goodbye"
        assert any("calling end_call" in m for m in messages), "end_call was never invoked"
        assert any("goodbye played -- hanging up" in m for m in messages)
        assert any("Hung up call via REST" in m for m in messages), "REST hangup not logged"
        assert any("calling check_order_status" in m for m in messages), "order tool not used"

        user_texts = [e["text"] for e in _events(records, "user_text")]
        print(f"[Log] user_text (GPT-Live): {user_texts}")
        assert len(user_texts) >= 3, f"Expected 3 user turns, GPT-Live heard: {user_texts}"
        assert _events(records, "session_end"), "No session_end event"

        transcript = best_transcript(client, [outbound_uuid, agent_uuid]).lower()
        assert any(w in transcript for w in ("plan", "pro", "team", "enterprise")), transcript
        assert any(
            w in transcript for w in ("order", "shipped", "processing", "delivered", "tracking")
        ), transcript

    def test_barge_in(self, plivo_setup):
        """Speaking over a long answer starts a new turn and gets the new question answered."""
        client = plivo_setup["client"]
        offset = len(read_log_records(LOG_PATH))
        outbound_uuid, agent_uuid = _place_call(client, plivo_setup["public_url"])

        try:
            time.sleep(16)  # greeting
            _say(client, outbound_uuid, "What plans do you offer and how much do they cost?")
            # Interrupt only once the (long) answer is really playing: the server logs
            # "response gap" at the first transcript fragment of the answer.
            playing = _wait_for_message(offset, "response gap", after="call_answered", timeout=30)
            assert playing, "The agent never started answering the question"
            time.sleep(3)
            _say(client, outbound_uuid, "Wait, stop. Can you check order T F 6 5 4 3 2 1?")
            time.sleep(20)
        finally:
            hangup_quietly(client, outbound_uuid, agent_uuid)

        time.sleep(2)
        records = _new_records(offset)
        _print_turns(records)
        turns = _events(records, "turn_complete")
        flushed = [t for t in turns if t["barge_in"]]
        print(f"[Log] turns with flushed audio (barge_in=true): {[t['turn'] for t in flushed]}")
        user_texts = [e["text"].lower() for e in _events(records, "user_text")]
        assert len(user_texts) >= 2, f"The interruption was not heard as a new turn: {user_texts}"
        assert any("order" in text or "stop" in text for text in user_texts[1:]), user_texts
        assert any("calling check_order_status" in r.get("message", "") for r in records), (
            "The question asked over the agent's answer was not acted on"
        )
        assert _events(records, "session_end"), "No session_end event"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
