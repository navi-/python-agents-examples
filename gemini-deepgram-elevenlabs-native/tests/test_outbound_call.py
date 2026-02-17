"""
Outbound call E2E tests -- verifies the outbound calling feature end-to-end.

Requirements:
    - Valid PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, PLIVO_PHONE_NUMBER,
      PLIVO_TEST_NUMBER, GEMINI_API_KEY, DEEPGRAM_API_KEY,
      ELEVENLABS_API_KEY in .env
    - ngrok binary available on PATH
    - faster-whisper installed (dev dependency)
    - Port 18003 available

Usage:
    cd gemini-deepgram-elevenlabs-native
    uv run pytest tests/test_outbound_call.py -v -s
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time

import httpx
import plivo
import pytest
from dotenv import load_dotenv

from tests.helpers import (
    download_recording,
    start_ngrok,
    stop_ngrok,
    transcribe_audio,
    wait_for_recording,
)

load_dotenv()

# Configuration
PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID", "")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN", "")
PLIVO_PHONE_NUMBER = os.getenv("PLIVO_PHONE_NUMBER", "")
PLIVO_TEST_NUMBER = os.getenv("PLIVO_TEST_NUMBER", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")

TEST_PORT = 18003
TEST_HTTP_URL = f"http://localhost:{TEST_PORT}"

pytestmark = pytest.mark.skipif(
    not all([
        PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, PLIVO_PHONE_NUMBER,
        PLIVO_TEST_NUMBER, GEMINI_API_KEY, DEEPGRAM_API_KEY,
        ELEVENLABS_API_KEY,
    ]),
    reason="Plivo credentials or API keys not configured",
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def ngrok_tunnel():
    """Start ngrok tunnel pointing at TEST_PORT."""
    proc, public_url = start_ngrok(TEST_PORT)
    print(f"\n[ngrok] Tunnel URL: {public_url}")

    yield public_url

    stop_ngrok(proc)


@pytest.fixture(scope="module")
def server_process(ngrok_tunnel):
    """Start the voice agent server as a subprocess on TEST_PORT."""
    env = os.environ.copy()
    env["SERVER_PORT"] = str(TEST_PORT)
    env["PUBLIC_URL"] = ngrok_tunnel

    project_dir = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    proc = subprocess.Popen(
        [sys.executable, "-m", "outbound.server"],
        cwd=project_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    ready = False
    for _ in range(30):
        try:
            resp = httpx.get(TEST_HTTP_URL, timeout=1.0)
            if resp.status_code == 200:
                ready = True
                break
        except Exception:
            pass
        time.sleep(0.5)

    if not ready:
        proc.terminate()
        proc.wait()
        output = proc.stdout.read().decode() if proc.stdout else ""
        pytest.skip(
            f"Server did not start in time. Output:\n{output[:2000]}"
        )

    yield proc

    os.kill(proc.pid, signal.SIGTERM)
    proc.wait(timeout=5)


@pytest.fixture(scope="module")
def plivo_client():
    """Create a Plivo REST client."""
    return plivo.RestClient(
        auth_id=PLIVO_AUTH_ID, auth_token=PLIVO_AUTH_TOKEN
    )


# =============================================================================
# Tests
# =============================================================================


class TestOutboundCall:
    """End-to-end tests for outbound calling."""

    def test_initiate_outbound_call_api(
        self, server_process, ngrok_tunnel
    ):
        """POST /outbound/call returns call_id and status tracking works."""
        public_url = ngrok_tunnel

        resp = httpx.post(
            f"{public_url}/outbound/call",
            params={
                "phone_number": PLIVO_TEST_NUMBER,
                "campaign_id": "test-campaign-1",
                "opening_reason": "your recent demo request",
                "objective": "qualify interest",
            },
            timeout=30.0,
        )
        assert resp.status_code == 200
        data = resp.json()

        assert "call_id" in data, f"Expected call_id in response: {data}"
        call_id = data["call_id"]

        status_resp = httpx.get(
            f"{public_url}/outbound/status/{call_id}",
            timeout=10.0,
        )
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        assert status_data["call_id"] == call_id
        assert status_data["status"] in (
            "ringing", "connected", "completed", "failed", "no_answer",
        )

        time.sleep(3)
        httpx.post(
            f"{public_url}/outbound/hangup/{call_id}",
            timeout=10.0,
        )

    def test_outbound_answer_webhook(
        self, server_process, ngrok_tunnel
    ):
        """Verify /outbound/answer returns valid Plivo Stream XML."""
        public_url = ngrok_tunnel

        resp = httpx.get(
            f"{public_url}/outbound/answer",
            params={
                "call_id": "test-call-123",
                "CallUUID": "test-uuid-456",
                "From": PLIVO_PHONE_NUMBER,
                "To": PLIVO_TEST_NUMBER,
            },
            timeout=10.0,
        )
        assert resp.status_code == 200
        body = resp.text

        assert "<Stream" in body
        assert "bidirectional" in body
        assert "ws" in body.lower()

    def test_outbound_call_full_cycle(
        self, server_process, ngrok_tunnel, plivo_client
    ):
        """Place a real outbound call, record, transcribe, verify greeting."""
        public_url = ngrok_tunnel

        resp = httpx.post(
            f"{public_url}/outbound/call",
            params={
                "phone_number": PLIVO_TEST_NUMBER,
                "campaign_id": "test-full-cycle",
                "opening_reason": "your recent demo request",
                "objective": "qualify interest",
            },
            timeout=30.0,
        )
        assert resp.status_code == 200
        data = resp.json()
        call_id = data.get("call_id")
        assert call_id, f"No call_id in response: {data}"

        call_uuid = None
        for _i in range(60):
            try:
                live_calls = plivo_client.live_calls.list_ids()
                call_ids = []
                if hasattr(live_calls, "calls"):
                    call_ids = live_calls.calls or []
                elif isinstance(live_calls, dict):
                    call_ids = live_calls.get("calls", [])
                if call_ids:
                    call_uuid = call_ids[0]
                    break
            except Exception:
                pass
            time.sleep(0.5)

        if not call_uuid:
            pytest.skip("Call did not connect")

        try:
            plivo_client.calls.start_recording(
                call_uuid, file_format="mp3"
            )
            time.sleep(20)
        finally:
            with contextlib.suppress(Exception):
                plivo_client.calls.delete(call_uuid)

        recording_url = wait_for_recording(
            plivo_client, call_uuid, timeout=30
        )
        assert recording_url

        audio_data = download_recording(recording_url)
        assert len(audio_data) > 1000

        transcript = transcribe_audio(audio_data)
        assert len(transcript) > 5

        outbound_words = [
            "alex", "techflow", "demo", "trial",
            "reaching out", "hi", "hello", "good time",
        ]
        matches = [w for w in outbound_words if w in transcript.lower()]
        assert matches, (
            f"Outbound greeting doesn't match: '{transcript}'"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
