"""
Live call E2E test -- places a real call through Plivo infrastructure.

Requirements:
    - Valid PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, PLIVO_PHONE_NUMBER,
      PLIVO_TEST_NUMBER, GEMINI_API_KEY, DEEPGRAM_API_KEY,
      ELEVENLABS_API_KEY in .env
    - ngrok binary available on PATH
    - faster-whisper installed (dev dependency)
    - Port 18002 available

Usage:
    cd gemini-deepgram-elevenlabs-native
    uv run pytest tests/test_live_call.py -v -s
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

TEST_PORT = 18002
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
def server_process():
    """Start the voice agent server as a subprocess on TEST_PORT."""
    env = os.environ.copy()
    env["SERVER_PORT"] = str(TEST_PORT)

    project_dir = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    proc = subprocess.Popen(
        [sys.executable, "-m", "inbound.server"],
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
def ngrok_tunnel(server_process):
    """Start ngrok tunnel pointing at the test server."""
    proc, public_url = start_ngrok(TEST_PORT)
    print(f"\n[ngrok] Tunnel URL: {public_url}")

    yield public_url

    stop_ngrok(proc)


@pytest.fixture(scope="module")
def plivo_configured(ngrok_tunnel):
    """Configure Plivo app and assign phone number."""
    client = plivo.RestClient(
        auth_id=PLIVO_AUTH_ID, auth_token=PLIVO_AUTH_TOKEN
    )
    public_url = ngrok_tunnel

    app_name = "Gemini_DG_EL_Agent_Test"
    answer_url = f"{public_url}/answer"
    hangup_url = f"{public_url}/hangup"

    apps = client.applications.list()
    existing_app = None
    for app_obj in apps["objects"]:
        if app_obj["app_name"] == app_name:
            existing_app = app_obj
            break

    if existing_app:
        client.applications.update(
            app_id=existing_app["app_id"],
            answer_url=answer_url,
            answer_method="POST",
            hangup_url=hangup_url,
            hangup_method="POST",
        )
        app_id = existing_app["app_id"]
    else:
        response = client.applications.create(
            app_name=app_name,
            answer_url=answer_url,
            answer_method="POST",
            hangup_url=hangup_url,
            hangup_method="POST",
        )
        app_id = response["app_id"]

    phone_digits = "".join(
        c for c in PLIVO_PHONE_NUMBER if c.isdigit()
    )
    client.numbers.update(number=phone_digits, app_id=app_id)

    yield {
        "client": client,
        "app_id": app_id,
        "public_url": public_url,
    }


# =============================================================================
# Tests
# =============================================================================


class TestLiveCall:
    """End-to-end tests that place a real call through Plivo."""

    def test_ngrok_tunnel_accessible(self, ngrok_tunnel):
        """Verify the ngrok tunnel reaches our server."""
        resp = httpx.get(ngrok_tunnel, timeout=10.0)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_answer_webhook_via_ngrok(self, plivo_configured):
        """Verify /answer returns valid Plivo XML through ngrok."""
        public_url = plivo_configured["public_url"]
        resp = httpx.get(
            f"{public_url}/answer",
            params={
                "CallUUID": "test-ngrok",
                "From": "+15551234567",
                "To": "+16572338892",
            },
            timeout=10.0,
        )
        assert resp.status_code == 200
        body = resp.text
        assert "<Stream" in body
        assert "bidirectional" in body

    def _place_call_and_wait(self, client, public_url):
        """Place an outbound call and return the live call_uuid."""
        hold_url = f"{public_url}/hold"
        from_digits = "".join(
            c for c in PLIVO_TEST_NUMBER if c.isdigit()
        )
        to_digits = "".join(
            c for c in PLIVO_PHONE_NUMBER if c.isdigit()
        )

        call_response = client.calls.create(
            from_=from_digits,
            to_=to_digits,
            answer_url=hold_url,
            answer_method="POST",
        )
        request_uuid = call_response["request_uuid"]

        call_uuid = None
        for _i in range(60):
            try:
                live_calls = client.live_calls.list_ids()
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

        assert call_uuid, (
            f"Call did not go live within 30s (request_uuid={request_uuid})"
        )
        return call_uuid

    def test_live_call_greeting(self, plivo_configured):
        """Place a real call, record it, transcribe, verify greeting."""
        client = plivo_configured["client"]
        public_url = plivo_configured["public_url"]

        call_uuid = self._place_call_and_wait(client, public_url)

        try:
            client.calls.start_recording(call_uuid, file_format="mp3")
            time.sleep(20)
        finally:
            with contextlib.suppress(Exception):
                client.calls.delete(call_uuid)

        recording_url = wait_for_recording(client, call_uuid, timeout=30)
        assert recording_url, (
            f"No recording found for call {call_uuid} within 30s"
        )

        audio_data = download_recording(recording_url)
        assert len(audio_data) > 1000

        transcript = transcribe_audio(audio_data)
        assert len(transcript) > 5

        greeting_words = [
            "hello", "hi", "welcome", "help", "how",
            "assist", "alex", "techflow",
        ]
        matches = [w for w in greeting_words if w in transcript.lower()]
        assert matches, (
            f"Greeting doesn't match expected content: '{transcript}'"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
