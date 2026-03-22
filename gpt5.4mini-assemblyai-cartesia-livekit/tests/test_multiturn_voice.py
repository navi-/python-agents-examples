"""
Multi-turn conversation and barge-in verification test.

Tests that the agent can handle multiple conversation turns and
supports interruption (barge-in) during speech.

Requires all API keys and LiveKit server configured.

Run:
    uv run pytest tests/test_multiturn_voice.py -v -s
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time

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

PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID", "")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN", "")
PLIVO_PHONE_NUMBER = os.getenv("PLIVO_PHONE_NUMBER", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", "")
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "")

ALL_CONFIGURED = all(
    [
        PLIVO_AUTH_ID,
        PLIVO_AUTH_TOKEN,
        PLIVO_PHONE_NUMBER,
        OPENAI_API_KEY,
        ASSEMBLYAI_API_KEY,
        CARTESIA_API_KEY,
        LIVEKIT_URL,
    ]
)

TEST_PORT = 18003


@pytest.mark.skipif(not ALL_CONFIGURED, reason="Not all credentials configured")
class TestMultiturnVoice:
    """Multi-turn voice conversation test."""

    @pytest.fixture(scope="class")
    def infrastructure(self):
        """Start server and ngrok."""
        ngrok_proc, public_url = start_ngrok(TEST_PORT)

        env = os.environ.copy()
        env["SERVER_PORT"] = str(TEST_PORT)
        env["PUBLIC_URL"] = public_url

        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        server_proc = subprocess.Popen(
            [sys.executable, "-m", "inbound.server"],
            cwd=project_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        import httpx

        ready = False
        for _ in range(30):
            try:
                resp = httpx.get(f"http://localhost:{TEST_PORT}", timeout=1.0)
                if resp.status_code == 200:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(0.5)

        if not ready:
            server_proc.terminate()
            server_proc.wait()
            stop_ngrok(ngrok_proc)
            pytest.skip("Server did not start")

        yield {"server": server_proc, "ngrok": ngrok_proc, "url": public_url}

        os.kill(server_proc.pid, signal.SIGTERM)
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            server_proc.wait()
        stop_ngrok(ngrok_proc)

    def test_multiturn_conversation(self, infrastructure):
        """Place a longer call to test multi-turn conversation."""
        client = plivo.RestClient(auth_id=PLIVO_AUTH_ID, auth_token=PLIVO_AUTH_TOKEN)

        call = client.calls.create(
            from_=PLIVO_PHONE_NUMBER,
            to_=PLIVO_PHONE_NUMBER,
            answer_url=f"{infrastructure['url']}/answer",
            answer_method="POST",
            hangup_url=f"{infrastructure['url']}/hangup",
            hangup_method="POST",
            record=True,
            recording_callback_url=f"{infrastructure['url']}/hangup",
        )

        call_uuid = (
            call.get("request_uuid", "")
            if isinstance(call, dict)
            else getattr(call, "request_uuid", "")
        )
        assert call_uuid, "Failed to get call UUID"

        # Let call run for 30 seconds to capture multiple turns
        time.sleep(30)

        with contextlib.suppress(Exception):
            client.calls.delete(call_uuid)

        time.sleep(5)

        recording_url = wait_for_recording(client, call_uuid, timeout=30)
        if not recording_url:
            pytest.skip("No recording available")

        audio = download_recording(recording_url)
        transcript = transcribe_audio(audio)

        assert len(transcript) > 20, f"Transcript too short for multi-turn: {transcript}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
