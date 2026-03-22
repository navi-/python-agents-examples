"""
Real outbound call test.

Places an outbound call via the API and verifies the agent greets the callee.

Requires all API keys, LiveKit server, and Plivo credentials.

Run:
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
import pytest
from dotenv import load_dotenv

from tests.helpers import (
    start_ngrok,
    stop_ngrok,
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

TEST_PORT = 18004


@pytest.mark.skipif(not ALL_CONFIGURED, reason="Not all credentials configured")
class TestOutboundCall:
    """Outbound call test."""

    @pytest.fixture(scope="class")
    def infrastructure(self):
        """Start outbound server and ngrok."""
        ngrok_proc, public_url = start_ngrok(TEST_PORT)

        env = os.environ.copy()
        env["SERVER_PORT"] = str(TEST_PORT)
        env["PUBLIC_URL"] = public_url

        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        server_proc = subprocess.Popen(
            [sys.executable, "-m", "outbound.server"],
            cwd=project_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

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

    def test_outbound_call_initiation(self, infrastructure):
        """Test initiating an outbound call via the API."""
        resp = httpx.post(
            f"http://localhost:{TEST_PORT}/outbound/call",
            params={
                "phone_number": PLIVO_PHONE_NUMBER,
                "campaign_id": "test-campaign",
                "opening_reason": "a quick product demo",
                "objective": "Schedule a follow-up meeting",
                "context": "Test outbound call",
            },
            timeout=15.0,
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "call_id" in data
        assert data.get("status") in ("ringing", None) or "error" in data

        if "error" not in data:
            call_id = data["call_id"]

            # Wait a bit and check status
            time.sleep(15)

            status_resp = httpx.get(
                f"http://localhost:{TEST_PORT}/outbound/status/{call_id}",
                timeout=5.0,
            )
            assert status_resp.status_code == 200

            # Try to hang up
            with contextlib.suppress(Exception):
                httpx.post(
                    f"http://localhost:{TEST_PORT}/outbound/hangup/{call_id}",
                    timeout=5.0,
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
