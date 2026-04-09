"""
Integration tests for GPT-5.2 LiveKit voice agent.

Test Levels:
1. Unit Tests - Phone normalization (offline, no API keys)
2. Local Integration - Optional server health check

Note: No audio conversion tests — LiveKit handles all audio transport
natively via SIP bridge. No WebSocket tests — LiveKit handles audio
through rooms, not direct WebSocket.

Run:
    uv run pytest tests/test_integration.py -v -k "unit"
    uv run pytest tests/test_integration.py -v -k "local"
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import httpx
import pytest
from dotenv import load_dotenv

from utils import normalize_phone_number

load_dotenv()

TEST_PORT = 18001
LOCAL_HTTP_URL = f"http://localhost:{TEST_PORT}"


# =============================================================================
# UNIT TESTS
# =============================================================================


class TestUnitPhoneNormalization:
    """Unit tests for phone number normalization."""

    def test_normalize_e164_format(self):
        result = normalize_phone_number("+16572338892")
        assert result == "16572338892"

    def test_normalize_with_spaces(self):
        result = normalize_phone_number("+1 657-233-8892")
        assert result == "16572338892"

    def test_normalize_local_format(self):
        result = normalize_phone_number("(657) 233-8892", "US")
        assert result == "16572338892"

    def test_normalize_empty(self):
        result = normalize_phone_number("")
        assert result == ""

    def test_normalize_international(self):
        result = normalize_phone_number("+442071234567")
        assert result == "442071234567"


class TestUnitCallManager:
    """Unit tests for outbound CallManager."""

    def test_create_call(self):
        from outbound.agent import CallManager

        mgr = CallManager()
        record = mgr.create_call(
            phone_number="+15551234567",
            campaign_id="test-campaign",
            opening_reason="Follow up on demo request",
            objective="Schedule a demo",
        )
        assert record.phone_number == "+15551234567"
        assert record.campaign_id == "test-campaign"
        assert record.status == "initiating"
        assert "demo request" in record.initial_message
        assert "{{opening_reason}}" not in record.system_prompt

    def test_update_status(self):
        from outbound.agent import CallManager

        mgr = CallManager()
        record = mgr.create_call(phone_number="+15551234567")
        mgr.update_status(record.call_id, "ringing", livekit_room_name="outbound-123")
        updated = mgr.get_call(record.call_id)
        assert updated.status == "ringing"
        assert updated.livekit_room_name == "outbound-123"

    def test_get_active_calls(self):
        from outbound.agent import CallManager

        mgr = CallManager()
        r1 = mgr.create_call(phone_number="+15551111111")
        r2 = mgr.create_call(phone_number="+15552222222")
        mgr.update_status(r1.call_id, "completed")
        active = mgr.get_active_calls()
        assert len(active) == 1
        assert active[0].call_id == r2.call_id

    def test_campaign_filter(self):
        from outbound.agent import CallManager

        mgr = CallManager()
        mgr.create_call(phone_number="+15551111111", campaign_id="A")
        mgr.create_call(phone_number="+15552222222", campaign_id="B")
        mgr.create_call(phone_number="+15553333333", campaign_id="A")
        assert len(mgr.get_calls_by_campaign("A")) == 2
        assert len(mgr.get_calls_by_campaign("B")) == 1


# =============================================================================
# LOCAL INTEGRATION TESTS
# =============================================================================


class TestLocalIntegration:
    """Integration tests for the outbound management server."""

    @pytest.fixture(scope="class")
    def server_process(self):
        """Start the outbound server as a subprocess."""
        env = os.environ.copy()
        env["SERVER_PORT"] = str(TEST_PORT)

        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
                resp = httpx.get(LOCAL_HTTP_URL, timeout=1.0)
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
            pytest.skip(f"Server did not start in time. Output:\n{output[:2000]}")

        yield proc

        os.kill(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    @pytest.mark.asyncio
    async def test_local_health_check(self, server_process):
        async with httpx.AsyncClient() as client:
            response = await client.get(LOCAL_HTTP_URL)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert "livekit" in data["service"]

    @pytest.mark.asyncio
    async def test_outbound_call_missing_number(self, server_process):
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{LOCAL_HTTP_URL}/outbound/call")
            assert response.status_code == 200
            data = response.json()
            assert "error" in data

    @pytest.mark.asyncio
    async def test_outbound_status_not_found(self, server_process):
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{LOCAL_HTTP_URL}/outbound/status/nonexistent-id"
            )
            assert response.status_code == 200
            data = response.json()
            assert data["error"] == "Call not found"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
