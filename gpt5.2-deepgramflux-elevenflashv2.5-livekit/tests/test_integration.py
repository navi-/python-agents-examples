"""
Integration tests for GPT-5.2 LiveKit voice agent.

Test Levels:
1. Unit Tests - Test individual components (audio conversion, phone normalization)
2. Local Integration - Test server endpoints without external services

Run tests:
    uv run pytest tests/test_integration.py -v

Run specific test level:
    uv run pytest tests/test_integration.py -v -k "unit"
    uv run pytest tests/test_integration.py -v -k "local"
"""

from __future__ import annotations

import math
import os
import signal
import struct
import subprocess
import sys
import time

import httpx
import pytest
from dotenv import load_dotenv

# Import audio functions from utils module
from utils import normalize_phone_number, pcm_to_ulaw, ulaw_to_pcm

load_dotenv()

# Configuration from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY", "")
PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID", "")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN", "")

TEST_PORT = 18001
LOCAL_HTTP_URL = f"http://localhost:{TEST_PORT}"


# =============================================================================
# UNIT TESTS - Test individual components
# =============================================================================


class TestUnitAudioConversion:
    """Unit tests for audio format conversion."""

    def test_ulaw_to_pcm_conversion(self):
        """Test mu-law to PCM conversion."""
        ulaw_silence = b"\xff" * 160
        pcm_audio = ulaw_to_pcm(ulaw_silence)

        samples = struct.unpack(f"{len(pcm_audio) // 2}h", pcm_audio)
        avg_amplitude = sum(abs(s) for s in samples) / len(samples)

        assert len(pcm_audio) == 320  # 160 samples * 2 bytes
        assert avg_amplitude < 100  # Should be near silence

    def test_pcm_to_ulaw_conversion(self):
        """Test PCM to mu-law conversion."""
        pcm_silence = b"\x00" * 320
        ulaw_audio = pcm_to_ulaw(pcm_silence)

        assert len(ulaw_audio) == 160  # Half the size

    def test_audio_roundtrip(self):
        """Test that audio survives roundtrip conversion."""
        samples = []
        for i in range(160):
            sample = int(16000 * math.sin(2 * math.pi * 440 * i / 8000))
            samples.append(sample)
        pcm_original = struct.pack(f"{len(samples)}h", *samples)

        ulaw = pcm_to_ulaw(pcm_original)
        pcm_restored = ulaw_to_pcm(ulaw)

        original_samples = struct.unpack(f"{len(pcm_original) // 2}h", pcm_original)
        restored_samples = struct.unpack(f"{len(pcm_restored) // 2}h", pcm_restored)

        # Check correlation (should be > 0.9)
        correlation = sum(
            o * r for o, r in zip(original_samples, restored_samples, strict=True)
        )
        orig_energy = sum(o * o for o in original_samples)
        rest_energy = sum(r * r for r in restored_samples)

        if orig_energy > 0 and rest_energy > 0:
            normalized_corr = correlation / (orig_energy * rest_energy) ** 0.5
            assert normalized_corr > 0.9, "Audio quality degraded too much"


class TestUnitPhoneNormalization:
    """Unit tests for phone number normalization."""

    def test_normalize_e164_format(self):
        """Test normalizing E.164 formatted numbers."""
        result = normalize_phone_number("+16572338892")
        assert result == "16572338892"

    def test_normalize_with_spaces(self):
        """Test normalizing numbers with spaces."""
        result = normalize_phone_number("+1 657-233-8892")
        assert result == "16572338892"

    def test_normalize_local_format(self):
        """Test normalizing local format numbers."""
        result = normalize_phone_number("(657) 233-8892", "US")
        assert result == "16572338892"

    def test_normalize_empty(self):
        """Test normalizing empty string."""
        result = normalize_phone_number("")
        assert result == ""


# =============================================================================
# LOCAL INTEGRATION TESTS
# =============================================================================


class TestLocalIntegration:
    """Integration tests using local server endpoints."""

    @pytest.fixture(scope="class")
    def server_process(self):
        """Start the inbound server as a subprocess."""
        env = os.environ.copy()
        env["SERVER_PORT"] = str(TEST_PORT)

        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
        """Test the health check endpoint."""
        async with httpx.AsyncClient() as client:
            response = await client.get(LOCAL_HTTP_URL)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert "gpt5.2-deepgramflux-elevenflashv2.5-livekit" in data["service"]

    @pytest.mark.asyncio
    async def test_local_answer_webhook(self, server_process):
        """Test the answer webhook returns valid XML."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{LOCAL_HTTP_URL}/answer",
                params={
                    "CallUUID": "test123",
                    "From": "+15551234567",
                    "To": "+16572338892",
                },
            )
            assert response.status_code == 200
            assert "application/xml" in response.headers["content-type"]
            # Without LIVEKIT_SIP_URI, should return a fallback message
            assert "<Speak" in response.text or "<Dial" in response.text

    @pytest.mark.asyncio
    async def test_local_hangup_webhook(self, server_process):
        """Test the hangup webhook."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{LOCAL_HTTP_URL}/hangup",
                data={
                    "CallUUID": "test123",
                    "Duration": "30",
                    "HangupCause": "NORMAL_CLEARING",
                },
            )
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_local_fallback_webhook(self, server_process):
        """Test the fallback webhook."""
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{LOCAL_HTTP_URL}/fallback")
            assert response.status_code == 200
            assert "application/xml" in response.headers["content-type"]
            assert "<Speak" in response.text

    @pytest.mark.asyncio
    async def test_local_hold_webhook(self, server_process):
        """Test the hold webhook."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{LOCAL_HTTP_URL}/hold")
            assert response.status_code == 200
            assert "application/xml" in response.headers["content-type"]
            assert "<Wait" in response.text


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
