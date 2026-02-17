"""End-to-end live tests for the SIP voice agent.

These tests require:
- A running SIP server (inbound/server.py)
- Plivo SIP trunk configured to point to the server
- Valid API keys for Gemini, Deepgram, Cartesia

Run with:
    uv run pytest tests/test_e2e_live.py -v -k "e2e"

These tests are meant to be run manually during development, not in CI.
"""

from __future__ import annotations

import asyncio
import math
import os
import struct

import pytest
from dotenv import load_dotenv

load_dotenv()

PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID", "")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN", "")
PLIVO_PHONE_NUMBER = os.getenv("PLIVO_PHONE_NUMBER", "")
SIP_SERVER_HOST = os.getenv("SIP_SERVER_HOST", "127.0.0.1")
SIP_SERVER_PORT = int(os.getenv("SIP_PORT", "5060"))


class TestE2ELive:
    """End-to-end live tests using real SIP calls."""

    @pytest.fixture
    def sip_configured(self):
        """Check if SIP server is configured for testing."""
        if not all([PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, PLIVO_PHONE_NUMBER]):
            pytest.skip("SIP/Plivo credentials not configured for live testing")

    @pytest.mark.asyncio
    async def test_e2e_rtp_loopback_with_tone(self, sip_configured):
        """Test RTP loopback with a real audio tone."""
        from sip import RTPSession

        # Set up two RTP sessions talking to each other
        session_a = RTPSession(local_port=19100, remote_addr=("127.0.0.1", 19101))
        session_b = RTPSession(local_port=19101, remote_addr=("127.0.0.1", 19100))

        await session_a.start()
        await session_b.start()

        try:
            # Generate 1 second of 440Hz tone
            for frame_idx in range(50):  # 50 * 20ms = 1 second
                tone = [
                    int(
                        8000
                        * math.sin(
                            2 * math.pi * 440 * (frame_idx * 160 + i) / 8000
                        )
                    )
                    for i in range(160)
                ]
                pcm_frame = struct.pack(f"{len(tone)}h", *tone)
                await session_a.send_audio(pcm_frame)
                await asyncio.sleep(0.02)

            # Collect received frames
            await asyncio.sleep(0.2)
            received_frames = []
            while True:
                frame = session_b.receive_audio_nowait()
                if frame is None:
                    break
                received_frames.append(frame)

            assert len(received_frames) > 10, (
                f"Expected >10 frames, got {len(received_frames)}"
            )

            # Check audio quality
            all_audio = b"".join(received_frames)
            samples = struct.unpack(f"{len(all_audio) // 2}h", all_audio)
            rms = (sum(s**2 for s in samples) / len(samples)) ** 0.5
            assert rms > 500, f"Audio RMS {rms} too low"

        finally:
            await session_a.stop()
            await session_b.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
