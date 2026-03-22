"""
E2E tests with real API connections (no phone call).

Tests the LiveKit agent pipeline with real AssemblyAI, OpenAI, and Cartesia APIs.
Requires valid API keys in .env.

Run:
    uv run pytest tests/test_e2e_live.py -v
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", "")
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")


@pytest.mark.skipif(not OPENAI_API_KEY, reason="OPENAI_API_KEY not set")
class TestOpenAIConnection:
    """Test OpenAI API connectivity."""

    @pytest.mark.asyncio
    async def test_openai_chat_completion(self):
        """Test basic OpenAI chat completion."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        response = await client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[{"role": "user", "content": "Say hello in one word."}],
            max_tokens=10,
        )
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0


@pytest.mark.skipif(
    not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]),
    reason="LiveKit credentials not set",
)
class TestLiveKitConnection:
    """Test LiveKit server connectivity."""

    @pytest.mark.asyncio
    async def test_livekit_room_create(self):
        """Test creating a LiveKit room."""
        from livekit import api as livekit_api

        lk = livekit_api.LiveKitAPI(
            url=LIVEKIT_URL,
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET,
        )

        room = await lk.room.create_room(
            livekit_api.CreateRoomRequest(name="test-e2e-room", empty_timeout=30)
        )
        assert room.name == "test-e2e-room"

        # Cleanup
        await lk.room.delete_room(livekit_api.DeleteRoomRequest(room="test-e2e-room"))
        await lk.aclose()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
