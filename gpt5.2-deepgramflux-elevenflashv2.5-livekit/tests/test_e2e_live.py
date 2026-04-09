"""
E2E live tests for LiveKit voice agent — tests with real API connections.

These tests require:
- Valid OPENAI_API_KEY, DEEPGRAM_API_KEY, ELEVEN_API_KEY
- LiveKit server access (LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
- No phone call required

Run:
    uv run pytest tests/test_e2e_live.py -v
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

LIVEKIT_URL = os.getenv("LIVEKIT_URL", "")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")

pytestmark = pytest.mark.skipif(
    not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]),
    reason="LiveKit credentials not configured",
)


@pytest.mark.asyncio
async def test_livekit_room_creation():
    """Test creating and deleting a LiveKit room."""
    from livekit.api import LiveKitAPI
    from livekit.protocol.room import CreateRoomRequest, DeleteRoomRequest

    lk_api = LiveKitAPI(
        url=LIVEKIT_URL,
        api_key=LIVEKIT_API_KEY,
        api_secret=LIVEKIT_API_SECRET,
    )

    room = await lk_api.room.create_room(
        CreateRoomRequest(name="test-e2e-room", empty_timeout=30)
    )
    assert room.name == "test-e2e-room"

    await lk_api.room.delete_room(
        DeleteRoomRequest(room="test-e2e-room")
    )
    await lk_api.aclose()


@pytest.mark.asyncio
async def test_livekit_list_sip_trunks():
    """Test listing SIP trunks (verifies SIP API access)."""
    from livekit.api import LiveKitAPI
    from livekit.protocol.sip import ListSIPInboundTrunkRequest

    lk_api = LiveKitAPI(
        url=LIVEKIT_URL,
        api_key=LIVEKIT_API_KEY,
        api_secret=LIVEKIT_API_SECRET,
    )

    trunks = await lk_api.sip.list_sip_inbound_trunk(
        ListSIPInboundTrunkRequest()
    )
    # Just verify we can list — may be empty
    assert trunks is not None

    await lk_api.aclose()
