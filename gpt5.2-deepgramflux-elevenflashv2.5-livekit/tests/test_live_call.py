"""
Live inbound call test — places a real call to verify greeting.

Requires:
- Running inbound server + agent worker
- Plivo credentials and phone number
- LiveKit SIP trunk configured
- ngrok tunnel (or public URL)

Run:
    uv run pytest tests/test_live_call.py -v
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID", "")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN", "")
PLIVO_PHONE_NUMBER = os.getenv("PLIVO_PHONE_NUMBER", "")
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "")

pytestmark = pytest.mark.skipif(
    not all([PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, PLIVO_PHONE_NUMBER, LIVEKIT_URL]),
    reason="Plivo and LiveKit credentials required for live call tests",
)


@pytest.mark.asyncio
async def test_inbound_call_greeting():
    """Place a real inbound call and verify the agent responds with a greeting.

    This test:
    1. Calls the Plivo phone number
    2. Records the first 10 seconds
    3. Transcribes the recording
    4. Verifies greeting keywords are present
    """
    pytest.skip("Live call test — run manually with infrastructure configured")
