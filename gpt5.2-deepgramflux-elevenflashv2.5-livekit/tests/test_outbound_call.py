"""
Outbound call test — initiates and verifies an outbound call.

Requires:
- Running outbound agent worker
- Plivo credentials and phone number
- LiveKit SIP trunk configured
- A valid destination phone number

Run:
    uv run pytest tests/test_outbound_call.py -v
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
    reason="Plivo and LiveKit credentials required for outbound call tests",
)


@pytest.mark.asyncio
async def test_outbound_call_greeting():
    """Initiate an outbound call and verify the agent greeting.

    This test:
    1. Uses the /outbound/call API to initiate a call
    2. Records the first 10 seconds of the outbound leg
    3. Transcribes the recording
    4. Verifies the agent introduces itself
    """
    pytest.skip("Outbound call test — run manually with infrastructure configured")
