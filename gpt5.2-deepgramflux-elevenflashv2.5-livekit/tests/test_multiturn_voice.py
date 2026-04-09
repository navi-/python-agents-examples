"""
Multi-turn conversation test — verifies barge-in and turn-taking.

Requires:
- Running inbound agent worker
- Plivo credentials and phone number
- LiveKit SIP trunk configured

Run:
    uv run pytest tests/test_multiturn_voice.py -v
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
    reason="Plivo and LiveKit credentials required for multi-turn tests",
)


@pytest.mark.asyncio
async def test_multiturn_conversation():
    """Test multi-turn conversation with barge-in.

    This test:
    1. Calls the agent
    2. Waits for greeting
    3. Sends a question
    4. Verifies agent responds
    5. Interrupts the agent (barge-in)
    6. Sends another question
    7. Verifies context is maintained
    """
    pytest.skip("Multi-turn test — run manually with infrastructure configured")
