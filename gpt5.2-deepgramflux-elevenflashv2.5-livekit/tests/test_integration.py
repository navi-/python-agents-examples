"""
Unit tests for GPT-5.2 LiveKit voice agent.

Tests phone normalization and CallManager logic. No external services needed.

Run:
    uv run pytest tests/test_integration.py -v
"""

from __future__ import annotations

from utils import normalize_phone_number


class TestUnitPhoneNormalization:
    """Unit tests for phone number normalization."""

    def test_normalize_e164_format(self):
        assert normalize_phone_number("+16572338892") == "16572338892"

    def test_normalize_with_spaces(self):
        assert normalize_phone_number("+1 657-233-8892") == "16572338892"

    def test_normalize_local_format(self):
        assert normalize_phone_number("(657) 233-8892", "US") == "16572338892"

    def test_normalize_empty(self):
        assert normalize_phone_number("") == ""

    def test_normalize_international(self):
        assert normalize_phone_number("+442071234567") == "442071234567"


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
