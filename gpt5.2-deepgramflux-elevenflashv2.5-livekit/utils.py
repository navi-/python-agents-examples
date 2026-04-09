"""Shared utilities for the LiveKit voice agent.

This module provides:
- Phone number normalization

For LiveKit framework examples, no audio conversion or VAD is needed here —
LiveKit handles audio transport, format conversion, and voice activity
detection internally via its SIP bridge and plugin system.
"""

from __future__ import annotations

import os

import phonenumbers
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# =============================================================================
# Configuration (only constants consumed by utility functions)
# =============================================================================

DEFAULT_COUNTRY_CODE = os.getenv("DEFAULT_COUNTRY_CODE", "US")

# =============================================================================
# Phone Number Utilities
# =============================================================================


def normalize_phone_number(phone: str, default_region: str = DEFAULT_COUNTRY_CODE) -> str:
    """Normalize phone number to E.164 format (digits only, no leading +)."""
    if not phone:
        return ""

    try:
        parsed = phonenumbers.parse(phone, default_region)
        e164 = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
        return e164.lstrip("+")
    except phonenumbers.NumberParseException as e:
        logger.warning(f"Failed to parse phone number '{phone}': {e}")
        return "".join(c for c in phone if c.isdigit())
