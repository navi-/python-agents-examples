"""Shared utilities for the LiveKit voice agent.

This module provides:
- Phone number normalization
- Plivo Zentrunk SIP trunk API helpers (REST, since the Plivo SDK lacks Zentrunk support)
"""

from __future__ import annotations

import os

import httpx
import phonenumbers
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

DEFAULT_COUNTRY_CODE = os.getenv("DEFAULT_COUNTRY_CODE", "US")

PLIVO_API_BASE = "https://api.plivo.com/v1/Account"


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


# =============================================================================
# Plivo Zentrunk SIP API
# =============================================================================


class PlivoZentrunk:
    """Plivo Zentrunk SIP trunking API client.

    The Plivo Python SDK does not support Zentrunk operations, so this
    uses the REST API directly via httpx.
    """

    def __init__(self, auth_id: str, auth_token: str) -> None:
        self.auth_id = auth_id
        self.auth_token = auth_token
        self.base_url = f"{PLIVO_API_BASE}/{auth_id}/Zentrunk"

    async def _request(
        self, method: str, path: str, json: dict | None = None
    ) -> dict:
        async with httpx.AsyncClient() as client:
            resp = await client.request(
                method,
                f"{self.base_url}{path}",
                json=json,
                auth=(self.auth_id, self.auth_token),
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json()

    # --- Origination URIs (where inbound calls are forwarded) ---

    async def list_origination_uris(self) -> list[dict]:
        data = await self._request("GET", "/URI/")
        return data.get("objects", [])

    async def create_origination_uri(self, name: str, uri: str) -> str:
        """Create an origination URI. Returns the uri_uuid."""
        data = await self._request("POST", "/URI/", json={"name": name, "uri": uri})
        uri_uuid = data.get("uri_uuid", "")
        logger.info(f"Created Plivo origination URI: {uri_uuid} → {uri}")
        return uri_uuid

    # --- Inbound Trunks ---

    async def list_trunks(self) -> list[dict]:
        data = await self._request("GET", "/Trunk/")
        return data.get("objects", [])

    async def create_inbound_trunk(
        self, name: str, primary_uri_uuid: str
    ) -> str:
        """Create an inbound trunk. Returns the trunk_id."""
        data = await self._request(
            "POST",
            "/Trunk/",
            json={
                "name": name,
                "trunk_direction": "inbound",
                "primary_uri_uuid": primary_uri_uuid,
            },
        )
        trunk_id = data.get("trunk_id", "")
        logger.info(f"Created Plivo inbound trunk: {trunk_id}")
        return trunk_id

    # --- Outbound Trunks ---

    async def create_credential(self, name: str, username: str, password: str) -> str:
        """Create SIP credentials. Returns the credential_uuid."""
        data = await self._request(
            "POST",
            "/Credential/",
            json={"name": name, "username": username, "password": password},
        )
        cred_uuid = data.get("credential_uuid", "")
        logger.info(f"Created Plivo SIP credential: {cred_uuid}")
        return cred_uuid

    async def list_credentials(self) -> list[dict]:
        data = await self._request("GET", "/Credential/")
        return data.get("objects", [])

    async def create_outbound_trunk(self, name: str, credential_uuid: str) -> tuple[str, str]:
        """Create an outbound trunk. Returns (trunk_id, termination_domain)."""
        data = await self._request(
            "POST",
            "/Trunk/",
            json={
                "name": name,
                "trunk_direction": "outbound",
                "credential_uuid": credential_uuid,
            },
        )
        trunk_id = data.get("trunk_id", "")
        domain = data.get("trunk_domain", "")
        logger.info(f"Created Plivo outbound trunk: {trunk_id}, domain: {domain}")
        return trunk_id, domain

    # --- Phone Number Mapping ---

    async def map_number_to_trunk(self, phone_number: str, trunk_id: str) -> bool:
        """Map a Plivo phone number to a Zentrunk inbound trunk."""
        url = f"{PLIVO_API_BASE}/{self.auth_id}/Number/{phone_number}/"
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url,
                json={"app_id": trunk_id},
                auth=(self.auth_id, self.auth_token),
                timeout=30.0,
            )
            if resp.status_code in (200, 202):
                logger.info(f"Mapped +{phone_number} to trunk {trunk_id}")
                return True
            logger.warning(f"Failed to map number to trunk: {resp.status_code} {resp.text}")
            return False
