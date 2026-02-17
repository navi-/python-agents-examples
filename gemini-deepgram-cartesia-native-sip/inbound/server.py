"""Inbound SIP server — handles incoming SIP calls + FastAPI health/status API.

Combines:
- AsyncSIPServer for SIP signaling and RTP media
- FastAPI for health checks and call monitoring
- Plivo SIP trunk auto-configuration via Zentrunk API
"""

from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime

import httpx
import plivo
import uvicorn
from fastapi import FastAPI
from loguru import logger

from inbound.agent import run_agent
from sip import AsyncSIPServer, RTPSession
from utils import (
    PLIVO_AUTH_ID,
    PLIVO_AUTH_TOKEN,
    PLIVO_PHONE_NUMBER,
    PUBLIC_URL,
    RTP_PORT_END,
    RTP_PORT_START,
    SERVER_PORT,
    SIP_HOST,
    SIP_PORT,
    normalize_phone_number,
)

# =============================================================================
# Plivo SIP Trunk Configuration
# =============================================================================

ZENTRUNK_BASE = "https://api.plivo.com/v1/Account/{auth_id}/Zentrunk"
TRUNK_NAME = "Gemini_SIP_Voice_Agent"
URI_NAME = "Gemini_SIP_Voice_Agent_URI"


def configure_plivo_sip_trunk() -> bool:
    """Configure Plivo SIP trunk to route inbound calls to our SIP server.

    This uses the Plivo Zentrunk API to:
    1. Create/update an Origination URI pointing to our server's SIP address
    2. Create/update an Inbound Trunk with that URI
    3. Associate the phone number with the trunk

    The server's public IP:port must be reachable from Plivo's network.
    Set PUBLIC_URL to your server's public IP or hostname.
    """
    if not all([PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, PLIVO_PHONE_NUMBER, PUBLIC_URL]):
        missing = []
        if not PLIVO_AUTH_ID:
            missing.append("PLIVO_AUTH_ID")
        if not PLIVO_AUTH_TOKEN:
            missing.append("PLIVO_AUTH_TOKEN")
        if not PLIVO_PHONE_NUMBER:
            missing.append("PLIVO_PHONE_NUMBER")
        if not PUBLIC_URL:
            missing.append("PUBLIC_URL")
        logger.warning(f"Skipping Plivo SIP trunk auto-config. Missing: {', '.join(missing)}")
        return False

    try:
        base_url = ZENTRUNK_BASE.format(auth_id=PLIVO_AUTH_ID)
        auth = (PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN)

        # Derive SIP origination URI from PUBLIC_URL
        # PUBLIC_URL can be an IP, hostname, or full URL
        sip_host = PUBLIC_URL.rstrip("/")
        for prefix in ("https://", "http://", "sip:"):
            if sip_host.startswith(prefix):
                sip_host = sip_host[len(prefix):]
        # Add SIP port if not already specified
        if ":" not in sip_host:
            sip_host = f"{sip_host}:{SIP_PORT}"
        origination_uri = sip_host

        # --- Step 1: Create or update Origination URI ---
        uri_uuid = _find_or_create_origination_uri(base_url, auth, origination_uri)
        if not uri_uuid:
            return False

        # --- Step 2: Create or update Inbound Trunk ---
        trunk_id = _find_or_create_inbound_trunk(base_url, auth, uri_uuid)
        if not trunk_id:
            return False

        # --- Step 3: Associate phone number with trunk ---
        # Plivo maps numbers to inbound trunks via numbers.update(app_id=trunk_id)
        client = plivo.RestClient(auth_id=PLIVO_AUTH_ID, auth_token=PLIVO_AUTH_TOKEN)
        phone_number = normalize_phone_number(PLIVO_PHONE_NUMBER)
        if not phone_number:
            logger.error(f"Invalid phone number format: {PLIVO_PHONE_NUMBER}")
            return False

        client.numbers.update(number=phone_number, app_id=trunk_id)

        logger.info(f"Plivo SIP trunk configured for +{phone_number}")
        logger.info(f"  Origination URI: {origination_uri}")
        logger.info(f"  Trunk ID: {trunk_id}")
        logger.info(f"  Trunk name: {TRUNK_NAME}")

        return True

    except plivo.exceptions.ValidationError as e:
        logger.error(f"Plivo validation error: {e}")
        return False
    except httpx.HTTPError as e:
        logger.error(f"Zentrunk API error: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to configure Plivo SIP trunk: {e}")
        return False


def _find_or_create_origination_uri(
    base_url: str, auth: tuple[str, str], origination_uri: str
) -> str:
    """Find existing Origination URI by name, or create a new one. Returns uri_uuid."""
    uri_url = f"{base_url}/URI/"

    # List existing URIs and look for ours
    resp = httpx.get(uri_url, auth=auth, timeout=15.0)
    resp.raise_for_status()
    data = resp.json()

    for uri_obj in data.get("objects", []):
        if uri_obj.get("name") == URI_NAME:
            uri_uuid = uri_obj["uri_uuid"]
            # Update the URI to point to current server address
            update_resp = httpx.post(
                f"{uri_url}{uri_uuid}/",
                auth=auth,
                json={"uri": origination_uri},
                timeout=15.0,
            )
            update_resp.raise_for_status()
            logger.info(f"Updated origination URI: {URI_NAME} → {origination_uri}")
            return uri_uuid

    # Create new URI
    create_resp = httpx.post(
        uri_url,
        auth=auth,
        json={"name": URI_NAME, "uri": origination_uri},
        timeout=15.0,
    )
    create_resp.raise_for_status()
    uri_uuid = create_resp.json().get("uri_uuid", "")
    logger.info(f"Created origination URI: {URI_NAME} → {origination_uri}")
    return uri_uuid


def _find_or_create_inbound_trunk(
    base_url: str, auth: tuple[str, str], uri_uuid: str
) -> str:
    """Find existing Inbound Trunk by name, or create a new one. Returns trunk_id."""
    trunk_url = f"{base_url}/Trunk/"

    # List existing trunks and look for ours
    resp = httpx.get(trunk_url, auth=auth, timeout=15.0)
    resp.raise_for_status()
    data = resp.json()

    for trunk_obj in data.get("objects", []):
        if trunk_obj.get("name") == TRUNK_NAME:
            trunk_id = trunk_obj["trunk_id"]
            # Update trunk to use current URI
            update_resp = httpx.post(
                f"{trunk_url}{trunk_id}/",
                auth=auth,
                json={"primary_uri_uuid": uri_uuid},
                timeout=15.0,
            )
            update_resp.raise_for_status()
            logger.info(f"Updated inbound trunk: {TRUNK_NAME}")
            return trunk_id

    # Create new inbound trunk
    create_resp = httpx.post(
        trunk_url,
        auth=auth,
        json={
            "name": TRUNK_NAME,
            "trunk_direction": "inbound",
            "primary_uri_uuid": uri_uuid,
        },
        timeout=15.0,
    )
    create_resp.raise_for_status()
    trunk_id = create_resp.json().get("trunk_id", "")
    logger.info(f"Created inbound trunk: {TRUNK_NAME}")
    return trunk_id


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(title="Gemini-Deepgram-Cartesia SIP Voice Agent (Inbound)")

# Track active calls for the status API
_active_calls: dict[str, dict] = {}
_start_time = datetime.now(UTC)

# SIP server reference (set during startup)
_sip_server: AsyncSIPServer | None = None


@app.get("/")
async def health_check():
    """Health check endpoint."""
    phone = normalize_phone_number(PLIVO_PHONE_NUMBER)
    return {
        "status": "ok",
        "service": "gemini-deepgram-cartesia-sip-inbound",
        "phone_number": f"+{phone}" if phone else "not configured",
        "uptime_seconds": (datetime.now(UTC) - _start_time).total_seconds(),
        "active_calls": len(_active_calls),
    }


@app.get("/calls")
async def list_active_calls():
    """List active SIP calls."""
    calls = []
    for call_id, info in _active_calls.items():
        calls.append({
            "call_id": call_id,
            "from_uri": info.get("from_uri", ""),
            "to_uri": info.get("to_uri", ""),
            "started_at": info.get("started_at", ""),
        })
    return {"active_calls": calls, "count": len(calls)}


# =============================================================================
# SIP Call Handler
# =============================================================================


async def on_new_call(rtp_session: RTPSession, metadata: dict) -> None:
    """Callback when SIP INVITE is answered — start voice agent."""
    call_id = metadata["call_id"]
    from_uri = metadata.get("from_uri", "")
    to_uri = metadata.get("to_uri", "")

    logger.info(f"New call: {call_id} from={from_uri} to={to_uri}")

    _active_calls[call_id] = {
        "from_uri": from_uri,
        "to_uri": to_uri,
        "started_at": datetime.now(UTC).isoformat(),
    }

    try:
        await run_agent(
            rtp_session=rtp_session,
            call_id=call_id,
            from_uri=from_uri,
            to_uri=to_uri,
        )
    finally:
        _active_calls.pop(call_id, None)
        logger.info(f"Call ended: {call_id}")


async def on_call_end(call_id: str) -> None:
    """Callback when SIP BYE is received."""
    _active_calls.pop(call_id, None)
    logger.info(f"Call terminated by remote: {call_id}")


# =============================================================================
# Server Startup
# =============================================================================


async def start_servers() -> None:
    """Start both SIP server and FastAPI concurrently."""
    global _sip_server

    _sip_server = AsyncSIPServer(
        host=SIP_HOST,
        sip_port=SIP_PORT,
        rtp_port_start=RTP_PORT_START,
        rtp_port_end=RTP_PORT_END,
        on_new_call=on_new_call,
        on_call_end=on_call_end,
    )

    await _sip_server.start()

    # Run FastAPI with uvicorn
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=SERVER_PORT,
        log_level="info",
    )
    server = uvicorn.Server(config)

    logger.info(f"FastAPI health API on port {SERVER_PORT}")
    logger.info(f"SIP server on port {SIP_PORT}")

    try:
        await server.serve()
    finally:
        await _sip_server.stop()


def main() -> None:
    """Entry point for the inbound server."""
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    logger.info("Starting Gemini-Deepgram-Cartesia SIP Inbound Server")

    if PLIVO_PHONE_NUMBER and PUBLIC_URL:
        logger.info("Configuring Plivo SIP trunk...")
        phone = normalize_phone_number(PLIVO_PHONE_NUMBER)
        if configure_plivo_sip_trunk():
            logger.info(f"Ready! Call +{phone} to test")
        else:
            logger.warning("Plivo SIP trunk auto-configuration failed. Configure manually.")
    else:
        logger.info(
            "To enable auto-configuration, set PUBLIC_URL and PLIVO_PHONE_NUMBER"
        )

    asyncio.run(start_servers())


if __name__ == "__main__":
    main()
