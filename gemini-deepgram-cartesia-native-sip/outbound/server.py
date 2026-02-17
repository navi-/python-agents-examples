"""Outbound SIP server — FastAPI API to trigger outbound calls via SIP.

Provides API endpoints to:
- Initiate outbound calls through a SIP trunk
- Monitor call status
- List active/completed calls

Includes Plivo SIP trunk auto-configuration:
- IP Access Control List (whitelist server's public IP)
- Outbound Trunk creation with IP ACL authentication
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from dataclasses import asdict
from datetime import UTC, datetime

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel

from inbound.agent import CartesiaTTS, GeminiLLM
from outbound.agent import CallManager, run_agent
from sip import SIPClient
from utils import (
    PLIVO_AUTH_ID,
    PLIVO_AUTH_TOKEN,
    PLIVO_FROM_NUMBER,
    PLIVO_SIP_TRUNK_HOST,
    PUBLIC_IP,
    PUBLIC_URL,
    SERVER_PORT,
    cartesia_to_plivo,
    normalize_phone_number,
)

# =============================================================================
# Plivo Outbound SIP Trunk Configuration
# =============================================================================

ZENTRUNK_BASE = "https://api.plivo.com/v1/Account/{auth_id}/Zentrunk"
OUTBOUND_TRUNK_NAME = "Gemini_SIP_Voice_Agent_Outbound"
IPACL_NAME = "Gemini_SIP_Voice_Agent_ACL"


def _get_public_ip() -> str:
    """Detect the server's public IP address.

    Tries to extract IP from PUBLIC_URL first, then falls back to an
    external service to determine the public IP.
    """
    # Try PUBLIC_URL if it looks like an IP address
    if PUBLIC_URL:
        host = PUBLIC_URL.rstrip("/")
        for prefix in ("https://", "http://", "sip:"):
            if host.startswith(prefix):
                host = host[len(prefix):]
        # Strip port if present
        if ":" in host:
            host = host.split(":")[0]
        # Check if it's an IP (not a hostname like ngrok)
        parts = host.split(".")
        if len(parts) == 4 and all(p.isdigit() for p in parts):
            return host

    # Auto-detect public IP
    try:
        resp = httpx.get("https://api.ipify.org", timeout=10.0)
        resp.raise_for_status()
        return resp.text.strip()
    except Exception as e:
        logger.warning(f"Could not auto-detect public IP: {e}")
        return ""


def configure_plivo_outbound_trunk() -> bool:
    """Configure Plivo outbound SIP trunk with IP ACL.

    This uses the Plivo Zentrunk API to:
    1. Create/update an IP Access Control List with our server's public IP
    2. Create/update an Outbound Trunk authenticated via that IP ACL

    The server's public IP must be whitelisted so Plivo accepts our SIP INVITEs.
    """
    if not all([PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN]):
        missing = []
        if not PLIVO_AUTH_ID:
            missing.append("PLIVO_AUTH_ID")
        if not PLIVO_AUTH_TOKEN:
            missing.append("PLIVO_AUTH_TOKEN")
        logger.warning(
            f"Skipping outbound trunk config. Missing: {', '.join(missing)}"
        )
        return False

    public_ip = _get_public_ip()
    if not public_ip:
        logger.warning(
            "Cannot configure outbound trunk: unable to determine public IP. "
            "Set PUBLIC_URL to your server's public IP address."
        )
        return False

    try:
        base_url = ZENTRUNK_BASE.format(auth_id=PLIVO_AUTH_ID)
        auth = (PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN)

        # --- Step 1: Create or update IP ACL with our public IP ---
        ipacl_uuid = _find_or_create_ip_acl(base_url, auth, public_ip)
        if not ipacl_uuid:
            return False

        # --- Step 2: Create or update Outbound Trunk with IP ACL ---
        trunk_id = _find_or_create_outbound_trunk(base_url, auth, ipacl_uuid)
        if not trunk_id:
            return False

        logger.info("Plivo outbound SIP trunk configured")
        logger.info(f"  Public IP (whitelisted): {public_ip}")
        logger.info(f"  IP ACL: {IPACL_NAME} ({ipacl_uuid})")
        logger.info(f"  Trunk: {OUTBOUND_TRUNK_NAME} ({trunk_id})")

        return True

    except httpx.HTTPError as e:
        logger.error(f"Zentrunk API error: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to configure outbound trunk: {e}")
        return False


def _find_or_create_ip_acl(
    base_url: str, auth: tuple[str, str], public_ip: str
) -> str:
    """Find existing IP ACL by name, or create a new one. Returns ipacl_uuid."""
    acl_url = f"{base_url}/IPAccessControlList/"

    # List existing IP ACLs
    resp = httpx.get(acl_url, auth=auth, timeout=15.0)
    resp.raise_for_status()
    data = resp.json()

    for acl_obj in data.get("objects", []):
        if acl_obj.get("name") == IPACL_NAME:
            ipacl_uuid = acl_obj["ipacl_uuid"]
            # Update IP ACL with current server IP
            update_resp = httpx.post(
                f"{acl_url}{ipacl_uuid}/",
                auth=auth,
                json={"ip_addresses": [public_ip]},
                timeout=15.0,
            )
            update_resp.raise_for_status()
            logger.info(f"Updated IP ACL: {IPACL_NAME} → {public_ip}")
            return ipacl_uuid

    # Create new IP ACL
    create_resp = httpx.post(
        acl_url,
        auth=auth,
        json={"name": IPACL_NAME, "ip_addresses": [public_ip]},
        timeout=15.0,
    )
    create_resp.raise_for_status()
    ipacl_uuid = create_resp.json().get("ipacl_uuid", "")
    logger.info(f"Created IP ACL: {IPACL_NAME} → {public_ip}")
    return ipacl_uuid


def _find_or_create_outbound_trunk(
    base_url: str, auth: tuple[str, str], ipacl_uuid: str
) -> str:
    """Find existing Outbound Trunk by name, or create a new one. Returns trunk_id."""
    trunk_url = f"{base_url}/Trunk/"

    # List existing trunks
    resp = httpx.get(trunk_url, auth=auth, timeout=15.0)
    resp.raise_for_status()
    data = resp.json()

    for trunk_obj in data.get("objects", []):
        if trunk_obj.get("name") == OUTBOUND_TRUNK_NAME:
            trunk_id = trunk_obj["trunk_id"]
            # Trunk already exists — IP ACL is updated separately
            logger.info(f"Found existing outbound trunk: {OUTBOUND_TRUNK_NAME}")
            return trunk_id

    # Create new outbound trunk
    create_resp = httpx.post(
        trunk_url,
        auth=auth,
        json={
            "name": OUTBOUND_TRUNK_NAME,
            "trunk_direction": "outbound",
            "ipacl_uuid": ipacl_uuid,
        },
        timeout=15.0,
    )
    create_resp.raise_for_status()
    trunk_id = create_resp.json().get("trunk_id", "")
    logger.info(f"Created outbound trunk: {OUTBOUND_TRUNK_NAME}")
    return trunk_id


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(title="Gemini-Deepgram-Cartesia SIP Voice Agent (Outbound)")

_call_manager = CallManager()
_sip_client: SIPClient | None = None
_start_time = datetime.now(UTC)
_background_tasks: set[asyncio.Task] = set()


class OutboundCallRequest(BaseModel):
    """Request body for initiating an outbound call."""

    phone_number: str
    campaign_id: str = ""
    opening_reason: str = ""
    objective: str = ""
    context: str = ""


@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "gemini-deepgram-cartesia-sip-outbound",
        "uptime_seconds": (datetime.now(UTC) - _start_time).total_seconds(),
        "active_calls": len(_call_manager.get_active_calls()),
    }


@app.post("/outbound/call")
async def outbound_initiate(request: OutboundCallRequest):
    """Initiate an outbound call through SIP trunk."""
    if not _sip_client:
        raise HTTPException(status_code=503, detail="SIP client not initialized")

    phone = normalize_phone_number(request.phone_number)
    if not phone:
        raise HTTPException(status_code=400, detail="Invalid phone number")

    # Create call record
    record = _call_manager.create_call(
        phone_number=phone,
        campaign_id=request.campaign_id,
        opening_reason=request.opening_reason,
        objective=request.objective,
        context=request.context,
    )

    # Start outbound call in background
    _background_tasks.add(
        asyncio.create_task(_place_call(record.call_id, phone, record))
    )

    return {
        "call_id": record.call_id,
        "phone_number": phone,
        "status": "initiating",
    }


@app.get("/outbound/calls")
async def list_calls():
    """List active outbound calls."""
    active = _call_manager.get_active_calls()
    return {
        "active_calls": [
            {
                "call_id": c.call_id,
                "phone_number": c.phone_number,
                "status": c.status,
                "created_at": c.created_at.isoformat(),
            }
            for c in active
        ],
        "count": len(active),
    }


@app.get("/outbound/call/{call_id}")
async def get_call(call_id: str):
    """Get status of a specific outbound call."""
    record = _call_manager.get_call(call_id)
    if not record:
        raise HTTPException(status_code=404, detail="Call not found")
    data = asdict(record)
    # Convert datetimes to ISO strings
    for key in ("created_at", "connected_at", "ended_at"):
        if data[key] is not None:
            data[key] = data[key].isoformat()
    return data


# =============================================================================
# Call Placement
# =============================================================================


async def _pre_generate_greeting(
    system_prompt: str, initial_message: str
) -> tuple[str, bytes]:
    """Pre-generate greeting text and audio during ring time.

    Returns (greeting_text, greeting_pcm8k). On failure returns ("", b"").
    """
    tts = CartesiaTTS()
    try:
        llm = GeminiLLM(system_prompt)
        await tts.connect()

        greeting_text = await llm.generate_response(initial_message)
        if not greeting_text:
            return "", b""

        tts_audio = await tts.synthesize(greeting_text)
        if not tts_audio:
            return greeting_text, b""

        pcm_8k = cartesia_to_plivo(tts_audio)
        logger.info(
            f"Greeting pre-generated ({len(pcm_8k)} bytes): "
            f"{greeting_text[:80]}..."
        )
        return greeting_text, pcm_8k

    except Exception as e:
        logger.warning(f"Greeting pre-generation failed: {e}")
        return "", b""
    finally:
        await tts.close()


async def _place_call(call_id: str, phone: str, record) -> None:
    """Place an outbound call via SIP trunk and run the agent.

    Pre-generates the greeting audio concurrently with the SIP INVITE so
    that when the callee answers, the greeting plays immediately (0ms latency).
    """
    to_uri = f"sip:{phone}@{PLIVO_SIP_TRUNK_HOST}"
    from_uri = f"sip:{PLIVO_FROM_NUMBER}@{PLIVO_SIP_TRUNK_HOST}"

    _call_manager.update_status(call_id, "ringing")
    logger.info(f"Placing outbound call {call_id} to {to_uri}")

    try:
        # Start greeting generation AND SIP INVITE concurrently.
        # Ring time (5-15s) >> generation time (~3-4s), so greeting is
        # ready by the time the callee answers.
        greeting_task = asyncio.create_task(
            _pre_generate_greeting(record.system_prompt, record.initial_message)
        )

        rtp_session, sip_call_id = await _sip_client.call(
            to_uri=to_uri,
            from_uri=from_uri,
            timeout=30.0,
        )

        _call_manager.update_status(
            call_id, "connected",
            sip_call_id=sip_call_id,
            connected_at=datetime.now(UTC),
        )

        # Get pre-generated greeting (should be ready; waits if still generating)
        greeting_text, greeting_audio = await greeting_task

        await run_agent(
            rtp_session=rtp_session,
            call_id=call_id,
            from_uri=from_uri,
            to_uri=to_uri,
            system_prompt=record.system_prompt,
            initial_message=record.initial_message,
            greeting_audio=greeting_audio,
            greeting_text=greeting_text,
        )

        _call_manager.update_status(
            call_id, "completed",
            ended_at=datetime.now(UTC),
            outcome="success",
        )

    except TimeoutError:
        _call_manager.update_status(
            call_id, "no_answer",
            ended_at=datetime.now(UTC),
            outcome="no_answer",
        )
        logger.warning(f"Outbound call {call_id} — no answer")

    except Exception as e:
        _call_manager.update_status(
            call_id, "failed",
            ended_at=datetime.now(UTC),
            outcome="failed",
            hangup_cause=str(e),
        )
        logger.error(f"Outbound call {call_id} failed: {e}")

    finally:
        # Hang up if still connected
        if _sip_client and record.sip_call_id:
            with contextlib.suppress(Exception):
                await _sip_client.hangup(record.sip_call_id)


# =============================================================================
# Server Startup
# =============================================================================


async def start_servers() -> None:
    """Start SIP client and FastAPI server."""
    global _sip_client

    import socket

    # Determine local IP for SIP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "127.0.0.1"

    # Use PUBLIC_IP for SDP/SIP headers (NAT traversal)
    # Falls back to auto-detected public IP, then local IP
    public_ip = PUBLIC_IP or _get_public_ip() or local_ip
    logger.info(f"Local IP: {local_ip}, Public IP for SDP: {public_ip}")

    _sip_client = SIPClient(
        local_ip=local_ip,
        local_port=5080,
        trunk_host=PLIVO_SIP_TRUNK_HOST,
        trunk_port=5060,
        public_ip=public_ip,
    )
    await _sip_client.start()

    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=SERVER_PORT,
        log_level="info",
    )
    server = uvicorn.Server(config)

    logger.info(f"FastAPI outbound API on port {SERVER_PORT}")
    logger.info(f"SIP client ready, trunk={PLIVO_SIP_TRUNK_HOST}")

    try:
        await server.serve()
    finally:
        await _sip_client.stop()


def main() -> None:
    """Entry point for the outbound server."""
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    logger.info("Starting Gemini-Deepgram-Cartesia SIP Outbound Server")

    # Configure Plivo outbound trunk with IP whitelisting
    if PLIVO_AUTH_ID and PLIVO_AUTH_TOKEN:
        logger.info("Configuring Plivo outbound SIP trunk...")
        if configure_plivo_outbound_trunk():
            logger.info("Outbound trunk ready")
        else:
            logger.warning(
                "Outbound trunk auto-configuration failed. Configure manually."
            )

    asyncio.run(start_servers())


if __name__ == "__main__":
    main()
