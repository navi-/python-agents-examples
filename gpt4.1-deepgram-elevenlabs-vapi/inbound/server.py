"""Standalone FastAPI server for inbound calls via Vapi orchestrator.

Vapi handles the entire voice pipeline (Deepgram STT -> GPT-4.1 -> ElevenLabs TTS)
and connects to Plivo via SIP trunking. This server receives webhook events from
Vapi for dynamic assistant configuration, tool execution, and call lifecycle events.

Architecture:
    Caller -> Plivo (SIP) -> Vapi (orchestrator) -> This Server (webhooks)
"""

from __future__ import annotations

import os

import requests as http_requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import Response
from loguru import logger

from inbound.agent import (
    GPT_MODEL,
    build_assistant_config,
    handle_tool_calls,
)
from utils import normalize_phone_number

load_dotenv()

# Server configuration
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID", "")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN", "")
PLIVO_PHONE_NUMBER = os.getenv("PLIVO_PHONE_NUMBER", "")
PUBLIC_URL = os.getenv("PUBLIC_URL", "")
VAPI_PRIVATE_KEY = os.getenv("VAPI_PRIVATE_KEY", "")

# Vapi SIP trunking constants
VAPI_SIGNALING_IPS = ["44.229.228.186", "44.238.177.138"]
VAPI_SIP_URI = "sip.vapi.ai;transport=udp"
PLIVO_BASE = f"https://api.plivo.com/v1/Account/{PLIVO_AUTH_ID}"
VAPI_BASE = "https://api.vapi.ai"

app = FastAPI(
    title="GPT-4.1 Voice Agent via Vapi (Inbound)",
    description="Inbound voice agent using Vapi with GPT-4.1, Deepgram, ElevenLabs, and Plivo",
    version="0.1.0",
)


# =============================================================================
# Vapi SIP Trunk Auto-Configuration
# =============================================================================


def _plivo_request(
    method: str, endpoint: str, json_data: dict | None = None
) -> http_requests.Response:
    """Make an authenticated request to the Plivo Zentrunk API."""
    url = f"{PLIVO_BASE}/{endpoint}"
    return http_requests.request(
        method, url, auth=(PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN), json=json_data, timeout=30
    )


def _vapi_request(
    method: str, endpoint: str, json_data: dict | None = None
) -> http_requests.Response:
    """Make an authenticated request to the Vapi API."""
    url = f"{VAPI_BASE}/{endpoint}"
    return http_requests.request(
        method,
        url,
        headers={
            "Authorization": f"Bearer {VAPI_PRIVATE_KEY}",
            "Content-Type": "application/json",
        },
        json=json_data,
        timeout=30,
    )


def configure_vapi_sip() -> bool:
    """Idempotent SIP trunk setup for Plivo <-> Vapi.

    Steps (each checks-before-creating):
    1. Plivo IP ACL — whitelist Vapi signaling IPs
    2. Plivo outbound trunk — Vapi -> Plivo -> PSTN
    3. Plivo origination URI — points to Vapi SIP endpoint
    4. Plivo inbound trunk — PSTN -> Plivo -> Vapi
    5. Assign phone number to inbound trunk
    6. Vapi SIP credential — register Plivo trunk in Vapi
    7. Vapi phone number import — BYO number
    """
    missing = []
    if not PLIVO_AUTH_ID:
        missing.append("PLIVO_AUTH_ID")
    if not PLIVO_AUTH_TOKEN:
        missing.append("PLIVO_AUTH_TOKEN")
    if not PLIVO_PHONE_NUMBER:
        missing.append("PLIVO_PHONE_NUMBER")
    if not VAPI_PRIVATE_KEY:
        missing.append("VAPI_PRIVATE_KEY")
    if missing:
        logger.warning(f"Skipping SIP auto-config. Missing: {', '.join(missing)}")
        return False

    try:
        # Step 1: IP ACL
        resp = _plivo_request("GET", "Zentrunk/IPAccessControlList/")
        ipacl_uuid = None
        if resp.status_code == 200:
            for acl in resp.json().get("objects", []):
                if acl.get("name") == "Vapi Signaling IPs":
                    ipacl_uuid = acl.get("ipacl_uuid")
                    logger.info(f"IP ACL already exists: {ipacl_uuid}")
                    break
        if not ipacl_uuid:
            resp = _plivo_request(
                "POST",
                "Zentrunk/IPAccessControlList/",
                {"name": "Vapi Signaling IPs", "ip_addresses": VAPI_SIGNALING_IPS},
            )
            if resp.status_code not in (200, 201):
                logger.error(f"Failed to create IP ACL: {resp.status_code} {resp.text}")
                return False
            ipacl_uuid = resp.json().get("ipacl_uuid", "")
            logger.info(f"Created IP ACL: {ipacl_uuid}")

        # Step 2: Outbound trunk
        resp = _plivo_request("GET", "Zentrunk/Trunk/")
        outbound_trunk_id = None
        if resp.status_code == 200:
            for trunk in resp.json().get("objects", []):
                if trunk.get("name") == "Vapi Outbound":
                    outbound_trunk_id = trunk.get("trunk_id")
                    logger.info(f"Outbound trunk already exists: {outbound_trunk_id}")
                    break
        if not outbound_trunk_id:
            resp = _plivo_request(
                "POST",
                "Zentrunk/Trunk/",
                {
                    "name": "Vapi Outbound",
                    "trunk_direction": "outbound",
                    "trunk_status": "enabled",
                    "secure": False,
                    "ipacl_uuid": ipacl_uuid,
                },
            )
            if resp.status_code not in (200, 201):
                logger.error(
                    f"Failed to create outbound trunk: {resp.status_code} {resp.text}"
                )
                return False
            outbound_trunk_id = resp.json().get("trunk_id", "")
            logger.info(f"Created outbound trunk: {outbound_trunk_id}")

        # Step 3: Origination URI
        resp = _plivo_request("GET", "Zentrunk/URI/")
        uri_uuid = None
        if resp.status_code == 200:
            for uri in resp.json().get("objects", []):
                if VAPI_SIP_URI in uri.get("uri", ""):
                    uri_uuid = uri.get("uri_uuid")
                    logger.info(f"Origination URI already exists: {uri_uuid}")
                    break
        if not uri_uuid:
            resp = _plivo_request(
                "POST",
                "Zentrunk/URI/",
                {"name": "Vapi Primary", "uri": VAPI_SIP_URI},
            )
            if resp.status_code not in (200, 201):
                logger.error(
                    f"Failed to create origination URI: {resp.status_code} {resp.text}"
                )
                return False
            uri_uuid = resp.json().get("uri_uuid", "")
            logger.info(f"Created origination URI: {uri_uuid}")

        # Step 4: Inbound trunk
        resp = _plivo_request("GET", "Zentrunk/Trunk/")
        inbound_trunk_id = None
        if resp.status_code == 200:
            for trunk in resp.json().get("objects", []):
                if trunk.get("name") == "Vapi Inbound":
                    inbound_trunk_id = trunk.get("trunk_id")
                    logger.info(f"Inbound trunk already exists: {inbound_trunk_id}")
                    break
        if not inbound_trunk_id:
            resp = _plivo_request(
                "POST",
                "Zentrunk/Trunk/",
                {
                    "name": "Vapi Inbound",
                    "trunk_direction": "inbound",
                    "trunk_status": "enabled",
                    "primary_uri_uuid": uri_uuid,
                },
            )
            if resp.status_code not in (200, 201):
                logger.error(
                    f"Failed to create inbound trunk: {resp.status_code} {resp.text}"
                )
                return False
            inbound_trunk_id = resp.json().get("trunk_id", "")
            logger.info(f"Created inbound trunk: {inbound_trunk_id}")

        # Step 5: Assign phone number to inbound trunk
        number = PLIVO_PHONE_NUMBER.lstrip("+")
        resp = _plivo_request("POST", f"Number/{number}/", {"app_id": inbound_trunk_id})
        if resp.status_code in (200, 201, 202):
            logger.info(f"Assigned {PLIVO_PHONE_NUMBER} to inbound trunk")
        else:
            logger.warning(
                f"Could not assign phone to trunk: {resp.status_code} {resp.text}. "
                "You may need to assign it manually in Plivo Console."
            )

        # Step 6: Vapi SIP credential
        resp = _vapi_request("GET", "credential")
        credential_id = None
        if resp.status_code == 200:
            for cred in resp.json():
                if cred.get("name") == "Plivo Zentrunk":
                    credential_id = cred.get("id")
                    logger.info(f"Vapi SIP credential already exists: {credential_id}")
                    break
        if not credential_id:
            sip_domain = f"{outbound_trunk_id}.zt.plivo.com"
            resp = _vapi_request(
                "POST",
                "credential",
                {
                    "provider": "byo-sip-trunk",
                    "name": "Plivo Zentrunk",
                    "gateways": [{"ip": sip_domain, "inboundEnabled": False}],
                },
            )
            if resp.status_code not in (200, 201):
                logger.error(
                    f"Failed to register SIP credential: {resp.status_code} {resp.text}"
                )
                return False
            credential_id = resp.json().get("id", "")
            logger.info(f"Registered Vapi SIP credential: {credential_id}")

        # Step 7: Vapi phone number import
        resp = _vapi_request("GET", "phone-number")
        phone_imported = False
        if resp.status_code == 200:
            for pn in resp.json():
                if pn.get("number") == PLIVO_PHONE_NUMBER:
                    phone_imported = True
                    logger.info(
                        f"Phone number already imported in Vapi: {pn.get('id')}"
                    )
                    break
        if not phone_imported:
            resp = _vapi_request(
                "POST",
                "phone-number",
                {
                    "provider": "byo-phone-number",
                    "number": PLIVO_PHONE_NUMBER,
                    "credentialId": credential_id,
                    "name": "Plivo SIP Number",
                    "numberE164CheckEnabled": False,
                },
            )
            if resp.status_code not in (200, 201):
                logger.error(
                    f"Failed to import phone number: {resp.status_code} {resp.text}"
                )
                return False
            logger.info(
                f"Imported phone number into Vapi: {resp.json().get('id', '')}"
            )

        return True

    except Exception as e:
        logger.error(f"SIP auto-configuration failed: {e}")
        return False


# =============================================================================
# Routes
# =============================================================================


@app.get("/")
async def health_check() -> dict:
    """Health check endpoint."""
    phone = normalize_phone_number(PLIVO_PHONE_NUMBER)
    return {
        "status": "ok",
        "service": "gpt4.1-deepgram-elevenlabs-vapi-inbound",
        "model": GPT_MODEL,
        "orchestrator": "vapi",
        "phone_number": f"+{phone}" if phone else "not configured",
    }


@app.get("/hold")
@app.post("/hold")
async def hold_webhook() -> Response:
    """Hold endpoint — keeps call alive silently (used during testing)."""
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response><Wait length=\"120\" /></Response>"
    )
    return Response(content=xml, media_type="application/xml")


@app.post("/vapi/webhook")
async def vapi_webhook(request: Request) -> dict:
    """Main Vapi webhook endpoint.

    Handles all Vapi server events:
    - assistant-request: Return dynamic assistant config for inbound calls
    - tool-calls: Execute tools and return results
    - status-update: Log call status changes
    - end-of-call-report: Log call completion details
    - conversation-update: Real-time transcript updates
    """
    body = await request.json()
    message = body.get("message", {})
    message_type = message.get("type", "")

    if message_type == "assistant-request":
        # Vapi is asking for assistant configuration for an inbound call
        call = message.get("call", {})
        from_number = call.get("customer", {}).get("number", "")
        logger.info(f"Assistant request for inbound call from {from_number}")

        server_url = f"{PUBLIC_URL}/vapi/webhook" if PUBLIC_URL else ""
        assistant_config = build_assistant_config(server_url=server_url)

        return {"assistant": assistant_config}

    if message_type == "tool-calls":
        logger.info("Processing tool calls from Vapi")
        results = await handle_tool_calls(message)
        return {"results": results}

    if message_type == "status-update":
        status = message.get("status", "")
        call = message.get("call", {})
        call_id = call.get("id", "")
        logger.info(f"Call status update: call_id={call_id}, status={status}")
        return {"ok": True}

    if message_type == "end-of-call-report":
        call = message.get("call", {})
        call_id = call.get("id", "")
        duration = message.get("durationSeconds", 0)
        ended_reason = message.get("endedReason", "")
        summary = message.get("summary", "")
        artifact = message.get("artifact", {})
        recording_url = artifact.get("recordingUrl", "")
        transcript = artifact.get("transcript", "")
        logger.info(
            f"Call ended: call_id={call_id}, duration={duration}s, "
            f"reason={ended_reason}, summary={summary[:100]}"
        )
        if recording_url:
            logger.info(f"Recording: {recording_url}")
        if transcript:
            logger.info(f"Transcript length: {len(transcript)} chars")
        return {"ok": True}

    if message_type == "transcript":
        role = message.get("role", "")
        text = message.get("transcript", "")
        logger.debug(f"Transcript [{role}]: {text[:100]}")
        return {"ok": True}

    if message_type == "user-interrupted":
        logger.debug("User interrupted assistant speech")
        return {"ok": True}

    if message_type == "conversation-update":
        # Real-time transcript — useful for logging or analytics
        return {"ok": True}

    if message_type == "speech-update":
        return {"ok": True}

    if message_type == "hang":
        # "hang" is a delay/lag notification from Vapi, NOT a hangup event
        logger.warning("Vapi hang (delay/lag) notification received")
        return {"ok": True}

    logger.debug(f"Unhandled Vapi event: {message_type}")
    return {"ok": True}


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run the inbound server."""
    logger.info(f"Starting GPT-4.1 Vapi Inbound Voice Agent on port {SERVER_PORT}")

    if PUBLIC_URL:
        logger.info(f"Vapi webhook URL: {PUBLIC_URL}/vapi/webhook")
    else:
        logger.warning(
            "PUBLIC_URL not set. Use ngrok or similar to expose this server, "
            "then set PUBLIC_URL and restart."
        )

    if PLIVO_PHONE_NUMBER and VAPI_PRIVATE_KEY:
        phone = normalize_phone_number(PLIVO_PHONE_NUMBER)
        if configure_vapi_sip():
            logger.info(f"Ready! Call +{phone} to test")
        else:
            logger.warning("SIP auto-configuration failed. Configure manually.")

    uvicorn.run("inbound.server:app", host="0.0.0.0", port=SERVER_PORT, log_level="info")


if __name__ == "__main__":
    main()
