"""Standalone FastAPI server for outbound calls via Vapi orchestrator.

Vapi handles the entire voice pipeline and connects to Plivo via SIP trunking.
This server manages outbound call initiation via Vapi API and handles webhook
events for tool execution and call lifecycle tracking.

Architecture:
    This Server -> Vapi API (initiate) -> Plivo (SIP) -> Callee
    Callee -> Plivo (SIP) -> Vapi (orchestrator) -> This Server (webhooks)
"""

from __future__ import annotations

import os
from datetime import datetime

import requests as http_requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request
from fastapi.responses import Response
from loguru import logger

from outbound.agent import (
    GPT_MODEL,
    CallManager,
    determine_outcome,
    handle_tool_calls,
    initiate_outbound_call,
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
    title="GPT-4.1 Voice Agent via Vapi (Outbound)",
    description="Outbound voice agent using Vapi with GPT-4.1, Deepgram, ElevenLabs, and Plivo",
    version="0.1.0",
)

call_manager = CallManager()


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
        "service": "gpt4.1-deepgram-elevenlabs-vapi-outbound",
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


@app.post("/outbound/call")
async def outbound_initiate(
    request: Request,
    phone_number: str = Query(default=""),
    campaign_id: str = Query(default=""),
    opening_reason: str = Query(default=""),
    objective: str = Query(default=""),
    context: str = Query(default=""),
) -> dict:
    """Initiate an outbound call via Vapi.

    Creates a call record, then uses the Vapi API to place a call.
    Vapi handles the SIP signaling through Plivo and manages the
    entire voice pipeline.
    """
    if not phone_number:
        return {"error": "phone_number is required"}

    record = call_manager.create_call(
        phone_number=phone_number,
        campaign_id=campaign_id,
        opening_reason=opening_reason,
        objective=objective,
        context=context,
    )

    try:
        server_url = f"{PUBLIC_URL}/vapi/webhook" if PUBLIC_URL else ""
        vapi_call_id = await initiate_outbound_call(record, server_url)

        call_manager.update_status(
            record.call_id,
            "ringing",
            vapi_call_id=vapi_call_id,
        )
        logger.info(
            f"Outbound call initiated: call_id={record.call_id}, "
            f"to={phone_number}, vapi_call_id={vapi_call_id}"
        )

        return {
            "call_id": record.call_id,
            "status": "ringing",
            "phone_number": phone_number,
            "vapi_call_id": vapi_call_id,
        }

    except Exception as e:
        logger.error(f"Failed to initiate outbound call: {e}")
        call_manager.update_status(record.call_id, "failed", outcome="failed")
        return {"error": str(e), "call_id": record.call_id}


@app.post("/vapi/webhook")
async def vapi_webhook(request: Request) -> dict:
    """Main Vapi webhook endpoint for outbound calls.

    Handles all Vapi server events:
    - tool-calls: Execute tools and return results
    - status-update: Track call status changes
    - end-of-call-report: Finalize call records
    - conversation-update: Real-time transcript updates
    """
    body = await request.json()
    message = body.get("message", {})
    message_type = message.get("type", "")

    if message_type == "tool-calls":
        logger.info("Processing tool calls from Vapi")
        results = await handle_tool_calls(message)
        return {"results": results}

    if message_type == "status-update":
        status = message.get("status", "")
        call = message.get("call", {})
        vapi_call_id = call.get("id", "")

        record = call_manager.get_call_by_vapi_id(vapi_call_id)
        if record:
            if status == "in-progress":
                call_manager.update_status(
                    record.call_id,
                    "connected",
                    connected_at=datetime.utcnow(),
                )
            logger.info(
                f"Call status update: call_id={record.call_id}, "
                f"vapi_status={status}"
            )
        else:
            logger.debug(f"Status update for unknown vapi_call_id={vapi_call_id}: {status}")

        return {"ok": True}

    if message_type == "end-of-call-report":
        call = message.get("call", {})
        vapi_call_id = call.get("id", "")
        duration = message.get("durationSeconds", 0)
        ended_reason = message.get("endedReason", "")
        summary = message.get("summary", "")
        artifact = message.get("artifact", {})
        recording_url = artifact.get("recordingUrl", "")
        transcript = artifact.get("transcript", "")

        record = call_manager.get_call_by_vapi_id(vapi_call_id)
        if record:
            outcome = determine_outcome(ended_reason, duration)
            call_manager.update_status(
                record.call_id,
                "completed",
                ended_at=datetime.utcnow(),
                duration=duration,
                ended_reason=ended_reason,
                outcome=outcome,
                transcript=transcript,
                recording_url=recording_url,
            )
            logger.info(
                f"Outbound call {record.call_id} completed: outcome={outcome}, "
                f"duration={duration}s, summary={summary[:100]}"
            )
            if recording_url:
                logger.info(f"Recording: {recording_url}")
        else:
            logger.debug(
                f"End-of-call report for unknown vapi_call_id={vapi_call_id}: "
                f"duration={duration}s"
            )

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
        return {"ok": True}

    if message_type == "speech-update":
        return {"ok": True}

    if message_type == "hang":
        # "hang" is a delay/lag notification from Vapi, NOT a hangup event
        logger.warning("Vapi hang (delay/lag) notification received")
        return {"ok": True}

    logger.debug(f"Unhandled Vapi event: {message_type}")
    return {"ok": True}


@app.get("/outbound/status/{call_id}")
async def outbound_status(call_id: str) -> dict:
    """Get status and details for an outbound call."""
    record = call_manager.get_call(call_id)
    if not record:
        return {"error": "Call not found"}

    return {
        "call_id": record.call_id,
        "phone_number": record.phone_number,
        "status": record.status,
        "campaign_id": record.campaign_id,
        "opening_reason": record.opening_reason,
        "objective": record.objective,
        "outcome": record.outcome,
        "duration": record.duration,
        "vapi_call_id": record.vapi_call_id,
        "transcript": record.transcript,
        "recording_url": record.recording_url,
        "created_at": record.created_at.isoformat(),
        "connected_at": record.connected_at.isoformat() if record.connected_at else None,
        "ended_at": record.ended_at.isoformat() if record.ended_at else None,
    }


@app.get("/outbound/campaign/{campaign_id}")
async def outbound_campaign(campaign_id: str) -> dict:
    """Get all calls for a campaign."""
    records = call_manager.get_calls_by_campaign(campaign_id)
    return {
        "campaign_id": campaign_id,
        "total": len(records),
        "calls": [
            {
                "call_id": r.call_id,
                "phone_number": r.phone_number,
                "status": r.status,
                "outcome": r.outcome,
                "duration": r.duration,
            }
            for r in records
        ],
    }


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run the outbound server."""
    logger.info(f"Starting GPT-4.1 Vapi Outbound Voice Agent on port {SERVER_PORT}")

    if PUBLIC_URL:
        logger.info(f"Vapi webhook URL: {PUBLIC_URL}/vapi/webhook")
    else:
        logger.warning("PUBLIC_URL not set. Outbound calls require a public webhook URL.")

    if PLIVO_PHONE_NUMBER and VAPI_PRIVATE_KEY:
        phone = normalize_phone_number(PLIVO_PHONE_NUMBER)
        if configure_vapi_sip():
            logger.info(f"SIP configured. Outbound from +{phone}")
        else:
            logger.warning("SIP auto-configuration failed. Configure manually.")

    uvicorn.run("outbound.server:app", host="0.0.0.0", port=SERVER_PORT, log_level="info")


if __name__ == "__main__":
    main()
