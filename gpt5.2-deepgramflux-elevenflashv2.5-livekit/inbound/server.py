"""FastAPI management server for inbound calls via LiveKit SIP.

On startup, auto-configures:
1. LiveKit inbound SIP trunk (with Krisp noise cancellation enabled)
2. LiveKit dispatch rule (routes inbound SIP calls to agent rooms)
3. Plivo phone number webhooks (answer/hangup URLs)

The agent worker must be started separately:
    uv run python -m inbound.agent dev

This server provides:
    /         — health check
    /answer   — Plivo answer webhook (returns SIP redirect XML)
    /hangup   — Plivo hangup webhook
"""

from __future__ import annotations

import asyncio
import os

import plivo
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request
from fastapi.responses import Response
from loguru import logger
from plivo import plivoxml

from utils import normalize_phone_number

load_dotenv()

# Server configuration
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID", "")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN", "")
PLIVO_PHONE_NUMBER = os.getenv("PLIVO_PHONE_NUMBER", "")
PUBLIC_URL = os.getenv("PUBLIC_URL", "")

# LiveKit configuration
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")

# Populated at startup by configure_livekit_sip() or from env
LIVEKIT_SIP_URI = os.getenv("LIVEKIT_SIP_URI", "")
LIVEKIT_SIP_TRUNK_ID = os.getenv("LIVEKIT_SIP_TRUNK_ID", "")

app = FastAPI(
    title="GPT-5.2 LiveKit Voice Agent (Inbound)",
    description="Inbound voice agent using LiveKit with Deepgram Flux, GPT-5.2-mini, ElevenLabs",
    version="0.1.0",
)


# =============================================================================
# LiveKit SIP Auto-Configuration
# =============================================================================


async def configure_livekit_sip() -> bool:
    """Create or reuse a LiveKit inbound SIP trunk and dispatch rule.

    Returns True if the SIP trunk is ready for inbound calls.
    Sets module-level LIVEKIT_SIP_TRUNK_ID with the trunk ID.
    """
    global LIVEKIT_SIP_TRUNK_ID

    if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
        missing = []
        if not LIVEKIT_URL:
            missing.append("LIVEKIT_URL")
        if not LIVEKIT_API_KEY:
            missing.append("LIVEKIT_API_KEY")
        if not LIVEKIT_API_SECRET:
            missing.append("LIVEKIT_API_SECRET")
        logger.warning(f"Skipping LiveKit SIP auto-config. Missing: {', '.join(missing)}")
        return False

    try:
        from livekit.api import LiveKitAPI
        from livekit.protocol.sip import (
            CreateSIPDispatchRuleRequest,
            CreateSIPInboundTrunkRequest,
            ListSIPDispatchRuleRequest,
            ListSIPInboundTrunkRequest,
            SIPDispatchRule,
            SIPDispatchRuleIndividual,
            SIPInboundTrunkInfo,
        )

        lk_api = LiveKitAPI(
            url=LIVEKIT_URL,
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET,
        )

        phone = normalize_phone_number(PLIVO_PHONE_NUMBER)
        trunk_name = "plivo-inbound-gpt52"

        # --- Inbound SIP Trunk ---
        # Check if trunk already exists
        existing_trunks = await lk_api.sip.list_sip_inbound_trunk(
            ListSIPInboundTrunkRequest()
        )
        trunk_id = ""
        for trunk in existing_trunks.items:
            if trunk.name == trunk_name:
                trunk_id = trunk.sip_trunk_id
                logger.info(f"Reusing existing inbound SIP trunk: {trunk_id}")
                break

        if not trunk_id:
            # Create new inbound trunk
            # numbers: Plivo phone numbers that will send SIP to LiveKit
            trunk_info = SIPInboundTrunkInfo(
                name=trunk_name,
                numbers=[f"+{phone}"] if phone else [],
                krisp_enabled=True,
            )
            result = await lk_api.sip.create_sip_inbound_trunk(
                CreateSIPInboundTrunkRequest(trunk=trunk_info)
            )
            trunk_id = result.sip_trunk_id
            logger.info(f"Created inbound SIP trunk: {trunk_id}")

        LIVEKIT_SIP_TRUNK_ID = trunk_id

        # --- Dispatch Rule ---
        # Check if dispatch rule already exists for this trunk
        existing_rules = await lk_api.sip.list_sip_dispatch_rule(
            ListSIPDispatchRuleRequest()
        )
        has_rule = False
        for rule in existing_rules.items:
            if trunk_id in rule.trunk_ids:
                has_rule = True
                logger.info(f"Reusing existing dispatch rule: {rule.sip_dispatch_rule_id}")
                break

        if not has_rule:
            # Create dispatch rule: route calls to individual rooms per caller
            # SIPDispatchRuleIndividual creates a unique room per call
            await lk_api.sip.create_sip_dispatch_rule(
                CreateSIPDispatchRuleRequest(
                    name=f"dispatch-{trunk_name}",
                    trunk_ids=[trunk_id],
                    rule=SIPDispatchRule(
                        dispatch_rule_individual=SIPDispatchRuleIndividual(
                            room_prefix="inbound-",
                        )
                    ),
                )
            )
            logger.info("Created SIP dispatch rule (individual rooms, prefix='inbound-')")

        await lk_api.aclose()

        logger.info(f"LiveKit SIP configured: trunk_id={trunk_id}")
        return True

    except Exception as e:
        logger.error(f"Failed to configure LiveKit SIP: {e}")
        return False


# =============================================================================
# Plivo Webhook Configuration
# =============================================================================


def configure_plivo_webhooks() -> bool:
    """Configure Plivo phone number with webhook URLs."""
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
        logger.warning(f"Skipping Plivo auto-config. Missing: {', '.join(missing)}")
        return False

    try:
        client = plivo.RestClient(auth_id=PLIVO_AUTH_ID, auth_token=PLIVO_AUTH_TOKEN)

        app_name = "GPT52_LiveKit_Agent"
        answer_url = f"{PUBLIC_URL}/answer"
        hangup_url = f"{PUBLIC_URL}/hangup"

        # Check if application already exists
        apps = client.applications.list()
        existing_app = None
        for app_obj in apps["objects"]:
            if app_obj["app_name"] == app_name:
                existing_app = app_obj
                break

        if existing_app:
            client.applications.update(
                app_id=existing_app["app_id"],
                answer_url=answer_url,
                answer_method="POST",
                hangup_url=hangup_url,
                hangup_method="POST",
            )
            app_id = existing_app["app_id"]
            logger.info(f"Updated Plivo application: {app_name}")
        else:
            response = client.applications.create(
                app_name=app_name,
                answer_url=answer_url,
                answer_method="POST",
                hangup_url=hangup_url,
                hangup_method="POST",
            )
            app_id = response["app_id"]
            logger.info(f"Created Plivo application: {app_name}")

        phone_number = normalize_phone_number(PLIVO_PHONE_NUMBER)
        if not phone_number:
            logger.error(f"Invalid phone number format: {PLIVO_PHONE_NUMBER}")
            return False

        client.numbers.update(number=phone_number, app_id=app_id)

        logger.info(f"Plivo webhooks configured for +{phone_number}")
        logger.info(f"  Answer URL: {answer_url}")
        logger.info(f"  Hangup URL: {hangup_url}")

        return True

    except plivo.exceptions.ValidationError as e:
        logger.error(f"Plivo validation error: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to configure Plivo: {e}")
        return False


# =============================================================================
# Routes
# =============================================================================


@app.get("/")
async def health_check() -> dict:
    """Health check endpoint."""
    from inbound.agent import LLM_MODEL

    phone = normalize_phone_number(PLIVO_PHONE_NUMBER)
    return {
        "status": "ok",
        "service": "gpt5.2-deepgramflux-elevenflashv2.5-livekit",
        "model": LLM_MODEL,
        "phone_number": f"+{phone}" if phone else "not configured",
        "sip_uri": LIVEKIT_SIP_URI or "not configured",
        "sip_trunk_id": LIVEKIT_SIP_TRUNK_ID or "not configured",
    }


@app.get("/answer")
@app.post("/answer")
async def answer_webhook(
    request: Request,
    CallUUID: str = Query(default=""),
    From: str = Query(default=""),
    To: str = Query(default=""),
) -> Response:
    """Plivo answer webhook — routes call to LiveKit via SIP.

    Returns Plivo XML that bridges the incoming call to a LiveKit room
    via the configured SIP trunk. LiveKit's dispatch rules then route
    the call to the appropriate agent worker.
    """
    call_uuid = CallUUID
    from_number = From
    to_number = To

    if request.method == "POST":
        try:
            form_data = await request.form()
            call_uuid = call_uuid or str(form_data.get("CallUUID", ""))
            from_number = from_number or str(form_data.get("From", ""))
            to_number = to_number or str(form_data.get("To", ""))
        except Exception:
            pass

    logger.info(f"Incoming call: CallUUID={call_uuid}, From={from_number}, To={to_number}")

    if not LIVEKIT_SIP_URI:
        logger.error("LIVEKIT_SIP_URI not configured — cannot route call")
        response = plivoxml.ResponseElement()
        response.add(
            plivoxml.SpeakElement(
                "We're sorry, the voice agent is not available right now. "
                "Please try again later.",
                voice="Polly.Joanna",
            )
        )
        response.add(plivoxml.HangupElement())
        return Response(content=response.to_string(), media_type="application/xml")

    # Route the call to LiveKit via SIP trunk
    # LiveKit's SIP bridge will create a room and the agent worker auto-joins
    response = plivoxml.ResponseElement()
    dial = plivoxml.DialElement()
    dial.add(
        plivoxml.SipElement(
            f"sip:{LIVEKIT_SIP_URI}",
            headers=f"X-Plivo-Call-UUID={call_uuid}&X-Plivo-From={from_number}",
        )
    )
    response.add(dial)

    logger.info(f"Routing call {call_uuid} to LiveKit SIP: {LIVEKIT_SIP_URI}")

    return Response(content=response.to_string(), media_type="application/xml")


@app.post("/hangup")
async def hangup_webhook(request: Request) -> Response:
    """Plivo hangup webhook — called when a call ends."""
    try:
        form_data = await request.form()
        logger.info(
            f"Call ended: CallUUID={form_data.get('CallUUID')}, "
            f"Duration={form_data.get('Duration')}s, "
            f"HangupCause={form_data.get('HangupCause')}"
        )
    except Exception as e:
        logger.warning(f"Error parsing hangup webhook: {e}")

    return Response(content="OK", media_type="text/plain")


@app.post("/fallback")
async def fallback_webhook(request: Request) -> Response:
    """Fallback webhook if primary answer webhook fails."""
    logger.warning("Fallback webhook triggered")

    response = plivoxml.ResponseElement()
    response.add(
        plivoxml.SpeakElement(
            "We're sorry, but we're experiencing technical difficulties. "
            "Please try again later.",
            voice="Polly.Joanna",
        )
    )
    response.add(plivoxml.HangupElement())

    return Response(content=response.to_string(), media_type="application/xml")


@app.get("/hold")
@app.post("/hold")
async def hold_webhook() -> Response:
    """Hold endpoint — keeps call alive silently (used during testing)."""
    response = plivoxml.ResponseElement()
    response.add(plivoxml.WaitElement(length=120))
    return Response(content=response.to_string(), media_type="application/xml")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run the inbound management server.

    On startup:
    1. Creates LiveKit inbound SIP trunk + dispatch rule (if credentials set)
    2. Configures Plivo phone number webhooks (if credentials set)
    3. Starts the FastAPI management server

    NOTE: The LiveKit agent worker must also be started separately:
        uv run python -m inbound.agent dev
    """
    logger.info(f"Starting LiveKit Inbound Voice Agent server on port {SERVER_PORT}")

    # Auto-configure LiveKit SIP trunk
    if LIVEKIT_URL and LIVEKIT_API_KEY and LIVEKIT_API_SECRET:
        logger.info("Configuring LiveKit SIP trunk...")
        asyncio.run(configure_livekit_sip())
    else:
        logger.info(
            "Set LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET "
            "to enable SIP auto-configuration"
        )

    # Auto-configure Plivo webhooks
    if PLIVO_PHONE_NUMBER and PUBLIC_URL:
        logger.info("Configuring Plivo webhooks...")
        phone = normalize_phone_number(PLIVO_PHONE_NUMBER)
        if configure_plivo_webhooks():
            logger.info(f"Ready! Call +{phone} to test")
        else:
            logger.warning("Plivo auto-configuration failed. Configure manually.")
    else:
        logger.info("To enable auto-configuration, set PUBLIC_URL and PLIVO_PHONE_NUMBER")

    if not LIVEKIT_SIP_URI:
        logger.warning(
            "LIVEKIT_SIP_URI not set. Set it to your LiveKit SIP endpoint "
            "(e.g., <trunk-id>.sip.livekit.cloud) so /answer can route calls."
        )

    logger.info(
        "Remember to start the agent worker: uv run python -m inbound.agent dev"
    )

    uvicorn.run("inbound.server:app", host="0.0.0.0", port=SERVER_PORT, log_level="info")


if __name__ == "__main__":
    main()
