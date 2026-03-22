"""Standalone FastAPI server for inbound calls with LiveKit agent worker."""

from __future__ import annotations

import asyncio
import os
import threading

import plivo
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request
from fastapi.responses import Response
from livekit import api as livekit_api
from livekit.agents import cli as livekit_cli
from loguru import logger
from plivo import plivoxml

from inbound.agent import create_worker_options
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
LIVEKIT_SIP_URI = os.getenv("LIVEKIT_SIP_URI", "")

app = FastAPI(
    title="GPT-5.4-mini AssemblyAI Cartesia LiveKit Voice Agent (Inbound)",
    description=(
        "Inbound voice agent using LiveKit with AssemblyAI STT, "
        "OpenAI GPT-5.4-mini LLM, and Cartesia TTS"
    ),
    version="0.1.0",
)


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

        app_name = "GPT54mini_AAI_Cartesia_LiveKit_Agent"
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
# LiveKit Room Management
# =============================================================================


async def create_sip_participant(call_uuid: str, from_number: str) -> str | None:
    """Create a LiveKit room and dispatch a SIP participant for the call.

    Args:
        call_uuid: Plivo call UUID.
        from_number: Caller's phone number.

    Returns:
        The room name if successful, None otherwise.
    """
    if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
        logger.warning("LiveKit credentials not configured, skipping SIP dispatch")
        return None

    try:
        room_name = f"inbound-{call_uuid}"
        lk_api = livekit_api.LiveKitAPI(
            url=LIVEKIT_URL,
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET,
        )

        # Create the room
        await lk_api.room.create_room(
            livekit_api.CreateRoomRequest(name=room_name, empty_timeout=300)
        )

        logger.info(f"Created LiveKit room: {room_name} for call {call_uuid}")
        await lk_api.aclose()
        return room_name

    except Exception as e:
        logger.error(f"Failed to create LiveKit room: {e}")
        return None


# =============================================================================
# Routes
# =============================================================================


@app.get("/")
async def health_check() -> dict:
    """Health check endpoint."""
    from inbound.agent import OPENAI_MODEL

    phone = normalize_phone_number(PLIVO_PHONE_NUMBER)
    return {
        "status": "ok",
        "service": "gpt5.4mini-assemblyai-cartesia-livekit",
        "model": OPENAI_MODEL,
        "stt": "assemblyai",
        "tts": "cartesia",
        "framework": "livekit",
        "phone_number": f"+{phone}" if phone else "not configured",
    }


@app.get("/answer")
@app.post("/answer")
async def answer_webhook(
    request: Request,
    CallUUID: str = Query(default=""),
    From: str = Query(default=""),
    To: str = Query(default=""),
) -> Response:
    """Plivo answer webhook - routes call to LiveKit via SIP.

    When Plivo receives an inbound call, this webhook returns XML that
    instructs Plivo to dial into the LiveKit SIP bridge, connecting the
    caller to the voice agent running in a LiveKit room.
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

    # Create LiveKit room for this call
    room_name = await create_sip_participant(call_uuid, from_number)

    if LIVEKIT_SIP_URI and room_name:
        # Route call to LiveKit via SIP
        sip_uri = f"sip:{room_name}@{LIVEKIT_SIP_URI}"
        logger.info(f"Routing call to LiveKit SIP: {sip_uri}")

        response = plivoxml.ResponseElement()
        dial = plivoxml.DialElement()
        dial.add(plivoxml.SipElement(sip_uri))
        response.add(dial)

        return Response(content=response.to_string(), media_type="application/xml")

    # Fallback: if LiveKit SIP is not configured, return hold
    logger.warning("LiveKit SIP not configured, placing call on hold")
    response = plivoxml.ResponseElement()
    response.add(
        plivoxml.SpeakElement(
            "Please wait while we connect you to an agent.",
            voice="Polly.Joanna",
        )
    )
    response.add(plivoxml.WaitElement(length=120))

    return Response(content=response.to_string(), media_type="application/xml")


@app.post("/hangup")
async def hangup_webhook(request: Request) -> Response:
    """Plivo hangup webhook - called when a call ends."""
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
            "We're sorry, but we're experiencing technical difficulties. Please try again later.",
            voice="Polly.Joanna",
        )
    )
    response.add(plivoxml.HangupElement())

    return Response(content=response.to_string(), media_type="application/xml")


@app.get("/hold")
@app.post("/hold")
async def hold_webhook() -> Response:
    """Hold endpoint - keeps call alive silently (used during testing)."""
    response = plivoxml.ResponseElement()
    response.add(plivoxml.WaitElement(length=120))
    return Response(content=response.to_string(), media_type="application/xml")


# =============================================================================
# LiveKit Agent Worker
# =============================================================================


def _run_livekit_worker() -> None:
    """Run the LiveKit agent worker in a background thread."""
    try:
        opts = create_worker_options()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        livekit_cli.run_app(opts)
    except Exception as e:
        logger.error(f"LiveKit worker error: {e}")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run the inbound server with LiveKit agent worker."""
    logger.info(
        f"Starting GPT-5.4-mini AssemblyAI Cartesia LiveKit "
        f"Inbound Voice Agent on port {SERVER_PORT}"
    )

    if PLIVO_PHONE_NUMBER and PUBLIC_URL:
        logger.info("Configuring Plivo webhooks...")
        phone = normalize_phone_number(PLIVO_PHONE_NUMBER)
        if configure_plivo_webhooks():
            logger.info(f"Ready! Call +{phone} to test")
        else:
            logger.warning("Plivo auto-configuration failed. Configure manually.")
    else:
        logger.info("To enable auto-configuration, set PUBLIC_URL and PLIVO_PHONE_NUMBER")

    # Start LiveKit agent worker in background
    if LIVEKIT_URL and LIVEKIT_API_KEY and LIVEKIT_API_SECRET:
        logger.info("Starting LiveKit agent worker...")
        worker_thread = threading.Thread(target=_run_livekit_worker, daemon=True)
        worker_thread.start()
    else:
        logger.warning(
            "LiveKit credentials not configured. "
            "Set LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET to enable."
        )

    uvicorn.run("inbound.server:app", host="0.0.0.0", port=SERVER_PORT, log_level="info")


if __name__ == "__main__":
    main()
