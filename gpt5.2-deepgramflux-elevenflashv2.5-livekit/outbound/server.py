"""FastAPI management server for outbound calls via LiveKit SIP.

Handles outbound call initiation via LiveKit's SIP API, Plivo webhook
integration, and call lifecycle management.

The agent worker must be started separately:
    uv run python -m outbound.agent dev
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request
from fastapi.responses import Response
from loguru import logger
from plivo import plivoxml

from outbound.agent import CallManager, determine_outcome
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
LIVEKIT_SIP_TRUNK_ID = os.getenv("LIVEKIT_SIP_TRUNK_ID", "")

app = FastAPI(
    title="GPT-5.2 LiveKit Voice Agent (Outbound)",
    description="Outbound voice agent using LiveKit with Deepgram Flux, GPT-5.2-mini, ElevenLabs",
    version="0.1.0",
)

call_manager = CallManager()


# =============================================================================
# Routes
# =============================================================================


@app.get("/")
async def health_check() -> dict:
    """Health check endpoint."""
    phone = normalize_phone_number(PLIVO_PHONE_NUMBER)
    return {
        "status": "ok",
        "service": "gpt5.2-deepgramflux-elevenflashv2.5-livekit-outbound",
        "phone_number": f"+{phone}" if phone else "not configured",
        "livekit_url": LIVEKIT_URL or "not configured",
    }


@app.post("/outbound/call")
async def outbound_initiate(
    request: Request,
    phone_number: str = Query(default=""),
    campaign_id: str = Query(default=""),
    opening_reason: str = Query(default=""),
    objective: str = Query(default=""),
    context: str = Query(default=""),
) -> dict:
    """Initiate an outbound call via LiveKit SIP.

    Creates a call record, then uses LiveKit's SIP API to place a call
    through the configured SIP trunk (Plivo). The outbound agent worker
    auto-joins the room when the SIP participant connects.
    """
    if not phone_number:
        return {"error": "phone_number is required"}

    if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
        return {"error": "LiveKit credentials not configured"}

    if not LIVEKIT_SIP_TRUNK_ID:
        return {"error": "LIVEKIT_SIP_TRUNK_ID not configured"}

    record = call_manager.create_call(
        phone_number=phone_number,
        campaign_id=campaign_id,
        opening_reason=opening_reason,
        objective=objective,
        context=context,
    )

    try:
        from livekit.api import LiveKitAPI
        from livekit.protocol.sip import CreateSIPParticipantRequest

        lk_api = LiveKitAPI(
            url=LIVEKIT_URL,
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET,
        )

        to_number = normalize_phone_number(phone_number)
        room_name = f"outbound-{record.call_id}"

        # Store metadata in the room for the agent to read
        room_metadata = json.dumps({
            "call_id": record.call_id,
            "system_prompt": record.system_prompt,
            "initial_message": record.initial_message,
            "is_outbound": True,
        })

        # Create SIP participant — LiveKit dials out through the SIP trunk
        sip_participant = await lk_api.sip.create_sip_participant(
            CreateSIPParticipantRequest(
                sip_trunk_id=LIVEKIT_SIP_TRUNK_ID,
                sip_call_to=f"+{to_number}",
                room_name=room_name,
                participant_identity=f"sip-{to_number}",
                participant_name=f"Caller +{to_number}",
                room_meta=room_metadata,
            )
        )

        call_manager.update_status(
            record.call_id, "ringing",
            livekit_room_name=room_name,
            plivo_request_uuid=getattr(sip_participant, "sip_call_id", ""),
        )

        logger.info(
            f"Outbound call initiated via LiveKit SIP: call_id={record.call_id}, "
            f"to=+{to_number}, room={room_name}"
        )

        await lk_api.aclose()

        return {
            "call_id": record.call_id,
            "status": "ringing",
            "phone_number": phone_number,
            "room_name": room_name,
        }

    except Exception as e:
        logger.error(f"Failed to initiate outbound call: {e}")
        call_manager.update_status(record.call_id, "failed", outcome="failed")
        return {"error": str(e), "call_id": record.call_id}


@app.post("/outbound/hangup")
async def outbound_hangup_webhook(request: Request) -> Response:
    """Plivo webhook when an outbound call ends."""
    try:
        form_data = await request.form()
        call_uuid = str(form_data.get("CallUUID", ""))
        duration = int(form_data.get("Duration", 0) or 0)
        hangup_cause = str(form_data.get("HangupCause", ""))

        logger.info(
            f"Outbound call ended: CallUUID={call_uuid}, "
            f"Duration={duration}s, HangupCause={hangup_cause}"
        )

        # Find and update the call record
        for record in call_manager.get_active_calls():
            if (
                record.plivo_call_uuid == call_uuid
                or record.plivo_request_uuid == call_uuid
            ):
                outcome = determine_outcome(hangup_cause, duration)
                call_manager.update_status(
                    record.call_id, "completed",
                    ended_at=datetime.utcnow(),
                    duration=duration,
                    hangup_cause=hangup_cause,
                    outcome=outcome,
                )
                logger.info(
                    f"Outbound call {record.call_id} completed: outcome={outcome}"
                )
                break
    except Exception as e:
        logger.warning(f"Error parsing outbound hangup webhook: {e}")

    return Response(content="OK", media_type="text/plain")


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
        "livekit_room_name": record.livekit_room_name,
        "created_at": record.created_at.isoformat(),
        "connected_at": (
            record.connected_at.isoformat() if record.connected_at else None
        ),
        "ended_at": record.ended_at.isoformat() if record.ended_at else None,
    }


@app.post("/outbound/hangup/{call_id}")
async def outbound_hangup_call(call_id: str) -> dict:
    """Programmatically end an active outbound call via LiveKit."""
    record = call_manager.get_call(call_id)
    if not record:
        return {"error": "Call not found"}

    if record.status not in ("ringing", "connected"):
        return {"error": f"Call is not active (status: {record.status})"}

    if not record.livekit_room_name:
        return {"error": "No LiveKit room name — call may not be connected yet"}

    try:
        from livekit.api import LiveKitAPI
        from livekit.protocol.room import DeleteRoomRequest

        lk_api = LiveKitAPI(
            url=LIVEKIT_URL,
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET,
        )

        # Deleting the room disconnects all participants and ends the call
        await lk_api.room.delete_room(
            DeleteRoomRequest(room=record.livekit_room_name)
        )
        await lk_api.aclose()

        call_manager.update_status(
            call_id, "completed",
            ended_at=datetime.utcnow(),
            outcome="success",
        )
        logger.info(f"Programmatically ended outbound call {call_id}")
        return {"call_id": call_id, "status": "completed"}
    except Exception as e:
        logger.error(f"Failed to end call {call_id}: {e}")
        return {"error": str(e)}


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


@app.get("/hold")
@app.post("/hold")
async def hold_webhook() -> Response:
    """Hold endpoint — keeps call alive silently (used for outbound A-leg)."""
    response = plivoxml.ResponseElement()
    response.add(plivoxml.WaitElement(length=120))
    return Response(content=response.to_string(), media_type="application/xml")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run the outbound management server.

    NOTE: The LiveKit agent worker must also be started separately:
        uv run python -m outbound.agent dev
    """
    logger.info(
        f"Starting LiveKit Outbound Voice Agent server on port {SERVER_PORT}"
    )

    if not LIVEKIT_URL:
        logger.warning("LIVEKIT_URL not set. Outbound calls will fail.")

    logger.info(
        "Remember to start the agent worker: uv run python -m outbound.agent dev"
    )

    uvicorn.run(
        "outbound.server:app", host="0.0.0.0", port=SERVER_PORT, log_level="info"
    )


if __name__ == "__main__":
    main()
