"""Outbound call management API.

Provides HTTP endpoints to trigger and manage outbound calls.
The agent worker (agent.py) must be running to handle audio.

Usage:
    uv run python -m outbound.server
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from loguru import logger

from outbound.agent import OUTBOUND_TRUNK_ID, CallManager
from utils import normalize_phone_number

load_dotenv()

SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
PLIVO_PHONE_NUMBER = os.getenv("PLIVO_PHONE_NUMBER", "")
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")

app = FastAPI(
    title="GPT-5.2 LiveKit Voice Agent (Outbound)",
    description="Outbound call management API",
    version="0.1.0",
)

call_manager = CallManager()


@app.get("/")
async def health_check() -> dict:
    """Health check endpoint."""
    phone = normalize_phone_number(PLIVO_PHONE_NUMBER)
    return {
        "status": "ok",
        "service": "gpt5.2-deepgramflux-elevenflashv2.5-livekit-outbound",
        "phone_number": f"+{phone}" if phone else "not configured",
        "livekit_url": LIVEKIT_URL or "not configured",
        "sip_trunk_id": OUTBOUND_TRUNK_ID or "not configured",
    }


@app.post("/outbound/call")
async def outbound_initiate(
    phone_number: str = Query(default=""),
    campaign_id: str = Query(default=""),
    opening_reason: str = Query(default=""),
    objective: str = Query(default=""),
    context: str = Query(default=""),
) -> dict:
    """Initiate an outbound call via LiveKit SIP.

    Creates a call record and a LiveKit SIP participant that dials
    through the outbound SIP trunk (Plivo).
    """
    if not phone_number:
        return {"error": "phone_number is required"}

    if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
        return {"error": "LiveKit credentials not configured"}

    trunk_id = OUTBOUND_TRUNK_ID
    if not trunk_id:
        return {"error": "Outbound SIP trunk not configured — start agent worker first"}

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

        room_metadata = json.dumps({
            "call_id": record.call_id,
            "system_prompt": record.system_prompt,
            "initial_message": record.initial_message,
            "is_outbound": True,
        })

        await lk_api.sip.create_sip_participant(
            CreateSIPParticipantRequest(
                sip_trunk_id=trunk_id,
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
        )

        logger.info(
            f"Outbound call initiated: call_id={record.call_id}, "
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
    """Programmatically end an active outbound call."""
    record = call_manager.get_call(call_id)
    if not record:
        return {"error": "Call not found"}

    if record.status not in ("ringing", "connected"):
        return {"error": f"Call is not active (status: {record.status})"}

    if not record.livekit_room_name:
        return {"error": "No LiveKit room — call may not be connected yet"}

    try:
        from livekit.api import LiveKitAPI
        from livekit.protocol.room import DeleteRoomRequest

        lk_api = LiveKitAPI(
            url=LIVEKIT_URL,
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET,
        )

        await lk_api.room.delete_room(
            DeleteRoomRequest(room=record.livekit_room_name)
        )
        await lk_api.aclose()

        call_manager.update_status(
            call_id, "completed",
            ended_at=datetime.utcnow(),
            outcome="success",
        )
        logger.info(f"Ended outbound call {call_id}")
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


def main() -> None:
    """Run the outbound call management server."""
    logger.info(f"Starting outbound call API on port {SERVER_PORT}")
    logger.info("Agent worker must be running: uv run python -m outbound.agent dev")
    uvicorn.run(
        "outbound.server:app", host="0.0.0.0", port=SERVER_PORT, log_level="info"
    )


if __name__ == "__main__":
    main()
