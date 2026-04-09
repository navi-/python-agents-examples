"""Optional HTTP wrapper for outbound call management.

NOT required — calls can be triggered directly via:
    from outbound.agent import initiate_call
    result = await initiate_call(phone_number="+15551234567")

This server is a convenience for triggering calls via curl/HTTP.

Usage (optional):
    uv run python -m outbound.server
"""

from __future__ import annotations

import os
from datetime import datetime

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from loguru import logger

from outbound.agent import OUTBOUND_TRUNK_ID, call_manager, initiate_call
from utils import normalize_phone_number

load_dotenv()

SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
PLIVO_PHONE_NUMBER = os.getenv("PLIVO_PHONE_NUMBER", "")
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")

app = FastAPI(
    title="GPT-5.2 LiveKit Voice Agent (Outbound)",
    description="Optional HTTP wrapper for outbound call management",
    version="0.1.0",
)


@app.get("/")
async def health_check() -> dict:
    phone = normalize_phone_number(PLIVO_PHONE_NUMBER)
    return {
        "status": "ok",
        "service": "gpt5.2-deepgramflux-elevenflashv2.5-livekit-outbound",
        "phone_number": f"+{phone}" if phone else "not configured",
        "livekit_url": LIVEKIT_URL or "not configured",
        "sip_trunk_id": OUTBOUND_TRUNK_ID or "not configured",
    }


@app.post("/outbound/call")
async def outbound_call(
    phone_number: str = Query(default=""),
    campaign_id: str = Query(default=""),
    opening_reason: str = Query(default=""),
    objective: str = Query(default=""),
    context: str = Query(default=""),
) -> dict:
    """Trigger an outbound call. Delegates to agent.initiate_call()."""
    if not phone_number:
        return {"error": "phone_number is required"}
    return await initiate_call(
        phone_number=phone_number,
        campaign_id=campaign_id,
        opening_reason=opening_reason,
        objective=objective,
        context=context,
    )


@app.get("/outbound/status/{call_id}")
async def outbound_status(call_id: str) -> dict:
    record = call_manager.get_call(call_id)
    if not record:
        return {"error": "Call not found"}
    return {
        "call_id": record.call_id,
        "phone_number": record.phone_number,
        "status": record.status,
        "outcome": record.outcome,
        "livekit_room_name": record.livekit_room_name,
        "created_at": record.created_at.isoformat(),
    }


@app.post("/outbound/hangup/{call_id}")
async def outbound_hangup(call_id: str) -> dict:
    record = call_manager.get_call(call_id)
    if not record:
        return {"error": "Call not found"}
    if not record.livekit_room_name:
        return {"error": "No LiveKit room — call may not be connected"}

    try:
        from livekit.api import LiveKitAPI
        from livekit.protocol.room import DeleteRoomRequest

        lk_api = LiveKitAPI(
            url=LIVEKIT_URL,
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET,
        )
        await lk_api.room.delete_room(DeleteRoomRequest(room=record.livekit_room_name))
        await lk_api.aclose()

        call_manager.update_status(call_id, "completed", ended_at=datetime.utcnow())
        return {"call_id": call_id, "status": "completed"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/outbound/campaign/{campaign_id}")
async def outbound_campaign(campaign_id: str) -> dict:
    records = call_manager.get_calls_by_campaign(campaign_id)
    return {
        "campaign_id": campaign_id,
        "total": len(records),
        "calls": [
            {"call_id": r.call_id, "phone_number": r.phone_number, "status": r.status}
            for r in records
        ],
    }


def main() -> None:
    logger.info(f"Starting optional outbound API on port {SERVER_PORT}")
    uvicorn.run("outbound.server:app", host="0.0.0.0", port=SERVER_PORT, log_level="info")


if __name__ == "__main__":
    main()
