"""Optional management server for inbound calls.

NOT required for call handling — the agent worker (agent.py) handles
everything via LiveKit SIP. This server exists only for health checks
and operational visibility.

Usage (optional):
    uv run python -m inbound.server
"""

from __future__ import annotations

import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from loguru import logger

from utils import normalize_phone_number

load_dotenv()

SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
PLIVO_PHONE_NUMBER = os.getenv("PLIVO_PHONE_NUMBER", "")
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "")

app = FastAPI(
    title="GPT-5.2 LiveKit Voice Agent (Inbound)",
    description="Health check for inbound voice agent",
    version="0.1.0",
)


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
        "livekit_url": LIVEKIT_URL or "not configured",
    }


def main() -> None:
    """Run the optional management server."""
    logger.info(f"Starting health check server on port {SERVER_PORT}")
    logger.info("This server is optional — the agent worker handles all calls.")
    uvicorn.run("inbound.server:app", host="0.0.0.0", port=SERVER_PORT, log_level="info")


if __name__ == "__main__":
    main()
