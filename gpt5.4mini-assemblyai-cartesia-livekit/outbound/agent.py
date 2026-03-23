"""Outbound voice agent -- LiveKit Agent + AgentSession + call state management.

Uses LiveKit Agent with AssemblyAI streaming STT, OpenAI GPT-5.4-mini streaming LLM,
and Cartesia streaming TTS. Includes CallManager for tracking outbound call lifecycle.

Status state machine:
    initiating -> ringing -> connected -> completed
                         |-> no_answer
                |-> failed
"""

from __future__ import annotations

import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from livekit import api as livekit_api
from livekit.agents import Agent, AgentSession, AutoSubscribe, JobContext, JobProcess, WorkerOptions
from livekit.plugins import assemblyai, cartesia, openai, silero
from loguru import logger

load_dotenv()

# Agent configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", "")

# Model configuration
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
CARTESIA_VOICE = os.getenv("CARTESIA_VOICE", "79a125e8-cd45-4c13-8a67-188112f4dd22")

# LiveKit configuration
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")

# =============================================================================
# System Prompt
# =============================================================================

_OUTBOUND_PROMPT_TEMPLATE = (Path(__file__).parent / "system_prompt.md").read_text().strip()


def build_outbound_prompt(
    opening_reason: str = "",
    objective: str = "",
    context: str = "",
) -> str:
    """Build a concrete outbound system prompt by substituting template variables."""
    prompt = _OUTBOUND_PROMPT_TEMPLATE
    prompt = prompt.replace("{{opening_reason}}", opening_reason)
    prompt = prompt.replace("{{objective}}", objective)
    prompt = prompt.replace("{{context}}", context)
    return prompt


# Default system prompt (no template substitution)
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", _OUTBOUND_PROMPT_TEMPLATE)

# =============================================================================
# Outbound Call Records
# =============================================================================


@dataclass
class OutboundCallRecord:
    """Tracks the state of a single outbound call."""

    call_id: str
    phone_number: str
    status: str = "initiating"  # initiating|ringing|connected|completed|failed|no_answer
    campaign_id: str = ""
    context: str = ""
    system_prompt: str = ""
    initial_message: str = ""
    opening_reason: str = ""
    objective: str = ""
    plivo_request_uuid: str = ""
    plivo_call_uuid: str = ""
    livekit_room: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    connected_at: datetime | None = None
    ended_at: datetime | None = None
    duration: int = 0
    hangup_cause: str = ""
    outcome: str = ""  # success|no_answer|busy|failed


def determine_outcome(hangup_cause: str, duration: int) -> str:
    """Map Plivo hangup cause and duration to a high-level outcome.

    See https://www.plivo.com/docs/voice/troubleshooting/hangup-causes/
    """
    cause = hangup_cause.upper() if hangup_cause else ""

    if cause in ("NO_ANSWER", "ORIGINATOR_CANCEL"):
        return "no_answer"
    if cause in ("USER_BUSY", "CALL_REJECTED"):
        return "busy"
    if cause in (
        "UNALLOCATED_NUMBER",
        "INVALID_NUMBER_FORMAT",
        "NO_ROUTE_DESTINATION",
        "NETWORK_OUT_OF_ORDER",
        "SERVICE_UNAVAILABLE",
        "RECOVERY_ON_TIMER_EXPIRE",
        "BEARERCAPABILITY_NOTAVAIL",
    ):
        return "failed"

    # If the call was answered and had meaningful duration, consider it success
    if duration > 0 or cause in ("NORMAL_CLEARING", ""):
        return "success"

    return "failed"


class CallManager:
    """Thread-safe manager for outbound call records."""

    def __init__(self) -> None:
        self._calls: dict[str, OutboundCallRecord] = {}
        self._lock = threading.Lock()

    def create_call(
        self,
        phone_number: str,
        campaign_id: str = "",
        opening_reason: str = "",
        objective: str = "",
        context: str = "",
    ) -> OutboundCallRecord:
        """Create and register a new outbound call record."""
        call_id = str(uuid.uuid4())
        system_prompt = build_outbound_prompt(opening_reason, objective, context)

        if opening_reason:
            initial_message = (
                "The call has been answered. Begin with your outbound greeting now. "
                "State your name, company, and that you are reaching out regarding: "
                f"{opening_reason}. Then ask if now is a good time."
            )
        else:
            initial_message = (
                "The call has been answered. Begin with your outbound greeting now. "
                "State your name, company, and why you are calling. "
                "Then ask if now is a good time."
            )

        record = OutboundCallRecord(
            call_id=call_id,
            phone_number=phone_number,
            campaign_id=campaign_id,
            opening_reason=opening_reason,
            objective=objective,
            context=context,
            system_prompt=system_prompt,
            initial_message=initial_message,
        )

        with self._lock:
            self._calls[call_id] = record

        return record

    def get_call(self, call_id: str) -> OutboundCallRecord | None:
        """Look up a call by its ID."""
        with self._lock:
            return self._calls.get(call_id)

    def update_status(self, call_id: str, status: str, **kwargs: Any) -> OutboundCallRecord | None:
        """Thread-safe status update with optional extra fields."""
        with self._lock:
            record = self._calls.get(call_id)
            if record is None:
                return None
            record.status = status
            for key, value in kwargs.items():
                if hasattr(record, key):
                    setattr(record, key, value)
            return record

    def get_active_calls(self) -> list[OutboundCallRecord]:
        """Return calls with status in (initiating, ringing, connected)."""
        with self._lock:
            return [
                r
                for r in self._calls.values()
                if r.status in ("initiating", "ringing", "connected")
            ]

    def get_calls_by_campaign(self, campaign_id: str) -> list[OutboundCallRecord]:
        """Return all calls for a given campaign."""
        with self._lock:
            return [r for r in self._calls.values() if r.campaign_id == campaign_id]

    def reset(self) -> None:
        """Clear all records (useful for testing)."""
        with self._lock:
            self._calls.clear()


# =============================================================================
# LiveKit Room Management for Outbound
# =============================================================================

# In-memory store mapping room names to outbound call context
_room_call_context: dict[str, dict[str, str]] = {}
_room_context_lock = threading.Lock()


def store_room_context(room_name: str, system_prompt: str, initial_message: str) -> None:
    """Store outbound call context for a LiveKit room."""
    with _room_context_lock:
        _room_call_context[room_name] = {
            "system_prompt": system_prompt,
            "initial_message": initial_message,
        }


def get_room_context(room_name: str) -> dict[str, str] | None:
    """Retrieve outbound call context for a LiveKit room."""
    with _room_context_lock:
        return _room_call_context.pop(room_name, None)


async def create_outbound_room(call_id: str) -> str | None:
    """Create a LiveKit room for an outbound call.

    Args:
        call_id: Internal call ID.

    Returns:
        The room name if successful, None otherwise.
    """
    if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
        logger.warning("LiveKit credentials not configured")
        return None

    try:
        room_name = f"outbound-{call_id}"
        lk_api = livekit_api.LiveKitAPI(
            url=LIVEKIT_URL,
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET,
        )

        await lk_api.room.create_room(
            livekit_api.CreateRoomRequest(name=room_name, empty_timeout=300)
        )

        logger.info(f"Created LiveKit room: {room_name} for outbound call {call_id}")
        await lk_api.aclose()
        return room_name

    except Exception as e:
        logger.error(f"Failed to create LiveKit room: {e}")
        return None


# =============================================================================
# Public API
# =============================================================================


def prewarm(proc: JobProcess) -> None:
    """Prewarm resources before agent starts handling jobs."""
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext) -> None:
    """LiveKit agent entrypoint for outbound voice calls.

    Args:
        ctx: LiveKit job context with room and participant info.
    """
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    logger.info(f"Outbound room connected: {ctx.room.name}")

    # Check for outbound call context
    room_context = get_room_context(ctx.room.name)
    system_prompt = room_context["system_prompt"] if room_context else SYSTEM_PROMPT
    initial_message = (
        room_context["initial_message"] if room_context else "Hello, I'm calling for help."
    )

    # Wait for the callee to join via SIP
    participant = await ctx.wait_for_participant()
    logger.info(f"Outbound participant joined: {participant.identity}")

    # Create agent with outbound system instructions
    agent = Agent(instructions=system_prompt)

    # Create session with streaming STT, LLM, TTS, and VAD
    session = AgentSession(
        stt=assemblyai.STT(),
        llm=openai.LLM(model=OPENAI_MODEL),
        tts=cartesia.TTS(voice=CARTESIA_VOICE),
        vad=ctx.proc.userdata["vad"],
    )

    # Start the session in the room
    session.start(agent=agent, room=ctx.room)

    # Send the outbound greeting
    session.say(initial_message)


def create_worker_options() -> WorkerOptions:
    """Create LiveKit WorkerOptions for the outbound agent.

    Returns:
        WorkerOptions configured for the outbound voice agent.
    """
    return WorkerOptions(
        entrypoint_fnc=entrypoint,
        prewarm_fnc=prewarm,
    )
