"""Outbound voice agent — LiveKit VoicePipelineAgent + call management.

Clone, configure .env, run. On startup this file:
1. Creates Plivo Zentrunk SIP credentials + outbound trunk (gets termination domain)
2. Creates a LiveKit outbound SIP trunk using the Plivo termination domain
3. Starts the agent worker

Calls triggered via initiate_call() or directly via LiveKit API.

Usage:
    uv run python -m outbound.agent dev
"""

from __future__ import annotations

import asyncio
import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# Agent configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5.2-mini")
ELEVEN_VOICE = os.getenv("ELEVEN_VOICE", "jessica")
ELEVEN_MODEL = os.getenv("ELEVEN_MODEL", "eleven_flash_v2_5")

# LiveKit configuration
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")

# Plivo configuration
PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID", "")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN", "")
PLIVO_PHONE_NUMBER = os.getenv("PLIVO_PHONE_NUMBER", "")

# System prompt
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


SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", _OUTBOUND_PROMPT_TEMPLATE)


# =============================================================================
# SIP Setup — Plivo outbound trunk + LiveKit outbound trunk
# =============================================================================

# Set by setup_sip_outbound(), used by initiate_call()
OUTBOUND_TRUNK_ID = ""


async def setup_sip_outbound() -> bool:
    """Set up the full outbound SIP pipeline: Plivo credentials → Plivo trunk → LiveKit trunk.

    Steps:
    1. Create Plivo Zentrunk SIP credentials (idempotent, reuses by name)
    2. Create Plivo Zentrunk outbound trunk → gets termination domain (XXXXXXX.zt.plivo.com)
    3. Create LiveKit outbound SIP trunk using the Plivo termination domain + credentials

    All steps are idempotent — safe to run on every startup.
    """
    global OUTBOUND_TRUNK_ID

    if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
        logger.warning("LiveKit credentials not set — skipping outbound SIP setup")
        return False

    if not all([PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN]):
        logger.warning("Plivo credentials not set — skipping outbound SIP setup")
        return False

    from utils import PlivoZentrunk, normalize_phone_number

    phone = normalize_phone_number(PLIVO_PHONE_NUMBER)

    # --- Plivo side: credentials + outbound trunk ---
    try:
        plivo = PlivoZentrunk(PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN)
        cred_name = "livekit-outbound-gpt52"
        trunk_name_plivo = "outbound-gpt52"

        # 1. Plivo SIP credential
        existing_creds = await plivo.list_credentials()
        cred_uuid = ""
        for c in existing_creds:
            if c.get("name") == cred_name:
                cred_uuid = c.get("credential_uuid", "")
                logger.info(f"[Plivo] Reusing SIP credential: {cred_uuid}")
                break

        if not cred_uuid:
            cred_uuid = await plivo.create_credential(
                cred_name, PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN
            )

        # 2. Plivo outbound trunk → termination domain
        existing_trunks = await plivo.list_trunks()
        plivo_trunk_id = ""
        plivo_domain = ""
        for t in existing_trunks:
            if t.get("name") == trunk_name_plivo:
                plivo_trunk_id = t.get("trunk_id", "")
                plivo_domain = t.get("trunk_domain", "")
                logger.info(f"[Plivo] Reusing outbound trunk: {plivo_trunk_id}")
                break

        if not plivo_trunk_id:
            plivo_trunk_id, plivo_domain = await plivo.create_outbound_trunk(
                trunk_name_plivo, cred_uuid
            )

        if not plivo_domain:
            logger.error("[Plivo] No termination domain returned — cannot create LiveKit trunk")
            return False

        logger.info(f"[Plivo] Termination domain: {plivo_domain}")

    except Exception as e:
        logger.error(f"[Plivo] Outbound trunk setup failed: {e}")
        return False

    # --- LiveKit side: outbound SIP trunk ---
    try:
        from livekit.api import LiveKitAPI
        from livekit.protocol.sip import (
            CreateSIPOutboundTrunkRequest,
            ListSIPOutboundTrunkRequest,
            SIPOutboundTrunkInfo,
        )

        lk_api = LiveKitAPI(
            url=LIVEKIT_URL,
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET,
        )

        trunk_name_lk = "plivo-outbound-gpt52"

        # 3. LiveKit outbound trunk
        existing = await lk_api.sip.list_sip_outbound_trunk(
            ListSIPOutboundTrunkRequest()
        )
        lk_trunk_id = ""
        for trunk in existing.items:
            if trunk.name == trunk_name_lk:
                lk_trunk_id = trunk.sip_trunk_id
                logger.info(f"[LiveKit] Reusing outbound trunk: {lk_trunk_id}")
                break

        if not lk_trunk_id:
            result = await lk_api.sip.create_sip_outbound_trunk(
                CreateSIPOutboundTrunkRequest(
                    trunk=SIPOutboundTrunkInfo(
                        name=trunk_name_lk,
                        address=plivo_domain,
                        numbers=[f"+{phone}"] if phone else [],
                        auth_username=PLIVO_AUTH_ID,
                        auth_password=PLIVO_AUTH_TOKEN,
                        transport=1,  # UDP
                    )
                )
            )
            lk_trunk_id = result.sip_trunk_id
            logger.info(f"[LiveKit] Created outbound trunk: {lk_trunk_id}")

        await lk_api.aclose()
        OUTBOUND_TRUNK_ID = lk_trunk_id

        logger.info("")
        logger.info("  Outbound SIP pipeline ready:")
        logger.info(f"  LiveKit {lk_trunk_id} → Plivo {plivo_domain} → PSTN")
        logger.info("")
        return True

    except Exception as e:
        logger.error(f"[LiveKit] Outbound trunk setup failed: {e}")
        return False


# =============================================================================
# Outbound Call Records
# =============================================================================


@dataclass
class OutboundCallRecord:
    """Tracks the state of a single outbound call."""

    call_id: str
    phone_number: str
    status: str = "initiating"
    campaign_id: str = ""
    context: str = ""
    system_prompt: str = ""
    initial_message: str = ""
    opening_reason: str = ""
    objective: str = ""
    livekit_room_name: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    connected_at: datetime | None = None
    ended_at: datetime | None = None
    duration: int = 0
    outcome: str = ""


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
        with self._lock:
            return self._calls.get(call_id)

    def update_status(
        self, call_id: str, status: str, **kwargs: Any
    ) -> OutboundCallRecord | None:
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
        with self._lock:
            return [
                r for r in self._calls.values()
                if r.status in ("initiating", "ringing", "connected")
            ]

    def get_calls_by_campaign(self, campaign_id: str) -> list[OutboundCallRecord]:
        with self._lock:
            return [r for r in self._calls.values() if r.campaign_id == campaign_id]

    def reset(self) -> None:
        with self._lock:
            self._calls.clear()


# =============================================================================
# Function Tools
# =============================================================================


def _build_tool_functions():
    from livekit.agents import llm

    @llm.function_tool
    async def send_sms(phone_number: str, message: str) -> str:
        """Send an SMS message to a phone number.

        Args:
            phone_number: The phone number to send the SMS to.
            message: The text message content.
        """
        logger.info(f"Tool call: send_sms(phone={phone_number})")
        return f"SMS sent successfully to {phone_number}."

    @llm.function_tool
    async def schedule_callback(
        phone_number: str, preferred_time: str, reason: str
    ) -> str:
        """Schedule a callback for the prospect at a preferred time.

        Args:
            phone_number: The phone number to call back.
            preferred_time: When the prospect would like to be called back.
            reason: The reason for the callback.
        """
        logger.info(f"Tool call: schedule_callback(phone={phone_number}, time={preferred_time})")
        return f"Callback scheduled for {preferred_time}."

    @llm.function_tool
    async def transfer_call(department: str) -> str:
        """Transfer the call to a human agent in the specified department.

        Args:
            department: The department to transfer to.
        """
        logger.info(f"Tool call: transfer_call(department={department})")
        return f"Transferring you to {department}. Please hold."

    @llm.function_tool
    async def end_call() -> str:
        """End the current phone call when the conversation is complete."""
        logger.info("Tool call: end_call()")
        return "The call has been ended. Goodbye!"

    return [send_sms, schedule_callback, transfer_call, end_call]


# =============================================================================
# LiveKit Agent
# =============================================================================


def _prewarm(proc) -> None:
    from livekit.plugins import silero

    proc.userdata["vad"] = silero.VAD.load()
    logger.info("Silero VAD model loaded (outbound)")


async def _entrypoint(ctx) -> None:
    from livekit.agents import AutoSubscribe, llm
    from livekit.agents.pipeline import VoicePipelineAgent
    from livekit.plugins import deepgram, elevenlabs, noise_cancellation
    from livekit.plugins import openai as openai_plugin
    from livekit.plugins.turn_detector.multilingual import MultilingualModel

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    logger.info(f"Connected to outbound room: {ctx.room.name}")

    participant = await ctx.wait_for_participant()
    logger.info(f"Outbound participant connected: {participant.identity}")

    system_prompt = SYSTEM_PROMPT
    initial_message = (
        "The call has been answered. Begin with your outbound greeting now. "
        "State your name, company, and why you are calling. "
        "Then ask if now is a good time."
    )

    room_metadata = ctx.room.metadata
    if room_metadata:
        import json

        try:
            metadata = json.loads(room_metadata)
            if metadata.get("system_prompt"):
                system_prompt = metadata["system_prompt"]
            if metadata.get("initial_message"):
                initial_message = metadata["initial_message"]
            logger.info(f"Loaded call metadata: call_id={metadata.get('call_id')}")
        except (json.JSONDecodeError, KeyError):
            logger.warning("Failed to parse room metadata, using defaults")

    initial_ctx = llm.ChatContext()
    initial_ctx.add_message(role="system", content=system_prompt)

    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(model="nova-3"),
        llm=openai_plugin.LLM(model=LLM_MODEL),
        tts=elevenlabs.TTS(voice=ELEVEN_VOICE, model=ELEVEN_MODEL),
        chat_ctx=initial_ctx,
        fnc_ctx=_build_tool_functions(),
        turn_detector=MultilingualModel(),
        noise_cancellation=noise_cancellation.BVC(),
    )

    agent.start(ctx.room, participant)
    logger.info("Outbound VoicePipelineAgent started")

    await agent.say(initial_message, allow_interruptions=True)


# =============================================================================
# Public API
# =============================================================================

call_manager = CallManager()


async def initiate_call(
    phone_number: str,
    campaign_id: str = "",
    opening_reason: str = "",
    objective: str = "",
    context: str = "",
) -> dict:
    """Initiate an outbound call via LiveKit SIP.

    Can be called from any async context — no HTTP server required.
    """
    import json

    from utils import normalize_phone_number

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

        logger.info(f"Outbound call initiated: call_id={record.call_id}, to=+{to_number}")

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


# =============================================================================
# Entry Point
# =============================================================================


def run_agent() -> None:
    """Set up SIP pipeline (Plivo + LiveKit), then start the agent worker.

    Usage: uv run python -m outbound.agent dev
    """
    asyncio.run(setup_sip_outbound())

    from livekit.agents import WorkerOptions, cli

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=_entrypoint,
            prewarm_fnc=_prewarm,
        )
    )


if __name__ == "__main__":
    run_agent()
