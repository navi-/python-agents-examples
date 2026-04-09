"""Inbound voice agent — LiveKit VoicePipelineAgent for incoming calls.

This is the only file needed for inbound calls. It:
1. Creates a LiveKit inbound SIP trunk + dispatch rule (one-time setup)
2. Runs the LiveKit agent worker that auto-joins rooms when callers connect

No separate server.py is required — Plivo sends SIP directly to LiveKit,
and the dispatch rule routes callers to individual rooms.

Usage:
    uv run python -m inbound.agent dev
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

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

# Plivo phone number registered on the SIP trunk
PLIVO_PHONE_NUMBER = os.getenv("PLIVO_PHONE_NUMBER", "")

# System prompt loaded from file
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    (Path(__file__).parent / "system_prompt.md").read_text().strip(),
)


# =============================================================================
# SIP Trunk Setup (runs once before the worker starts)
# =============================================================================


async def setup_sip_inbound() -> str | None:
    """Create or reuse a LiveKit inbound SIP trunk and dispatch rule.

    Returns the trunk ID if successful, None otherwise.
    """
    if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
        logger.warning("LiveKit credentials not set — skipping SIP trunk setup")
        return None

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

        from utils import normalize_phone_number

        lk_api = LiveKitAPI(
            url=LIVEKIT_URL,
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET,
        )

        phone = normalize_phone_number(PLIVO_PHONE_NUMBER)
        trunk_name = "plivo-inbound-gpt52"

        # Check for existing trunk
        existing = await lk_api.sip.list_sip_inbound_trunk(
            ListSIPInboundTrunkRequest()
        )
        trunk_id = ""
        for trunk in existing.items:
            if trunk.name == trunk_name:
                trunk_id = trunk.sip_trunk_id
                logger.info(f"Reusing inbound SIP trunk: {trunk_id}")
                break

        if not trunk_id:
            result = await lk_api.sip.create_sip_inbound_trunk(
                CreateSIPInboundTrunkRequest(
                    trunk=SIPInboundTrunkInfo(
                        name=trunk_name,
                        numbers=[f"+{phone}"] if phone else [],
                        krisp_enabled=True,
                    )
                )
            )
            trunk_id = result.sip_trunk_id
            logger.info(f"Created inbound SIP trunk: {trunk_id}")

        # Check for existing dispatch rule
        rules = await lk_api.sip.list_sip_dispatch_rule(
            ListSIPDispatchRuleRequest()
        )
        has_rule = any(trunk_id in rule.trunk_ids for rule in rules.items)

        if not has_rule:
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
            logger.info("Created dispatch rule (individual rooms, prefix='inbound-')")

        await lk_api.aclose()
        return trunk_id

    except Exception as e:
        logger.error(f"SIP trunk setup failed: {e}")
        return None


# =============================================================================
# Function Tools
# =============================================================================


def _build_tool_functions():
    """Build LiveKit-compatible function tools for the agent."""
    from livekit.agents import llm

    @llm.function_tool
    async def check_order_status(order_id: str) -> str:
        """Check the status of an order by its ID.

        Args:
            order_id: The order ID to look up.
        """
        logger.info(f"Tool call: check_order_status(order_id={order_id})")
        return (
            f"Order {order_id} is currently being processed. "
            "Estimated delivery is in two to three business days."
        )

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
        """Schedule a callback for the caller at a preferred time.

        Args:
            phone_number: The phone number to call back.
            preferred_time: When the caller would like to be called back.
            reason: The reason for the callback.
        """
        logger.info(f"Tool call: schedule_callback(phone={phone_number}, time={preferred_time})")
        return f"Callback scheduled for {preferred_time}."

    @llm.function_tool
    async def transfer_call(department: str) -> str:
        """Transfer the call to a human agent in the specified department.

        Args:
            department: The department to transfer to (e.g., 'billing', 'support', 'sales').
        """
        logger.info(f"Tool call: transfer_call(department={department})")
        return f"Transferring you to {department}. Please hold."

    @llm.function_tool
    async def end_call() -> str:
        """End the current phone call when the conversation is complete."""
        logger.info("Tool call: end_call()")
        return "The call has been ended. Goodbye!"

    return [check_order_status, send_sms, schedule_callback, transfer_call, end_call]


# =============================================================================
# LiveKit Agent
# =============================================================================


def _prewarm(proc) -> None:
    """Pre-warm: load Silero VAD model."""
    from livekit.plugins import silero

    proc.userdata["vad"] = silero.VAD.load()
    logger.info("Silero VAD model loaded")


async def _entrypoint(ctx) -> None:
    """LiveKit agent entrypoint — creates and starts the VoicePipelineAgent."""
    from livekit.agents import AutoSubscribe, llm
    from livekit.agents.pipeline import VoicePipelineAgent
    from livekit.plugins import deepgram, elevenlabs, noise_cancellation
    from livekit.plugins import openai as openai_plugin
    from livekit.plugins.turn_detector.multilingual import MultilingualModel

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    logger.info(f"Connected to room: {ctx.room.name}")

    participant = await ctx.wait_for_participant()
    logger.info(f"Participant joined: {participant.identity}")

    initial_ctx = llm.ChatContext()
    initial_ctx.add_message(role="system", content=SYSTEM_PROMPT)

    # Turn detection: Silero VAD (speech presence, barge-in) +
    # MultilingualModel (135M transformer, semantic end-of-turn on partial transcripts)
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
    logger.info("VoicePipelineAgent started")

    await agent.say(
        "Hello! Thank you for calling. How can I help you today?",
        allow_interruptions=True,
    )


# =============================================================================
# Public API
# =============================================================================


def run_agent() -> None:
    """Set up SIP trunk, then start the LiveKit agent worker.

    Usage: uv run python -m inbound.agent dev
    """
    # One-time SIP trunk + dispatch rule setup
    trunk_id = asyncio.run(setup_sip_inbound())
    if trunk_id:
        logger.info(f"SIP trunk ready: {trunk_id}")
    else:
        logger.warning("SIP trunk not configured — agent will still start but no calls will route")

    from livekit.agents import WorkerOptions, cli

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=_entrypoint,
            prewarm_fnc=_prewarm,
        )
    )


if __name__ == "__main__":
    run_agent()
