"""Inbound voice agent — LiveKit VoicePipelineAgent for incoming calls.

Clone, configure .env, run. On startup this file:
1. Creates a LiveKit inbound SIP trunk + dispatch rule
2. Creates a Plivo Zentrunk origination URI + inbound trunk
3. Maps your Plivo phone number to the trunk
4. Starts the agent worker

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

# LiveKit SIP endpoint — per-project, found in LiveKit Cloud Project Settings
# Format: {project_id}.sip.livekit.cloud (e.g., vjnxecm0tjk.sip.livekit.cloud)
LIVEKIT_SIP_ENDPOINT = os.getenv("LIVEKIT_SIP_ENDPOINT", "")

# Plivo configuration
PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID", "")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN", "")
PLIVO_PHONE_NUMBER = os.getenv("PLIVO_PHONE_NUMBER", "")

# System prompt loaded from file
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    (Path(__file__).parent / "system_prompt.md").read_text().strip(),
)


# =============================================================================
# SIP Setup — LiveKit + Plivo (runs once before the worker starts)
# =============================================================================


async def setup_sip_inbound() -> bool:
    """Set up the full inbound SIP pipeline: LiveKit trunk → Plivo trunk → phone number.

    Steps:
    1. Create LiveKit inbound SIP trunk (idempotent, reuses by name)
    2. Create LiveKit dispatch rule (routes calls to individual rooms)
    3. Derive the LiveKit SIP URI from the trunk ID
    4. Create Plivo Zentrunk origination URI pointing to LiveKit SIP
    5. Create Plivo Zentrunk inbound trunk with that origination URI
    6. Map the Plivo phone number to the Plivo inbound trunk

    All steps are idempotent — safe to run on every startup.
    """
    from utils import PlivoZentrunk, normalize_phone_number

    phone = normalize_phone_number(PLIVO_PHONE_NUMBER)
    if not phone:
        logger.warning("PLIVO_PHONE_NUMBER not set — skipping SIP setup")
        return False

    # --- LiveKit side ---
    if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
        logger.warning("LiveKit credentials not set — skipping SIP setup")
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

        trunk_name = "plivo-inbound-gpt52"

        # 1. LiveKit inbound SIP trunk
        existing = await lk_api.sip.list_sip_inbound_trunk(
            ListSIPInboundTrunkRequest()
        )
        lk_trunk_id = ""
        for trunk in existing.items:
            if trunk.name == trunk_name:
                lk_trunk_id = trunk.sip_trunk_id
                logger.info(f"[LiveKit] Reusing inbound trunk: {lk_trunk_id}")
                break

        if not lk_trunk_id:
            result = await lk_api.sip.create_sip_inbound_trunk(
                CreateSIPInboundTrunkRequest(
                    trunk=SIPInboundTrunkInfo(
                        name=trunk_name,
                        numbers=[f"+{phone}"],
                        krisp_enabled=True,
                    )
                )
            )
            lk_trunk_id = result.sip_trunk_id
            logger.info(f"[LiveKit] Created inbound trunk: {lk_trunk_id}")

        # 2. LiveKit dispatch rule
        rules = await lk_api.sip.list_sip_dispatch_rule(
            ListSIPDispatchRuleRequest()
        )
        has_rule = any(lk_trunk_id in rule.trunk_ids for rule in rules.items)

        if not has_rule:
            await lk_api.sip.create_sip_dispatch_rule(
                CreateSIPDispatchRuleRequest(
                    name=f"dispatch-{trunk_name}",
                    trunk_ids=[lk_trunk_id],
                    rule=SIPDispatchRule(
                        dispatch_rule_individual=SIPDispatchRuleIndividual(
                            room_prefix="inbound-",
                        )
                    ),
                )
            )
            logger.info("[LiveKit] Created dispatch rule (prefix='inbound-')")

        await lk_api.aclose()

        # 3. LiveKit SIP endpoint (per-project, from Project Settings)
        if not LIVEKIT_SIP_ENDPOINT:
            logger.warning(
                "[LiveKit] LIVEKIT_SIP_ENDPOINT not set. "
                "Find it in LiveKit Cloud → Project Settings → SIP URI. "
                "Format: {project_id}.sip.livekit.cloud"
            )
            await lk_api.aclose()
            return True  # LiveKit trunk created, but can't configure Plivo without endpoint

        lk_sip_uri = f"{LIVEKIT_SIP_ENDPOINT};transport=tcp"
        logger.info(f"[LiveKit] SIP endpoint: {lk_sip_uri}")

    except Exception as e:
        logger.error(f"[LiveKit] SIP setup failed: {e}")
        return False

    # --- Plivo Zentrunk side ---
    if not all([PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN]):
        logger.warning("Plivo credentials not set — LiveKit trunk created but Plivo not configured")
        logger.info(f"  Manually point Plivo Zentrunk at: {lk_sip_uri}")
        return True

    try:
        plivo = PlivoZentrunk(PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN)
        uri_name = f"livekit-{trunk_name}"
        trunk_name_plivo = f"inbound-{trunk_name}"

        # 4. Plivo origination URI → LiveKit SIP endpoint
        existing_uris = await plivo.list_origination_uris()
        uri_uuid = ""
        for uri in existing_uris:
            if uri.get("name") == uri_name:
                uri_uuid = uri.get("uri_uuid", "")
                logger.info(f"[Plivo] Reusing origination URI: {uri_uuid}")
                break

        if not uri_uuid:
            uri_uuid = await plivo.create_origination_uri(uri_name, lk_sip_uri)

        # 5. Plivo inbound trunk
        existing_trunks = await plivo.list_trunks()
        plivo_trunk_id = ""
        for t in existing_trunks:
            if t.get("name") == trunk_name_plivo:
                plivo_trunk_id = t.get("trunk_id", "")
                logger.info(f"[Plivo] Reusing inbound trunk: {plivo_trunk_id}")
                break

        if not plivo_trunk_id:
            plivo_trunk_id = await plivo.create_inbound_trunk(trunk_name_plivo, uri_uuid)

        # 6. Map phone number to trunk
        await plivo.map_number_to_trunk(phone, plivo_trunk_id)

        logger.info("")
        logger.info("  Inbound SIP pipeline ready:")
        logger.info(f"  Phone +{phone} → Plivo trunk {plivo_trunk_id} → LiveKit {lk_sip_uri}")
        logger.info("")
        return True

    except Exception as e:
        logger.error(f"[Plivo] Zentrunk setup failed: {e}")
        logger.info(f"  Manually point Plivo Zentrunk at: {lk_sip_uri}")
        return True  # LiveKit side succeeded


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
# Entry Point
# =============================================================================


def run_agent() -> None:
    """Set up SIP pipeline (LiveKit + Plivo), then start the agent worker.

    Usage: uv run python -m inbound.agent dev
    """
    asyncio.run(setup_sip_inbound())

    from livekit.agents import WorkerOptions, cli

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=_entrypoint,
            prewarm_fnc=_prewarm,
        )
    )


if __name__ == "__main__":
    run_agent()
