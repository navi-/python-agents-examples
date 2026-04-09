"""Inbound voice agent — LiveKit VoicePipelineAgent for incoming calls.

Uses LiveKit Agents framework with:
- Deepgram Flux STT (streaming speech-to-text via /listen/v2)
- OpenAI GPT-5.2-mini LLM
- ElevenLabs Flash v2.5 TTS
- Silero VAD (voice activity detection)
- Krisp BVC (background voice cancellation / noise cancellation)

The agent runs as a LiveKit worker that auto-joins rooms when SIP participants
connect via Plivo telephony.

Usage:
    # Run as standalone worker:
    uv run python -m inbound.agent dev

    # Or import and start programmatically:
    from inbound.agent import run_agent
    run_agent()
"""

from __future__ import annotations

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

# System prompt loaded from file
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    (Path(__file__).parent / "system_prompt.md").read_text().strip(),
)


# =============================================================================
# Function Tools
# =============================================================================


def _build_tool_functions():
    """Build LiveKit-compatible function tools for the agent.

    Imported lazily to avoid loading livekit at module level.
    """
    from livekit.agents import llm

    @llm.function_tool
    async def check_order_status(
        order_id: str,
    ) -> str:
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
    async def send_sms(
        phone_number: str,
        message: str,
    ) -> str:
        """Send an SMS message to a phone number.

        Args:
            phone_number: The phone number to send the SMS to.
            message: The text message content.
        """
        logger.info(f"Tool call: send_sms(phone_number={phone_number}, message={message[:50]}...)")
        return f"SMS sent successfully to {phone_number}."

    @llm.function_tool
    async def schedule_callback(
        phone_number: str,
        preferred_time: str,
        reason: str,
    ) -> str:
        """Schedule a callback for the caller at a preferred time.

        Args:
            phone_number: The phone number to call back.
            preferred_time: When the caller would like to be called back.
            reason: The reason for the callback.
        """
        logger.info(
            f"Tool call: schedule_callback(phone={phone_number}, "
            f"time={preferred_time}, reason={reason})"
        )
        return f"Callback scheduled for {preferred_time}. A specialist will call {phone_number}."

    @llm.function_tool
    async def transfer_call(
        department: str,
    ) -> str:
        """Transfer the call to a human agent in the specified department.

        Args:
            department: The department to transfer to (e.g., 'billing', 'support', 'sales').
        """
        logger.info(f"Tool call: transfer_call(department={department})")
        return f"Transferring you to the {department} department now. Please hold."

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
    """Pre-warm the agent process by loading the Silero VAD model."""
    from livekit.plugins import silero

    proc.userdata["vad"] = silero.VAD.load()
    logger.info("Silero VAD model pre-loaded")


async def _entrypoint(ctx) -> None:
    """LiveKit agent entrypoint — creates and starts the VoicePipelineAgent."""
    from livekit.agents import AutoSubscribe, llm
    from livekit.agents.pipeline import VoicePipelineAgent
    from livekit.plugins import deepgram, elevenlabs, noise_cancellation
    from livekit.plugins import openai as openai_plugin

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    logger.info(f"Connected to room: {ctx.room.name}")

    participant = await ctx.wait_for_participant()
    logger.info(f"Participant joined: {participant.identity}")

    # Build initial chat context with system prompt
    initial_ctx = llm.ChatContext()
    initial_ctx.add_message(role="system", content=SYSTEM_PROMPT)

    # Build function tools
    tools = _build_tool_functions()

    # Create the voice pipeline agent
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(model="nova-3"),
        llm=openai_plugin.LLM(model=LLM_MODEL),
        tts=elevenlabs.TTS(
            voice=ELEVEN_VOICE,
            model=ELEVEN_MODEL,
        ),
        chat_ctx=initial_ctx,
        fnc_ctx=tools,
        noise_cancellation=noise_cancellation.BVC(),
    )

    agent.start(ctx.room, participant)
    logger.info("VoicePipelineAgent started")

    # Send initial greeting
    await agent.say(
        "Hello! Thank you for calling. How can I help you today?",
        allow_interruptions=True,
    )


# =============================================================================
# Public API
# =============================================================================


def run_agent() -> None:
    """Start the LiveKit agent worker.

    This function blocks and runs the agent worker process, which connects
    to the LiveKit server and auto-joins rooms when SIP participants connect.

    Run with: uv run python -m inbound.agent dev
    """
    from livekit.agents import WorkerOptions, cli

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=_entrypoint,
            prewarm_fnc=_prewarm,
        )
    )


if __name__ == "__main__":
    run_agent()
