"""Voice agent using LiveKit with AssemblyAI STT, OpenAI GPT-5.4-mini, and Cartesia TTS.

This module provides the LiveKit Agent for inbound calls:
- Uses LiveKit Agents framework for orchestration
- AssemblyAI for real-time streaming speech-to-text
- OpenAI GPT-5.4-mini for streaming language model
- Cartesia for streaming text-to-speech
- Silero VAD for voice activity detection

Usage:
    python -m inbound.server
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
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

# System prompt loaded from file
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    (Path(__file__).parent / "system_prompt.md").read_text().strip(),
)


def prewarm(proc: JobProcess) -> None:
    """Prewarm resources before agent starts handling jobs.

    Loads the Silero VAD model into the process so it is ready when
    the first call arrives, avoiding cold-start latency.
    """
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext) -> None:
    """LiveKit agent entrypoint for inbound voice calls.

    Called by the LiveKit framework when a new participant joins a room
    (typically via SIP from Plivo). Creates an Agent with the system prompt
    and an AgentSession with streaming STT, LLM, and TTS components.

    Args:
        ctx: LiveKit job context with room and participant info.
    """
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    logger.info(f"Room connected: {ctx.room.name}")

    # Wait for the first participant (the caller via SIP)
    participant = await ctx.wait_for_participant()
    logger.info(f"Participant joined: {participant.identity}")

    # Create the agent with system instructions
    agent = Agent(instructions=SYSTEM_PROMPT)

    # Create session with streaming STT, LLM, TTS, and VAD
    session = AgentSession(
        stt=assemblyai.STT(),
        llm=openai.LLM(model=OPENAI_MODEL),
        tts=cartesia.TTS(voice=CARTESIA_VOICE),
        vad=ctx.proc.userdata["vad"],
    )

    # Start the session in the room
    session.start(agent=agent, room=ctx.room)

    # Send initial greeting
    session.say("Hello! Thank you for calling. How can I help you today?")


def create_worker_options() -> WorkerOptions:
    """Create LiveKit WorkerOptions for the inbound agent.

    Returns:
        WorkerOptions configured for the inbound voice agent.
    """
    return WorkerOptions(
        entrypoint_fnc=entrypoint,
        prewarm_fnc=prewarm,
    )
