"""Outbound voice agent -- Pipecat + Gemini Live pipeline + call state management.

Loads the outbound system prompt and provides run_agent() for handling
outbound call WebSocket sessions, plus CallManager for tracking call lifecycle.

Status state machine:
    initiating -> ringing -> connected -> completed
                         |-> no_answer
                |-> failed
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
from pipecat.frames.frames import LLMMessagesAppendFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.serializers.plivo import PlivoFrameSerializer
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

load_dotenv()

# Agent configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash-native-audio-preview-12-2025")
GEMINI_VOICE = os.getenv("GEMINI_VOICE", "Puck")

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

    def update_status(
        self, call_id: str, status: str, **kwargs: Any
    ) -> OutboundCallRecord | None:
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
                r for r in self._calls.values()
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
# Public API
# =============================================================================


async def run_agent(
    websocket,
    call_id: str,
    stream_id: str,
    auth_id: str = "",
    auth_token: str = "",
    system_prompt: str | None = None,
    initial_message: str = "Hello, I'm calling for help.",
) -> PipelineTask:
    """Run the Pipecat pipeline for an outbound voice agent session.

    Args:
        websocket: FastAPI WebSocket connection from Plivo
        call_id: Unique identifier for this call
        stream_id: Plivo stream identifier
        auth_id: Plivo auth ID for serializer
        auth_token: Plivo auth token for serializer
        system_prompt: Custom system prompt (from CallManager)
        initial_message: Initial message to trigger the agent greeting

    Returns:
        The PipelineTask instance
    """
    prompt = system_prompt or SYSTEM_PROMPT

    # Create Plivo serializer with authentication
    serializer = PlivoFrameSerializer(
        stream_id=stream_id,
        call_id=call_id,
        auth_id=auth_id or os.getenv("PLIVO_AUTH_ID", ""),
        auth_token=auth_token or os.getenv("PLIVO_AUTH_TOKEN", ""),
    )

    # Create transport with Plivo serializer
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_enabled=True,
            serializer=serializer,
        ),
    )

    # Initialize Gemini Live service for speech-to-speech
    llm = GeminiLiveLLMService(
        api_key=GEMINI_API_KEY,
        model=GEMINI_MODEL,
        voice_id=GEMINI_VOICE,
        system_instruction=prompt,
    )

    # Create the pipeline
    pipeline = Pipeline(
        [
            transport.input(),
            llm,
            transport.output(),
        ]
    )

    logger.info("Creating outbound pipeline task...")

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # Queue the outbound greeting
    async def send_greeting():
        await asyncio.sleep(1.5)  # Wait for Gemini session to connect
        logger.info("Sending outbound greeting message to Gemini...")
        await task.queue_frames(
            [LLMMessagesAppendFrame(
                [{"role": "user", "content": initial_message}],
                run_llm=True,
            )]
        )

    _greeting_task = asyncio.create_task(send_greeting())  # noqa: RUF006

    logger.info("Running outbound pipeline...")

    runner = PipelineRunner()
    await runner.run(task)

    logger.info("Outbound pipeline completed")
    return task
