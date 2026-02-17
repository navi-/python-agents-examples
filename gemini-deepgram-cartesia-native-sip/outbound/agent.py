"""Outbound voice agent — VoiceAgent engine + call state management.

Loads the outbound system prompt and provides run_agent() for handling
outbound call RTP sessions, plus CallManager for tracking call lifecycle.

Status state machine:
    initiating -> ringing -> connected -> completed
                         |-> no_answer
                |-> failed
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from inbound.agent import CartesiaTTS, DeepgramSTT, GeminiLLM
from utils import (
    SileroVADProcessor,
    cartesia_to_plivo,
    pcm8k_to_vad,
)

if TYPE_CHECKING:
    from sip import RTPSession

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
    sip_call_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    connected_at: datetime | None = None
    ended_at: datetime | None = None
    duration: int = 0
    hangup_cause: str = ""
    outcome: str = ""  # success|no_answer|busy|failed


def determine_outcome(hangup_cause: str, duration: int) -> str:
    """Map hangup cause and duration to a high-level outcome."""
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
    ):
        return "failed"

    # If the call was answered and had meaningful duration, consider it success
    if duration > 0 or cause in ("NORMAL_CLEARING", ""):
        return "success"

    return "failed"


class CallManager:
    """Thread-safe manager for outbound call records."""

    def __init__(self) -> None:
        import threading

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
                "State your name, company, and why you are calling. Then ask if now is a good time."
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
# Outbound Voice Agent
# =============================================================================


class OutboundVoiceAgent:
    """Voice agent for outbound calls over RTP.

    Same STT+LLM+TTS+VAD pipeline as inbound, but with an initial greeting
    message that is spoken first.
    """

    def __init__(
        self,
        rtp_session: RTPSession,
        call_id: str,
        from_uri: str = "",
        to_uri: str = "",
        system_prompt: str | None = None,
        initial_message: str = "",
        greeting_audio: bytes = b"",
        greeting_text: str = "",
    ):
        self._rtp = rtp_session
        self.call_id = call_id
        self.from_uri = from_uri
        self.to_uri = to_uri
        self.initial_message = initial_message
        self._greeting_audio = greeting_audio

        self._running = False
        self._send_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._vad = SileroVADProcessor()
        self._is_responding = False
        self._is_playing = False  # True while TTS audio is being sent to RTP
        self._current_response_task: asyncio.Task | None = None
        self._last_bargein_time: float = 0.0  # Cooldown to prevent repeated barge-ins

        prompt = system_prompt or SYSTEM_PROMPT
        self._llm = GeminiLLM(prompt)
        self._tts = CartesiaTTS()
        self._stt = DeepgramSTT(on_transcript=self._handle_transcript)

        # Seed LLM history with the pre-generated greeting so it knows what
        # it already said and can continue the conversation coherently
        if greeting_text and initial_message:
            self._llm.seed_history(initial_message, greeting_text)

    async def _handle_transcript(self, transcript: str) -> None:
        """Handle transcribed text from STT."""
        if not transcript.strip():
            return

        logger.info(f"Processing user input: {transcript}")

        if self._current_response_task and not self._current_response_task.done():
            self._current_response_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._current_response_task

        self._current_response_task = asyncio.create_task(
            self._process_response(transcript)
        )

    async def _process_response(self, transcript: str) -> None:
        """Generate LLM response, synthesize TTS, and queue for sending."""
        try:
            self._is_responding = True

            response_text = await self._llm.generate_response(transcript)
            if not response_text:
                self._is_responding = False
                return

            tts_audio = await self._tts.synthesize(response_text)
            if not tts_audio:
                self._is_responding = False
                return

            pcm_8k = cartesia_to_plivo(tts_audio)
            self._is_playing = True
            await self._send_queue.put(pcm_8k)

        except asyncio.CancelledError:
            logger.debug("Response generation cancelled (barge-in)")
            raise
        finally:
            self._is_responding = False

    async def run(self) -> None:
        """Run the outbound voice agent session."""
        logger.info(f"Starting outbound agent for call {self.call_id}")
        self._running = True

        tasks = []
        try:
            await self._stt.connect()
            await self._tts.connect()

            # Start RTP tasks immediately so audio flows
            tasks = [
                asyncio.create_task(self._receive_from_rtp(), name="rtp_rx"),
                asyncio.create_task(self._send_to_rtp(), name="rtp_tx"),
            ]

            if self._greeting_audio:
                # Play pre-generated greeting immediately (zero latency)
                logger.info("Playing pre-generated greeting (0ms latency)")
                self._is_playing = True
                await self._send_queue.put(self._greeting_audio)
            elif self.initial_message:
                # Fallback: generate greeting now (adds ~3-4s latency)
                logger.warning("No pre-generated greeting, generating live")
                greeting_text = await self._llm.generate_response(self.initial_message)
                if greeting_text:
                    greeting_audio = await self._tts.synthesize(greeting_text)
                    if greeting_audio:
                        pcm_8k = cartesia_to_plivo(greeting_audio)
                        self._is_playing = True
                        await self._send_queue.put(pcm_8k)

            done, _pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                if task.exception():
                    logger.error(f"Task {task.get_name()} failed: {task.exception()}")

        except Exception as e:
            logger.error(f"Outbound agent error: {e}")
        finally:
            self._running = False
            await self._stt.close()
            await self._tts.close()
            if self._current_response_task and not self._current_response_task.done():
                self._current_response_task.cancel()
            for task in tasks:
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
            logger.info(f"Outbound agent ended for call {self.call_id}")

    async def _receive_from_rtp(self) -> None:
        """Receive PCM16 8kHz audio from RTP, send to STT and run VAD."""
        try:
            while self._running and self._rtp.is_running:
                try:
                    pcm_data = await asyncio.wait_for(
                        self._rtp.receive_audio(), timeout=0.1
                    )
                except TimeoutError:
                    continue

                await self._stt.send_audio(pcm_data)

                vad_audio = pcm8k_to_vad(pcm_data)
                speech_started, _speech_ended = self._vad.process(vad_audio)

                if speech_started and self._is_playing:
                    now = asyncio.get_event_loop().time()
                    if now - self._last_bargein_time > 2.0:
                        logger.info("Barge-in detected, cancelling response")
                        self._last_bargein_time = now
                        await self._trigger_interruption()

            if not self._rtp.is_running:
                logger.info(f"RTP session ended (remote hangup) for call {self.call_id}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"RTP receiver error: {e}")

    async def _trigger_interruption(self) -> None:
        """Cancel current response and drain send queue."""
        if self._current_response_task and not self._current_response_task.done():
            self._current_response_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._current_response_task

        while not self._send_queue.empty():
            try:
                self._send_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        self._is_responding = False
        self._is_playing = False

    async def _send_to_rtp(self) -> None:
        """Dequeue audio and send to RTPSession."""
        try:
            while self._running and self._rtp.is_running:
                try:
                    pcm_data = await asyncio.wait_for(
                        self._send_queue.get(), timeout=0.1
                    )
                except TimeoutError:
                    if self._is_playing and self._send_queue.empty():
                        self._is_playing = False
                    continue

                await self._rtp.send_audio(pcm_data)
                if self._send_queue.empty():
                    self._is_playing = False

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"RTP sender error: {e}")


# =============================================================================
# Public API
# =============================================================================


async def run_agent(
    rtp_session: RTPSession,
    call_id: str,
    from_uri: str = "",
    to_uri: str = "",
    system_prompt: str | None = None,
    initial_message: str = "",
    greeting_audio: bytes = b"",
    greeting_text: str = "",
) -> None:
    """Run a voice agent session for an outbound SIP call."""
    agent = OutboundVoiceAgent(
        rtp_session=rtp_session,
        call_id=call_id,
        from_uri=from_uri,
        to_uri=to_uri,
        system_prompt=system_prompt,
        initial_message=initial_message,
        greeting_audio=greeting_audio,
        greeting_text=greeting_text,
    )
    await agent.run()
