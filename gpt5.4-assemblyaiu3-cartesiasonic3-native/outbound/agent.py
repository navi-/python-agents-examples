"""Outbound voice agent — AssemblyAI STT + GPT-5.4-mini LLM + Cartesia TTS + call state management.

Loads the outbound system prompt and provides run_agent() for handling
outbound call WebSocket sessions, plus CallManager for tracking call lifecycle.

Status state machine:
    initiating -> ringing -> connected -> completed
                         |-> no_answer
                |-> failed
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import random
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

from dotenv import load_dotenv
from loguru import logger
from openai import AsyncOpenAI

from utils import cartesia_to_plivo

load_dotenv()

# Agent configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")
ASSEMBLYAI_MODEL = os.getenv("ASSEMBLYAI_MODEL", "u3-rt-pro")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", "")
CARTESIA_MODEL = os.getenv("CARTESIA_MODEL", "sonic-3")
CARTESIA_VOICE_ID = os.getenv("CARTESIA_VOICE_ID", "6ccbfb76-1fc6-48f7-b71d-91ac6298247b")
CARTESIA_API_VERSION = os.getenv("CARTESIA_API_VERSION", "2025-04-16")

ASSEMBLYAI_WS_URL = "wss://streaming.assemblyai.com/v3/ws"
CARTESIA_WS_URL = "wss://api.cartesia.ai/tts/websocket"

END_OF_TURN_CONFIDENCE = float(os.getenv("END_OF_TURN_CONFIDENCE", "0.4"))
MIN_END_OF_TURN_SILENCE = int(os.getenv("MIN_END_OF_TURN_SILENCE_MS", "160"))
MAX_TURN_SILENCE = int(os.getenv("MAX_TURN_SILENCE_MS", "2400"))

if TYPE_CHECKING:
    from fastapi import WebSocket

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
# Tool Functions — replace these with your actual implementations
# =============================================================================


async def check_order_status(order_number: str | None, email: str | None) -> dict[str, Any]:
    """Look up order status. Replace with your actual implementation."""
    logger.info(f"Checking order: number={order_number}, email={email}")

    if not order_number and not email:
        return {"status": "error", "message": "Need order number or email"}

    statuses = [
        {
            "status": "shipped",
            "order_number": order_number or f"TF-{random.randint(100000, 999999)}",
            "shipping_carrier": "FedEx",
            "tracking_number": f"FX{random.randint(1000000000, 9999999999)}",
            "estimated_delivery": (datetime.now() + timedelta(days=2)).strftime("%B %d"),
            "items": "TechFlow Pro Annual Subscription",
        },
        {
            "status": "processing",
            "order_number": order_number or f"TF-{random.randint(100000, 999999)}",
            "message": "Order is being prepared and will ship within 24 hours",
            "items": "TechFlow Teams License (5 seats)",
        },
        {
            "status": "delivered",
            "order_number": order_number or f"TF-{random.randint(100000, 999999)}",
            "delivered_date": (datetime.now() - timedelta(days=1)).strftime("%B %d"),
            "signed_by": "Front Desk",
            "items": "TechFlow Enterprise Setup Kit",
        },
    ]
    return random.choice(statuses)


async def send_sms(phone_number: str, message: str) -> dict[str, Any]:
    """Send SMS to customer. Replace with your actual implementation."""
    logger.info(f"Sending SMS to {phone_number}: {message[:50]}...")

    if not phone_number:
        return {"status": "error", "message": "Phone number required"}

    return {
        "status": "sent",
        "phone_number": phone_number,
        "message_preview": message[:50] + "..." if len(message) > 50 else message,
        "confirmation_id": f"SMS{random.randint(100000, 999999)}",
    }


async def schedule_callback(
    phone_number: str, reason: str, preferred_time: str, department: str
) -> dict[str, Any]:
    """Schedule a callback. Replace with your actual implementation."""
    logger.info(f"Scheduling callback: {phone_number}, {department}")

    if not phone_number:
        return {"status": "error", "message": "Phone number required"}

    return {
        "status": "scheduled",
        "callback_id": f"CB{random.randint(100000, 999999)}",
        "phone_number": phone_number,
        "department": department,
        "scheduled_time": preferred_time or "within 2 business hours",
        "reason": reason,
    }


async def transfer_call(department: str, reason: str) -> dict[str, Any]:
    """Transfer call to human agent. Replace with your actual implementation."""
    logger.info(f"Transferring to {department}: {reason}")

    return {
        "status": "transferring",
        "department": department,
        "reason": reason,
        "estimated_wait": "less than 2 minutes",
    }


# =============================================================================
# Tool Definitions
# =============================================================================

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "check_order_status",
            "description": "Look up the status of a customer's order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_number": {
                        "type": "string",
                        "description": "Order number (usually starts with TF-)",
                    },
                    "email": {
                        "type": "string",
                        "description": "Customer's email if order number unavailable",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_sms",
            "description": "Send a text message to the customer's phone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "phone_number": {"type": "string", "description": "Phone number"},
                    "message": {"type": "string", "description": "Message content"},
                },
                "required": ["phone_number", "message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_callback",
            "description": "Schedule a callback from a specialist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "phone_number": {"type": "string", "description": "Phone number"},
                    "reason": {"type": "string", "description": "Why callback is needed"},
                    "preferred_time": {"type": "string", "description": "Preferred time"},
                    "department": {"type": "string", "description": "Department"},
                },
                "required": ["phone_number", "reason", "department"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transfer_call",
            "description": "Transfer call to human agent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "department": {"type": "string", "description": "Department"},
                    "reason": {"type": "string", "description": "Transfer reason"},
                },
                "required": ["department", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "end_call",
            "description": "End the call gracefully.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Reason for ending"},
                    "resolution": {
                        "type": "string",
                        "description": "How issue was resolved",
                    },
                },
            },
        },
    },
]


# =============================================================================
# Voice Agent
# =============================================================================


class VoiceAgent:
    """Manages a voice conversation using AssemblyAI STT, GPT LLM, and Cartesia TTS."""

    def __init__(
        self,
        websocket: WebSocket,
        call_id: str,
        from_number: str = "",
        to_number: str = "",
        system_prompt: str | None = None,
        initial_message: str = "Hello, I'm calling for help.",
    ):
        self.websocket = websocket
        self.call_id = call_id
        self.from_number = from_number
        self.to_number = to_number
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.initial_message = initial_message

        self._running = False
        self._send_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._assemblyai_ws = None
        self._cartesia_ws = None
        self._openai = AsyncOpenAI(api_key=OPENAI_API_KEY)

        self._messages: list[dict[str, Any]] = []
        self._is_responding = False
        self._response_task: asyncio.Task | None = None
        self._current_context_id: str | None = None

    def _build_system_prompt(self) -> str:
        """Build the system prompt with call context."""
        prompt = self.system_prompt

        if self.from_number:
            call_time = datetime.now().strftime("%I:%M %p on %A, %B %d")
            prompt += f"""

## Current Call Context
- Caller's phone number: {self.from_number}
- Call ID: {self.call_id}
- Time: {call_time}

You can use the caller's phone number for SMS or callbacks without asking."""

        return prompt

    async def run(self) -> None:
        """Run the voice agent session."""
        import websockets

        logger.info(f"Starting voice agent session for call {self.call_id}")
        self._running = True

        self._messages = [{"role": "system", "content": self._build_system_prompt()}]

        aai_params = urlencode({
            "sample_rate": "8000",
            "encoding": "pcm_mulaw",
            "speech_model": ASSEMBLYAI_MODEL,
            "format_turns": "false",
            "end_of_turn_confidence_threshold": str(END_OF_TURN_CONFIDENCE),
            "min_end_of_turn_silence_when_confident": str(MIN_END_OF_TURN_SILENCE),
            "max_turn_silence": str(MAX_TURN_SILENCE),
        })
        aai_url = f"{ASSEMBLYAI_WS_URL}?{aai_params}"
        aai_headers = {"Authorization": ASSEMBLYAI_API_KEY}

        cartesia_params = urlencode({
            "api_key": CARTESIA_API_KEY,
            "cartesia_version": CARTESIA_API_VERSION,
        })
        cartesia_url = f"{CARTESIA_WS_URL}?{cartesia_params}"

        try:
            async with (
                websockets.connect(
                    aai_url, additional_headers=aai_headers, max_size=None
                ) as aai_ws,
                websockets.connect(cartesia_url, max_size=None) as cartesia_ws,
            ):
                self._assemblyai_ws = aai_ws
                self._cartesia_ws = cartesia_ws

                begin_msg = json.loads(await aai_ws.recv())
                if begin_msg.get("type") != "Begin":
                    logger.error(f"Expected Begin event, got: {begin_msg}")
                    return
                logger.info(f"AssemblyAI session started: id={begin_msg.get('id')}")

                await self._generate_response(self.initial_message)

                await self._run_streaming_tasks()

        except Exception as e:
            logger.error(f"Voice agent session error: {e}")
        finally:
            self._running = False
            if self._response_task and not self._response_task.done():
                self._response_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._response_task
            self._assemblyai_ws = None
            self._cartesia_ws = None
            logger.info(f"Voice agent session ended for call {self.call_id}")

    async def _run_streaming_tasks(self) -> None:
        """Run the concurrent streaming tasks."""
        tasks = [
            asyncio.create_task(self._receive_from_plivo(), name="plivo_rx"),
            asyncio.create_task(self._receive_from_assemblyai(), name="assemblyai_rx"),
            asyncio.create_task(self._send_to_plivo(), name="plivo_tx"),
        ]

        try:
            done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task.exception():
                    logger.error(f"Task {task.get_name()} failed: {task.exception()}")
        finally:
            self._running = False
            for task in tasks:
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

    async def _receive_from_plivo(self) -> None:
        """Receive audio from Plivo and forward to AssemblyAI for transcription.

        AssemblyAI requires audio chunks between 50ms and 1000ms. Plivo sends
        160-byte chunks (20ms at 8kHz μ-law), so we buffer to 100ms (800 bytes)
        before forwarding.
        """
        AAI_MIN_CHUNK = 800  # 100ms at 8kHz μ-law
        aai_buffer = bytearray()

        try:
            while self._running:
                data = await self.websocket.receive_text()
                message = json.loads(data)
                event = message.get("event")

                if event == "media":
                    payload = message.get("media", {}).get("payload", "")
                    if payload:
                        mulaw_audio = base64.b64decode(payload)
                        aai_buffer.extend(mulaw_audio)
                        if len(aai_buffer) >= AAI_MIN_CHUNK and self._assemblyai_ws:
                            await self._assemblyai_ws.send(bytes(aai_buffer))
                            aai_buffer.clear()

                elif event == "text":
                    # Text injection (for testing) — bypass STT, send directly to LLM
                    text = message.get("text", "").strip()
                    if text:
                        logger.info(f"Text event received: '{text}'")
                        await self._generate_response(text)

                elif event == "stop":
                    if aai_buffer and self._assemblyai_ws:
                        await self._assemblyai_ws.send(bytes(aai_buffer))
                        aai_buffer.clear()
                    logger.info("Plivo stream stopped")
                    break

        except Exception as e:
            if "1000" not in str(e):
                logger.error(f"Plivo receiver error: {e}")

    async def _receive_from_assemblyai(self) -> None:
        """Receive Turn events from AssemblyAI and handle turn detection."""
        try:
            async for raw_message in self._assemblyai_ws:
                if not self._running:
                    return

                message = json.loads(raw_message)
                msg_type = message.get("type", "")

                if msg_type == "Turn":
                    transcript = message.get("transcript", "").strip()
                    end_of_turn = message.get("end_of_turn", False)

                    if not transcript:
                        continue

                    if self._is_responding:
                        logger.info(f"Barge-in detected: '{transcript[:50]}...'")
                        await self._handle_barge_in()

                    if end_of_turn:
                        logger.info(f"User turn complete: '{transcript}'")
                        await self._generate_response(transcript)

                elif msg_type == "Termination":
                    logger.info(
                        f"AssemblyAI session terminated: "
                        f"audio={message.get('audio_duration_seconds', 0)}s, "
                        f"session={message.get('session_duration_seconds', 0)}s"
                    )
                    break

                elif msg_type == "Error":
                    logger.error(f"AssemblyAI error: {message}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            if "close" not in str(e).lower():
                logger.error(f"AssemblyAI receiver error: {e}")

    async def _handle_barge_in(self) -> None:
        """Handle user interruption (barge-in) during agent response."""
        if self._response_task and not self._response_task.done():
            self._response_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._response_task
            self._response_task = None

        if self._current_context_id and self._cartesia_ws:
            try:
                await self._cartesia_ws.send(json.dumps({
                    "context_id": self._current_context_id,
                    "cancel": True,
                }))
            except Exception as e:
                logger.debug(f"Error cancelling Cartesia context: {e}")
            self._current_context_id = None

        while not self._send_queue.empty():
            try:
                self._send_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        try:
            await self.websocket.send_text(json.dumps({"event": "clearAudio"}))
        except Exception as e:
            logger.debug(f"Error sending clearAudio: {e}")

        self._is_responding = False
        logger.info("Barge-in handled: response cancelled, audio cleared")

    async def _generate_response(self, user_text: str) -> None:
        """Start generating an agent response for the user's text."""
        self._messages.append({"role": "user", "content": user_text})

        if self._response_task and not self._response_task.done():
            self._response_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._response_task

        self._is_responding = True
        self._response_task = asyncio.create_task(
            self._response_pipeline(), name="response_pipeline"
        )

    async def _response_pipeline(self) -> None:
        """Pipeline: GPT streaming -> Cartesia TTS -> Plivo audio queue."""
        context_id = str(uuid.uuid4())
        self._current_context_id = context_id
        full_response = ""

        try:
            stream = await self._openai.chat.completions.create(
                model=OPENAI_MODEL,
                messages=self._messages,
                tools=TOOL_DEFINITIONS,
                stream=True,
                max_completion_tokens=500,
            )

            tool_calls_data: dict[int, dict[str, str]] = {}
            text_buffer = ""

            async for chunk in stream:
                if not self._running:
                    break

                choice = chunk.choices[0] if chunk.choices else None
                if not choice:
                    continue

                delta = choice.delta

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_data:
                            tool_calls_data[idx] = {
                                "id": tc.id or "",
                                "name": "",
                                "arguments": "",
                            }
                        if tc.id:
                            tool_calls_data[idx]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                tool_calls_data[idx]["name"] = tc.function.name
                            if tc.function.arguments:
                                tool_calls_data[idx]["arguments"] += tc.function.arguments
                    continue

                if delta.content:
                    text = delta.content
                    full_response += text
                    text_buffer += text

                    if any(text_buffer.endswith(p) for p in [".", "!", "?", ",", ";", ":"]):
                        await self._send_to_cartesia(text_buffer, context_id)
                        text_buffer = ""

                if choice.finish_reason:
                    break

            if text_buffer.strip():
                await self._send_to_cartesia(text_buffer, context_id, is_last=True)
            else:
                await self._send_to_cartesia("", context_id, is_last=True)

            if tool_calls_data:
                await self._handle_tool_calls(tool_calls_data, context_id)
                return

            await self._collect_cartesia_audio(context_id)

            if full_response:
                self._messages.append({"role": "assistant", "content": full_response})

        except asyncio.CancelledError:
            logger.debug("Response pipeline cancelled (barge-in)")
            raise
        except Exception as e:
            logger.error(f"Response pipeline error: {e}")
        finally:
            self._is_responding = False
            self._current_context_id = None

    async def _send_to_cartesia(
        self, text: str, context_id: str, is_last: bool = False
    ) -> None:
        """Send a text chunk to Cartesia for TTS synthesis."""
        if not self._cartesia_ws:
            return
        if not text.strip() and not is_last:
            return

        message = {
            "context_id": context_id,
            "model_id": CARTESIA_MODEL,
            "transcript": text,
            "voice": {"mode": "id", "id": CARTESIA_VOICE_ID},
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": 24000,
            },
            "language": "en",
            "continue": not is_last,
        }

        try:
            await self._cartesia_ws.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending to Cartesia: {e}")

    async def _collect_cartesia_audio(self, context_id: str) -> None:
        """Collect audio responses from Cartesia for a given context."""
        if not self._cartesia_ws:
            return

        try:
            while True:
                raw = await asyncio.wait_for(self._cartesia_ws.recv(), timeout=10.0)
                message = json.loads(raw)

                if message.get("context_id") != context_id:
                    continue

                if message.get("type") == "chunk" and message.get("data"):
                    pcm_24k = base64.b64decode(message["data"])
                    plivo_audio = cartesia_to_plivo(pcm_24k)
                    await self._send_queue.put(plivo_audio)

                if message.get("done", False):
                    break

        except TimeoutError:
            logger.debug("Cartesia audio collection timed out")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug(f"Cartesia audio collection error: {e}")

    async def _handle_tool_calls(
        self, tool_calls_data: dict[int, dict[str, str]], context_id: str
    ) -> None:
        """Execute tool calls and generate a follow-up response."""
        tool_calls_msg = []
        for idx in sorted(tool_calls_data):
            tc = tool_calls_data[idx]
            tool_calls_msg.append({
                "id": tc["id"],
                "type": "function",
                "function": {"name": tc["name"], "arguments": tc["arguments"]},
            })

        self._messages.append({
            "role": "assistant",
            "tool_calls": tool_calls_msg,
        })

        for tc in tool_calls_msg:
            name = tc["function"]["name"]
            try:
                raw_args = tc["function"]["arguments"]
                args = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                args = {}

            logger.info(f"Tool call: {name} with args: {args}")
            result = await self._execute_tool(name, args)

            self._messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps(result),
            })

        self._is_responding = True
        self._current_context_id = str(uuid.uuid4())
        await self._response_pipeline_no_tools()

    async def _response_pipeline_no_tools(self) -> None:
        """Generate a follow-up response (no tools) after tool execution."""
        context_id = self._current_context_id or str(uuid.uuid4())
        full_response = ""

        try:
            stream = await self._openai.chat.completions.create(
                model=OPENAI_MODEL,
                messages=self._messages,
                stream=True,
                max_completion_tokens=500,
            )

            text_buffer = ""

            async for chunk in stream:
                if not self._running:
                    break

                choice = chunk.choices[0] if chunk.choices else None
                if not choice or not choice.delta or not choice.delta.content:
                    if choice and choice.finish_reason:
                        break
                    continue

                text = choice.delta.content
                full_response += text
                text_buffer += text

                if any(text_buffer.endswith(p) for p in [".", "!", "?", ",", ";", ":"]):
                    await self._send_to_cartesia(text_buffer, context_id)
                    text_buffer = ""

            if text_buffer.strip():
                await self._send_to_cartesia(text_buffer, context_id, is_last=True)
            else:
                await self._send_to_cartesia("", context_id, is_last=True)

            await self._collect_cartesia_audio(context_id)

            if full_response:
                self._messages.append({"role": "assistant", "content": full_response})

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Follow-up response error: {e}")
        finally:
            self._is_responding = False
            self._current_context_id = None

    async def _execute_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool function by name."""
        try:
            if name == "check_order_status":
                return await check_order_status(
                    order_number=args.get("order_number"),
                    email=args.get("email"),
                )
            elif name == "send_sms":
                return await send_sms(
                    phone_number=args.get("phone_number", ""),
                    message=args.get("message", ""),
                )
            elif name == "schedule_callback":
                return await schedule_callback(
                    phone_number=args.get("phone_number", ""),
                    reason=args.get("reason", ""),
                    preferred_time=args.get("preferred_time", ""),
                    department=args.get("department", "general"),
                )
            elif name == "transfer_call":
                return await transfer_call(
                    department=args.get("department", "support"),
                    reason=args.get("reason", "Customer requested transfer"),
                )
            elif name == "end_call":
                logger.info(f"Ending call: {args.get('reason')}")
                self._running = False
                return {"status": "call_ending", "reason": args.get("reason", "")}
            else:
                return {"error": f"Unknown function: {name}"}
        except Exception as e:
            logger.error(f"Error in tool {name}: {e}")
            return {"error": str(e)}

    async def _send_to_plivo(self) -> None:
        """Send queued audio to Plivo WebSocket in 20ms chunks."""
        PLIVO_CHUNK_SIZE = 160
        audio_buffer = bytearray()

        try:
            while self._running:
                try:
                    audio = await asyncio.wait_for(self._send_queue.get(), timeout=0.1)
                    audio_buffer.extend(audio)

                    while len(audio_buffer) >= PLIVO_CHUNK_SIZE:
                        chunk = bytes(audio_buffer[:PLIVO_CHUNK_SIZE])
                        audio_buffer = audio_buffer[PLIVO_CHUNK_SIZE:]

                        message = {
                            "event": "playAudio",
                            "media": {
                                "contentType": "audio/x-mulaw",
                                "sampleRate": 8000,
                                "payload": base64.b64encode(chunk).decode("utf-8"),
                            },
                        }
                        await self.websocket.send_text(json.dumps(message))

                except TimeoutError:
                    continue

        except asyncio.CancelledError:
            pass


# =============================================================================
# Public API
# =============================================================================


async def run_agent(
    websocket: WebSocket,
    call_id: str,
    from_number: str = "",
    to_number: str = "",
    system_prompt: str | None = None,
    initial_message: str = "Hello, I'm calling for help.",
) -> None:
    """Run a voice agent session for an outbound call."""
    agent = VoiceAgent(
        websocket=websocket,
        call_id=call_id,
        from_number=from_number,
        to_number=to_number,
        system_prompt=system_prompt,
        initial_message=initial_message,
    )
    await agent.run()
