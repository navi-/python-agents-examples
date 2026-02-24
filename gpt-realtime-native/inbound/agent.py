"""Inbound voice agent — GPTRealtimeVoiceAgent engine for incoming calls.

Loads the inbound system prompt and provides run_agent() for handling
inbound call WebSocket sessions.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from loguru import logger

from utils import (
    SileroVADProcessor,
    openai_to_plivo,
    plivo_to_openai,
    plivo_to_vad,
)

load_dotenv()

# Agent configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GPT_REALTIME_MODEL = os.getenv("GPT_REALTIME_MODEL", "gpt-realtime-1.5")
GPT_REALTIME_VOICE = os.getenv("GPT_REALTIME_VOICE", "alloy")
OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime"

if TYPE_CHECKING:
    from fastapi import WebSocket

# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = (Path(__file__).parent / "system_prompt.md").read_text().strip()
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", SYSTEM_PROMPT)

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
# GPT Realtime Voice Agent
# =============================================================================


class GPTRealtimeVoiceAgent:
    """Manages a voice conversation session between Plivo and OpenAI GPT Realtime API."""

    def __init__(
        self,
        websocket: WebSocket,
        call_id: str,
        from_number: str = "",
        to_number: str = "",
        system_prompt: str | None = None,
    ):
        self.websocket = websocket
        self.call_id = call_id
        self.from_number = from_number
        self.to_number = to_number
        self.system_prompt = system_prompt or SYSTEM_PROMPT

        self._running = False
        self._send_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._vad = SileroVADProcessor()
        self._openai_ws = None
        self._is_responding = False
        self._interruption_event = asyncio.Event()
        # VAD is disabled until the initial greeting finishes playing
        self._vad_enabled_at: float = float("inf")

    def _build_tools(self) -> list[dict[str, Any]]:
        """Build tool definitions for OpenAI function calling."""
        return [
            {
                "type": "function",
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
            {
                "type": "function",
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
            {
                "type": "function",
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
            {
                "type": "function",
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
            {
                "type": "function",
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
        ]

    async def _handle_function_call(self, name: str, call_id: str, arguments: str) -> None:
        """Execute a function call and send the result back to OpenAI."""
        try:
            args = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            args = {}

        logger.info(f"Function call: {name} with args: {args}")

        try:
            if name == "check_order_status":
                result = await check_order_status(
                    order_number=args.get("order_number"),
                    email=args.get("email"),
                )
            elif name == "send_sms":
                result = await send_sms(
                    phone_number=args.get("phone_number", ""),
                    message=args.get("message", ""),
                )
            elif name == "schedule_callback":
                result = await schedule_callback(
                    phone_number=args.get("phone_number", ""),
                    reason=args.get("reason", ""),
                    preferred_time=args.get("preferred_time", ""),
                    department=args.get("department", "general"),
                )
            elif name == "transfer_call":
                result = await transfer_call(
                    department=args.get("department", "support"),
                    reason=args.get("reason", "Customer requested transfer"),
                )
            elif name == "end_call":
                logger.info(f"Ending call: {args.get('reason')}")
                self._running = False
                result = {"status": "call_ending", "reason": args.get("reason", "")}
            else:
                result = {"error": f"Unknown function: {name}"}

        except Exception as e:
            logger.error(f"Error in function {name}: {e}")
            result = {"error": str(e)}

        await self._openai_send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(result),
                },
            }
        )
        await self._openai_send(
            {
                "type": "response.create",
            }
        )

    async def _openai_send(self, message: dict[str, Any]) -> None:
        """Send a JSON message to the OpenAI WebSocket."""
        if self._openai_ws:
            await self._openai_ws.send(json.dumps(message))

    def _build_session_config(self) -> dict[str, Any]:
        """Build OpenAI Realtime GA session configuration."""
        system_prompt = self.system_prompt

        if self.from_number:
            call_time = datetime.now().strftime("%I:%M %p on %A, %B %d")
            system_prompt += f"""

## Current Call Context
- Caller's phone number: {self.from_number}
- Call ID: {self.call_id}
- Time: {call_time}

You can use the caller's phone number for SMS or callbacks without asking."""

        return {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": GPT_REALTIME_MODEL,
                "instructions": system_prompt,
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "turn_detection": None,
                    },
                    "output": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "voice": GPT_REALTIME_VOICE,
                    },
                },
                "tools": self._build_tools(),
            },
        }

    async def run(self) -> None:
        """Run the voice bot session."""
        import websockets

        logger.info(f"Starting GPT Realtime bot session for call {self.call_id}")
        self._running = True

        ws_url = f"{OPENAI_REALTIME_URL}?model={GPT_REALTIME_MODEL}"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        }

        try:
            async with websockets.connect(
                ws_url,
                additional_headers=headers,
                max_size=None,
            ) as openai_ws:
                self._openai_ws = openai_ws
                logger.info("Connected to OpenAI GPT Realtime API")

                await self._openai_send(self._build_session_config())

                while True:
                    msg = json.loads(await openai_ws.recv())
                    if msg.get("type") == "session.updated":
                        logger.info("OpenAI session configured")
                        break
                    elif msg.get("type") == "error":
                        logger.error(f"OpenAI session error: {msg}")
                        return

                # Trigger initial greeting from the system prompt
                await self._openai_send({"type": "response.create"})

                await self._run_streaming_tasks(openai_ws)

        except Exception as e:
            logger.error(f"Bot session error: {e}")
        finally:
            self._running = False
            self._openai_ws = None
            logger.info(f"Bot session ended for call {self.call_id}")

    async def _run_streaming_tasks(self, openai_ws) -> None:
        """Run the concurrent streaming tasks."""
        tasks = [
            asyncio.create_task(self._receive_from_plivo(), name="plivo_rx"),
            asyncio.create_task(self._receive_from_openai(openai_ws), name="openai_rx"),
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

    async def _trigger_interruption(self) -> None:
        """Cancel the current OpenAI response, drain the audio queue, and clear Plivo playback."""
        logger.info("Barge-in: cancelling response and clearing audio")
        await self._openai_send({"type": "response.cancel"})
        self._is_responding = False

        # Drain queued audio
        while not self._send_queue.empty():
            try:
                self._send_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Tell Plivo to stop playing buffered audio immediately
        self._interruption_event.set()
        try:
            await self.websocket.send_text(json.dumps({"event": "clearAudio"}))
        except Exception:
            pass

    async def _receive_from_plivo(self) -> None:
        """Receive audio from Plivo, run VAD, and forward to OpenAI."""
        try:
            while self._running:
                data = await self.websocket.receive_text()
                message = json.loads(data)
                event = message.get("event")

                if event == "media":
                    payload = message.get("media", {}).get("payload", "")
                    if payload:
                        mulaw_audio = base64.b64decode(payload)

                        # Always forward audio to OpenAI
                        pcm_24k = plivo_to_openai(mulaw_audio)
                        audio_b64 = base64.b64encode(pcm_24k).decode("utf-8")
                        await self._openai_send(
                            {
                                "type": "input_audio_buffer.append",
                                "audio": audio_b64,
                            }
                        )

                        # Skip VAD during grace period (avoids false triggers from
                        # initial noise/echo while the greeting plays)
                        if time.monotonic() < self._vad_enabled_at:
                            continue

                        vad_audio = plivo_to_vad(mulaw_audio)
                        speech_started, speech_ended = self._vad.process(vad_audio)

                        if speech_started and self._is_responding:
                            await self._trigger_interruption()

                        if speech_ended:
                            logger.debug("VAD: speech ended — committing audio buffer")
                            await self._openai_send({"type": "input_audio_buffer.commit"})
                            await self._openai_send({"type": "response.create"})
                            self._vad.reset()

                elif event == "text":
                    text = message.get("text", "")
                    if text:
                        logger.info(f"Injecting text: {text[:50]}...")
                        await self._openai_send(
                            {
                                "type": "conversation.item.create",
                                "item": {
                                    "type": "message",
                                    "role": "user",
                                    "content": [{"type": "input_text", "text": text}],
                                },
                            }
                        )
                        await self._openai_send({"type": "response.create"})

                elif event == "stop":
                    logger.info("Plivo stream stopped")
                    break

        except Exception as e:
            if "1000" not in str(e):
                logger.error(f"Plivo receiver error: {e}")

    async def _receive_from_openai(self, openai_ws) -> None:
        """Receive events from OpenAI and queue audio for Plivo."""
        first_response_done = False
        try:
            async for raw_message in openai_ws:
                if not self._running:
                    return

                message = json.loads(raw_message)
                event_type = message.get("type", "")

                if event_type == "response.output_audio.delta":
                    # Only queue audio if we haven't been interrupted
                    if not self._interruption_event.is_set():
                        audio_b64 = message.get("delta", "")
                        if audio_b64:
                            pcm_24k = base64.b64decode(audio_b64)
                            plivo_audio = openai_to_plivo(pcm_24k)
                            await self._send_queue.put(plivo_audio)

                elif event_type == "response.created":
                    self._is_responding = True
                    self._interruption_event.clear()

                elif event_type in ("response.done", "response.cancelled"):
                    self._is_responding = False
                    self._interruption_event.clear()
                    logger.debug(f"OpenAI: {event_type}")

                    # Enable VAD after the first response (greeting) finishes,
                    # with a small grace period to ignore echo/ring artifacts
                    if not first_response_done:
                        first_response_done = True
                        self._vad_enabled_at = time.monotonic() + 0.5
                        self._vad.reset()
                        logger.info("VAD enabled (grace period 0.5s)")

                elif event_type == "response.function_call_arguments.done":
                    await self._handle_function_call(
                        name=message.get("name", ""),
                        call_id=message.get("call_id", ""),
                        arguments=message.get("arguments", ""),
                    )

                elif event_type == "response.output_audio_transcript.delta":
                    transcript = message.get("delta", "")
                    if transcript:
                        logger.debug(f"Agent: {transcript}")

                elif event_type == "error":
                    error = message.get("error", message)
                    logger.error(f"OpenAI error: {error}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            if "close" not in str(e).lower():
                logger.error(f"OpenAI receiver error: {e}")

    async def _send_to_plivo(self) -> None:
        """Send queued audio to Plivo WebSocket in 20ms chunks."""
        PLIVO_CHUNK_SIZE = 160
        audio_buffer = bytearray()

        try:
            while self._running:
                try:
                    audio = await asyncio.wait_for(self._send_queue.get(), timeout=0.1)

                    # If interrupted, discard audio and reset buffer
                    if self._interruption_event.is_set():
                        audio_buffer.clear()
                        continue

                    audio_buffer.extend(audio)

                    while len(audio_buffer) >= PLIVO_CHUNK_SIZE:
                        # Check again in case interruption happened mid-send
                        if self._interruption_event.is_set():
                            audio_buffer.clear()
                            break

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
) -> None:
    """Run a voice agent session for an incoming call."""
    agent = GPTRealtimeVoiceAgent(
        websocket=websocket,
        call_id=call_id,
        from_number=from_number,
        to_number=to_number,
        system_prompt=system_prompt,
    )
    await agent.run()
