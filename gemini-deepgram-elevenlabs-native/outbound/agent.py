"""Outbound voice agent -- Gemini LLM + Deepgram STT + ElevenLabs TTS + call state.

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
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiohttp
from dotenv import load_dotenv
from google import genai
from google.genai import types
from loguru import logger

from utils import (
    SileroVADProcessor,
    elevenlabs_to_plivo,
    plivo_to_deepgram,
    plivo_to_vad,
)

if TYPE_CHECKING:
    from fastapi import WebSocket

load_dotenv()

# =============================================================================
# Agent Configuration
# =============================================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
DEEPGRAM_MODEL = os.getenv("DEEPGRAM_MODEL", "nova-2-phonecall")
DEEPGRAM_SAMPLE_RATE = 8000

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_flash_v2_5")

# =============================================================================
# System Prompt
# =============================================================================

_OUTBOUND_PROMPT_TEMPLATE = (
    (Path(__file__).parent / "system_prompt.md").read_text().strip()
)


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
    status: str = "initiating"
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
    outcome: str = ""


def determine_outcome(hangup_cause: str, duration: int) -> str:
    """Map Plivo hangup cause and duration to a high-level outcome."""
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
                "The call has been answered. Begin with your outbound "
                "greeting now. State your name, company, and that you are "
                f"reaching out regarding: {opening_reason}. "
                "Then ask if now is a good time."
            )
        else:
            initial_message = (
                "The call has been answered. Begin with your outbound "
                "greeting now. State your name, company, and why you are "
                "calling. Then ask if now is a good time."
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
                r
                for r in self._calls.values()
                if r.status in ("initiating", "ringing", "connected")
            ]

    def get_calls_by_campaign(
        self, campaign_id: str
    ) -> list[OutboundCallRecord]:
        """Return all calls for a given campaign."""
        with self._lock:
            return [
                r
                for r in self._calls.values()
                if r.campaign_id == campaign_id
            ]

    def reset(self) -> None:
        """Clear all records (useful for testing)."""
        with self._lock:
            self._calls.clear()


# =============================================================================
# Deepgram STT Client
# =============================================================================


class DeepgramSTT:
    """Real-time speech-to-text using Deepgram WebSocket API."""

    def __init__(self, on_transcript: callable):
        self.on_transcript = on_transcript
        self._ws = None
        self._session: aiohttp.ClientSession | None = None
        self._running = False
        self._receive_task: asyncio.Task | None = None

    async def connect(self) -> None:
        """Connect to Deepgram WebSocket."""
        self._session = aiohttp.ClientSession()
        self._running = True

        url = (
            f"wss://api.deepgram.com/v1/listen"
            f"?model={DEEPGRAM_MODEL}"
            f"&encoding=linear16"
            f"&sample_rate={DEEPGRAM_SAMPLE_RATE}"
            f"&channels=1"
            f"&punctuate=true"
            f"&interim_results=false"
        )

        headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}

        self._ws = await self._session.ws_connect(url, headers=headers)
        logger.info("Connected to Deepgram STT")

        self._receive_task = asyncio.create_task(self._receive_loop())

    async def _receive_loop(self) -> None:
        """Receive transcription results from Deepgram."""
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if data.get("type") == "Results":
                        channel = data.get("channel", {})
                        alternatives = channel.get("alternatives", [])
                        if alternatives:
                            transcript = alternatives[0].get(
                                "transcript", ""
                            )
                            if transcript.strip():
                                logger.info(
                                    f"STT transcript: {transcript}"
                                )
                                await self.on_transcript(transcript)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(
                        f"Deepgram WebSocket error: {msg.data}"
                    )
                    break
        except Exception as e:
            if self._running:
                logger.error(f"Deepgram receive error: {e}")

    async def send_audio(self, pcm_audio: bytes) -> None:
        """Send PCM audio to Deepgram."""
        if self._ws and not self._ws.closed:
            await self._ws.send_bytes(pcm_audio)

    async def close(self) -> None:
        """Close the Deepgram connection."""
        self._running = False
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._receive_task
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session:
            await self._session.close()


# =============================================================================
# ElevenLabs TTS Client
# =============================================================================


class ElevenLabsTTS:
    """Text-to-speech using ElevenLabs HTTP API."""

    def __init__(self):
        self._session: aiohttp.ClientSession | None = None

    async def connect(self) -> None:
        """Initialize the HTTP session."""
        self._session = aiohttp.ClientSession()
        logger.info("ElevenLabs TTS client initialized")

    async def synthesize(self, text: str) -> bytes:
        """Synthesize text to speech using ElevenLabs HTTP API."""
        if not self._session:
            await self.connect()

        url = (
            f"https://api.elevenlabs.io/v1/text-to-speech/"
            f"{ELEVENLABS_VOICE_ID}?output_format=pcm_24000"
        )
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "model_id": ELEVENLABS_MODEL,
        }

        try:
            async with self._session.post(
                url, headers=headers, json=payload
            ) as resp:
                if resp.status == 200:
                    audio_data = await resp.read()
                    logger.debug(
                        f"ElevenLabs TTS: synthesized {len(audio_data)} bytes"
                    )
                    return audio_data
                else:
                    error = await resp.text()
                    logger.error(
                        f"ElevenLabs TTS error {resp.status}: {error}"
                    )
                    return b""
        except Exception as e:
            logger.error(f"ElevenLabs TTS request failed: {e}")
            return b""

    async def close(self) -> None:
        """Close the session."""
        if self._session:
            await self._session.close()


# =============================================================================
# Gemini LLM Client
# =============================================================================


class GeminiLLM:
    """Conversational LLM using Google Gemini."""

    def __init__(self, system_prompt: str):
        self._client = genai.Client(api_key=GEMINI_API_KEY)
        self._system_prompt = system_prompt
        self._conversation_history: list[types.Content] = []

    async def generate_response(self, user_text: str) -> str:
        """Generate a response to user input."""
        self._conversation_history.append(
            types.Content(role="user", parts=[types.Part(text=user_text)])
        )

        try:
            response = await self._client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=self._conversation_history,
                config=types.GenerateContentConfig(
                    system_instruction=self._system_prompt,
                    temperature=0.7,
                    max_output_tokens=256,
                ),
            )

            assistant_text = response.text or ""
            logger.info(f"LLM response: {assistant_text[:100]}...")

            self._conversation_history.append(
                types.Content(
                    role="model", parts=[types.Part(text=assistant_text)]
                )
            )

            return assistant_text

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return (
                "I'm sorry, I'm having trouble processing that. "
                "Could you please repeat?"
            )


# =============================================================================
# Voice Agent
# =============================================================================


class VoiceAgent:
    """Voice agent that orchestrates STT, LLM, TTS, and VAD for outbound."""

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
        self.initial_message = initial_message

        self._running = False
        self._processing_lock = asyncio.Lock()
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._vad = SileroVADProcessor()
        self._is_responding = False

        prompt = system_prompt or SYSTEM_PROMPT
        if from_number:
            prompt += f"\n\nCurrent caller's phone number: {from_number}"

        self._llm = GeminiLLM(prompt)
        self._tts = ElevenLabsTTS()
        self._stt = DeepgramSTT(on_transcript=self._handle_transcript)

    async def _handle_transcript(self, transcript: str) -> None:
        """Handle transcribed text from STT."""
        async with self._processing_lock:
            if not transcript.strip():
                return

            logger.info(f"Processing user input: {transcript}")
            self._is_responding = True

            response_text = await self._llm.generate_response(transcript)
            if not response_text:
                self._is_responding = False
                return

            tts_audio = await self._tts.synthesize(response_text)
            if not tts_audio:
                self._is_responding = False
                return

            ulaw_audio = elevenlabs_to_plivo(tts_audio)
            await self._audio_queue.put(ulaw_audio)

    async def run(self) -> None:
        """Run the voice agent session."""
        logger.info(f"Starting outbound voice agent for call {self.call_id}")
        self._running = True

        try:
            await self._stt.connect()
            await self._tts.connect()

            # Generate initial greeting from LLM
            response_text = await self._llm.generate_response(
                self.initial_message
            )
            if response_text:
                greeting_audio = await self._tts.synthesize(response_text)
                if greeting_audio:
                    ulaw_audio = elevenlabs_to_plivo(greeting_audio)
                    await self._audio_queue.put(ulaw_audio)

            await self._run_streaming_tasks()

        except Exception as e:
            logger.error(f"Voice agent error: {e}")
        finally:
            self._running = False
            await self._stt.close()
            await self._tts.close()
            logger.info(f"Voice agent ended for call {self.call_id}")

    async def _run_streaming_tasks(self) -> None:
        """Run the concurrent streaming tasks."""
        tasks = [
            asyncio.create_task(
                self._receive_from_plivo(), name="plivo_rx"
            ),
            asyncio.create_task(
                self._receive_from_deepgram(), name="deepgram_rx"
            ),
            asyncio.create_task(self._send_to_plivo(), name="plivo_tx"),
        ]

        try:
            done, _pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                if task.exception():
                    logger.error(
                        f"Task {task.get_name()} failed: "
                        f"{task.exception()}"
                    )
        finally:
            self._running = False
            for task in tasks:
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

    async def _receive_from_plivo(self) -> None:
        """Receive audio from Plivo, run VAD, and forward to STT."""
        try:
            while self._running:
                data = await self.websocket.receive_text()
                message = json.loads(data)
                event = message.get("event")

                if event == "media":
                    payload = message.get("media", {}).get("payload", "")
                    if payload:
                        ulaw_audio = base64.b64decode(payload)

                        pcm_audio = plivo_to_deepgram(ulaw_audio)
                        await self._stt.send_audio(pcm_audio)

                        vad_audio = plivo_to_vad(ulaw_audio)
                        speech_started, speech_ended = self._vad.process(
                            vad_audio
                        )

                        if speech_started and self._is_responding:
                            logger.info(
                                "Barge-in detected, clearing audio queue"
                            )
                            self._is_responding = False
                            while not self._audio_queue.empty():
                                try:
                                    self._audio_queue.get_nowait()
                                except asyncio.QueueEmpty:
                                    break
                            await self.websocket.send_text(
                                json.dumps({"event": "clearAudio"})
                            )

                        if speech_ended:
                            logger.debug("VAD: speech ended")
                            self._vad.reset()

                elif event == "stop":
                    logger.info("Received stop event from Plivo")
                    break

        except Exception as e:
            if "1000" not in str(e):
                logger.error(f"Plivo receiver error: {e}")

    async def _receive_from_deepgram(self) -> None:
        """Monitor Deepgram STT - transcripts handled via callback."""
        try:
            while self._running:
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass

    async def _send_to_plivo(self) -> None:
        """Send queued audio to Plivo WebSocket in 20ms chunks."""
        PLIVO_CHUNK_SIZE = 160
        audio_buffer = bytearray()

        try:
            while self._running:
                try:
                    audio = await asyncio.wait_for(
                        self._audio_queue.get(), timeout=0.1
                    )
                    audio_buffer.extend(audio)

                    while len(audio_buffer) >= PLIVO_CHUNK_SIZE:
                        chunk = bytes(audio_buffer[:PLIVO_CHUNK_SIZE])
                        audio_buffer = audio_buffer[PLIVO_CHUNK_SIZE:]

                        message = {
                            "event": "playAudio",
                            "media": {
                                "contentType": "audio/x-mulaw",
                                "sampleRate": 8000,
                                "payload": base64.b64encode(chunk).decode(
                                    "utf-8"
                                ),
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
