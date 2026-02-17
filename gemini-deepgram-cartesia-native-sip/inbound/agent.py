"""Inbound voice agent — Deepgram STT + Gemini LLM + Cartesia TTS + Silero VAD.

Handles incoming SIP calls via RTPSession. Audio flows:
  RTP (PCM16 8kHz from RTPSession) → Deepgram STT → Gemini LLM → Cartesia TTS → RTP

Silero VAD runs on incoming audio for barge-in detection.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import aiohttp
from google import genai
from google.genai import types
from loguru import logger

from utils import (
    CARTESIA_API_KEY,
    CARTESIA_MODEL,
    CARTESIA_SAMPLE_RATE,
    CARTESIA_VOICE_ID,
    DEEPGRAM_API_KEY,
    DEEPGRAM_MODEL,
    DEEPGRAM_SAMPLE_RATE,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    SileroVADProcessor,
    cartesia_to_plivo,
    pcm8k_to_vad,
)

if TYPE_CHECKING:
    from sip import RTPSession

# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = (Path(__file__).parent / "system_prompt.md").read_text().strip()
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", SYSTEM_PROMPT)

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
                            transcript = alternatives[0].get("transcript", "")
                            if transcript.strip():
                                logger.info(f"STT transcript: {transcript}")
                                await self.on_transcript(transcript)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"Deepgram WebSocket error: {msg.data}")
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
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session:
            await self._session.close()


# =============================================================================
# Cartesia TTS Client
# =============================================================================


class CartesiaTTS:
    """Text-to-speech using Cartesia HTTP API."""

    def __init__(self):
        self._session: aiohttp.ClientSession | None = None

    async def connect(self) -> None:
        """Initialize the HTTP session."""
        self._session = aiohttp.ClientSession()
        logger.info("Cartesia TTS client initialized")

    async def synthesize(self, text: str) -> bytes:
        """Synthesize text to speech using Cartesia HTTP API.

        Returns PCM16 audio at 24kHz.
        """
        if not self._session:
            await self.connect()

        url = "https://api.cartesia.ai/tts/bytes"
        headers = {
            "X-API-Key": CARTESIA_API_KEY,
            "Cartesia-Version": "2024-06-10",
            "Content-Type": "application/json",
        }
        payload = {
            "model_id": CARTESIA_MODEL,
            "transcript": text,
            "voice": {"mode": "id", "id": CARTESIA_VOICE_ID},
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": CARTESIA_SAMPLE_RATE,
            },
        }

        try:
            async with self._session.post(url, headers=headers, json=payload) as resp:
                if resp.status == 200:
                    audio_data = await resp.read()
                    logger.debug(f"Cartesia TTS: synthesized {len(audio_data)} bytes")
                    return audio_data
                else:
                    error = await resp.text()
                    logger.error(f"Cartesia TTS error {resp.status}: {error}")
                    return b""
        except Exception as e:
            logger.error(f"Cartesia TTS request failed: {e}")
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

    def seed_history(self, user_text: str, model_text: str) -> None:
        """Pre-seed conversation history (e.g., with a pre-generated greeting)."""
        self._conversation_history.append(
            types.Content(role="user", parts=[types.Part(text=user_text)])
        )
        self._conversation_history.append(
            types.Content(role="model", parts=[types.Part(text=model_text)])
        )

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
                types.Content(role="model", parts=[types.Part(text=assistant_text)])
            )

            return assistant_text

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return "I'm sorry, I'm having trouble processing that. Could you please repeat?"


# =============================================================================
# Voice Agent
# =============================================================================


class VoiceAgent:
    """Voice agent that orchestrates RTP + STT + LLM + TTS + VAD.

    Receives PCM16 8kHz audio from RTPSession, sends to Deepgram for
    transcription, gets responses from Gemini, synthesizes with Cartesia,
    and sends back through RTP. Silero VAD handles barge-in detection.
    """

    def __init__(
        self,
        rtp_session: RTPSession,
        call_id: str,
        from_uri: str = "",
        to_uri: str = "",
        system_prompt: str | None = None,
    ):
        self._rtp = rtp_session
        self.call_id = call_id
        self.from_uri = from_uri
        self.to_uri = to_uri

        self._running = False
        self._send_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._vad = SileroVADProcessor()
        self._is_responding = False
        self._current_response_task: asyncio.Task | None = None

        prompt = system_prompt or SYSTEM_PROMPT
        if from_uri:
            prompt += f"\n\nCurrent caller: {from_uri}"

        self._llm = GeminiLLM(prompt)
        self._tts = CartesiaTTS()
        self._stt = DeepgramSTT(on_transcript=self._handle_transcript)

    async def _handle_transcript(self, transcript: str) -> None:
        """Handle transcribed text from STT — cancel any existing response and start new."""
        if not transcript.strip():
            return

        logger.info(f"Processing user input: {transcript}")

        # Cancel any in-progress response
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

            # Get LLM response
            response_text = await self._llm.generate_response(transcript)
            if not response_text:
                self._is_responding = False
                return

            # Synthesize speech (Cartesia returns PCM16 24kHz)
            tts_audio = await self._tts.synthesize(response_text)
            if not tts_audio:
                self._is_responding = False
                return

            # Resample 24kHz → 8kHz for RTP
            pcm_8k = cartesia_to_plivo(tts_audio)

            # Queue PCM16 8kHz audio for sending via RTP
            await self._send_queue.put(pcm_8k)

        except asyncio.CancelledError:
            logger.debug("Response generation cancelled (barge-in)")
            raise
        finally:
            self._is_responding = False

    async def run(self) -> None:
        """Run the voice agent session."""
        logger.info(f"Starting voice agent for call {self.call_id}")
        self._running = True

        tasks = []
        try:
            await self._stt.connect()
            await self._tts.connect()

            # Synthesize and queue greeting
            greeting = (
                "Hello! This is Alex from TechFlow, powered by Gemini, Deepgram, "
                "and Cartesia over direct SIP. How can I help you today?"
            )
            greeting_audio = await self._tts.synthesize(greeting)
            if greeting_audio:
                pcm_8k = cartesia_to_plivo(greeting_audio)
                await self._send_queue.put(pcm_8k)

            # Run concurrent tasks
            tasks = [
                asyncio.create_task(self._receive_from_rtp(), name="rtp_rx"),
                asyncio.create_task(self._send_to_rtp(), name="rtp_tx"),
            ]

            done, _pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                if task.exception():
                    logger.error(f"Task {task.get_name()} failed: {task.exception()}")

        except Exception as e:
            logger.error(f"Voice agent error: {e}")
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
            logger.info(f"Voice agent ended for call {self.call_id}")

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

                # Send to Deepgram for transcription
                await self._stt.send_audio(pcm_data)

                # Run Silero VAD for barge-in
                vad_audio = pcm8k_to_vad(pcm_data)
                speech_started, _speech_ended = self._vad.process(vad_audio)

                if speech_started and self._is_responding:
                    logger.info("Barge-in detected, cancelling response")
                    await self._trigger_interruption()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"RTP receiver error: {e}")

    async def _trigger_interruption(self) -> None:
        """Cancel current response and drain send queue."""
        # Cancel in-progress response task
        if self._current_response_task and not self._current_response_task.done():
            self._current_response_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._current_response_task

        # Drain send queue (RTPSession will send silence automatically)
        while not self._send_queue.empty():
            try:
                self._send_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        self._is_responding = False

    async def _send_to_rtp(self) -> None:
        """Dequeue audio and send to RTPSession."""
        try:
            while self._running and self._rtp.is_running:
                try:
                    pcm_data = await asyncio.wait_for(
                        self._send_queue.get(), timeout=0.1
                    )
                except TimeoutError:
                    continue

                # RTPSession handles 20ms chunking and timing internally
                await self._rtp.send_audio(pcm_data)

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
) -> None:
    """Run a voice agent session for an incoming SIP call."""
    agent = VoiceAgent(
        rtp_session=rtp_session,
        call_id=call_id,
        from_uri=from_uri,
        to_uri=to_uri,
        system_prompt=system_prompt,
    )
    await agent.run()
