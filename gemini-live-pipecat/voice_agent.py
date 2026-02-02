"""
Voice agent using Pipecat with Google Gemini Live API for speech-to-speech conversations.

This module provides a voice agent that:
- Uses Pipecat framework for orchestration
- Connects to Google's Gemini Live API for real-time speech processing
- Handles bidirectional audio streaming with Plivo telephony
- Supports function calling for actions during conversations

Usage:
    python voice_agent.py
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import Response
from loguru import logger
from pipecat.frames.frames import LLMMessagesUpdateFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.serializers.plivo import PlivoFrameSerializer
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from plivo import plivoxml

load_dotenv()

# Configuration
SERVER_PORT = int(os.getenv("SERVER_PORT", "8080"))
PUBLIC_URL = os.getenv("PUBLIC_URL", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash-native-audio-preview-12-2025")
GEMINI_VOICE = os.getenv("GEMINI_VOICE", "Puck")
PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID", "")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN", "")

# System prompt for the voice agent
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    """You are a helpful voice assistant. Keep your responses concise and natural
for voice conversation. Be friendly and professional. Ask clarifying questions
when needed. Your responses will be spoken aloud, so avoid special characters
and speak naturally.""",
)

# Store active pipeline tasks
active_tasks: dict[str, PipelineTask] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for FastAPI startup/shutdown."""
    logger.info("Starting Gemini Live Pipecat Voice Agent")
    yield
    logger.info("Shutting down...")
    for task in active_tasks.values():
        await task.cancel()


app = FastAPI(
    title="Gemini Live Pipecat Voice Agent",
    description="Voice agent using Pipecat with Google Gemini Live API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
async def health_check() -> dict:
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "gemini-live-pipecat",
        "model": GEMINI_MODEL,
    }


@app.post("/answer")
async def answer_webhook(request: Request) -> Response:
    """
    Plivo calls this endpoint when a call comes in.
    Returns XML to start streaming audio via WebSocket.
    """
    base_url = PUBLIC_URL
    if not base_url:
        logger.error("PUBLIC_URL not configured")
        return Response(content="PUBLIC_URL not configured", status_code=500)

    # Create Plivo XML response
    response = plivoxml.ResponseElement()

    # Start audio stream to WebSocket
    ws_url = base_url.replace("https://", "wss://").replace("http://", "ws://")
    ws_url = f"{ws_url}/ws/stream"

    logger.info(f"WebSocket URL: {ws_url}")

    stream = plivoxml.StreamElement(
        ws_url,
        bidirectional=True,
        keepCallAlive=True,
        contentType="audio/x-mulaw;rate=8000",
    )

    response.add(stream)

    return Response(content=response.to_string(), media_type="application/xml")


@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    WebSocket endpoint that handles real-time audio streaming from Plivo.
    Uses Pipecat with Gemini Live for speech-to-speech processing.
    """
    logger.info("WebSocket connection request received")

    await websocket.accept()
    logger.info("WebSocket connection accepted")

    call_id = None

    try:
        # Read the start message from Plivo
        logger.info("Waiting for start message from Plivo...")
        start_data = await websocket.receive_text()
        start_message = json.loads(start_data)

        logger.info(f"Received start message: {start_message}")

        # Extract Plivo-specific IDs from the start event
        start_info = start_message.get("start", {})
        stream_id = start_info.get("streamId")
        call_id = start_info.get("callId")

        if not stream_id or not call_id:
            logger.error("Missing stream_id or call_id")
            await websocket.close()
            return

        # Create Plivo serializer with authentication
        serializer = PlivoFrameSerializer(
            stream_id=stream_id,
            call_id=call_id,
            auth_id=PLIVO_AUTH_ID,
            auth_token=PLIVO_AUTH_TOKEN,
        )

        # Create transport with Plivo serializer
        transport = FastAPIWebsocketTransport(
            websocket=websocket,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                add_wav_header=False,
                serializer=serializer,
            ),
        )

        # Initialize Gemini Live service for speech-to-speech
        llm = GeminiLiveLLMService(
            api_key=GEMINI_API_KEY,
            model=GEMINI_MODEL,
            voice_id=GEMINI_VOICE,
            system_instruction=SYSTEM_PROMPT,
        )

        # Create the pipeline
        # For Gemini Multimodal Live, audio flows directly through the LLM service
        # which handles both speech recognition and speech synthesis natively
        pipeline = Pipeline(
            [
                transport.input(),  # Receive audio from Plivo
                llm,  # Gemini Multimodal Live (speech-to-speech)
                transport.output(),  # Send audio to Plivo
            ]
        )

        logger.info("Creating pipeline task...")

        # Create and run the pipeline task
        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
        )

        # Store the task
        active_tasks[call_id] = task

        logger.info("Queuing initial greeting message...")

        # Queue initial message to trigger greeting
        await task.queue_frames(
            [LLMMessagesUpdateFrame([{"role": "user", "content": "Hello!"}], run_llm=True)]
        )

        logger.info("Running pipeline...")

        # Run the pipeline
        runner = PipelineRunner()
        await runner.run(task)

        logger.info("Pipeline completed")

    except Exception as e:
        logger.error(f"Error in WebSocket handler: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Cleanup
        if call_id and call_id in active_tasks:
            del active_tasks[call_id]

        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting Gemini Live Pipecat Voice Agent on port {SERVER_PORT}")
    logger.info(f"Webhook URL: {PUBLIC_URL}/answer")

    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
