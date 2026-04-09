"""Shared test helpers for ngrok, recording, and transcription.

Provides utilities for:
- Starting ngrok tunnels for webhook testing
- Recording call audio via Plivo
- Transcribing audio for verification
"""

from __future__ import annotations

import os
import subprocess
import time

from loguru import logger


def start_ngrok(port: int = 8000) -> str | None:
    """Start an ngrok tunnel and return the public URL.

    Returns:
        The public HTTPS URL or None if ngrok is not available.
    """
    try:
        proc = subprocess.Popen(
            ["ngrok", "http", str(port), "--log=stdout"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(3)

        import httpx

        resp = httpx.get("http://localhost:4040/api/tunnels", timeout=5.0)
        tunnels = resp.json().get("tunnels", [])
        for tunnel in tunnels:
            if tunnel.get("proto") == "https":
                return tunnel["public_url"]

        logger.warning("No HTTPS tunnel found")
        proc.terminate()
        return None
    except Exception as e:
        logger.warning(f"Failed to start ngrok: {e}")
        return None


def transcribe_audio(audio_path: str) -> str:
    """Transcribe an audio file using faster-whisper.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Transcribed text.
    """
    try:
        from faster_whisper import WhisperModel

        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(audio_path, beam_size=5)
        return " ".join(segment.text.strip() for segment in segments)
    except ImportError:
        logger.warning("faster-whisper not installed, skipping transcription")
        return ""


def get_plivo_client():
    """Create a Plivo REST client from environment variables."""
    import plivo

    auth_id = os.getenv("PLIVO_AUTH_ID", "")
    auth_token = os.getenv("PLIVO_AUTH_TOKEN", "")
    if not auth_id or not auth_token:
        return None
    return plivo.RestClient(auth_id=auth_id, auth_token=auth_token)
