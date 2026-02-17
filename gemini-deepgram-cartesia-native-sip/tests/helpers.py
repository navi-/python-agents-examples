"""Shared test helpers for live call tests."""

from __future__ import annotations

import os
import tempfile
import time

import httpx
import plivo


def wait_for_recording(
    client: plivo.RestClient, call_uuid: str, timeout: float = 30.0
) -> str | None:
    """Poll for a recording to appear for the given call UUID. Returns URL or None."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            recordings = client.recordings.list(call_uuid=call_uuid)
            objects = recordings.get("objects", []) if isinstance(recordings, dict) else []
            if not objects and hasattr(recordings, "objects"):
                objects = recordings.objects or []
            if objects:
                rec = objects[0]
                if isinstance(rec, dict):
                    url = rec.get("recording_url", "")
                else:
                    url = getattr(rec, "recording_url", "")
                if url:
                    return url
        except Exception:
            pass
        time.sleep(3)
    return None


def download_recording(url: str) -> bytes:
    """Download recording audio from URL."""
    resp = httpx.get(url, follow_redirects=True, timeout=30.0)
    resp.raise_for_status()
    return resp.content


def transcribe_audio(audio_data: bytes) -> str:
    """Transcribe MP3 audio using faster-whisper."""
    from faster_whisper import WhisperModel

    model = WhisperModel("base", device="cpu", compute_type="int8")

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(audio_data)
        tmp_path = f.name

    try:
        segments, _ = model.transcribe(tmp_path, language="en")
        return " ".join(seg.text.strip() for seg in segments).strip()
    finally:
        os.unlink(tmp_path)
