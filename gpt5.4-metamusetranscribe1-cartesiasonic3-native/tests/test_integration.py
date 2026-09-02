"""
Integration tests for the Babel voice agent
(Meta Muse Voice Transcribe + GPT-5.4 mini + Cartesia Sonic-3).

Test Levels:
1. Unit Tests - audio conversion, phone normalization, Muse event parsing,
   language-tag parsing, diarized turn formatting (offline, no API keys)
2. Local Integration - starts the server, tests the Plivo WebSocket flow
3. API Integration - OpenAI, Muse Voice Transcribe, Cartesia connections
4. Plivo Integration - Plivo API configuration

Run tests:
    uv run pytest tests/test_integration.py -v

Run specific test level:
    uv run pytest tests/test_integration.py -v -k "unit"
    uv run pytest tests/test_integration.py -v -k "local"
    uv run pytest tests/test_integration.py -v -k "muse"
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import signal
import struct
import subprocess
import sys
import time
import uuid

import httpx
import plivo
import pytest
import websockets
from dotenv import load_dotenv

from utils import pcm_to_ulaw, ulaw_to_pcm

load_dotenv()

# Configuration from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
META_API_KEY = os.getenv("META_API_KEY", "")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", "")
PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID", "")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN", "")
PLIVO_PHONE_NUMBER = os.getenv("PLIVO_PHONE_NUMBER", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")

TEST_PORT = 18001
LOCAL_WS_URL = f"ws://localhost:{TEST_PORT}/ws"
LOCAL_HTTP_URL = f"http://localhost:{TEST_PORT}"


# =============================================================================
# UNIT TESTS - Audio conversion
# =============================================================================


class TestUnitAudioConversion:
    """Unit tests for audio format conversion."""

    def test_ulaw_to_pcm_conversion(self):
        """Test u-law to PCM conversion."""
        ulaw_silence = b"\xff" * 160
        pcm_audio = ulaw_to_pcm(ulaw_silence)

        samples = struct.unpack(f"{len(pcm_audio) // 2}h", pcm_audio)
        avg_amplitude = sum(abs(s) for s in samples) / len(samples)

        assert len(pcm_audio) == 320  # 160 samples * 2 bytes
        assert avg_amplitude < 100  # Should be near silence

    def test_pcm_to_ulaw_conversion(self):
        """Test PCM to u-law conversion."""
        pcm_silence = b"\x00" * 320
        ulaw_audio = pcm_to_ulaw(pcm_silence)

        assert len(ulaw_audio) == 160  # Half the size

    def test_audio_roundtrip(self):
        """Test that audio survives roundtrip conversion."""
        samples = []
        for i in range(160):
            sample = int(16000 * math.sin(2 * math.pi * 440 * i / 8000))
            samples.append(sample)
        pcm_original = struct.pack(f"{len(samples)}h", *samples)

        ulaw = pcm_to_ulaw(pcm_original)
        pcm_restored = ulaw_to_pcm(ulaw)

        original_samples = struct.unpack(f"{len(pcm_original) // 2}h", pcm_original)
        restored_samples = struct.unpack(f"{len(pcm_restored) // 2}h", pcm_restored)

        correlation = sum(o * r for o, r in zip(original_samples, restored_samples, strict=True))
        orig_energy = sum(o * o for o in original_samples)
        rest_energy = sum(r * r for r in restored_samples)

        if orig_energy > 0 and rest_energy > 0:
            normalized_corr = correlation / (orig_energy * rest_energy) ** 0.5
            assert normalized_corr > 0.9, "Audio quality degraded too much"

    def test_plivo_to_muse_pcm16_resample(self):
        """plivo_to_muse converts u-law 8kHz to PCM16 24kHz (pcm16 input mode)."""
        from utils import plivo_to_muse

        # 160 bytes u-law = 20ms at 8kHz (one Plivo packet)
        pcm_24k = plivo_to_muse(b"\xff" * 160)

        # 8kHz -> 24kHz = 3x samples, 2 bytes each
        assert len(pcm_24k) == 160 * 3 * 2

    def test_cartesia_to_plivo_conversion(self):
        """Test Cartesia (PCM16 24kHz) to Plivo (u-law 8kHz) conversion."""
        from utils import cartesia_to_plivo

        # 480 samples at 24kHz = 20ms, 2 bytes each
        ulaw_8k = cartesia_to_plivo(b"\x00" * 960)

        assert len(ulaw_8k) == 160  # 20ms of u-law at 8kHz

    def test_plivo_to_vad_shape(self):
        """plivo_to_vad yields float32 16kHz samples in [-1, 1]."""
        import numpy as np

        from utils import plivo_to_vad

        audio = plivo_to_vad(b"\xff" * 160)
        assert audio.dtype == np.float32
        assert len(audio) == 320  # 160 samples * 2x resample
        assert float(np.abs(audio).max()) <= 1.0


# =============================================================================
# UNIT TESTS - Phone normalization
# =============================================================================


class TestUnitPhoneNormalization:
    """Unit tests for phone number normalization."""

    def test_normalize_e164_format(self):
        from utils import normalize_phone_number

        assert normalize_phone_number("+16572338892") == "16572338892"

    def test_normalize_with_spaces(self):
        from utils import normalize_phone_number

        assert normalize_phone_number("+1 657-233-8892") == "16572338892"

    def test_normalize_local_format(self):
        from utils import normalize_phone_number

        assert normalize_phone_number("(657) 233-8892", "US") == "16572338892"


# =============================================================================
# UNIT TESTS - Muse Voice Transcribe event parsing
# =============================================================================


class TestUnitMuseTranscribeSTT:
    """Unit tests for MuseTranscribeSTT message parsing (no network)."""

    def test_session_config_shape(self):
        from inbound.agent import MuseTranscribeSTT

        stt = MuseTranscribeSTT(keywords=["Babel", "paella"], context="restaurant line")
        cfg = stt.build_session_config()

        assert cfg["type"] == "transcription_session.update"
        session = cfg["session"]
        assert session["input_audio_format"] in ("g711_ulaw", "pcm16")
        transcription = session["input_audio_transcription"]
        assert transcription["model"].startswith("muse-voice-transcribe")
        assert transcription["keywords"] == ["Babel", "paella"]
        assert "paella" in transcription["prompt"]
        assert transcription["diarization"]["enabled"] is True
        assert session["turn_detection"]["create_response"] is False

    def test_delta_accumulates_then_completes(self):
        from inbound.agent import MuseTranscribeSTT

        stt = MuseTranscribeSTT()
        seg1 = stt.parse_event(
            {
                "type": "conversation.item.input_audio_transcription.delta",
                "item_id": "item_1",
                "delta": "Hola, ",
                "speaker": 2,
            }
        )
        seg2 = stt.parse_event(
            {
                "type": "conversation.item.input_audio_transcription.delta",
                "item_id": "item_1",
                "delta": "quiero una paella",
            }
        )
        assert seg1 is not None and not seg1.is_final
        assert seg2 is not None and seg2.text == "Hola, quiero una paella"
        assert seg2.speaker == "2"  # speaker remembered from the first delta
        assert stt.pending_text() == "Hola, quiero una paella"

        final = stt.parse_event(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item_1",
                "transcript": "Hola, quiero una paella.",
                "speaker": "2",
            }
        )
        assert final is not None and final.is_final
        assert final.text == "Hola, quiero una paella."
        assert final.speaker == "2"
        assert stt.pending_text() == ""

    def test_completed_without_transcript_uses_partials(self):
        from inbound.agent import MuseTranscribeSTT

        stt = MuseTranscribeSTT()
        stt.parse_event(
            {
                "type": "conversation.item.input_audio_transcription.delta",
                "item_id": "x",
                "delta": "two tacos",
            }
        )
        final = stt.parse_event(
            {"type": "conversation.item.input_audio_transcription.completed", "item_id": "x"}
        )
        assert final is not None and final.text == "two tacos"

    def test_extract_speaker_tolerates_shapes(self):
        from inbound.agent import MuseTranscribeSTT

        assert MuseTranscribeSTT.extract_speaker({"speaker": 1}) == "1"
        assert MuseTranscribeSTT.extract_speaker({"speaker_id": "SPEAKER_03"}) == "SPEAKER_03"
        assert MuseTranscribeSTT.extract_speaker({"item": {"speaker_label": "2"}}) == "2"
        assert MuseTranscribeSTT.extract_speaker({"segments": [{"speaker": 4}]}) == "4"
        assert MuseTranscribeSTT.extract_speaker({"type": "noise"}) == ""

    def test_ignores_unrelated_events(self):
        from inbound.agent import MuseTranscribeSTT

        stt = MuseTranscribeSTT()
        assert stt.parse_event({"type": "input_audio_buffer.speech_started"}) is None
        assert stt.parse_event({"type": "transcription_session.updated"}) is None
        assert stt.parse_event({"type": "response.text.delta", "delta": "x"}) is None

    def test_update_biasing_merges_keywords_offline(self):
        from inbound.agent import MuseTranscribeSTT

        stt = MuseTranscribeSTT(keywords=["Babel"])
        asyncio.run(stt.update_biasing(keywords=["Priya", "Babel", "Carlos"]))
        assert stt.keywords == ["Babel", "Priya", "Carlos"]


# =============================================================================
# UNIT TESTS - Reply language tag + diarized turn formatting
# =============================================================================


class TestUnitLanguageTagParser:
    """Unit tests for the <xx> reply language tag parser."""

    def test_tag_split_across_deltas(self):
        from inbound.agent import LanguageTagParser

        p = LanguageTagParser()
        assert p.feed("<") == ""
        assert p.feed("es") == ""
        assert p.feed("> Claro") == "Claro"
        assert p.language == "es"
        assert p.feed(" que si.") == " que si."

    def test_no_tag_passes_through(self):
        from inbound.agent import LanguageTagParser

        p = LanguageTagParser()
        assert p.feed("Welcome") == "Welcome"
        assert p.language == "en"

    def test_unknown_language_falls_back(self):
        from inbound.agent import LanguageTagParser

        p = LanguageTagParser()
        assert p.feed("<xx> hello") == "hello"
        assert p.language == "en"

    def test_region_tag_and_hindi(self):
        from inbound.agent import LanguageTagParser

        p = LanguageTagParser()
        assert p.feed("<hi-IN> ज़रूर") == "ज़रूर"
        assert p.language == "hi"

    def test_flush_returns_held_text(self):
        from inbound.agent import LanguageTagParser

        p = LanguageTagParser()
        assert p.feed("<e") == ""
        assert p.flush() == "<e"


class TestUnitTurnFormatting:
    """Diarized segments become speaker-tagged lines for the LLM."""

    def test_format_turn_with_names(self):
        from inbound.agent import TranscriptSegment, VoiceAgent

        agent = VoiceAgent.__new__(VoiceAgent)
        from inbound.agent import TableState

        agent.table = TableState()
        agent.table.speakers["2"] = {"name": "Priya"}
        text = agent._format_turn(
            [
                TranscriptSegment(text="Two tacos please.", speaker="1", is_final=True),
                TranscriptSegment(text="और एक पनीर टिक्का।", speaker="2", is_final=True),
                TranscriptSegment(text="no speaker info", speaker="", is_final=True),
            ]
        )
        assert text.splitlines() == [
            "[Speaker 1] Two tacos please.",
            "[Speaker 2 (Priya)] और एक पनीर टिक्का।",
            "no speaker info",
        ]

    def test_normalize_speaker_label(self):
        from inbound.agent import normalize_speaker_label

        assert normalize_speaker_label("Speaker 2") == "2"
        assert normalize_speaker_label("speaker_3") == "3"
        assert normalize_speaker_label(1) == "1"
        assert normalize_speaker_label(None) == ""


# =============================================================================
# LOCAL INTEGRATION TESTS
# =============================================================================


class TestLocalIntegration:
    """Integration tests using local WebSocket connection."""

    @pytest.fixture(scope="class")
    def server_process(self):
        """Start the inbound server as a subprocess."""
        env = os.environ.copy()
        env["SERVER_PORT"] = str(TEST_PORT)

        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        proc = subprocess.Popen(
            [sys.executable, "-m", "inbound.server"],
            cwd=project_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        ready = False
        for _ in range(30):
            try:
                resp = httpx.get(LOCAL_HTTP_URL, timeout=1.0)
                if resp.status_code == 200:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(0.5)

        if not ready:
            proc.terminate()
            proc.wait()
            output = proc.stdout.read().decode() if proc.stdout else ""
            pytest.skip(f"Server did not start in time. Output:\n{output[:2000]}")

        yield proc

        os.kill(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    @pytest.mark.asyncio
    async def test_local_health_check(self, server_process):
        """Test the health check endpoint."""
        async with httpx.AsyncClient() as client:
            response = await client.get(LOCAL_HTTP_URL)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_local_answer_webhook(self, server_process):
        """Test the answer webhook returns valid XML."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{LOCAL_HTTP_URL}/answer",
                params={"CallUUID": "test123", "From": "+15551234567", "To": "+16572338892"},
            )
            assert response.status_code == 200
            assert "application/xml" in response.headers["content-type"]
            assert "<Stream" in response.text
            assert "bidirectional" in response.text

    @pytest.mark.asyncio
    async def test_local_websocket_connection(self, server_process):
        """Test WebSocket connection and audio reception (needs API keys)."""
        if not all([OPENAI_API_KEY, META_API_KEY, CARTESIA_API_KEY]):
            pytest.skip("OPENAI_API_KEY, META_API_KEY, CARTESIA_API_KEY required")

        body_data = {"call_uuid": "test123", "from": "+15551234567", "to": "+16572338892"}
        body_b64 = base64.b64encode(json.dumps(body_data).encode()).decode()
        ws_url = f"{LOCAL_WS_URL}?body={body_b64}"

        async with websockets.connect(ws_url, close_timeout=2) as ws:
            start_event = {
                "event": "start",
                "start": {"callId": str(uuid.uuid4()), "streamId": str(uuid.uuid4())},
            }
            await ws.send(json.dumps(start_event))

            audio_received = False
            try:
                async with asyncio.timeout(30):
                    while True:
                        message = await ws.recv()
                        data = json.loads(message)
                        if data.get("event") == "playAudio":
                            audio_received = True
                            break
            except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                pass

            assert audio_received, "No audio received from server"

    @pytest.mark.asyncio
    async def test_local_audio_quality(self, server_process):
        """Test audio quality from the agent (needs API keys)."""
        if not all([OPENAI_API_KEY, META_API_KEY, CARTESIA_API_KEY]):
            pytest.skip("OPENAI_API_KEY, META_API_KEY, CARTESIA_API_KEY required")

        body_data = {"call_uuid": "test123", "from": "+15551234567", "to": "+16572338892"}
        body_b64 = base64.b64encode(json.dumps(body_data).encode()).decode()
        ws_url = f"{LOCAL_WS_URL}?body={body_b64}"

        audio_chunks = []

        async with websockets.connect(ws_url, close_timeout=2) as ws:
            start_event = {
                "event": "start",
                "start": {"callId": str(uuid.uuid4()), "streamId": str(uuid.uuid4())},
            }
            await ws.send(json.dumps(start_event))

            start_time = time.time()
            while time.time() - start_time < 30:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=0.5)
                    data = json.loads(message)
                    if data.get("event") == "playAudio":
                        payload = data.get("media", {}).get("payload", "")
                        if payload:
                            audio_chunks.append(base64.b64decode(payload))
                except asyncio.TimeoutError:
                    silence = base64.b64encode(b"\xff" * 160).decode()
                    await ws.send(json.dumps({"event": "media", "media": {"payload": silence}}))
                except websockets.exceptions.ConnectionClosed:
                    break

                if len(audio_chunks) >= 20:
                    break

        assert len(audio_chunks) > 0, "No audio chunks received"

        combined_audio = b"".join(audio_chunks)
        pcm_audio = ulaw_to_pcm(combined_audio)
        samples = struct.unpack(f"{len(pcm_audio) // 2}h", pcm_audio)

        rms = (sum(s**2 for s in samples) / len(samples)) ** 0.5
        assert rms > 500, f"Audio RMS {rms} too low - may be silence"


# =============================================================================
# API INTEGRATION TESTS
# =============================================================================


class TestOpenAIIntegration:
    """Integration tests for the OpenAI chat completions API."""

    @pytest.fixture
    def openai_configured(self):
        if not OPENAI_API_KEY:
            pytest.skip("OPENAI_API_KEY not configured")

    @pytest.mark.asyncio
    async def test_openai_chat_completion(self, openai_configured):
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        resp = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": "Reply with the single word: pong"}],
            max_completion_tokens=5,
        )
        assert resp.choices[0].message.content


class TestMuseIntegration:
    """Integration tests for Meta Muse Voice Transcribe realtime sessions."""

    @pytest.fixture
    def muse_configured(self):
        if not META_API_KEY:
            pytest.skip("META_API_KEY not configured")

    @pytest.mark.asyncio
    async def test_muse_session_opens(self, muse_configured):
        """A transcription session can be opened, configured and fed silence."""
        from inbound.agent import MuseTranscribeSTT, TranscriptSegment

        queue: asyncio.Queue[TranscriptSegment] = asyncio.Queue()
        stt = MuseTranscribeSTT(on_segment=queue)
        await stt.connect()
        try:
            # 500ms of u-law silence in 20ms frames
            for _ in range(25):
                await stt.send_audio(b"\xff" * 160)
            await asyncio.sleep(1.0)
            assert stt.errors == 0, "Muse reported an error during session setup"
        finally:
            await stt.close()


class TestCartesiaIntegration:
    """Integration tests for the Cartesia TTS WebSocket."""

    @pytest.fixture
    def cartesia_configured(self):
        if not CARTESIA_API_KEY:
            pytest.skip("CARTESIA_API_KEY not configured")

    @pytest.mark.asyncio
    async def test_cartesia_synthesis(self, cartesia_configured):
        from inbound.agent import (
            CARTESIA_API_VERSION,
            CARTESIA_MODEL,
            CARTESIA_VOICE_ID,
            CARTESIA_WS_URL,
        )

        url = (
            f"{CARTESIA_WS_URL}?api_key={CARTESIA_API_KEY}&cartesia_version={CARTESIA_API_VERSION}"
        )
        context_id = str(uuid.uuid4())
        total = 0
        async with websockets.connect(url, max_size=None) as ws:
            await ws.send(
                json.dumps(
                    {
                        "context_id": context_id,
                        "model_id": CARTESIA_MODEL,
                        "transcript": "Welcome to Bistro Mundo.",
                        "voice": {"mode": "id", "id": CARTESIA_VOICE_ID},
                        "output_format": {
                            "container": "raw",
                            "encoding": "pcm_s16le",
                            "sample_rate": 24000,
                        },
                        "language": "en",
                        "continue": False,
                    }
                )
            )
            async with asyncio.timeout(20):
                while True:
                    msg = json.loads(await ws.recv())
                    if msg.get("context_id") != context_id:
                        continue
                    if msg.get("type") == "chunk" and msg.get("data"):
                        total += len(base64.b64decode(msg["data"]))
                    if msg.get("done"):
                        break
        assert total > 24000, f"Cartesia returned too little audio: {total} bytes"


# =============================================================================
# PLIVO INTEGRATION TESTS
# =============================================================================


class TestPlivoIntegration:
    """Integration tests for Plivo API."""

    @pytest.fixture
    def plivo_configured(self):
        if not all([PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, PLIVO_PHONE_NUMBER]):
            pytest.skip("Plivo credentials not configured")

    def test_plivo_credentials_valid(self, plivo_configured):
        client = plivo.RestClient(auth_id=PLIVO_AUTH_ID, auth_token=PLIVO_AUTH_TOKEN)
        account = client.account.get()
        assert account is not None

    def test_plivo_phone_number_exists(self, plivo_configured):
        from utils import normalize_phone_number

        client = plivo.RestClient(auth_id=PLIVO_AUTH_ID, auth_token=PLIVO_AUTH_TOKEN)
        number_digits = normalize_phone_number(PLIVO_PHONE_NUMBER)
        try:
            number = client.numbers.get(number=number_digits)
            assert number is not None
        except plivo.exceptions.ResourceNotFoundError:
            pytest.fail(f"Phone number {PLIVO_PHONE_NUMBER} not found")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
