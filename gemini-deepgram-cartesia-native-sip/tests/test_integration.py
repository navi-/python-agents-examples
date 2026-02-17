"""
Integration tests for Gemini-Deepgram-Cartesia SIP Voice Agent.

Test Levels:
1. Unit Tests - Test individual components (audio conversion, SIP parsing, RTP)
2. Local Integration - Test SIP server setup and health endpoints
3. Deepgram/Gemini Integration - Test API connections
4. Plivo Integration - Test Plivo API configuration

Run tests:
    uv run pytest tests/test_integration.py -v

Run specific test level:
    uv run pytest tests/test_integration.py -v -k "unit"
    uv run pytest tests/test_integration.py -v -k "local"
"""

from __future__ import annotations

import asyncio
import math
import os
import struct

import pytest
from dotenv import load_dotenv

from sip import (
    RTP_PAYLOAD_TYPE_PCMU,
    RTPPacket,
    RTPSession,
    SIPMessage,
    build_sdp,
    extract_remote_rtp_info,
    parse_sdp,
)
from utils import (
    cartesia_to_plivo,
    normalize_phone_number,
    pcm8k_to_vad,
    pcm_to_ulaw,
    resample_audio,
    ulaw_to_pcm,
)

load_dotenv()

# Configuration from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID", "")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN", "")
PLIVO_PHONE_NUMBER = os.getenv("PLIVO_PHONE_NUMBER", "")


# =============================================================================
# UNIT TESTS — Audio Conversion
# =============================================================================


class TestUnitAudioConversion:
    """Unit tests for audio format conversion."""

    def test_ulaw_to_pcm_conversion(self):
        """Test μ-law to PCM conversion."""
        ulaw_silence = b"\xff" * 160
        pcm_audio = ulaw_to_pcm(ulaw_silence)

        samples = struct.unpack(f"{len(pcm_audio) // 2}h", pcm_audio)
        avg_amplitude = sum(abs(s) for s in samples) / len(samples)

        assert len(pcm_audio) == 320  # 160 samples * 2 bytes
        assert avg_amplitude < 100  # Should be near silence

    def test_pcm_to_ulaw_conversion(self):
        """Test PCM to μ-law conversion."""
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

        # Check correlation (should be > 0.9)
        correlation = sum(
            o * r for o, r in zip(original_samples, restored_samples, strict=True)
        )
        orig_energy = sum(o * o for o in original_samples)
        rest_energy = sum(r * r for r in restored_samples)

        if orig_energy > 0 and rest_energy > 0:
            normalized_corr = correlation / (orig_energy * rest_energy) ** 0.5
            assert normalized_corr > 0.9, "Audio quality degraded too much"

    def test_resample_audio_same_rate(self):
        """Test resampling with same rate returns same data."""
        pcm = b"\x00\x01" * 80
        result = resample_audio(pcm, 8000, 8000)
        assert result == pcm

    def test_resample_audio_upsample(self):
        """Test upsampling from 8kHz to 24kHz."""
        # 160 samples at 8kHz = 20ms
        samples = [int(8000 * math.sin(2 * math.pi * 440 * i / 8000)) for i in range(160)]
        pcm_8k = struct.pack(f"{len(samples)}h", *samples)

        pcm_24k = resample_audio(pcm_8k, 8000, 24000)
        # Should be ~3x the samples
        assert len(pcm_24k) // 2 == 480

    def test_cartesia_to_plivo(self):
        """Test Cartesia 24kHz → 8kHz conversion."""
        # 480 samples at 24kHz = 20ms
        samples_24k = [int(8000 * math.sin(2 * math.pi * 440 * i / 24000)) for i in range(480)]
        pcm_24k = struct.pack(f"{len(samples_24k)}h", *samples_24k)

        pcm_8k = cartesia_to_plivo(pcm_24k)
        # Should be ~1/3 the samples
        assert len(pcm_8k) // 2 == 160

    def test_pcm8k_to_vad(self):
        """Test PCM 8kHz → VAD float32 16kHz conversion."""
        pcm_8k = b"\x00" * 320  # 160 samples at 8kHz
        vad_audio = pcm8k_to_vad(pcm_8k)

        # Should be float32, upsampled to 16kHz (320 samples)
        assert vad_audio.dtype.name == "float32"
        assert len(vad_audio) == 320  # 2x samples due to 8kHz→16kHz


# =============================================================================
# UNIT TESTS — Phone Normalization
# =============================================================================


class TestUnitPhoneNormalization:
    """Unit tests for phone number normalization."""

    def test_normalize_e164_format(self):
        """Test normalizing E.164 formatted numbers."""
        result = normalize_phone_number("+16572338892")
        assert result == "16572338892"

    def test_normalize_with_spaces(self):
        """Test normalizing numbers with spaces."""
        result = normalize_phone_number("+1 657-233-8892")
        assert result == "16572338892"

    def test_normalize_local_format(self):
        """Test normalizing local format numbers."""
        result = normalize_phone_number("(657) 233-8892")
        assert result == "16572338892"

    def test_normalize_empty(self):
        """Test normalizing empty string."""
        result = normalize_phone_number("")
        assert result == ""


# =============================================================================
# UNIT TESTS — SIP Message Parsing
# =============================================================================


class TestUnitSIPMessage:
    """Unit tests for SIP message parsing and creation."""

    def test_parse_invite(self):
        """Test parsing a SIP INVITE request."""
        raw = (
            b"INVITE sip:agent@192.168.1.100:5060 SIP/2.0\r\n"
            b"Via: SIP/2.0/UDP 10.0.0.1:5060;branch=z9hG4bK776asdhds\r\n"
            b"From: <sip:user@10.0.0.1>;tag=12345\r\n"
            b"To: <sip:agent@192.168.1.100>\r\n"
            b"Call-ID: abc123@10.0.0.1\r\n"
            b"CSeq: 1 INVITE\r\n"
            b"Contact: <sip:user@10.0.0.1:5060>\r\n"
            b"Content-Type: application/sdp\r\n"
            b"Content-Length: 100\r\n"
            b"\r\n"
            b"v=0\r\n"
            b"o=- 123 123 IN IP4 10.0.0.1\r\n"
            b"s=Session\r\n"
            b"c=IN IP4 10.0.0.1\r\n"
            b"t=0 0\r\n"
            b"m=audio 30000 RTP/AVP 0\r\n"
        )

        msg = SIPMessage.parse(raw)

        assert msg.method == "INVITE"
        assert msg.request_uri == "sip:agent@192.168.1.100:5060"
        assert msg.call_id == "abc123@10.0.0.1"
        assert msg.cseq == "1 INVITE"
        assert "10.0.0.1" in msg.via
        assert "v=0" in msg.body

    def test_parse_200_ok(self):
        """Test parsing a SIP 200 OK response."""
        raw = (
            b"SIP/2.0 200 OK\r\n"
            b"Via: SIP/2.0/UDP 10.0.0.1:5060;branch=z9hG4bK776\r\n"
            b"From: <sip:user@10.0.0.1>;tag=12345\r\n"
            b"To: <sip:agent@192.168.1.100>;tag=67890\r\n"
            b"Call-ID: abc123@10.0.0.1\r\n"
            b"CSeq: 1 INVITE\r\n"
            b"Content-Length: 0\r\n"
            b"\r\n"
        )

        msg = SIPMessage.parse(raw)

        assert msg.status_code == 200
        assert msg.status_text == "OK"
        assert msg.call_id == "abc123@10.0.0.1"
        assert msg.method is None

    def test_parse_bye(self):
        """Test parsing a SIP BYE request."""
        raw = (
            b"BYE sip:agent@192.168.1.100 SIP/2.0\r\n"
            b"Via: SIP/2.0/UDP 10.0.0.1:5060;branch=z9hG4bK999\r\n"
            b"From: <sip:user@10.0.0.1>;tag=12345\r\n"
            b"To: <sip:agent@192.168.1.100>;tag=67890\r\n"
            b"Call-ID: abc123@10.0.0.1\r\n"
            b"CSeq: 2 BYE\r\n"
            b"Content-Length: 0\r\n"
            b"\r\n"
        )

        msg = SIPMessage.parse(raw)

        assert msg.method == "BYE"
        assert msg.call_id == "abc123@10.0.0.1"

    def test_to_bytes_roundtrip(self):
        """Test that SIP message survives serialize/parse roundtrip."""
        msg = SIPMessage(
            status_code=200,
            status_text="OK",
        )
        msg.headers["Via"] = "SIP/2.0/UDP 10.0.0.1:5060"
        msg.headers["From"] = "<sip:user@10.0.0.1>;tag=12345"
        msg.headers["To"] = "<sip:agent@10.0.0.2>;tag=67890"
        msg.headers["Call-ID"] = "test123"
        msg.headers["CSeq"] = "1 INVITE"
        msg.headers["Content-Length"] = "0"

        raw = msg.to_bytes()
        parsed = SIPMessage.parse(raw)

        assert parsed.status_code == 200
        assert parsed.call_id == "test123"

    def test_extract_tag(self):
        """Test tag extraction from From/To headers."""
        msg = SIPMessage()
        msg.headers["From"] = "<sip:user@10.0.0.1>;tag=abc123"

        # Using the server's _extract_tag helper indirectly
        from_header = msg.from_header
        tag = ""
        for part in from_header.split(";"):
            part = part.strip()
            if part.startswith("tag="):
                tag = part[4:]
        assert tag == "abc123"


# =============================================================================
# UNIT TESTS — SDP
# =============================================================================


class TestUnitSDP:
    """Unit tests for SDP parsing and building."""

    def test_parse_sdp(self):
        """Test parsing SDP body."""
        sdp = (
            "v=0\r\n"
            "o=- 123 123 IN IP4 10.0.0.1\r\n"
            "s=Session\r\n"
            "c=IN IP4 10.0.0.1\r\n"
            "t=0 0\r\n"
            "m=audio 30000 RTP/AVP 0 8\r\n"
            "a=rtpmap:0 PCMU/8000\r\n"
            "a=ptime:20\r\n"
        )

        info = parse_sdp(sdp)

        assert info["remote_ip"] == "10.0.0.1"
        assert info["remote_port"] == 30000
        assert 0 in info["codecs"]
        assert 8 in info["codecs"]
        assert info["ptime"] == 20

    def test_build_sdp(self):
        """Test building SDP offer."""
        sdp = build_sdp("192.168.1.100", 10000)

        assert "192.168.1.100" in sdp
        assert "m=audio 10000" in sdp
        assert "PCMU/8000" in sdp
        assert "a=ptime:20" in sdp
        assert "a=sendrecv" in sdp

    def test_extract_remote_rtp_info(self):
        """Test extracting remote RTP info from SDP."""
        sdp = (
            "v=0\r\n"
            "o=- 123 123 IN IP4 10.0.0.5\r\n"
            "s=Session\r\n"
            "c=IN IP4 10.0.0.5\r\n"
            "t=0 0\r\n"
            "m=audio 12345 RTP/AVP 0\r\n"
        )

        ip, port = extract_remote_rtp_info(sdp)

        assert ip == "10.0.0.5"
        assert port == 12345


# =============================================================================
# UNIT TESTS — RTP Packet
# =============================================================================


class TestUnitRTPPacket:
    """Unit tests for RTP packet encoding/decoding."""

    def test_rtp_packet_create(self):
        """Test creating an RTP packet."""
        payload = b"\xff" * 160  # G.711 silence
        pkt = RTPPacket(
            payload_type=RTP_PAYLOAD_TYPE_PCMU,
            sequence=100,
            timestamp=16000,
            ssrc=0x12345678,
            payload=payload,
        )

        raw = pkt.to_bytes()

        # Header: 12 bytes + payload: 160 bytes
        assert len(raw) == 172

    def test_rtp_packet_roundtrip(self):
        """Test RTP packet survives encode/decode roundtrip."""
        payload = bytes(range(160))
        pkt = RTPPacket(
            payload_type=RTP_PAYLOAD_TYPE_PCMU,
            sequence=42,
            timestamp=3360,
            ssrc=0xAABBCCDD,
            payload=payload,
        )

        raw = pkt.to_bytes()
        parsed = RTPPacket.parse(raw)

        assert parsed.payload_type == RTP_PAYLOAD_TYPE_PCMU
        assert parsed.sequence == 42
        assert parsed.timestamp == 3360
        assert parsed.ssrc == 0xAABBCCDD
        assert parsed.payload == payload

    def test_rtp_packet_too_short(self):
        """Test that parsing short data raises ValueError."""
        with pytest.raises(ValueError, match="too short"):
            RTPPacket.parse(b"\x80\x00")


# =============================================================================
# UNIT TESTS — RTP Session
# =============================================================================


class TestUnitRTPSession:
    """Unit tests for RTP session."""

    @pytest.mark.asyncio
    async def test_rtp_session_start_stop(self):
        """Test starting and stopping an RTP session."""
        session = RTPSession(local_port=19000, remote_addr=("127.0.0.1", 19001))
        await session.start()

        assert session.is_running
        assert session.local_port == 19000

        await session.stop()
        assert not session.is_running

    @pytest.mark.asyncio
    async def test_rtp_session_send_audio(self):
        """Test queueing audio for sending."""
        session = RTPSession(local_port=19002, remote_addr=("127.0.0.1", 19003))
        await session.start()

        try:
            # Queue PCM16 audio (320 bytes = 1 frame)
            pcm_frame = b"\x00" * 320
            await session.send_audio(pcm_frame)

            # Give send loop time to process
            await asyncio.sleep(0.05)
        finally:
            await session.stop()

    @pytest.mark.asyncio
    async def test_rtp_session_loopback(self):
        """Test RTP loopback — send to self and receive."""
        # Bind two sessions that talk to each other
        session_a = RTPSession(local_port=19010, remote_addr=("127.0.0.1", 19011))
        session_b = RTPSession(local_port=19011, remote_addr=("127.0.0.1", 19010))

        await session_a.start()
        await session_b.start()

        try:
            # Session A sends audio
            # 160 samples of 440Hz tone at 8kHz, as PCM16
            tone = [int(16000 * math.sin(2 * math.pi * 440 * i / 8000)) for i in range(160)]
            pcm_frame = struct.pack(f"{len(tone)}h", *tone)
            await session_a.send_audio(pcm_frame)

            # Wait for send loop to fire
            await asyncio.sleep(0.1)

            # Session B should receive PCM data (decoded from μ-law)
            pcm_received = session_b.receive_audio_nowait()
            # May not arrive instantly due to timing, so try a few times
            for _ in range(10):
                if pcm_received is not None:
                    break
                await asyncio.sleep(0.025)
                pcm_received = session_b.receive_audio_nowait()

            assert pcm_received is not None, "No audio received via RTP loopback"
            assert len(pcm_received) == 320  # 160 PCM16 samples

        finally:
            await session_a.stop()
            await session_b.stop()


# =============================================================================
# UNIT TESTS — Call Manager
# =============================================================================


class TestUnitCallManager:
    """Unit tests for outbound CallManager."""

    def test_create_call(self):
        """Test creating a call record."""
        from outbound.agent import CallManager

        mgr = CallManager()
        record = mgr.create_call(
            phone_number="15551234567",
            campaign_id="test-campaign",
            opening_reason="your free trial",
            objective="qualify the lead",
        )

        assert record.phone_number == "15551234567"
        assert record.status == "initiating"
        assert record.campaign_id == "test-campaign"
        assert "free trial" in record.system_prompt

    def test_update_status(self):
        """Test updating call status."""
        from outbound.agent import CallManager

        mgr = CallManager()
        record = mgr.create_call(phone_number="15551234567")
        mgr.update_status(record.call_id, "connected")

        updated = mgr.get_call(record.call_id)
        assert updated.status == "connected"

    def test_get_active_calls(self):
        """Test getting active calls."""
        from outbound.agent import CallManager

        mgr = CallManager()
        r1 = mgr.create_call(phone_number="15551111111")
        r2 = mgr.create_call(phone_number="15552222222")
        mgr.update_status(r1.call_id, "connected")
        mgr.update_status(r2.call_id, "completed")

        active = mgr.get_active_calls()
        assert len(active) == 1
        assert active[0].call_id == r1.call_id

    def test_determine_outcome(self):
        """Test outcome determination."""
        from outbound.agent import determine_outcome

        assert determine_outcome("NO_ANSWER", 0) == "no_answer"
        assert determine_outcome("USER_BUSY", 0) == "busy"
        assert determine_outcome("NORMAL_CLEARING", 30) == "success"
        assert determine_outcome("UNALLOCATED_NUMBER", 0) == "failed"
        assert determine_outcome("", 10) == "success"


# =============================================================================
# LOCAL INTEGRATION TESTS
# =============================================================================


class TestLocalIntegration:
    """Integration tests for SIP server + FastAPI health API."""

    @pytest.mark.asyncio
    async def test_sip_server_start_stop(self):
        """Test starting and stopping the SIP server."""
        from sip import AsyncSIPServer

        server = AsyncSIPServer(
            host="0.0.0.0",
            sip_port=15060,
            rtp_port_start=15000,
            rtp_port_end=15100,
        )
        await server.start()

        calls = server.get_active_calls()
        assert len(calls) == 0

        await server.stop()

    @pytest.mark.asyncio
    async def test_sip_server_handles_options(self):
        """Test SIP server responds to OPTIONS."""
        from sip import AsyncSIPServer, SIPMessage

        received_responses = []

        class _TestProtocol(asyncio.DatagramProtocol):
            def __init__(self):
                self.transport = None

            def connection_made(self, transport):
                self.transport = transport

            def datagram_received(self, data, addr):
                try:
                    msg = SIPMessage.parse(data)
                    received_responses.append(msg)
                except Exception:
                    pass

        server = AsyncSIPServer(host="0.0.0.0", sip_port=15061)
        await server.start()

        try:
            loop = asyncio.get_running_loop()
            transport, _protocol = await loop.create_datagram_endpoint(
                _TestProtocol, local_addr=("127.0.0.1", 0)
            )

            # Send OPTIONS
            options = SIPMessage(method="OPTIONS", request_uri="sip:agent@127.0.0.1:15061")
            options.headers["Via"] = "SIP/2.0/UDP 127.0.0.1:9999;branch=z9hG4bKtest"
            options.headers["From"] = "<sip:test@127.0.0.1>;tag=test1"
            options.headers["To"] = "<sip:agent@127.0.0.1>"
            options.headers["Call-ID"] = "options-test-1"
            options.headers["CSeq"] = "1 OPTIONS"
            options.headers["Content-Length"] = "0"

            transport.sendto(options.to_bytes(), ("127.0.0.1", 15061))

            # Wait for response
            await asyncio.sleep(0.2)

            assert len(received_responses) > 0
            resp = received_responses[0]
            assert resp.status_code == 200

            transport.close()
        finally:
            await server.stop()


# =============================================================================
# DEEPGRAM/GEMINI INTEGRATION TESTS
# =============================================================================


class TestDeepgramIntegration:
    """Integration tests for Deepgram STT API."""

    @pytest.fixture
    def deepgram_configured(self):
        if not DEEPGRAM_API_KEY:
            pytest.skip("DEEPGRAM_API_KEY not configured")

    @pytest.mark.asyncio
    async def test_deepgram_connection(self, deepgram_configured):
        """Test Deepgram WebSocket connection."""
        from inbound.agent import DeepgramSTT

        transcripts = []

        async def on_transcript(text):
            transcripts.append(text)

        stt = DeepgramSTT(on_transcript=on_transcript)
        await stt.connect()

        # Send silence to verify connection works
        silence = b"\x00" * 320
        await stt.send_audio(silence)
        await asyncio.sleep(1)

        await stt.close()
        # No assertion on transcript — just verify no crash


class TestGeminiIntegration:
    """Integration tests for Gemini LLM API."""

    @pytest.fixture
    def gemini_configured(self):
        if not GEMINI_API_KEY:
            pytest.skip("GEMINI_API_KEY not configured")

    @pytest.mark.asyncio
    async def test_gemini_response(self, gemini_configured):
        """Test Gemini generates a response."""
        from inbound.agent import GeminiLLM

        llm = GeminiLLM("You are a helpful assistant. Keep responses brief.")
        response = await llm.generate_response("Say hello in one sentence.")

        assert len(response) > 0
        assert isinstance(response, str)


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
        """Test that Plivo credentials are valid."""
        import plivo

        client = plivo.RestClient(auth_id=PLIVO_AUTH_ID, auth_token=PLIVO_AUTH_TOKEN)
        account = client.account.get()
        assert account is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
