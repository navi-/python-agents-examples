"""SIP protocol, SDP negotiation, and RTP session for direct SIP/RTP voice calls.

This module provides a pure-Python asyncio implementation of:
- SIP message parsing and creation (INVITE, BYE, ACK, 200 OK, etc.)
- SDP offer/answer for media negotiation (G.711 PCMU/PCMA at 8kHz)
- RTP packet encoding/decoding (RFC 3550)
- Bidirectional RTP audio streaming with precise 20ms timing
- Async SIP server for handling inbound calls
- SIP UAC (User Agent Client) for placing outbound calls

Based on the sip-to-ai reference implementation pattern.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from utils import pcm_to_ulaw, ulaw_to_pcm

# =============================================================================
# SIP Message Parsing/Creation
# =============================================================================

# Standard SIP port
DEFAULT_SIP_PORT = 5060

# RTP constants
RTP_HEADER_SIZE = 12
RTP_PAYLOAD_TYPE_PCMU = 0
RTP_PAYLOAD_TYPE_PCMA = 8
RTP_CLOCK_RATE = 8000
RTP_FRAME_DURATION = 0.020  # 20ms
RTP_FRAME_SAMPLES = 160  # 8000 * 0.020


def _random_branch() -> str:
    """Generate a random SIP branch parameter."""
    return f"z9hG4bK{random.randint(100000, 999999)}"


@dataclass
class SIPMessage:
    """Parsed SIP message (request or response)."""

    method: str | None = None  # INVITE, BYE, ACK, OPTIONS, CANCEL
    status_code: int | None = None  # 100, 180, 200, etc.
    status_text: str = ""
    request_uri: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    body: str = ""
    raw_first_line: str = ""

    @classmethod
    def parse(cls, data: bytes) -> SIPMessage:
        """Parse raw SIP message bytes into a SIPMessage."""
        text = data.decode("utf-8", errors="replace")
        parts = text.split("\r\n\r\n", 1)
        header_section = parts[0]
        body = parts[1] if len(parts) > 1 else ""

        lines = header_section.split("\r\n")
        first_line = lines[0]

        msg = cls(body=body, raw_first_line=first_line)

        # Parse first line — request or response
        if first_line.startswith("SIP/2.0"):
            # Response: SIP/2.0 200 OK
            parts_fl = first_line.split(" ", 2)
            msg.status_code = int(parts_fl[1])
            msg.status_text = parts_fl[2] if len(parts_fl) > 2 else ""
        else:
            # Request: INVITE sip:user@host SIP/2.0
            parts_fl = first_line.split(" ", 2)
            msg.method = parts_fl[0]
            msg.request_uri = parts_fl[1] if len(parts_fl) > 1 else ""

        # Parse headers
        for line in lines[1:]:
            if ":" in line:
                key, value = line.split(":", 1)
                # Store with original case as key, strip whitespace
                msg.headers[key.strip()] = value.strip()

        return msg

    def to_bytes(self) -> bytes:
        """Serialize SIP message to bytes."""
        lines = []

        if self.status_code is not None:
            lines.append(f"SIP/2.0 {self.status_code} {self.status_text}")
        elif self.method:
            lines.append(f"{self.method} {self.request_uri} SIP/2.0")

        for key, value in self.headers.items():
            lines.append(f"{key}: {value}")

        header_text = "\r\n".join(lines) + "\r\n\r\n"
        if self.body:
            header_text += self.body

        return header_text.encode("utf-8")

    def get_header(self, name: str) -> str:
        """Case-insensitive header lookup."""
        name_lower = name.lower()
        for key, value in self.headers.items():
            if key.lower() == name_lower:
                return value
        return ""

    @property
    def call_id(self) -> str:
        return self.get_header("Call-ID") or self.get_header("i")

    @property
    def from_header(self) -> str:
        return self.get_header("From") or self.get_header("f")

    @property
    def to_header(self) -> str:
        return self.get_header("To") or self.get_header("t")

    @property
    def cseq(self) -> str:
        return self.get_header("CSeq")

    @property
    def via(self) -> str:
        return self.get_header("Via") or self.get_header("v")

    @property
    def contact(self) -> str:
        return self.get_header("Contact") or self.get_header("m")


# =============================================================================
# SDP Parsing/Building
# =============================================================================


def parse_sdp(body: str) -> dict[str, Any]:
    """Parse SDP body and extract media information.

    Returns dict with keys: remote_ip, remote_port, codecs, ptime.
    """
    result: dict[str, Any] = {
        "remote_ip": "",
        "remote_port": 0,
        "codecs": [],
        "ptime": 20,
    }

    for line in body.strip().splitlines():
        line = line.strip()
        if line.startswith("c=IN IP4 "):
            result["remote_ip"] = line.split()[-1]
        elif line.startswith("m=audio "):
            parts = line.split()
            result["remote_port"] = int(parts[1])
            # Codec payload types follow: m=audio PORT RTP/AVP 0 8 ...
            result["codecs"] = [int(p) for p in parts[3:] if p.isdigit()]
        elif line.startswith("a=ptime:"):
            result["ptime"] = int(line.split(":")[1])

    return result


def build_sdp(local_ip: str, rtp_port: int) -> str:
    """Build SDP answer offering PCMU (0) at 8kHz."""
    session_id = int(time.time())
    return (
        "v=0\r\n"
        f"o=- {session_id} {session_id} IN IP4 {local_ip}\r\n"
        "s=Voice Agent\r\n"
        f"c=IN IP4 {local_ip}\r\n"
        "t=0 0\r\n"
        f"m=audio {rtp_port} RTP/AVP 0\r\n"
        "a=rtpmap:0 PCMU/8000\r\n"
        "a=ptime:20\r\n"
        "a=sendrecv\r\n"
    )


def extract_remote_rtp_info(sdp_body: str) -> tuple[str, int]:
    """Extract remote IP and RTP port from SDP body."""
    info = parse_sdp(sdp_body)
    return info["remote_ip"], info["remote_port"]


# =============================================================================
# RTP Packet
# =============================================================================


@dataclass
class RTPPacket:
    """RFC 3550 RTP packet."""

    payload_type: int = RTP_PAYLOAD_TYPE_PCMU
    sequence: int = 0
    timestamp: int = 0
    ssrc: int = 0
    payload: bytes = b""

    @classmethod
    def parse(cls, data: bytes) -> RTPPacket:
        """Parse raw RTP packet bytes."""
        if len(data) < RTP_HEADER_SIZE:
            raise ValueError(f"RTP packet too short: {len(data)} bytes")

        # First byte: V(2) P(1) X(1) CC(4)
        first_byte = data[0]
        cc = first_byte & 0x0F
        # Second byte: M(1) PT(7)
        second_byte = data[1]
        payload_type = second_byte & 0x7F

        sequence = struct.unpack("!H", data[2:4])[0]
        timestamp = struct.unpack("!I", data[4:8])[0]
        ssrc = struct.unpack("!I", data[8:12])[0]

        header_size = RTP_HEADER_SIZE + cc * 4
        payload = data[header_size:]

        return cls(
            payload_type=payload_type,
            sequence=sequence,
            timestamp=timestamp,
            ssrc=ssrc,
            payload=payload,
        )

    def to_bytes(self) -> bytes:
        """Serialize RTP packet to bytes."""
        # V=2, P=0, X=0, CC=0
        first_byte = 0x80
        # M=0, PT
        second_byte = self.payload_type & 0x7F

        header = struct.pack(
            "!BBHII",
            first_byte,
            second_byte,
            self.sequence & 0xFFFF,
            self.timestamp & 0xFFFFFFFF,
            self.ssrc,
        )
        return header + self.payload


# =============================================================================
# RTP Session — Bidirectional audio over UDP
# =============================================================================


class _RTPProtocol(asyncio.DatagramProtocol):
    """asyncio DatagramProtocol for receiving RTP packets."""

    def __init__(self, rx_queue: asyncio.Queue[bytes]):
        self._rx_queue = rx_queue
        self.transport: asyncio.DatagramTransport | None = None

    def connection_made(self, transport: asyncio.DatagramTransport) -> None:
        self.transport = transport

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        try:
            pkt = RTPPacket.parse(data)
            # Decode G.711 μ-law to PCM16
            pcm_data = ulaw_to_pcm(pkt.payload)
            with contextlib.suppress(asyncio.QueueFull):
                self._rx_queue.put_nowait(pcm_data)
        except Exception:
            pass  # Ignore malformed packets

    def error_received(self, exc: Exception) -> None:
        logger.warning(f"RTP UDP error: {exc}")


class RTPSession:
    """Bidirectional RTP audio over UDP with precise 20ms timing."""

    def __init__(self, local_port: int, remote_addr: tuple[str, int]):
        self._local_port = local_port
        self._remote_addr = remote_addr
        self._rx_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=500)
        self._tx_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=500)
        self._transport: asyncio.DatagramTransport | None = None
        self._protocol: _RTPProtocol | None = None
        self._send_task: asyncio.Task | None = None
        self._running = False
        self._ssrc = random.randint(0, 0xFFFFFFFF)
        self._sequence = random.randint(0, 0xFFFF)
        self._timestamp = random.randint(0, 0xFFFFFFFF)

    async def start(self) -> None:
        """Bind UDP socket and start send loop."""
        loop = asyncio.get_running_loop()
        self._protocol = _RTPProtocol(self._rx_queue)

        self._transport, _ = await loop.create_datagram_endpoint(
            lambda: self._protocol,
            local_addr=("0.0.0.0", self._local_port),
        )

        self._running = True
        self._send_task = asyncio.create_task(self._send_loop(), name="rtp_tx")
        logger.info(
            f"RTP session started: local={self._local_port}, "
            f"remote={self._remote_addr[0]}:{self._remote_addr[1]}"
        )

    async def _send_loop(self) -> None:
        """Send RTP packets at precise 20ms intervals using absolute time scheduling."""
        loop = asyncio.get_running_loop()
        start_time = loop.time()
        frame_count = 0

        # G.711 μ-law silence (0xFF = zero amplitude)
        silence_ulaw = b"\xff" * RTP_FRAME_SAMPLES

        try:
            while self._running:
                frame_count += 1
                target_time = start_time + frame_count * RTP_FRAME_DURATION
                now = loop.time()
                if now < target_time:
                    await asyncio.sleep(target_time - now)

                # Get audio from tx queue or send silence
                try:
                    pcm_data = self._tx_queue.get_nowait()
                    ulaw_data = pcm_to_ulaw(pcm_data)
                    # Ensure exactly 160 bytes
                    if len(ulaw_data) < RTP_FRAME_SAMPLES:
                        ulaw_data += b"\xff" * (RTP_FRAME_SAMPLES - len(ulaw_data))
                    elif len(ulaw_data) > RTP_FRAME_SAMPLES:
                        ulaw_data = ulaw_data[:RTP_FRAME_SAMPLES]
                except asyncio.QueueEmpty:
                    ulaw_data = silence_ulaw

                # Build and send RTP packet
                pkt = RTPPacket(
                    payload_type=RTP_PAYLOAD_TYPE_PCMU,
                    sequence=self._sequence,
                    timestamp=self._timestamp,
                    ssrc=self._ssrc,
                    payload=ulaw_data,
                )
                if self._transport and not self._transport.is_closing():
                    self._transport.sendto(pkt.to_bytes(), self._remote_addr)

                self._sequence = (self._sequence + 1) & 0xFFFF
                self._timestamp = (self._timestamp + RTP_FRAME_SAMPLES) & 0xFFFFFFFF

        except asyncio.CancelledError:
            pass

    async def send_audio(self, pcm_data: bytes) -> None:
        """Queue PCM16 8kHz audio for sending. Data is chunked into 20ms frames."""
        # Split into 20ms frames (320 bytes PCM16 = 160 samples)
        frame_size = RTP_FRAME_SAMPLES * 2  # 16-bit samples
        for i in range(0, len(pcm_data), frame_size):
            chunk = pcm_data[i : i + frame_size]
            if len(chunk) < frame_size:
                chunk += b"\x00" * (frame_size - len(chunk))
            try:
                self._tx_queue.put_nowait(chunk)
            except asyncio.QueueFull:
                break  # Backpressure — drop frames

    async def receive_audio(self) -> bytes:
        """Get next PCM16 8kHz frame from receive queue."""
        return await self._rx_queue.get()

    def receive_audio_nowait(self) -> bytes | None:
        """Non-blocking receive. Returns None if no audio available."""
        try:
            return self._rx_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def stop(self) -> None:
        """Close UDP socket and cancel tasks."""
        self._running = False
        if self._send_task and not self._send_task.done():
            self._send_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._send_task
        if self._transport:
            self._transport.close()
        logger.info(f"RTP session stopped: local={self._local_port}")

    @property
    def local_port(self) -> int:
        return self._local_port

    @property
    def is_running(self) -> bool:
        return self._running


# =============================================================================
# Async SIP Server
# =============================================================================


@dataclass
class ActiveCall:
    """Tracks an active SIP call."""

    call_id: str
    from_uri: str
    to_uri: str
    remote_sip_addr: tuple[str, int]
    rtp_session: RTPSession
    local_rtp_port: int
    from_tag: str = ""
    to_tag: str = ""


class _SIPProtocol(asyncio.DatagramProtocol):
    """asyncio DatagramProtocol for SIP UDP transport."""

    def __init__(self, message_handler: Callable):
        self._handler = message_handler
        self.transport: asyncio.DatagramTransport | None = None

    def connection_made(self, transport: asyncio.DatagramTransport) -> None:
        self.transport = transport

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        asyncio.get_running_loop().create_task(self._handler(data, addr))

    def error_received(self, exc: Exception) -> None:
        logger.warning(f"SIP UDP error: {exc}")


class AsyncSIPServer:
    """UDP SIP server managing inbound calls."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        sip_port: int = DEFAULT_SIP_PORT,
        rtp_port_start: int = 10000,
        rtp_port_end: int = 20000,
        on_new_call: Callable | None = None,
        on_call_end: Callable | None = None,
    ):
        self._host = host
        self._sip_port = sip_port
        self._on_new_call = on_new_call
        self._on_call_end = on_call_end
        self._calls: dict[str, ActiveCall] = {}
        self._rtp_port_pool: set[int] = set(range(rtp_port_start, rtp_port_end, 2))
        self._transport: asyncio.DatagramTransport | None = None
        self._protocol: _SIPProtocol | None = None
        self._local_ip = host if host != "0.0.0.0" else ""
        self._running = False

    def _allocate_rtp_port(self) -> int | None:
        """Allocate an RTP port from the pool."""
        if not self._rtp_port_pool:
            return None
        return self._rtp_port_pool.pop()

    def _release_rtp_port(self, port: int) -> None:
        """Return an RTP port to the pool."""
        self._rtp_port_pool.add(port)

    def _get_local_ip(self) -> str:
        """Get the local IP address for SDP."""
        if self._local_ip:
            return self._local_ip
        # Try to determine local IP
        import socket

        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            self._local_ip = ip
            return ip
        except Exception:
            return "127.0.0.1"

    def _extract_tag(self, header_value: str) -> str:
        """Extract tag parameter from From/To header."""
        for part in header_value.split(";"):
            part = part.strip()
            if part.startswith("tag="):
                return part[4:]
        return ""

    def _extract_uri(self, header_value: str) -> str:
        """Extract SIP URI from header value (strip display name and angle brackets)."""
        if "<" in header_value:
            start = header_value.index("<") + 1
            end = header_value.index(">")
            return header_value[start:end]
        # No angle brackets, take up to first semicolon
        return header_value.split(";")[0].strip()

    async def start(self) -> None:
        """Bind UDP on SIP port and start listening."""
        loop = asyncio.get_running_loop()
        self._protocol = _SIPProtocol(self._handle_message)

        self._transport, _ = await loop.create_datagram_endpoint(
            lambda: self._protocol,
            local_addr=(self._host, self._sip_port),
        )

        self._running = True
        local_ip = self._get_local_ip()
        logger.info(f"SIP server listening on {local_ip}:{self._sip_port}")

    async def _send_sip(self, msg: SIPMessage, addr: tuple[str, int]) -> None:
        """Send a SIP message to the given address."""
        if self._transport and not self._transport.is_closing():
            self._transport.sendto(msg.to_bytes(), addr)

    async def _handle_message(self, data: bytes, addr: tuple[str, int]) -> None:
        """Route incoming SIP message to appropriate handler."""
        try:
            msg = SIPMessage.parse(data)
        except Exception as e:
            logger.warning(f"Failed to parse SIP message from {addr}: {e}")
            return

        if msg.method == "INVITE":
            await self._handle_invite(msg, addr)
        elif msg.method == "ACK":
            logger.debug(f"ACK received for call {msg.call_id}")
        elif msg.method == "BYE":
            await self._handle_bye(msg, addr)
        elif msg.method == "OPTIONS":
            await self._handle_options(msg, addr)
        elif msg.method == "CANCEL":
            await self._handle_cancel(msg, addr)
        elif msg.status_code is not None:
            logger.debug(
                f"SIP response {msg.status_code} {msg.status_text} "
                f"for call {msg.call_id}"
            )

    async def _handle_invite(self, msg: SIPMessage, addr: tuple[str, int]) -> None:
        """Handle incoming INVITE — negotiate media and start RTP."""
        call_id = msg.call_id
        logger.info(f"INVITE received: {call_id} from {addr}")

        # Check for re-INVITE (call already exists)
        if call_id in self._calls:
            logger.info(f"Re-INVITE for existing call {call_id}, sending 200 OK")
            existing = self._calls[call_id]
            await self._send_200_ok(msg, addr, existing.local_rtp_port)
            return

        # Send 100 Trying
        trying = self._build_response(msg, 100, "Trying", addr)
        await self._send_sip(trying, addr)

        # Parse SDP to get remote RTP endpoint
        if not msg.body:
            logger.warning("INVITE has no SDP body")
            error_resp = self._build_response(msg, 400, "Bad Request (no SDP)", addr)
            await self._send_sip(error_resp, addr)
            return

        remote_ip, remote_port = extract_remote_rtp_info(msg.body)
        if not remote_ip or not remote_port:
            logger.warning("Could not extract RTP info from SDP")
            error_resp = self._build_response(msg, 400, "Bad Request (bad SDP)", addr)
            await self._send_sip(error_resp, addr)
            return

        # Allocate local RTP port
        local_rtp_port = self._allocate_rtp_port()
        if local_rtp_port is None:
            logger.error("No RTP ports available")
            error_resp = self._build_response(msg, 503, "Service Unavailable", addr)
            await self._send_sip(error_resp, addr)
            return

        # Create RTP session
        rtp_session = RTPSession(local_rtp_port, (remote_ip, remote_port))
        await rtp_session.start()

        # Generate To tag
        to_tag = f"{random.randint(100000, 999999)}"

        # Store call
        from_uri = self._extract_uri(msg.from_header)
        to_uri = self._extract_uri(msg.to_header)
        from_tag = self._extract_tag(msg.from_header)

        call = ActiveCall(
            call_id=call_id,
            from_uri=from_uri,
            to_uri=to_uri,
            remote_sip_addr=addr,
            rtp_session=rtp_session,
            local_rtp_port=local_rtp_port,
            from_tag=from_tag,
            to_tag=to_tag,
        )
        self._calls[call_id] = call

        # Send 200 OK with SDP
        await self._send_200_ok(msg, addr, local_rtp_port, to_tag=to_tag)

        logger.info(
            f"Call {call_id} answered: RTP local={local_rtp_port}, "
            f"remote={remote_ip}:{remote_port}"
        )

        # Notify callback
        if self._on_new_call:
            task = asyncio.create_task(
                self._safe_callback(
                    self._on_new_call,
                    rtp_session,
                    {
                        "call_id": call_id,
                        "from_uri": from_uri,
                        "to_uri": to_uri,
                        "remote_addr": addr,
                    },
                )
            )
            task.add_done_callback(lambda t: None)  # prevent GC

    async def _safe_callback(self, callback: Callable, *args: Any) -> None:
        """Run callback safely, catching exceptions."""
        try:
            await callback(*args)
        except Exception as e:
            logger.error(f"Callback error: {e}")

    async def _send_200_ok(
        self,
        invite_msg: SIPMessage,
        addr: tuple[str, int],
        rtp_port: int,
        to_tag: str = "",
    ) -> None:
        """Send 200 OK response with SDP answer."""
        local_ip = self._get_local_ip()
        sdp_body = build_sdp(local_ip, rtp_port)

        response = self._build_response(
            invite_msg, 200, "OK", addr, body=sdp_body, to_tag=to_tag
        )
        response.headers["Content-Type"] = "application/sdp"
        response.headers["Content-Length"] = str(len(sdp_body))

        await self._send_sip(response, addr)

    def _build_response(
        self,
        request: SIPMessage,
        status_code: int,
        status_text: str,
        addr: tuple[str, int],
        body: str = "",
        to_tag: str = "",
    ) -> SIPMessage:
        """Build a SIP response from a request."""
        local_ip = self._get_local_ip()

        # Build To header with tag
        to_header = request.to_header
        if to_tag and ";tag=" not in to_header:
            to_header = f"{to_header};tag={to_tag}"

        response = SIPMessage(
            status_code=status_code,
            status_text=status_text,
            body=body,
        )
        response.headers["Via"] = request.via
        response.headers["From"] = request.from_header
        response.headers["To"] = to_header
        response.headers["Call-ID"] = request.call_id
        response.headers["CSeq"] = request.cseq
        response.headers["Contact"] = f"<sip:{local_ip}:{self._sip_port}>"
        response.headers["User-Agent"] = "VoiceAgent/1.0"
        if not body:
            response.headers["Content-Length"] = "0"

        return response

    async def _handle_bye(self, msg: SIPMessage, addr: tuple[str, int]) -> None:
        """Handle BYE — stop RTP and clean up call."""
        call_id = msg.call_id
        logger.info(f"BYE received for call {call_id}")

        # Send 200 OK
        response = self._build_response(msg, 200, "OK", addr)
        await self._send_sip(response, addr)

        # Clean up call
        await self._cleanup_call(call_id)

    async def _handle_options(self, msg: SIPMessage, addr: tuple[str, int]) -> None:
        """Handle OPTIONS (keepalive/capability query)."""
        response = self._build_response(msg, 200, "OK", addr)
        response.headers["Allow"] = "INVITE, ACK, BYE, CANCEL, OPTIONS"
        response.headers["Accept"] = "application/sdp"
        await self._send_sip(response, addr)

    async def _handle_cancel(self, msg: SIPMessage, addr: tuple[str, int]) -> None:
        """Handle CANCEL — terminate pending call."""
        call_id = msg.call_id
        logger.info(f"CANCEL received for call {call_id}")

        # Send 200 OK for CANCEL
        response = self._build_response(msg, 200, "OK", addr)
        await self._send_sip(response, addr)

        # Clean up
        await self._cleanup_call(call_id)

    async def _cleanup_call(self, call_id: str) -> None:
        """Stop RTP session and clean up call resources."""
        call = self._calls.pop(call_id, None)
        if call is None:
            return

        await call.rtp_session.stop()
        self._release_rtp_port(call.local_rtp_port)
        logger.info(f"Call {call_id} cleaned up")

        if self._on_call_end:
            try:
                await self._on_call_end(call_id)
            except Exception as e:
                logger.error(f"on_call_end callback error: {e}")

    async def send_bye(self, call_id: str) -> None:
        """Send BYE to terminate a call from our side."""
        call = self._calls.get(call_id)
        if not call:
            logger.warning(f"Cannot send BYE — call {call_id} not found")
            return

        local_ip = self._get_local_ip()
        bye = SIPMessage(method="BYE", request_uri=call.from_uri)
        bye.headers["Via"] = f"SIP/2.0/UDP {local_ip}:{self._sip_port};branch={_random_branch()}"
        bye.headers["From"] = f"<{call.to_uri}>;tag={call.to_tag}"
        bye.headers["To"] = f"<{call.from_uri}>;tag={call.from_tag}"
        bye.headers["Call-ID"] = call_id
        bye.headers["CSeq"] = "2 BYE"
        bye.headers["Contact"] = f"<sip:{local_ip}:{self._sip_port}>"
        bye.headers["Content-Length"] = "0"

        await self._send_sip(bye, call.remote_sip_addr)
        await self._cleanup_call(call_id)

    def get_active_calls(self) -> dict[str, ActiveCall]:
        """Return active calls."""
        return dict(self._calls)

    async def stop(self) -> None:
        """Stop all active calls and close SIP socket."""
        self._running = False
        # Stop all calls
        for call_id in list(self._calls.keys()):
            await self._cleanup_call(call_id)
        # Close SIP transport
        if self._transport:
            self._transport.close()
        logger.info("SIP server stopped")


# =============================================================================
# SIP UAC (User Agent Client) — for outbound calls
# =============================================================================


class SIPClient:
    """SIP User Agent Client for placing outbound calls through a SIP trunk.

    Args:
        local_ip: Local IP address for binding sockets.
        local_port: Local SIP port.
        trunk_host: SIP trunk hostname (e.g. sip.plivo.com).
        trunk_port: SIP trunk port.
        public_ip: Public IP to advertise in SDP and SIP headers.
            Required for NAT traversal — Plivo sends RTP to this address.
            If empty, falls back to local_ip.
        rtp_port_start: Start of RTP port range.
        rtp_port_end: End of RTP port range.
    """

    def __init__(
        self,
        local_ip: str,
        local_port: int = 5080,
        trunk_host: str = "",
        trunk_port: int = 5060,
        public_ip: str = "",
        rtp_port_start: int = 20000,
        rtp_port_end: int = 30000,
    ):
        self._local_ip = local_ip
        self._local_port = local_port
        self._public_ip = public_ip or local_ip
        self._trunk_host = trunk_host
        self._trunk_port = trunk_port
        self._rtp_port_pool: set[int] = set(range(rtp_port_start, rtp_port_end, 2))
        self._transport: asyncio.DatagramTransport | None = None
        self._protocol: _SIPProtocol | None = None
        self._pending_invites: dict[str, asyncio.Future] = {}
        self._calls: dict[str, ActiveCall] = {}
        self._keepalive_task: asyncio.Task | None = None
        self._running = False

    def _allocate_rtp_port(self) -> int | None:
        if not self._rtp_port_pool:
            return None
        return self._rtp_port_pool.pop()

    def _release_rtp_port(self, port: int) -> None:
        self._rtp_port_pool.add(port)

    async def start(self) -> None:
        """Start the SIP client UDP listener and keepalive task."""
        loop = asyncio.get_running_loop()
        self._protocol = _SIPProtocol(self._handle_response)
        self._transport, _ = await loop.create_datagram_endpoint(
            lambda: self._protocol,
            local_addr=("0.0.0.0", self._local_port),
        )
        self._running = True
        self._keepalive_task = asyncio.create_task(
            self._keepalive_loop(), name="sip_keepalive"
        )
        logger.info(f"SIP client started on {self._local_ip}:{self._local_port}")

    async def _keepalive_loop(self) -> None:
        """Send periodic in-dialog SIP keepalives to keep NAT/ALG bindings alive.

        Uses the active call's dialog info (Call-ID, From/To tags) so that
        SIP ALGs recognize the keepalive as part of the existing session.
        Also sends CRLF keepalive (RFC 5626) as a fallback.
        """
        trunk_addr = (self._trunk_host, self._trunk_port)
        cseq_counter = 100  # Start high to avoid conflicts with INVITE CSeq
        try:
            while self._running:
                await asyncio.sleep(15)
                if not self._calls or not self._transport or self._transport.is_closing():
                    continue

                # Send CRLF keepalive (RFC 5626 §4.4.1) — simplest NAT binding refresh
                self._transport.sendto(b"\r\n\r\n", trunk_addr)

                # Send in-dialog OPTIONS for each active call
                for call_id, call in list(self._calls.items()):
                    cseq_counter += 1
                    options = SIPMessage(
                        method="OPTIONS",
                        request_uri=call.to_uri,
                    )
                    options.headers["Via"] = (
                        f"SIP/2.0/UDP {self._public_ip}:{self._local_port}"
                        f";rport;branch={_random_branch()}"
                    )
                    options.headers["From"] = (
                        f"<{call.from_uri}>;tag={call.from_tag}"
                    )
                    to_hdr = f"<{call.to_uri}>"
                    if call.to_tag:
                        to_hdr += f";tag={call.to_tag}"
                    options.headers["To"] = to_hdr
                    options.headers["Call-ID"] = call_id
                    options.headers["CSeq"] = f"{cseq_counter} OPTIONS"
                    options.headers["Max-Forwards"] = "70"
                    options.headers["Content-Length"] = "0"
                    self._transport.sendto(options.to_bytes(), trunk_addr)
                    logger.debug(
                        f"SIP keepalive OPTIONS sent for call {call_id}"
                    )
        except asyncio.CancelledError:
            pass

    async def _handle_response(self, data: bytes, addr: tuple[str, int]) -> None:
        """Handle SIP responses to our requests."""
        try:
            msg = SIPMessage.parse(data)
        except Exception as e:
            logger.warning(f"Failed to parse SIP response: {e}")
            return

        call_id = msg.call_id
        if not call_id:
            return

        if msg.status_code == 200 and call_id in self._pending_invites:
            # 200 OK — send ACK and complete the call setup
            future = self._pending_invites.pop(call_id)

            # Log all response headers for debugging
            logger.info(f"200 OK for call {call_id}: {dict(msg.headers)}")

            # Check for Session-Expires (RFC 4028)
            session_expires = msg.get_header("Session-Expires")
            if session_expires:
                logger.warning(
                    f"Session-Expires: {session_expires} — "
                    "call will be terminated if not refreshed"
                )

            # Parse SDP
            remote_ip, remote_port = extract_remote_rtp_info(msg.body)

            # Send ACK
            await self._send_ack(msg, addr)

            if not future.done():
                future.set_result((remote_ip, remote_port, msg))

        elif msg.status_code and msg.status_code >= 400:
            future = self._pending_invites.pop(call_id, None)
            if future and not future.done():
                future.set_exception(
                    ConnectionError(f"SIP {msg.status_code} {msg.status_text}")
                )
        elif msg.method == "BYE":
            # Remote hung up
            logger.info(f"BYE received for call {call_id} — remote party hung up")
            response = SIPMessage(status_code=200, status_text="OK")
            response.headers["Via"] = msg.via
            response.headers["From"] = msg.from_header
            response.headers["To"] = msg.to_header
            response.headers["Call-ID"] = call_id
            response.headers["CSeq"] = msg.cseq
            response.headers["Content-Length"] = "0"
            if self._transport:
                self._transport.sendto(response.to_bytes(), addr)

            call = self._calls.pop(call_id, None)
            if call:
                await call.rtp_session.stop()
                self._release_rtp_port(call.local_rtp_port)
                logger.info(f"Call {call_id} cleaned up (BYE)")

    async def _send_ack(self, ok_msg: SIPMessage, addr: tuple[str, int]) -> None:
        """Send ACK for a 200 OK."""
        ack = SIPMessage(
            method="ACK",
            request_uri=ok_msg.get_header("Contact").strip("<>") or ok_msg.to_header,
        )
        ack.headers["Via"] = (
            f"SIP/2.0/UDP {self._public_ip}:{self._local_port}"
            f";rport;branch={_random_branch()}"
        )
        ack.headers["From"] = ok_msg.to_header  # Swap for ACK direction
        ack.headers["To"] = ok_msg.from_header
        ack.headers["Call-ID"] = ok_msg.call_id
        # CSeq for ACK matches the INVITE CSeq number
        cseq_num = ok_msg.cseq.split()[0] if ok_msg.cseq else "1"
        ack.headers["CSeq"] = f"{cseq_num} ACK"
        ack.headers["Content-Length"] = "0"

        if self._transport:
            self._transport.sendto(ack.to_bytes(), addr)

    async def call(
        self,
        to_uri: str,
        from_uri: str = "sip:agent@localhost",
        timeout: float = 30.0,
    ) -> tuple[RTPSession, str]:
        """Place an outbound call. Returns (RTPSession, call_id) on answer.

        Args:
            to_uri: SIP URI to call (e.g. sip:15551234567@sip.plivo.com)
            from_uri: Our SIP URI
            timeout: Seconds to wait for answer
        """
        call_id = f"{random.randint(100000, 999999)}@{self._public_ip}"
        branch = f"z9hG4bK{random.randint(100000, 999999)}"
        from_tag = f"{random.randint(100000, 999999)}"

        local_rtp_port = self._allocate_rtp_port()
        if local_rtp_port is None:
            raise RuntimeError("No RTP ports available")

        # Use public IP in SDP so Plivo can send RTP back to us
        sdp_body = build_sdp(self._public_ip, local_rtp_port)

        invite = SIPMessage(method="INVITE", request_uri=to_uri)
        invite.headers["Via"] = (
            f"SIP/2.0/UDP {self._public_ip}:{self._local_port}"
            f";rport;branch={branch}"
        )
        invite.headers["From"] = f"<{from_uri}>;tag={from_tag}"
        invite.headers["To"] = f"<{to_uri}>"
        invite.headers["Call-ID"] = call_id
        invite.headers["CSeq"] = "1 INVITE"
        invite.headers["Contact"] = f"<sip:{self._public_ip}:{self._local_port}>"
        invite.headers["Content-Type"] = "application/sdp"
        invite.headers["Content-Length"] = str(len(sdp_body))
        invite.headers["Max-Forwards"] = "70"
        invite.headers["User-Agent"] = "VoiceAgent/1.0"
        invite.headers["Supported"] = "timer"
        invite.headers["Session-Expires"] = "1800;refresher=uac"
        invite.headers["Min-SE"] = "90"
        invite.body = sdp_body

        # Set up future for response
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending_invites[call_id] = future

        # Send INVITE
        trunk_addr = (self._trunk_host, self._trunk_port)
        if self._transport:
            self._transport.sendto(invite.to_bytes(), trunk_addr)
        logger.info(f"INVITE sent to {to_uri} via {trunk_addr}")

        try:
            remote_ip, remote_port, ok_msg = await asyncio.wait_for(future, timeout)
        except TimeoutError:
            self._pending_invites.pop(call_id, None)
            self._release_rtp_port(local_rtp_port)
            raise TimeoutError(f"No answer for call to {to_uri}") from None

        # Extract to_tag from 200 OK To header
        to_tag = ""
        to_header = ok_msg.to_header or ""
        for part in to_header.split(";"):
            p = part.strip()
            if p.startswith("tag="):
                to_tag = p[4:]
                break

        # Start RTP session
        rtp_session = RTPSession(local_rtp_port, (remote_ip, remote_port))
        await rtp_session.start()

        self._calls[call_id] = ActiveCall(
            call_id=call_id,
            from_uri=from_uri,
            to_uri=to_uri,
            remote_sip_addr=trunk_addr,
            rtp_session=rtp_session,
            local_rtp_port=local_rtp_port,
            from_tag=from_tag,
            to_tag=to_tag,
        )

        return rtp_session, call_id

    async def hangup(self, call_id: str) -> None:
        """Send BYE to terminate an outbound call."""
        call = self._calls.pop(call_id, None)
        if not call:
            return

        bye = SIPMessage(method="BYE", request_uri=call.to_uri)
        bye.headers["Via"] = (
            f"SIP/2.0/UDP {self._public_ip}:{self._local_port}"
            f";rport;branch={_random_branch()}"
        )
        bye.headers["From"] = f"<{call.from_uri}>;tag={call.from_tag}"
        bye.headers["To"] = f"<{call.to_uri}>"
        bye.headers["Call-ID"] = call_id
        bye.headers["CSeq"] = "2 BYE"
        bye.headers["Content-Length"] = "0"

        if self._transport:
            self._transport.sendto(bye.to_bytes(), call.remote_sip_addr)

        await call.rtp_session.stop()
        self._release_rtp_port(call.local_rtp_port)

    async def stop(self) -> None:
        """Stop client and clean up all calls."""
        self._running = False
        if self._keepalive_task and not self._keepalive_task.done():
            self._keepalive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._keepalive_task
        for call_id in list(self._calls.keys()):
            await self.hangup(call_id)
        if self._transport:
            self._transport.close()
