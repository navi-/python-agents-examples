# GPT-5.2 Mini + Deepgram Flux + ElevenLabs Flash v2.5 — LiveKit Voice Agent

LiveKit Agents framework orchestration with Plivo SIP trunking — fully automated setup. On startup, the agent creates all SIP trunks (LiveKit + Plivo Zentrunk), dispatch rules, and phone number mappings. No manual Console steps. STT is Deepgram Flux (`/listen/v2` WebSocket, conversational audio). LLM is OpenAI `gpt-5.2-mini` via Responses API with 5 function tools. TTS is ElevenLabs `eleven_flash_v2_5` (low-latency streaming). Turn detection uses Silero VAD (speech presence, barge-in) + LiveKit MultilingualModel (135M transformer, semantic end-of-turn on partial transcripts in a 4-turn sliding window). Noise cancellation is Krisp BVC. LiveKit handles all audio transport and format conversion natively.

- Inbound: `python -m inbound.agent dev` — auto-creates LiveKit inbound trunk + dispatch rule, Plivo origination URI + inbound trunk, maps phone number. Zero manual config beyond .env
- Outbound: `python -m outbound.agent dev` — auto-creates Plivo SIP credentials + outbound trunk (derives termination domain), LiveKit outbound trunk. Calls via `initiate_call()`
- Barge-in handled by `VoicePipelineAgent` with `allow_interruptions=True` — Silero VAD detects speech onset, pipeline cancels in-flight TTS

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- LiveKit server — [self-hosted](https://docs.livekit.io/home/self-hosting/local/) (local dev) or [Cloud](https://cloud.livekit.io/) (free tier)
- TCP tunnel for SIP (e.g., [ngrok](https://ngrok.com/)) if running LiveKit locally
- OpenAI, Deepgram, ElevenLabs API keys
- Plivo account with Zentrunk SIP trunking enabled

## Quick Start

```bash
git clone <repo> && cd gpt5.2-deepgramflux-elevenflashv2.5-livekit
uv sync
cp .env.example .env
# Fill in: LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET,
#          OPENAI_API_KEY, DEEPGRAM_API_KEY, ELEVEN_API_KEY,
#          PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, PLIVO_PHONE_NUMBER
uv run python -m inbound.agent dev
```

That's it. On startup the agent:

1. **[LiveKit]** Creates inbound SIP trunk (Krisp enabled) + dispatch rule (`inbound-*` rooms)
2. **[Plivo]** Creates Zentrunk origination URI → LiveKit SIP endpoint
3. **[Plivo]** Creates Zentrunk inbound trunk with the origination URI
4. **[Plivo]** Maps your phone number to the trunk
5. Starts the agent worker — auto-joins rooms when callers connect

Call your Plivo number. Done.

## Outbound Calls

```bash
uv run python -m outbound.agent dev
```

On startup: creates Plivo SIP credentials → Plivo outbound trunk (gets termination domain) → LiveKit outbound trunk.

Trigger a call:

```python
from outbound.agent import initiate_call

result = await initiate_call(
    phone_number="+15551234567",
    opening_reason="Follow up on your demo request",
)
```

## What Happens on Startup

### Inbound (`inbound/agent.py`)

```
setup_sip_inbound()
├── LiveKit: create_sip_inbound_trunk(krisp=True)
├── LiveKit: create_sip_dispatch_rule(prefix="inbound-")
├── Plivo:  create_origination_uri(→ LiveKit SIP endpoint)
├── Plivo:  create_inbound_trunk(uri=above)
└── Plivo:  map_number_to_trunk(phone, trunk)

_entrypoint()
├── VoicePipelineAgent(vad=Silero, stt=Deepgram, llm=GPT-5.2, tts=ElevenLabs)
├── turn_detector=MultilingualModel()
├── noise_cancellation=BVC()
└── agent.say("Hello! How can I help you today?")
```

### Outbound (`outbound/agent.py`)

```
setup_sip_outbound()
├── Plivo:  create_credential(auth_id, auth_token)
├── Plivo:  create_outbound_trunk(cred) → termination domain
└── LiveKit: create_sip_outbound_trunk(address=domain, auth=plivo)

initiate_call(phone, reason, objective)
└── LiveKit: create_sip_participant(trunk, phone, room_metadata)
```

## Project Structure

```
gpt5.2-deepgramflux-elevenflashv2.5-livekit/
├── inbound/
│   ├── agent.py              # SIP setup + agent worker
│   └── system_prompt.md
├── outbound/
│   ├── agent.py              # SIP setup + agent worker + initiate_call()
│   └── system_prompt.md
├── utils.py                  # Phone normalization + PlivoZentrunk API client
├── tests/
├── pyproject.toml
├── .env.example
├── Dockerfile
└── README.md
```

## Local Development (Self-Hosted LiveKit)

LiveKit SIP requires 3 services: `livekit-server`, `livekit-sip`, and Redis. The easiest way is docker-compose:

### 1. Start LiveKit infrastructure

```bash
docker compose up -d
```

This starts Redis, livekit-server (port 7880), and livekit-sip (port 5060) with dev keys (`devkey`/`secret`).

### 2. Tunnel the SIP port

Plivo needs to reach your local SIP service. Tunnel port 5060:

```bash
ngrok tcp 5060
# Output: Forwarding tcp://0.tcp.ngrok.io:12345 -> localhost:5060
```

### 3. Configure .env

```bash
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
LIVEKIT_SIP_ENDPOINT=0.tcp.ngrok.io:12345   # tunnel address from step 2
OPENAI_API_KEY=...
DEEPGRAM_API_KEY=...
ELEVEN_API_KEY=...
PLIVO_AUTH_ID=...
PLIVO_AUTH_TOKEN=...
PLIVO_PHONE_NUMBER=...
```

### 4. Run the agent

```bash
uv run python -m inbound.agent dev
```

On startup: creates SIP trunks (LiveKit + Plivo), maps phone number, starts worker. Call your Plivo number.

## LiveKit Cloud Setup

Alternatively, use [LiveKit Cloud](https://cloud.livekit.io/) (free tier):

```bash
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=<from dashboard>
LIVEKIT_API_SECRET=<from dashboard>
LIVEKIT_SIP_ENDPOINT=<project_id>.sip.livekit.cloud   # from Project Settings → SIP URI
```

## Deployment

Same `.env`, same command. The SIP setup is idempotent — checks for existing trunks/rules by name, reuses them. Safe to restart.

```bash
# Docker
docker build -t livekit-agent .
docker run --env-file .env livekit-agent

# Or directly
uv run python -m inbound.agent start
```

## Testing

```bash
uv run pytest tests/test_integration.py -v       # Unit tests (offline)
uv run pytest tests/test_e2e_live.py -v           # E2E (needs LiveKit creds)
```

## License

MIT
