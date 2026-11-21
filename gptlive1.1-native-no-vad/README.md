# GPT-Live 1.1 Native (no client VAD) — Plivo Voice Agent

Plivo bidirectional audio streaming bridged over raw `websockets` + asyncio (no SDK/framework) to OpenAI GPT-Live `gpt-live-1.1` (`wss://api.openai.com/v1/live/sessions`, `Authorization: Bearer`), a full-duplex speech-to-speech model that listens while it speaks. `session.start` sets one audio format for both directions, `audio/pcmu` at 8000 Hz, so Plivo μ-law 8kHz (base64 JSON, 160B/20ms) goes in as `session.input_audio.append` and comes back as `session.output_audio.delta` with no transcoding or resampling.
- No client-side VAD and no turn commit: turn-taking, backchannels and interruptions are decided by the model from the continuous caller stream; voice `marin` and model ids are `.env` config passed verbatim.
- Barge-in: GPT-Live stops speaking by itself and sends no interruption event; when caller speech arrives (`session.input_transcript.delta`) while more than 300 ms of agent audio is still unplayed (send queue + 20 ms per chunk already sent to Plivo), the queue is drained and Plivo gets `clearAudio`.
- Tools run in Responses delegation (`delegation.type: responses`, backend model `gpt-6-luna`): 5 function calls arrive in `response.event` envelopes and are answered with `response.item.create` + `response.create`.
- `end_call` hangs up via Plivo REST `calls.delete` once the goodbye audio has been quiet for 1 s and its Plivo `checkpoint` is acknowledged with `playedStream` (15 s fallback deadline); `session.close` then waits for `session.closed` usage.

## Features

- **One model, full duplex**: GPT-Live hears the caller while it is speaking. There is no STT/LLM/TTS chain to orchestrate and no VAD to tune; the model decides when to talk, when to keep going through an "mm-hm", and when to stop.
- **μ-law 8kHz end to end**: `audio/pcmu` at 8000 Hz is Plivo's own format, so audio is forwarded as is. No `numpy`, `scipy` or `torch` at runtime.
- **Tools through delegation**: the voice model hands tool work to a backend model (`OPENAI_DELEGATION_MODEL`), which calls this example's five functions (order status, SMS, callback, transfer, end call).
- **Inbound and outbound**: inbound calls via `/answer`; outbound calls with Plivo's Make Call API and a per-call `greeting` on the answer URL.
- **Authenticated webhooks and stream**: every Plivo webhook and the `/ws` stream must carry a valid Plivo V3 signature.
- **One command to try it**: `--tunnel` opens a Cloudflare quick tunnel and points your Plivo number at it.
- **Observability**: structured JSON events per turn and session, optional Redis Streams sink and OpenTelemetry spans.

## Verification status

**The GPT-Live WebSocket session itself has not been run against the live API yet**: the environment this example was written in could not open outbound WebSockets. It was built from OpenAI's published GPT-Live documentation (read 2026-11-21) and verified offline against `tests/fake_live.py`, a local server that speaks the documented wire protocol. Checked against the real API over HTTPS (2026-11-21): `gpt-live-1.1` and `gpt-6-luna` are available models; `gpt-6-luna` accepts this example's `DELEGATION_INSTRUCTIONS` and five function tools through the Responses API, returns a `function_call` output item (`call_id`, `name`, JSON-string `arguments`) and turns a `function_call_output` into one spoken-style sentence; the text-to-speech helper the e2e test uses for caller audio works.

| Verified offline (155 unit + 7 local integration tests) | To confirm on the first live run |
|---|---|
| `session.start` shape, handshake, pre-start audio buffering | `gpt-live-1.1` accepts `audio/pcmu` at 8000 on the primary WebSocket (the schema allows it; no docs example uses it) |
| μ-law pass-through, 160-byte `playAudio` framing | Whether output audio arrives in real time or in bursts (decides how often the barge-in flush fires) |
| Turn segmentation from transcript deltas, structured events | How promptly `session.input_transcript.delta` follows caller speech |
| Barge-in flush for burst and real-time delivery | The nested completion event name inside `response.event` (`response.completed`; a 1 s fallback covers a different name) |
| Delegated function calls, `end_call` hangup via checkpoint | Greeting wording: the model is instructed, not forced, to read it exactly |
| Plivo signature checks on webhooks and `/ws`, outbound answer URL | `session.closed` usage fields |

Run `uv run pytest tests/test_integration.py -v -k GPTLive` and `uv run pytest tests/test_e2e_live.py -v -s` with `OPENAI_API_KEY` set to check the right-hand column; the e2e test prints how far ahead of real time audio arrives and whether `clearAudio` was needed.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- An OpenAI API key with access to GPT-Live (`gpt-live-1.1`) and the delegation model ([platform.openai.com](https://platform.openai.com))
- A Plivo account with a phone number
- For a local run, `cloudflared` for `--tunnel` (or ngrok); for hosting, Docker

## Quick Start

GPT-Live's WebSocket API has no telephony, so a small bridge server has to sit between Plivo and OpenAI. That server is this example. Plivo just needs a public URL to reach it.

```
Caller ──► Plivo ══ WebSocket ══► this server (bridge) ══ WebSocket ══► OpenAI GPT-Live
                                        ▲
                         reachable at PUBLIC_URL (tunnel or host)
```

| | Option A: one command (local) | Option B: Docker (any host) |
|---|---|---|
| **Use for** | Trying it on your laptop | Running on a server |
| **Public URL** | Created for you (free Cloudflare quick tunnel) | Your host's HTTPS URL |
| **Needs** | `uv`, `cloudflared` | Docker, a host that allows long-lived WebSockets |
| **Plivo number setup** | Automatic | Automatic when `PUBLIC_URL` is set |

For both options, first put four values in `.env`:

```bash
git clone https://github.com/plivo/python-agents-examples && cd python-agents-examples/gptlive1.1-native-no-vad
cp .env.example .env   # set OPENAI_API_KEY, PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, PLIVO_PHONE_NUMBER
```

### Option A: one command, local

[Install `cloudflared`](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/downloads/) once, then start the agent with a tunnel:

```bash
uv run python -m inbound.server --tunnel
```

What `--tunnel` does:

1. Starts `cloudflared tunnel --url http://localhost:$SERVER_PORT` (inbound, default 8000; the outbound server uses `$OUTBOUND_SERVER_PORT`, default 8001). The quick tunnel needs no Cloudflare account.
2. Uses the `https://<random>.trycloudflare.com` URL as `PUBLIC_URL`.
3. Points your Plivo number at it, creating or updating the `GPT_Live_Native` Plivo application. Plivo only accepts a URL once it can resolve the hostname, so for a new tunnel the server retries in the background for up to 3 minutes. In testing this took about 70 s.
4. Logs `Ready! Call +<number> to talk to the agent (Ctrl+C to stop)` once the server accepts connections and Plivo has accepted the URL.

Call the number. Ctrl+C stops the server and the tunnel. The URL changes on every run and the number is re-pointed each time.

For outbound calls, run `uv run python -m outbound.server --tunnel`. It listens on port 8001, so it can run at the same time as the inbound server, each with its own tunnel URL. Once it accepts connections it logs `Ready!` with a ready-to-paste cURL for Plivo's Make Call API with this server's answer URL; see [Outbound Calls](#outbound-calls). In short:

```bash
set -a && source .env && set +a   # PLIVO_AUTH_ID / PLIVO_AUTH_TOKEN in the shell
curl -X POST "https://api.plivo.com/v1/Account/$PLIVO_AUTH_ID/Call/" \
  -u "$PLIVO_AUTH_ID:$PLIVO_AUTH_TOKEN" -H "Content-Type: application/json" \
  -d '{"from": "<your Plivo number>", "to": "+1234567890",
       "answer_url": "https://<random>.trycloudflare.com/outbound/answer?greeting=Hi%2C%20this%20is%20Alex%20from%20TechFlow.%20Is%20now%20a%20good%20time%20for%20a%20quick%20chat%3F",
       "hangup_url": "https://<random>.trycloudflare.com/outbound/hangup",
       "answer_method": "POST", "hangup_method": "POST"}'
```

### Option B: Docker

```bash
docker build -t gptlive1.1-native-no-vad .
docker run --env-file .env -e PUBLIC_URL=https://your-host.example.com -p 8000:8000 gptlive1.1-native-no-vad
```

The container listens on port 8000 and configures the Plivo number from `PUBLIC_URL` on startup. For outbound calls, append `uv run python -m outbound.server` to `docker run` and publish port 8001 instead:

```bash
docker run --env-file .env -e PUBLIC_URL=https://your-outbound-host.example.com -p 8001:8001 gptlive1.1-native-no-vad \
  uv run python -m outbound.server
```

Then place calls with Plivo's Make Call API using `answer_url=https://your-outbound-host.example.com/outbound/answer`. See [Deployment](#deployment) for suitable hosts.

### Other ways to expose the server

Without `--tunnel`, any tunnel works. For example, run `ngrok http 8000`, set `PUBLIC_URL` to its HTTPS URL, then `uv run python -m inbound.server`. For outbound, tunnel port 8001 (`ngrok http 8001`) instead.

To run both servers at once behind fixed URLs, give each process its own `PUBLIC_URL` on the command line, e.g. `PUBLIC_URL=https://out.example.com uv run python -m outbound.server`. A variable set in the shell takes precedence over `.env`, because `load_dotenv()` does not override variables that are already set.


## Project Structure

```
gptlive1.1-native-no-vad/
├── inbound/
│   ├── agent.py              # GPTLiveVoiceAgent: session.start, 3 asyncio tasks, tools, hangup
│   ├── server.py             # FastAPI: /answer, /ws, /hangup, /fallback, /hold; --tunnel
│   └── system_prompt.md      # Inbound prompt (delegation, backchannel, interruption policies)
├── outbound/
│   ├── agent.py              # Same agent; greeting from the answer_url, outbound call context
│   ├── server.py             # FastAPI: /outbound/answer, /outbound/hangup, /ws; --tunnel
│   └── system_prompt.md      # Outbound prompt
├── utils.py                  # normalize_phone_number, pass-through audio wrappers, quick tunnel
├── tests/
│   ├── fake_live.py          # Local stand-in for the GPT-Live WebSocket API
│   ├── helpers.py            # Signing, servers, tunnels, recordings; test-only μ-law codec
│   ├── test_integration.py   # Unit + local integration (offline) + direct GPT-Live check
│   ├── test_e2e_live.py      # Real GPT-Live, simulated Plivo, spoken caller turns, WAV capture
│   ├── test_live_call.py     # Real inbound call
│   ├── test_multiturn_voice.py  # Real multi-turn call + barge-in
│   └── test_outbound_call.py # Real outbound call
├── pyproject.toml
├── .env.example
├── Dockerfile
└── README.md
```

## How It Works

```
Plivo (μ-law 8kHz, 20ms frames)
   │  media ──► plivo_rx ──► session.input_audio.append (base64 μ-law) ──►┐
   │                                                                       │  GPT-Live
   │  playAudio ◄── plivo_tx ◄── send queue ◄── openai_rx ◄── session.output_audio.delta
   │  clearAudio ◄── barge-in flush ◄──────── session.input_transcript.delta
   │  checkpoint / playedStream (goodbye before hangup)
                                   openai_rx ◄── response.event (function_call) 
                                   openai_rx ──► response.item.create + response.create
```

Three asyncio tasks run per call (`_run_streaming_tasks`):

| Task | Method | Job |
|---|---|---|
| `plivo_rx` | `_receive_from_plivo()` | Decode Plivo `media` frames and append them to the session; handle `playedStream`, `stop` |
| `openai_rx` | `_receive_from_openai()` | `session.start` handshake, then dispatch GPT-Live events: queue audio, track turns, run delegated function calls |
| `plivo_tx` | `_send_to_plivo()` | Chunk queued audio into 160-byte `playAudio` messages, send checkpoints, run the 100 ms housekeeping tick |

### Session start

`session.start` is the first message on the socket and nothing else is sent until `session.started` arrives (10 s timeout). Caller audio that arrives earlier is buffered (2 s cap) and flushed in order. Model, instructions, audio format and voice are fixed for the life of the session.

```json
{
  "type": "session.start",
  "session": {
    "model": "gpt-live-1.1",
    "instructions": "<system_prompt.md + call context>",
    "audio": {"format": {"type": "audio/pcmu", "rate": 8000}, "output": {"voice": "marin"}},
    "input": [{"type": "message", "role": "developer",
               "content": [{"type": "input_text", "text": "... <say>greeting</say>"}]}],
    "delegation": {"type": "responses", "responses": {
      "model": "gpt-6-luna", "instructions": "<backend prompt + call context>",
      "tools": ["<5 function tools>"], "tool_choice": "auto"}}
  }
}
```

### Greeting

GPT-Live has no field that speaks literal text. Following OpenAI's guidance, the greeting is a developer message in `session.input` that tells the model to greet now and read the line inside `<say>` tags exactly. The model generates the speech, so treat the wording as very likely rather than guaranteed. Inbound uses `AGENT_GREETING`; outbound uses the `greeting` query param of the answer URL.

### Turn-taking and barge-in

There is nothing to commit and nothing to cancel: the caller's audio streams in continuously (silence included) and the model decides when to speak. The prompt's "Backchannel Policy" and "Interruption Policy" sections steer that behaviour.

GPT-Live sends no "user started speaking" event, and it cannot take back audio it has already delivered. The agent therefore tracks how much agent audio the caller has not heard yet: bytes in the send queue plus audio already sent to Plivo but not yet played (Plivo plays 20 ms per chunk, so the playback clock is known without a round trip). When a caller transcript fragment arrives after agent speech and that backlog is above `BARGE_IN_FLUSH_BACKLOG_MS` (300 ms), the queue is drained and `clearAudio` is sent. If GPT-Live delivers audio in real time the backlog stays near zero and nothing is flushed; the model simply stops.

### Turns in the logs

Transcript deltas are fragments with session-timeline `start_ms`/`end_ms` and no turn markers. The agent cuts turns where the speaker changes: caller fragments, then agent fragments, until the caller speaks again. The greeting is turn 1. `response_gap_ms` is the gap on GPT-Live's own timeline between the end of the caller's speech and the start of the reply.

### Ending the call

`end_call` (run by the backend model) marks the hangup as pending. GPT-Live has no end-of-speech event, so the agent watches the audio itself: once no chunk with speech has arrived for 1 s (`END_CALL_QUIET_S`), it queues a Plivo `checkpoint` behind the last audio and hangs up over REST when Plivo answers `playedStream`. If nothing is spoken after `end_call`, it stops waiting after 4 s; 15 s is the hard limit. At teardown the agent sends `session.close` and waits up to 3 s for `session.closed`, which confirms the session was finalized and carries its usage.

## Webhook authentication

Both servers are exposed on a public URL, so they accept only requests and audio streams that come from Plivo. The checks below always run; there is no setting to turn them off:

- **Plivo webhooks** (`/answer`, `/hangup`, `/fallback`, `/hold`, `/outbound/answer`, `/outbound/hangup`) must carry a valid [Plivo V3 signature](https://www.plivo.com/docs/voice/concepts/signature-validation) (`X-Plivo-Signature-V3` + `X-Plivo-Signature-V3-Nonce`, an HMAC-SHA256 signature keyed with `PLIVO_AUTH_TOKEN`). `verify_plivo_signature()`, a FastAPI dependency in each `server.py`, checks them with the Plivo SDK's `validate_v3_signature()`. The signed params are the form fields for POST and the query string for GET. Otherwise the request gets **403** and a warning is logged with the path and reason (`missing …` or `signature mismatch`). Signatures are never logged.
- **`/ws`** (the audio stream): Plivo sends the same two headers when it connects, and the same `verify_plivo_signature()` checks them before the WebSocket is accepted. Plivo signs the stream URL as `http://<host>/ws`: the `http` scheme, and no query string. A connection without a valid signature gets **403**, a warning is logged (`Rejected Plivo stream /ws: …`), and no agent is started. The signature does not cover `?body=`.

**`PUBLIC_URL` must be exactly the URL Plivo is configured to call.** Plivo signs the URL it requested. Behind a tunnel or reverse proxy, this server sees `http://localhost:8000/...` instead, so the signed URL is rebuilt as `PUBLIC_URL` (trailing `/` stripped) + request path + raw query string. `request.url` is not used. For `/ws` the signed URL is rebuilt as `http://` + the host of `PUBLIC_URL` + request path. The scheme (`https` vs `http`), the host and any path prefix must match the answer and hangup URLs in the Plivo application or Make Call request. `PUBLIC_URL=https://agent.example.com` and `https://agent.example.com/` are equivalent. `http://agent.example.com` or another hostname for the same server is not. The outbound `answer_url` query string (`greeting`) is part of what Plivo signs: for a POST, the Plivo SDK's base string is `<url>?<query params sorted, URL-decoded>.<form params sorted, name+value concatenated>`. Editing the query string therefore invalidates the signature.

**`--tunnel`**: the quick-tunnel URL becomes `PUBLIC_URL` at startup, and the verifier reads it on each request, so signatures are checked against the tunnel URL. The inbound server points the number at that same URL. For outbound, use the tunnel URL from the `Ready!` line in your `answer_url`.

**`PLIVO_AUTH_TOKEN` is required**: with it empty, the server refuses to start. Tests use a dummy token and sign their requests the way Plivo does. Use a subaccount's auth token if the number and calls belong to a subaccount, because Plivo signs with the token of the account that owns the call.


## Audio Formats

| Leg | Format | Notes |
|---|---|---|
| Plivo → server | μ-law 8kHz mono, base64 in JSON `media` events | 160 bytes per 20 ms frame |
| Server → GPT-Live | `session.input_audio.append`, base64 μ-law 8kHz | Same bytes, re-wrapped (`plivo_to_openai` is a pass-through) |
| GPT-Live → server | `session.output_audio.delta`, base64 μ-law 8kHz | `audio.format` covers both directions |
| Server → Plivo | `playAudio`, `audio/x-mulaw`, `sampleRate` 8000 | Re-chunked to exactly 160 bytes (`openai_to_plivo` is a pass-through) |

## Outbound Calls

You place outbound calls with [Plivo's Make Call API](https://www.plivo.com/docs/voice/api/call/make-a-call). This server has no dial endpoint and keeps no call records: it answers Plivo's webhooks and bridges audio.

```
your shell / backend ──POST /v1/Account/{auth_id}/Call/──► Plivo ──dials──► callee
                                                             │ callee answers
                        /outbound/answer?greeting=…       ◄─┘ (answer_url)
                          └─► <Stream> ─► /ws ─► run_agent(greeting)
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/outbound/answer` | GET/POST | Plivo answer webhook (your `answer_url`). Reads the optional query param `greeting`, plus Plivo's `CallUUID`/`From`/`To`/`ParentCallUUID`/`SIP-*` fields, and returns `<Stream>` to `/ws` with them in the base64 `body` |
| `/outbound/hangup` | POST | Plivo hangup webhook (your optional `hangup_url`); logs `CallUUID`, `Duration`, `HangupCause` |
| `/ws` | WebSocket | Plivo audio stream (Plivo signature required); runs the agent |
| `/` | GET | Health check |

Once it accepts connections, the server logs a `Ready!` line with the cURL below, its own `PUBLIC_URL` and `PLIVO_PHONE_NUMBER` filled in. Credentials stay shell references (`set -a && source .env && set +a` exports them):

```bash
curl -X POST "https://api.plivo.com/v1/Account/$PLIVO_AUTH_ID/Call/" \
  -u "$PLIVO_AUTH_ID:$PLIVO_AUTH_TOKEN" -H "Content-Type: application/json" \
  -d '{"from": "+14155550100", "to": "+1234567890",
       "answer_url": "https://your-host.example.com/outbound/answer?greeting=Hi%2C%20this%20is%20Alex%20from%20TechFlow.%20I%27m%20reaching%20out%20because%20you%20requested%20a%20demo.%20Is%20now%20a%20good%20time%20for%20a%20quick%20chat%3F",
       "hangup_url": "https://your-host.example.com/outbound/hangup",
       "answer_method": "POST", "hangup_method": "POST"}'
```

URL-encode the query value. From Python, `plivo.RestClient().calls.create(from_=..., to_=..., answer_url=..., answer_method="POST", hangup_url=...)` does the same (see `tests/test_outbound_call.py`).

| `answer_url` param | Used for | When missing |
|---|---|---|
| `greeting` | The line the model is told to read exactly when the callee answers (a developer message in `session.input` with the text in `<say>` tags). GPT-Live generates the speech, so wording can differ slightly; see [Greeting](#greeting) | `DEFAULT_OUTBOUND_GREETING` ("Hi, this is Alex from TechFlow, following up on your recent interest in our products. Is now a good time for a quick chat?") |

The greeting is the only per-call input. The system prompt is `outbound/system_prompt.md` as is (plus the call context); it tells the model that it opens with the greeting it is given, so write the prompt for your use case there. For status, duration or recordings, use Plivo's Call API and call logs, or your `hangup_url`.


## Function Calling

GPT-Live does not call functions itself. In Responses delegation the voice model hands a task to the backend model, which calls the functions; the voice model speaks the outcome.

| Function | Purpose |
|---|---|
| `check_order_status` | Look up an order by number or email (mock data) |
| `send_sms` | Text information to the caller (mock) |
| `schedule_callback` | Schedule a specialist callback (mock) |
| `transfer_call` | Transfer to a human (mock) |
| `end_call` | Hang up after the goodbye has played |

Flow: `session.delegation.created` → `response.event` envelopes carrying the backend model's Responses events → a `response.output_item.done` whose item is a `function_call` runs the function (10 s timeout) → when the backend response completes, every result is returned as `response.item.create` (`function_call_output`) followed by one `response.create`. If no completion event arrives within 1 s of the last call, the results are returned anyway. Replace the mock implementations in `inbound/agent.py` and `outbound/agent.py` with your own.

## Observability

Set `LOG_FORMAT=json` for one JSON object per log line. Structured events:

| Event | Fields |
|---|---|
| `call_answered` | `call_id`, `leg_call_id`, `from_number`, `to_number`, `sip_headers`, `stream_id` |
| `user_text` / `agent_text` | `call_id`, `turn`, `text` |
| `turn_complete` | `turn`, `barge_in`, `user_text`, `agent_text`, `response_gap_ms`, `plivo_rx_bytes`, `plivo_tx_chunks` |
| `session_end` | `duration_s`, `turns`, `barge_ins`, `errors`, `response_gap_avg_ms`, `rx_bytes`, `tx_chunks`, `openai_session_id`, `openai_close_reason`, `openai_usage` |

Optional sinks: `REDIS_EVENTS_URL` publishes every log record to a Redis Stream (`uv sync --extra streaming`); with `uv sync --extra observability`, OpenTelemetry spans `session`, `plivo_rx`, `openai_rx`, `plivo_tx` and `tool.<name>` are exported to `OTEL_EXPORTER_OTLP_ENDPOINT`.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI API key (GPT-Live and the delegation model) |
| `OPENAI_LIVE_MODEL` | `gpt-live-1.1` | `session.model`, passed verbatim |
| `OPENAI_LIVE_VOICE` | `marin` | `session.audio.output.voice`, passed verbatim |
| `OPENAI_DELEGATION_MODEL` | `gpt-6-luna` | `session.delegation.responses.model`, passed verbatim |
| `OPENAI_DELEGATION_REASONING_EFFORT` | unset | `delegation.responses.reasoning.effort`; omitted when empty |
| `OPENAI_LIVE_URL` | `wss://api.openai.com/v1/live/sessions` | Override for a proxy or the test fake |
| `AGENT_GREETING` | "Hi, this is Alex from TechFlow. I'm built with OpenAI GPT-Live on Plivo. How can I help you today?" | Inbound greeting |
| `PLIVO_AUTH_ID` / `PLIVO_AUTH_TOKEN` | — | Plivo credentials; the token also keys the signature checks (required) |
| `PLIVO_PHONE_NUMBER` | — | Number to point at this server on startup |
| `PUBLIC_URL` | — | Public HTTPS URL of this server; not needed with `--tunnel` |
| `DEFAULT_COUNTRY_CODE` | `US` | Region for phone numbers without a country prefix |
| `SERVER_PORT` / `OUTBOUND_SERVER_PORT` | `8000` / `8001` | Inbound / outbound server ports |
| `LOG_LEVEL` | `normal` | `verbose`, `normal`, `quiet` |
| `LOG_FORMAT` / `LOG_FILE` | `text` / unset | `json` for structured logs; optional JSON log file |
| `REDIS_EVENTS_URL` / `REDIS_STREAM_KEY` | unset / `voice-agent:events` | Redis Streams sink |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | unset | OTLP endpoint for traces |

Timing constants (`BARGE_IN_FLUSH_BACKLOG_MS`, `END_CALL_QUIET_S`, `END_CALL_GRACE_S`, …) are at the top of each `agent.py`.

### System prompt

The prompt is `inbound/system_prompt.md` / `outbound/system_prompt.md`; there is no env override. Edit the file, or mount your own over it:

```bash
docker run --env-file .env -v ./my_prompt.md:/app/inbound/system_prompt.md -p 8000:8000 gptlive1.1-native-no-vad
```

GPT-Live instructions should stay short and say when to delegate. Keep the "Delegation Policy", "Backchannel Policy" and "Interruption Policy" sections when you rewrite the prompt: they are how turn-taking and tool use are steered. Backend procedures belong in `DELEGATION_INSTRUCTIONS` in `agent.py`.

## Dependencies

Runtime: `fastapi`, `uvicorn`, `websockets`, `plivo`, `python-dotenv`, `python-multipart`, `loguru`, `phonenumbers`. No audio or ML libraries: nothing is transcoded. Extras: `streaming` (Redis), `observability` (OpenTelemetry, Traceloop).

## Pricing

GPT-Live bills the voice session by duration (per second), and the delegation model separately at its own token rates. See [OpenAI's pricing page](https://developers.openai.com/api/docs/pricing) for current numbers. Plivo bills the call minutes.

## Testing

```bash
# Unit tests (offline, no API keys)
uv run pytest tests/test_integration.py -v -k "unit"

# Local server against the fake GPT-Live server (offline, no API keys)
uv run pytest tests/test_integration.py -v -k "local"

# This example's session.start against the real API (needs OPENAI_API_KEY)
uv run pytest tests/test_integration.py -v -k "GPTLive"

# E2E with real GPT-Live, no phone call: greeting, spoken questions, tool call, goodbye, barge-in.
# Caller turns are synthesized with OpenAI text-to-speech.
uv run pytest tests/test_e2e_live.py -v -s

# Real calls (need Plivo creds, PLIVO_TEST_NUMBER, ngrok)
uv run pytest tests/test_live_call.py tests/test_multiturn_voice.py tests/test_outbound_call.py -v -s
```

### Voice samples

The e2e test can keep each simulated call as a stereo 8 kHz WAV, caller on the left channel and agent on the right, exactly as the audio crossed the Plivo stream:

```bash
SAVE_CALL_AUDIO_DIR=samples uv run pytest tests/test_e2e_live.py -v -s
# samples/e2e-greeting.wav, samples/e2e-conversation.wav, samples/e2e-barge-in.wav
```

## Deployment

Use any host that runs containers and keeps WebSocket connections open for the length of a call:

| Host | Works? | Notes |
|---|---|---|
| Fly.io, Render, Railway, a VM or Kubernetes | ✅ | Long-lived WebSockets supported |
| Google Cloud Run | ✅ | Raise the request timeout (max 60 min) above your longest call |
| Vercel/Netlify functions, AWS Lambda | ❌ | No long-lived WebSocket server |

```bash
docker build -t gptlive1.1-native-no-vad .

# Inbound (default): configures the Plivo number from PUBLIC_URL on startup
docker run -p 8000:8000 --env-file .env -e PUBLIC_URL=https://your-host.example.com gptlive1.1-native-no-vad

# Outbound (listens on 8001, OUTBOUND_SERVER_PORT)
docker run -p 8001:8001 --env-file .env -e PUBLIC_URL=https://your-host.example.com gptlive1.1-native-no-vad \
  uv run python -m outbound.server
```

The image exposes 8000 (inbound) and 8001 (outbound). It is based on `python:3.12-slim` by default; pass `--build-arg BASE_IMAGE=...` to change it. It runs `uv sync --locked --no-install-project --no-dev --extra streaming`, so the Redis sink is available but the `observability` extra is not installed. The image does not include `cloudflared`, so `--tunnel` is for local runs only.


## Troubleshooting

### `--tunnel`: "cloudflared not found" or no "Ready!" line

- **"cloudflared not found":** [install `cloudflared`](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/downloads/) so it's on your `PATH`, then run again.
- **`Waiting for Plivo to accept …` for a while:** this is normal. Plivo rejects a new `trycloudflare.com` hostname (`Must be a valid url`) until it resolves, and the server keeps retrying for up to 3 minutes before `Ready!`. With `outbound.server --tunnel`, Plivo's Make Call API rejects the new `answer_url` the same way during that window; retry the call after a minute.
- **`Plivo did not accept … within 180s`:** restart with `--tunnel` to get a new URL, or use ngrok or a deployed host with a fixed `PUBLIC_URL`. Quick tunnels are meant for development and have no uptime guarantee.

### Calls rejected: 403 on `/answer` / `/outbound/answer`, or the call drops at once

The log shows `Rejected Plivo webhook POST /answer: signature mismatch …`. The URL rebuilt from `PUBLIC_URL` doesn't match the one Plivo signed. Check the following:

- `PUBLIC_URL` has the same scheme and host as the application's answer URL, or the `answer_url` you passed to Make Call. For example, don't mix `http://` and `https://`, and don't mix an old tunnel hostname with a new one.
- `PLIVO_AUTH_TOKEN` belongs to the account (or subaccount) that owns the number or call.
- Nothing between Plivo and the server rewrites the path or query string.

`missing X-Plivo-Signature-V3` means the request did not come from Plivo, for example a hand-made `curl`. To call the server by hand, sign the request with your `PLIVO_AUTH_TOKEN` the way Plivo does (see `plivo_signature_headers()` in `tests/helpers.py`).

`Rejected Plivo stream /ws: signature mismatch …` means `/answer` was accepted but the audio stream was refused, so the call connects and then drops. Check that the host in `PUBLIC_URL` is the host Plivo connects to for the stream, and that nothing in between rewrites the path. To open `/ws` by hand, send the headers from `stream_signature_headers()` in `tests/helpers.py`.

### 401 or `error` right after connecting

`GPT-Live error during session.start: …` in the log carries OpenAI's own code and message. A 401 on connect means `OPENAI_API_KEY` is missing or has no GPT-Live access. An unknown-model error names the field: check `OPENAI_LIVE_MODEL` and `OPENAI_DELEGATION_MODEL` against the models your organization can use.

### Silence on the call / no greeting

Check the log for `session.started`. If it is there but no `first audio chunk sent to Plivo` follows, the model did not act on the greeting message: confirm `AGENT_GREETING` (or the `greeting` query param) is not empty.

### The agent talks over the caller, or cuts itself off

Interruption behaviour is the model's, steered by the prompt's "Interruption Policy" and "Backchannel Policy". If buffered audio keeps playing after the model has stopped, lower `BARGE_IN_FLUSH_BACKLOG_MS`; if short acknowledgements cut answers off, raise it.

### Tool results never spoken

The log should show `calling <function>` followed (at `LOG_LEVEL=verbose`) by `returned N function result(s) + response.create`. If the first line is missing, the voice model did not delegate: make the trigger explicit in the prompt's "Delegation Policy".

### Call does not hang up after goodbye

`end_call` must be called by the backend model (`calling end_call` in the log). The hangup then waits for the goodbye to go quiet and for Plivo's `playedStream`; without Plivo credentials the REST hangup is skipped and only the stream is closed.

## Implementation Notes

- GPT-Live expects a continuous input stream paced at its sample rate. Plivo provides exactly that (one 20 ms frame every 20 ms), so frames are forwarded as they arrive, silence included.
- `audio.format` is a single object for both directions; there is no separate input/output format, and it cannot change mid-session.
- The session's `expires_at` is logged at `LOG_LEVEL=verbose`; when a session expires GPT-Live sends `session.closed` (`reason: expired`) and the call ends.
- `error` events answer one client event and do not end the session; they are logged and counted in `session_end.errors`.
- The housekeeping tick in `plivo_tx` is time-based rather than idle-based, because GPT-Live may stream audio without gaps.
