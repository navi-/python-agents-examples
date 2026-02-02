# Contributing

Guidelines for contributing voice agent examples to this repository.

## Project Naming Convention

Each example project uses Plivo as the telephony platform. Project names follow this structure:

```
{llm}-{stt}-{tts}-{framework}
```

| Component | Description |
|-----------|-------------|
| `llm` | Large Language Model provider (e.g., `openai`, `gemini`, `claude`) |
| `stt` | Speech-to-Text provider (e.g., `deepgram`, `whisper`, `gemini`) |
| `tts` | Text-to-Speech provider (e.g., `cartesia`, `elevenlabs`, `gemini`) |
| `framework` | Orchestration framework, or `native` if none |

### Framework Values

| Value | Meaning |
|-------|---------|
| `pipecat` | Uses the Pipecat framework |
| `livekit` | Uses the LiveKit Agents framework |
| `native` | Direct API integration, no orchestration framework |

### Speech-to-Speech (S2S) Models

For S2S models that handle speech input and output natively (like Gemini Live, GPT-4o Realtime), use the product name followed by the framework:

```
{product-name}-{framework}
```

Examples: `gemini-live-native`, `gpt4o-realtime-native`, `gemini-live-pipecat`

### Examples

| Project Name | LLM | STT | TTS | Framework |
|--------------|-----|-----|-----|-----------|
| `openai-deepgram-cartesia-pipecat` | OpenAI | Deepgram | Cartesia | Pipecat |
| `openai-deepgram-elevenlabs-pipecat` | OpenAI | Deepgram | ElevenLabs | Pipecat |
| `gemini-live-native` | Gemini (S2S) | Gemini (S2S) | Gemini (S2S) | None |

## Project Structure

Each example project should be a self-contained directory at the repository root:

```
{project-name}/
├── README.md           # Setup and usage instructions
├── requirements.txt    # or pyproject.toml
├── .env.example        # Environment variable template
├── server.py           # Main entry point (or app.py)
└── ...
```

## Adding a New Example

1. Create a new directory following the naming convention
2. Include a README.md with:
   - Brief description of the voice agent
   - Prerequisites and dependencies
   - Setup and configuration steps
   - How to run the example
3. Use `.env.example` for environment variables (never commit actual credentials)
4. Test that the example works end-to-end with Plivo
