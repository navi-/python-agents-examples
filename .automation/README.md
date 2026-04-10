# Voice Agent Example Generator

Agentic automation that generates production-ready voice agent examples when new AI models release. Uses a **Planner → Generator → Evaluator → Fixer** loop powered by Claude API. Also generates READMEs and Plivo.com/docs documentation.

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Trigger    │────▶│   Planner   │────▶│  Generator  │────▶│  Evaluator  │
│  Detection   │     │             │     │  (Claude)   │     │             │
└─────────────┘     └─────────────┘     └──────┬──────┘     └──────┬──────┘
                                               │                    │
                                               │    ┌──────────┐   │
                                               │◀───│  Fixer   │◀──┘
                                               │    │ (Claude) │  (if failures)
                                               │    └──────────┘
                                               ▼
                                    ┌──────────────────────┐
                                    │  README + Plivo Docs │
                                    │  Generation          │
                                    └──────────┬───────────┘
                                               ▼
                                        ┌─────────────┐
                                        │ Git Workflow │
                                        │ branch/push  │
                                        └─────────────┘
```

### Pipeline Components

1. **Trigger Detection** (`triggers.py`): Compares `known-models.yaml` manifest against the component registry. New models trigger generation.

2. **Planner** (`planner.py`): Computes all valid `{LLM × STT × TTS × Orchestration}` combinations, skipping existing directories.

3. **Generator** (`generator.py`): Claude API produces each code file using existing examples as references. Files generated in dependency order.

4. **README Generator** (`readme_gen.py`): Dedicated Claude-powered README generator. Produces first-principles structured READMEs with demo description (≤5 lines text only), architecture diagrams, audio pipeline tables, VAD tuning, and troubleshooting.

5. **Evaluator** (`evaluator.py`): Multi-stage validation:
   - Structure: all canonical files exist
   - Code quality: `__future__` annotations, loguru, no hardcoded keys
   - Config placement: server config in server.py, audio config in utils.py
   - Audio pipeline: PLIVO_CHUNK_SIZE=160, μ-law content types
   - README quality: demo length ≤5 lines, no diagrams in demo, ≥5 required sections
   - Lint: ruff passes
   - Unit tests: offline tests pass
   - Self-review: Claude checks for logical correctness
   - Validation script: `scripts/validate-example.sh`

6. **Fixer** (`fixer.py`): Takes evaluation failures and uses Claude to produce corrected files. Loop repeats up to 3 times.

7. **Plivo Docs Generator** (`plivo_docs.py`): Produces documentation for plivo.com/docs in three content types (see below).

8. **Orchestrator** (`orchestrator.py`): Runs the full pipeline. Each example gets code + README + Plivo docs guide.

## Documentation for plivo.com/docs

The docs generator produces three content types:

### Guides (`docs/guides/`)
Per-stack tutorials. One guide per example directory. Step-by-step: prerequisites → setup → build the agent → test → deploy.

```bash
# Generate a guide for a specific example
uv run generate-examples docs guide --example gpt4.1mini-deepgram-elevenlabs-native

# Generate guides for ALL existing examples
uv run generate-examples docs all
```

### Reference (`docs/reference/`)
**Component Type → Provider → Model Series hierarchy.** Each component type (LLM, STT, TTS, S2S) contains provider pillar pages, and each provider pillar links to model series spoke pages.

```
docs/reference/
├── llm/
│   ├── openai.md              ← Provider pillar (overview, comparison table, auth)
│   ├── openai/
│   │   ├── gpt-4.1.md         ← Model series (benchmarks, pricing, migration guide)
│   │   └── gpt-5.4.md         ← Model series (breaking changes, migration from 4.1)
│   ├── anthropic.md
│   ├── anthropic/
│   │   └── claude-4.md
│   ├── google.md
│   │   └── gemini-2.0.md
│   └── xai.md
│       └── grok-3.md
├── stt/
│   ├── deepgram.md → deepgram/nova-3.md
│   ├── sarvam.md → sarvam/saaras-v3.md
│   ├── openai.md → openai/whisper-1.md
│   └── google.md → google/cloud-stt-v2.md
├── tts/
│   ├── elevenlabs.md → elevenlabs/ws-streaming.md
│   ├── openai.md → openai/tts.md
│   ├── xai.md → xai/grok-tts.md
│   ├── cartesia.md → cartesia/sonic.md
│   └── google.md → google/cloud-tts-v1.md
└── s2s/
    ├── xai.md → xai/grok-voice.md
    ├── openai.md → openai/gpt-realtime.md
    └── google.md → google/gemini-live.md
```

**Provider pillar pages** contain: auth setup, common patterns, model series comparison table, links to each series spoke.

**Model series spoke pages** contain mandatory unique content to prevent thin content:
- Breaking changes from previous series
- Model variants table (specs, context window, pricing tier)
- Benchmarks & capabilities unique to this series
- Migration guide (concrete code diff from previous series)
- Known issues & workarounds
- Example projects using this series

**Hub-spoke linking**: Every series page links back to its provider pillar. Provider pillars collect external link equity and distribute to spokes.

```bash
# Generate a single provider's pillar + all series pages
uv run generate-examples docs reference --component-type llm --provider openai

# Generate all LLM reference pages
uv run generate-examples docs reference --component-type llm

# Generate ALL reference pages
uv run generate-examples docs reference

# Generate everything (guides + references + concepts)
uv run generate-examples docs all
```

### Concepts (`docs/concepts/`)
Architecture and decision pages. Universal patterns that don't change per model.

- `voice-agent-architecture.md` — Plivo WebSocket protocol, pipeline patterns
- `audio-pipeline.md` — μ-law/PCM conversion, resampling, chunking
- `vad-turn-detection.md` — Silero VAD config, echo rejection, barge-in
- `choosing-your-stack.md` — Provider comparison, trade-offs, compatibility

```bash
# Generate a specific concept page
uv run generate-examples docs concepts --topic audio-pipeline

# Generate all concept pages
uv run generate-examples docs all
```

### Why this structure

- **Component-type grouping** matches information architecture — Realtime API under S2S, Whisper under STT
- **Provider pillar pages** target high-volume queries ("Plivo OpenAI LLM integration")
- **Model series spoke pages** target long-tail queries ("Plivo GPT-5.4 migration guide")
- **Every model version gets its own page** — benchmarks, pricing, examples all differ even when API contract is the same
- **Hub-spoke linking** concentrates link equity on pillar pages while spokes rank for specific queries
- **Thin content prevention** — mandatory unique sections (breaking changes, migration guide, benchmarks) ensure >30% unique content per page

## Quick Start

```bash
cd .automation
uv sync

export ANTHROPIC_API_KEY=sk-ant-...

# List all registered components
uv run generate-examples list-components

# Plan (dry run)
uv run generate-examples trigger --type llm --key gpt5.4mini --dry-run

# Generate examples + READMEs + Plivo docs guides
uv run generate-examples trigger --type llm --key gpt5.4mini

# Generate ALL Plivo docs (guides + references + concepts)
uv run generate-examples docs all

# Detect new models and auto-generate
uv run generate-examples detect --provider openai --auto-generate
```

## Adding a New Provider/Model

### Step 1: Add to Registry

Edit `voice_agent_generator/registry.py`:

```python
"newmodel": LLMComponent(
    name="New Model",
    short_name="newmodel",
    provider="provider",
    api_style="http",
    env_vars=["PROVIDER_API_KEY"],
    dependencies=["httpx>=0.27.0"],
    model_id="new-model-v1",
    doc_url="https://docs.provider.com/api",
),
```

### Step 2: Update Manifest

Add to `known-models.yaml`:

```yaml
provider:
  llms:
    - key: newmodel
      model_id: new-model-v1
```

### Step 3: Generate

```bash
uv run generate-examples trigger --type llm --key newmodel
```

Then regenerate:

```bash
uv run generate-examples docs reference --component-type llm --provider provider
```

## Combination Math

| Trigger Type | Combinations Generated |
|---|---|
| New LLM | LLM × all_STTs × all_TTSs = 1 × 4 × 5 = 20 examples |
| New STT | all_LLMs × STT × all_TTSs = 7 × 1 × 5 = 35 examples |
| New TTS | all_LLMs × all_STTs × TTS = 7 × 4 × 1 = 28 examples |
| Voice Native | Single example (S2S is self-contained) |

Use `--max-examples N` to limit batch size.

## File Organization

```
.automation/                          # Hidden dir — not an example
├── pyproject.toml                    # Package definition
├── known-models.yaml                 # Model manifest for trigger detection
├── README.md                         # This file
├── docs/                             # Generated Plivo docs output
│   ├── sidebar.json                  # Navigation config
│   ├── concepts/                     # Architecture pages
│   ├── reference/                    # Per-type pages (llm/, stt/, tts/, s2s/)
│   └── guides/                       # Per-example tutorials
└── voice_agent_generator/
    ├── __init__.py
    ├── cli.py                        # Click CLI entry point
    ├── registry.py                   # Component definitions (LLMs, STTs, TTSs)
    ├── planner.py                    # Combination generator
    ├── generator.py                  # Claude-powered code generator
    ├── readme_gen.py                 # First-principles README generator
    ├── plivo_docs.py                 # Plivo.com/docs generator (guides + reference + concepts)
    ├── evaluator.py                  # Multi-stage validator (incl. README quality)
    ├── fixer.py                      # Auto-fix from eval feedback
    ├── orchestrator.py               # Full pipeline loop
    └── triggers.py                   # New model detection
```
