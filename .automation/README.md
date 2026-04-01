# Voice Agent Example Generator

Agentic automation that generates production-ready voice agent examples when new AI models release. Uses a **Planner → Generator → Evaluator → Fixer** loop powered by Claude API.

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
                                        ┌─────────────┐
                                        │ Git Workflow │
                                        │ branch/push  │
                                        └─────────────┘
```

### Components

1. **Trigger Detection** (`triggers.py`): Compares `known-models.yaml` manifest against the component registry. New models not yet in the registry trigger generation.

2. **Planner** (`planner.py`): Computes all valid `{LLM × STT × TTS × Orchestration}` combinations for the triggered component, skipping already-existing example directories.

3. **Generator** (`generator.py`): Uses Claude API to produce each file for an example, using existing examples as references. Generates files in dependency order (utils.py → agent.py → server.py → tests → config).

4. **Evaluator** (`evaluator.py`): Multi-stage validation:
   - Structure: all 24 canonical files exist
   - Code quality: `__future__` annotations, loguru, no hardcoded keys
   - Config placement: server config in server.py, audio config in utils.py
   - Audio pipeline: PLIVO_CHUNK_SIZE=160, μ-law content types
   - Lint: ruff passes
   - Unit tests: offline tests pass
   - Self-review: Claude reviews generated code for logical correctness
   - Validation script: `scripts/validate-example.sh` passes

5. **Fixer** (`fixer.py`): Takes evaluation failures, infers which files are affected, and uses Claude to produce corrected versions. Re-evaluated after fixing.

6. **Orchestrator** (`orchestrator.py`): Runs the full loop up to 3 fix iterations per example.

## Quick Start

```bash
cd .automation
uv sync

# Set your Anthropic API key
export ANTHROPIC_API_KEY=sk-ant-...

# List all registered components
uv run generate-examples list-components

# Plan what would be generated for a new LLM (dry run)
uv run generate-examples trigger --type llm --key gpt5.4mini --dry-run

# Generate all STT×TTS combinations for a new LLM
uv run generate-examples trigger --type llm --key gpt5.4mini

# Generate all LLM×TTS combinations for a new STT (e.g., Deepgram Nova 4)
# First add the new STT to registry.py, then:
uv run generate-examples trigger --type stt --key deepgram-nova4

# Generate a single voice-native example
uv run generate-examples trigger --type voice_native --key grok-voice

# Detect new models from a provider
uv run generate-examples detect --provider openai

# Detect and auto-generate
uv run generate-examples detect --provider openai --auto-generate

# Limit number of examples (useful for testing)
uv run generate-examples trigger --type llm --key gpt5.4mini --max-examples 2
```

## Adding a New Provider/Model

### Step 1: Add to Registry

Edit `voice_agent_generator/registry.py` and add the new component:

```python
# In LLMS dict:
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

## Combination Math

When triggered by a new component, the planner generates cross-products:

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
└── voice_agent_generator/
    ├── __init__.py
    ├── cli.py                        # Click CLI entry point
    ├── registry.py                   # Component definitions (LLMs, STTs, TTSs)
    ├── planner.py                    # Combination generator
    ├── generator.py                  # Claude-powered code generator
    ├── evaluator.py                  # Multi-stage validator
    ├── fixer.py                      # Auto-fix from eval feedback
    ├── orchestrator.py               # Full pipeline loop
    └── triggers.py                   # New model detection
```

The `.automation/` directory is a dotfile — invisible by default, clearly separate from the voice agent examples in the repo root.
