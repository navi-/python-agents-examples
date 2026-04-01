"""README generator — produces detailed READMEs from first principles.

README structure (designed from what a developer actually needs):

1. **Demo description** (H1 to first H2): ≤5 lines plain text/bullets.
   This is the ONLY constraint from the hosting app. No diagrams, no code blocks.
   Dense technical pipeline trace naming every component, protocol, format.

2. **What this does** (first H2): One paragraph — what problem does this solve,
   what's the user experience (call a number, talk to an AI agent).

3. **Architecture**: ASCII pipeline diagram showing the data flow.
   Component table with service, protocol, model, audio format.

4. **Audio pipeline**: Hop-by-hop table. Every format conversion named.
   This is the section developers spend the most time debugging.

5. **VAD & turn detection**: Only for native orchestration. Thresholds,
   echo rejection, barge-in steps. This is the #2 debugging topic.

6. **Get started**: Prerequisites → install → configure → run.
   No wall of text — numbered steps with code blocks.

7. **Configuration reference**: Full env var table. Every knob documented.

8. **Extending**: How to add tools/functions, change the system prompt,
   swap providers. The "what next" after getting it running.

9. **Testing**: All test commands with what each level tests.

10. **Deploying**: Docker one-liner + what env vars to set in production.

11. **Troubleshooting**: Top 3-4 failure modes with fixes.

12. **Project structure**: File tree at the END (reference, not orientation).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import anthropic
from loguru import logger

from .planner import ExamplePlan


def _describe_audio_hops(plan: ExamplePlan) -> str:
    """Describe the audio conversion hops for this example."""
    lines = []

    if plan.is_voice_native:
        vn = plan.voice_native
        lines.append(f"- Plivo (μ-law 8kHz) → Agent: decode μ-law, resample "
                      f"8kHz → {vn.input_sample_rate}Hz {vn.input_format}")
        lines.append("- Agent → Silero VAD: decode μ-law, resample 8kHz → 16kHz float32")
        lines.append(f"- {vn.name} → Agent: {vn.output_format} at {vn.output_sample_rate}Hz")
        lines.append(f"- Agent → Plivo: resample {vn.output_sample_rate}Hz → 8kHz, "
                      f"encode μ-law, chunk to 160 bytes")
    else:
        stt, tts = plan.stt, plan.tts
        lines.append("- Plivo (μ-law 8kHz) → Agent: decode μ-law to PCM16")
        if stt.needs_resample_from_plivo:
            lines.append(f"- Agent → {stt.name} STT: resample 8kHz → "
                          f"{stt.input_sample_rate}Hz {stt.input_format}")
        else:
            lines.append(f"- Agent → {stt.name} STT: {stt.input_format} at "
                          f"{stt.input_sample_rate}Hz (no resample needed)")
        if plan.orchestration.needs_vad_in_utils:
            lines.append("- Agent → Silero VAD: decode μ-law, resample 8kHz → 16kHz float32")
        lines.append(f"- {tts.name} TTS → Agent: {tts.output_format} at "
                      f"{tts.output_sample_rate}Hz")
        if tts.needs_resample_to_plivo:
            lines.append(f"- Agent → Plivo: resample {tts.output_sample_rate}Hz → 8kHz, "
                          f"encode μ-law, chunk to 160 bytes")
        else:
            if tts.output_format == "mulaw":
                lines.append("- Agent → Plivo: already μ-law 8kHz, chunk to 160 bytes")
            else:
                lines.append("- Agent → Plivo: encode to μ-law, chunk to 160 bytes")

    return "\n".join(lines)


def _build_component_block(plan: ExamplePlan) -> str:
    """Build structured component info for the prompt."""
    if plan.is_voice_native:
        vn = plan.voice_native
        return textwrap.dedent(f"""\
            Type: Voice-native (S2S) — single API for STT+LLM+TTS.
            Provider: {vn.provider} | Model: `{vn.model_id}` | Protocol: {vn.api_style}
            Input: {vn.input_format} {vn.input_sample_rate}Hz | Output: {vn.output_format} {vn.output_sample_rate}Hz
            Env vars: {', '.join(vn.env_vars)}
            {f'Notes: {vn.notes}' if vn.notes else ''}
            Orchestration: {plan.orchestration.name} | VAD: {'Silero (client-side)' if plan.orchestration.needs_vad_in_utils else 'framework-managed'}
        """)

    llm, stt, tts = plan.llm, plan.stt, plan.tts
    return textwrap.dedent(f"""\
        Type: Pipeline — separate STT → LLM → TTS.

        LLM: {llm.name} | Provider: {llm.provider} | Model: `{llm.model_id}`
          API: {llm.api_style} | Streaming: {llm.streaming} | Tools: {llm.supports_tools}
          Max tokens: {llm.max_tokens_default} | Env: {', '.join(llm.env_vars)}

        STT: {stt.name} | Provider: {stt.provider} | Protocol: {stt.api_style}
          Input: {stt.input_format} {stt.input_sample_rate}Hz
          Resample from Plivo 8kHz: {stt.needs_resample_from_plivo}
          {f'Model: `{stt.model_id}`' if stt.model_id else ''}
          Env: {', '.join(stt.env_vars)}

        TTS: {tts.name} | Provider: {tts.provider} | Protocol: {tts.api_style}
          Output: {tts.output_format} {tts.output_sample_rate}Hz
          Resample to Plivo 8kHz: {tts.needs_resample_to_plivo}
          {f'Voice: `{tts.voice_id_default}`' if tts.voice_id_default else ''}
          {f'Notes: {tts.notes}' if tts.notes else ''}
          Env: {', '.join(tts.env_vars)}

        Orchestration: {plan.orchestration.name}
        VAD: {'Silero (client-side, in utils.py)' if plan.orchestration.needs_vad_in_utils else 'framework-managed (vad_enabled=True)'}
    """)


def _build_readme_prompt(plan: ExamplePlan, reference_readme: str) -> str:
    """Build the prompt for generating a README from first principles."""
    component_block = _build_component_block(plan)
    audio_hops = _describe_audio_hops(plan)

    return textwrap.dedent(f"""\
        Generate a README.md for the voice agent example `{plan.dir_name}`.

        ## Components
        {component_block}

        ## Audio Pipeline Hops
        {audio_hops}

        ## Environment Variables
        {', '.join(plan.all_env_vars)}
        Plus: PUBLIC_URL, SERVER_PORT, DEFAULT_COUNTRY_CODE

        ## Dependencies
        {', '.join(plan.all_dependencies)}

        ## README Structure (follow this EXACTLY)

        ### 1. Title (H1)
        Format: `# {{LLM}} + {{STT}} + {{TTS}} — {{Orchestration}} Voice Agent`
        For S2S: `# {{API Name}} — {{Orchestration}} Voice Agent`

        ### 2. Demo description (between H1 and first H2)
        HARD CONSTRAINT: ≤5 lines of PLAIN TEXT and/or BULLET LISTS.
        NO tables, NO diagrams, NO code blocks, NO images.
        This text is extracted by the hosting app as the example's description.

        What to include in those ≤5 lines:
        - Orchestration approach (native asyncio / Pipecat pipeline / etc.)
        - Full pipeline trace: Plivo → STT → LLM → TTS → Plivo
        - Each component: service, model name, protocol (WS/HTTP), audio format, sample rate
        - VAD: engine, frame size, thresholds (if native)
        - Barge-in behavior: what gets cancelled, what Plivo event is sent
        - Audio conversions: which hops need resampling, which don't
        - No marketing adjectives. Every word is a technical fact.
        - No latency numbers (those go in detailed sections below).

        ### 3. What This Does (first H2)
        One short paragraph: a phone call comes in via Plivo, user speaks,
        AI responds in real time. Name the use case (customer service demo).

        ### 4. Architecture
        - ASCII pipeline diagram (data flow from phone to components and back)
        - Component summary table: Component | Service | Protocol | Model | Audio Format

        ### 5. Audio Pipeline
        - Hop-by-hop table: Hop | Format | Sample Rate | Frame Size | Notes
        - List the actual conversion function names from utils.py
          (e.g., `plivo_to_deepgram()`, `elevenlabs_to_plivo()`)

        ### 6. VAD & Turn Detection (native orchestration only)
        - VAD parameter table with values and WHY each value was chosen
        - Echo rejection: explain the probability ranges for echo vs speech
        - Turn state machine (ASCII)
        - Barge-in: numbered steps of what happens

        ### 7. Per-Service Sections
        One H2 per service (STT, LLM, TTS) with:
        - Endpoint URL pattern
        - Protocol details (WebSocket message format, HTTP request shape)
        - Model name and key config
        - Audio format details

        ### 8. Concurrent Tasks (native) or Pipeline Structure (framework)
        - Task table: Task Name | Method | Role
        - How tasks coordinate (queues, events, locks)

        ### 9. Get Started
        Prerequisites (bullet list) → numbered steps:
        1. cd into dir, 2. uv sync, 3. cp .env, 4. start ngrok, 5. run server
        Keep it tight. No paragraphs between steps.

        ### 10. Configuration
        Full env var table: Variable | Description | Default/Required

        ### 11. Extending the Agent
        - How to add a new tool/function (3-step: define fn, add schema, add handler)
        - How to change the system prompt
        - How to swap a provider (what files to touch)

        ### 12. Testing
        All test commands with one-line description of what each tests.
        Group by: unit (offline) → local integration → API integration → live calls.

        ### 13. Deployment
        Docker build + run commands. What env vars to set in production.

        ### 14. Troubleshooting
        Top 4 failure modes developers hit. Each: symptom → cause → fix.

        ### 15. Project Structure
        File tree with one-line description per file.

        ## Reference
        Use this existing README for tone and density calibration, but follow
        the structure above (not the reference's structure):

        <reference>
        {reference_readme[:8000]}
        </reference>

        Output ONLY the README markdown. No wrapping fences.
    """)


class ReadmeGenerator:
    """Generates high-quality READMEs for voice agent examples."""

    def __init__(self, repo_root: Path, model: str = "claude-sonnet-4-6"):
        self.repo_root = repo_root
        self.model = model
        self.client = anthropic.Anthropic()

        # Load a reference README for tone calibration
        ref_path = repo_root / "gpt4.1mini-sarvam-elevenlabs-native" / "README.md"
        self.reference_readme = ref_path.read_text() if ref_path.exists() else ""

    def generate(self, plan: ExamplePlan) -> str:
        """Generate a complete README for the given example plan."""
        logger.info(f"Generating README for {plan.dir_name}")

        prompt = _build_readme_prompt(plan, self.reference_readme)

        system = textwrap.dedent("""\
            You are a technical writer who produces developer documentation for
            voice agent examples. Your READMEs are:

            - Fact-dense: every sentence conveys technical information
            - Scannable: developers find what they need in <10 seconds
            - Correct: sample rates, formats, protocols are verified
            - Practical: code snippets are copy-pasteable and work

            You write for developers who will:
            1. Skim the demo description to decide if this stack fits their needs
            2. Follow Get Started to run it locally in <5 minutes
            3. Read Architecture to understand the data flow
            4. Reference Audio Pipeline when debugging format issues
            5. Read Extending when customizing for their use case

            Do not use filler phrases. Do not describe what the README contains.
            Just write the README.
        """)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=16000,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0].text

            # Strip any wrapping fences
            lines = content.strip().split("\n")
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                return "\n".join(lines) + "\n"

            return content

        except Exception as e:
            logger.error(f"README generation failed: {e}")
            return _fallback_readme(plan)


def _fallback_readme(plan: ExamplePlan) -> str:
    """Generate a minimal README if Claude API fails."""
    title_parts = []
    if plan.is_voice_native:
        title_parts.append(plan.voice_native.name)
    else:
        title_parts.extend([plan.llm.name, plan.stt.name, plan.tts.name])
    title = " + ".join(title_parts)

    return textwrap.dedent(f"""\
        # {title} — {plan.orchestration.name.title()} Voice Agent

        Voice agent using {', '.join(title_parts)} with Plivo telephony \
and {'Silero VAD' if plan.orchestration.needs_vad_in_utils else 'framework-managed VAD'}.

        ## What This Does

        A phone call comes in via Plivo, the caller speaks to an AI agent that \
uses {title} for real-time voice conversation.

        ## Get Started

        ```bash
        cd {plan.dir_name}
        uv sync
        cp .env.example .env
        # Edit .env with your API keys

        ngrok http 8000
        uv run python -m inbound.server
        ```

        ## Configuration

        See `.env.example` for all configuration options.

        ## Testing

        ```bash
        uv run pytest tests/test_integration.py -v -k "unit"
        ```

        ## Deployment

        ```bash
        docker build -t {plan.dir_name} .
        docker run -p 8000:8000 --env-file .env {plan.dir_name}
        ```
    """)
