"""Plivo docs generator — produces documentation for plivo.com/docs.

Generates three types of content:

1. **Guides** (per AI stack combination):
   Step-by-step tutorials that walk a developer through building a voice agent
   with a specific stack. One guide per example.
   Path: docs/guides/{example-name}.md

2. **Reference** (per component type):
   Comprehensive reference pages organized by role (STT, LLM, TTS, VAD).
   Each page covers ALL providers for that role. When a new model releases,
   you update one page — not create a new one.
   Path: docs/reference/{component-type}.md

3. **Concepts** (architecture & decisions):
   Explains the universal patterns: Plivo WebSocket protocol, audio pipeline,
   turn detection strategies, orchestration approaches.
   Path: docs/concepts/{topic}.md

Why this structure (not per-model-series):
- Users search by task ("add Deepgram STT to Plivo") not by model version
- Per-model docs create 90% duplication (GPT-4.1 vs GPT-5.4 are near-identical)
- LLM-SEO: "Plivo voice agent Deepgram STT" beats "Plivo GPT-4.1-mini voice agent"
- New model releases update a reference page, not fragment the docs
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import anthropic
from loguru import logger

from .planner import ExamplePlan
from .registry import (
    LLMS,
    ORCHESTRATIONS,
    STTS,
    TTSS,
    VOICE_NATIVE,
    LLMComponent,
    STTComponent,
    TTSComponent,
    VoiceNativeComponent,
)


# =============================================================================
# Concepts — universal architecture pages
# =============================================================================

CONCEPT_TOPICS = {
    "voice-agent-architecture": {
        "title": "Voice Agent Architecture on Plivo",
        "seo_description": "Learn how to build real-time voice agents on Plivo using WebSocket streaming, AI providers for STT/LLM/TTS, and client-side VAD for turn detection.",
        "sections": [
            "Overview — what is a Plivo voice agent",
            "Plivo WebSocket protocol (start/media/stop events, playAudio/clearAudio)",
            "Audio format: μ-law 8kHz, 160-byte chunks (20ms), base64 encoding",
            "Answer webhook XML with <Stream> element",
            "Pipeline patterns: S2S (voice-native) vs STT→LLM→TTS (pipeline)",
            "Orchestration approaches: native asyncio vs Pipecat vs LiveKit",
            "Inbound vs outbound call flows",
        ],
    },
    "audio-pipeline": {
        "title": "Audio Pipeline & Format Conversion",
        "seo_description": "Understand audio format conversion between Plivo telephony (μ-law 8kHz) and AI services (PCM 16/24kHz). G.711 codec, resampling, and chunking explained.",
        "sections": [
            "Plivo audio format: G.711 μ-law, 8kHz, mono, 160 bytes per frame",
            "G.711 μ-law encode/decode (lookup table approach vs audioop)",
            "Resampling with scipy.signal.resample (8kHz ↔ 16kHz ↔ 24kHz)",
            "Common conversion patterns: plivo_to_{api}, {api}_to_plivo",
            "Frame size and chunking: why 160 bytes matters",
            "Audio quality: telephony band (300-3400Hz) limits",
        ],
    },
    "vad-turn-detection": {
        "title": "VAD & Turn Detection for Voice Agents",
        "seo_description": "Configure Silero VAD for voice agent turn detection on Plivo. Threshold tuning, echo rejection, barge-in handling, and turn state machines.",
        "sections": [
            "Why client-side VAD (Silero) vs server-side VAD",
            "Silero VAD: ONNX model, 512-sample frames at 16kHz (32ms)",
            "Threshold tuning: start_threshold, end_threshold, min_silence_ms",
            "Echo rejection: agent playback registers 0.5-0.74, real speech 0.93+",
            "Turn state machine: IDLE → SPEAKING → CHECK_TRANSCRIPT → COMMIT",
            "Barge-in: cancel tasks, drain queue, send clearAudio to Plivo",
            "Framework VAD: vad_enabled=True in Pipecat/LiveKit",
        ],
    },
    "choosing-your-stack": {
        "title": "Choosing Your AI Stack for Plivo Voice Agents",
        "seo_description": "Compare STT, LLM, and TTS providers for building voice agents on Plivo. Deepgram vs Sarvam, OpenAI vs Anthropic vs Google, ElevenLabs vs Cartesia.",
        "sections": [
            "Decision framework: latency vs cost vs quality vs language support",
            "S2S APIs (Grok Voice, GPT Realtime, Gemini Live) — when to use them",
            "STT providers: Deepgram (fast, English), Sarvam (Indian languages), Whisper (accuracy)",
            "LLM providers: OpenAI (tools), Anthropic (reasoning), Google (multimodal), xAI (speed)",
            "TTS providers: ElevenLabs (quality), Cartesia (speed), Grok TTS (no-resample)",
            "Orchestration: native (control) vs Pipecat (speed) vs LiveKit (scale)",
            "Audio format compatibility matrix",
        ],
    },
}


# =============================================================================
# Guide generator — one guide per example
# =============================================================================


def _build_guide_prompt(plan: ExamplePlan, agent_py: str, utils_py: str, server_py: str) -> str:
    """Build the prompt for generating a Plivo docs guide."""
    if plan.is_voice_native:
        vn = plan.voice_native
        stack_desc = f"the {vn.name} speech-to-speech API ({vn.model_id})"
    else:
        stack_desc = (
            f"{plan.stt.name} for speech-to-text, "
            f"{plan.llm.name} ({plan.llm.model_id}) for the LLM, "
            f"and {plan.tts.name} for text-to-speech"
        )

    return textwrap.dedent(f"""\
        Write a step-by-step guide for plivo.com/docs on building a real-time voice
        agent using {stack_desc} with Plivo telephony.

        Example directory: `{plan.dir_name}`
        Orchestration: {plan.orchestration.name}

        ## Target audience
        Developers who have a Plivo account and want to add AI voice capabilities.
        They may not know audio processing or WebSocket protocols deeply.

        ## Guide structure

        ### Frontmatter (YAML)
        ```yaml
        ---
        title: "Build a Voice Agent with {stack_desc}"
        description: "<SEO description, 150-160 chars>"
        slug: "{plan.dir_name}"
        sidebar_label: "{plan.dir_name}"
        tags: [voice-agent, {', '.join(_get_tags(plan))}]
        ---
        ```

        ### Body sections (use H2 for each):

        1. **What you'll build**: 2-3 sentences + architecture bullet list.
           Name every component, its role, and the protocol used.

        2. **Prerequisites**: Account requirements, API keys needed, tools to install.

        3. **Set up the project**: Create directory, install deps with uv, create .env.
           Include the full .env.example content inline.

        4. **Understand the audio pipeline**: Explain Plivo's μ-law 8kHz format,
           the conversions needed for each AI service. Include the hop-by-hop table.
           This is what makes developers click — they need to understand the audio path.

        5. **Build the voice agent** (the core section):
           Walk through building agent.py step by step:
           a. Connect to AI service(s)
           b. Set up the three async tasks (native) or pipeline (framework)
           c. Handle Plivo audio → forward to STT
           d. Process transcripts → send to LLM
           e. Receive TTS audio → send back to Plivo
           f. Implement VAD and barge-in

           Include REAL code snippets from the example (not pseudocode).
           Each snippet should be 10-30 lines showing one concept.

        6. **Set up the server**: The FastAPI endpoints (/answer, /ws, /hangup).
           Show the answer webhook XML with <Stream>.

        7. **Run and test locally**:
           - Start ngrok
           - Run the server
           - Make a test call
           - What to expect (greeting, conversation flow)

        8. **Add function calling** (if LLM supports tools):
           Show how to define a tool, add the schema, handle the call.
           One concrete example (e.g., check_order_status).

        9. **Deploy to production**:
           Dockerfile, docker build/run, what env vars to set.
           Production considerations (HTTPS, monitoring, error handling).

        10. **Next steps**: Link to other guides with different stacks.
            Link to reference pages for each component.

        ## Important rules
        - Use code blocks with language hints (```python, ```bash, ```xml)
        - Every code snippet must be REAL code from the example, not pseudo-code
        - Explain WHY, not just HOW (why 160-byte chunks? why 0.85 VAD threshold?)
        - Include the SEO-friendly frontmatter
        - Cross-link to concept pages: [Audio Pipeline](/docs/concepts/audio-pipeline),
          [VAD & Turn Detection](/docs/concepts/vad-turn-detection)
        - Do NOT include the full agent.py — show key methods with explanations
        - Use callout blocks for important notes: `> **Note**: ...`

        ## Source code for reference
        Only use code from these files (do not invent code):

        <agent-py>
        {agent_py[:6000]}
        </agent-py>

        <utils-py>
        {utils_py[:3000]}
        </utils-py>

        <server-py>
        {server_py[:3000]}
        </server-py>

        Output ONLY the guide markdown (including frontmatter). No wrapping fences.
    """)


def _get_tags(plan: ExamplePlan) -> list[str]:
    """Generate SEO tags for a guide."""
    tags = ["plivo"]
    if plan.is_voice_native:
        tags.extend([plan.voice_native.provider, "speech-to-speech"])
    else:
        tags.extend([plan.llm.provider, plan.stt.provider, plan.tts.provider])
        tags.append("stt")
        tags.append("tts")
    tags.append(plan.orchestration.name)
    if plan.orchestration.needs_vad_in_utils:
        tags.append("silero-vad")
    return list(dict.fromkeys(tags))  # deduplicate


# =============================================================================
# Reference page generator — one page per component type
# =============================================================================


def _build_reference_prompt(
    component_type: str,
    components: dict,
    existing_examples: list[str],
) -> str:
    """Build the prompt for a reference documentation page."""

    if component_type == "stt":
        title = "Speech-to-Text (STT) Providers"
        seo = "Compare STT providers for Plivo voice agents: Deepgram, Sarvam, Whisper, Google Cloud. Configuration, audio formats, and integration code."
        intro = "Every voice agent needs to convert caller speech to text. This page covers all supported STT providers, their configuration, and how to integrate each one."
    elif component_type == "llm":
        title = "LLM Providers"
        seo = "Use OpenAI, Anthropic Claude, Google Gemini, or xAI Grok as the LLM in your Plivo voice agent. Configuration, function calling, and streaming setup."
        intro = "The LLM processes transcribed speech and generates responses. This page covers all supported LLM providers and how to configure each one."
    elif component_type == "tts":
        title = "Text-to-Speech (TTS) Providers"
        seo = "Compare TTS providers for Plivo voice agents: ElevenLabs, Cartesia, OpenAI TTS, Grok TTS. Audio formats, streaming protocols, and voice selection."
        intro = "TTS converts the LLM's text response into audio that plays to the caller. This page covers all supported TTS providers and their audio format details."
    elif component_type == "voice_native":
        title = "Speech-to-Speech (S2S) APIs"
        seo = "Use Grok Voice, GPT Realtime, or Gemini Live for end-to-end speech-to-speech voice agents on Plivo. No separate STT/TTS needed."
        intro = "S2S APIs handle speech recognition, language model reasoning, and speech synthesis in a single API call. This page covers all supported S2S providers."
    else:
        title = component_type.title()
        seo = ""
        intro = ""

    # Build component details
    component_details = []
    for key, comp in components.items():
        if isinstance(comp, STTComponent):
            component_details.append(textwrap.dedent(f"""\
                ### {comp.name}
                - Key: `{key}`
                - Provider: {comp.provider}
                - Protocol: {comp.api_style}
                - Input format: {comp.input_format} at {comp.input_sample_rate}Hz
                - Needs resample from Plivo 8kHz: {comp.needs_resample_from_plivo}
                - Model: `{comp.model_id}`
                - Env var: {', '.join(comp.env_vars)}
                - API docs: {comp.doc_url}
            """))
        elif isinstance(comp, LLMComponent):
            component_details.append(textwrap.dedent(f"""\
                ### {comp.name}
                - Key: `{key}`
                - Provider: {comp.provider}
                - Protocol: {comp.api_style}
                - Streaming: {comp.streaming}
                - Function calling: {comp.supports_tools}
                - Model: `{comp.model_id}`
                - Max tokens default: {comp.max_tokens_default}
                - Env var: {', '.join(comp.env_vars)}
                - API docs: {comp.doc_url}
            """))
        elif isinstance(comp, TTSComponent):
            component_details.append(textwrap.dedent(f"""\
                ### {comp.name}
                - Key: `{key}`
                - Provider: {comp.provider}
                - Protocol: {comp.api_style}
                - Output format: {comp.output_format} at {comp.output_sample_rate}Hz
                - Needs resample to Plivo 8kHz: {comp.needs_resample_to_plivo}
                - Default voice: `{comp.voice_id_default}`
                - Env var: {', '.join(comp.env_vars)}
                - API docs: {comp.doc_url}
                {f'- Notes: {comp.notes}' if comp.notes else ''}
            """))
        elif isinstance(comp, VoiceNativeComponent):
            component_details.append(textwrap.dedent(f"""\
                ### {comp.name}
                - Key: `{key}`
                - Provider: {comp.provider}
                - Protocol: {comp.api_style}
                - Input: {comp.input_format} at {comp.input_sample_rate}Hz
                - Output: {comp.output_format} at {comp.output_sample_rate}Hz
                - Model: `{comp.model_id}`
                - Env var: {', '.join(comp.env_vars)}
                - API docs: {comp.doc_url}
            """))

    return textwrap.dedent(f"""\
        Write a reference documentation page for plivo.com/docs about {title}.

        ## Frontmatter
        ```yaml
        ---
        title: "{title}"
        description: "{seo}"
        slug: "reference/{component_type}-providers"
        sidebar_label: "{title}"
        tags: [voice-agent, reference, {component_type}]
        ---
        ```

        ## Introduction
        {intro}

        ## Components to document
        {chr(10).join(component_details)}

        ## Page structure

        1. **Introduction**: What role this component plays in the voice agent pipeline.
           Why choosing the right provider matters (latency, cost, language support).

        2. **Quick comparison table**: All providers side-by-side with key specs.
           Columns: Provider | Protocol | Audio Format | Sample Rate | Key Feature

        3. **Per-provider sections** (H2 each):
           For each provider:
           a. Overview (1-2 sentences, what makes it unique)
           b. Configuration (env vars, model selection)
           c. Audio format details (input/output format, sample rates, resample needed?)
           d. Integration code snippet (the key connection/streaming code, 15-25 lines)
           e. Example projects that use this provider (link to guides)

        4. **Audio format compatibility matrix**:
           Table showing which providers work at which sample rates and what
           conversion is needed for Plivo's 8kHz μ-law.

        5. **Switching providers**:
           What files to change when swapping one provider for another.
           (utils.py for audio conversion, agent.py for API client, .env for keys)

        ## Rules
        - Code snippets should show REAL patterns from the example codebase
        - Include the conversion function names (plivo_to_deepgram, elevenlabs_to_plivo, etc.)
        - Cross-link to guides: [Build with {{STT}} + {{LLM}} + {{TTS}}](/docs/guides/{{example-name}})
        - Cross-link to concepts: [Audio Pipeline](/docs/concepts/audio-pipeline)
        - SEO: use the provider names and "Plivo voice agent" naturally in headings

        ## Existing examples in the repo
        {', '.join(existing_examples)}

        Output ONLY the page markdown (including frontmatter). No wrapping fences.
    """)


# =============================================================================
# Concept page generator
# =============================================================================


def _build_concept_prompt(topic_key: str, topic: dict) -> str:
    """Build the prompt for a concept documentation page."""
    return textwrap.dedent(f"""\
        Write a concept documentation page for plivo.com/docs.

        ## Frontmatter
        ```yaml
        ---
        title: "{topic['title']}"
        description: "{topic['seo_description']}"
        slug: "concepts/{topic_key}"
        sidebar_label: "{topic['title']}"
        tags: [voice-agent, concepts, {topic_key}]
        ---
        ```

        ## Sections to cover
        {chr(10).join(f'- {s}' for s in topic['sections'])}

        ## Guidelines
        - This is a CONCEPT page, not a tutorial. Explain the WHY and WHAT, not step-by-step HOW.
        - Use diagrams (ASCII) where they clarify data flow or state machines.
        - Include code snippets only to illustrate concepts (not full implementations).
        - Cross-link to guides and reference pages throughout.
        - Write for developers who understand HTTP/WebSockets but may not know telephony or audio processing.
        - SEO: use "Plivo voice agent" and component names naturally.
        - Each section should be 3-5 paragraphs or include a table/diagram.
        - End with a "Next steps" section linking to guides.

        Output ONLY the page markdown (including frontmatter). No wrapping fences.
    """)


# =============================================================================
# Main docs generator class
# =============================================================================


class PlivoDocsGenerator:
    """Generates documentation for plivo.com/docs.

    Produces three content types:
    - Guides: per-stack tutorials (one per example)
    - Reference: per-component-type pages (STT, LLM, TTS, Voice-Native)
    - Concepts: architecture and decision pages
    """

    def __init__(self, repo_root: Path, model: str = "claude-sonnet-4-6"):
        self.repo_root = repo_root
        self.model = model
        self.client = anthropic.Anthropic()
        self.docs_dir = repo_root / ".automation" / "docs"

    def generate_guide(self, plan: ExamplePlan) -> str:
        """Generate a guide for a specific example."""
        logger.info(f"Generating Plivo docs guide for {plan.dir_name}")

        # Read source files for code snippets
        example_dir = self.repo_root / plan.dir_name
        agent_py = self._read(example_dir / "inbound" / "agent.py")
        utils_py = self._read(example_dir / "utils.py")
        server_py = self._read(example_dir / "inbound" / "server.py")

        if not agent_py:
            logger.warning(f"No agent.py found for {plan.dir_name} — using placeholder")
            agent_py = "# Agent code not yet generated"

        prompt = _build_guide_prompt(plan, agent_py, utils_py, server_py)
        return self._call_claude(prompt, "guide")

    def generate_reference(self, component_type: str) -> str:
        """Generate a reference page for a component type."""
        logger.info(f"Generating Plivo docs reference for {component_type}")

        component_map = {
            "stt": STTS,
            "llm": LLMS,
            "tts": TTSS,
            "voice_native": VOICE_NATIVE,
        }

        components = component_map.get(component_type, {})
        if not components:
            raise ValueError(f"Unknown component type: {component_type}")

        existing = [d.name for d in self.repo_root.iterdir()
                    if d.is_dir() and not d.name.startswith(".")]

        prompt = _build_reference_prompt(component_type, components, existing)
        return self._call_claude(prompt, "reference")

    def generate_concept(self, topic_key: str) -> str:
        """Generate a concept page."""
        logger.info(f"Generating Plivo docs concept: {topic_key}")

        topic = CONCEPT_TOPICS.get(topic_key)
        if not topic:
            raise ValueError(f"Unknown concept topic: {topic_key}. "
                             f"Available: {list(CONCEPT_TOPICS.keys())}")

        prompt = _build_concept_prompt(topic_key, topic)
        return self._call_claude(prompt, "concept")

    def generate_all_references(self) -> dict[str, str]:
        """Generate all reference pages."""
        results = {}
        for comp_type in ["stt", "llm", "tts", "voice_native"]:
            results[comp_type] = self.generate_reference(comp_type)
        return results

    def generate_all_concepts(self) -> dict[str, str]:
        """Generate all concept pages."""
        results = {}
        for topic_key in CONCEPT_TOPICS:
            results[topic_key] = self.generate_concept(topic_key)
        return results

    def write_docs(
        self,
        guides: dict[str, str] | None = None,
        references: dict[str, str] | None = None,
        concepts: dict[str, str] | None = None,
    ) -> Path:
        """Write all generated docs to disk."""
        if guides:
            guides_dir = self.docs_dir / "guides"
            guides_dir.mkdir(parents=True, exist_ok=True)
            for name, content in guides.items():
                (guides_dir / f"{name}.md").write_text(content)
                logger.info(f"  Wrote guide: {name}.md")

        if references:
            ref_dir = self.docs_dir / "reference"
            ref_dir.mkdir(parents=True, exist_ok=True)
            for comp_type, content in references.items():
                (ref_dir / f"{comp_type}-providers.md").write_text(content)
                logger.info(f"  Wrote reference: {comp_type}-providers.md")

        if concepts:
            concepts_dir = self.docs_dir / "concepts"
            concepts_dir.mkdir(parents=True, exist_ok=True)
            for topic, content in concepts.items():
                (concepts_dir / f"{topic}.md").write_text(content)
                logger.info(f"  Wrote concept: {topic}.md")

        # Generate sidebar/navigation config
        self._write_sidebar(guides or {}, references or {}, concepts or {})

        return self.docs_dir

    def _write_sidebar(
        self,
        guides: dict[str, str],
        references: dict[str, str],
        concepts: dict[str, str],
    ) -> None:
        """Generate a sidebar navigation file for the docs site."""
        sidebar = {
            "docs": [
                {
                    "type": "category",
                    "label": "Concepts",
                    "items": [f"concepts/{k}" for k in concepts],
                },
                {
                    "type": "category",
                    "label": "Reference",
                    "items": [f"reference/{k}-providers" for k in references],
                },
                {
                    "type": "category",
                    "label": "Guides",
                    "items": [f"guides/{k}" for k in guides],
                },
            ]
        }

        import json
        sidebar_path = self.docs_dir / "sidebar.json"
        sidebar_path.write_text(json.dumps(sidebar, indent=2) + "\n")
        logger.info("  Wrote sidebar.json")

    def _read(self, path: Path) -> str:
        """Read a file, return empty string if not found."""
        try:
            return path.read_text()
        except FileNotFoundError:
            return ""

    def _call_claude(self, prompt: str, doc_type: str) -> str:
        """Call Claude API to generate documentation."""
        system = textwrap.dedent(f"""\
            You are a technical documentation writer for Plivo's developer docs
            (plivo.com/docs). You write {doc_type} documentation that is:

            - Technically precise: sample rates, protocols, formats are correct
            - Developer-friendly: code snippets are real, copy-pasteable, and tested
            - SEO-optimized: headings include searchable terms naturally
            - Scannable: developers find what they need quickly
            - Cross-linked: references to related guides, concepts, and API docs

            Your documentation targets developers building real-time voice agents
            on Plivo's telephony platform. They have API keys and want to ship.

            Include YAML frontmatter with title, description (≤160 chars for SEO),
            slug, sidebar_label, and tags.
        """)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=16000,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0].text

            # Strip wrapping fences
            lines = content.strip().split("\n")
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                return "\n".join(lines) + "\n"

            return content

        except Exception as e:
            logger.error(f"Docs generation failed: {e}")
            return f"---\ntitle: Generation failed\n---\n\nError: {e}\n"
