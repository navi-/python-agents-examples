"""Plivo docs generator — produces documentation for plivo.com/docs.

Generates three types of content:

1. **Guides** (per AI stack combination):
   Step-by-step tutorials that walk a developer through building a voice agent
   with a specific stack. One guide per example.
   Path: docs/guides/{example-name}.md

2. **Reference** (Component Type → Provider → Model Series):
   Hierarchical reference pages organized by component type (LLM, STT, TTS, S2S),
   then by provider, then by model series.

   - Provider pillar pages: overview, auth, model series comparison table
     Path: docs/reference/{type}/{provider}.md
   - Model series spoke pages: version-specific integration, benchmarks, migration
     Path: docs/reference/{type}/{provider}/{series-slug}.md

   Hub-spoke linking: series pages link back to their provider pillar.
   Provider pillars collect external link equity and distribute to spokes.

3. **Concepts** (architecture & decisions):
   Explains the universal patterns: Plivo WebSocket protocol, audio pipeline,
   turn detection strategies, orchestration approaches.
   Path: docs/concepts/{topic}.md

Why Component Type → Provider → Model Series:
- Component-type grouping matches info architecture (Realtime under S2S, Whisper under STT)
- Provider pillar pages target high-volume queries ("Plivo OpenAI LLM integration")
- Model series spokes target long-tail queries ("Plivo GPT-5.4 migration guide")
- Every model version gets its own page (benchmarks, pricing, examples differ)
- Hub-spoke linking concentrates link equity on pillars while spokes rank for specifics
- Thin content prevention via mandatory unique sections (>30% unique per page)
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import anthropic
from loguru import logger

from .planner import ExamplePlan
from .registry import (
    COMPONENT_TYPE_LABELS,
    get_reference_hierarchy,
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
# Reference: Provider pillar page (Component Type > Provider overview)
# =============================================================================


def _build_provider_pillar_prompt(
    component_type: str,
    provider_key: str,
    provider_data: dict,
    existing_examples: list[str],
) -> str:
    """Build prompt for a provider pillar page within a component type."""
    display_name = provider_data["display_name"]
    type_label = COMPONENT_TYPE_LABELS.get(component_type, component_type.upper())
    series_map = provider_data["series"]

    series_summaries = []
    for slug, sdata in series_map.items():
        model_names = ", ".join(c.name for _, c in sdata["models"])
        series_summaries.append(
            f"- **{sdata['series_name']}** (slug: `{slug}`): {model_names}"
        )

    model_details = []
    for slug, sdata in series_map.items():
        for key, comp in sdata["models"]:
            lines = [f"### {comp.name} (`{comp.model_id}`)"]
            lines.append(f"- Series: {sdata['series_name']} (`{slug}`)")
            lines.append(f"- API version: `{comp.api_version}` ({comp.api_style})")
            if hasattr(comp, "max_tokens_param"):
                lines.append(f"- Max tokens param: `{comp.max_tokens_param}`")
            if hasattr(comp, "input_sample_rate"):
                lines.append(
                    f"- Input: {comp.input_format} at {comp.input_sample_rate}Hz"
                )
            if hasattr(comp, "output_sample_rate"):
                lines.append(
                    f"- Output: {comp.output_format} at {comp.output_sample_rate}Hz"
                )
            lines.append(f"- Env vars: {', '.join(comp.env_vars)}")
            if comp.integration_notes:
                lines.append("- Integration notes:")
                for note in comp.integration_notes:
                    lines.append(f"  - {note}")
            model_details.append("\n".join(lines))

    seo = (
        f"{display_name} {type_label} integration for Plivo voice agents. "
        f"Setup, audio formats, model comparison, and code examples."
    )[:160]

    series_pages = "\n".join(
        f"- [{sdata['series_name']}]"
        f"(/docs/reference/{component_type}/{provider_key}/{slug})"
        for slug, sdata in series_map.items()
    )

    return textwrap.dedent(f"""\
        Write a PROVIDER PILLAR reference page for plivo.com/docs.

        Provider: {display_name}
        Component type: {type_label}
        This page lives at: `reference/{component_type}/{provider_key}`

        ## Frontmatter
        ```yaml
        ---
        title: "{display_name} {type_label} Integration for Plivo Voice Agents"
        description: "{seo}"
        slug: "reference/{component_type}/{provider_key}"
        sidebar_label: "{display_name}"
        tags: [voice-agent, reference, {component_type}, {provider_key}]
        ---
        ```

        ## Model series under this provider
        {chr(10).join(series_summaries)}

        ## All models
        {chr(10).join(model_details)}

        ## Page structure (PILLAR PAGE)

        This is a HUB page. It provides an overview and links to detailed series pages.

        1. **Provider overview** (2-3 paragraphs):
           What {display_name} offers for {type_label} in voice agents.
           Auth setup (API key env var, how to obtain).

        2. **Model series comparison table**:
           All series side-by-side. Columns vary by component type:
           - LLM: Series | Models | API Version | Max Tokens Param | Tools | Key Difference
           - STT: Series | Models | Protocol | Sample Rate | Resample Needed | Key Difference
           - TTS: Series | Models | Protocol | Output Rate | Output Format | Key Difference
           - S2S: Series | Models | Protocol | In/Out Rate | Key Difference

        3. **Common integration pattern**:
           Code shared across ALL series (auth, connection setup, imports).
           Show 15-20 lines of REAL code patterns.

        4. **Audio format** (if STT/TTS/S2S):
           Plivo 8kHz μ-law ↔ this provider's format.
           Conversion function names (plivo_to_X, X_to_plivo).

        5. **Model series pages** (links):
           {series_pages}

        6. **Related guides**: Link to example guides using this provider.

        ## Internal linking rules
        - Link to each model series spoke page
        - Link to concept pages: [Audio Pipeline](/docs/concepts/audio-pipeline)
        - Do NOT link to other providers' pages (keeps link equity in this cluster)
        - SEO: use "{display_name} {type_label} Plivo voice agent" in headings

        ## Existing examples in the repo
        {', '.join(existing_examples)}

        Output ONLY the page markdown (including frontmatter). No wrapping fences.
    """)


# =============================================================================
# Reference: Model series spoke page (Component Type > Provider > Series)
# =============================================================================


def _build_model_series_prompt(
    component_type: str,
    provider_key: str,
    series_slug: str,
    series_data: dict,
    provider_display: str,
    existing_examples: list[str],
) -> str:
    """Build prompt for a model series spoke page.

    Includes mandatory unique sections to prevent thin content.
    """
    type_label = COMPONENT_TYPE_LABELS.get(component_type, component_type.upper())
    series_name = series_data["series_name"]
    models = series_data["models"]

    model_details = []
    all_integration_notes = []
    for key, comp in models:
        lines = [f"### {comp.name} (`{comp.model_id}`)"]
        lines.append(f"- Registry key: `{key}`")
        lines.append(f"- API version: `{comp.api_version}` ({comp.api_style})")
        if hasattr(comp, "max_tokens_param"):
            lines.append(f"- Max tokens param: `{comp.max_tokens_param}`")
        if hasattr(comp, "streaming"):
            lines.append(f"- Streaming: {comp.streaming}")
        if hasattr(comp, "supports_tools"):
            lines.append(f"- Function calling: {comp.supports_tools}")
        if hasattr(comp, "input_sample_rate"):
            lines.append(f"- Input: {comp.input_format} at {comp.input_sample_rate}Hz")
            lines.append(f"- Resample from Plivo: {comp.needs_resample_from_plivo}")
        if hasattr(comp, "output_sample_rate"):
            out_fmt = getattr(comp, "output_format", "unknown")
            lines.append(f"- Output: {out_fmt} at {comp.output_sample_rate}Hz")
            if hasattr(comp, "needs_resample_to_plivo"):
                lines.append(f"- Resample to Plivo: {comp.needs_resample_to_plivo}")
        lines.append(f"- Env vars: {', '.join(comp.env_vars)}")
        lines.append(f"- API docs: {comp.doc_url}")
        model_details.append("\n".join(lines))
        all_integration_notes.extend(comp.integration_notes)

    notes_block = (
        "\n".join(f"- {n}" for n in all_integration_notes)
        if all_integration_notes else "None provided."
    )

    seo = (
        f"{series_name} — {provider_display} {type_label} on Plivo. "
        f"Integration, benchmarks, migration guide, and examples."
    )[:160]

    pillar_link = f"/docs/reference/{component_type}/{provider_key}"

    return textwrap.dedent(f"""\
        Write a MODEL SERIES SPOKE reference page for plivo.com/docs.

        Series: {series_name}
        Provider: {provider_display}
        Component type: {type_label}
        This page lives at: `reference/{component_type}/{provider_key}/{series_slug}`

        ## Frontmatter
        ```yaml
        ---
        title: "{series_name} — {provider_display} {type_label} on Plivo"
        description: "{seo}"
        slug: "reference/{component_type}/{provider_key}/{series_slug}"
        sidebar_label: "{series_name}"
        tags: [voice-agent, reference, {component_type}, {provider_key}, {series_slug}]
        ---
        ```

        ## Models in this series
        {chr(10).join(model_details)}

        ## Integration notes from registry
        {notes_block}

        ## MANDATORY page structure (spoke page — THIN CONTENT PREVENTION)

        Every section below is REQUIRED. These sections ensure >30% unique content
        that cannot be duplicated across other model series pages.

        1. **Breadcrumb / back link**:
           `[← {provider_display} {type_label} Overview]({pillar_link})`

        2. **Series overview** (2-3 sentences):
           What makes {series_name} different from other series by this provider.
           When to choose this series over alternatives.

        3. **Model variants table** (UNIQUE per series):
           All models in this series with specs.
           Columns: Model | model_id | Context Window | Max Output | Cost Tier | Speed

        4. **Breaking changes from previous series** (UNIQUE per series):
           What changed from the prior series. Be specific:
           - Parameter renames (e.g., `max_tokens` → `max_completion_tokens`)
           - Endpoint changes, auth changes, response format changes
           - If this is the first/only series, write "Initial release — no breaking changes."
           Use `> **Breaking change**: ...` callout blocks.

        5. **Integration code** (15-30 lines):
           Complete connection + streaming code for this specific series.
           Must use the EXACT model_id and parameters for this series.

        6. **Audio format** (if STT/TTS/S2S):
           Conversion details specific to this series.

        7. **Benchmarks & capabilities** (UNIQUE per series):
           What this series excels at vs others.
           Speed, accuracy, language support, tool calling quality.
           Include concrete numbers where available.

        8. **Pricing & rate limits** (UNIQUE per series):
           Pricing tier for this series.
           Tokens per minute, requests per minute limits.

        9. **Migration guide** (UNIQUE per series):
           Concrete code diff showing what lines change when upgrading
           TO this series from the previous one.
           Show before/after code blocks.
           If first series: "This is the baseline — see newer series for migration guides."

        10. **Known issues & workarounds** (UNIQUE per series):
            Version-specific edge cases, bugs, limitations.
            If none known: "No known issues at this time."

        11. **Example projects** (UNIQUE per series):
            Links to example directories in the repo that use this specific series.

        ## Internal linking rules
        - MUST link back to provider pillar: [{provider_display} {type_label}]({pillar_link})
        - Do NOT link to other providers' pages
        - Cross-link to concept pages where relevant
        - SEO: use "{series_name} {provider_display} Plivo voice agent" in headings

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
    - Reference: Component Type → Provider pillar → Model Series spoke
    - Concepts: architecture and decision pages
    """

    def __init__(self, repo_root: Path, model: str = "claude-sonnet-4-6"):
        self.repo_root = repo_root
        self.model = model
        self.client = anthropic.Anthropic()
        self.docs_dir = repo_root / ".automation" / "docs"

    # ----- Guides -----

    def generate_guide(self, plan: ExamplePlan) -> str:
        """Generate a guide for a specific example."""
        logger.info(f"Generating Plivo docs guide for {plan.dir_name}")

        example_dir = self.repo_root / plan.dir_name
        agent_py = self._read(example_dir / "inbound" / "agent.py")
        utils_py = self._read(example_dir / "utils.py")
        server_py = self._read(example_dir / "inbound" / "server.py")

        if not agent_py:
            logger.warning(f"No agent.py found for {plan.dir_name} — using placeholder")
            agent_py = "# Agent code not yet generated"

        prompt = _build_guide_prompt(plan, agent_py, utils_py, server_py)
        return self._call_claude(prompt, "guide")

    # ----- Reference: provider pillar -----

    def generate_provider_pillar(self, component_type: str, provider_key: str) -> str:
        """Generate a provider pillar page under a component type."""
        hierarchy = get_reference_hierarchy()
        type_data = hierarchy.get(component_type, {})
        provider_data = type_data.get(provider_key)
        if not provider_data:
            available = ", ".join(sorted(type_data.keys()))
            raise ValueError(
                f"Unknown provider '{provider_key}' for {component_type}. "
                f"Available: {available}"
            )

        logger.info(f"Generating pillar: reference/{component_type}/{provider_key}")

        existing = self._list_examples()
        prompt = _build_provider_pillar_prompt(
            component_type, provider_key, provider_data, existing
        )
        return self._call_claude(prompt, "reference")

    # ----- Reference: model series spoke -----

    def generate_model_series_page(
        self, component_type: str, provider_key: str, series_slug: str,
    ) -> str:
        """Generate a model series spoke page."""
        hierarchy = get_reference_hierarchy()
        type_data = hierarchy.get(component_type, {})
        provider_data = type_data.get(provider_key)
        if not provider_data:
            raise ValueError(f"Unknown provider '{provider_key}' for {component_type}")

        series_data = provider_data["series"].get(series_slug)
        if not series_data:
            available = ", ".join(sorted(provider_data["series"].keys()))
            raise ValueError(
                f"Unknown series '{series_slug}' for "
                f"{provider_key}/{component_type}. Available: {available}"
            )

        logger.info(
            f"Generating series: "
            f"reference/{component_type}/{provider_key}/{series_slug}"
        )

        existing = self._list_examples()
        prompt = _build_model_series_prompt(
            component_type, provider_key, series_slug,
            series_data, provider_data["display_name"], existing,
        )
        return self._call_claude(prompt, "reference")

    # ----- Reference: provider (pillar + all series) -----

    def generate_provider_references(
        self, component_type: str, provider_key: str,
    ) -> dict[str, str]:
        """Generate pillar + all series pages for one provider.

        Returns dict like:
            {"llm/openai": "...", "llm/openai/gpt-4.1": "...", ...}
        """
        hierarchy = get_reference_hierarchy()
        provider_data = hierarchy.get(component_type, {}).get(provider_key)
        if not provider_data:
            raise ValueError(
                f"Unknown provider '{provider_key}' for {component_type}"
            )

        results = {}

        pillar_key = f"{component_type}/{provider_key}"
        results[pillar_key] = self.generate_provider_pillar(
            component_type, provider_key
        )

        for series_slug in provider_data["series"]:
            series_key = f"{component_type}/{provider_key}/{series_slug}"
            results[series_key] = self.generate_model_series_page(
                component_type, provider_key, series_slug,
            )

        return results

    # ----- Reference: all -----

    def generate_all_references(self) -> dict[str, str]:
        """Generate ALL reference pages across all types/providers/series."""
        hierarchy = get_reference_hierarchy()
        results = {}

        for comp_type in sorted(hierarchy.keys()):
            for provider_key in sorted(hierarchy[comp_type].keys()):
                provider_results = self.generate_provider_references(
                    comp_type, provider_key,
                )
                results.update(provider_results)

        return results

    # ----- Concepts -----

    def generate_concept(self, topic_key: str) -> str:
        """Generate a concept page."""
        logger.info(f"Generating Plivo docs concept: {topic_key}")

        topic = CONCEPT_TOPICS.get(topic_key)
        if not topic:
            raise ValueError(
                f"Unknown concept topic: {topic_key}. "
                f"Available: {list(CONCEPT_TOPICS.keys())}"
            )

        prompt = _build_concept_prompt(topic_key, topic)
        return self._call_claude(prompt, "concept")

    def generate_all_concepts(self) -> dict[str, str]:
        """Generate all concept pages."""
        results = {}
        for topic_key in CONCEPT_TOPICS:
            results[topic_key] = self.generate_concept(topic_key)
        return results

    # ----- Write docs to disk -----

    def write_docs(
        self,
        guides: dict[str, str] | None = None,
        references: dict[str, str] | None = None,
        concepts: dict[str, str] | None = None,
    ) -> Path:
        """Write all generated docs to disk.

        references keys are paths like "llm/openai" or "llm/openai/gpt-4.1".
        """
        if guides:
            guides_dir = self.docs_dir / "guides"
            guides_dir.mkdir(parents=True, exist_ok=True)
            for name, content in guides.items():
                (guides_dir / f"{name}.md").write_text(content)
                logger.info(f"  Wrote guide: {name}.md")

        if references:
            ref_dir = self.docs_dir / "reference"
            for path_key, content in references.items():
                out_path = ref_dir / f"{path_key}.md"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(content)
                logger.info(f"  Wrote reference: {path_key}.md")

        if concepts:
            concepts_dir = self.docs_dir / "concepts"
            concepts_dir.mkdir(parents=True, exist_ok=True)
            for topic, content in concepts.items():
                (concepts_dir / f"{topic}.md").write_text(content)
                logger.info(f"  Wrote concept: {topic}.md")

        self._write_sidebar(guides or {}, references or {}, concepts or {})
        return self.docs_dir

    # ----- Sidebar: Mintlify nested navigation -----

    def _write_sidebar(
        self,
        guides: dict[str, str],
        references: dict[str, str],
        concepts: dict[str, str],
    ) -> None:
        """Generate Mintlify-compatible docs.json with nested groups."""
        hierarchy = get_reference_hierarchy()

        ref_categories = []
        for comp_type in ["llm", "stt", "tts", "s2s"]:
            if comp_type not in hierarchy:
                continue
            type_label = COMPONENT_TYPE_LABELS.get(comp_type, comp_type.upper())

            provider_groups = []
            for prov_key, prov_data in sorted(hierarchy[comp_type].items()):
                pillar_path = f"reference/{comp_type}/{prov_key}"
                if pillar_path not in references:
                    continue

                items = [pillar_path]
                for series_slug in sorted(prov_data["series"].keys()):
                    series_path = (
                        f"reference/{comp_type}/{prov_key}/{series_slug}"
                    )
                    if series_path in references:
                        items.append(series_path)

                provider_groups.append({
                    "group": prov_data["display_name"],
                    "pages": items,
                })

            if provider_groups:
                ref_categories.append({
                    "group": type_label,
                    "pages": provider_groups,
                })

        sidebar = {
            "navigation": [
                {
                    "group": "Concepts",
                    "pages": [f"concepts/{k}" for k in concepts],
                },
                {
                    "group": "Reference",
                    "pages": ref_categories,
                },
                {
                    "group": "Guides",
                    "pages": [f"guides/{k}" for k in guides],
                },
            ]
        }

        sidebar_path = self.docs_dir / "docs.json"
        sidebar_path.parent.mkdir(parents=True, exist_ok=True)
        sidebar_path.write_text(json.dumps(sidebar, indent=2) + "\n")
        logger.info("  Wrote docs.json (Mintlify navigation)")

    # ----- Helpers -----

    def _list_examples(self) -> list[str]:
        """List existing example directories in the repo."""
        return [
            d.name for d in self.repo_root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]

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
