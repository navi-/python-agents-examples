"""Generator — uses Claude API to produce voice agent example code.

Reads reference examples from the repo, builds a comprehensive prompt with
component specs, and asks Claude to generate each file for a new example.
The generator operates file-by-file to keep context focused and quality high.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import anthropic
from loguru import logger

from .planner import ExamplePlan

# Files to generate in order (dependencies first)
GENERATION_ORDER = [
    "utils.py",
    "inbound/agent.py",
    "inbound/server.py",
    "inbound/system_prompt.md",
    "outbound/agent.py",
    "outbound/server.py",
    "outbound/system_prompt.md",
    "pyproject.toml",
    ".env.example",
    ".gitignore",
    ".pre-commit-config.yaml",
    "Dockerfile",
    "README.md",
    "tests/conftest.py",
    "tests/helpers.py",
    "tests/test_integration.py",
    "tests/test_e2e_live.py",
    "tests/test_live_call.py",
    "tests/test_multiturn_voice.py",
    "tests/test_outbound_call.py",
]

# Files that are nearly identical across examples and can be copied
COPY_FILES = [
    "inbound/__init__.py",
    "outbound/__init__.py",
    "tests/__init__.py",
]

INIT_PY_CONTENT = '"""Package init."""\n'

# Maximum retries for generation
MAX_RETRIES = 2


def _read_file(path: Path) -> str:
    """Read a file, return empty string if not found."""
    try:
        return path.read_text()
    except FileNotFoundError:
        return ""


def _read_reference_files(repo_root: Path, ref_name: str) -> dict[str, str]:
    """Read all files from a reference example."""
    ref_dir = repo_root / ref_name
    files = {}
    for rel_path in GENERATION_ORDER:
        content = _read_file(ref_dir / rel_path)
        if content:
            files[rel_path] = content
    return files


def _build_component_spec(plan: ExamplePlan) -> str:
    """Build a human-readable spec of the components for this example."""
    lines = ["## Component Specification\n"]

    if plan.is_voice_native:
        vn = plan.voice_native
        lines.append(f"**Type**: Voice-native (S2S) — single API handles STT+LLM+TTS")
        lines.append(f"**Provider**: {vn.provider}")
        lines.append(f"**Model**: `{vn.model_id}`")
        lines.append(f"**API style**: {vn.api_style}")
        lines.append(f"**Input**: {vn.input_format} at {vn.input_sample_rate}Hz")
        lines.append(f"**Output**: {vn.output_format} at {vn.output_sample_rate}Hz")
        lines.append(f"**Env vars**: {', '.join(vn.env_vars)}")
        if vn.notes:
            lines.append(f"**Notes**: {vn.notes}")
    else:
        llm = plan.llm
        stt = plan.stt
        tts = plan.tts
        lines.append("**Type**: Pipeline (separate STT → LLM → TTS)\n")

        lines.append(f"### LLM: {llm.name}")
        lines.append(f"- Provider: {llm.provider}")
        lines.append(f"- Model: `{llm.model_id}`")
        lines.append(f"- API style: {llm.api_style}")
        lines.append(f"- Streaming: {llm.streaming}")
        lines.append(f"- Supports tools: {llm.supports_tools}")
        lines.append(f"- Max tokens default: {llm.max_tokens_default}")
        lines.append(f"- Env vars: {', '.join(llm.env_vars)}")
        if llm.doc_url:
            lines.append(f"- API docs: {llm.doc_url}")
        if llm.notes:
            lines.append(f"- Notes: {llm.notes}")

        lines.append(f"\n### STT: {stt.name}")
        lines.append(f"- Provider: {stt.provider}")
        lines.append(f"- API style: {stt.api_style}")
        lines.append(f"- Input sample rate: {stt.input_sample_rate}Hz")
        lines.append(f"- Input format: {stt.input_format}")
        lines.append(f"- Needs resample from Plivo 8kHz: {stt.needs_resample_from_plivo}")
        lines.append(f"- Env vars: {', '.join(stt.env_vars)}")
        if stt.model_id:
            lines.append(f"- Model: `{stt.model_id}`")
        if stt.doc_url:
            lines.append(f"- API docs: {stt.doc_url}")

        lines.append(f"\n### TTS: {tts.name}")
        lines.append(f"- Provider: {tts.provider}")
        lines.append(f"- API style: {tts.api_style}")
        lines.append(f"- Output sample rate: {tts.output_sample_rate}Hz")
        lines.append(f"- Output format: {tts.output_format}")
        lines.append(f"- Needs resample to Plivo 8kHz: {tts.needs_resample_to_plivo}")
        lines.append(f"- Env vars: {', '.join(tts.env_vars)}")
        if tts.voice_id_default:
            lines.append(f"- Default voice ID: `{tts.voice_id_default}`")
        if tts.doc_url:
            lines.append(f"- API docs: {tts.doc_url}")
        if tts.notes:
            lines.append(f"- Notes: {tts.notes}")

    lines.append(f"\n### Orchestration: {plan.orchestration.name}")
    lines.append(f"- Needs VAD in utils.py: {plan.orchestration.needs_vad_in_utils}")
    if plan.orchestration.framework_deps:
        lines.append(f"- Framework deps: {', '.join(plan.orchestration.framework_deps)}")

    lines.append(f"\n### Dependencies (all)")
    for dep in plan.all_dependencies:
        lines.append(f"- `{dep}`")

    lines.append(f"\n### Env vars (all)")
    for var in plan.all_env_vars:
        lines.append(f"- `{var}`")

    return "\n".join(lines)


def _build_system_prompt(plan: ExamplePlan, claude_md: str) -> str:
    """Build the system prompt for the generator agent."""
    return textwrap.dedent(f"""\
        You are an expert voice agent engineer. You generate production-ready
        voice agent examples for the python-agents-examples repository.

        You MUST follow the project constitution (CLAUDE.md) exactly. Every file
        must match the canonical structure, naming, and code quality standards.

        Here is the project constitution:

        <claude-md>
        {claude_md}
        </claude-md>

        Important rules:
        1. `from __future__ import annotations` at top of every .py file
        2. Use `loguru` for logging, `python-dotenv` for env vars
        3. No hardcoded API keys — always `os.getenv()`
        4. PLIVO_CHUNK_SIZE = 160 in agent.py (not utils.py)
        5. Audio: Plivo sends/receives base64 μ-law at 8kHz
        6. playAudio format: {{"event": "playAudio", "media": {{"contentType": "audio/x-mulaw", "sampleRate": 8000, "payload": "<base64>"}}}}
        7. All utility functions (audio conversion, VAD, phone normalization) go in utils.py
        8. Server config (SERVER_PORT, PLIVO_AUTH_ID, etc.) goes in server.py
        9. API keys, model names, voice names go in agent.py
        10. The example directory name is: {plan.dir_name}
    """)


def _build_file_prompt(
    plan: ExamplePlan,
    target_file: str,
    reference_files: dict[str, str],
    already_generated: dict[str, str],
    component_spec: str,
) -> str:
    """Build the user prompt for generating a specific file."""
    parts = [
        f"Generate the file `{target_file}` for the voice agent example `{plan.dir_name}`.\n",
        component_spec,
        "\n## Reference Example Files\n",
        f"The reference example is `{plan.reference_example}`. "
        "Adapt the patterns below to the new component specification above.\n",
    ]

    # Include the matching reference file
    ref_content = reference_files.get(target_file, "")
    if ref_content:
        parts.append(f"\n### Reference: {plan.reference_example}/{target_file}\n")
        parts.append(f"```\n{ref_content}\n```\n")

    # Include already-generated files for this example (for consistency)
    if already_generated:
        parts.append("\n## Already Generated Files (for this example)\n")
        parts.append("Use these for import consistency and cross-file references.\n")
        for fname, content in already_generated.items():
            # Only include relevant files, not all of them
            if _is_relevant_context(target_file, fname):
                parts.append(f"\n### {fname} (already generated)\n```\n{content}\n```\n")

    parts.append(
        f"\n## Instructions\n"
        f"Output ONLY the file content for `{target_file}`. No markdown fences, "
        f"no explanation — just the raw file content ready to write to disk.\n"
        f"Adapt the reference to use the components specified above. "
        f"Keep the same structure, patterns, and quality level.\n"
    )

    return "\n".join(parts)


def _is_relevant_context(target: str, generated: str) -> bool:
    """Determine if a previously generated file is relevant context for the target."""
    # utils.py is always relevant
    if generated == "utils.py":
        return True
    # agent.py is relevant for server.py and tests
    if "agent.py" in generated and ("server.py" in target or "test" in target):
        return True
    # inbound files are relevant to outbound (same patterns)
    if "inbound" in generated and "outbound" in target:
        return True
    # server.py is relevant for tests
    if "server.py" in generated and "test" in target:
        return True
    # pyproject.toml is relevant for Dockerfile
    if generated == "pyproject.toml" and target == "Dockerfile":
        return True
    return False


class ExampleGenerator:
    """Generates a complete voice agent example using Claude API."""

    def __init__(
        self,
        repo_root: Path,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 16000,
    ):
        self.repo_root = repo_root
        self.model = model
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic()
        self.claude_md = _read_file(repo_root / "CLAUDE.md")

    def generate_example(
        self,
        plan: ExamplePlan,
        dry_run: bool = False,
    ) -> dict[str, str]:
        """Generate all files for a single example.

        Returns dict of {relative_path: content} for all generated files.
        """
        logger.info(f"Generating example: {plan.dir_name}")
        logger.info(f"  Reference: {plan.reference_example}")
        logger.info(f"  Components: LLM={plan.llm and plan.llm.name}, "
                     f"STT={plan.stt and plan.stt.name}, "
                     f"TTS={plan.tts and plan.tts.name}, "
                     f"VoiceNative={plan.voice_native and plan.voice_native.name}")

        # Read reference files
        reference_files = {}
        if plan.reference_example:
            reference_files = _read_reference_files(self.repo_root, plan.reference_example)

        component_spec = _build_component_spec(plan)
        system_prompt = _build_system_prompt(plan, self.claude_md)

        generated: dict[str, str] = {}

        # Copy static files
        for copy_file in COPY_FILES:
            generated[copy_file] = INIT_PY_CONTENT

        # Generate each file in order
        for target_file in GENERATION_ORDER:
            logger.info(f"  Generating: {target_file}")

            user_prompt = _build_file_prompt(
                plan, target_file, reference_files, generated, component_spec
            )

            if dry_run:
                generated[target_file] = f"# DRY RUN: {target_file} for {plan.dir_name}\n"
                continue

            content = self._call_claude(system_prompt, user_prompt, target_file)
            if content:
                generated[target_file] = content
            else:
                logger.error(f"  Failed to generate: {target_file}")

        return generated

    def _call_claude(self, system: str, user: str, target_file: str) -> str:
        """Call Claude API to generate a single file."""
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                content = response.content[0].text

                # Strip any accidental markdown fences
                content = self._strip_fences(content, target_file)
                return content

            except anthropic.APIError as e:
                logger.warning(f"  API error (attempt {attempt + 1}): {e}")
                if attempt == MAX_RETRIES:
                    logger.error(f"  Giving up on {target_file} after {MAX_RETRIES + 1} attempts")
                    return ""
            except Exception as e:
                logger.error(f"  Unexpected error generating {target_file}: {e}")
                return ""

        return ""

    @staticmethod
    def _strip_fences(content: str, filename: str) -> str:
        """Remove markdown code fences if the model wrapped its output."""
        lines = content.strip().split("\n")
        if not lines:
            return content

        # Check if wrapped in ```python ... ``` or ``` ... ```
        first = lines[0].strip()
        if first.startswith("```"):
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            return "\n".join(lines) + "\n"

        return content


def write_example(
    repo_root: Path,
    plan: ExamplePlan,
    files: dict[str, str],
) -> Path:
    """Write generated files to disk."""
    example_dir = repo_root / plan.dir_name
    logger.info(f"Writing {len(files)} files to {example_dir}")

    for rel_path, content in files.items():
        file_path = example_dir / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        logger.debug(f"  Wrote: {rel_path} ({len(content)} bytes)")

    return example_dir
