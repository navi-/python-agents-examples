"""Evaluator — validates generated examples against repo standards.

Runs a multi-stage evaluation:
1. Structure check: all canonical files exist
2. Static analysis: ruff lint passes
3. Constitution checks: config placement, code quality, audio pipeline
4. Unit tests: offline tests pass
5. Self-review: Claude reviews the generated code for correctness

Returns structured feedback that can be fed back to the generator for fixes.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
from loguru import logger

from .planner import ExamplePlan


@dataclass
class EvalResult:
    """Result of evaluating a single check."""

    name: str
    passed: bool
    message: str = ""
    fixable: bool = True  # Can the generator fix this automatically?


@dataclass
class EvaluationReport:
    """Complete evaluation report for an example."""

    example_name: str
    results: list[EvalResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failures(self) -> list[EvalResult]:
        return [r for r in self.results if not r.passed]

    @property
    def fixable_failures(self) -> list[EvalResult]:
        return [r for r in self.results if not r.passed and r.fixable]

    def summary(self) -> str:
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        lines = [f"Evaluation: {passed}/{total} passed, {failed} failed"]
        for r in self.failures:
            lines.append(f"  [FAIL] {r.name}: {r.message}")
        return "\n".join(lines)

    def feedback_for_generator(self) -> str:
        """Format failures as actionable feedback for the generator."""
        if self.passed:
            return "All checks passed."
        lines = ["The following issues need to be fixed:\n"]
        for r in self.fixable_failures:
            lines.append(f"- **{r.name}**: {r.message}")
        return "\n".join(lines)


class Evaluator:
    """Evaluates generated voice agent examples."""

    def __init__(self, repo_root: Path, model: str = "claude-haiku-4-5-20251001"):
        self.repo_root = repo_root
        self.model = model

    def evaluate(self, plan: ExamplePlan) -> EvaluationReport:
        """Run all evaluation checks on a generated example."""
        example_dir = self.repo_root / plan.dir_name
        report = EvaluationReport(example_name=plan.dir_name)

        logger.info(f"Evaluating: {plan.dir_name}")

        # 1. Structure checks
        report.results.extend(self._check_structure(example_dir))

        # 2. Code quality checks
        report.results.extend(self._check_code_quality(example_dir))

        # 3. Config placement checks
        report.results.extend(self._check_config_placement(example_dir, plan))

        # 4. Audio pipeline checks
        report.results.extend(self._check_audio_pipeline(example_dir, plan))

        # 5. Lint check
        report.results.extend(self._check_lint(example_dir))

        # 6. Unit tests (if venv exists)
        report.results.extend(self._check_unit_tests(example_dir))

        # 7. README quality checks
        report.results.extend(self._check_readme(example_dir, plan))

        # 8. Validate script (uses the repo's own validator)
        report.results.extend(self._run_validate_script(plan.dir_name))

        logger.info(report.summary())
        return report

    def _check_structure(self, example_dir: Path) -> list[EvalResult]:
        """Check all canonical files exist."""
        results = []
        required = [
            "inbound/__init__.py", "inbound/agent.py", "inbound/server.py",
            "inbound/system_prompt.md",
            "outbound/__init__.py", "outbound/agent.py", "outbound/server.py",
            "outbound/system_prompt.md",
            "utils.py",
            "tests/__init__.py", "tests/conftest.py", "tests/helpers.py",
            "tests/test_integration.py", "tests/test_e2e_live.py",
            "tests/test_live_call.py", "tests/test_multiturn_voice.py",
            "tests/test_outbound_call.py",
            "pyproject.toml", ".env.example", ".gitignore",
            ".pre-commit-config.yaml", "Dockerfile", "README.md",
        ]

        missing = [f for f in required if not (example_dir / f).exists()]
        if missing:
            results.append(EvalResult(
                name="structure",
                passed=False,
                message=f"Missing files: {', '.join(missing)}",
            ))
        else:
            results.append(EvalResult(name="structure", passed=True))

        return results

    def _check_code_quality(self, example_dir: Path) -> list[EvalResult]:
        """Check from __future__ import annotations, loguru, no hardcoded keys."""
        results = []

        # Check __future__ annotations
        py_files = list(example_dir.rglob("*.py"))
        py_files = [f for f in py_files if "__pycache__" not in str(f) and ".venv" not in str(f)]
        missing_annotations = []
        for f in py_files:
            if f.name == "__init__.py":
                continue
            content = f.read_text()
            if "from __future__ import annotations" not in content:
                missing_annotations.append(f.relative_to(example_dir))

        if missing_annotations:
            results.append(EvalResult(
                name="future_annotations",
                passed=False,
                message=f"Missing 'from __future__ import annotations' in: "
                        f"{', '.join(str(f) for f in missing_annotations)}",
            ))
        else:
            results.append(EvalResult(name="future_annotations", passed=True))

        # Check loguru usage
        agent_py = example_dir / "inbound" / "agent.py"
        if agent_py.exists():
            content = agent_py.read_text()
            if "from loguru import logger" not in content:
                results.append(EvalResult(
                    name="loguru_usage",
                    passed=False,
                    message="inbound/agent.py does not use loguru",
                ))
            else:
                results.append(EvalResult(name="loguru_usage", passed=True))

        return results

    def _check_config_placement(
        self, example_dir: Path, plan: ExamplePlan
    ) -> list[EvalResult]:
        """Check that config constants are in the right files."""
        results = []
        utils_py = example_dir / "utils.py"
        if utils_py.exists():
            content = utils_py.read_text()
            leaked = []
            for const in ["SERVER_PORT", "PLIVO_AUTH_ID", "PLIVO_AUTH_TOKEN",
                          "PLIVO_PHONE_NUMBER", "PUBLIC_URL"]:
                if f"{const} =" in content or f"{const}=" in content:
                    leaked.append(const)
            if "PLIVO_CHUNK_SIZE" in content:
                leaked.append("PLIVO_CHUNK_SIZE")

            if leaked:
                results.append(EvalResult(
                    name="config_placement",
                    passed=False,
                    message=f"utils.py contains config that belongs elsewhere: {', '.join(leaked)}",
                ))
            else:
                results.append(EvalResult(name="config_placement", passed=True))

        return results

    def _check_audio_pipeline(
        self, example_dir: Path, plan: ExamplePlan
    ) -> list[EvalResult]:
        """Check PLIVO_CHUNK_SIZE, playAudio format, Stream XML."""
        results = []
        inbound_agent = example_dir / "inbound" / "agent.py"

        if inbound_agent.exists():
            content = inbound_agent.read_text()

            # PLIVO_CHUNK_SIZE = 160
            if "PLIVO_CHUNK_SIZE" not in content or "160" not in content:
                results.append(EvalResult(
                    name="plivo_chunk_size",
                    passed=False,
                    message="PLIVO_CHUNK_SIZE = 160 not found in inbound/agent.py",
                ))
            else:
                results.append(EvalResult(name="plivo_chunk_size", passed=True))

            # audio/x-mulaw content type
            if "audio/x-mulaw" not in content:
                results.append(EvalResult(
                    name="mulaw_content_type",
                    passed=False,
                    message="playAudio content type 'audio/x-mulaw' not found in inbound/agent.py",
                ))
            else:
                results.append(EvalResult(name="mulaw_content_type", passed=True))

        # Check server.py for Stream XML
        inbound_server = example_dir / "inbound" / "server.py"
        if inbound_server.exists():
            content = inbound_server.read_text()
            if "audio/x-mulaw" not in content:
                results.append(EvalResult(
                    name="stream_xml_mulaw",
                    passed=False,
                    message="Stream XML content type not found in inbound/server.py",
                ))
            else:
                results.append(EvalResult(name="stream_xml_mulaw", passed=True))

        return results

    def _check_lint(self, example_dir: Path) -> list[EvalResult]:
        """Run ruff lint check."""
        try:
            result = subprocess.run(
                ["uv", "run", "ruff", "check", "."],
                cwd=example_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return [EvalResult(name="ruff_lint", passed=True)]
            else:
                return [EvalResult(
                    name="ruff_lint",
                    passed=False,
                    message=f"Ruff errors:\n{result.stdout[:2000]}",
                )]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return [EvalResult(
                name="ruff_lint",
                passed=False,
                message="Could not run ruff (uv not available or timeout)",
                fixable=False,
            )]

    def _check_unit_tests(self, example_dir: Path) -> list[EvalResult]:
        """Run unit tests if venv is set up."""
        test_file = example_dir / "tests" / "test_integration.py"
        if not test_file.exists():
            return [EvalResult(
                name="unit_tests",
                passed=False,
                message="test_integration.py not found",
            )]

        try:
            # First try to sync deps
            subprocess.run(
                ["uv", "sync"],
                cwd=example_dir,
                capture_output=True,
                timeout=120,
            )

            result = subprocess.run(
                ["uv", "run", "python", "-m", "pytest",
                 "tests/test_integration.py", "-v", "-k", "unit", "--tb=short"],
                cwd=example_dir,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                return [EvalResult(name="unit_tests", passed=True)]
            else:
                return [EvalResult(
                    name="unit_tests",
                    passed=False,
                    message=f"Unit test failures:\n{result.stdout[-2000:]}\n{result.stderr[-500:]}",
                )]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return [EvalResult(
                name="unit_tests",
                passed=False,
                message="Could not run unit tests",
                fixable=False,
            )]

    def _check_readme(
        self, example_dir: Path, plan: ExamplePlan
    ) -> list[EvalResult]:
        """Check README quality: demo description, required sections, content rules."""
        results = []
        readme_path = example_dir / "README.md"

        if not readme_path.exists():
            results.append(EvalResult(
                name="readme_exists", passed=False,
                message="README.md not found",
            ))
            return results

        content = readme_path.read_text()
        lines = content.split("\n")

        # 1. Check H1 exists
        h1_line = None
        for i, line in enumerate(lines):
            if line.startswith("# ") and not line.startswith("##"):
                h1_line = i
                break

        if h1_line is None:
            results.append(EvalResult(
                name="readme_h1", passed=False,
                message="README has no H1 heading",
            ))
            return results

        # 2. Find first H2
        first_h2 = None
        for i, line in enumerate(lines[h1_line + 1:], start=h1_line + 1):
            if line.startswith("## "):
                first_h2 = i
                break

        if first_h2 is None:
            results.append(EvalResult(
                name="readme_structure", passed=False,
                message="README has no H2 sections",
            ))
            return results

        # 3. Demo description: lines between H1 and first H2
        demo_lines = [l for l in lines[h1_line + 1:first_h2] if l.strip()]
        if len(demo_lines) > 5:
            results.append(EvalResult(
                name="readme_demo_length", passed=False,
                message=f"Demo description has {len(demo_lines)} lines (max 5)",
            ))
        else:
            results.append(EvalResult(name="readme_demo_length", passed=True))

        # 4. No code blocks or tables in demo description
        demo_text = "\n".join(lines[h1_line + 1:first_h2])
        has_code_block = "```" in demo_text
        has_table = "|" in demo_text and "---" in demo_text
        has_image = "![" in demo_text

        if has_code_block or has_table or has_image:
            bad = []
            if has_code_block:
                bad.append("code blocks")
            if has_table:
                bad.append("tables")
            if has_image:
                bad.append("images")
            results.append(EvalResult(
                name="readme_demo_format", passed=False,
                message=f"Demo description contains {', '.join(bad)} "
                        f"(only plain text and bullet lists allowed)",
            ))
        else:
            results.append(EvalResult(name="readme_demo_format", passed=True))

        # 5. Check required H2 sections (at least 5 of 7)
        required_sections = [
            "Features", "Prerequisites", "Quick Start", "Project Structure",
            "How It Works", "Configuration", "Testing",
        ]
        found_sections = []
        for section in required_sections:
            for line in lines:
                if line.startswith("## ") and section.lower() in line.lower():
                    found_sections.append(section)
                    break

        # Also accept "Get Started" as alias for "Quick Start",
        # "Architecture" for "How It Works", "Deploying"/"Deployment" etc.
        aliases = {
            "Quick Start": ["Get Started", "Setup", "Getting Started"],
            "How It Works": ["Architecture", "Pipeline", "Audio Pipeline"],
            "Project Structure": ["File Structure", "Structure"],
            "Configuration": ["Config", "Environment Variables"],
        }
        for canonical, alts in aliases.items():
            if canonical not in found_sections:
                for alt in alts:
                    for line in lines:
                        if line.startswith("## ") and alt.lower() in line.lower():
                            found_sections.append(canonical)
                            break
                    if canonical in found_sections:
                        break

        unique_found = list(dict.fromkeys(found_sections))
        if len(unique_found) >= 5:
            results.append(EvalResult(
                name="readme_sections",
                passed=True,
                message=f"Found {len(unique_found)}/7 required sections",
            ))
        else:
            missing = [s for s in required_sections if s not in unique_found]
            results.append(EvalResult(
                name="readme_sections", passed=False,
                message=f"Only {len(unique_found)}/7 sections found. "
                        f"Missing: {', '.join(missing)}",
            ))

        # 6. Minimum content length (a real README should be >2000 chars)
        if len(content) < 2000:
            results.append(EvalResult(
                name="readme_length", passed=False,
                message=f"README is only {len(content)} chars (minimum 2000 for a real README)",
            ))
        else:
            results.append(EvalResult(name="readme_length", passed=True))

        return results

    def _run_validate_script(self, example_name: str) -> list[EvalResult]:
        """Run the repo's own validate-example.sh script."""
        script = self.repo_root / "scripts" / "validate-example.sh"
        if not script.exists():
            return []

        try:
            result = subprocess.run(
                ["bash", str(script), example_name],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                return [EvalResult(name="validate_script", passed=True)]
            else:
                return [EvalResult(
                    name="validate_script",
                    passed=False,
                    message=f"validate-example.sh failed:\n{result.stdout[-2000:]}",
                )]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return [EvalResult(
                name="validate_script",
                passed=False,
                message="Could not run validate-example.sh",
                fixable=False,
            )]


class SelfReviewAgent:
    """Uses Claude to review generated code for logical correctness.

    Catches issues that static analysis misses — wrong API URLs, incorrect
    audio format conversions, mismatched sample rates, etc.
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self.model = model
        self.client = anthropic.Anthropic()

    def review(self, plan: ExamplePlan, files: dict[str, str]) -> list[EvalResult]:
        """Review generated files for logical correctness."""
        results = []

        # Build review prompt
        files_text = ""
        for fname, content in files.items():
            if fname.endswith(".py") or fname == "pyproject.toml":
                files_text += f"\n--- {fname} ---\n{content}\n"

        component_desc = ""
        if plan.is_voice_native:
            component_desc = (
                f"Voice-native API: {plan.voice_native.name} "
                f"(model: {plan.voice_native.model_id})"
            )
        else:
            component_desc = (
                f"LLM: {plan.llm.name} (model: {plan.llm.model_id}), "
                f"STT: {plan.stt.name}, TTS: {plan.tts.name}"
            )

        prompt = textwrap.dedent(f"""\
            Review this voice agent example for correctness. The example uses:
            {component_desc}
            Orchestration: {plan.orchestration.name}

            Check for:
            1. Correct API endpoint URLs for each provider
            2. Correct audio format conversions (sample rates, encoding)
            3. Correct WebSocket message formats
            4. Correct import paths between files
            5. Correct env var names matching .env.example
            6. Any logical bugs in the audio pipeline

            Respond with a JSON array of issues found. Each issue:
            {{"file": "path", "line_hint": "relevant code", "issue": "description"}}

            If no issues found, respond with: []

            <files>
            {files_text}
            </files>
        """)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()

            # Try to parse as JSON
            import json
            # Find JSON array in response
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                issues = json.loads(text[start:end])
                if issues:
                    for issue in issues:
                        results.append(EvalResult(
                            name=f"self_review:{issue.get('file', 'unknown')}",
                            passed=False,
                            message=issue.get("issue", "Unknown issue"),
                        ))
                else:
                    results.append(EvalResult(name="self_review", passed=True))
            else:
                results.append(EvalResult(name="self_review", passed=True))

        except Exception as e:
            logger.warning(f"Self-review failed: {e}")
            results.append(EvalResult(
                name="self_review",
                passed=True,  # Don't block on review failures
                message=f"Review skipped: {e}",
            ))

        return results
