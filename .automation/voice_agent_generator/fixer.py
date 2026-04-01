"""Fixer — takes evaluation feedback and regenerates broken files.

Part of the planner-generator-evaluator loop. When the evaluator finds issues,
the fixer uses Claude to produce corrected versions of the affected files.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import anthropic
from loguru import logger

from .evaluator import EvaluationReport, EvalResult
from .generator import _read_file
from .planner import ExamplePlan


class Fixer:
    """Fixes generated files based on evaluation feedback."""

    def __init__(self, repo_root: Path, model: str = "claude-sonnet-4-6"):
        self.repo_root = repo_root
        self.model = model
        self.client = anthropic.Anthropic()
        self.claude_md = _read_file(repo_root / "CLAUDE.md")

    def fix(
        self,
        plan: ExamplePlan,
        files: dict[str, str],
        report: EvaluationReport,
    ) -> dict[str, str]:
        """Fix files based on evaluation failures.

        Returns updated files dict with corrected content.
        """
        if report.passed:
            return files

        fixable = report.fixable_failures
        if not fixable:
            logger.warning("No fixable failures found — cannot auto-fix")
            return files

        logger.info(f"Fixing {len(fixable)} issues in {plan.dir_name}")

        # Group failures by affected file
        file_issues = self._group_issues_by_file(fixable, files)

        for target_file, issues in file_issues.items():
            if target_file not in files:
                logger.warning(f"  Cannot fix {target_file} — not in generated files")
                continue

            logger.info(f"  Fixing: {target_file} ({len(issues)} issues)")
            fixed = self._fix_file(plan, target_file, files[target_file], issues, files)
            if fixed:
                files[target_file] = fixed

        return files

    def _group_issues_by_file(
        self,
        failures: list[EvalResult],
        files: dict[str, str],
    ) -> dict[str, list[EvalResult]]:
        """Map evaluation failures to the files that need fixing."""
        file_issues: dict[str, list[EvalResult]] = {}

        for failure in failures:
            # Determine which file(s) this failure affects
            affected = self._infer_affected_files(failure, files)
            for f in affected:
                file_issues.setdefault(f, []).append(failure)

        return file_issues

    def _infer_affected_files(
        self, failure: EvalResult, files: dict[str, str]
    ) -> list[str]:
        """Infer which files a failure affects based on the check name."""
        name = failure.name.lower()

        if "self_review:" in name:
            # Extract filename from "self_review:path/to/file.py"
            fname = failure.name.split(":", 1)[1]
            if fname in files:
                return [fname]

        # Map check names to files
        mappings = {
            "structure": [],  # Can't fix missing files this way
            "future_annotations": [],  # Need to fix specific files from message
            "loguru_usage": ["inbound/agent.py"],
            "config_placement": ["utils.py"],
            "plivo_chunk_size": ["inbound/agent.py"],
            "mulaw_content_type": ["inbound/agent.py"],
            "stream_xml_mulaw": ["inbound/server.py"],
            "ruff_lint": [],  # Parse ruff output for file names
            "unit_tests": ["utils.py", "tests/test_integration.py"],
            "validate_script": [],  # Too broad
        }

        for key, default_files in mappings.items():
            if key in name:
                if default_files:
                    return default_files
                # Try to extract filenames from the message
                return self._extract_files_from_message(failure.message, files)

        return []

    def _extract_files_from_message(
        self, message: str, files: dict[str, str]
    ) -> list[str]:
        """Extract file paths mentioned in an error message."""
        found = []
        for fname in files:
            # Check for the filename or its basename in the message
            if fname in message or fname.split("/")[-1] in message:
                found.append(fname)
        return found if found else []

    def _fix_file(
        self,
        plan: ExamplePlan,
        target_file: str,
        current_content: str,
        issues: list[EvalResult],
        all_files: dict[str, str],
    ) -> str | None:
        """Use Claude to fix a specific file."""
        issues_text = "\n".join(f"- [{i.name}] {i.message}" for i in issues)

        # Include relevant context files
        context_files = ""
        if target_file != "utils.py" and "utils.py" in all_files:
            context_files += f"\n--- utils.py ---\n{all_files['utils.py']}\n"
        if "agent.py" in target_file and "inbound/server.py" in all_files:
            context_files += (
                f"\n--- inbound/server.py ---\n{all_files['inbound/server.py']}\n"
            )

        prompt = textwrap.dedent(f"""\
            Fix the following file for the voice agent example `{plan.dir_name}`.

            ## Issues to fix
            {issues_text}

            ## Current file content: {target_file}
            ```
            {current_content}
            ```
            {f"## Context files{context_files}" if context_files else ""}

            ## Project rules (from CLAUDE.md)
            - `from __future__ import annotations` at top of every .py file
            - Use `loguru` for logging
            - PLIVO_CHUNK_SIZE = 160 in agent.py
            - playAudio uses "audio/x-mulaw" content type
            - No server config (SERVER_PORT, PLIVO_AUTH_ID, etc.) in utils.py
            - Ruff lint rules: E, W, F, I, B, UP, SIM, RUF (line-length 100)

            Output ONLY the corrected file content. No markdown fences, no explanation.
        """)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=16000,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0].text

            # Strip fences
            lines = content.strip().split("\n")
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                return "\n".join(lines) + "\n"

            return content

        except Exception as e:
            logger.error(f"  Fix failed for {target_file}: {e}")
            return None
