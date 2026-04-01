"""Orchestrator — runs the full planner → generator → evaluator → fixer loop.

This is the core agentic loop:
1. Planner computes all example combinations to generate
2. Generator produces code files using Claude
3. Evaluator validates each example (structure, lint, tests, self-review)
4. Fixer corrects any issues found by the evaluator
5. Re-evaluate to confirm fixes
6. Git workflow: branch, commit, push

Repeats the generate-evaluate-fix cycle up to MAX_FIX_ITERATIONS times.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.table import Table

from .evaluator import EvaluationReport, Evaluator, SelfReviewAgent
from .fixer import Fixer
from .generator import ExampleGenerator, write_example
from .planner import ExamplePlan, TriggerEvent, plan_examples
from .plivo_docs import PlivoDocsGenerator
from .readme_gen import ReadmeGenerator

MAX_FIX_ITERATIONS = 3
console = Console()


@dataclass
class GenerationResult:
    """Result of generating a single example."""

    plan: ExamplePlan
    files: dict[str, str] = field(default_factory=dict)
    reports: list[EvaluationReport] = field(default_factory=list)
    success: bool = False
    iterations: int = 0
    error: str = ""


@dataclass
class OrchestratorResult:
    """Result of the full orchestration run."""

    trigger: TriggerEvent
    results: list[GenerationResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def succeeded(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def failed(self) -> int:
        return self.total - self.succeeded


class Orchestrator:
    """Runs the full example generation pipeline."""

    def __init__(
        self,
        repo_root: Path,
        generator_model: str = "claude-sonnet-4-6",
        reviewer_model: str = "claude-haiku-4-5-20251001",
        dry_run: bool = False,
        skip_existing: bool = True,
        max_examples: int | None = None,
    ):
        self.repo_root = repo_root
        self.dry_run = dry_run
        self.skip_existing = skip_existing
        self.max_examples = max_examples

        self.generator = ExampleGenerator(repo_root, model=generator_model)
        self.readme_gen = ReadmeGenerator(repo_root, model=generator_model)
        self.docs_gen = PlivoDocsGenerator(repo_root, model=generator_model)
        self.evaluator = Evaluator(repo_root)
        self.reviewer = SelfReviewAgent(model=reviewer_model)
        self.fixer = Fixer(repo_root, model=generator_model)

    def run(self, trigger: TriggerEvent) -> OrchestratorResult:
        """Execute the full pipeline for a trigger event."""
        result = OrchestratorResult(trigger=trigger)

        # Phase 1: Plan
        console.rule("[bold blue]Phase 1: Planning")
        plans = plan_examples(trigger, self.repo_root, skip_existing=self.skip_existing)

        if self.max_examples and len(plans) > self.max_examples:
            logger.info(f"Limiting to {self.max_examples} examples (of {len(plans)} planned)")
            plans = plans[:self.max_examples]

        if not plans:
            console.print("[yellow]No new examples to generate (all combinations exist)")
            return result

        self._print_plan_table(plans)

        # Phase 2-4: Generate → Evaluate → Fix for each example
        for i, plan in enumerate(plans, 1):
            console.rule(f"[bold green]Example {i}/{len(plans)}: {plan.dir_name}")
            gen_result = self._generate_one(plan)
            result.results.append(gen_result)

        # Summary
        self._print_summary(result)
        return result

    def _generate_one(self, plan: ExamplePlan) -> GenerationResult:
        """Generate, evaluate, and fix a single example."""
        gen_result = GenerationResult(plan=plan)

        try:
            # Generate
            console.print(f"[cyan]Generating {plan.dir_name}...")
            files = self.generator.generate_example(plan, dry_run=self.dry_run)
            gen_result.files = files

            if self.dry_run:
                gen_result.success = True
                console.print("[yellow]  (dry run — skipping evaluation)")
                return gen_result

            # Generate README with dedicated generator (higher quality)
            console.print(f"[cyan]  Generating README...")
            readme_content = self.readme_gen.generate(plan)
            files["README.md"] = readme_content

            # Write files to disk
            write_example(self.repo_root, plan, files)

            # Generate Plivo docs guide (in .automation/docs/guides/)
            console.print(f"[cyan]  Generating Plivo docs guide...")
            try:
                guide_content = self.docs_gen.generate_guide(plan)
                guides_dir = self.repo_root / ".automation" / "docs" / "guides"
                guides_dir.mkdir(parents=True, exist_ok=True)
                (guides_dir / f"{plan.dir_name}.md").write_text(guide_content)
            except Exception as e:
                logger.warning(f"  Plivo docs guide generation failed: {e}")

            # Evaluate → Fix loop
            for iteration in range(MAX_FIX_ITERATIONS + 1):
                gen_result.iterations = iteration + 1
                console.print(f"[cyan]  Evaluation round {iteration + 1}...")

                # Evaluate
                report = self.evaluator.evaluate(plan)

                # Self-review (only on first iteration to save API calls)
                if iteration == 0:
                    review_results = self.reviewer.review(plan, files)
                    report.results.extend(review_results)

                gen_result.reports.append(report)

                if report.passed:
                    gen_result.success = True
                    console.print(f"[green]  ✓ All checks passed (iteration {iteration + 1})")
                    break

                # Report failures
                console.print(f"[red]  {len(report.failures)} failures found")
                for f in report.failures:
                    console.print(f"[red]    • {f.name}: {f.message[:100]}")

                if iteration == MAX_FIX_ITERATIONS:
                    console.print("[red]  ✗ Max fix iterations reached")
                    gen_result.error = report.feedback_for_generator()
                    break

                # Fix
                if report.fixable_failures:
                    console.print(f"[yellow]  Fixing {len(report.fixable_failures)} issues...")
                    files = self.fixer.fix(plan, files, report)
                    gen_result.files = files
                    # Re-write fixed files
                    write_example(self.repo_root, plan, files)
                else:
                    console.print("[red]  No fixable failures — cannot auto-fix")
                    gen_result.error = "Unfixable failures: " + report.feedback_for_generator()
                    break

        except Exception as e:
            logger.error(f"Error generating {plan.dir_name}: {e}")
            gen_result.error = str(e)

        return gen_result

    def _print_plan_table(self, plans: list[ExamplePlan]) -> None:
        """Print a table of planned examples."""
        table = Table(title="Planned Examples")
        table.add_column("Directory", style="cyan")
        table.add_column("LLM")
        table.add_column("STT")
        table.add_column("TTS")
        table.add_column("Orchestration")
        table.add_column("Reference")

        for plan in plans:
            table.add_row(
                plan.dir_name,
                plan.llm.name if plan.llm else (plan.voice_native.name if plan.voice_native else "—"),
                plan.stt.name if plan.stt else "—",
                plan.tts.name if plan.tts else "—",
                plan.orchestration.name,
                plan.reference_example or "—",
            )

        console.print(table)

    def _print_summary(self, result: OrchestratorResult) -> None:
        """Print final summary table."""
        console.rule("[bold blue]Summary")
        table = Table(title="Generation Results")
        table.add_column("Example", style="cyan")
        table.add_column("Status")
        table.add_column("Iterations")
        table.add_column("Notes")

        for r in result.results:
            status = "[green]✓ PASS" if r.success else "[red]✗ FAIL"
            notes = ""
            if r.error:
                notes = r.error[:80]
            table.add_row(r.plan.dir_name, status, str(r.iterations), notes)

        console.print(table)
        console.print(
            f"\n[bold]Total: {result.total} | "
            f"[green]Passed: {result.succeeded}[/green] | "
            f"[red]Failed: {result.failed}[/red]"
        )


def git_workflow(
    repo_root: Path,
    results: list[GenerationResult],
    branch_name: str | None = None,
    push: bool = False,
) -> None:
    """Create git branch, commit successful examples, optionally push."""
    successful = [r for r in results if r.success]
    if not successful:
        console.print("[yellow]No successful examples to commit")
        return

    # Determine branch name
    if not branch_name:
        dir_names = [r.plan.dir_name for r in successful]
        if len(dir_names) == 1:
            branch_name = dir_names[0]
        else:
            branch_name = f"generated-examples-{len(dir_names)}"

    # Create branch
    subprocess.run(["git", "checkout", "-b", branch_name], cwd=repo_root, check=False)

    # Stage and commit each example
    for r in successful:
        example_dir = r.plan.dir_name
        subprocess.run(["git", "add", example_dir], cwd=repo_root, check=True)

    # Commit
    dir_list = ", ".join(r.plan.dir_name for r in successful)
    commit_msg = f"Add generated voice agent examples: {dir_list}"
    subprocess.run(["git", "commit", "-m", commit_msg], cwd=repo_root, check=True)
    console.print(f"[green]Committed {len(successful)} examples on branch {branch_name}")

    # Push
    if push:
        subprocess.run(
            ["git", "push", "-u", "origin", branch_name],
            cwd=repo_root,
            check=True,
        )
        console.print(f"[green]Pushed to origin/{branch_name}")
