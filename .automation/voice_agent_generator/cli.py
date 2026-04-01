"""CLI — command-line interface for the voice agent generator.

Usage:
    # Generate all examples for a new LLM
    generate-examples trigger --type llm --key gpt5.4mini

    # Generate all examples for a new STT
    generate-examples trigger --type stt --key deepgram

    # Generate all examples for a new TTS
    generate-examples trigger --type tts --key elevenlabs

    # Generate a single voice-native example
    generate-examples trigger --type voice_native --key grok-voice

    # List all registered components
    generate-examples list-components

    # Plan without generating (dry run)
    generate-examples trigger --type llm --key gpt5.4mini --dry-run

    # Detect new model releases and generate
    generate-examples detect --provider openai
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from loguru import logger
from rich.console import Console
from rich.table import Table

from .orchestrator import Orchestrator, git_workflow
from .planner import TriggerEvent, plan_examples
from .registry import LLMS, ORCHESTRATIONS, STTS, TTSS, VOICE_NATIVE
from .triggers import ModelReleaseDetector

console = Console()


def _find_repo_root() -> Path:
    """Find the repository root (parent of .automation/)."""
    # Walk up from this file to find the repo root
    current = Path(__file__).resolve().parent.parent.parent
    if (current / "CLAUDE.md").exists():
        return current
    # Fallback: use CWD
    cwd = Path.cwd()
    if (cwd / "CLAUDE.md").exists():
        return cwd
    if (cwd.parent / "CLAUDE.md").exists():
        return cwd.parent
    raise click.ClickException("Cannot find repo root (CLAUDE.md not found)")


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def main(verbose: bool) -> None:
    """Voice Agent Example Generator — automated example creation."""
    level = "DEBUG" if verbose else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=level)


@main.command()
@click.option("--type", "component_type", required=True,
              type=click.Choice(["llm", "stt", "tts", "voice_native"]),
              help="Type of component that triggered generation")
@click.option("--key", "component_key", required=True,
              help="Registry key for the component (e.g., gpt4.1mini, deepgram)")
@click.option("--orchestration", default="native",
              type=click.Choice(["native", "pipecat", "livekit", "vapi"]),
              help="Target orchestration style")
@click.option("--dry-run", is_flag=True, help="Plan and show what would be generated")
@click.option("--max-examples", type=int, default=None,
              help="Limit number of examples to generate")
@click.option("--model", default="claude-sonnet-4-6",
              help="Claude model for code generation")
@click.option("--branch", default=None, help="Git branch name for commits")
@click.option("--push", is_flag=True, help="Push to remote after committing")
@click.option("--no-skip-existing", is_flag=True,
              help="Regenerate even if example directory exists")
def trigger(
    component_type: str,
    component_key: str,
    orchestration: str,
    dry_run: bool,
    max_examples: int | None,
    model: str,
    branch: str | None,
    push: bool,
    no_skip_existing: bool,
) -> None:
    """Generate examples triggered by a new component release."""
    repo_root = _find_repo_root()

    trigger_event = TriggerEvent(
        component_type=component_type,
        component_key=component_key,
        orchestration=orchestration,
    )

    console.rule(f"[bold]Trigger: new {component_type} '{component_key}'")

    if dry_run:
        # Just plan and show
        plans = plan_examples(trigger_event, repo_root, skip_existing=not no_skip_existing)
        if not plans:
            console.print("[yellow]No new examples needed")
            return

        table = Table(title=f"Would generate {len(plans)} examples")
        table.add_column("Directory", style="cyan")
        table.add_column("LLM")
        table.add_column("STT")
        table.add_column("TTS")
        table.add_column("Reference")

        for p in plans:
            table.add_row(
                p.dir_name,
                p.llm.name if p.llm else (p.voice_native.name if p.voice_native else "—"),
                p.stt.name if p.stt else "—",
                p.tts.name if p.tts else "—",
                p.reference_example or "—",
            )
        console.print(table)
        return

    # Full run
    orchestrator = Orchestrator(
        repo_root=repo_root,
        generator_model=model,
        dry_run=False,
        skip_existing=not no_skip_existing,
        max_examples=max_examples,
    )

    result = orchestrator.run(trigger_event)

    # Git workflow
    if result.succeeded > 0 and not dry_run:
        git_workflow(repo_root, result.results, branch_name=branch, push=push)


@main.command("list-components")
def list_components() -> None:
    """List all registered components."""
    console.rule("[bold]LLMs")
    table = Table()
    table.add_column("Key", style="cyan")
    table.add_column("Name")
    table.add_column("Provider")
    table.add_column("Model ID")
    table.add_column("API Style")
    for key, llm in LLMS.items():
        table.add_row(key, llm.name, llm.provider, llm.model_id, llm.api_style)
    console.print(table)

    console.rule("[bold]STTs")
    table = Table()
    table.add_column("Key", style="cyan")
    table.add_column("Name")
    table.add_column("Provider")
    table.add_column("Sample Rate")
    table.add_column("Format")
    for key, stt in STTS.items():
        table.add_row(key, stt.name, stt.provider, str(stt.input_sample_rate), stt.input_format)
    console.print(table)

    console.rule("[bold]TTSs")
    table = Table()
    table.add_column("Key", style="cyan")
    table.add_column("Name")
    table.add_column("Provider")
    table.add_column("Sample Rate")
    table.add_column("Format")
    for key, tts in TTSS.items():
        table.add_row(key, tts.name, tts.provider, str(tts.output_sample_rate), tts.output_format)
    console.print(table)

    console.rule("[bold]Voice-Native (S2S)")
    table = Table()
    table.add_column("Key", style="cyan")
    table.add_column("Name")
    table.add_column("Provider")
    table.add_column("Model ID")
    for key, vn in VOICE_NATIVE.items():
        table.add_row(key, vn.name, vn.provider, vn.model_id)
    console.print(table)

    console.rule("[bold]Orchestrations")
    table = Table()
    table.add_column("Key", style="cyan")
    table.add_column("VAD in utils")
    table.add_column("Framework Deps")
    for key, orch in ORCHESTRATIONS.items():
        table.add_row(
            key,
            str(orch.needs_vad_in_utils),
            ", ".join(orch.framework_deps) if orch.framework_deps else "—",
        )
    console.print(table)


@main.command()
@click.option("--type", "component_type", required=True,
              type=click.Choice(["llm", "stt", "tts", "voice_native"]),
              help="Type of component to show combinations for")
@click.option("--key", "component_key", required=True,
              help="Registry key for the component")
@click.option("--orchestration", default="native")
def plan(component_type: str, component_key: str, orchestration: str) -> None:
    """Show what examples would be generated (without generating)."""
    repo_root = _find_repo_root()
    trigger_event = TriggerEvent(
        component_type=component_type,
        component_key=component_key,
        orchestration=orchestration,
    )
    plans = plan_examples(trigger_event, repo_root)
    if not plans:
        console.print("[yellow]No new examples needed — all combinations already exist")
        return

    for p in plans:
        existing = "NEW" if not (repo_root / p.dir_name).is_dir() else "EXISTS"
        console.print(f"  [{existing}] {p.dir_name} (ref: {p.reference_example})")

    console.print(f"\n[bold]{len(plans)} examples would be generated")


@main.command()
@click.option("--provider", required=True,
              help="Provider to check for new models (openai, google, xai, deepgram, etc.)")
@click.option("--auto-generate", is_flag=True,
              help="Automatically generate examples for detected new models")
@click.option("--model", default="claude-sonnet-4-6",
              help="Claude model for code generation")
def detect(provider: str, auto_generate: bool, model: str) -> None:
    """Detect new model releases from a provider and optionally generate examples."""
    repo_root = _find_repo_root()
    detector = ModelReleaseDetector(repo_root)

    console.rule(f"[bold]Detecting new models from {provider}")
    triggers = detector.detect(provider)

    if not triggers:
        console.print(f"[yellow]No new models detected for {provider}")
        return

    for t in triggers:
        console.print(f"  [NEW] {t.component_type}: {t.component_key}")

    if auto_generate:
        for t in triggers:
            orchestrator = Orchestrator(
                repo_root=repo_root,
                generator_model=model,
            )
            result = orchestrator.run(t)
            if result.succeeded > 0:
                git_workflow(repo_root, result.results)


@main.group()
def docs() -> None:
    """Generate documentation for plivo.com/docs."""


@docs.command("guide")
@click.option("--example", required=True, help="Example directory name")
@click.option("--model", default="claude-sonnet-4-6", help="Claude model")
def docs_guide(example: str, model: str) -> None:
    """Generate a Plivo docs guide for a specific example."""
    from .plivo_docs import PlivoDocsGenerator

    repo_root = _find_repo_root()
    gen = PlivoDocsGenerator(repo_root, model=model)

    # Build a minimal plan from the example directory
    plan = _plan_from_existing(example, repo_root)
    if not plan:
        console.print(f"[red]Cannot build plan for {example}")
        return

    content = gen.generate_guide(plan)
    output_dir = repo_root / ".automation" / "docs" / "guides"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{example}.md"
    output_path.write_text(content)
    console.print(f"[green]Wrote guide: {output_path}")


@docs.command("reference")
@click.option("--provider", required=True,
              help="Provider to generate reference for (e.g., openai, deepgram, elevenlabs)")
@click.option("--model", default="claude-sonnet-4-6", help="Claude model")
def docs_reference(provider: str, model: str) -> None:
    """Generate a Plivo docs reference page for a provider."""
    from .plivo_docs import PlivoDocsGenerator

    repo_root = _find_repo_root()
    gen = PlivoDocsGenerator(repo_root, model=model)

    content = gen.generate_provider_reference(provider)
    output_dir = repo_root / ".automation" / "docs" / "reference"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{provider}.md"
    output_path.write_text(content)
    console.print(f"[green]Wrote reference: {output_path}")


@docs.command("concepts")
@click.option("--topic", default=None, help="Specific topic (or all if omitted)")
@click.option("--model", default="claude-sonnet-4-6", help="Claude model")
def docs_concepts(topic: str | None, model: str) -> None:
    """Generate Plivo docs concept pages."""
    from .plivo_docs import CONCEPT_TOPICS, PlivoDocsGenerator

    repo_root = _find_repo_root()
    gen = PlivoDocsGenerator(repo_root, model=model)

    if topic:
        if topic not in CONCEPT_TOPICS:
            console.print(f"[red]Unknown topic: {topic}")
            console.print(f"Available: {', '.join(CONCEPT_TOPICS.keys())}")
            return
        content = gen.generate_concept(topic)
        output_dir = repo_root / ".automation" / "docs" / "concepts"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{topic}.md"
        output_path.write_text(content)
        console.print(f"[green]Wrote concept: {output_path}")
    else:
        concepts = gen.generate_all_concepts()
        output_dir = repo_root / ".automation" / "docs" / "concepts"
        output_dir.mkdir(parents=True, exist_ok=True)
        for t, content in concepts.items():
            (output_dir / f"{t}.md").write_text(content)
            console.print(f"[green]Wrote concept: {t}.md")


@docs.command("all")
@click.option("--model", default="claude-sonnet-4-6", help="Claude model")
def docs_all(model: str) -> None:
    """Generate all Plivo docs (guides for existing examples + references + concepts)."""
    from .plivo_docs import PlivoDocsGenerator

    repo_root = _find_repo_root()
    gen = PlivoDocsGenerator(repo_root, model=model)

    console.rule("[bold]Generating Plivo docs")

    # Concepts
    console.print("[cyan]Generating concept pages...")
    concepts = gen.generate_all_concepts()

    # References
    console.print("[cyan]Generating reference pages...")
    references = gen.generate_all_references()

    # Guides for all existing examples
    console.print("[cyan]Generating guides for existing examples...")
    guides = {}
    existing_dirs = sorted(
        d.name for d in repo_root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
        and d.name not in ("scripts", "__pycache__")
        and (d / "inbound" / "agent.py").exists()
    )

    for example in existing_dirs:
        plan = _plan_from_existing(example, repo_root)
        if plan:
            content = gen.generate_guide(plan)
            guides[example] = content
            console.print(f"  [green]Guide: {example}")

    # Write everything
    gen.write_docs(guides=guides, references=references, concepts=concepts)
    console.print(f"\n[bold green]Generated {len(concepts)} concepts, "
                  f"{len(references)} references, {len(guides)} guides")
    console.print(f"Output: {gen.docs_dir}")


def _plan_from_existing(example_name: str, repo_root: Path) -> ExamplePlan | None:
    """Build a minimal ExamplePlan from an existing example directory.

    Infers components from the directory name and registry.
    """
    from .planner import ExamplePlan

    # Parse the directory name to identify components
    parts = example_name.split("-")

    # Check if it's a voice-native example
    for vn_key, vn in VOICE_NATIVE.items():
        if example_name.startswith(vn.short_name):
            orch_name = parts[-1] if parts[-1] in ORCHESTRATIONS else "native"
            orch = ORCHESTRATIONS.get(orch_name, ORCHESTRATIONS["native"])
            return ExamplePlan(
                dir_name=example_name,
                voice_native=vn,
                orchestration=orch,
            )

    # Try to match LLM, STT, TTS from registry
    orch_name = parts[-1] if parts[-1] in ORCHESTRATIONS else "native"
    orch = ORCHESTRATIONS.get(orch_name, ORCHESTRATIONS["native"])

    matched_llm = None
    matched_stt = None
    matched_tts = None

    for key, llm in LLMS.items():
        if llm.short_name in example_name:
            matched_llm = llm
            break

    for key, stt in STTS.items():
        if stt.short_name in example_name:
            matched_stt = stt
            break

    for key, tts in TTSS.items():
        if tts.short_name in example_name:
            matched_tts = tts
            break

    if matched_llm or matched_stt or matched_tts:
        return ExamplePlan(
            dir_name=example_name,
            llm=matched_llm,
            stt=matched_stt,
            tts=matched_tts,
            orchestration=orch,
        )

    return None


if __name__ == "__main__":
    main()
