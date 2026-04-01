"""Planner — generates valid example combinations from a trigger event.

When a new component releases (e.g., a new Deepgram STT model, a new GPT LLM),
the planner computes all valid {LLM, STT, TTS, orchestration} combinations
that should be generated, filtering out already-existing examples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel

from .registry import (
    LLMS,
    ORCHESTRATIONS,
    STTS,
    TTSS,
    VOICE_NATIVE,
    LLMComponent,
    OrchestrationStyle,
    STTComponent,
    TTSComponent,
    VoiceNativeComponent,
)


class TriggerEvent(BaseModel):
    """Describes what triggered the generation run."""

    component_type: str  # "llm" | "stt" | "tts" | "voice_native"
    component_key: str  # key in registry, e.g. "gpt4.1mini"
    orchestration: str = "native"  # which orchestration to target
    generate_all_combinations: bool = True  # if True, cross with all compatible components


@dataclass
class ExamplePlan:
    """A single example to generate."""

    dir_name: str  # e.g. "gpt4.1mini-deepgram-elevenlabs-native"
    llm: LLMComponent | None = None
    stt: STTComponent | None = None
    tts: TTSComponent | None = None
    voice_native: VoiceNativeComponent | None = None
    orchestration: OrchestrationStyle = field(default_factory=lambda: ORCHESTRATIONS["native"])
    reference_example: str = ""  # existing example to use as template

    @property
    def is_voice_native(self) -> bool:
        return self.voice_native is not None

    @property
    def all_env_vars(self) -> list[str]:
        """Collect all unique env vars needed."""
        env_vars: list[str] = []
        if self.voice_native:
            env_vars.extend(self.voice_native.env_vars)
        else:
            if self.llm:
                env_vars.extend(self.llm.env_vars)
            if self.stt:
                env_vars.extend(self.stt.env_vars)
            if self.tts:
                env_vars.extend(self.tts.env_vars)
        # Plivo is always needed
        env_vars.extend(["PLIVO_AUTH_ID", "PLIVO_AUTH_TOKEN", "PLIVO_PHONE_NUMBER"])
        return list(dict.fromkeys(env_vars))  # deduplicate, preserve order

    @property
    def all_dependencies(self) -> list[str]:
        """Collect all unique Python dependencies."""
        deps: list[str] = [
            "fastapi>=0.115.0",
            "uvicorn[standard]>=0.30.0",
            "websockets>=15.0",
            "plivo>=4.59.0",
            "python-dotenv>=1.0.0",
            "python-multipart>=0.0.9",
            "loguru>=0.7.0",
            "numpy>=1.24.0",
            "scipy>=1.11.0",
            "phonenumbers>=8.13.0",
        ]
        if self.orchestration.needs_vad_in_utils:
            deps.extend(["silero-vad>=5.1", "torch>=2.0.0"])

        if self.voice_native:
            deps.extend(self.voice_native.dependencies)
        else:
            if self.llm:
                deps.extend(self.llm.dependencies)
            if self.stt:
                deps.extend(self.stt.dependencies)
            if self.tts:
                deps.extend(self.tts.dependencies)

        deps.extend(self.orchestration.framework_deps)
        return list(dict.fromkeys(deps))


def build_dir_name(
    llm: LLMComponent | None = None,
    stt: STTComponent | None = None,
    tts: TTSComponent | None = None,
    voice_native: VoiceNativeComponent | None = None,
    orchestration: OrchestrationStyle | None = None,
) -> str:
    """Build the canonical directory name from components.

    Convention: {ai-provider}-{optional-stt}-{optional-tts}-{orchestration}
    Voice-native: {short_name}-{orchestration}
    Pipeline: {llm}-{stt}-{tts}-{orchestration}
    """
    orch = orchestration or ORCHESTRATIONS["native"]
    if voice_native:
        return f"{voice_native.short_name}-{orch.name}"
    parts = []
    if llm:
        parts.append(llm.short_name)
    if stt:
        parts.append(stt.short_name)
    if tts:
        parts.append(tts.short_name)
    parts.append(orch.name)
    return "-".join(parts)


def find_best_reference(plan: ExamplePlan, repo_root: Path) -> str:
    """Find the best existing example to use as a reference/template.

    Preference order:
    1. Same orchestration + same LLM provider
    2. Same orchestration + any provider
    3. grok-voice-native (the primary reference)
    """
    existing = [d.name for d in repo_root.iterdir() if d.is_dir() and not d.name.startswith(".")]

    # Try same orchestration + same provider
    orch_name = plan.orchestration.name
    provider = ""
    if plan.llm:
        provider = plan.llm.provider
    elif plan.voice_native:
        provider = plan.voice_native.provider

    for ex in existing:
        if ex.endswith(f"-{orch_name}") and provider and provider in ex.lower():
            if (repo_root / ex / "inbound" / "agent.py").exists():
                return ex

    # Try same orchestration type
    for ex in existing:
        if ex.endswith(f"-{orch_name}"):
            if (repo_root / ex / "inbound" / "agent.py").exists():
                return ex

    # Fallback to primary reference
    if (repo_root / "grok-voice-native" / "inbound" / "agent.py").exists():
        return "grok-voice-native"

    return ""


def plan_examples(
    trigger: TriggerEvent,
    repo_root: Path,
    skip_existing: bool = True,
) -> list[ExamplePlan]:
    """Generate a list of ExamplePlans from a trigger event.

    For a new LLM:    cross with all STTs × all TTSs
    For a new STT:    cross with all LLMs × all TTSs
    For a new TTS:    cross with all LLMs × all STTs
    For voice_native: single example (S2S APIs are self-contained)
    """
    existing_dirs = set()
    if skip_existing:
        existing_dirs = {
            d.name for d in repo_root.iterdir() if d.is_dir() and not d.name.startswith(".")
        }

    orch = ORCHESTRATIONS.get(trigger.orchestration, ORCHESTRATIONS["native"])
    plans: list[ExamplePlan] = []

    if trigger.component_type == "voice_native":
        vn = VOICE_NATIVE.get(trigger.component_key)
        if not vn:
            raise ValueError(f"Unknown voice_native component: {trigger.component_key}")
        dir_name = build_dir_name(voice_native=vn, orchestration=orch)
        if dir_name not in existing_dirs:
            plan = ExamplePlan(
                dir_name=dir_name, voice_native=vn, orchestration=orch
            )
            plan.reference_example = find_best_reference(plan, repo_root)
            plans.append(plan)
        return plans

    if trigger.component_type == "llm":
        llm = LLMS.get(trigger.component_key)
        if not llm:
            raise ValueError(f"Unknown LLM: {trigger.component_key}")
        for stt_key, stt in STTS.items():
            for tts_key, tts in TTSS.items():
                dir_name = build_dir_name(llm=llm, stt=stt, tts=tts, orchestration=orch)
                if dir_name not in existing_dirs:
                    plan = ExamplePlan(
                        dir_name=dir_name, llm=llm, stt=stt, tts=tts, orchestration=orch
                    )
                    plan.reference_example = find_best_reference(plan, repo_root)
                    plans.append(plan)

    elif trigger.component_type == "stt":
        stt = STTS.get(trigger.component_key)
        if not stt:
            raise ValueError(f"Unknown STT: {trigger.component_key}")
        for llm_key, llm in LLMS.items():
            for tts_key, tts in TTSS.items():
                dir_name = build_dir_name(llm=llm, stt=stt, tts=tts, orchestration=orch)
                if dir_name not in existing_dirs:
                    plan = ExamplePlan(
                        dir_name=dir_name, llm=llm, stt=stt, tts=tts, orchestration=orch
                    )
                    plan.reference_example = find_best_reference(plan, repo_root)
                    plans.append(plan)

    elif trigger.component_type == "tts":
        tts = TTSS.get(trigger.component_key)
        if not tts:
            raise ValueError(f"Unknown TTS: {trigger.component_key}")
        for llm_key, llm in LLMS.items():
            for stt_key, stt in STTS.items():
                dir_name = build_dir_name(llm=llm, stt=stt, tts=tts, orchestration=orch)
                if dir_name not in existing_dirs:
                    plan = ExamplePlan(
                        dir_name=dir_name, llm=llm, stt=stt, tts=tts, orchestration=orch
                    )
                    plan.reference_example = find_best_reference(plan, repo_root)
                    plans.append(plan)
    else:
        raise ValueError(f"Unknown component_type: {trigger.component_type}")

    return plans
