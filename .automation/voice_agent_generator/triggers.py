"""Trigger detection — identifies new model releases and maps them to generation triggers.

Strategies for detecting new models:
1. API-based: Query provider APIs for available models, diff against registry
2. Config-based: Read a YAML manifest of known models, compare to registry
3. Manual: Accept explicit trigger events via CLI

The detector outputs TriggerEvent objects that feed into the planner.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from loguru import logger

from .planner import TriggerEvent
from .registry import LLMS, STTS, TTSS, VOICE_NATIVE


class ModelReleaseDetector:
    """Detects new model releases by comparing known models against the registry.

    Uses a manifest file (.automation/known-models.yaml) to track what models
    each provider currently offers. When a model appears in the manifest but
    not in the registry, it's a candidate for generation.

    The manifest can be updated manually or by a scheduled job that queries
    provider APIs.
    """

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.manifest_path = repo_root / ".automation" / "known-models.yaml"
        self._manifest: dict | None = None

    @property
    def manifest(self) -> dict:
        if self._manifest is None:
            self._manifest = self._load_manifest()
        return self._manifest

    def _load_manifest(self) -> dict:
        """Load the known-models manifest."""
        if not self.manifest_path.exists():
            logger.warning(f"No manifest found at {self.manifest_path}")
            return {}
        with open(self.manifest_path) as f:
            return yaml.safe_load(f) or {}

    def detect(self, provider: str) -> list[TriggerEvent]:
        """Detect new models from a provider that aren't in the registry yet.

        Returns TriggerEvent objects for each new model found.
        """
        triggers: list[TriggerEvent] = []

        provider_models = self.manifest.get(provider, {})
        if not provider_models:
            logger.info(f"No models listed for provider '{provider}' in manifest")
            return triggers

        # Check LLMs
        for model_info in provider_models.get("llms", []):
            key = model_info.get("key", "")
            if key and key not in LLMS:
                triggers.append(TriggerEvent(
                    component_type="llm",
                    component_key=key,
                ))
                logger.info(f"  New LLM detected: {key}")

        # Check STTs
        for model_info in provider_models.get("stts", []):
            key = model_info.get("key", "")
            if key and key not in STTS:
                triggers.append(TriggerEvent(
                    component_type="stt",
                    component_key=key,
                ))
                logger.info(f"  New STT detected: {key}")

        # Check TTSs
        for model_info in provider_models.get("ttss", []):
            key = model_info.get("key", "")
            if key and key not in TTSS:
                triggers.append(TriggerEvent(
                    component_type="tts",
                    component_key=key,
                ))
                logger.info(f"  New TTS detected: {key}")

        # Check voice-native
        for model_info in provider_models.get("voice_native", []):
            key = model_info.get("key", "")
            if key and key not in VOICE_NATIVE:
                triggers.append(TriggerEvent(
                    component_type="voice_native",
                    component_key=key,
                ))
                logger.info(f"  New voice-native detected: {key}")

        return triggers

    def detect_all(self) -> list[TriggerEvent]:
        """Detect new models across all providers."""
        triggers: list[TriggerEvent] = []
        for provider in self.manifest:
            triggers.extend(self.detect(provider))
        return triggers


def create_trigger_from_api_discovery(
    provider: str,
    component_type: str,
    model_id: str,
    short_name: str,
    **kwargs,
) -> dict:
    """Create a manifest entry for a newly discovered model.

    This is used by external scripts that query provider APIs and want to
    update the manifest with new models they find.
    """
    return {
        "key": short_name,
        "model_id": model_id,
        "discovered_at": None,  # Set by caller
        **kwargs,
    }
