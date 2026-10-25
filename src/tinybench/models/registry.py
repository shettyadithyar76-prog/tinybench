"""Registry of available model backends and verified model catalogue."""

from __future__ import annotations

from typing import Any
from tinybench.exceptions import ModelNotFoundError
from tinybench.models.base import BaseModelAdapter, ModelMetadata
from tinybench.models.mock import MockModelAdapter
from tinybench.models.ollama_backend import OLLAMA_CATALOGUE, OllamaModelAdapter
from tinybench.models.transformers_backend import KNOWN_MODELS, TransformersModelAdapter

# Exactly the three approved real open-weight models
APPROVED_MODEL_IDS: list[str] = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "gemma4:e2b",
]

# Canonical aliases for convenience across CLI and UI
MODEL_ALIASES: dict[str, str] = {
    # Llama 3.2 1B Instruct
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3.2": "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/llama-3.2-1b-instruct": "meta-llama/Llama-3.2-1B-Instruct",
    # SmolLM2 1.7B Instruct
    "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "smollm2-1.7b-instruct": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "smollm2": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "huggingfacetb/smollm2-1.7b-instruct": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    # Gemma 4 E2B
    "gemma4-e2b": "gemma4:e2b",
    "gemma-4-e2b": "gemma4:e2b",
    "gemma4": "gemma4:e2b",
    "google/gemma-4-e2b": "gemma4:e2b",
    "google/gemma-4-E2B": "gemma4:e2b",
}


class ModelRegistry:
    """Single source of truth for the approved TinyBench model catalogue."""

    APPROVED_MODELS = APPROVED_MODEL_IDS

    @classmethod
    def list_available_models(cls) -> list[ModelMetadata]:
        """Return verified metadata for exactly the three approved models."""
        models: list[ModelMetadata] = []

        # 1. Llama 3.2 1B Instruct
        llama_adapter = TransformersModelAdapter("meta-llama/Llama-3.2-1B-Instruct")
        models.append(llama_adapter.get_metadata())

        # 2. SmolLM2 1.7B Instruct
        smol_adapter = TransformersModelAdapter("HuggingFaceTB/SmolLM2-1.7B-Instruct")
        models.append(smol_adapter.get_metadata())

        # 3. Gemma 4 E2B
        gemma_adapter = OllamaModelAdapter("gemma4:e2b")
        models.append(gemma_adapter.get_metadata())

        return models

    @classmethod
    def get_adapter(cls, name_or_id: str, revision: str | None = None) -> BaseModelAdapter:
        """Instantiate model adapter by approved model identifier or alias."""
        key = name_or_id.strip()
        canonical = MODEL_ALIASES.get(key.lower(), key)

        if canonical == "meta-llama/Llama-3.2-1B-Instruct":
            return TransformersModelAdapter(model_id=canonical, revision=revision)
        elif canonical == "HuggingFaceTB/SmolLM2-1.7B-Instruct":
            return TransformersModelAdapter(model_id=canonical, revision=revision)
        elif canonical == "gemma4:e2b":
            return OllamaModelAdapter(model_id=canonical, revision=revision)
        elif canonical.startswith("mock"):
            # Internal test simulator for unit tests and CI pipelines only
            mode = "accurate"
            if "noisy" in canonical:
                mode = "noisy"
            elif "failing" in canonical:
                mode = "failing"
            elif "slow" in canonical:
                mode = "slow"
            return MockModelAdapter(model_id=canonical, mode=mode)
        else:
            raise ModelNotFoundError(
                f"Model '{name_or_id}' is not in the approved TinyBench catalogue. "
                f"Approved models: {', '.join(APPROVED_MODEL_IDS)}"
            )

    @classmethod
    def is_approved_model(cls, name_or_id: str) -> bool:
        """Check whether a given name or alias maps to an approved model."""
        key = name_or_id.strip()
        canonical = MODEL_ALIASES.get(key.lower(), key)
        return canonical in APPROVED_MODEL_IDS
