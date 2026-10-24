"""Registry of available model backends and factory methods."""

from __future__ import annotations

from typing import Callable
from tinybench.models.base import BaseModelAdapter, ModelMetadata
from tinybench.models.mock import MockModelAdapter
from tinybench.models.transformers_backend import KNOWN_MODELS, TransformersModelAdapter

# Built-in aliases mapped to canonical identifiers
MODEL_ALIASES: dict[str, str] = {
    "mock": "mock",
    "mock-noisy": "mock-noisy",
    "mock-slow": "mock-slow",
    "mock-failing": "mock-failing",
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "smollm2-360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
}


class ModelRegistry:
    """Manages model adapters, metadata discovery, and instantiation."""

    @classmethod
    def list_available_models(cls) -> list[ModelMetadata]:
        """Return metadata for all standard registered models and mocks."""
        models: list[ModelMetadata] = [
            MockModelAdapter(model_id="mock", mode="accurate").get_metadata(),
            MockModelAdapter(model_id="mock-noisy", mode="noisy", noise_rate=0.3).get_metadata(),
            MockModelAdapter(model_id="mock-slow", mode="accurate", simulated_latency=0.03).get_metadata(),
        ]

        for model_id in KNOWN_MODELS.keys():
            adapter = TransformersModelAdapter(model_id=model_id)
            models.append(adapter.get_metadata())

        return models

    @classmethod
    def get_adapter(cls, name_or_id: str, revision: str | None = None) -> BaseModelAdapter:
        """Instantiate appropriate model adapter by alias or repo id."""
        key = name_or_id.strip()
        canonical = MODEL_ALIASES.get(key.lower(), key)

        if canonical == "mock":
            return MockModelAdapter(model_id="mock", mode="accurate")
        elif canonical == "mock-noisy":
            return MockModelAdapter(model_id="mock-noisy", mode="noisy", noise_rate=0.3)
        elif canonical == "mock-slow":
            return MockModelAdapter(model_id="mock-slow", mode="accurate", simulated_latency=0.03)
        elif canonical == "mock-failing":
            return MockModelAdapter(model_id="mock-failing", mode="failing")
        else:
            # Treat as Transformers model (HF Hub or local folder)
            return TransformersModelAdapter(model_id=canonical, revision=revision)
