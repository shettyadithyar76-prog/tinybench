"""Model adapters, registries, and backend definitions."""

from tinybench.models.base import (
    BaseModelAdapter,
    GenerationResult,
    ModelMetadata,
)
from tinybench.models.downloader import download_model, is_model_installed
from tinybench.models.mock import MockModelAdapter
from tinybench.models.ollama_backend import OllamaModelAdapter
from tinybench.models.registry import APPROVED_MODEL_IDS, MODEL_ALIASES, ModelRegistry
from tinybench.models.transformers_backend import TransformersModelAdapter

__all__ = [
    "APPROVED_MODEL_IDS",
    "BaseModelAdapter",
    "GenerationResult",
    "MODEL_ALIASES",
    "ModelMetadata",
    "MockModelAdapter",
    "ModelRegistry",
    "OllamaModelAdapter",
    "TransformersModelAdapter",
    "download_model",
    "is_model_installed",
]
