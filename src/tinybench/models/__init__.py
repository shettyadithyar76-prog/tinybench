"""Model adapters, registries, and backend definitions."""

from tinybench.models.base import (
    BaseModelAdapter,
    GenerationResult,
    ModelMetadata,
)
from tinybench.models.mock import MockModelAdapter
from tinybench.models.transformers_backend import TransformersModelAdapter
from tinybench.models.registry import ModelRegistry

__all__ = [
    "BaseModelAdapter",
    "GenerationResult",
    "ModelMetadata",
    "MockModelAdapter",
    "TransformersModelAdapter",
    "ModelRegistry",
]
