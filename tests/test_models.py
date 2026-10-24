"""Unit tests for model adapters and registry."""

import pytest

from tinybench.config import GenerationConfig
from tinybench.exceptions import ModelInferenceError, ModelLoadError, ResourceLimitError
from tinybench.models.mock import MockModelAdapter
from tinybench.models.registry import ModelRegistry
from tinybench.models.transformers_backend import TransformersModelAdapter


def test_mock_adapter_lifecycle():
    adapter = MockModelAdapter(model_id="mock", mode="accurate")
    assert not adapter.is_loaded

    # Generate before load raises error
    with pytest.raises(ModelLoadError):
        adapter.generate("Test prompt", GenerationConfig())

    init_time = adapter.load()
    assert adapter.is_loaded
    assert init_time >= 0.0

    res = adapter.generate("What is 15 apples and 27 more?", GenerationConfig())
    assert "42" in res.text
    assert res.latency_seconds >= 0.0

    adapter.unload()
    assert not adapter.is_loaded


def test_mock_noisy_mode():
    adapter = MockModelAdapter(model_id="mock-noisy", mode="noisy", noise_rate=1.0)
    adapter.load()
    res = adapter.generate("What is 15 apples and 27 more?", GenerationConfig())
    # Noisy mode synthesizes deliberate non-target answers
    assert "35" in res.text


def test_mock_failing_mode():
    adapter = MockModelAdapter(model_id="mock-failing", mode="failing")
    adapter.load()
    with pytest.raises(ModelInferenceError):
        adapter.generate("Trigger error here", GenerationConfig())


def test_mock_metadata():
    adapter = MockModelAdapter(model_id="mock")
    meta = adapter.get_metadata()
    assert meta.model_id == "mock"
    assert meta.backend_type == "mock"
    assert meta.license == "MIT"
    assert meta.license_is_osi_approved is True
    assert meta.estimated_ram_mb > 0


def test_model_registry_resolution():
    models = ModelRegistry.list_available_models()
    assert len(models) >= 3
    ids = [m.model_id for m in models]
    assert "mock" in ids

    # Alias lookups
    m1 = ModelRegistry.get_adapter("mock")
    assert isinstance(m1, MockModelAdapter)

    m2 = ModelRegistry.get_adapter("smollm2-135m")
    assert isinstance(m2, TransformersModelAdapter)
    assert m2.model_id == "HuggingFaceTB/SmolLM2-135M-Instruct"

    m3 = ModelRegistry.get_adapter("custom/org-model-v1")
    assert isinstance(m3, TransformersModelAdapter)
    assert m3.model_id == "custom/org-model-v1"


def test_transformers_metadata_without_loading():
    adapter = TransformersModelAdapter(model_id="HuggingFaceTB/SmolLM2-135M-Instruct")
    meta = adapter.get_metadata()
    assert meta.family == "SmolLM2"
    assert meta.parameters == "135M"
    assert meta.license == "Apache-2.0"
    assert meta.license_is_osi_approved is True
    assert meta.estimated_ram_mb == 400.0
