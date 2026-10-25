"""Unit tests for model adapters, verified catalogue, and registry."""

import os
import pytest
from pathlib import Path

from tinybench.config import GenerationConfig
from tinybench.exceptions import (
    GatedModelAccessError,
    ModelInferenceError,
    ModelLoadError,
    ModelNotFoundError,
    ResourceLimitError,
    RuntimeUnavailableError,
)
from tinybench.models.downloader import check_disk_space_for_model, is_model_installed
from tinybench.models.mock import MockModelAdapter
from tinybench.models.ollama_backend import (
    OllamaModelAdapter,
    check_ollama_daemon,
    is_ollama_model_installed,
)
from tinybench.models.registry import APPROVED_MODEL_IDS, ModelRegistry
from tinybench.models.transformers_backend import TransformersModelAdapter


def test_catalogue_contains_exactly_three_approved_models():
    """Verify that the catalogue contains exactly the 3 approved real models and zero mocks."""
    models = ModelRegistry.list_available_models()
    assert len(models) == 3, f"Expected exactly 3 models, got {len(models)}"

    model_ids = [m.model_id for m in models]
    assert model_ids == [
        "meta-llama/Llama-3.2-1B-Instruct",
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "gemma4:e2b",
    ]

    # Verify no mock models appear in the available catalogue
    for m in models:
        assert "mock" not in m.model_id.lower()
        assert m.backend_type in ("transformers_cpu", "ollama")


def test_approved_model_metadata_verification():
    """Verify official repository IDs, licenses, parameters, and URLs for all 3 models."""
    models = {m.model_id: m for m in ModelRegistry.list_available_models()}

    # 1. Llama 3.2 1B Instruct
    llama = models["meta-llama/Llama-3.2-1B-Instruct"]
    assert llama.model_name == "Llama 3.2 1B Instruct"
    assert llama.developer == "Meta AI"
    assert llama.parameters == "1.23B"
    assert llama.variant == "Instruct"
    assert "Llama 3.2 Community License" in llama.license
    assert "https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/blob/main/LICENSE.txt" == llama.license_url
    assert llama.download_size == "2.47 GB"
    assert llama.supported_runtime == "transformers_cpu"
    assert llama.quantization == "None (BF16 unquantized)"
    assert llama.recommended_ram_mb == 3500.0

    # 2. SmolLM2 1.7B Instruct
    smol = models["HuggingFaceTB/SmolLM2-1.7B-Instruct"]
    assert smol.model_name == "SmolLM2 1.7B Instruct"
    assert smol.developer == "Hugging Face"
    assert smol.parameters == "1.71B"
    assert smol.variant == "Instruct"
    assert smol.license == "Apache-2.0"
    assert "https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/blob/main/LICENSE" == smol.license_url
    assert smol.license_is_osi_approved is True
    assert smol.download_size == "3.42 GB"
    assert smol.supported_runtime == "transformers_cpu"
    assert smol.quantization == "None (FP16/BF16 unquantized)"
    assert smol.recommended_ram_mb == 4000.0

    # 3. Gemma 4 E2B
    gemma = models["gemma4:e2b"]
    assert gemma.model_name == "Gemma 4 E2B"
    assert gemma.developer == "Google DeepMind"
    assert gemma.parameters == "2B effective"
    assert gemma.variant == "E2B (Edge Multimodal)"
    assert gemma.license == "Apache-2.0"
    assert gemma.license_url == "https://gemma.ai/models/gemma-4/"
    assert gemma.license_is_osi_approved is True
    assert gemma.supported_runtime == "ollama"
    assert gemma.runtime_model_id == "gemma4:e2b"
    assert gemma.download_size == "1.8 GB"
    assert gemma.quantization == "4-bit (q4_K_M)"
    assert gemma.recommended_ram_mb == 2500.0


def test_invalid_model_fails_safely():
    """Ensure unapproved or fictional model identifiers raise ModelNotFoundError."""
    with pytest.raises(ModelNotFoundError):
        ModelRegistry.get_adapter("nonexistent-fictional-model")

    with pytest.raises(ModelNotFoundError):
        ModelRegistry.get_adapter("qwen2.5-0.5b")  # Removed unapproved model

    with pytest.raises(ModelNotFoundError):
        ModelRegistry.get_adapter("smollm2-135m")  # Removed unapproved model


def test_model_alias_resolution():
    """Verify canonical alias mapping for approved models."""
    # Llama alias
    a1 = ModelRegistry.get_adapter("llama-3.2-1b")
    assert isinstance(a1, TransformersModelAdapter)
    assert a1.model_id == "meta-llama/Llama-3.2-1B-Instruct"

    # SmolLM2 alias
    a2 = ModelRegistry.get_adapter("smollm2-1.7b")
    assert isinstance(a2, TransformersModelAdapter)
    assert a2.model_id == "HuggingFaceTB/SmolLM2-1.7B-Instruct"

    # Gemma 4 alias
    a3 = ModelRegistry.get_adapter("gemma4-e2b")
    assert isinstance(a3, OllamaModelAdapter)
    assert a3.model_id == "gemma4:e2b"


def test_gated_llama_without_token_raises_gated_error():
    """Llama 3.2 1B requires authentication / agreement; raises GatedModelAccessError without token."""
    old_token = os.environ.pop("HF_TOKEN", None)
    old_hub_token = os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
    try:
        adapter = TransformersModelAdapter("meta-llama/Llama-3.2-1B-Instruct")
        # If model is not already cached locally, load should safely raise GatedModelAccessError
        if not is_model_installed("meta-llama/Llama-3.2-1B-Instruct"):
            with pytest.raises(GatedModelAccessError):
                adapter.load()
    finally:
        if old_token:
            os.environ["HF_TOKEN"] = old_token
        if old_hub_token:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = old_hub_token


def test_ollama_runtime_checks():
    """Verify Ollama runtime health checks and error handling when daemon is down."""
    # On an arbitrary unused port, daemon check should report false cleanly without crashing
    reachable, msg = check_ollama_daemon(base_url="http://127.0.0.1:54321", timeout=0.5)
    assert reachable is False
    assert "not reachable" in msg

    adapter = OllamaModelAdapter(model_id="gemma4:e2b", base_url="http://127.0.0.1:54321")
    with pytest.raises(RuntimeUnavailableError):
        adapter.load()


def test_mock_adapter_lifecycle():
    """Ensure internal test mock adapter operates deterministically for unit testing."""
    adapter = MockModelAdapter(model_id="mock", mode="accurate")
    assert not adapter.is_loaded

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
