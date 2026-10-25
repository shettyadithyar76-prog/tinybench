# TinyBench — Model Integration Guide

This guide explains how to integrate, register, and benchmark new open-weight causal language models or custom simulator backends in TinyBench.

---

## 1. Architecture Overview

Model adapters in TinyBench isolate framework-specific execution (e.g. Hugging Face Transformers, ONNX Runtime, llama.cpp, or mock engines) behind a common interface.

All model adapters subclass `BaseModelAdapter` from `tinybench.models.base`.

---

## 2. Implementing a Model Adapter

To create a new adapter, inherit from `BaseModelAdapter` and implement the abstract methods:

```python
from __future__ import annotations
from typing import Any
from tinybench.config import GenerationConfig
from tinybench.models.base import BaseModelAdapter, ModelMetadata


class MyCustomAdapter(BaseModelAdapter):
    """Custom model adapter for local CPU inference."""

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        super().__init__(model_name, **kwargs)
        self._is_loaded = False

    def load(self) -> None:
        """Initialize the model and tokenizer into memory."""
        # Load weights, tokenizer, or runtime session here
        self._is_loaded = True

    def generate(self, prompt: str, config: GenerationConfig) -> str:
        """Generate text from a prompt given generation constraints."""
        if not self._is_loaded:
            self.load()
        # Execute forward pass and return decoded output text
        return f"Response to: {prompt}"

    def unload(self) -> None:
        """Release weights and free system RAM."""
        self._is_loaded = False

    def get_metadata(self) -> ModelMetadata:
        """Return model metadata for reporting and safety checks."""
        return ModelMetadata(
            model_id=self.model_name,
            model_name="My Custom Model",
            backend_type="custom",
            parameters="135M",
            estimated_ram_mb=400.0,
            license="Apache-2.0",
            description="Custom local model adapter.",
        )
```

---

## 3. Registering Your Model in `ModelRegistry`

To make your model selectable from the CLI (`--model <name>`) and visible in the Streamlit Dashboard, register it in `src/tinybench/models/registry.py`:

```python
# In tinybench/models/registry.py
ModelRegistry.register(
    name="my-custom-model",
    adapter_class=MyCustomAdapter,
    metadata=ModelMetadata(
        model_id="my-custom-model",
        model_name="My Custom Model 135M",
        backend_type="custom",
        parameters="135M",
        estimated_ram_mb=450.0,
        license="Apache-2.0",
        description="Fast local CPU model.",
    ),
)
```

---

## 4. Hardware Safety & Memory Pre-Flight Checks

TinyBench enforces memory safety before loading models onto ordinary CPU hardware:
1. **`estimated_ram_mb`**: Each model declares its expected resident memory footprint.
2. **Pre-flight check**: TinyBench checks `psutil.virtual_memory().available`. If available RAM is below the model's requirement, a clear warning or error is surfaced before allocation occurs.
3. **Clean teardown**: `unload()` releases tensors and invokes garbage collection (`gc.collect()`) to prevent memory leaks across successive runs.

---

## 5. Benchmarking Your Model

### Via CLI:
```bash
python -m tinybench.cli evaluate --model my-custom-model --limit 10
```

### Via Interactive Streamlit Dashboard:
1. Launch the dashboard: `streamlit run app.py`
2. Navigate to **Run Benchmark**.
3. Select your model from the dropdown or enter its Hugging Face repo ID.
4. Click **🚀 Launch Benchmark Run**.
