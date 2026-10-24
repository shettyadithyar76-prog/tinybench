# Contributor Developer Guide

This guide walks you through extending TinyBench with new model backends, benchmark categories, and metrics.

---

## 1. How to Add a New Model Adapter

All model backends inherit from `BaseModelAdapter` located in `src/tinybench/models/base.py`.

### Step 1: Create Your Adapter Class
Create a new file in `src/tinybench/models/` (e.g. `onnx_backend.py`):

```python
from tinybench.config import GenerationConfig
from tinybench.models.base import BaseModelAdapter, GenerationResult, ModelMetadata
import time

class OnnxModelAdapter(BaseModelAdapter):
    def __init__(self, model_id: str, onnx_path: str):
        super().__init__(model_id=model_id)
        self.onnx_path = onnx_path

    def load(self) -> float:
        t0 = time.perf_counter()
        # Initialize your runtime session here
        self._is_loaded = True
        return round(time.perf_counter() - t0, 4)

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_id=self.model_id,
            model_name="ONNX Runtime Model",
            family="ONNX",
            parameters="135M",
            license="Apache-2.0",
            license_is_osi_approved=True,
            estimated_ram_mb=300.0,
            backend_type="onnx_cpu",
            description="ONNX Runtime CPU quantized model.",
        )

    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        t0 = time.perf_counter()
        # Run inference using ONNX session
        generated_text = "sample text"
        return GenerationResult(
            text=generated_text,
            prompt_tokens=len(prompt.split()),
            generated_tokens=len(generated_text.split()),
            latency_seconds=round(time.perf_counter() - t0, 4),
        )

    def unload(self) -> None:
        self._is_loaded = False
```

### Step 2: Register in `ModelRegistry`
In `src/tinybench/models/registry.py`:
- Import your adapter.
- Add an alias in `MODEL_ALIASES`.
- Return an instance in `ModelRegistry.get_adapter()`.

### Step 3: Add Unit Tests
Add tests in `tests/test_models.py` verifying that initialization, generation, and unloading function as expected.

---

## 2. How to Add a New Benchmark Category

1. Add your category to `TaskCategory` enum in `src/tinybench/benchmarks/schema.py`:
   ```python
   class TaskCategory(str, Enum):
       ...
       CODE_COMPLETION = "code_completion"
   ```
2. Add corresponding examples into `datasets/basic_v1.jsonl` with clear `expected_answer` and deterministic `scoring_method`.
3. In `src/tinybench/runner.py`, add grading dispatch in `_evaluate_single_task` if your category requires specialized evaluation logic.
4. Run `python -m tinybench.cli validate-dataset datasets/basic_v1.jsonl` to ensure the dataset passes schema checks.

---

## 3. How to Add an Evaluation Metric

1. Add your metric function in `src/tinybench/metrics/` (e.g. `src/tinybench/metrics/rouge.py` or `f1.py`).
2. Expose the metric in `src/tinybench/metrics/__init__.py`.
3. Integrate the metric into `BenchmarkRunner._evaluate_single_task` or summary aggregation.
4. Write unit tests in `tests/test_metrics.py`.
