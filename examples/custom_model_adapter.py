"""Example: Implementing and registering a custom model adapter."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tinybench.config import BenchmarkConfig, GenerationConfig
from tinybench.models.base import BaseModelAdapter, GenerationResult, ModelMetadata
from tinybench.reporting import format_markdown_summary
from tinybench.runner import BenchmarkRunner


class RuleBasedRegexAdapter(BaseModelAdapter):
    """A custom adapter implementing basic pattern matching rules."""

    def __init__(self, model_id: str = "custom-regex") -> None:
        super().__init__(model_id=model_id)

    def load(self) -> float:
        t0 = time.perf_counter()
        # Custom loading / setup logic here
        self._is_loaded = True
        return round(time.perf_counter() - t0, 4)

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_id=self.model_id,
            model_name="Custom Regex Heuristic Model",
            family="Heuristic",
            parameters="0M",
            license="MIT",
            license_is_osi_approved=True,
            estimated_ram_mb=10.0,
            backend_type="custom_heuristic",
            description="Demonstration of a custom model adapter registered by an open-source contributor.",
        )

    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        t0 = time.perf_counter()
        p = prompt.lower()
        if "capital" in p and "france" in p:
            reply = "Paris"
        elif "speed of light" in p:
            reply = "A"
        elif "15 apples" in p:
            reply = "42"
        else:
            reply = "I don't know."

        return GenerationResult(
            text=reply,
            prompt_tokens=len(prompt.split()),
            generated_tokens=len(reply.split()),
            latency_seconds=round(time.perf_counter() - t0, 4),
        )

    def unload(self) -> None:
        self._is_loaded = False


def main():
    print("Testing Custom Model Adapter...")
    adapter = RuleBasedRegexAdapter()
    adapter.load()
    meta = adapter.get_metadata()
    print(f"Loaded: {meta.model_name} (RAM requirement: {meta.estimated_ram_mb} MB)")

    res = adapter.generate("What is the capital city of France?", GenerationConfig())
    print(f"Generated: '{res.text}' in {res.latency_seconds:.4f}s")
    adapter.unload()


if __name__ == "__main__":
    main()
