"""Example: Evaluating a real Hugging Face open-weight model on CPU."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tinybench.config import BenchmarkConfig, GenerationConfig
from tinybench.models.transformers_backend import TransformersModelAdapter
from tinybench.reporting import format_markdown_summary
from tinybench.runner import BenchmarkRunner


def main():
    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    print(f"Preparing CPU evaluation for: {model_name}")

    # Inspect estimated RAM before starting
    adapter = TransformersModelAdapter(model_name)
    meta = adapter.get_metadata()
    print(f"Model: {meta.model_name}")
    print(f"Parameters: {meta.parameters}")
    print(f"Estimated RAM: {meta.estimated_ram_mb:.0f} MB")
    print(f"License: {meta.license}")

    # Configure a 3-example smoke test
    config = BenchmarkConfig(
        dataset_path="datasets/basic_v1.jsonl",
        model_name=model_name,
        limit=3,
        generation_config=GenerationConfig(
            max_new_tokens=32,
            temperature=0.0,
        ),
    )

    runner = BenchmarkRunner(config)
    try:
        result = runner.run()
        print("\n" + format_markdown_summary(result))
    except Exception as exc:
        print(f"Evaluation error: {exc}")
        print("Note: Run with network access to download model weights if not cached.")


if __name__ == "__main__":
    main()
