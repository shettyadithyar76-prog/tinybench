"""Example: Running an end-to-end evaluation with the mock backend."""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tinybench.config import BenchmarkConfig, GenerationConfig
from tinybench.reporting import format_markdown_summary
from tinybench.runner import BenchmarkRunner


def main():
    print("Running TinyBench Mock Evaluation Example...")

    # Configure benchmark
    config = BenchmarkConfig(
        dataset_path="datasets/basic_v1.jsonl",
        model_name="mock",
        limit=10,
        generation_config=GenerationConfig(
            max_new_tokens=32,
            temperature=0.0,
            seed=42,
        ),
    )

    runner = BenchmarkRunner(config)

    def progress_callback(current, total, task_result):
        status = "PASSED" if task_result.is_correct else "FAILED"
        print(f"[{current:02d}/{total:02d}] {task_result.task_id} ({task_result.category}): {status} ({task_result.latency_seconds:.3f}s)")

    result = runner.run(progress_callback=progress_callback)

    # Print markdown summary
    print("\n" + format_markdown_summary(result))


if __name__ == "__main__":
    main()
