"""Example: Comparing multiple model evaluations programmatically."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tinybench.config import BenchmarkConfig
from tinybench.reporting import generate_comparison_table
from tinybench.runner import BenchmarkRunner


def main():
    models_to_compare = ["mock", "mock-noisy"]
    results = []

    for model_name in models_to_compare:
        print(f"Benchmarking model: {model_name}...")
        config = BenchmarkConfig(
            dataset_path="datasets/basic_v1.jsonl",
            model_name=model_name,
            limit=10,
        )
        runner = BenchmarkRunner(config)
        res = runner.run()
        results.append(res)

    print("\nComparison Results:")
    table = generate_comparison_table(results)
    fmt = "{:<25} {:<10} {:<12} {:<14} {:<14}"
    print(fmt.format("Model", "Examples", "Accuracy", "Exact Match", "Median Latency"))
    print("-" * 75)
    for row in table:
        print(fmt.format(
            row["Model Name"],
            str(row["Total"]),
            f"{row['Accuracy (%)']:.1f}%",
            f"{row['Exact Match (%)']:.1f}%",
            f"{row['Median Latency (s)']:.3f} s",
        ))


if __name__ == "__main__":
    main()
