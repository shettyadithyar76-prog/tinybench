"""Exporting, reporting, and comparison tools for benchmark results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from tinybench.runner import RunResult


def export_run_json(result: RunResult, output_path: str | Path) -> Path:
    """Export complete RunResult schema to a JSON file.

    Args:
        result: The evaluation run result.
        output_path: Target file path.

    Returns:
        Resolved Path to written file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, indent=2)
    return path


def export_run_csv(
    result: RunResult,
    output_path: str | Path,
    detail_level: str = "tasks",
) -> Path:
    """Export benchmark result to CSV.

    Args:
        result: RunResult object.
        output_path: Destination path.
        detail_level: Either 'tasks' (per-task breakdown) or 'summary' (one-row run summary).

    Returns:
        Resolved Path to written CSV file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        if detail_level == "summary":
            writer = csv.writer(f)
            writer.writerow([
                "run_id",
                "timestamp",
                "model_id",
                "model_name",
                "family",
                "dataset_name",
                "total_examples",
                "successful_examples",
                "failed_examples",
                "accuracy_percent",
                "exact_match_percent",
                "avg_compliance",
                "init_latency_s",
                "median_latency_s",
                "mean_latency_s",
                "p95_latency_s",
                "peak_ram_mb",
                "memory_delta_mb",
                "os",
                "python_version",
            ])
            writer.writerow([
                result.run_id,
                result.timestamp,
                result.model_metadata.model_id,
                result.model_metadata.model_name,
                result.model_metadata.family,
                result.dataset_name,
                result.total_examples,
                result.successful_examples,
                result.failed_examples,
                result.accuracy,
                result.exact_match_rate,
                result.avg_compliance_score,
                result.latency_summary.init_latency_seconds,
                result.latency_summary.median_latency_seconds,
                result.latency_summary.mean_latency_seconds,
                result.latency_summary.p95_latency_seconds,
                result.memory_measurement.peak_rss_mb,
                result.memory_measurement.memory_delta_mb,
                f"{result.system_info.os_name} {result.system_info.architecture}",
                result.system_info.python_version,
            ])
        else:
            # Per-task breakdown
            writer = csv.writer(f)
            writer.writerow([
                "task_id",
                "category",
                "is_correct",
                "exact_match",
                "compliance_score",
                "latency_seconds",
                "prompt_tokens",
                "generated_tokens",
                "prompt",
                "expected_answer",
                "generated_text",
                "error",
            ])
            for t in result.task_results:
                writer.writerow([
                    t.task_id,
                    t.category,
                    t.is_correct,
                    t.exact_match,
                    t.compliance_score,
                    t.latency_seconds,
                    t.prompt_tokens or "",
                    t.generated_tokens or "",
                    t.prompt,
                    str(t.expected_answer),
                    t.generated_text,
                    t.error or "",
                ])

    return path


def generate_comparison_table(results: list[RunResult]) -> list[dict[str, Any]]:
    """Produce standardized comparison records across multiple runs."""
    rows: list[dict[str, Any]] = []
    for r in results:
        rows.append({
            "Run ID": r.run_id,
            "Model Name": r.model_metadata.model_name,
            "Model ID": r.model_metadata.model_id,
            "Dataset": r.dataset_name,
            "Total": r.total_examples,
            "Accuracy (%)": r.accuracy,
            "Exact Match (%)": r.exact_match_rate,
            "Compliance": r.avg_compliance_score,
            "Median Latency (s)": r.latency_summary.median_latency_seconds,
            "Mean Latency (s)": r.latency_summary.mean_latency_seconds,
            "Init Latency (s)": r.latency_summary.init_latency_seconds,
            "Peak RAM (MB)": r.memory_measurement.peak_rss_mb,
            "RAM Delta (MB)": r.memory_measurement.memory_delta_mb,
            "Timestamp": r.timestamp,
        })
    return rows


def format_markdown_summary(result: RunResult) -> str:
    """Format an evaluation run into a clean markdown table summary."""
    lines: list[str] = [
        f"# TinyBench Evaluation Summary — {result.model_metadata.model_name}",
        f"- **Run ID**: `{result.run_id}`",
        f"- **Timestamp**: {result.timestamp}",
        f"- **Dataset**: `{result.dataset_name}` ({result.total_examples} examples)",
        f"- **Platform**: {result.system_info.os_name} ({result.system_info.processor}) | Python {result.system_info.python_version}",
        "",
        "## Overall Metrics",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| **Accuracy** | {result.accuracy:.2f}% |",
        f"| **Exact Match** | {result.exact_match_rate:.2f}% |",
        f"| **Instruction Compliance** | {result.avg_compliance_score:.4f} |",
        f"| **Median Latency** | {result.latency_summary.median_latency_seconds:.3f} s |",
        f"| **Mean Latency** | {result.latency_summary.mean_latency_seconds:.3f} s |",
        f"| **Init Latency** | {result.latency_summary.init_latency_seconds:.3f} s |",
        f"| **Peak Process RSS** | {result.memory_measurement.peak_rss_mb:.1f} MB |",
        f"| **Memory Delta** | {result.memory_measurement.memory_delta_mb:.1f} MB |",
        f"| **Successful / Failed** | {result.successful_examples} / {result.failed_examples} |",
        "",
        "## Category Breakdown",
        "| Category | Count | Accuracy | Exact Match | Compliance | Mean Latency |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for cat_name, cm in sorted(result.category_metrics.items()):
        lines.append(
            f"| `{cat_name}` | {cm.total} | {cm.accuracy:.1f}% | {cm.exact_match_rate:.1f}% | {cm.avg_compliance_score:.2f} | {cm.mean_latency_seconds:.3f} s |"
        )

    return "\n".join(lines)
