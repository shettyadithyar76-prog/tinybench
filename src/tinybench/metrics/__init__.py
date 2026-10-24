"""Evaluation metrics for accuracy, exact match, compliance, latency, and memory."""

from tinybench.metrics.exact_match import (
    compute_exact_match,
    normalize_text,
)
from tinybench.metrics.accuracy import (
    calculate_accuracy,
    compute_multiple_choice,
    compute_numeric_match,
    extract_mcq_choice,
    extract_numbers,
)
from tinybench.metrics.compliance import (
    evaluate_instruction_compliance,
    evaluate_json_schema_compliance,
    extract_json_from_text,
)
from tinybench.metrics.latency import (
    LatencySummary,
    compute_latency_summary,
)
from tinybench.metrics.memory import (
    MemoryMeasurement,
    MemoryTracker,
)

__all__ = [
    "compute_exact_match",
    "normalize_text",
    "calculate_accuracy",
    "compute_multiple_choice",
    "compute_numeric_match",
    "extract_mcq_choice",
    "extract_numbers",
    "evaluate_instruction_compliance",
    "evaluate_json_schema_compliance",
    "extract_json_from_text",
    "LatencySummary",
    "compute_latency_summary",
    "MemoryMeasurement",
    "MemoryTracker",
]
