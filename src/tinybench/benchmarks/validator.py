"""Validation engine for benchmark dataset integrity and health."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError

from tinybench.benchmarks.schema import BenchmarkItem


class DatasetValidationReport(BaseModel):
    """Detailed summary of dataset validation results."""
    file_path: str = Field(..., description="Path to evaluated dataset.")
    is_valid: bool = Field(..., description="True if dataset conforms to all strict schema requirements.")
    total_items: int = Field(default=0, description="Total valid items found.")
    category_counts: dict[str, int] = Field(default_factory=dict, description="Item distribution across categories.")
    scoring_method_counts: dict[str, int] = Field(default_factory=dict, description="Distribution across scoring methods.")
    errors: list[str] = Field(default_factory=list, description="List of fatal validation errors.")
    warnings: list[str] = Field(default_factory=list, description="Non-fatal warnings regarding balance or structure.")


def validate_dataset_file(file_path: str | Path) -> DatasetValidationReport:
    """Perform comprehensive validation on a dataset file.

    Args:
        file_path: Path to .jsonl file to validate.

    Returns:
        DatasetValidationReport with details on errors, warnings, and item breakdown.
    """
    path = Path(file_path)
    errors: list[str] = []
    warnings: list[str] = []
    category_counter: Counter[str] = Counter()
    scoring_counter: Counter[str] = Counter()
    seen_ids: set[str] = set()
    total_valid = 0

    if not path.exists():
        return DatasetValidationReport(
            file_path=str(path),
            is_valid=False,
            errors=[f"File does not exist: {path}"],
        )

    if not path.is_file():
        return DatasetValidationReport(
            file_path=str(path),
            is_valid=False,
            errors=[f"Path is not a regular file: {path}"],
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                raw = line.strip()
                if not raw or raw.startswith("#"):
                    continue

                try:
                    data = json.loads(raw)
                except json.JSONDecodeError as err:
                    errors.append(f"Line {line_idx}: Invalid JSON syntax ({err})")
                    continue

                try:
                    item = BenchmarkItem.model_validate(data)
                except ValidationError as err:
                    errors.append(f"Line {line_idx}: Schema error ({err.errors()[0]['msg']})")
                    continue

                if item.task_id in seen_ids:
                    errors.append(f"Line {line_idx}: Duplicate task_id '{item.task_id}'")
                seen_ids.add(item.task_id)

                category_counter[item.category] += 1
                scoring_counter[item.scoring_method] += 1
                total_valid += 1

    except Exception as exc:
        errors.append(f"Read error: {exc}")

    if total_valid == 0 and not errors:
        errors.append("Dataset file contains 0 benchmark examples.")

    # Check for category balance warnings
    for cat, count in category_counter.items():
        if count < 3:
            warnings.append(f"Category '{cat}' has only {count} examples (recommend at least 3 for statistical relevance).")

    return DatasetValidationReport(
        file_path=str(path),
        is_valid=len(errors) == 0,
        total_items=total_valid,
        category_counts=dict(category_counter),
        scoring_method_counts=dict(scoring_counter),
        errors=errors,
        warnings=warnings,
    )
