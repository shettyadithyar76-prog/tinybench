"""Unit tests for dataset schema, loading, and validation."""

import json
from pathlib import Path
import pytest

from tinybench.benchmarks.loader import load_dataset
from tinybench.benchmarks.schema import BenchmarkItem
from tinybench.benchmarks.validator import validate_dataset_file
from tinybench.exceptions import DatasetValidationError


def test_load_valid_dataset(sample_jsonl_file: Path, sample_items: list[BenchmarkItem]):
    """Verify loading from a properly formatted JSONL file."""
    items = load_dataset(sample_jsonl_file)
    assert len(items) == len(sample_items)
    assert items[0].task_id == "test_mcq_1"
    assert items[0].category == "multiple_choice"


def test_load_nonexistent_file(tmp_path: Path):
    """Verify DatasetValidationError when target file does not exist."""
    with pytest.raises(DatasetValidationError, match="not found"):
        load_dataset(tmp_path / "does_not_exist.jsonl")


def test_load_corrupt_json(tmp_path: Path):
    """Verify error on malformed JSON line."""
    bad_file = tmp_path / "corrupt.jsonl"
    bad_file.write_text('{this is not valid json}\n', encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="JSON syntax error"):
        load_dataset(bad_file)


def test_load_schema_violation(tmp_path: Path):
    """Verify validation error when required schema fields are missing."""
    bad_file = tmp_path / "invalid_schema.jsonl"
    # Missing prompt and expected_answer
    bad_file.write_text('{"task_id": "test_1", "category": "factual_qa"}\n', encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="Validation error"):
        load_dataset(bad_file)


def test_load_duplicate_task_id(tmp_path: Path):
    """Verify duplicate task_id detection."""
    dup_file = tmp_path / "duplicate.jsonl"
    line = json.dumps({
        "task_id": "dup_1",
        "category": "factual_qa",
        "prompt": "Test prompt",
        "expected_answer": "42",
        "scoring_method": "exact_match",
    })
    dup_file.write_text(f"{line}\n{line}\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="Duplicate task_id"):
        load_dataset(dup_file)


def test_category_filtering(sample_jsonl_file: Path):
    """Verify filtering tasks by category."""
    items = load_dataset(sample_jsonl_file, categories=["multiple_choice", "factual_qa"])
    assert len(items) == 2
    cats = {item.category for item in items}
    assert cats == {"multiple_choice", "factual_qa"}


def test_limit_parameter(sample_jsonl_file: Path):
    """Verify limiting max loaded examples."""
    items = load_dataset(sample_jsonl_file, limit=3)
    assert len(items) == 3


def test_validate_dataset_file_report(sample_jsonl_file: Path):
    """Verify DatasetValidationReport structure and metrics."""
    report = validate_dataset_file(sample_jsonl_file)
    assert report.is_valid is True
    assert report.total_items == 5
    assert len(report.errors) == 0
    assert "multiple_choice" in report.category_counts


def test_actual_basic_v1_dataset():
    """Verify the real bundled datasets/basic_v1.jsonl is 100% valid."""
    path = Path("datasets/basic_v1.jsonl")
    assert path.is_file(), "Bundled basic_v1.jsonl must exist"
    report = validate_dataset_file(path)
    assert report.is_valid is True, f"Dataset errors: {report.errors}"
    assert report.total_items >= 30, "Benchmark must have >= 30 items"
    # Verify all 5 categories are present
    expected_cats = {
        "multiple_choice",
        "factual_qa",
        "arithmetic_reasoning",
        "instruction_following",
        "structured_output",
    }
    assert set(report.category_counts.keys()) == expected_cats
