"""Unit tests for reporting and export tools."""

import csv
import json
from pathlib import Path
import pytest

from tinybench.reporting import (
    export_run_csv,
    export_run_json,
    format_markdown_summary,
    generate_comparison_table,
)
from tinybench.runner import RunResult


def test_export_run_json(tmp_path: Path, sample_run_result: RunResult):
    out_file = tmp_path / "result.json"
    p = export_run_json(sample_run_result, out_file)
    assert p.is_file()

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["run_id"] == sample_run_result.run_id
    # Ensure it deserializes cleanly into RunResult
    reloaded = RunResult.model_validate(data)
    assert reloaded.run_id == sample_run_result.run_id


def test_export_run_csv_tasks_level(tmp_path: Path, sample_run_result: RunResult):
    out_file = tmp_path / "result_tasks.csv"
    p = export_run_csv(sample_run_result, out_file, detail_level="tasks")
    assert p.is_file()

    with open(p, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
    # Header + 5 tasks
    assert len(reader) == 6
    assert reader[0][0] == "task_id"
    assert reader[1][0] == "test_mcq_1"


def test_export_run_csv_summary_level(tmp_path: Path, sample_run_result: RunResult):
    out_file = tmp_path / "result_summary.csv"
    p = export_run_csv(sample_run_result, out_file, detail_level="summary")
    assert p.is_file()

    with open(p, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
    # Header + 1 summary row
    assert len(reader) == 2
    assert reader[0][0] == "run_id"
    assert reader[1][0] == sample_run_result.run_id


def test_generate_comparison_table(sample_run_result: RunResult):
    rows = generate_comparison_table([sample_run_result])
    assert len(rows) == 1
    assert rows[0]["Run ID"] == sample_run_result.run_id
    assert rows[0]["Accuracy (%)"] == 100.0


def test_format_markdown_summary(sample_run_result: RunResult):
    md = format_markdown_summary(sample_run_result)
    assert "# TinyBench Evaluation Summary" in md
    assert "Accuracy" in md
    assert "Category Breakdown" in md
