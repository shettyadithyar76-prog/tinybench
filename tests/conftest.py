"""Pytest fixtures for TinyBench testing suite."""

import json
import pytest
from pathlib import Path
from typing import Generator

from tinybench.benchmarks.schema import BenchmarkItem
from tinybench.config import BenchmarkConfig, GenerationConfig, SystemInfo
from tinybench.metrics.latency import LatencySummary
from tinybench.metrics.memory import MemoryMeasurement
from tinybench.models.base import ModelMetadata
from tinybench.runner import CategoryMetrics, RunResult, TaskResult


@pytest.fixture
def sample_items() -> list[BenchmarkItem]:
    """Provide a list of diverse in-memory BenchmarkItems."""
    return [
        BenchmarkItem(
            task_id="test_mcq_1",
            category="multiple_choice",
            prompt="What is 1+1?\nA) 2\nB) 3\nAnswer with letter.",
            expected_answer="A",
            scoring_method="multiple_choice",
            metadata={"options": ["A", "B"]},
        ),
        BenchmarkItem(
            task_id="test_fact_1",
            category="factual_qa",
            prompt="What is the capital of Italy?",
            expected_answer="Rome",
            scoring_method="exact_match",
            metadata={"aliases": ["rome"]},
        ),
        BenchmarkItem(
            task_id="test_math_1",
            category="arithmetic_reasoning",
            prompt="Calculate 10 * 5.",
            expected_answer=50,
            scoring_method="numeric",
            metadata={"tolerance": 0.0},
        ),
        BenchmarkItem(
            task_id="test_inst_1",
            category="instruction_following",
            prompt="Say hello in exactly 2 words.",
            expected_answer="hello world",
            scoring_method="instruction_compliance",
            metadata={"exact_word_count": 2, "required_keywords": ["hello"]},
        ),
        BenchmarkItem(
            task_id="test_struct_1",
            category="structured_output",
            prompt="Output JSON with key name.",
            expected_answer={"name": "test"},
            scoring_method="json_schema",
            metadata={"required_keys": ["name"]},
        ),
    ]


@pytest.fixture
def sample_jsonl_file(tmp_path: Path, sample_items: list[BenchmarkItem]) -> Path:
    """Create a temporary valid .jsonl dataset file."""
    fpath = tmp_path / "test_dataset.jsonl"
    with open(fpath, "w", encoding="utf-8") as f:
        for item in sample_items:
            f.write(json.dumps(item.model_dump()) + "\n")
    return fpath


@pytest.fixture
def sample_run_result(sample_items: list[BenchmarkItem]) -> RunResult:
    """Provide a realistic mock RunResult."""
    meta = ModelMetadata(
        model_id="mock",
        model_name="Mock Model",
        family="Mock",
        parameters="0M",
        license="MIT",
        license_is_osi_approved=True,
        estimated_ram_mb=10.0,
        backend_type="mock",
        revision="v1",
        description="Test mock",
    )
    task_res = [
        TaskResult(
            task_id=it.task_id,
            category=it.category,
            prompt=it.prompt,
            expected_answer=it.expected_answer,
            generated_text=str(it.expected_answer),
            is_correct=True,
            exact_match=True,
            compliance_score=1.0,
            latency_seconds=0.01,
        )
        for it in sample_items
    ]
    lat = LatencySummary(
        init_latency_seconds=0.05,
        first_generation_latency_seconds=0.01,
        mean_latency_seconds=0.01,
        median_latency_seconds=0.01,
        p95_latency_seconds=0.01,
        min_latency_seconds=0.01,
        max_latency_seconds=0.01,
        total_generation_seconds=0.05,
        successful_count=5,
        failed_count=0,
    )
    mem = MemoryMeasurement(
        initial_rss_mb=50.0,
        peak_rss_mb=55.0,
        final_rss_mb=52.0,
        memory_delta_mb=5.0,
    )
    sys_info = SystemInfo.capture()
    return RunResult(
        run_id="run_test_12345",
        timestamp="2026-10-24T12:00:00Z",
        model_metadata=meta,
        dataset_name="test_dataset.jsonl",
        dataset_path="datasets/test_dataset.jsonl",
        generation_config=GenerationConfig(),
        system_info=sys_info,
        total_examples=5,
        successful_examples=5,
        failed_examples=0,
        accuracy=100.0,
        exact_match_rate=100.0,
        avg_compliance_score=1.0,
        latency_summary=lat,
        memory_measurement=mem,
        category_metrics={
            "multiple_choice": CategoryMetrics(
                category="multiple_choice",
                total=1,
                correct=1,
                accuracy=100.0,
                exact_match_rate=100.0,
                avg_compliance_score=1.0,
                mean_latency_seconds=0.01,
            )
        },
        task_results=task_res,
    )
