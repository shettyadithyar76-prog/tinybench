"""Unit tests for benchmark runner pipeline and execution logic."""

from pathlib import Path
import pytest

from tinybench.config import BenchmarkConfig, GenerationConfig
from tinybench.runner import BenchmarkRunner, TaskResult


def test_runner_end_to_end(sample_jsonl_file: Path):
    """Run full benchmark execution with mock model."""
    cfg = BenchmarkConfig(
        dataset_path=str(sample_jsonl_file),
        model_name="mock",
        generation_config=GenerationConfig(seed=42),
    )
    runner = BenchmarkRunner(cfg)

    progress_events: list[tuple[int, int]] = []

    def on_prog(curr: int, total: int, res: TaskResult):
        progress_events.append((curr, total))
        assert isinstance(res, TaskResult)

    result = runner.run(progress_callback=on_prog)

    # Verify execution counts
    assert len(progress_events) == 5
    assert result.total_examples == 5
    assert result.successful_examples == 5
    assert result.failed_examples == 0
    assert result.accuracy > 0.0

    # Verify metadata captured
    assert result.model_metadata.model_id == "mock"
    assert result.system_info.processor is not None
    assert result.latency_summary.mean_latency_seconds >= 0.0
    assert result.memory_measurement.peak_rss_mb > 0.0

    # Verify category metrics
    assert "multiple_choice" in result.category_metrics
    assert result.category_metrics["multiple_choice"].total == 1


def test_runner_with_failing_model(tmp_path: Path):
    """Ensure runner handles task failure gracefully without crashing pipeline."""
    bad_dataset = tmp_path / "fail_dataset.jsonl"
    bad_dataset.write_text(
        '{"task_id": "err_1", "category": "factual_qa", "prompt": "error please", "expected_answer": "42", "scoring_method": "exact_match"}\n'
        '{"task_id": "ok_1", "category": "factual_qa", "prompt": "normal question", "expected_answer": "42", "scoring_method": "exact_match"}\n',
        encoding="utf-8",
    )

    cfg = BenchmarkConfig(
        dataset_path=str(bad_dataset),
        model_name="mock-failing",
    )
    runner = BenchmarkRunner(cfg)
    result = runner.run()

    assert result.total_examples == 2
    assert result.failed_examples == 1
    assert result.successful_examples == 1
    # Check that error is recorded on task result
    assert result.task_results[0].error is not None
    assert result.task_results[1].error is None


def test_runner_reproducibility(sample_jsonl_file: Path):
    """Verify identical seeds produce identical results on mock backend."""
    cfg1 = BenchmarkConfig(
        dataset_path=str(sample_jsonl_file),
        model_name="mock-noisy",
        generation_config=GenerationConfig(seed=123),
    )
    cfg2 = BenchmarkConfig(
        dataset_path=str(sample_jsonl_file),
        model_name="mock-noisy",
        generation_config=GenerationConfig(seed=123),
    )

    res1 = BenchmarkRunner(cfg1).run()
    res2 = BenchmarkRunner(cfg2).run()

    for t1, t2 in zip(res1.task_results, res2.task_results):
        assert t1.generated_text == t2.generated_text
        assert t1.is_correct == t2.is_correct
