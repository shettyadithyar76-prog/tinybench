"""Optional integration test for real local model execution on CPU."""

import os
from pathlib import Path
import pytest

from tinybench.config import BenchmarkConfig, GenerationConfig
from tinybench.runner import BenchmarkRunner

RUN_REAL_TESTS = os.environ.get("RUN_REAL_MODEL_TESTS", "0") == "1"


@pytest.mark.skipif(
    not RUN_REAL_TESTS,
    reason="Real model integration tests require RUN_REAL_MODEL_TESTS=1 and network/weights.",
)
def test_real_model_inference_cpu(tmp_path: Path):
    """Optional smoke test evaluating SmolLM2-135M on CPU with 1 example."""
    test_ds = tmp_path / "single_task.jsonl"
    test_ds.write_text(
        '{"task_id": "real_1", "category": "factual_qa", "prompt": "What is the capital of France?", "expected_answer": "Paris", "scoring_method": "exact_match"}\n',
        encoding="utf-8",
    )

    cfg = BenchmarkConfig(
        dataset_path=str(test_ds),
        model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
        limit=1,
        generation_config=GenerationConfig(max_new_tokens=16, temperature=0.0),
        output_dir=str(tmp_path / "runs"),
    )
    runner = BenchmarkRunner(cfg)
    result = runner.run()

    assert result.total_examples == 1
    assert result.successful_examples == 1
    assert result.task_results[0].error is None
    assert len(result.task_results[0].generated_text) > 0
