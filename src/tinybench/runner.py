"""Benchmark execution runner and evaluation pipeline."""

from __future__ import annotations

import datetime
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Callable
from pydantic import BaseModel, Field

from tinybench.benchmarks.loader import load_dataset
from tinybench.benchmarks.schema import BenchmarkItem, ScoringMethod
from tinybench.config import BenchmarkConfig, GenerationConfig, SystemInfo
from tinybench.metrics.accuracy import (
    calculate_accuracy,
    compute_multiple_choice,
    compute_numeric_match,
)
from tinybench.metrics.compliance import (
    evaluate_instruction_compliance,
    evaluate_json_schema_compliance,
)
from tinybench.metrics.exact_match import compute_exact_match
from tinybench.metrics.latency import LatencySummary, compute_latency_summary
from tinybench.metrics.memory import MemoryMeasurement, MemoryTracker
from tinybench.models.base import BaseModelAdapter, ModelMetadata
from tinybench.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


class TaskResult(BaseModel):
    """Execution and evaluation result for a single benchmark task."""
    task_id: str = Field(..., description="Task identifier.")
    category: str = Field(..., description="Benchmark category.")
    prompt: str = Field(..., description="Prompt presented to model.")
    expected_answer: Any = Field(..., description="Canonical target answer.")
    generated_text: str = Field(..., description="Raw model completion.")
    thinking_text: str | None = Field(default=None, description="Internal reasoning tokens / chain-of-thought.")
    is_correct: bool = Field(default=False, description="Whether answer satisfies target accuracy criteria.")
    exact_match: bool = Field(default=False, description="Normalized string equality check.")
    compliance_score: float = Field(default=1.0, description="Instruction compliance score (0.0 to 1.0).")
    compliance_details: dict[str, bool] = Field(default_factory=dict, description="Outcomes of constraint checks.")
    latency_seconds: float = Field(default=0.0, description="Task generation duration in seconds.")
    prompt_tokens: int | None = Field(default=None, description="Input tokens counted.")
    generated_tokens: int | None = Field(default=None, description="Output tokens counted.")
    error: str | None = Field(default=None, description="Error message if inference or scoring failed.")


class CategoryMetrics(BaseModel):
    """Aggregate metrics breakdown for a specific category."""
    category: str = Field(..., description="Benchmark category name.")
    total: int = Field(default=0, description="Total tasks evaluated in this category.")
    correct: int = Field(default=0, description="Number of correct tasks.")
    accuracy: float = Field(default=0.0, description="Accuracy percentage.")
    exact_match_rate: float = Field(default=0.0, description="Exact match percentage.")
    avg_compliance_score: float = Field(default=1.0, description="Average compliance score.")
    mean_latency_seconds: float = Field(default=0.0, description="Mean task latency in seconds.")


class RunResult(BaseModel):
    """Complete evaluation report and raw artifacts for a benchmark run."""
    run_id: str = Field(..., description="Unique run identifier.")
    timestamp: str = Field(..., description="ISO 8601 execution timestamp.")
    model_metadata: ModelMetadata = Field(..., description="Profile of tested model.")
    dataset_name: str = Field(..., description="Dataset filename.")
    dataset_path: str = Field(..., description="Filesystem path to source dataset.")
    generation_config: GenerationConfig = Field(..., description="Generation parameters.")
    system_info: SystemInfo = Field(..., description="Hardware and software platform info.")
    total_examples: int = Field(default=0, description="Total examples attempted.")
    successful_examples: int = Field(default=0, description="Examples generated without fatal crash.")
    failed_examples: int = Field(default=0, description="Failed examples.")
    accuracy: float = Field(default=0.0, description="Overall accuracy percentage.")
    exact_match_rate: float = Field(default=0.0, description="Overall exact match percentage.")
    avg_compliance_score: float = Field(default=1.0, description="Mean instruction compliance score.")
    latency_summary: LatencySummary = Field(..., description="Latency profile.")
    memory_measurement: MemoryMeasurement = Field(..., description="Process memory consumption.")
    category_metrics: dict[str, CategoryMetrics] = Field(default_factory=dict, description="Breakdown by category.")
    task_results: list[TaskResult] = Field(default_factory=list, description="Detailed per-task outcomes.")
    is_demo: bool = Field(default=False, description="Whether this run was executed with a mock/simulated model.")
    runtime_version: str | None = Field(default=None, description="Inference runtime engine version.")
    provenance: dict[str, Any] = Field(default_factory=dict, description="Execution provenance and environment metadata.")


class BenchmarkRunner:
    """Executes a benchmark configuration against a model adapter."""

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config

    def run(
        self,
        progress_callback: Callable[[int, int, TaskResult], None] | None = None,
    ) -> RunResult:
        """Run the evaluation suite end-to-end.

        Args:
            progress_callback: Optional callable receiving (current_index, total_count, latest_task_result).

        Returns:
            Structured RunResult containing full metrics, per-task outputs, and system metadata.
        """
        run_id = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        dataset_path = Path(self.config.dataset_path)

        # 1. Load benchmark dataset
        items = load_dataset(
            file_path=dataset_path,
            categories=self.config.categories,
            limit=self.config.limit,
        )

        # 2. Instantiate model backend
        adapter = ModelRegistry.get_adapter(
            name_or_id=self.config.model_name,
            revision=self.config.model_revision,
        )
        model_meta = adapter.get_metadata()

        # 3. Initialize memory and latency tracking
        mem_tracker = MemoryTracker()
        mem_tracker.start()

        init_duration = 0.0
        try:
            init_duration = adapter.load()
        except Exception as exc:
            logger.error(f"Failed to load model {self.config.model_name}: {exc}")
            raise

        mem_tracker.sample()

        task_results: list[TaskResult] = []
        latencies: list[float] = []
        failed_count = 0

        total_items = len(items)

        # 4. Iterate over benchmark tasks
        for idx, item in enumerate(items, start=1):
            t_res = self._evaluate_single_task(adapter, item)
            task_results.append(t_res)

            if t_res.error:
                failed_count += 1
            else:
                latencies.append(t_res.latency_seconds)

            mem_tracker.sample()

            if progress_callback:
                progress_callback(idx, total_items, t_res)

        # 5. Unload model to free memory
        adapter.unload()
        mem_measurement = mem_tracker.finish()

        # 6. Compute aggregate latency
        lat_summary = compute_latency_summary(
            latencies=latencies,
            init_time=init_duration,
            failed_count=failed_count,
        )

        # 7. Compute aggregate quality scores
        overall_accuracy = calculate_accuracy([r.is_correct for r in task_results])
        overall_em = calculate_accuracy([r.exact_match for r in task_results])

        compliance_scores = [r.compliance_score for r in task_results]
        avg_compliance = (
            round(sum(compliance_scores) / len(compliance_scores), 4)
            if compliance_scores
            else 1.0
        )

        # 8. Compute per-category breakdown
        cat_metrics = self._compute_category_breakdown(task_results)

        # 9. Capture system info & execution provenance
        sys_info = SystemInfo.capture()
        is_demo = "mock" in adapter.model_id.lower() or model_meta.backend_type == "mock"

        runtime_ver = "mock"
        if model_meta.backend_type == "transformers_cpu":
            try:
                import transformers
                runtime_ver = f"transformers {transformers.__version__}"
            except Exception:
                runtime_ver = "transformers"
        elif model_meta.backend_type == "ollama":
            from tinybench.models.ollama_backend import check_ollama_daemon
            _, msg = check_ollama_daemon()
            runtime_ver = msg

        provenance = {
            "model_id": model_meta.model_id,
            "model_name": model_meta.model_name,
            "developer": getattr(model_meta, "developer", "Not verified"),
            "parameters": model_meta.parameters,
            "variant": getattr(model_meta, "variant", "Instruct"),
            "quantization": getattr(model_meta, "quantization", "Not verified"),
            "weight_formats": getattr(model_meta, "weight_formats", []),
            "runtime": model_meta.backend_type,
            "runtime_version": runtime_ver,
            "os": f"{sys_info.os_name} ({sys_info.architecture})",
            "cpu": sys_info.processor,
            "cores": f"{sys_info.cpu_physical_cores} Physical / {sys_info.cpu_logical_cores} Threads",
            "dataset": dataset_path.name,
            "max_new_tokens": self.config.generation_config.max_new_tokens,
            "temperature": self.config.generation_config.temperature,
            "is_demo": is_demo,
        }

        result = RunResult(
            run_id=run_id,
            timestamp=timestamp,
            model_metadata=model_meta,
            dataset_name=dataset_path.name,
            dataset_path=str(dataset_path),
            generation_config=self.config.generation_config,
            system_info=sys_info,
            total_examples=len(task_results),
            successful_examples=len(task_results) - failed_count,
            failed_examples=failed_count,
            accuracy=overall_accuracy,
            exact_match_rate=overall_em,
            avg_compliance_score=avg_compliance,
            latency_summary=lat_summary,
            memory_measurement=mem_measurement,
            category_metrics=cat_metrics,
            task_results=task_results,
            is_demo=is_demo,
            runtime_version=runtime_ver,
            provenance=provenance,
        )

        return result

    def _evaluate_single_task(
        self,
        adapter: BaseModelAdapter,
        item: BenchmarkItem,
    ) -> TaskResult:
        """Run model inference on one item and grade output."""
        thinking_text = None
        try:
            gen_res = adapter.generate(item.prompt, self.config.generation_config)
            output_text = gen_res.text
            latency = gen_res.latency_seconds
            p_tokens = gen_res.prompt_tokens
            g_tokens = gen_res.generated_tokens
            if hasattr(gen_res, "metadata") and isinstance(gen_res.metadata, dict):
                thinking_text = gen_res.metadata.get("thinking")
            error_msg = None
        except Exception as exc:
            logger.warning(f"Error evaluating task {item.task_id}: {exc}")
            output_text = ""
            latency = 0.0
            p_tokens = None
            g_tokens = None
            error_msg = str(exc)

        # Compute metrics based on scoring method
        is_correct = False
        exact_match = False
        compliance_score = 1.0
        compliance_details: dict[str, bool] = {}

        if error_msg is None:
            # String exact match check
            expected_str = str(item.expected_answer)
            aliases = item.metadata.get("aliases")
            exact_match = compute_exact_match(output_text, expected_str, aliases=aliases)

            # Category-specific correctness evaluation
            if item.scoring_method == ScoringMethod.MULTIPLE_CHOICE.value:
                options = item.metadata.get("options", ["A", "B", "C", "D"])
                is_correct = compute_multiple_choice(
                    output_text,
                    expected_str,
                    options=options,
                )
            elif item.scoring_method == ScoringMethod.NUMERIC.value:
                tol = float(item.metadata.get("tolerance", 0.0))
                is_correct = compute_numeric_match(output_text, item.expected_answer, tolerance=tol)
            elif item.scoring_method == ScoringMethod.EXACT_MATCH.value:
                is_correct = exact_match
            elif item.scoring_method == ScoringMethod.INSTRUCTION_COMPLIANCE.value:
                comp_score, details = evaluate_instruction_compliance(output_text, item.metadata)
                compliance_score = comp_score
                compliance_details = details
                is_correct = (comp_score >= 0.99)
            elif item.scoring_method == ScoringMethod.JSON_SCHEMA.value:
                all_ok, comp_score, details = evaluate_json_schema_compliance(output_text, item.metadata)
                compliance_score = comp_score
                compliance_details = details
                is_correct = all_ok
            else:
                is_correct = exact_match

        return TaskResult(
            task_id=item.task_id,
            category=item.category,
            prompt=item.prompt,
            expected_answer=item.expected_answer,
            generated_text=output_text,
            thinking_text=thinking_text,
            is_correct=is_correct,
            exact_match=exact_match,
            compliance_score=compliance_score,
            compliance_details=compliance_details,
            latency_seconds=latency,
            prompt_tokens=p_tokens,
            generated_tokens=g_tokens,
            error=error_msg,
        )

    def _compute_category_breakdown(
        self,
        task_results: list[TaskResult],
    ) -> dict[str, CategoryMetrics]:
        """Aggregate metrics per benchmark category."""
        grouped: dict[str, list[TaskResult]] = {}
        for r in task_results:
            grouped.setdefault(r.category, []).append(r)

        breakdown: dict[str, CategoryMetrics] = {}
        for cat, results in grouped.items():
            total = len(results)
            correct_count = sum(1 for r in results if r.is_correct)
            acc = round((correct_count / total) * 100.0, 2) if total > 0 else 0.0
            em_rate = round((sum(1 for r in results if r.exact_match) / total) * 100.0, 2) if total > 0 else 0.0
            avg_comp = round(sum(r.compliance_score for r in results) / total, 4) if total > 0 else 1.0

            valid_lats = [r.latency_seconds for r in results if r.error is None]
            mean_lat = round(sum(valid_lats) / len(valid_lats), 4) if valid_lats else 0.0

            breakdown[cat] = CategoryMetrics(
                category=cat,
                total=total,
                correct=correct_count,
                accuracy=acc,
                exact_match_rate=em_rate,
                avg_compliance_score=avg_comp,
                mean_latency_seconds=mean_lat,
            )

        return breakdown
