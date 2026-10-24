"""Dataset schema and definitions for evaluation tasks."""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, model_validator


class TaskCategory(str, Enum):
    """Core standardized benchmark categories."""
    MULTIPLE_CHOICE = "multiple_choice"
    FACTUAL_QA = "factual_qa"
    ARITHMETIC_REASONING = "arithmetic_reasoning"
    INSTRUCTION_FOLLOWING = "instruction_following"
    STRUCTURED_OUTPUT = "structured_output"


class ScoringMethod(str, Enum):
    """Scoring algorithms used to verify model responses."""
    MULTIPLE_CHOICE = "multiple_choice"
    EXACT_MATCH = "exact_match"
    NUMERIC = "numeric"
    INSTRUCTION_COMPLIANCE = "instruction_compliance"
    JSON_SCHEMA = "json_schema"


class BenchmarkItem(BaseModel):
    """Single benchmark evaluation example."""
    task_id: str = Field(..., min_length=1, description="Unique identifier for the benchmark task.")
    category: str = Field(..., min_length=1, description="Benchmark category (e.g. multiple_choice, factual_qa).")
    prompt: str = Field(..., min_length=1, description="Full prompt text presented to the model.")
    expected_answer: Any = Field(..., description="Canonical expected answer or specification.")
    scoring_method: str = Field(..., min_length=1, description="Scoring method to use for this task.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Scoring metadata and constraints.")
    description: str | None = Field(default=None, description="Human-readable description of the task objective.")

    @model_validator(mode="after")
    def validate_task_spec(self) -> BenchmarkItem:
        # Validate that category & scoring method have appropriate metadata
        if self.scoring_method == ScoringMethod.MULTIPLE_CHOICE.value:
            if "options" not in self.metadata or not isinstance(self.metadata["options"], list):
                raise ValueError(f"Task '{self.task_id}': multiple_choice requires 'options' list in metadata.")
        elif self.scoring_method == ScoringMethod.JSON_SCHEMA.value:
            if not self.metadata.get("is_array"):
                if "required_keys" not in self.metadata or not isinstance(self.metadata["required_keys"], list):
                    raise ValueError(f"Task '{self.task_id}': json_schema requires 'required_keys' list in metadata (or 'is_array': true).")
            elif "min_items" in self.metadata and not isinstance(self.metadata["min_items"], int):
                raise ValueError(f"Task '{self.task_id}': 'min_items' must be an integer.")
        return self
