"""Configuration models and system hardware information collector."""

from __future__ import annotations

import os
import platform
import psutil
from typing import Any
from pydantic import BaseModel, Field


class GenerationConfig(BaseModel):
    """Parameters controlling text generation across model backends."""
    max_new_tokens: int = Field(default=64, ge=1, le=2048, description="Maximum new tokens to generate.")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Sampling temperature; 0.0 is deterministic greedy decoding.")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling threshold.")
    seed: int = Field(default=42, description="Random seed for reproducible generation.")
    stop_sequences: list[str] = Field(default_factory=list, description="Optional stop strings to halt generation.")


class BenchmarkConfig(BaseModel):
    """Configuration for an evaluation run."""
    dataset_path: str = Field(..., description="Path to the .jsonl benchmark dataset.")
    model_name: str = Field(default="mock", description="Registered model adapter identifier.")
    model_revision: str | None = Field(default=None, description="Specific checkpoint commit / revision.")
    categories: list[str] | None = Field(default=None, description="Subset of categories to evaluate.")
    limit: int | None = Field(default=None, ge=1, description="Maximum number of items to evaluate.")
    generation_config: GenerationConfig = Field(default_factory=GenerationConfig, description="Generation settings.")
    output_dir: str = Field(default="runs", description="Directory to save run results.")
    system_prompt: str | None = Field(default=None, description="Optional system instruction for instruction-tuned models.")
    timeout_per_task: float = Field(default=30.0, ge=1.0, description="Inference timeout per task in seconds.")


class SystemInfo(BaseModel):
    """Hardware and environment metadata for research reproducibility."""
    os_name: str = Field(..., description="Operating system name (Windows, Linux, Darwin).")
    os_version: str = Field(..., description="OS kernel/build version.")
    architecture: str = Field(..., description="CPU architecture (e.g. AMD64, x86_64, arm64).")
    processor: str = Field(..., description="Processor brand/model string.")
    cpu_physical_cores: int = Field(..., description="Number of physical CPU cores.")
    cpu_logical_cores: int = Field(..., description="Number of logical CPU threads.")
    total_ram_mb: float = Field(..., description="Total system RAM in megabytes.")
    available_ram_mb: float = Field(..., description="Available system RAM at run time in megabytes.")
    python_version: str = Field(..., description="Python runtime version.")
    torch_version: str | None = Field(default=None, description="PyTorch version if installed.")
    transformers_version: str | None = Field(default=None, description="Hugging Face Transformers version if installed.")

    @classmethod
    def capture(cls) -> SystemInfo:
        """Capture current system hardware and library environment."""
        torch_ver: str | None = None
        try:
            import torch  # type: ignore
            torch_ver = getattr(torch, "__version__", None)
        except Exception:
            pass

        transformers_ver: str | None = None
        try:
            import transformers  # type: ignore
            transformers_ver = getattr(transformers, "__version__", None)
        except Exception:
            pass

        mem = psutil.virtual_memory()
        return cls(
            os_name=platform.system(),
            os_version=platform.version(),
            architecture=platform.machine(),
            processor=platform.processor() or "Unknown CPU",
            cpu_physical_cores=psutil.cpu_count(logical=False) or 1,
            cpu_logical_cores=psutil.cpu_count(logical=True) or 1,
            total_ram_mb=round(mem.total / (1024 * 1024), 2),
            available_ram_mb=round(mem.available / (1024 * 1024), 2),
            python_version=platform.python_version(),
            torch_version=torch_ver,
            transformers_version=transformers_ver,
        )
