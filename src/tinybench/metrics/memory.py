"""Memory consumption measurement and platform diagnostics."""

from __future__ import annotations

import os
import psutil
from pydantic import BaseModel, Field


class MemoryMeasurement(BaseModel):
    """Memory profile of an evaluation run."""
    initial_rss_mb: float = Field(..., description="Resident Set Size before model loading/inference in MB.")
    peak_rss_mb: float = Field(..., description="Highest observed Resident Set Size during run in MB.")
    final_rss_mb: float = Field(..., description="Resident Set Size at conclusion of run in MB.")
    memory_delta_mb: float = Field(..., description="Net change in memory (final - initial) in MB.")
    measurement_method: str = Field(
        default="psutil.Process().memory_info().rss",
        description="Technique used to measure memory.",
    )
    notes: str = Field(
        default=(
            "Process Resident Set Size (RSS) captures physical memory currently allocated to the Python "
            "process. Note: Process RSS does not include system-wide page cache, GPU memory, or out-of-process "
            "driver allocations."
        ),
        description="Platform and methodological qualifications.",
    )


class MemoryTracker:
    """Monitors process resident memory throughout benchmark execution."""

    def __init__(self) -> None:
        self.process = psutil.Process(os.getpid())
        self.initial_rss_mb: float = 0.0
        self.peak_rss_mb: float = 0.0

    def start(self) -> None:
        """Record baseline memory before model loading or test execution."""
        mem_info = self.process.memory_info()
        self.initial_rss_mb = round(mem_info.rss / (1024 * 1024), 2)
        self.peak_rss_mb = self.initial_rss_mb

    def sample(self) -> float:
        """Sample current RSS and update peak if higher. Returns current MB."""
        try:
            current_mb = round(self.process.memory_info().rss / (1024 * 1024), 2)
            if current_mb > self.peak_rss_mb:
                self.peak_rss_mb = current_mb
            return current_mb
        except Exception:
            return self.peak_rss_mb

    def finish(self) -> MemoryMeasurement:
        """Finalize tracking and return structured measurement."""
        final_mb = self.sample()
        delta_mb = round(self.peak_rss_mb - self.initial_rss_mb, 2)
        return MemoryMeasurement(
            initial_rss_mb=self.initial_rss_mb,
            peak_rss_mb=self.peak_rss_mb,
            final_rss_mb=final_mb,
            memory_delta_mb=delta_mb,
        )
