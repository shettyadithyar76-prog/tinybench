"""Latency measurement and statistics computation using monotonic clock."""

from __future__ import annotations

import statistics
from pydantic import BaseModel, Field


class LatencySummary(BaseModel):
    """Aggregate latency statistics for an evaluation run."""
    init_latency_seconds: float = Field(default=0.0, description="Model weights and tokenizer initialization duration.")
    first_generation_latency_seconds: float = Field(default=0.0, description="Latency of the first generated sample (cold start).")
    mean_latency_seconds: float = Field(default=0.0, description="Arithmetic mean of per-task generation latencies.")
    median_latency_seconds: float = Field(default=0.0, description="Median generation latency.")
    p95_latency_seconds: float = Field(default=0.0, description="95th percentile generation latency.")
    min_latency_seconds: float = Field(default=0.0, description="Fastest individual inference latency.")
    max_latency_seconds: float = Field(default=0.0, description="Slowest individual inference latency.")
    total_generation_seconds: float = Field(default=0.0, description="Total cumulative generation time across all tasks.")
    successful_count: int = Field(default=0, description="Number of tasks completed without fatal error.")
    failed_count: int = Field(default=0, description="Number of tasks that encountered generation failure.")


def compute_latency_summary(
    latencies: list[float],
    init_time: float = 0.0,
    failed_count: int = 0,
) -> LatencySummary:
    """Calculate descriptive statistics from recorded task latencies.

    Args:
        latencies: List of generation durations in seconds (monotonic clock).
        init_time: Model setup/load time in seconds.
        failed_count: Number of failed generation attempts.

    Returns:
        LatencySummary with mean, median, p95, min, max.
    """
    if not latencies:
        return LatencySummary(
            init_latency_seconds=round(init_time, 4),
            successful_count=0,
            failed_count=failed_count,
        )

    first_gen = latencies[0]
    total_time = sum(latencies)
    mean_lat = statistics.mean(latencies)
    median_lat = statistics.median(latencies)
    min_lat = min(latencies)
    max_lat = max(latencies)

    # 95th percentile calculation
    if len(latencies) >= 2:
        sorted_lats = sorted(latencies)
        p95_index = int(0.95 * (len(sorted_lats) - 1))
        p95_lat = sorted_lats[p95_index]
    else:
        p95_lat = latencies[0]

    return LatencySummary(
        init_latency_seconds=round(init_time, 4),
        first_generation_latency_seconds=round(first_gen, 4),
        mean_latency_seconds=round(mean_lat, 4),
        median_latency_seconds=round(median_lat, 4),
        p95_latency_seconds=round(p95_lat, 4),
        min_latency_seconds=round(min_lat, 4),
        max_latency_seconds=round(max_lat, 4),
        total_generation_seconds=round(total_time, 4),
        successful_count=len(latencies),
        failed_count=failed_count,
    )
