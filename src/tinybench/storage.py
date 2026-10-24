"""Persistent storage engine for benchmark run history using SQLite and JSON."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from tinybench.runner import RunResult


class RunDatabase:
    """Manages persistent SQLite indexing and JSON storage of benchmark runs."""

    def __init__(self, storage_dir: str | Path = "runs") -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.storage_dir / "runs.db"
        self._init_db()

    def _init_db(self) -> None:
        """Create sqlite tables if not present."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    run_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    family TEXT NOT NULL,
                    dataset_name TEXT NOT NULL,
                    total_examples INTEGER NOT NULL,
                    successful_examples INTEGER NOT NULL,
                    failed_examples INTEGER NOT NULL,
                    accuracy REAL NOT NULL,
                    exact_match_rate REAL NOT NULL,
                    avg_compliance_score REAL NOT NULL,
                    median_latency REAL NOT NULL,
                    mean_latency REAL NOT NULL,
                    init_latency REAL NOT NULL,
                    peak_memory_mb REAL NOT NULL,
                    memory_delta_mb REAL NOT NULL,
                    json_path TEXT NOT NULL
                )
            """)
            conn.commit()

    def save_run(self, result: RunResult) -> Path:
        """Save RunResult both as detailed JSON artifact and SQLite index entry."""
        json_path = self.storage_dir / f"{result.run_id}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, indent=2)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO evaluation_runs (
                    run_id, timestamp, model_name, model_id, family,
                    dataset_name, total_examples, successful_examples, failed_examples,
                    accuracy, exact_match_rate, avg_compliance_score,
                    median_latency, mean_latency, init_latency,
                    peak_memory_mb, memory_delta_mb, json_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.run_id,
                    result.timestamp,
                    result.model_metadata.model_name,
                    result.model_metadata.model_id,
                    result.model_metadata.family,
                    result.dataset_name,
                    result.total_examples,
                    result.successful_examples,
                    result.failed_examples,
                    result.accuracy,
                    result.exact_match_rate,
                    result.avg_compliance_score,
                    result.latency_summary.median_latency_seconds,
                    result.latency_summary.mean_latency_seconds,
                    result.latency_summary.init_latency_seconds,
                    result.memory_measurement.peak_rss_mb,
                    result.memory_measurement.memory_delta_mb,
                    str(json_path),
                ),
            )
            conn.commit()

        return json_path

    def list_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        """List summary records of all stored runs ordered by most recent."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT run_id, timestamp, model_name, model_id, family,
                       dataset_name, total_examples, successful_examples, failed_examples,
                       accuracy, exact_match_rate, avg_compliance_score,
                       median_latency, mean_latency, init_latency,
                       peak_memory_mb, memory_delta_mb, json_path
                FROM evaluation_runs
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_run(self, run_id: str) -> RunResult | None:
        """Load complete RunResult object from disk for specified run_id."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT json_path FROM evaluation_runs WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            if not row:
                # Also check direct json file in storage_dir
                direct_path = self.storage_dir / f"{run_id}.json"
                if direct_path.is_file():
                    with open(direct_path, "r", encoding="utf-8") as f:
                        return RunResult.model_validate(json.load(f))
                return None

            json_path = Path(row[0])
            if not json_path.is_file():
                return None

            with open(json_path, "r", encoding="utf-8") as f:
                return RunResult.model_validate(json.load(f))

    def delete_run(self, run_id: str) -> bool:
        """Remove run from database and delete its JSON file if present."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT json_path FROM evaluation_runs WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            if not row:
                return False

            json_path = Path(row[0])
            if json_path.exists():
                try:
                    json_path.unlink()
                except Exception:
                    pass

            cursor.execute("DELETE FROM evaluation_runs WHERE run_id = ?", (run_id,))
            conn.commit()
            return True
