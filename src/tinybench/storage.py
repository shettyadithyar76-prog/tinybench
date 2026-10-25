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
        """Create sqlite tables if not present and apply migrations."""
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
                    json_path TEXT NOT NULL,
                    is_demo INTEGER DEFAULT 0
                )
            """)
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(evaluation_runs)")
            cols = [c[1] for c in cursor.fetchall()]
            if "is_demo" not in cols:
                conn.execute("ALTER TABLE evaluation_runs ADD COLUMN is_demo INTEGER DEFAULT 0")
            # Ensure legacy mock runs are safely tagged as demo runs
            conn.execute("UPDATE evaluation_runs SET is_demo = 1 WHERE model_id LIKE 'mock%' OR model_name LIKE '%Mock%'")
            # Normalize all Windows backslash paths to cross-platform POSIX forward slashes
            conn.execute("UPDATE evaluation_runs SET json_path = REPLACE(json_path, '\\', '/')")
            conn.commit()

    def save_run(self, result: RunResult) -> Path:
        """Save RunResult both as detailed JSON artifact and SQLite index entry."""
        json_path = self.storage_dir / f"{result.run_id}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, indent=2)

        is_demo_val = 1 if getattr(result, "is_demo", False) else 0

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO evaluation_runs (
                    run_id, timestamp, model_name, model_id, family,
                    dataset_name, total_examples, successful_examples, failed_examples,
                    accuracy, exact_match_rate, avg_compliance_score,
                    median_latency, mean_latency, init_latency,
                    peak_memory_mb, memory_delta_mb, json_path, is_demo
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    json_path.as_posix(),
                    is_demo_val,
                ),
            )
            conn.commit()

        return json_path

    def list_runs(self, limit: int = 100, real_only: bool = False) -> list[dict[str, Any]]:
        """List summary records of stored runs ordered by most recent."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            query = """
                SELECT run_id, timestamp, model_name, model_id, family,
                       dataset_name, total_examples, successful_examples, failed_examples,
                       accuracy, exact_match_rate, avg_compliance_score,
                       median_latency, mean_latency, init_latency,
                       peak_memory_mb, memory_delta_mb, json_path, is_demo
                FROM evaluation_runs
            """
            params: list[Any] = []
            if real_only:
                query += " WHERE is_demo = 0"
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, tuple(params))
            return [dict(row) for row in cursor.fetchall()]

    def get_run(self, run_id: str) -> RunResult | None:
        """Load complete RunResult object from disk for specified run_id."""
        # 1. Direct path lookup in storage_dir (always cross-platform)
        direct_path = self.storage_dir / f"{run_id}.json"
        if direct_path.is_file():
            with open(direct_path, "r", encoding="utf-8") as f:
                return RunResult.model_validate(json.load(f))

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT json_path FROM evaluation_runs WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            if not row:
                return None

            raw_path = str(row[0]).replace("\\", "/")
            json_path = Path(raw_path)
            if not json_path.is_file():
                # Cross-platform fallback: look up by filename in storage_dir
                fallback = self.storage_dir / json_path.name
                if fallback.is_file():
                    json_path = fallback
                else:
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
