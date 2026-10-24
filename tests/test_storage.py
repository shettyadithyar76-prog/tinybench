"""Unit tests for SQLite persistent run database."""

from pathlib import Path
import pytest

from tinybench.runner import RunResult
from tinybench.storage import RunDatabase


def test_sqlite_storage_lifecycle(tmp_path: Path, sample_run_result: RunResult):
    db = RunDatabase(storage_dir=tmp_path / "runs")
    assert db.db_path.is_file()

    # Save run
    json_path = db.save_run(sample_run_result)
    assert json_path.is_file()

    # List runs
    runs = db.list_runs()
    assert len(runs) == 1
    assert runs[0]["run_id"] == sample_run_result.run_id
    assert runs[0]["accuracy"] == sample_run_result.accuracy

    # Fetch run
    loaded = db.get_run(sample_run_result.run_id)
    assert loaded is not None
    assert loaded.run_id == sample_run_result.run_id
    assert len(loaded.task_results) == len(sample_run_result.task_results)

    # Delete run
    deleted = db.delete_run(sample_run_result.run_id)
    assert deleted is True
    assert db.get_run(sample_run_result.run_id) is None
    assert not json_path.exists()
