"""Dataset loading utility for JSONL benchmark datasets."""

from __future__ import annotations

import json
from pathlib import Path
from pydantic import ValidationError

from tinybench.benchmarks.schema import BenchmarkItem
from tinybench.exceptions import DatasetValidationError


def load_dataset(
    file_path: str | Path,
    categories: list[str] | None = None,
    limit: int | None = None,
) -> list[BenchmarkItem]:
    """Load and validate benchmark items from a JSONL file.

    Args:
        file_path: Path to the .jsonl file.
        categories: Optional list of category names to filter by.
        limit: Optional maximum number of items to return after filtering.

    Returns:
        List of validated BenchmarkItem objects.

    Raises:
        DatasetValidationError: If file does not exist, has bad JSON, or schema violation.
    """
    path = Path(file_path)
    if not path.is_file():
        raise DatasetValidationError(f"Dataset file not found: {file_path}")

    items: list[BenchmarkItem] = []
    seen_ids: set[str] = set()

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                clean_line = line.strip()
                if not clean_line or clean_line.startswith("#"):
                    continue

                try:
                    record = json.loads(clean_line)
                except json.JSONDecodeError as err:
                    raise DatasetValidationError(
                        f"JSON syntax error in '{path}' at line {line_idx}: {err}"
                    ) from err

                try:
                    item = BenchmarkItem.model_validate(record)
                except ValidationError as val_err:
                    raise DatasetValidationError(
                        f"Validation error in '{path}' at line {line_idx}: {val_err}"
                    ) from val_err

                if item.task_id in seen_ids:
                    raise DatasetValidationError(
                        f"Duplicate task_id '{item.task_id}' found at line {line_idx} in '{path}'"
                    )
                seen_ids.add(item.task_id)

                if categories and item.category not in categories:
                    continue

                items.append(item)
                if limit and len(items) >= limit:
                    break

    except DatasetValidationError:
        raise
    except Exception as e:
        raise DatasetValidationError(f"Failed to read dataset file '{file_path}': {e}") from e

    if not items:
        if categories:
            raise DatasetValidationError(
                f"No items found in '{file_path}' matching categories: {categories}"
            )
        raise DatasetValidationError(f"Dataset file '{file_path}' contains no valid benchmark records.")

    return items
