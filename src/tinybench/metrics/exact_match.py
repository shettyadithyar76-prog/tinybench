"""Exact match metric with reproducible text normalization."""

from __future__ import annotations

import re
import string


def normalize_text(text: str) -> str:
    """Normalize text for consistent string matching.

    Rules applied:
    1. Strip leading and trailing whitespace.
    2. Convert to lowercase.
    3. Remove enclosing quotation marks (single/double/backticks).
    4. Remove leading/trailing periods or punctuation.
    5. Strip standalone articles ('a', 'an', 'the') at word boundaries.
    6. Collapse internal consecutive whitespace to a single space.

    Normalization preserves internal hyphens and digits so numbers (e.g. -5, 42)
    and compounds remain distinct.
    """
    if not text:
        return ""

    s = text.strip().lower()

    # Strip enclosing quotes
    s = s.strip("\"'`“”‘’")

    # Remove standalone English articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)

    # Remove punctuation except hyphens and decimals attached to numbers
    # We replace punctuation with spaces unless it's between digits
    cleaned_chars = []
    for idx, char in enumerate(s):
        if char in string.punctuation:
            # Preserve decimal point in numbers like 3.14
            if char == "." and 0 < idx < len(s) - 1 and s[idx - 1].isdigit() and s[idx + 1].isdigit():
                cleaned_chars.append(char)
            # Preserve negative sign in numbers like -5
            elif char == "-" and idx < len(s) - 1 and s[idx + 1].isdigit() and (idx == 0 or s[idx - 1].isspace()):
                cleaned_chars.append(char)
            else:
                cleaned_chars.append(" ")
        else:
            cleaned_chars.append(char)

    s = "".join(cleaned_chars)
    # Collapse whitespace
    s = " ".join(s.split())
    return s.strip()


def compute_exact_match(
    prediction: str,
    ground_truth: str,
    aliases: list[str] | None = None,
) -> bool:
    """Check if normalized prediction matches normalized ground truth or any alias.

    Args:
        prediction: Raw model output.
        ground_truth: Canonical expected answer.
        aliases: Optional list of acceptable alternative phrasing.

    Returns:
        True if exact match succeeds, False otherwise.
    """
    norm_pred = normalize_text(prediction)
    norm_gt = normalize_text(ground_truth)

    if not norm_pred:
        return False

    if norm_pred == norm_gt:
        return True

    if aliases:
        for alias in aliases:
            if norm_pred == normalize_text(alias):
                return True

    return False
