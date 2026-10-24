"""Accuracy computation utilities for multiple-choice, numeric, and boolean tasks."""

from __future__ import annotations

import re
from typing import Any


def extract_mcq_choice(prediction: str, valid_options: list[str]) -> str | None:
    """Extract a multiple-choice option letter from raw model response.

    Looks for patterns like:
    - 'A'
    - 'A)'
    - '(A)'
    - 'Option A'
    - 'Answer: A'
    - 'The correct option is A'
    """
    if not prediction:
        return None

    clean = prediction.strip()
    valid_set = {opt.upper() for opt in valid_options}

    # Case 1: First single character is one of the valid options
    first_token = clean.split()[0].rstrip(").,;:").upper() if clean.split() else ""
    if first_token in valid_set:
        return first_token

    # Case 2: Regex for explicit 'Answer: [A-D]' or 'Option [A-D]'
    opts_pattern = "|".join(re.escape(opt) for opt in valid_options)
    match = re.search(
        rf"(?:answer|option|choice|is)\s*[:=\-]?\s*\(?([A-Za-z])\)?",
        clean,
        re.IGNORECASE,
    )
    if match and match.group(1).upper() in valid_set:
        return match.group(1).upper()

    # Case 3: Match standalone option surrounded by boundary or parenthesis
    match_paren = re.search(rf"\b\(?({opts_pattern})\)?\b", clean, re.IGNORECASE)
    if match_paren and match_paren.group(1).upper() in valid_set:
        return match_paren.group(1).upper()

    return None


def compute_multiple_choice(
    prediction: str,
    expected_option: str,
    options: list[str],
) -> bool:
    """Determine whether the model selected the correct multiple choice option."""
    selected = extract_mcq_choice(prediction, options)
    if not selected:
        return False
    return selected.upper() == expected_option.strip().upper()


def extract_numbers(text: str) -> list[float]:
    """Extract all numbers (integers, floats, negative values) from text."""
    # Matches integers and floats like 42, -15, 3.1415, 1,000 (after comma strip)
    clean = text.replace(",", "")
    pattern = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    matches = re.findall(pattern, clean)
    numbers: list[float] = []
    for m in matches:
        try:
            numbers.append(float(m))
        except ValueError:
            continue
    return numbers


def compute_numeric_match(
    prediction: str,
    expected: float | int,
    tolerance: float = 0.0,
) -> bool:
    """Check if model output contains the target numerical value within tolerance."""
    extracted = extract_numbers(prediction)
    if not extracted:
        return False

    expected_val = float(expected)
    for val in extracted:
        if abs(val - expected_val) <= tolerance:
            return True
    return False


def calculate_accuracy(results: list[bool]) -> float:
    """Calculate percentage accuracy from a list of boolean correctness checks."""
    if not results:
        return 0.0
    return round((sum(1 for r in results if r) / len(results)) * 100.0, 2)
