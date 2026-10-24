"""Rule-based deterministic instruction compliance and structured output verification."""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json_from_text(text: str) -> tuple[bool, Any, str | None]:
    """Attempt to parse JSON from raw text or markdown code blocks."""
    clean = text.strip()

    # Direct parse attempt
    try:
        parsed = json.loads(clean)
        return True, parsed, None
    except json.JSONDecodeError:
        pass

    # Look for ```json ... ``` or ``` ... ``` code blocks
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean, re.IGNORECASE)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1).strip())
            return True, parsed, None
        except json.JSONDecodeError as err:
            return False, None, f"Failed parsing markdown JSON block: {err}"

    # Look for first { ... } or [ ... ]
    match_obj = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", clean)
    if match_obj:
        try:
            parsed = json.loads(match_obj.group(1))
            return True, parsed, None
        except json.JSONDecodeError as err:
            return False, None, f"Failed parsing extracted JSON substring: {err}"

    return False, None, "No valid JSON structure found in output"


def evaluate_instruction_compliance(
    prediction: str,
    metadata: dict[str, Any],
) -> tuple[float, dict[str, bool]]:
    """Evaluate deterministic instruction constraints on a model response.

    Returns:
        tuple of (compliance_score: float between 0.0 and 1.0, details_dict: dict[rule_name, bool])
    """
    if not metadata:
        return 1.0, {}

    checks: dict[str, bool] = {}
    clean_text = prediction.strip()
    words = clean_text.split()

    # Rule: exact_word_count
    if "exact_word_count" in metadata:
        target = int(metadata["exact_word_count"])
        checks[f"exact_words_{target}"] = (len(words) == target)

    # Rule: max_words
    if "max_words" in metadata:
        max_limit = int(metadata["max_words"])
        checks[f"max_words_{max_limit}"] = (len(words) <= max_limit)

    # Rule: min_words
    if "min_words" in metadata:
        min_limit = int(metadata["min_words"])
        checks[f"min_words_{min_limit}"] = (len(words) >= min_limit)

    # Rule: required_keywords
    if "required_keywords" in metadata:
        kws = metadata["required_keywords"]
        for kw in kws:
            checks[f"contains_{kw}"] = bool(re.search(re.escape(kw), clean_text, re.IGNORECASE))

    # Rule: forbidden_keywords
    if "forbidden_keywords" in metadata:
        f_kws = metadata["forbidden_keywords"]
        for fkw in f_kws:
            checks[f"not_contains_{fkw}"] = not bool(re.search(re.escape(fkw), clean_text, re.IGNORECASE))

    # Rule: starts_with
    if "starts_with" in metadata:
        pfx = str(metadata["starts_with"]).strip()
        checks[f"starts_with_{pfx}"] = clean_text.lower().startswith(pfx.lower())

    # Rule: ends_with
    if "ends_with" in metadata:
        sfx = str(metadata["ends_with"]).strip().rstrip(".,;!")
        clean_end = clean_text.rstrip(".,;! \n\r")
        checks[f"ends_with_{sfx}"] = clean_end.lower().endswith(sfx.lower())

    # Rule: must_be_uppercase
    if metadata.get("must_be_uppercase") is True:
        # Check that alphabetic characters are all upper
        letters = [c for c in clean_text if c.isalpha()]
        checks["must_be_uppercase"] = bool(letters and all(c.isupper() for c in letters))

    # Rule: must_be_lowercase
    if metadata.get("must_be_lowercase") is True:
        letters = [c for c in clean_text if c.isalpha()]
        checks["must_be_lowercase"] = bool(letters and all(c.islower() for c in letters))

    # Calculate overall compliance score
    if not checks:
        return 1.0, {}

    passed = sum(1 for passed_flag in checks.values() if passed_flag)
    score = round(passed / len(checks), 4)
    return score, checks


def evaluate_json_schema_compliance(
    prediction: str,
    metadata: dict[str, Any],
) -> tuple[bool, float, dict[str, bool]]:
    """Evaluate JSON schema constraints.

    Returns:
        tuple of (is_valid_json: bool, compliance_score: float, details: dict[str, bool])
    """
    valid_json, parsed, _ = extract_json_from_text(prediction)
    checks: dict[str, bool] = {"valid_json_syntax": valid_json}

    if not valid_json:
        return False, 0.0, checks

    if metadata.get("is_array") is True:
        checks["is_array"] = isinstance(parsed, list)
        if "min_items" in metadata and isinstance(parsed, list):
            checks[f"min_items_{metadata['min_items']}"] = len(parsed) >= int(metadata["min_items"])
    else:
        checks["is_object"] = isinstance(parsed, dict)
        if isinstance(parsed, dict):
            required_keys = metadata.get("required_keys", [])
            for key in required_keys:
                checks[f"has_key_{key}"] = (key in parsed)

            key_types = metadata.get("key_types", {})
            for key, expected_type in key_types.items():
                if key in parsed:
                    if expected_type in ("number", "int", "float"):
                        checks[f"type_{key}_is_number"] = isinstance(parsed[key], (int, float))
                    elif expected_type == "string":
                        checks[f"type_{key}_is_string"] = isinstance(parsed[key], str)
                    elif expected_type == "bool":
                        checks[f"type_{key}_is_bool"] = isinstance(parsed[key], bool)
                    elif expected_type == "array":
                        checks[f"type_{key}_is_array"] = isinstance(parsed[key], list)

    passed = sum(1 for ok in checks.values() if ok)
    score = round(passed / len(checks), 4)
    all_passed = (passed == len(checks))
    return all_passed, score, checks
