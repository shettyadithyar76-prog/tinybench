"""Unit tests for evaluation metrics and measurement tools."""

import pytest

from tinybench.metrics.accuracy import (
    calculate_accuracy,
    compute_multiple_choice,
    compute_numeric_match,
    extract_mcq_choice,
    extract_numbers,
)
from tinybench.metrics.compliance import (
    evaluate_instruction_compliance,
    evaluate_json_schema_compliance,
    extract_json_from_text,
)
from tinybench.metrics.exact_match import compute_exact_match, normalize_text
from tinybench.metrics.latency import compute_latency_summary
from tinybench.metrics.memory import MemoryTracker


# 1. Exact Match & Normalization Tests
def test_normalize_text_cases():
    assert normalize_text("  The Eiffel Tower! ") == "eiffel tower"
    assert normalize_text('"An Apple"') == "apple"
    assert normalize_text("Result: 42.") == "result 42"
    assert normalize_text("Value: 3.1415") == "value 3.1415"
    assert normalize_text("Temp: -5 degrees") == "temp -5 degrees"
    assert normalize_text("") == ""


def test_compute_exact_match():
    assert compute_exact_match("Paris.", "paris") is True
    assert compute_exact_match("The United States", "united states") is True
    assert compute_exact_match("London", "Paris") is False
    assert compute_exact_match("JPY", "Yen", aliases=["yen", "jpy"]) is True
    assert compute_exact_match("", "anything") is False


# 2. Accuracy & MCQ Tests
def test_extract_mcq_choice():
    assert extract_mcq_choice("A", ["A", "B", "C", "D"]) == "A"
    assert extract_mcq_choice("b)", ["A", "B", "C", "D"]) == "B"
    assert extract_mcq_choice("(C)", ["A", "B", "C", "D"]) == "C"
    assert extract_mcq_choice("Option D is correct.", ["A", "B", "C", "D"]) == "D"
    assert extract_mcq_choice("Answer: A", ["A", "B", "C", "D"]) == "A"
    assert extract_mcq_choice("I don't know the answer", ["A", "B", "C", "D"]) is None


def test_compute_multiple_choice():
    assert compute_multiple_choice("The correct answer is B", "B", ["A", "B", "C", "D"]) is True
    assert compute_multiple_choice("Option A", "B", ["A", "B", "C", "D"]) is False


def test_numeric_extraction_and_match():
    assert extract_numbers("The total is 42 apples.") == [42.0]
    assert extract_numbers("Between -15.5 and 100,000") == [-15.5, 100000.0]

    assert compute_numeric_match("The answer is 42", 42) is True
    assert compute_numeric_match("It equals 3.14", 3.14, tolerance=0.01) is True
    assert compute_numeric_match("No numbers here", 10) is False
    assert compute_numeric_match("Value: 100", 90, tolerance=5.0) is False


def test_calculate_accuracy():
    assert calculate_accuracy([True, True, True, False]) == 75.0
    assert calculate_accuracy([]) == 0.0
    assert calculate_accuracy([False, False]) == 0.0


# 3. Instruction Compliance Tests
def test_instruction_compliance_rules():
    # Exact word count
    score, details = evaluate_instruction_compliance("blue vast sky", {"exact_word_count": 3})
    assert score == 1.0
    assert details["exact_words_3"] is True

    score, _ = evaluate_instruction_compliance("blue vast sky today", {"exact_word_count": 3})
    assert score == 0.0

    # Required & forbidden keywords
    meta = {
        "required_keywords": ["sunshine", "beach"],
        "forbidden_keywords": ["rain", "storm"],
    }
    score, det = evaluate_instruction_compliance("Enjoying warm sunshine at the beach.", meta)
    assert score == 1.0
    assert det["contains_sunshine"] is True
    assert det["not_contains_rain"] is True

    score_fail, det_fail = evaluate_instruction_compliance("Enjoying sunshine while rain pours.", meta)
    assert score_fail < 1.0
    assert det_fail["not_contains_rain"] is False

    # Prefix and uppercase
    score_pfx, _ = evaluate_instruction_compliance(
        "ANSWER: YES",
        {"starts_with": "Answer:", "must_be_uppercase": True},
    )
    assert score_pfx == 1.0


# 4. JSON Schema Compliance Tests
def test_json_extraction():
    ok, data, _ = extract_json_from_text('{"name": "Alice", "age": 30}')
    assert ok is True
    assert data["name"] == "Alice"

    # Markdown block
    md = "Here is the JSON:\n```json\n{\"status\": \"active\"}\n```\nDone."
    ok, data, _ = extract_json_from_text(md)
    assert ok is True
    assert data["status"] == "active"

    # Malformed
    ok, _, err = extract_json_from_text("not a json string at all")
    assert ok is False


def test_json_schema_compliance():
    meta = {
        "required_keys": ["movie", "rating"],
        "key_types": {"rating": "number", "movie": "string"},
    }
    ok, score, details = evaluate_json_schema_compliance(
        '{"movie": "Inception", "rating": 9}',
        meta,
    )
    assert ok is True
    assert score == 1.0
    assert details["type_rating_is_number"] is True

    # Missing key
    ok2, score2, details2 = evaluate_json_schema_compliance(
        '{"movie": "Inception"}',
        meta,
    )
    assert ok2 is False
    assert details2["has_key_rating"] is False


# 5. Latency & Memory Tests
def test_latency_summary_statistics():
    lats = [0.1, 0.2, 0.3, 0.4, 0.5]
    summary = compute_latency_summary(lats, init_time=1.0)
    assert summary.successful_count == 5
    assert summary.init_latency_seconds == 1.0
    assert summary.first_generation_latency_seconds == 0.1
    assert summary.mean_latency_seconds == 0.3
    assert summary.median_latency_seconds == 0.3
    assert summary.min_latency_seconds == 0.1
    assert summary.max_latency_seconds == 0.5


def test_memory_tracker():
    tracker = MemoryTracker()
    tracker.start()
    assert tracker.initial_rss_mb > 0
    tracker.sample()
    measurement = tracker.finish()
    assert measurement.peak_rss_mb >= measurement.initial_rss_mb
    assert "rss" in measurement.measurement_method.lower()
