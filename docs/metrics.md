# TinyBench Metrics & Research Methodology

This document outlines the evaluation algorithms, measurement techniques, and methodological principles underpinning TinyBench.

---

## 1. Quality Metrics

### A. Task Accuracy
- **Definition**: The percentage of tasks where the model's generated output satisfies the objective correctness criterion.
- **Scoring Logic**:
  - **Multiple Choice**: Extracts the selected option letter (e.g., `A`, `(B)`, `Option C`, `Answer: D`) using regular expressions and matches against the answer key.
  - **Numeric Reasoning**: Scans the text for integer and floating-point candidates. An answer is marked correct if any candidate falls within `expected ± tolerance`.
  - **Factual QA**: Evaluates whether the normalized output matches the canonical expected answer or any documented aliases.

### B. Exact Match (EM)
- **Definition**: String-level equality between the generated text and expected answer after applying standardized text normalization.
- **Normalization Pipeline**:
  1. Trims leading and trailing whitespace.
  2. Converts all characters to lowercase.
  3. Strips enclosing single quotes, double quotes, and backticks.
  4. Removes standalone English grammatical articles (`a`, `an`, `the`) at word boundaries.
  5. Strips punctuation while preserving decimal numbers (e.g. `3.14`) and negative signs (e.g. `-5`).
  6. Collapses multiple spaces into a single space.
- **Design Rule**: Normalization is deliberately conservative to avoid equating fundamentally different answers (e.g., `12` vs `21`).

### C. Instruction Compliance
- **Definition**: Quantitative measurement of how strictly the model adhered to arbitrary formatting and structural constraints, calculated independently of factual accuracy.
- **Evaluated Constraints**:
  - **Exact / Max / Min Word Count**: Verifies space-delimited word counts.
  - **Required Keywords**: Verifies presence of mandatory tokens.
  - **Forbidden Keywords**: Verifies absence of restricted tokens (negative constraints).
  - **Formatting Prefixes & Suffixes**: Checks starts-with / ends-with patterns.
  - **Casing**: Verifies all-uppercase or all-lowercase requirements.
  - **JSON Schema**: Parses output into JSON and asserts key existence, array dimensions, and scalar types (`number`, `string`, `bool`).
- **Scoring**: Expressed as a scalar from `0.0` to `1.0` reflecting the fraction of active rules satisfied.

---

## 2. Efficiency & Resource Metrics

### A. Latency Measurement
- **Clock**: Monotonic timing via `time.perf_counter()`, ensuring immunity from system wall-clock adjustments or daylight saving shifts.
- **Reported Statistics**:
  - **Initialization Latency**: Duration required to instantiate the tokenizer and load model weights into CPU memory.
  - **First-Generation Latency (Cold Start)**: Duration of the first task inference, capturing JIT compilation or cache allocation overhead.
  - **Median Latency**: Robust central tendency resistant to one-off system hiccups.
  - **Mean Latency**: Standard arithmetic average generation duration.
  - **P95 Latency**: 95th percentile duration highlighting tail latency.

### B. Memory Consumption Profiling
- **Metric**: Process Resident Set Size (RSS) measured via `psutil.Process().memory_info().rss`.
- **Reported Values**: Baseline RSS (prior to loading), Peak RSS (highest observed point), and Memory Delta.
- **Methodological Limitations**:
  > Process RSS captures physical RAM allocated directly to the Python process.
  > It does not capture OS-level disk cache, shared OS dynamic libraries, or memory isolated within external device drivers. Measurements may vary slightly across Windows, macOS, and Linux due to operating system virtual memory management policies.

---

## 3. Research Integrity & Ethical Reporting

TinyBench adheres to the following principles:

1. **Comparable Conditions**: When comparing models, TinyBench uses identical datasets, identical random seeds, and identical generation parameters (`max_new_tokens`, `temperature`, `top_p`).
2. **No Extrapolated Superiority Claims**: Scoring higher on a 35-item benchmark dataset demonstrates strong performance on those specific prompt formats, but does **not** prove a model is universally superior across broader natural language domains.
3. **Open Weights vs. Open Source**: TinyBench refers to models like SmolLM2 and Qwen2.5 as **open-weight models**. Open weights are not automatically equivalent to an OSI-approved open-source software license. Licenses must always be inspected individually.
4. **Transparent Failures**: Tasks that fail due to out-of-memory errors, invalid JSON syntax, or inference timeouts are explicitly preserved in the results with stack traces rather than silently omitted.
