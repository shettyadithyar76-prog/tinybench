# TinyBench Architecture

TinyBench is engineered as a decoupled, modular benchmarking toolkit for evaluating small open-weight language models on CPU hardware.

## System Component Diagram

```
+-------------------------------------------------------------+
|                      User Interfaces                        |
|   +-----------------------+     +-----------------------+   |
|   |  CLI (tinybench.cli)  |     |  Dashboard (app.py)   |   |
|   +-----------+-----------+     +-----------+-----------+   |
+---------------|-----------------------------|---------------+
                |                             |
                +--------------+--------------+
                               |
                               v
+-------------------------------------------------------------+
|                    Benchmark Runner Core                    |
|                    (tinybench.runner)                       |
|                                                             |
|  1. Load & Validate Dataset (tinybench.benchmarks)          |
|  2. Initialize Resource Tracker (tinybench.metrics.memory)  |
|  3. Instantiate Model Adapter (tinybench.models)            |
|  4. Execute Sequential Tasks & Collect Latencies            |
|  5. Grade Outcomes (tinybench.metrics)                      |
|  6. Aggregate Scores & Category Breakdown                   |
|  7. Release Model & Finalize Memory Profiling               |
+-------------------------------------------------------------+
          |                         |                     |
          v                         v                     v
+-------------------+     +-------------------+  +-------------------+
|  Model Adapters   |     | Metrics & Grading |  | Storage & Export  |
|  - MockAdapter    |     | - Exact Match     |  | - JSON Artifacts  |
|  - Transformers   |     | - MCQ Option      |  | - SQLite DB       |
|    (CPU PyTorch)  |     | - Numeric Match   |  | - CSV Summary     |
|  - Custom Registry|     | - Compliance      |  | - Markdown Report |
|                   |     | - Monotonic Clock |  |                   |
+-------------------+     +-------------------+  +-------------------+
```

## Core Modules

### 1. `tinybench.models`
Defines the `BaseModelAdapter` abstract base class. Model backends isolate all inference code so the evaluation runner does not depend on any specific deep learning framework.
- **`MockModelAdapter`**: Lightweight deterministic simulator with configurable noise and latency. Enables instant unit testing and CI verification with zero external downloads.
- **`TransformersModelAdapter`**: Hugging Face Transformers adapter configured specifically for CPU execution (`device_map="cpu"`, `torch_dtype=float32`, `low_cpu_mem_usage=True`). Includes RAM pre-flight checks and clean memory de-allocation.
- **`ModelRegistry`**: Factory and catalog mapping model names, aliases, and Hugging Face Hub IDs to adapter instances.

### 2. `tinybench.benchmarks`
- **`schema.py`**: Pydantic schema for `BenchmarkItem` covering `task_id`, `category`, `prompt`, `expected_answer`, `scoring_method`, and `metadata`.
- **`loader.py`**: Safe line-by-line JSONL parser with line-numbered error tracking and duplicate detection.
- **`validator.py`**: Automated integrity verification returning comprehensive reports on syntax, schema conformity, and category distributions.

### 3. `tinybench.metrics`
- **`accuracy.py`**: Objective scoring utilities for multiple-choice option extraction and numerical tolerance comparison.
- **`exact_match.py`**: Documented text normalizer (lowercase, quote stripping, article stripping, decimal preservation) for string equality tests.
- **`compliance.py`**: Deterministic rule checker evaluating word counts, forbidden/required keywords, prefixes, suffixes, case rules, and JSON schema syntax.
- **`latency.py`**: Monotonic clock statistics (`time.perf_counter()`) computing cold-start, mean, median, min, max, and p95 latency.
- **`memory.py`**: Process-level Resident Set Size (RSS) tracker measuring baseline, peak RAM, and memory delta during evaluation.

### 4. `tinybench.storage` & `reporting`
- **`storage.py`**: Dual-layer persistence engine saving rich, machine-readable JSON files per run, indexed by a lightweight SQLite database (`runs.db`) for fast dashboard queries.
- **`reporting.py`**: Formatters for exportable CSV tables (summary and per-task details), markdown summaries, and cross-model comparison matrices.
