# Hacktoberfest Beginner-Friendly Issues 🎃

Welcome to Hacktoberfest! We have designed 10 modular, well-scoped tasks with clear acceptance criteria. Pick one, comment on the issue or PR, and start hacking!

---

### Task 1: Add a New Benchmark Category (`code_completion`)
- **Scope**: Introduce a new evaluation category testing basic Python syntax generation and single-line completions.
- **Files to Modify**:
  - `src/tinybench/benchmarks/schema.py`: Add `CODE_COMPLETION = "code_completion"` to `TaskCategory`.
  - `datasets/basic_v1.jsonl`: Add 5 original Python completion tasks (e.g. reverse a string, find maximum, check palindrome).
  - `src/tinybench/runner.py`: Ensure scoring correctly verifies Python code snippets.
  - `tests/test_dataset.py`: Verify new category parses and validates.
- **Acceptance Criteria**:
  - `validate-dataset` passes without warnings.
  - New category appears in the CLI breakdown and Streamlit category charts.
  - Unit tests verify code completion scoring.

---

### Task 2: Add a New Model Adapter (`llama.cpp` / GGUF)
- **Scope**: Implement a `LlamaCppModelAdapter` that loads quantized `.gguf` weights on CPU.
- **Files to Modify**:
  - `src/tinybench/models/llamacpp_backend.py`: Implement subclass of `BaseModelAdapter`.
  - `src/tinybench/models/registry.py`: Register alias `llamacpp` and wire factory instantiation.
  - `tests/test_models.py`: Add unit test with mocked runtime.
- **Acceptance Criteria**:
  - Gracefully handles case when `llama-cpp-python` is not installed by raising `ModelLoadError`.
  - Reports parameter count and accurate GGUF memory requirements.
  - Implements `load()`, `generate()`, and `unload()` lifecycle.

---

### Task 3: Improve Answer Normalization
- **Scope**: Extend `normalize_text` to strip markdown bold/italic asterisks (`**answer**` -> `answer`) and normalize mathematical expressions (e.g. `1/2` vs `0.5`).
- **Files to Modify**:
  - `src/tinybench/metrics/exact_match.py`: Update regex cleaning rules.
  - `tests/test_metrics.py`: Add test cases for markdown-decorated answers and mathematical fractions.
- **Acceptance Criteria**:
  - Correctly strips leading/trailing markdown asterisks without modifying inner text.
  - Existing exact match tests continue to pass.
  - Preserves distinction between genuinely different values.

---

### Task 4: Add an Evaluation Metric (Token-level F1 / SQuAD style)
- **Scope**: Implement unigram Token F1 score comparing prediction tokens to ground truth tokens.
- **Files to Modify**:
  - `src/tinybench/metrics/f1.py`: Create `compute_token_f1(prediction: str, ground_truth: str) -> float`.
  - `src/tinybench/metrics/__init__.py`: Export new function.
  - `src/tinybench/runner.py`: Include `token_f1` in `TaskResult` and aggregate in `RunResult`.
  - `tests/test_metrics.py`: Add comprehensive test cases with known precision, recall, and F1.
- **Acceptance Criteria**:
  - Accurately computes harmonic mean of precision and recall over word tokens.
  - Unit test covers identical strings (1.0), completely disjoint strings (0.0), and partial overlaps.

---

### Task 5: Improve Latency Measurement (Time-To-First-Token / TTFT)
- **Scope**: Add support for measuring Time-To-First-Token when using streaming generation.
- **Files to Modify**:
  - `src/tinybench/models/base.py`: Add `ttft_seconds: float | None` to `GenerationResult`.
  - `src/tinybench/metrics/latency.py`: Add `mean_ttft_seconds` to `LatencySummary`.
  - `tests/test_metrics.py`: Add test verifying TTFT statistics computation.
- **Acceptance Criteria**:
  - `LatencySummary` serializes properly to JSON and CSV.
  - Mock model simulates TTFT measurement without breaking existing tests.

---

### Task 6: Add Resource Measurement for Another Platform (macOS / Linux)
- **Scope**: Enhance `SystemInfo.capture()` and `MemoryTracker` to support platform-specific hardware details (e.g. detecting Apple Silicon M-series chips or Linux `/proc/meminfo`).
- **Files to Modify**:
  - `src/tinybench/config.py`: Improve CPU brand string detection using `platform.uname()`.
  - `src/tinybench/metrics/memory.py`: Add platform-specific notes or peak RSS retrieval via `resource.getrusage` on POSIX systems.
  - `tests/test_metrics.py`: Verify cross-platform stability.
- **Acceptance Criteria**:
  - Runs cleanly across Windows, macOS, and Linux without crashing.
  - All tests pass on GitHub Actions matrix across Ubuntu and Windows.

---

### Task 7: Improve CSV and JSON Reporting (Export to Parquet / Markdown Summary File)
- **Scope**: Add an export option to save a standalone `SUMMARY.md` alongside JSON and CSV exports.
- **Files to Modify**:
  - `src/tinybench/reporting.py`: Add `export_run_markdown(result: RunResult, output_path: Path) -> Path`.
  - `src/tinybench/cli.py`: Add `--format markdown` to the `export` command.
  - `tests/test_reporting.py`: Add test verifying markdown export file creation.
- **Acceptance Criteria**:
  - `python -m tinybench.cli export --run-id <ID> --format markdown` generates valid markdown.
  - Unit tests verify formatting and file generation.

---

### Task 8: Add Documentation and Tutorial Notebook
- **Scope**: Create a Jupyter tutorial notebook `examples/quickstart_tutorial.ipynb` walking new users through TinyBench.
- **Files to Create**:
  - `examples/quickstart_tutorial.ipynb`: Step-by-step tutorial covering dataset validation, running mock and real evaluations, and plotting results.
- **Acceptance Criteria**:
  - Notebook runs from top to bottom without errors using mock models.
  - Clear markdown explanations accompanying each code cell.

---

### Task 9: Improve the Streamlit Dashboard (Dark Mode / Filtering by Prompt Text)
- **Scope**: Add a search bar in the Evaluation Results tab to filter tasks by prompt keywords.
- **Files to Modify**:
  - `app.py`: Add `st.text_input("Search prompt text...")` to task filter controls.
- **Acceptance Criteria**:
  - Typing a keyword instantly filters task list.
  - Clear empty state if search matches zero tasks.
  - Responsive and accessible styling.

---

### Task 10: Add Tests for Edge Cases
- **Scope**: Expand `tests/test_metrics.py` and `tests/test_dataset.py` with edge case tests (Unicode emojis, empty strings, extremely long prompts, numerical edge cases like scientific notation).
- **Files to Modify**:
  - `tests/test_metrics.py`: Add `test_scientific_notation_numeric()`, `test_unicode_normalization()`.
  - `tests/test_dataset.py`: Add `test_unicode_dataset_items()`.
- **Acceptance Criteria**:
  - Minimum 6 new edge-case unit test functions.
  - 100% test pass rate with `pytest`.
