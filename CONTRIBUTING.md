# Contributing to TinyBench ⚡

Welcome to TinyBench! We are excited to collaborate with open-source contributors during **Hacktoberfest** and beyond.

TinyBench is designed to be beginner-friendly, modular, and accessible: you do not need expensive GPU hardware or paid cloud credits to contribute.

---

## 🛠️ Development Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/tinybench.git
cd tinybench
```

### 2. Set Up a Virtual Environment (Python 3.11+)
```bash
# On Linux/macOS
python3 -m venv venv
source venv/bin/activate

# On Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install in Editable Mode
```bash
# Base installation with test tools
pip install -e ".[dev]"

# Or full installation including models and dashboard
pip install -e ".[all]"
```

---

## 🧪 Running Automated Tests

TinyBench tests are designed to execute completely in mock mode without downloading large neural network weights.

```bash
# Run the complete test suite
pytest

# Run with verbose output and coverage
pytest -v --cov=tinybench
```

All 40+ unit and integration tests should pass on ordinary CPU hardware in under 15 seconds.

---

## 💻 Running the Streamlit Dashboard

To verify UI changes locally:
```bash
streamlit run app.py
```
Then navigate to `http://localhost:8501`.

---

## 🏷️ Beginner-Friendly Hacktoberfest Tasks

Looking for an issue to pick up? We maintain curated, bite-sized tasks in [docs/beginner_issues.md](docs/beginner_issues.md):

1. **Add a new benchmark category** (e.g. basic code completion, translation).
2. **Add a new model adapter** (e.g. `llama.cpp` wrapper or ONNX Runtime).
3. **Improve answer normalization** (e.g. multilingual or math expression normalization).
4. **Add an evaluation metric** (e.g. ROUGE-1 / BLEU / Token F1).
5. **Improve latency measurement** (e.g. Time-To-First-Token tracking).
6. **Add resource measurement for another platform** (e.g. macOS Apple Silicon memory, Linux cgroups).
7. **Improve CSV/JSON reporting** (e.g. Excel export, Parquet export, LaTeX tables).
8. **Add documentation and examples** (e.g. Jupyter notebook tutorials).
9. **Improve the Streamlit dashboard** (e.g. dark mode toggle, filtering improvements).
10. **Add tests for edge cases** (e.g. unicode text, long sequences, malformed JSON).

See the [Beginner Issues Guide](docs/beginner_issues.md) for step-by-step acceptance criteria for each task!

---

## 📏 Coding Standards & Guidelines

- **Type Hints**: Use typing annotations (`from __future__ import annotations`, `list[str]`, `dict[str, Any]`, etc.).
- **Validation**: Use Pydantic models for configuration and structured data models.
- **Error Handling**: Raise subclasses of `TinyBenchError` (from `tinybench.exceptions`) rather than bare exceptions.
- **Docstrings**: Provide clear Google/Sphinx style docstrings on all public functions, classes, and methods.
- **Deterministic**: Tests must never rely on live model downloads unless explicitly marked with `@pytest.mark.skipif`.

---

## 🚀 Submitting a Pull Request (PR)

1. Create a feature branch: `git checkout -b feature/my-improvement`.
2. Commit your changes: `git commit -m "feat: add token f1 evaluation metric"`.
3. Verify all tests pass locally: `pytest`.
4. Push your branch to GitHub: `git push origin feature/my-improvement`.
5. Open a Pull Request referencing the relevant issue number.
6. A maintainer will review your PR, offer constructive feedback, and approve your Hacktoberfest contribution!
