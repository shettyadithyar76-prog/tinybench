# TinyBench ⚡

[![Tests](https://github.com/your-username/tinybench/actions/workflows/tests.yml/badge.svg)](https://github.com/your-username/tinybench/actions)
[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Hardware](https://img.shields.io/badge/hardware-CPU%20Only-orange.svg)]()
[![Hacktoberfest](https://img.shields.io/badge/Hacktoberfest-2026%20Ready-blueviolet.svg)](docs/beginner_issues.md)

**A lightweight, reproducible open-weight language model benchmarking toolkit engineered specifically for ordinary CPU-only laptops.**

---

## 🌟 Why TinyBench?

Evaluating language models often requires expensive GPU clusters, paid inference APIs, and heavyweight benchmarking suites designed for servers. 

Students, hobbyists, and independent researchers working on everyday laptops are frequently left behind. Furthermore, comparing models fairly is challenging due to varying prompt templates, non-deterministic decoders, and lack of hardware memory transparency.

**TinyBench solves this by providing:**
- 💻 **CPU-First Execution**: Run local models (like `SmolLM2-135M` and `Qwen2.5-0.5B`) directly on your laptop's CPU. No CUDA, no GPU, and no paid API keys required.
- 🎯 **Objective Grading**: Deterministic evaluation of factual recall, multiple-choice questions, arithmetic reasoning, and strict rule-based instruction compliance.
- ⏱️ **Monotonic Latency & RAM Profiling**: Accurate measurement of cold-start initialization, mean/median inference latency, and peak resident memory (RSS).
- 🧪 **Zero-Download Mock Engine**: Fully functional deterministic mock backends allow contributors to run tests, experiment with pipelines, and preview the UI in seconds without downloading gigabytes of weights.
- 📊 **Streamlit Dashboard & CLI**: Clean, research-oriented dashboard with Plotly charts, SQLite run history, and single-click CSV/JSON exports.

---

## 🚀 Quickstart

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/tinybench.git
cd tinybench

# Create and activate a virtual environment (Python 3.11+)
python -m venv venv
# On Windows:
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate

# Install with development & dashboard dependencies
pip install -e ".[all]"
```

### 2. Verify with Zero-Download Mock Evaluation

Run an immediate 10-example evaluation without downloading any model weights:

```bash
python -m tinybench.cli evaluate --model mock --limit 10
```

Output:
```text
# TinyBench Evaluation Summary — TinyBench Mock Engine (Accurate)
- Run ID: run_20261024_122550_3727f8
- Dataset: basic_v1.jsonl (10 examples)
- Accuracy: 100.00% | Exact Match: 90.00% | Median Latency: 0.005 s
- Peak Process RSS: 34.8 MB
```

### 3. Launch the Interactive Dashboard

```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to explore interactive Plotly charts, model comparisons, and per-task outputs.

---

## 💻 Command-Line Interface (CLI)

TinyBench comes with an intuitive CLI:

| Command | Description | Example |
| :--- | :--- | :--- |
| `list-models` | Show registered models, parameter counts, licenses, and RAM requirements. | `python -m tinybench.cli list-models` |
| `validate-dataset` | Verify a dataset's JSON syntax, schema compliance, and balance. | `python -m tinybench.cli validate-dataset datasets/basic_v1.jsonl` |
| `evaluate` | Run a benchmark against a mock or open-weight model. | `python -m tinybench.cli evaluate --model smollm2-135m --limit 5` |
| `compare` | Compare two or more models or historical runs side-by-side. | `python -m tinybench.cli compare --models mock,mock-noisy --limit 10` |
| `export` | Export run results to machine-readable CSV or JSON. | `python -m tinybench.cli export --run-id <RUN_ID> --format csv` |
| `history` | List previous benchmark runs from the local SQLite index. | `python -m tinybench.cli history` |

---

## 🤖 Supported Open-Weight Models

TinyBench supports both built-in CPU configurations and custom Hugging Face Causal LM checkpoints:

| Model ID | Family | Parameters | Est. RAM | License | OSI Approved |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `mock` | MockEngine | 0M (Simulated) | ~5 MB | MIT | ✅ Yes |
| `mock-noisy` | MockEngine | 0M (Simulated) | ~5 MB | MIT | ✅ Yes |
| `smollm2-135m` | SmolLM2 | 135M | ~400 MB | Apache-2.0 | ✅ Yes |
| `qwen2.5-0.5b` | Qwen2.5 | 490M | ~1,200 MB | Apache-2.0 | ✅ Yes |
| `smollm2-360m` | SmolLM2 | 360M | ~900 MB | Apache-2.0 | ✅ Yes |
| Custom HF Repo | Any | User-defined | Configurable | Check Repo | Dependent |

> [!NOTE]
> **Open Weights vs Open Source**: TinyBench refers to models like SmolLM2 and Qwen2.5 as *open-weight models*. Model weights may be available for local execution, but open weights are not automatically equivalent to an OSI-approved software license. Always verify model licenses before commercial use.

---

## 📁 Standard Benchmark Dataset (`basic_v1.jsonl`)

The initial benchmark dataset contains **35 original, carefully crafted tasks** (7 per category) specifically designed for deterministic CPU grading:

1. **`multiple_choice`**: Objective 4-option questions testing physics, geography, biology, chemistry, and CS.
2. **`factual_qa`**: Direct factual recall with normalized string matching and aliases (e.g. world capitals, currency).
3. **`arithmetic_reasoning`**: Word problems, multiplication, division, linear algebra, and geometry perimeter.
4. **`instruction_following`**: Strict deterministic constraints (exact word counts, forbidden keywords, required prefixes, casing rules).
5. **`structured_output`**: Valid JSON object and array synthesis with required schema keys and typed fields.

---

## 🔬 Research Methodology & Integrity

TinyBench is designed to foster sound empirical practices:
- **Reproducibility**: Supports fixed random seeds (`seed=42`) and records operating system, processor, Python version, and PyTorch versions with every run.
- **Controlled Comparison**: Side-by-side comparisons run against identical benchmark tasks with identical decoding configurations.
- **Transparent Errors**: Failed cases (malformed JSON, timeouts, out-of-memory errors) are tracked and displayed, never silently hidden.
- **Methodological Qualifications**: We document that process-level Resident Set Size (RSS) captures process physical memory, but does not capture external kernel drivers or OS file caches.
- **No Over-Claims**: Achieving high accuracy on `basic_v1.jsonl` does not imply universal general intelligence.

---

## 🎃 Contributing to Hacktoberfest

TinyBench is an official Hacktoberfest project! We welcome contributors of all experience levels.

- Read our [Contributing Guide](CONTRIBUTING.md) to set up your environment.
- Check out the [10 Curated Beginner Issues](docs/beginner_issues.md) with step-by-step acceptance criteria.
- Review our [Project Roadmap](docs/roadmap.md) for long-term technical direction.
- Adhere to the [Code of Conduct](CODE_OF_CONDUCT.md).

---

## 📄 License

TinyBench is licensed under the [Apache License, Version 2.0](LICENSE).
