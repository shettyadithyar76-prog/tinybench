# TinyBench Project Roadmap 🗺️

TinyBench is an open-source initiative designed to make language model benchmarking democratic, reproducible, and accessible on everyday hardware.

---

## 📌 Phase 1: Foundation (v0.1.0 — Current Release) ✅
- [x] Standardized `BenchmarkItem` dataset schema and streaming loader.
- [x] Curated 35-item `basic_v1.jsonl` dataset across 5 core categories.
- [x] Objective deterministic grading (Accuracy, Exact Match, Rule-based Instruction Compliance, Numeric Match, JSON Schema).
- [x] Monotonic latency tracking (mean, median, p95, cold start) and process RSS memory tracking.
- [x] Deterministic mock adapters for instant zero-download CI testing.
- [x] Hugging Face Transformers CPU backend for SmolLM2 and Qwen2.5.
- [x] Complete CLI (`list-models`, `validate-dataset`, `evaluate`, `compare`, `export`, `history`).
- [x] Streamlit web dashboard with interactive Plotly visualizer and SQLite persistence.
- [x] 40+ unit and integration tests passing in under 15 seconds.

---

## 📌 Phase 2: Quantization & Alternate Backends (v0.2.0)
- [ ] **GGUF / `llama.cpp` Adapter**: Support 4-bit and 8-bit quantized models (`.gguf`) for even lower memory consumption on 4GB-8GB RAM laptops.
- [ ] **ONNX Runtime Backend**: Integrate CPU-optimized ONNX runtime with graph optimizations.
- [ ] **Apple Silicon Accelerate / NEON Support**: Enhance CPU execution on ARM architectures (M1/M2/M3/M4).

---

## 📌 Phase 3: Enhanced Diagnostics & Datasets (v0.3.0)
- [ ] **Streaming Latency / TTFT**: Track Time-To-First-Token and sustained tokens-per-second (TPS) generation throughput.
- [ ] **Dataset v2 Expansion**:
  - Add `code_completion` (Python syntax and docstring test cases).
  - Add `safety_refusal` (evaluating benign compliance vs adversarial refusal).
  - Add multilingual translation tasks (Spanish, French, German, Hindi).
- [ ] **Token F1 & ROUGE Scores**: Add optional overlap metrics alongside exact match.

---

## 📌 Phase 4: Ecosystem & Sharing (v1.0.0)
- [ ] **Public Leaderboard Export**: Single-click generation of static HTML / GitHub Markdown leaderboard cards.
- [ ] **Hugging Face Hub Dataset Sync**: Pull benchmark datasets directly from Hugging Face Datasets without requiring local files.
- [ ] **Plugin Architecture**: Dynamically discover external third-party model adapters and custom evaluators.
