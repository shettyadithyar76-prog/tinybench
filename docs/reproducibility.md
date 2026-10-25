# TinyBench — Reproducibility & Scientific Methodology

This document outlines the reproducibility standards, hardware telemetry protocols, and grading transparency enforced across TinyBench.

---

## 1. The Reproducibility Challenge in LLM Benchmarking

Many modern LLM leaderboards suffer from reproducibility crises caused by:
- **Non-deterministic decoding**: Unfixed seeds or stochastic sampling temperatures.
- **LLM-as-a-judge variability**: Using proprietary external models that undergo silent background updates.
- **Hidden hardware context**: Presenting throughput and latency numbers without documenting memory budgets, CPU core counts, or thread concurrency.
- **Unstated answer extraction rules**: Arbitrary prompt wrappers and regex matching rules that vary across implementations.

TinyBench directly eliminates these issues through deterministic design.

---

## 2. Deterministic Generation Standards

1. **Greedy Decoding Default**:
   - By default, `temperature=0.0` is enforced for benchmark runs, ensuring deterministic argmax token selection.
2. **Fixed Random Seeds**:
   - `seed=42` is set across PyTorch (`torch.manual_seed`), Python (`random.seed`), and NumPy (`np.random.seed`).
3. **Deterministic Answer Extraction**:
   - All evaluators use explicit, unit-tested regex and normalization rules (see [`docs/metrics.md`](metrics.md)). No probabilistic judges are utilized.

---

## 3. Telemetry & Hardware Metadata Capture

Every benchmark execution saves a `SystemInfo` record inside the run artifact:
- **Operating System**: OS name, release, and architecture (e.g., `Windows 11`, `Ubuntu 24.04 LTS`, `macOS Darwin`).
- **CPU Cores**: Physical cores vs logical hyperthreads.
- **Python Version**: Exact interpreter build.
- **PyTorch Version**: Framework version and CPU backend flags.
- **Available System RAM**: Resident memory measured prior to model allocation.

---

## 4. Latency & Memory Measurement Protocols

### Latency Measurement:
- Uses `time.perf_counter()` (high-resolution monotonic clock) for microsecond precision.
- Distinguishes between **cold-start latency** (first task, including weights and cache warm-up) and **steady-state inference latency**.
- Reports mean, median, min, max, and p95 latency percentiles.

### Memory Profiling:
- Measures **Resident Set Size (RSS)** via `psutil.Process().memory_info().rss`.
- Measures **Peak RSS** across task execution.
- *Methodological note*: Process RSS captures the physical memory assigned to the Python process. It does not measure external OS page cache or kernel drivers.

---

## 5. Machine-Readable Audit Trails

Every completed evaluation produces an immutable JSON artifact containing:
- Run ID (timestamped hash)
- Complete configuration parameters
- Hardware environment snapshot
- Aggregated category metrics
- Granular per-task inputs, model outputs, expected answers, latency values, and error traces

These artifacts can be exported to CSV or queried via SQLite (`runs/runs.db`) for meta-analyses and independent verification.
