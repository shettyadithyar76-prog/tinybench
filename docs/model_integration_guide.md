# TinyBench — Approved Model Catalogue & Integration Guide

This guide documents the single source of truth for TinyBench's approved open-weight language models, their official upstream repositories, licensing, hardware safety, and download workflows.

---

## 1. Approved Model Catalogue

TinyBench standardizes on exactly three verified open-weight models optimized for local CPU-only benchmarking:

| Model Display Name | Exact Upstream Repository | Developer | Parameter Count | Quantization | Download Size | Official License | Supported Runtime |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Llama 3.2 1B Instruct** | `meta-llama/Llama-3.2-1B-Instruct` | Meta AI | ~1.23B | None (BF16) | 2.47 GB | [Llama 3.2 Community License](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/blob/main/LICENSE.txt) | `transformers_cpu` |
| **SmolLM2 1.7B Instruct** | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | Hugging Face | ~1.71B | None (FP16/BF16) | 3.42 GB | [Apache-2.0](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/blob/main/LICENSE) | `transformers_cpu` |
| **Gemma 4 E2B** | `gemma4:e2b` | Google DeepMind | ~2B effective | 4-bit (q4_K_M) | 1.8 GB | [Apache-2.0](https://gemma.ai/models/gemma-4/) | `ollama` |

---

## 2. Model Specific Requirements & Authentication

### Llama 3.2 1B Instruct (`meta-llama/Llama-3.2-1B-Instruct`)
- **License**: Gated community license.
- **Access Approval**: Users must accept Meta's license agreement at [Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct).
- **Authentication**: Set the `HF_TOKEN` environment variable or provide your Hugging Face access token when prompted.
- **RAM Footprint**: ~3.5 GB recommended system RAM for CPU inference.

### SmolLM2 1.7B Instruct (`HuggingFaceTB/SmolLM2-1.7B-Instruct`)
- **License**: Apache-2.0 (Permissive open source).
- **Access Approval**: Public open download; no token or gating required.
- **RAM Footprint**: ~4.0 GB recommended system RAM for CPU inference.

### Gemma 4 E2B (`gemma4:e2b`)
- **License**: Apache-2.0.
- **Runtime**: Local [Ollama](https://ollama.com) runtime daemon (`http://localhost:11434`).
- **Ollama Verification**: Start the Ollama service (`ollama serve`) and pull the model with `ollama pull gemma4:e2b` or via TinyBench's download manager.
- **RAM Footprint**: ~2.5 GB recommended system RAM for 4-bit quantized execution.

---

## 3. Explicit Model Downloads

TinyBench never downloads weights automatically on launch. Model downloads must be explicitly initiated:

### Via CLI:
```bash
# Download SmolLM2 1.7B Instruct
tinybench download --model smollm2-1.7b

# Download gated Llama 3.2 with HF_TOKEN
tinybench download --model llama-3.2-1b --token <YOUR_HF_TOKEN>

# Pull Gemma 4 E2B into Ollama
tinybench download --model gemma4-e2b
```

### Via Streamlit UI:
1. Navigate to **Run Benchmark**.
2. Select the desired approved model.
3. Review disk space and hardware compatibility.
4. Click **⬇️ Download & Verify Model**.

---

## 4. Hardware Safety & Pre-Flight Checks

Before allocating model weights onto CPU:
1. **Memory Budget**: TinyBench inspects `psutil.virtual_memory().available` against the model's `minimum_ram_mb`. If insufficient, execution is prevented to protect host stability.
2. **Disk Space Check**: `check_disk_space_for_model` verifies that at least 2.5–4.5 GB of free disk space is available before downloading.
3. **Clean Teardown**: `unload()` clears model references and invokes `gc.collect()` after evaluation completes.

---

## 5. Benchmarking Execution

```bash
# Evaluate SmolLM2 on basic benchmark dataset (10 examples)
tinybench evaluate --model smollm2-1.7b --limit 10

# Compare evaluation runs
tinybench compare --models smollm2-1.7b,llama-3.2-1b --limit 10
```
