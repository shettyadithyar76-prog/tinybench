# TinyBench Benchmark Dataset Specification (`basic_v1.jsonl`)

This specification defines the schema, categories, scoring methods, and validation rules for TinyBench benchmark datasets.

---

## 1. Overview & Objectives

TinyBench benchmarks language models on ordinary CPU hardware using deterministic, transparent, and reproducible grading criteria. The standard bundled dataset is located at [`datasets/basic_v1.jsonl`](../datasets/basic_v1.jsonl).

### Core Principles:
- **Zero LLM-as-a-Judge Bias**: Grading relies entirely on deterministic string normalization, multiple-choice parsing, mathematical equivalence, and JSON schema validation.
- **CPU Accessibility**: Task lengths and target answers are sized to allow small open-weight models (e.g. `SmolLM2-135M`, `Qwen2.5-0.5B`) to generate responses within seconds on CPU.
- **Machine-Readable**: Every line is a standalone, valid JSON object following JSON Lines (`.jsonl`) format.

---

## 2. Dataset Schema

Each entry in the JSONL dataset conforms to the `BenchmarkItem` Pydantic model:

```json
{
  "task_id": "mcq_001",
  "category": "multiple_choice",
  "prompt": "What is the primary gas found in Earth's atmosphere?\nA) Oxygen\nB) Nitrogen\nC) Carbon Dioxide\nD) Argon\nAnswer with the correct letter:",
  "expected_answer": "B",
  "scoring_method": "multiple_choice",
  "metadata": {
    "domain": "Earth Science",
    "difficulty": "elementary"
  }
}
```

### Field Definitions:

| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `task_id` | `string` | **Yes** | Unique identifier for the task (e.g., `mcq_001`, `fact_005`). Must be unique across the entire dataset. |
| `category` | `string` | **Yes** | One of the 5 supported categories: `multiple_choice`, `factual_qa`, `arithmetic_reasoning`, `instruction_following`, `structured_output`. |
| `prompt` | `string` | **Yes** | The complete input prompt delivered to the model's generation pipeline. |
| `expected_answer` | `string` \| `number` \| `object` | **Yes** | The ground-truth reference answer used for grading. |
| `scoring_method` | `string` | **Yes** | The evaluation method applied: `multiple_choice`, `exact_match`, `numeric_match`, `instruction_compliance`, or `json_schema`. |
| `metadata` | `object` | No | Optional dictionary containing subject domain, difficulty level, or rule constraints. |

---

## 3. Supported Task Categories & Scoring Rules

### 3.1. `multiple_choice`
- **Purpose**: Evaluates factual knowledge across physics, biology, chemistry, and computer science.
- **Format**: 4 distinct lettered options (A, B, C, D).
- **Scoring Method**: `multiple_choice`
  - Extracts the choice letter from common patterns (e.g., `B`, `(B)`, `Option B`, `Answer: B`).
  - Strict letter extraction prevents false positives from incidental word matches.

### 3.2. `factual_qa`
- **Purpose**: Evaluates direct factual recall (e.g., world capitals, inventions, currencies).
- **Scoring Method**: `exact_match`
  - Normalizes text by lowercasing, stripping leading/trailing punctuation, removing English articles (`a`, `an`, `the`), and trimming whitespace.
  - Compares normalized model output with normalized expected answer.

### 3.3. `arithmetic_reasoning`
- **Purpose**: Evaluates elementary multi-step math, integer multiplication, linear equations, and perimeter geometry.
- **Scoring Method**: `numeric_match`
  - Extracts numerical values (including negatives and floating-point numbers) from the output string.
  - Evaluates numerical equality within a floating-point tolerance of `1e-4`.

### 3.4. `instruction_following`
- **Purpose**: Evaluates adherence to deterministic structural constraints without LLM judges.
- **Scoring Method**: `instruction_compliance`
  - Evaluates rules defined in `metadata.rules`:
    - `exact_word_count`: Exact number of whitespace-delimited words.
    - `max_word_count`: Upper bound on word count.
    - `forbidden_words`: Substrings that must NOT appear in the output.
    - `required_words`: Substrings that MUST appear in the output.
    - `required_prefix`: Required leading substring.
    - `required_suffix`: Required trailing substring.
    - `casing`: Forced `UPPERCASE` or `lowercase`.

### 3.5. `structured_output`
- **Purpose**: Evaluates valid JSON object and array synthesis.
- **Scoring Method**: `json_schema`
  - Locates and extracts JSON code fences or raw JSON substrings.
  - Verifies JSON syntax using `json.loads()`.
  - Verifies presence and types of all required top-level schema keys specified in `metadata.required_keys`.

---

## 4. Validating a Dataset

Use the TinyBench CLI to validate any existing or newly authored dataset:

```bash
python -m tinybench.cli validate-dataset datasets/basic_v1.jsonl
```

The validator verifies:
1. Syntactic validity of every JSON line.
2. Pydantic schema conformity for all items.
3. Task ID uniqueness (flags duplicate IDs).
4. Category distribution and balance statistics.
