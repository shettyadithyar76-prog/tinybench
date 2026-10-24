# TinyBench Datasets

This directory contains versioned benchmark datasets for evaluating small open-weight language models on CPU hardware.

## Dataset: `basic_v1.jsonl`

- **Version**: 1.0.0
- **Total Examples**: 35 (7 per category)
- **Format**: JSON Lines (`.jsonl`), UTF-8 encoded
- **License**: Creative Commons Zero (CC0) / Public Domain dedication. All questions and prompts were author-crafted specifically for TinyBench.

### Categories

1. **`multiple_choice`** (7 items): Evaluates knowledge retrieval and letter-option adherence (A, B, C, D) across physics, geography, computer science, biology, and chemistry.
2. **`factual_qa`** (7 items): Tests direct factual recall with exact-match canonical answers (world capitals, historical milestones, planetary science).
3. **`arithmetic_reasoning`** (7 items): Tests multi-step and single-step arithmetic, percentage calculations, linear equations, and geometric perimeter reasoning.
4. **`instruction_following`** (7 items): Tests deterministic rule-following including exact word counts, forbidden keywords, required prefixes, uppercase formatting, and ending tokens.
5. **`structured_output`** (7 items): Evaluates generation of syntactically valid JSON objects and arrays with specific schema keys and typed fields.

### Record Schema

Each line in the `.jsonl` file must be a self-contained JSON object satisfying:

```json
{
  "task_id": "unique_string_id",
  "category": "multiple_choice | factual_qa | arithmetic_reasoning | instruction_following | structured_output",
  "prompt": "Full prompt text provided to the model",
  "expected_answer": "Canonical expected answer (string, number, or object)",
  "scoring_method": "multiple_choice | exact_match | numeric | instruction_compliance | json_schema",
  "metadata": {
    "options": ["A", "B", "C", "D"],
    "required_keywords": ["keyword1"],
    "forbidden_keywords": ["forbidden1"],
    "exact_word_count": 3,
    "starts_with": "Answer: ",
    "ends_with": "confirmed",
    "required_keys": ["key1", "key2"]
  },
  "description": "Short human-readable summary of the test purpose"
}
```

### Intended Use and Limitations

- **Intended Use**: Rapid sanity-checking, CPU latency benchmarking, and reproducible regression testing of small language models (< 1B parameters) without requiring expensive compute.
- **Limitations**: With 35 examples, `basic_v1` is not a replacement for massive academic suites like MMLU or GSM8k. It is calibrated for quick, local CPU evaluation and clear deterministic grading. A model scoring well on `basic_v1` is not guaranteed to be universally superior.
