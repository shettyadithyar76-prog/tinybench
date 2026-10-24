"""Deterministic mock model adapters for zero-download testing and CI."""

from __future__ import annotations

import json
import random
import re
import time
from typing import Any

from tinybench.config import GenerationConfig
from tinybench.exceptions import ModelInferenceError, ModelLoadError
from tinybench.models.base import BaseModelAdapter, GenerationResult, ModelMetadata


class MockModelAdapter(BaseModelAdapter):
    """Deterministic, lightweight model simulator that requires no downloads or GPU."""

    def __init__(
        self,
        model_id: str = "mock",
        mode: str = "accurate",
        simulated_latency: float = 0.005,
        noise_rate: float = 0.0,
    ) -> None:
        super().__init__(model_id=model_id)
        self.mode = mode
        self.simulated_latency = simulated_latency
        self.noise_rate = noise_rate
        self.load_duration = 0.0

    def load(self) -> float:
        """Simulate loading model weights."""
        t0 = time.perf_counter()
        time.sleep(self.simulated_latency)
        self._is_loaded = True
        self.load_duration = round(time.perf_counter() - t0, 4)
        return self.load_duration

    def get_metadata(self) -> ModelMetadata:
        """Return mock model metadata."""
        return ModelMetadata(
            model_id=self.model_id,
            model_name=f"TinyBench Mock Engine ({self.mode.capitalize()})",
            family="MockEngine",
            parameters="0M (Simulated)",
            license="MIT",
            license_is_osi_approved=True,
            estimated_ram_mb=5.0,
            backend_type="mock",
            revision="v1.0.0",
            description=(
                "Deterministic mock inference backend for unit testing, CI pipelines, and "
                "rapid UI prototyping without downloading neural network weights."
            ),
        )

    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """Generate simulated responses based on prompt patterns."""
        if not self._is_loaded:
            raise ModelLoadError("Mock model must be loaded before calling generate().")

        t0 = time.perf_counter()
        if self.simulated_latency > 0:
            time.sleep(self.simulated_latency)

        # Failure mode simulation
        if self.mode == "failing" and "error" in prompt.lower():
            raise ModelInferenceError("Simulated inference exception triggered by mock engine.")

        # Simulate noise if noise_rate > 0
        rng = random.Random(config.seed + hash(prompt) % 10000)
        is_noisy = rng.random() < self.noise_rate

        reply = self._synthesize_response(prompt, is_noisy)
        latency = round(time.perf_counter() - t0, 4)

        prompt_tokens = max(1, len(prompt.split()))
        gen_tokens = max(1, len(reply.split()))

        return GenerationResult(
            text=reply,
            prompt_tokens=prompt_tokens,
            generated_tokens=gen_tokens,
            latency_seconds=latency,
            metadata={"mock_mode": self.mode, "was_noised": is_noisy},
        )

    def _synthesize_response(self, prompt: str, is_noisy: bool) -> str:
        """Produce pattern-matched response for prompt."""
        p_lower = prompt.lower()

        # MCQ pattern: "A) ... B) ... C) ... D)"
        if "answer with the option letter" in p_lower or "a)" in p_lower and "b)" in p_lower:
            if "speed of light" in p_lower:
                return "B" if is_noisy else "A"
            if "largest ocean" in p_lower:
                return "A" if is_noisy else "D"
            if "binary search tree" in p_lower:
                return "C" if is_noisy else "B"
            if "powerhouse" in p_lower:
                return "B" if is_noisy else "C"
            if "electrical resistance" in p_lower:
                return "A" if is_noisy else "D"
            if "symbol for gold" in p_lower:
                return "A" if is_noisy else "B"
            if "earth's atmosphere" in p_lower:
                return "A" if is_noisy else "C"
            return "B" if is_noisy else "A"

        # Math queries
        if "15 apples" in p_lower and "27 more" in p_lower:
            return "35" if is_noisy else "42"
        if "12 multiplied by 8" in p_lower:
            return "84" if is_noisy else "96"
        if "240 miles in 4 hours" in p_lower:
            return "50" if is_noisy else "60"
        if "25 percent of 200" in p_lower:
            return "40" if is_noisy else "50"
        if "5 red balls" in p_lower and "7 blue balls" in p_lower:
            return "10" if is_noisy else "9"
        if "3x + 6 = 24" in p_lower:
            return "5" if is_noisy else "6"
        if "perimeter of a rectangle that has a length of 10" in p_lower:
            return "24" if is_noisy else "28"

        # Factual QA
        if "capital city of france" in p_lower:
            return "Lyon" if is_noisy else "Paris"
        if "first person to walk on the moon" in p_lower:
            return "Buzz Aldrin" if is_noisy else "Neil Armstrong"
        if "continents are there" in p_lower:
            return "6" if is_noisy else "7"
        if "boiling point of water" in p_lower:
            return "90" if is_noisy else "100"
        if "romeo and juliet" in p_lower:
            return "Charles Dickens" if is_noisy else "William Shakespeare"
        if "official currency of japan" in p_lower:
            return "Won" if is_noisy else "Yen"
        if "largest by mass" in p_lower or "largest planet" in p_lower:
            return "Saturn" if is_noisy else "Jupiter"

        # Instruction Following
        if "exactly three words" in p_lower:
            return "blue and wide sky" if is_noisy else "blue vast bright"
        if "without using the words 'liquid' or 'drink'" in p_lower:
            if is_noisy:
                return "Water is a clear liquid that people drink."
            return "Water is a vital molecular substance composed of hydrogen and oxygen."
        if "start your response with 'answer: '" in p_lower:
            return "The city is Madrid." if is_noisy else "Answer: Madrid"
        if "all uppercase" in p_lower:
            return "yellow" if is_noisy else "YELLOW"
        if "'sunshine' and the word 'beach'" in p_lower:
            return "It was warm at the shore." if is_noisy else "The warm sunshine bathed the golden beach."
        if "numbered list" in p_lower:
            return "11, 13, 17" if is_noisy else "1. 11\n2. 13\n3. 17"
        if "final word must be 'confirmed'" in p_lower:
            return "10 finished" if is_noisy else "5 plus 5 equals 10 confirmed"

        # Structured Output JSON
        if "tokyo" in p_lower and "city" in p_lower:
            return '{"city": "Tokyo"}' if is_noisy else '{"city": "Tokyo", "country": "Japan"}'
        if "book" in p_lower and "title" in p_lower:
            return '{"title": "1984"}' if is_noisy else '{"title": "1984", "author": "George Orwell", "year": 1949}'
        if "movie" in p_lower and "rating" in p_lower:
            return '{"movie": "Inception", "rating": "nine"}' if is_noisy else '{"movie": "Inception", "rating": 9}'
        if "primary colors" in p_lower and "array" in p_lower:
            return '["red"]' if is_noisy else '["red", "blue", "yellow"]'
        if "capital=value" in p_lower or "capital of canada" in p_lower:
            return "Ottawa" if is_noisy else "CAPITAL=Ottawa"
        if "status" in p_lower and "code" in p_lower:
            return '{"status": "ok", "code": "200"}' if is_noisy else '{"status": "ok", "code": 200}'
        if "skills" in p_lower and "name" in p_lower:
            return '{"name": "Alice", "age": 28, "skills": ["python", "machine learning"]}'

        return "Sample response"

    def unload(self) -> None:
        """Release mock resources."""
        self._is_loaded = False
