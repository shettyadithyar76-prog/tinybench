"""Hugging Face Transformers model adapter for local CPU inference."""

from __future__ import annotations

import gc
import logging
import os
import time
from typing import Any
import psutil

from tinybench.config import GenerationConfig
from tinybench.exceptions import (
    ModelInferenceError,
    ModelLoadError,
    ResourceLimitError,
)
from tinybench.models.base import BaseModelAdapter, GenerationResult, ModelMetadata

logger = logging.getLogger(__name__)

# Known metadata catalog for standard CPU-friendly open-weight models
KNOWN_MODELS: dict[str, dict[str, Any]] = {
    "HuggingFaceTB/SmolLM2-135M-Instruct": {
        "model_name": "SmolLM2 135M Instruct",
        "family": "SmolLM2",
        "parameters": "135M",
        "license": "Apache-2.0",
        "license_is_osi_approved": True,
        "estimated_ram_mb": 400.0,
        "description": "Ultra-lightweight instruction-tuned model trained by Hugging Face on synthetic & curated tokens.",
    },
    "Qwen/Qwen2.5-0.5B-Instruct": {
        "model_name": "Qwen 2.5 0.5B Instruct",
        "family": "Qwen2.5",
        "parameters": "490M",
        "license": "Apache-2.0",
        "license_is_osi_approved": True,
        "estimated_ram_mb": 1200.0,
        "description": "Compact high-capability open-weight language model optimized for coding, mathematics, and reasoning.",
    },
    "HuggingFaceTB/SmolLM2-360M-Instruct": {
        "model_name": "SmolLM2 360M Instruct",
        "family": "SmolLM2",
        "parameters": "360M",
        "license": "Apache-2.0",
        "license_is_osi_approved": True,
        "estimated_ram_mb": 900.0,
        "description": "360M parameter SmolLM2 model offering stronger reasoning while remaining accessible on 8GB RAM laptops.",
    },
}


class TransformersModelAdapter(BaseModelAdapter):
    """Local CPU model backend utilizing Hugging Face Transformers and PyTorch."""

    def __init__(
        self,
        model_id: str,
        revision: str | None = None,
        dtype: str = "float32",
    ) -> None:
        super().__init__(model_id=model_id, revision=revision)
        self.dtype = dtype
        self.model: Any = None
        self.tokenizer: Any = None
        self.load_duration: float = 0.0

    def get_metadata(self) -> ModelMetadata:
        """Lookup or infer model metadata and memory requirements."""
        known = KNOWN_MODELS.get(self.model_id)
        if known:
            return ModelMetadata(
                model_id=self.model_id,
                model_name=known["model_name"],
                family=known["family"],
                parameters=known["parameters"],
                license=known["license"],
                license_is_osi_approved=known["license_is_osi_approved"],
                estimated_ram_mb=known["estimated_ram_mb"],
                backend_type="transformers_cpu",
                revision=self.revision,
                description=known["description"],
            )

        # Fallback for user-specified HuggingFace model repo
        return ModelMetadata(
            model_id=self.model_id,
            model_name=self.model_id.split("/")[-1],
            family="Custom HF",
            parameters="Custom",
            license="Refer to HF Hub card",
            license_is_osi_approved=False,
            estimated_ram_mb=1500.0,
            backend_type="transformers_cpu",
            revision=self.revision,
            description="Custom Hugging Face model loaded from local disk or Hugging Face Hub.",
        )

    def load(self) -> float:
        """Load tokenizer and model weights onto CPU with memory checks."""
        meta = self.get_metadata()
        avail_ram_mb = psutil.virtual_memory().available / (1024 * 1024)

        if avail_ram_mb < meta.estimated_ram_mb:
            raise ResourceLimitError(
                f"Insufficient memory: {self.model_id} requires ~{meta.estimated_ram_mb:.0f} MB RAM, "
                f"but only {avail_ram_mb:.0f} MB is available on this system."
            )

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as err:
            raise ModelLoadError(
                f"Transformers and PyTorch are required to run {self.model_id}: {err}"
            ) from err

        t0 = time.perf_counter()
        try:
            logger.info(f"Loading tokenizer for {self.model_id}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                revision=self.revision,
            )

            # Ensure pad token is defined
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            torch_dtype = torch.float32 if self.dtype == "float32" else torch.bfloat16

            logger.info(f"Loading model weights for {self.model_id} on CPU...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                revision=self.revision,
                torch_dtype=torch_dtype,
                device_map="cpu",
                low_cpu_mem_usage=True,
            )
            self.model.eval()

            self._is_loaded = True
            self.load_duration = round(time.perf_counter() - t0, 4)
            return self.load_duration

        except Exception as exc:
            self._is_loaded = False
            raise ModelLoadError(f"Failed to load Transformers model '{self.model_id}': {exc}") from exc

    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """Run CPU inference on prompt."""
        if not self._is_loaded or self.model is None or self.tokenizer is None:
            raise ModelLoadError("Model is not loaded. Call load() before generate().")

        try:
            import torch

            # Seed PyTorch for reproducibility
            torch.manual_seed(config.seed)

            # Format input using chat template if model has one
            if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
                messages = [{"role": "user", "content": prompt}]
                try:
                    formatted_input = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    formatted_input = prompt
            else:
                formatted_input = prompt

            inputs = self.tokenizer(formatted_input, return_tensors="pt")
            input_ids = inputs["input_ids"]
            prompt_token_count = int(input_ids.shape[1])

            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": config.max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }

            if config.temperature > 0.0:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = config.temperature
                gen_kwargs["top_p"] = config.top_p
            else:
                gen_kwargs["do_sample"] = False

            t0 = time.perf_counter()
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    **gen_kwargs,
                )
            latency = round(time.perf_counter() - t0, 4)

            # Extract generated tokens only (strip prompt)
            new_tokens = output_ids[0][prompt_token_count:]
            generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            gen_token_count = int(len(new_tokens))

            # Apply stop sequences if specified
            for stop_seq in config.stop_sequences:
                if stop_seq in generated_text:
                    generated_text = generated_text.split(stop_seq)[0].strip()

            return GenerationResult(
                text=generated_text,
                prompt_tokens=prompt_token_count,
                generated_tokens=gen_token_count,
                latency_seconds=latency,
                metadata={"dtype": str(self.dtype), "device": "cpu"},
            )

        except Exception as exc:
            raise ModelInferenceError(f"Generation error with model '{self.model_id}': {exc}") from exc

    def unload(self) -> None:
        """Release PyTorch CPU model weights and invoke garbage collection."""
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
        gc.collect()
        try:
            import torch
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
