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
    GatedModelAccessError,
    ModelInferenceError,
    ModelLoadError,
    ResourceLimitError,
)
from tinybench.models.base import BaseModelAdapter, GenerationResult, ModelMetadata

logger = logging.getLogger(__name__)

# Verified metadata catalog for approved Hugging Face open-weight models
KNOWN_MODELS: dict[str, dict[str, Any]] = {
    "meta-llama/Llama-3.2-1B-Instruct": {
        "model_name": "Llama 3.2 1B Instruct",
        "family": "Llama",
        "developer": "Meta AI",
        "parameters": "1.23B",
        "variant": "Instruct",
        "license": "Llama 3.2 Community License",
        "license_url": "https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/blob/main/LICENSE.txt",
        "license_is_osi_approved": False,
        "supported_runtime": "transformers_cpu",
        "runtime_model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "weight_formats": ["safetensors"],
        "downloadable_artifacts": ["model.safetensors (2.47 GB)", "config.json", "tokenizer.json", "generation_config.json"],
        "download_size": "2.47 GB",
        "quantization": "None (BF16 unquantized)",
        "minimum_ram_mb": 2500.0,
        "recommended_ram_mb": 3500.0,
        "estimated_ram_mb": 3500.0,
        "status": "requires_access_approval",
        "status_reason": "Gated model. Requires accepting Meta's license at https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct and configuring HF_TOKEN environment variable.",
        "source_links": [
            "https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct",
            "https://llama.meta.com/",
        ],
        "verification_timestamp": "2026-10-25T20:30:00Z",
        "description": "Meta's ultra-compact 1.23B multilingual instruction-tuned model, optimized for on-device reasoning and text generation.",
    },
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": {
        "model_name": "SmolLM2 1.7B Instruct",
        "family": "SmolLM2",
        "developer": "Hugging Face",
        "parameters": "1.71B",
        "variant": "Instruct",
        "license": "Apache-2.0",
        "license_url": "https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/blob/main/LICENSE",
        "license_is_osi_approved": True,
        "supported_runtime": "transformers_cpu",
        "runtime_model_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "weight_formats": ["safetensors"],
        "downloadable_artifacts": ["model.safetensors (3.42 GB)", "config.json", "tokenizer.json", "generation_config.json"],
        "download_size": "3.42 GB",
        "quantization": "None (FP16/BF16 unquantized)",
        "minimum_ram_mb": 3000.0,
        "recommended_ram_mb": 4000.0,
        "estimated_ram_mb": 4000.0,
        "status": "available",
        "status_reason": "Open weights available for direct download under Apache-2.0 without gating.",
        "source_links": [
            "https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct",
        ],
        "verification_timestamp": "2026-10-25T20:30:00Z",
        "description": "Hugging Face's compact 1.71B instruction model trained on 11 trillion tokens, delivering high-grade reasoning on local consumer CPUs.",
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
        """Lookup verified model metadata and memory requirements."""
        known = KNOWN_MODELS.get(self.model_id)
        if known:
            # Dynamically check status for gated models
            status = known["status"]
            status_reason = known["status_reason"]
            if self.model_id == "meta-llama/Llama-3.2-1B-Instruct":
                token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
                if token:
                    status = "available"
                    status_reason = "Authenticated via HF_TOKEN environment variable."
                else:
                    status = "requires_access_approval"
                    status_reason = "Gated repository. Configure HF_TOKEN environment variable after accepting terms at https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct."

            return ModelMetadata(
                model_id=self.model_id,
                model_name=known["model_name"],
                family=known["family"],
                developer=known.get("developer", "Not verified"),
                parameters=known["parameters"],
                variant=known.get("variant", "Instruct"),
                license=known["license"],
                license_url=known.get("license_url", "Not verified"),
                license_is_osi_approved=known["license_is_osi_approved"],
                supported_runtime=known.get("supported_runtime", "transformers_cpu"),
                runtime_model_id=known.get("runtime_model_id", self.model_id),
                weight_formats=known.get("weight_formats", ["safetensors"]),
                downloadable_artifacts=known.get("downloadable_artifacts", []),
                download_size=known.get("download_size", "Not verified"),
                quantization=known.get("quantization", "Not verified"),
                minimum_ram_mb=known.get("minimum_ram_mb", 2000.0),
                recommended_ram_mb=known.get("recommended_ram_mb", 4000.0),
                estimated_ram_mb=known["estimated_ram_mb"],
                status=status,
                status_reason=status_reason,
                source_links=known.get("source_links", []),
                verification_timestamp=known.get("verification_timestamp", "2026-10-25T20:30:00Z"),
                backend_type="transformers_cpu",
                revision=self.revision,
                description=known["description"],
            )

        # Fallback for unrecognized models
        return ModelMetadata(
            model_id=self.model_id,
            model_name=self.model_id.split("/")[-1],
            family="Custom HF",
            developer="Not verified",
            parameters="Not verified",
            variant="Not verified",
            license="Refer to HF Hub card",
            license_url="Not verified",
            license_is_osi_approved=False,
            supported_runtime="transformers_cpu",
            runtime_model_id=self.model_id,
            weight_formats=["safetensors"],
            downloadable_artifacts=["Not verified"],
            download_size="Not verified",
            quantization="Not verified",
            minimum_ram_mb=2000.0,
            recommended_ram_mb=4000.0,
            estimated_ram_mb=3000.0,
            status="unavailable",
            status_reason="Unverified model identity not listed in TinyBench approved catalogue.",
            source_links=[f"https://huggingface.co/{self.model_id}"],
            verification_timestamp="2026-10-25T20:30:00Z",
            backend_type="transformers_cpu",
            revision=self.revision,
            description="Custom Hugging Face model not part of the approved TinyBench catalogue.",
        )

    def load(self) -> float:
        """Load tokenizer and model weights onto CPU with hardware safety checks."""
        meta = self.get_metadata()

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if self.model_id == "meta-llama/Llama-3.2-1B-Instruct" and not token:
            # Check if model snapshot exists in local HF cache before giving up
            from tinybench.models.downloader import is_model_installed
            if not is_model_installed(self.model_id):
                raise GatedModelAccessError(
                    "meta-llama/Llama-3.2-1B-Instruct is a gated model. "
                    "You must accept Meta's license at https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct "
                    "and supply your Hugging Face access token via the HF_TOKEN environment variable."
                )

        avail_ram_mb = psutil.virtual_memory().available / (1024 * 1024)
        if avail_ram_mb < meta.minimum_ram_mb:
            raise ResourceLimitError(
                f"Insufficient system memory: {self.model_id} requires at least {meta.minimum_ram_mb:.0f} MB RAM, "
                f"but only {avail_ram_mb:.0f} MB is available on this system."
            )

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as err:
            raise ModelLoadError(
                f"Transformers and PyTorch are required to run {self.model_id}: {err}"
            ) from err

        # Check whether `accelerate` is available
        try:
            import accelerate  # noqa: F401
            _has_accelerate = True
        except ImportError:
            _has_accelerate = False

        t0 = time.perf_counter()
        try:
            logger.info(f"Loading tokenizer for {self.model_id}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                revision=self.revision,
                token=token,
            )

            # Ensure pad token is defined
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            torch_dtype = torch.float32 if self.dtype == "float32" else torch.bfloat16

            logger.info(f"Loading model weights for {self.model_id} on CPU...")
            load_kwargs: dict[str, Any] = {
                "revision": self.revision,
                "torch_dtype": torch_dtype,
                "token": token,
            }
            if _has_accelerate:
                load_kwargs["device_map"] = "cpu"
                load_kwargs["low_cpu_mem_usage"] = True

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **load_kwargs,
            )
            self.model.eval()

            self._is_loaded = True
            self.load_duration = round(time.perf_counter() - t0, 4)
            return self.load_duration

        except Exception as exc:
            self._is_loaded = False
            msg = str(exc)
            if "gated" in msg.lower() or "401" in msg or "403" in msg:
                raise GatedModelAccessError(
                    f"Access denied for gated model '{self.model_id}'. "
                    "Verify you have been granted access on Hugging Face and that HF_TOKEN is configured: "
                    f"{exc}"
                ) from exc
            raise ModelLoadError(f"Failed to load Transformers model '{self.model_id}': {exc}") from exc

    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """Run CPU inference on prompt with measured latency and token metrics."""
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
                metadata={
                    "dtype": str(self.dtype),
                    "device": "cpu",
                    "tokens_per_second": round(gen_token_count / latency, 2) if latency > 0 else 0.0,
                },
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
