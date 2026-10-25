"""Ollama runtime model adapter for local CPU inference."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any
import urllib.error
import urllib.request
import psutil

from tinybench.config import GenerationConfig
from tinybench.exceptions import (
    ModelInferenceError,
    ModelLoadError,
    ResourceLimitError,
    RuntimeUnavailableError,
)
from tinybench.models.base import BaseModelAdapter, GenerationResult, ModelMetadata

logger = logging.getLogger(__name__)

def get_ollama_base_url() -> str:
    """Retrieve effective Ollama daemon endpoint URL from environment or default."""
    url = os.environ.get("OLLAMA_HOST") or os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434"
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"
    return url.rstrip("/")

DEFAULT_OLLAMA_URL = get_ollama_base_url()

# Verified metadata for approved Ollama models
OLLAMA_CATALOGUE: dict[str, dict[str, Any]] = {
    "gemma4:e2b": {
        "model_name": "Gemma 4 E2B",
        "family": "Gemma 4",
        "developer": "Google DeepMind",
        "parameters": "2B effective",
        "variant": "E2B (Edge Multimodal)",
        "license": "Apache-2.0",
        "license_url": "https://gemma.ai/models/gemma-4/",
        "license_is_osi_approved": True,
        "supported_runtime": "ollama",
        "runtime_model_id": "gemma4:e2b",
        "weight_formats": ["Ollama blob", "GGUF"],
        "downloadable_artifacts": ["gemma4:e2b (Ollama manifest & Q4_K_M blob)"],
        "download_size": "1.8 GB",
        "quantization": "4-bit (q4_K_M)",
        "minimum_ram_mb": 1800.0,
        "recommended_ram_mb": 2500.0,
        "estimated_ram_mb": 2500.0,
        "source_links": [
            "https://ollama.com/library/gemma4",
            "https://gemma.ai/models/gemma-4/",
        ],
        "verification_timestamp": "2026-10-25T20:30:00Z",
        "description": "Google DeepMind's ultra-efficient 2B edge model with native audio/vision understanding and extended context, built for low-resource deployment.",
    },
    "deepseek-r1:1.5b": {
        "model_name": "DeepSeek R1 Distill 1.5B",
        "family": "DeepSeek-R1",
        "developer": "DeepSeek AI",
        "parameters": "1.5B",
        "variant": "Distill Qwen (Reasoning)",
        "license": "MIT",
        "license_url": "https://opensource.org/licenses/MIT",
        "license_is_osi_approved": True,
        "supported_runtime": "ollama",
        "runtime_model_id": "deepseek-r1:1.5b",
        "weight_formats": ["Ollama blob", "GGUF"],
        "downloadable_artifacts": ["deepseek-r1:1.5b (Ollama manifest & Q4_K_M blob)"],
        "download_size": "1.1 GB",
        "quantization": "4-bit (q4_K_M)",
        "minimum_ram_mb": 1400.0,
        "recommended_ram_mb": 2200.0,
        "estimated_ram_mb": 1800.0,
        "source_links": [
            "https://ollama.com/library/deepseek-r1",
            "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        ],
        "verification_timestamp": "2026-10-26T00:00:00Z",
        "description": "DeepSeek's distilled 1.5B reasoning model with native chain-of-thought capabilities, optimized for edge reasoning on CPU.",
    },
}


def check_ollama_daemon(base_url: str | None = None, timeout: float = 2.0) -> tuple[bool, str]:
    """Check if local Ollama service daemon is reachable and responding.

    Returns:
        Tuple of (is_reachable, version_or_error_message).
    """
    effective_url = (base_url or get_ollama_base_url()).rstrip("/")
    url = f"{effective_url}/api/version"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                version = data.get("version", "unknown")
                return True, f"Ollama {version} active"
    except Exception as exc:
        return False, (
            f"Ollama daemon is not reachable at {effective_url}. "
            "Please ensure Ollama is installed (https://ollama.com) and running ('ollama serve'), "
            "or configure OLLAMA_HOST if using a remote or tunneled daemon."
        )
    return False, "Ollama daemon returned non-200 status code."


def is_ollama_model_installed(
    model_tag: str = "gemma4:e2b",
    base_url: str | None = None,
    timeout: float = 2.0,
) -> bool:
    """Verify whether a specific model tag is already pulled and available in Ollama."""
    effective_url = (base_url or get_ollama_base_url()).rstrip("/")
    is_running, _ = check_ollama_daemon(effective_url, timeout=timeout)
    if not is_running:
        return False

    url = f"{effective_url}/api/tags"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                installed_names = [m.get("name", "") for m in data.get("models", [])]
                # Check exact match or with :latest
                clean_tag = model_tag.lower().strip()
                for name in installed_names:
                    name_lower = name.lower()
                    if name_lower == clean_tag or name_lower.startswith(f"{clean_tag}:"):
                        return True
    except Exception as exc:
        logger.debug(f"Failed to query Ollama tags: {exc}")
    return False


def is_ollama_model_running(
    model_tag: str = "gemma4:e2b",
    base_url: str | None = None,
    timeout: float = 2.0,
) -> bool:
    """Check if model is currently resident in Ollama's active memory via /api/ps."""
    effective_url = (base_url or get_ollama_base_url()).rstrip("/")
    url = f"{effective_url}/api/ps"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                running_names = [m.get("name", "").lower() for m in data.get("models", [])]
                clean = model_tag.lower().strip()
                return any(clean in name or name.startswith(clean) for name in running_names)
    except Exception:
        pass
    return False


class OllamaModelAdapter(BaseModelAdapter):
    """Adapter for running verified models via the local Ollama runtime engine."""

    def __init__(
        self,
        model_id: str = "gemma4:e2b",
        base_url: str | None = None,
        revision: str | None = None,
    ) -> None:
        super().__init__(model_id=model_id, revision=revision)
        self.base_url = (base_url or get_ollama_base_url()).rstrip("/")
        self.load_duration: float = 0.0

    def get_metadata(self) -> ModelMetadata:
        """Lookup verified model metadata and check local Ollama availability."""
        known = OLLAMA_CATALOGUE.get(self.model_id)
        if not known:
            # Fallback for unexpected Ollama tag
            return ModelMetadata(
                model_id=self.model_id,
                model_name=self.model_id,
                family="Ollama",
                developer="Not verified",
                parameters="Not verified",
                variant="Not verified",
                license="Refer to Ollama library",
                license_url="Not verified",
                license_is_osi_approved=False,
                supported_runtime="ollama",
                runtime_model_id=self.model_id,
                weight_formats=["Ollama blob"],
                downloadable_artifacts=["Not verified"],
                download_size="Not verified",
                quantization="Not verified",
                minimum_ram_mb=2000.0,
                recommended_ram_mb=3000.0,
                estimated_ram_mb=2500.0,
                status="unavailable",
                status_reason=f"Model tag '{self.model_id}' is not in the approved TinyBench catalogue.",
                source_links=[f"https://ollama.com/library/{self.model_id.split(':')[0]}"],
                verification_timestamp="2026-10-25T20:30:00Z",
                backend_type="ollama",
                revision=self.revision,
                description="Custom Ollama model tag.",
            )

        # Dynamic status check against local Ollama service
        is_running, daemon_msg = check_ollama_daemon(self.base_url)
        if not is_running:
            status = "unavailable"
            status_reason = daemon_msg
        else:
            is_pulled = is_ollama_model_installed(self.model_id, self.base_url)
            if is_pulled:
                status = "available"
                status_reason = "Model is installed locally in Ollama and ready for CPU benchmarking."
            else:
                status = "available"
                status_reason = f"Ollama daemon active. Model '{self.model_id}' requires download via 'ollama pull {self.model_id}'."

        return ModelMetadata(
            model_id=self.model_id,
            model_name=known["model_name"],
            family=known["family"],
            developer=known["developer"],
            parameters=known["parameters"],
            variant=known["variant"],
            license=known["license"],
            license_url=known["license_url"],
            license_is_osi_approved=known["license_is_osi_approved"],
            supported_runtime=known["supported_runtime"],
            runtime_model_id=known["runtime_model_id"],
            weight_formats=known["weight_formats"],
            downloadable_artifacts=known["downloadable_artifacts"],
            download_size=known["download_size"],
            quantization=known["quantization"],
            minimum_ram_mb=known["minimum_ram_mb"],
            recommended_ram_mb=known["recommended_ram_mb"],
            estimated_ram_mb=known["estimated_ram_mb"],
            status=status,
            status_reason=status_reason,
            source_links=known["source_links"],
            verification_timestamp=known["verification_timestamp"],
            backend_type="ollama",
            revision=self.revision,
            description=known["description"],
        )

    def load(self) -> float:
        """Verify Ollama runtime health and model presence before running benchmark."""
        is_running, err_msg = check_ollama_daemon(self.base_url)
        if not is_running:
            raise RuntimeUnavailableError(err_msg)

        if not is_ollama_model_installed(self.model_id, self.base_url):
            raise ModelLoadError(
                f"Model '{self.model_id}' is not installed in local Ollama storage. "
                f"Execute 'ollama pull {self.model_id}' or initiate download from TinyBench UI before benchmarking."
            )

        meta = self.get_metadata()
        is_already_resident = is_ollama_model_running(self.model_id, self.base_url)
        avail_ram_mb = psutil.virtual_memory().available / (1024 * 1024)

        if not is_already_resident and avail_ram_mb < meta.minimum_ram_mb:
            raise ResourceLimitError(
                f"Insufficient memory: {self.model_id} requires at least {meta.minimum_ram_mb:.0f} MB RAM to load, "
                f"but only {avail_ram_mb:.0f} MB is available on this system."
            )

        t0 = time.perf_counter()
        # Ping model with an empty warmup to preload weights into memory
        try:
            req_data = json.dumps({
                "model": self.model_id,
                "prompt": "",
                "stream": False,
                "keep_alive": "5m",
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=req_data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                if resp.status == 200:
                    self._is_loaded = True
                    self.load_duration = round(time.perf_counter() - t0, 4)
                    return self.load_duration
        except Exception as exc:
            self._is_loaded = False
            raise ModelLoadError(f"Failed to preload Ollama model '{self.model_id}': {exc}") from exc

        self._is_loaded = True
        self.load_duration = round(time.perf_counter() - t0, 4)
        return self.load_duration

    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """Run genuine CPU inference through local Ollama API and return measured metrics."""
        if not self._is_loaded:
            # Auto-verify load
            self.load()

        options: dict[str, Any] = {
            "num_predict": config.max_new_tokens,
            "seed": config.seed,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }
        if config.stop_sequences:
            options["stop"] = config.stop_sequences

        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }

        t0 = time.perf_counter()
        try:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=req_data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120.0) as resp:
                if resp.status != 200:
                    raise ModelInferenceError(f"Ollama API returned HTTP {resp.status}")
                raw_bytes = resp.read()
                data = json.loads(raw_bytes.decode("utf-8"))

            wall_latency = round(time.perf_counter() - t0, 4)
            reply_text = data.get("response", "").strip()
            # If model is a reasoning model (e.g. DeepSeek-R1), strip <think>...</think> for answer evaluation
            clean_answer = re.sub(r"<think>.*?</think>", "", reply_text, flags=re.DOTALL).strip()
            effective_text = clean_answer if clean_answer else reply_text

            prompt_tokens = data.get("prompt_eval_count")
            generated_tokens = data.get("eval_count")
            eval_duration_ns = data.get("eval_duration", 0)
            tok_per_sec = (
                round(generated_tokens / (eval_duration_ns / 1e9), 2)
                if (generated_tokens and eval_duration_ns > 0)
                else (round(generated_tokens / wall_latency, 2) if generated_tokens and wall_latency > 0 else None)
            )

            return GenerationResult(
                text=effective_text,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                latency_seconds=wall_latency,
                metadata={
                    "runtime": "ollama",
                    "quantization": "4-bit (q4_K_M)",
                    "tokens_per_second": tok_per_sec,
                    "raw_response": reply_text,
                    "has_thinking_tags": "<think>" in reply_text,
                    "total_duration_ns": data.get("total_duration"),
                    "load_duration_ns": data.get("load_duration"),
                },
            )
        except Exception as exc:
            raise ModelInferenceError(f"Ollama generation failed for '{self.model_id}': {exc}") from exc

    def unload(self) -> None:
        """Instruct Ollama to evict model from RAM."""
        try:
            req_data = json.dumps({
                "model": self.model_id,
                "keep_alive": 0,
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=req_data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5.0) as _:
                pass
        except Exception:
            pass
        self._is_loaded = False
