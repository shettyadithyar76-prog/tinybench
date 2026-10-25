"""Model artifact download and local cache integrity manager."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Callable
import urllib.error
import urllib.request
import psutil

from tinybench.exceptions import (
    GatedModelAccessError,
    ModelDownloadError,
    ResourceLimitError,
    RuntimeUnavailableError,
)
from tinybench.models.ollama_backend import (
    check_ollama_daemon,
    get_ollama_base_url,
    is_ollama_model_installed,
)

logger = logging.getLogger(__name__)

# Required free disk space in bytes
REQUIRED_DISK_BYTES: dict[str, int] = {
    "meta-llama/Llama-3.2-1B-Instruct": int(3.5 * 1024 * 1024 * 1024),   # 3.5 GB
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": int(4.5 * 1024 * 1024 * 1024), # 4.5 GB
    "gemma4:e2b": int(2.5 * 1024 * 1024 * 1024),                         # 2.5 GB
    "deepseek-r1:1.5b": int(1.5 * 1024 * 1024 * 1024),                   # 1.5 GB
}


def is_model_installed(model_id: str) -> bool:
    """Check whether a model's weights and configuration are cached locally."""
    clean_id = model_id.strip()

    # Ollama runtime model
    if clean_id in ("gemma4:e2b", "deepseek-r1:1.5b") or ":" in clean_id or clean_id.startswith(("gemma4", "deepseek")):
        return is_ollama_model_installed(clean_id)

    # Hugging Face transformers model
    try:
        from huggingface_hub import try_to_load_from_cache

        # Check for config and weight files
        config_cached = try_to_load_from_cache(repo_id=clean_id, filename="config.json")
        weights_cached = try_to_load_from_cache(repo_id=clean_id, filename="model.safetensors")

        if config_cached is not None and isinstance(config_cached, (str, Path)):
            # If sharded weights, check model.safetensors.index.json
            if weights_cached is not None:
                return True
            index_cached = try_to_load_from_cache(repo_id=clean_id, filename="model.safetensors.index.json")
            if index_cached is not None:
                return True
    except Exception as exc:
        logger.debug(f"Cache check error for {clean_id}: {exc}")

    return False


def check_disk_space_for_model(model_id: str, target_dir: Path | None = None) -> None:
    """Ensure sufficient disk space is free before initiating a model download."""
    required_bytes = REQUIRED_DISK_BYTES.get(model_id, int(3.0 * 1024 * 1024 * 1024))
    check_path = target_dir or Path.home()
    usage = psutil.disk_usage(str(check_path))

    if usage.free < required_bytes:
        req_gb = required_bytes / (1024 ** 3)
        avail_gb = usage.free / (1024 ** 3)
        raise ResourceLimitError(
            f"Insufficient disk space to download {model_id}: "
            f"requires at least {req_gb:.1f} GB, but only {avail_gb:.1f} GB is free on disk."
        )


def download_model(
    model_id: str,
    token: str | None = None,
    progress_callback: Callable[[str, float | None], None] | None = None,
) -> str:
    """Download and verify model weights into local storage.

    Never downloads automatically. Must be explicitly invoked by user action.
    """
    clean_id = model_id.strip()

    # 1. Disk space pre-check
    check_disk_space_for_model(clean_id)

    # 2. Ollama model download path
    if clean_id in ("gemma4:e2b", "deepseek-r1:1.5b") or ":" in clean_id or clean_id.startswith(("gemma4", "deepseek")):
        ollama_url = get_ollama_base_url()
        is_running, daemon_msg = check_ollama_daemon(ollama_url)
        if not is_running:
            raise RuntimeUnavailableError(daemon_msg)

        if progress_callback:
            progress_callback(f"Connecting to Ollama daemon ({ollama_url}) for {clean_id}...", 0.05)

        try:
            req_data = json.dumps({"name": clean_id, "stream": True}).encode("utf-8")
            req = urllib.request.Request(
                f"{ollama_url}/api/pull",
                data=req_data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=600.0) as resp:
                for line in resp:
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line.decode("utf-8"))
                        status = chunk.get("status", "")
                        total = chunk.get("total", 0)
                        completed = chunk.get("completed", 0)
                        pct = (completed / total) if (total and completed) else None
                        if progress_callback:
                            progress_callback(f"Ollama: {status}", pct)
                        if "error" in chunk:
                            raise ModelDownloadError(f"Ollama pull error: {chunk['error']}")
                    except json.JSONDecodeError:
                        continue

            # Verify installation
            if is_ollama_model_installed(clean_id, base_url=ollama_url):
                if progress_callback:
                    progress_callback(f"Successfully installed {clean_id} in Ollama.", 1.0)
                return f"ollama:{clean_id}"
            else:
                raise ModelDownloadError(f"Ollama download finished but {clean_id} was not detected in tags.")

        except Exception as exc:
            if isinstance(exc, (ModelDownloadError, RuntimeUnavailableError)):
                raise
            raise ModelDownloadError(f"Failed to pull Ollama model '{clean_id}': {exc}") from exc

    # 3. Hugging Face transformers model download path
    effective_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if clean_id == "meta-llama/Llama-3.2-1B-Instruct" and not effective_token:
        raise GatedModelAccessError(
            "meta-llama/Llama-3.2-1B-Instruct is a gated repository requiring Meta license agreement. "
            "Please visit https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct to accept the terms, "
            "then provide your Hugging Face access token."
        )

    if progress_callback:
        progress_callback(f"Initiating Hugging Face download for {clean_id}...", 0.05)

    try:
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(
            repo_id=clean_id,
            token=effective_token,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.txt",
                "*.model",
            ],
        )

        # Integrity check: verify configuration and weight files exist
        local_path = Path(local_dir)
        config_file = local_path / "config.json"
        if not config_file.exists():
            raise ModelDownloadError(
                f"Integrity check failed: 'config.json' missing from downloaded snapshot at {local_dir}"
            )

        weight_files = list(local_path.glob("*.safetensors"))
        if not weight_files:
            raise ModelDownloadError(
                f"Integrity check failed: No '.safetensors' weights found in downloaded snapshot at {local_dir}"
            )

        if progress_callback:
            progress_callback(f"Verified {len(weight_files)} weight files for {clean_id}.", 1.0)

        return str(local_path)

    except Exception as exc:
        msg = str(exc)
        if "gated" in msg.lower() or "401" in msg or "403" in msg:
            raise GatedModelAccessError(
                f"Access denied for gated model '{clean_id}'. "
                "Ensure you accepted the license terms on Hugging Face and provided a valid token: "
                f"{exc}"
            ) from exc
        raise ModelDownloadError(f"Failed to download '{clean_id}' from Hugging Face Hub: {exc}") from exc
