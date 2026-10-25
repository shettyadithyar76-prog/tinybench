"""Custom exceptions for the TinyBench toolkit."""

class TinyBenchError(Exception):
    """Base exception for all TinyBench errors."""
    pass


class DatasetValidationError(TinyBenchError):
    """Raised when a benchmark dataset file is invalid or malformed."""
    pass


class ModelNotFoundError(TinyBenchError):
    """Raised when a requested model is not found in the approved catalogue."""
    pass


class ModelDownloadError(TinyBenchError):
    """Raised when downloading model weights fails."""
    pass


class GatedModelAccessError(ModelDownloadError):
    """Raised when a model requires user license approval or HF_TOKEN."""
    pass


class RuntimeUnavailableError(TinyBenchError):
    """Raised when a required inference runtime (e.g. Ollama daemon) is unavailable."""
    pass


class ModelLoadError(TinyBenchError):
    """Raised when loading a model adapter fails."""
    pass


class ModelInferenceError(TinyBenchError):
    """Raised when model text generation fails."""
    pass


class MetricComputationError(TinyBenchError):
    """Raised when an evaluation metric cannot be computed."""
    pass


class ResourceLimitError(TinyBenchError):
    """Raised when system memory or resources are insufficient for the model."""
    pass
