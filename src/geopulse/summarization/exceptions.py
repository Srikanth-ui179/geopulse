"""Exceptions for the local Ollama summarization subsystem."""


class OllamaError(Exception):
    """Base exception for Ollama client and inference failures."""


class OllamaUnavailableError(OllamaError):
    """Raised when the Ollama HTTP server cannot be reached."""


class OllamaTimeoutError(OllamaError):
    """Raised when an Ollama request times out."""


class OllamaModelNotFoundError(OllamaError):
    """Raised when the requested model is not installed in the local Ollama instance."""


class OllamaResponseError(OllamaError):
    """Raised when Ollama returns a malformed or empty response."""


class SummaryValidationError(OllamaError):
    """Raised when LLM-generated output fails schema validation."""
