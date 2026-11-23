"""Exceptions for the Geopulse summarization subsystem (Ollama and Gemini)."""


class LLMError(Exception):
    """Base exception for all LLM client and inference failures."""


class OllamaError(LLMError):
    """Base exception for Ollama client and inference failures."""


class OllamaUnavailableError(OllamaError):
    """Raised when the Ollama HTTP server cannot be reached."""


class OllamaTimeoutError(OllamaError):
    """Raised when an Ollama request times out."""


class OllamaModelNotFoundError(OllamaError):
    """Raised when the requested model is not installed in the local Ollama instance."""


class OllamaResponseError(OllamaError):
    """Raised when Ollama returns a malformed or empty response."""


class SummaryValidationError(LLMError):
    """Raised when LLM-generated output fails schema validation."""


class GeminiError(LLMError):
    """Base exception for Gemini client and API failures."""


class GeminiConfigError(GeminiError):
    """Raised when Gemini configuration (e.g. GEMINI_API_KEY) is missing or invalid."""


class GeminiAuthError(GeminiError):
    """Raised when Gemini API authentication or authorization fails (e.g. 401 or 403)."""


class GeminiQuotaError(GeminiError):
    """Raised when Gemini API rate limits or quota are exhausted (e.g. 429)."""


class GeminiModelNotFoundError(GeminiError):
    """Raised when the requested Gemini model is not found (404)."""


class GeminiUnavailableError(GeminiError):
    """Raised when the Gemini service is unavailable (e.g. 5xx or connection error)."""


class GeminiTimeoutError(GeminiError):
    """Raised when a Gemini API request times out."""


class GeminiResponseError(GeminiError):
    """Raised when Gemini returns a malformed payload or content is blocked."""


# Explicitly defined eligible exceptions for fallback from Gemini to Ollama in 'auto' mode.
# Transient server errors, network connectivity/timeout errors, quota exhaustion,
# missing models, and response/schema errors trigger fallback. Authentication and
# configuration errors are non-transient and do NOT fall back.
GEMINI_FALLBACK_ELIGIBLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    GeminiUnavailableError,
    GeminiTimeoutError,
    GeminiQuotaError,
    GeminiModelNotFoundError,
    GeminiResponseError,
)
