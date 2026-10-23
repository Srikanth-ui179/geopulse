"""Local Ollama HTTP client and mocked test double."""

import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import httpx

from geopulse.summarization.exceptions import (
    OllamaModelNotFoundError,
    OllamaResponseError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)


class BaseOllamaClient(ABC):
    """Abstract interface for local Ollama interaction."""

    @property
    @abstractmethod
    def model(self) -> str:
        """The identifier of the active LLM model."""
        ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: str | None = None,
        format_type: str | None = "json",
    ) -> str:
        """Generate text completion from Ollama.

        Returns raw string content (typically JSON when format_type='json').
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether the Ollama HTTP daemon is reachable."""
        ...


class OllamaClient(BaseOllamaClient):
    """Synchronous HTTP client for local Ollama inference."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2:3b",
        connect_timeout: float = 5.0,
        read_timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._model = model
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout

        self._timeout = httpx.Timeout(
            connect=self.connect_timeout,
            read=self.read_timeout,
            write=10.0,
            pool=5.0,
        )

    @property
    def model(self) -> str:
        return self._model

    def is_available(self) -> bool:
        try:
            with httpx.Client(timeout=httpx.Timeout(connect=2.0, read=2.0)) as client:
                res = client.get(f"{self.base_url}/api/tags")
                return res.status_code == 200
        except (httpx.HTTPError, OSError):
            return False

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        format_type: str | None = "json",
    ) -> str:
        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        if format_type:
            payload["format"] = format_type

        url = f"{self.base_url}/api/generate"
        try:
            with httpx.Client(timeout=self._timeout) as client:
                res = client.post(url, json=payload)
                res.raise_for_status()
                data = res.json()
        except httpx.ConnectError as exc:
            raise OllamaUnavailableError(
                f"Ollama is unreachable at {self.base_url}. Ensure 'ollama serve' is running."
            ) from exc
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(
                f"Ollama request timed out after {self.read_timeout}s."
            ) from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise OllamaModelNotFoundError(
                    f"Model '{self._model}' not found in Ollama at {self.base_url}. "
                    f"Run 'ollama pull {self._model}' to install it."
                ) from exc
            raise OllamaResponseError(
                f"Ollama returned HTTP error {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except (json.JSONDecodeError, KeyError) as exc:
            raise OllamaResponseError(f"Invalid JSON returned by Ollama API: {exc}") from exc
        except httpx.HTTPError as exc:
            raise OllamaUnavailableError(f"HTTP communication with Ollama failed: {exc}") from exc

        raw_response = data.get("response")
        if raw_response is None:
            raise OllamaResponseError("Ollama payload missing 'response' field")

        return raw_response


class MockOllamaClient(BaseOllamaClient):
    """In-memory mocked Ollama client for offline tests and dry-run CLI demonstrations."""

    def __init__(
        self,
        model: str = "mock-llama3.2:3b",
        canned_response: str | None = None,
        should_timeout: bool = False,
        should_be_unavailable: bool = False,
        should_model_not_found: bool = False,
        should_return_malformed: bool = False,
        custom_generator: Callable[[str], str] | None = None,
    ) -> None:
        self._model = model
        self.canned_response = canned_response
        self.should_timeout = should_timeout
        self.should_be_unavailable = should_be_unavailable
        self.should_model_not_found = should_model_not_found
        self.should_return_malformed = should_return_malformed
        self.custom_generator = custom_generator
        self.calls: list[dict[str, Any]] = []

    @property
    def model(self) -> str:
        return self._model

    def is_available(self) -> bool:
        return not self.should_be_unavailable

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        format_type: str | None = "json",
    ) -> str:
        self.calls.append({"prompt": prompt, "system": system, "format": format_type})

        if self.should_be_unavailable:
            raise OllamaUnavailableError("Mocked: Ollama service is unavailable")

        if self.should_timeout:
            raise OllamaTimeoutError("Mocked: Ollama request timed out")

        if self.should_model_not_found:
            raise OllamaModelNotFoundError(f"Mocked: Model '{self._model}' not found")

        if self.should_return_malformed:
            return "{ unparseable json: [incomplete "

        if self.custom_generator is not None:
            return self.custom_generator(prompt)

        if self.canned_response is not None:
            return self.canned_response

        # Determine appropriate confidence qualification from prompt context
        if (
            "Clustering Confidence Label: single_source" in prompt
            or "Distinct Reporting Sources: 1 " in prompt
        ):
            conf_qual = "Single-source reporting; unverified independently."
        elif "Clustering Confidence Label: disputed" in prompt:
            conf_qual = "Disputed reporting: conflicting accounts between sources."
        elif "Clustering Confidence Label: corroborated" in prompt:
            conf_qual = (
                "Reported across multiple outlets; military claims unconfirmed independently."
            )
        else:
            conf_qual = "Preliminary reporting; underlying details unconfirmed."

        if "Event Nature: Strategic Analysis" in prompt or "The Endless Chip War" in prompt:
            return json.dumps(
                {
                    "headline": (
                        "Strategic analysis assesses global semiconductor rivalry "
                        "and supply chain security"
                    ),
                    "what_happened": (
                        "A strategic analysis examines global semiconductor competition, "
                        "technological sovereignty, and national security supply chains, "
                        "highlighting strategic risks across critical technology sectors."
                    ),
                    "key_details": {
                        "actors": ["Semiconductor manufacturers", "Defence policy makers"],
                        "location": "Global",
                        "timing": (
                            "Date not specified in source"
                            if "Date not specified in source" in prompt
                            else "October 22, 2026"
                        ),
                        "action": "Strategic analysis of semiconductor supply chain policy",
                    },
                    "significance": (
                        "Disruptions to the semiconductor supply chain directly affect "
                        "defence electronics, advanced radar, and sovereign industrial capacity."
                    ),
                    "confidence_qualification": conf_qual,
                }
            )

        # Default synthetic well-structured JSON response based on prompt
        return json.dumps(
            {
                "headline": "Targeted military strikes reported across conflict sector",
                "what_happened": (
                    "Forces executed coordinated strikes against opposing military facilities. "
                    "Initial reporting indicates disruption to radar and supply lines."
                ),
                "key_details": {
                    "actors": ["Regional military forces"],
                    "location": "Reported conflict zone",
                    "timing": "Within the last 24 hours",
                    "action": "Airstrikes and artillery strikes",
                },
                "significance": (
                    "Could degrade operational readiness in the sector if sustained."
                ),
                "confidence_qualification": conf_qual,
            }
        )
