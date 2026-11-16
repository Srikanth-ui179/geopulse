"""Model provider clients for local Ollama and Google Gemini."""

import json
import logging
import re
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import SecretStr

from geopulse.summarization.exceptions import (
    GeminiAuthError,
    GeminiConfigError,
    GeminiModelNotFoundError,
    GeminiQuotaError,
    GeminiResponseError,
    GeminiTimeoutError,
    GeminiUnavailableError,
    OllamaModelNotFoundError,
    OllamaResponseError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)

if TYPE_CHECKING:
    from geopulse.config import Settings

logger = logging.getLogger(__name__)


# Stage-specific generation budgets tuned for local Ollama CPU inference (llama3.2:3b @ ~8.3 tok/s)
# Mathematical budget rationale:
# Compact claim schema (integer source index 'src' + max 2 claims + short evidence spans)
# requires ~70-95 tokens for 1 central claim, ~140-220 tokens for central + supporting claim.
# Longest realistic 2-claim output with 25-word spans is ~260-280 tokens.
# - Initial claims: 320 tokens (~38.6s max; +33% safety headroom over realistic outputs)
# - Entailment verifier: 160 tokens (~19.3s max; accommodates 1-2 compact claim verdicts)
# - Bounded corrective retries: 320 tokens (~38.6s max)
# - Telegram QA bot answers: 256 tokens (~30.9s max)
# Total worst-case generation for 2-stage event: 38.6s + 19.3s = 57.9s (safely inside 120s).
STAGE_BUDGET_CLAIMS: int = 512
STAGE_BUDGET_VERIFIER: int = 160
STAGE_BUDGET_CORRECTION: int = 512
STAGE_BUDGET_QA: int = 256

DEFAULT_OLLAMA_OPTIONS: dict[str, Any] = {
    "num_predict": STAGE_BUDGET_CLAIMS,
    "temperature": 0.1,
    "repeat_penalty": 1.15,
}

TRANSIENT_STATUS_CODES: frozenset[int] = frozenset({502, 503, 504})


class BaseLLMClient(ABC):
    """Abstract interface for LLM inference interaction."""

    @property
    @abstractmethod
    def model(self) -> str:
        """The identifier of the active LLM model."""
        ...

    @property
    @abstractmethod
    def provider(self) -> str:
        """The identifier of the active provider ('ollama' or 'gemini')."""
        ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: str | None = None,
        format_type: str | None = "json",
        options: dict[str, Any] | None = None,
    ) -> str:
        """Generate text completion from LLM.

        Returns raw string content (typically JSON when format_type='json').
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether the provider endpoint is reachable."""
        ...


class BaseOllamaClient(BaseLLMClient):
    """Abstract interface for local Ollama interaction."""

    @property
    def provider(self) -> str:
        return "ollama"


class OllamaClient(BaseOllamaClient):
    """Synchronous HTTP client for local Ollama inference."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2:3b",
        connect_timeout: float = 5.0,
        read_timeout: float = 120.0,
        max_retries: int = 1,
        request_timeout: float = 180.0,
        max_abandoned_workers: int = 2,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._model = model
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.max_abandoned_workers = max_abandoned_workers
        self._transport = transport
        self._client: httpx.Client | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._abandoned_workers_count: int = 0
        self._abandoned_lock = threading.Lock()
        self.last_metrics: dict[str, Any] = {}

        self._timeout = httpx.Timeout(
            connect=self.connect_timeout,
            read=self.read_timeout,
            write=10.0,
            pool=5.0,
        )

    @property
    def abandoned_workers_count(self) -> int:
        """Return the number of currently active abandoned worker threads."""
        with self._abandoned_lock:
            return self._abandoned_workers_count

    def _track_abandoned_worker(self, future: Any) -> None:
        """Register a timed-out future to track and release when the background worker exits."""
        with self._abandoned_lock:
            if not future.done():
                self._abandoned_workers_count += 1
                decremented = False

                def _release_abandoned(_fut: Any) -> None:
                    nonlocal decremented
                    with self._abandoned_lock:
                        if not decremented:
                            decremented = True
                            self._abandoned_workers_count = max(
                                0, self._abandoned_workers_count - 1
                            )

                future.add_done_callback(_release_abandoned)

    def _get_client(self) -> httpx.Client:
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(
                timeout=self._timeout,
                transport=self._transport,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            )
        return self._client

    def _get_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="ollama-worker",
            )
        return self._executor

    def _execute_post(
        self,
        client: httpx.Client,
        url: str,
        payload: dict[str, Any],
        timeout: httpx.Timeout,
    ) -> httpx.Response:
        res = client.post(url, json=payload, timeout=timeout)
        _ = res.content
        return res

    def close(self) -> None:
        """Safely release HTTP client and executor resources without blocking or masking errors."""
        client = self._client
        self._client = None
        executor = self._executor
        self._executor = None

        if client is not None:
            try:
                if not client.is_closed:
                    client.close()
            except Exception as exc:
                logger.debug("Error closing HTTP client during cleanup: %s", exc)

        if executor is not None:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception as exc:
                logger.debug("Error shutting down worker executor during cleanup: %s", exc)

    @property
    def model(self) -> str:
        return self._model

    def is_available(self) -> bool:
        try:
            with httpx.Client(timeout=httpx.Timeout(2.0)) as client:
                res = client.get(f"{self.base_url}/api/tags")
                return res.status_code == 200
        except (httpx.HTTPError, OSError):
            return False

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        format_type: str | None = "json",
        options: dict[str, Any] | None = None,
    ) -> str:
        merged_options = {**DEFAULT_OLLAMA_OPTIONS, **(options or {})}
        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": merged_options,
        }
        if system:
            payload["system"] = system
        if format_type:
            payload["format"] = format_type

        url = f"{self.base_url}/api/generate"
        overall_start = time.perf_counter()

        with self._abandoned_lock:
            if self._abandoned_workers_count >= self.max_abandoned_workers:
                logger.error(
                    "Ollama request rejected immediately: %d abandoned worker(s) "
                    "active (limit: %d)",
                    self._abandoned_workers_count,
                    self.max_abandoned_workers,
                )
                raise OllamaTimeoutError(
                    f"Ollama request rejected: {self._abandoned_workers_count} abandoned "
                    f"worker(s) are still active (limit: {self.max_abandoned_workers}). "
                    f"Wait for background Ollama inference to conclude before sending new requests."
                )

        deadline = overall_start + self.request_timeout

        for attempt in range(self.max_retries + 1):
            now = time.perf_counter()
            remaining_time = deadline - now
            if remaining_time <= 0:
                elapsed = time.perf_counter() - overall_start
                logger.error(
                    "Ollama request timed out after %.2fs (prompt_chars=%d, timeout=%.1fs)",
                    elapsed,
                    len(prompt),
                    self.request_timeout,
                )
                self.close()
                raise OllamaTimeoutError(
                    f"Ollama request exceeded wall-clock deadline of {self.request_timeout:.1f}s "
                    f"(elapsed={elapsed:.2f}s)."
                )

            start_time = time.perf_counter()
            client = self._get_client()
            attempt_timeout = httpx.Timeout(
                connect=min(self.connect_timeout, remaining_time),
                read=min(self.read_timeout, remaining_time),
                write=min(10.0, remaining_time),
                pool=min(5.0, remaining_time),
            )
            try:
                executor = self._get_executor()
                future = executor.submit(self._execute_post, client, url, payload, attempt_timeout)
                try:
                    res = future.result(timeout=remaining_time)
                except FuturesTimeoutError as exc:
                    self._track_abandoned_worker(future)
                    self.close()
                    elapsed = time.perf_counter() - overall_start
                    logger.error(
                        "Ollama request timed out after %.2fs (prompt_chars=%d, timeout=%.1fs)",
                        elapsed,
                        len(prompt),
                        self.request_timeout,
                    )
                    raise OllamaTimeoutError(
                        f"Ollama request exceeded wall-clock deadline "
                        f"of {self.request_timeout:.1f}s (elapsed={elapsed:.2f}s)."
                    ) from exc

                elapsed = time.perf_counter() - start_time

                # Check for token repeat limit error in 500 responses
                if res.status_code == 500 and "token repeat limit reached" in res.text.lower():
                    logger.warning(
                        "Ollama token repeat limit reached (attempt %d/%d): "
                        "prompt_chars=%d, elapsed=%.2fs",
                        attempt + 1,
                        self.max_retries + 1,
                        len(prompt),
                        elapsed,
                    )
                    if attempt < self.max_retries:
                        payload["options"]["repeat_penalty"] = 1.25
                        payload["options"]["num_predict"] = min(
                            payload["options"].get("num_predict", 768), 512
                        )
                        backoff = min(1.0, 0.5 * (attempt + 1))
                        if deadline - time.perf_counter() > backoff:
                            time.sleep(backoff)
                            continue
                        elapsed = time.perf_counter() - overall_start
                        logger.error(
                            "Ollama request timed out after %.2fs (prompt_chars=%d, timeout=%.1fs)",
                            elapsed,
                            len(prompt),
                            self.request_timeout,
                        )
                        self.close()
                        raise OllamaTimeoutError(
                            f"Ollama request exceeded wall-clock deadline "
                            f"of {self.request_timeout:.1f}s (elapsed={elapsed:.2f}s)."
                        )
                    raise OllamaResponseError(
                        "Ollama prediction aborted: token repeat limit reached"
                    )

                # Handle transient gateway / server errors (502, 503, 504)
                if res.status_code in TRANSIENT_STATUS_CODES:
                    logger.warning(
                        "Ollama transient HTTP %d (attempt %d/%d): elapsed=%.2fs",
                        res.status_code,
                        attempt + 1,
                        self.max_retries + 1,
                        elapsed,
                    )
                    if attempt < self.max_retries:
                        backoff = min(1.0, 0.5 * (attempt + 1))
                        if deadline - time.perf_counter() > backoff:
                            time.sleep(backoff)
                            continue
                        elapsed = time.perf_counter() - overall_start
                        logger.error(
                            "Ollama request timed out after %.2fs (prompt_chars=%d, timeout=%.1fs)",
                            elapsed,
                            len(prompt),
                            self.request_timeout,
                        )
                        self.close()
                        raise OllamaTimeoutError(
                            f"Ollama request exceeded wall-clock deadline "
                            f"of {self.request_timeout:.1f}s (elapsed={elapsed:.2f}s)."
                        )
                    res.raise_for_status()

                res.raise_for_status()
                data = res.json()
                raw_response = data.get("response")
                if raw_response is None:
                    raise OllamaResponseError("Ollama payload missing 'response' field")

                # Extract sanitized performance and token usage telemetry
                prompt_tokens = data.get("prompt_eval_count")
                eval_tokens = data.get("eval_count")
                eval_duration_ns = data.get("eval_duration")
                prompt_eval_duration_ns = data.get("prompt_eval_duration")
                done_reason = data.get("done_reason")  # e.g., "stop" or "length"

                tok_per_sec = 0.0
                if eval_tokens and eval_duration_ns and eval_duration_ns > 0:
                    tok_per_sec = eval_tokens / (eval_duration_ns / 1e9)

                prompt_tok_per_sec = 0.0
                prompt_eval_sec = 0.0
                if prompt_tokens and prompt_eval_duration_ns and prompt_eval_duration_ns > 0:
                    prompt_eval_sec = prompt_eval_duration_ns / 1e9
                    prompt_tok_per_sec = prompt_tokens / prompt_eval_sec

                eval_sec = (eval_duration_ns / 1e9) if eval_duration_ns else 0.0

                self.last_metrics = {
                    "model": self._model,
                    "prompt_tokens": prompt_tokens,
                    "prompt_eval_seconds": round(prompt_eval_sec, 2),
                    "prompt_eval_tokens_per_second": round(prompt_tok_per_sec, 2),
                    "eval_tokens": eval_tokens,
                    "eval_seconds": round(eval_sec, 2),
                    "eval_tokens_per_second": round(tok_per_sec, 2),
                    "elapsed_seconds": round(elapsed, 2),
                    "total_wall_clock_seconds": round(time.perf_counter() - overall_start, 2),
                    "done_reason": done_reason or "unknown",
                }

                if done_reason == "length":
                    logger.warning(
                        "Ollama response truncated: hit num_predict limit (%d tokens); "
                        "model=%s, elapsed=%.2fs",
                        eval_tokens or 0,
                        self._model,
                        elapsed,
                    )

                logger.info(
                    "Ollama generate completed: model=%s, prompt=%s tokens (%.2fs, %.1f tok/s), "
                    "eval=%s tokens (%.2fs, %.1f tok/s), done_reason=%s, elapsed=%.2fs",
                    self._model,
                    prompt_tokens,
                    prompt_eval_sec,
                    prompt_tok_per_sec,
                    eval_tokens,
                    eval_sec,
                    tok_per_sec,
                    done_reason or "unknown",
                    elapsed,
                )
                return raw_response

            except (httpx.RemoteProtocolError, ConnectionResetError, httpx.ReadError) as exc:
                elapsed = time.perf_counter() - start_time
                logger.warning(
                    "Ollama connection reset or dropped (attempt %d/%d): %s (elapsed=%.2fs)",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                    elapsed,
                )
                self.close()
                if attempt < self.max_retries:
                    backoff = min(1.0, 0.5 * (attempt + 1))
                    if deadline - time.perf_counter() > backoff:
                        time.sleep(backoff)
                        continue
                    elapsed = time.perf_counter() - overall_start
                    logger.error(
                        "Ollama request timed out after %.2fs (prompt_chars=%d, timeout=%.1fs)",
                        elapsed,
                        len(prompt),
                        self.request_timeout,
                    )
                    raise OllamaTimeoutError(
                        f"Ollama request exceeded wall-clock deadline "
                        f"of {self.request_timeout:.1f}s (elapsed={elapsed:.2f}s)."
                    ) from exc
                msg = (
                    f"Ollama connection reset after {attempt + 1} "
                    f"attempts at {self.base_url}: {exc}"
                )
                raise OllamaUnavailableError(msg) from exc

            except httpx.ConnectError as exc:
                elapsed = time.perf_counter() - start_time
                logger.warning(
                    "Ollama connect failed (attempt %d/%d): %s (elapsed=%.2fs)",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                    elapsed,
                )
                self.close()
                if attempt < self.max_retries:
                    backoff = min(1.0, 0.5 * (attempt + 1))
                    if deadline - time.perf_counter() > backoff:
                        time.sleep(backoff)
                        continue
                    elapsed = time.perf_counter() - overall_start
                    logger.error(
                        "Ollama request timed out after %.2fs (prompt_chars=%d, timeout=%.1fs)",
                        elapsed,
                        len(prompt),
                        self.request_timeout,
                    )
                    raise OllamaTimeoutError(
                        f"Ollama request exceeded wall-clock deadline "
                        f"of {self.request_timeout:.1f}s (elapsed={elapsed:.2f}s)."
                    ) from exc
                raise OllamaUnavailableError(
                    f"Ollama is unreachable at {self.base_url}. Ensure 'ollama serve' is running."
                ) from exc

            except httpx.TimeoutException as exc:
                elapsed = time.perf_counter() - overall_start
                logger.error(
                    "Ollama request timed out after %.2fs (prompt_chars=%d, timeout=%.1fs)",
                    elapsed,
                    len(prompt),
                    self.request_timeout,
                )
                self.close()
                raise OllamaTimeoutError(
                    f"Ollama request timed out after {self.request_timeout:.1f}s "
                    f"(elapsed={elapsed:.2f}s)."
                ) from exc

            except httpx.HTTPStatusError as exc:
                self.close()
                if exc.response.status_code == 404:
                    raise OllamaModelNotFoundError(
                        f"Model '{self._model}' not found in Ollama at {self.base_url}. "
                        f"Run 'ollama pull {self._model}' to install it."
                    ) from exc
                raise OllamaResponseError(
                    f"Ollama returned HTTP error {exc.response.status_code}: {exc.response.text}"
                ) from exc

            except (json.JSONDecodeError, KeyError) as exc:
                self.close()
                raise OllamaResponseError(f"Invalid JSON returned by Ollama API: {exc}") from exc


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
        connection_resets_count: int = 0,
        transient_error_status: int | None = None,
        transient_error_count: int = 0,
        token_repeat_error_count: int = 0,
        request_timeout: float = 180.0,
        max_retries: int = 1,
        custom_generator: Callable[[str], str] | None = None,
    ) -> None:
        self._model = model
        self.canned_response = canned_response
        self.should_timeout = should_timeout
        self.should_be_unavailable = should_be_unavailable
        self.should_model_not_found = should_model_not_found
        self.should_return_malformed = should_return_malformed
        self.connection_resets_remaining = connection_resets_count
        self.transient_error_status = transient_error_status
        self.transient_errors_remaining = transient_error_count
        self.token_repeat_errors_remaining = token_repeat_error_count
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.custom_generator = custom_generator
        self.calls: list[dict[str, Any]] = []
        self.last_metrics: dict[str, Any] = {}

    @property
    def model(self) -> str:
        return self._model

    @property
    def abandoned_workers_count(self) -> int:
        return 0

    def close(self) -> None:
        """No-op resource release for mocked Ollama client."""
        pass

    def is_available(self) -> bool:
        return not self.should_be_unavailable

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        format_type: str | None = "json",
        options: dict[str, Any] | None = None,
    ) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "system": system,
                "format": format_type,
                "options": options,
            }
        )

        if self.connection_resets_remaining > 0:
            self.connection_resets_remaining -= 1
            raise OllamaUnavailableError("Mocked: Connection reset by peer")

        if self.transient_errors_remaining > 0:
            self.transient_errors_remaining -= 1
            status = self.transient_error_status or 503
            raise OllamaResponseError(
                f"Mocked: Ollama returned HTTP error {status}: Service Unavailable"
            )

        if self.token_repeat_errors_remaining > 0:
            self.token_repeat_errors_remaining -= 1
            raise OllamaResponseError(
                "Mocked: Ollama prediction aborted: token repeat limit reached"
            )

        if self.should_be_unavailable:
            raise OllamaUnavailableError("Mocked: Ollama service is unavailable")

        if self.should_timeout:
            raise OllamaTimeoutError("Mocked: Ollama request timed out")

        if self.should_model_not_found:
            raise OllamaModelNotFoundError(f"Mocked: Model '{self._model}' not found")

        if self.should_return_malformed:
            return "{ unparseable json: [incomplete "

        if self.custom_generator is not None:
            resp = self.custom_generator(prompt)
        elif self.canned_response is not None:
            resp = self.canned_response
        else:
            resp = _generate_mock_summary_json(prompt)

        self.last_metrics = {
            "model": self._model,
            "prompt_tokens": len(prompt.split()),
            "eval_tokens": len(resp.split()),
            "eval_tokens_per_second": 8.3,
            "elapsed_seconds": 0.05,
            "done_reason": "stop",
        }
        return resp


def _generate_mock_summary_json(prompt: str) -> str:
    """Generate deterministic, grounded JSON summary fixtures based on prompt contents."""
    articles_section = (
        prompt.split("SOURCE ARTICLES:")[-1] if "SOURCE ARTICLES:" in prompt else prompt
    )
    prompt_lower = articles_section.lower()
    if (
        "Clustering Confidence Label: single_source" in prompt
        or "Distinct Reporting Sources: 1 " in prompt
    ):
        conf_qual = "Single-source reporting; unverified independently."
    elif "Clustering Confidence Label: disputed" in prompt:
        conf_qual = "Disputed reporting: conflicting accounts between sources."
    elif "Clustering Confidence Label: corroborated" in prompt:
        conf_qual = "Reported across multiple outlets; military claims unconfirmed independently."
    else:
        conf_qual = "Preliminary reporting; underlying details unconfirmed."

    if (
        "endless chip war" in prompt_lower
        or "semiconductor" in prompt_lower
        or "chip war" in prompt_lower
    ):
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
                    "timing": None,
                    "action": "Strategic analysis of semiconductor supply chain policy",
                },
                "significance": (
                    "Disruptions to the semiconductor supply chain directly affect "
                    "defence electronics, advanced radar, and sovereign industrial capacity."
                ),
                "confidence_qualification": conf_qual,
            }
        )

    prompt_lower = prompt.lower()
    is_petraeus = "petraeus" in prompt_lower and (
        "urgent revolutionary" in prompt_lower
        or "autonomous war" in prompt_lower
        or "petraeus says" in prompt_lower
    )
    if is_petraeus:
        return json.dumps(
            {
                "headline": (
                    "Petraeus urges 'urgent revolutionary institutional change' "
                    "for US military autonomous war readiness"
                ),
                "what_happened": (
                    "Retired Army Gen. David Petraeus stated that the U.S. military requires "
                    "'urgent revolutionary institutional change' to prepare for modern "
                    "autonomous war. While praising the Pentagon's new Autonomous Warfare "
                    "Command, Petraeus warned that rapid preparation is essential to stay "
                    "ahead of evolving conflict doctrines."
                ),
                "key_details": {
                    "actors": [
                        "Ret. Gen. David Petraeus",
                        "Pentagon",
                        "Autonomous Warfare Command",
                    ],
                    "location": "United States",
                    "timing": (
                        "Date not specified in source"
                        if "Date not specified in source" in prompt
                        else "October 2026"
                    ),
                    "action": ("Call for institutional reform and autonomous warfare preparation"),
                },
                "significance": (
                    "Highlights prominent military leadership debate over the pace of "
                    "institutional transformation required for drone and autonomous systems."
                ),
                "confidence_qualification": conf_qual,
            }
        )

    is_e5 = (
        "e5 defence" in prompt_lower
        or "e5 defense" in prompt_lower
        or "e5 group" in prompt_lower
        or "group of five" in prompt_lower
    )
    if is_e5:
        return json.dumps(
            {
                "headline": (
                    "E5 Defence Ministers issue joint statement on European "
                    "security and NATO cooperation"
                ),
                "what_happened": (
                    "Defence Ministers of the E5 group (France, Germany, Italy, Poland, "
                    "and the UK) issued a joint statement affirming that Europe must assume "
                    "greater responsibility for collective defence within NATO and sustain "
                    "support for Ukraine."
                ),
                "key_details": {
                    "actors": [
                        "E5 Defence Ministers",
                        "France",
                        "Germany",
                        "Italy",
                        "Poland",
                        "UK",
                    ],
                    "location": "Europe",
                    "timing": (
                        "Date not specified in source"
                        if "Date not specified in source" in prompt
                        else "October 26, 2026"
                    ),
                    "action": (
                        "Joint statement on collective European security and NATO deterrence"
                    ),
                },
                "significance": (
                    "Reaffirms European leadership and coordinated defence targets within NATO."
                ),
                "confidence_qualification": conf_qual,
            }
        )

    is_uk_france = (
        "uk-france" in prompt_lower or "french nuclear" in prompt_lower
    ) and "exercise" in prompt_lower
    if is_uk_france:
        return json.dumps(
            {
                "headline": ("UK-France deepen nuclear defence ties in first joint air exercise"),
                "what_happened": (
                    "RAF Typhoon fighters joined French Rafale jets "
                    "in a French nuclear drill, marking the first "
                    "joint air exercise supporting bilateral "
                    "nuclear defence ties."
                ),
                "key_details": {
                    "actors": [
                        "Royal Air Force",
                        "French Air and Space Force",
                        "UK Ministry of Defence",
                    ],
                    "location": "France",
                    "timing": (
                        "Date not specified in source"
                        if "Date not specified in source" in prompt
                        else "October 2026"
                    ),
                    "action": ("Joint air exercise supporting bilateral nuclear defence posture"),
                },
                "significance": (
                    "Signals closer bilateral deterrence "
                    "integration between Europe's two nuclear "
                    "powers under the Lancaster House framework."
                ),
                "confidence_qualification": conf_qual,
            }
        )

    # Event-specific fixture handling for standard test datasets
    # 1. US and UK strikes on Houthi targets in Yemen
    if "houthi" in prompt_lower and (
        "yemen" in prompt_lower or "airstrikes" in prompt_lower or "radar" in prompt_lower
    ):
        return json.dumps(
            {
                "headline": (
                    "US and UK launch joint airstrikes on Houthi military positions in Yemen"
                ),
                "what_happened": (
                    "American and British fighter aircraft carried out coordinated "
                    "airstrikes against Houthi radar and missile sites across western Yemen. "
                    "The operations followed repeated missile and drone attacks on Red Sea "
                    "commercial maritime shipping."
                ),
                "key_details": {
                    "actors": ["US military", "British forces", "Houthi forces"],
                    "location": "Yemen",
                    "timing": (
                        "Date not specified in source"
                        if "Date not specified in source" in prompt
                        else "October 22, 2026"
                    ),
                    "action": "Joint airstrikes against radar and missile infrastructure",
                },
                "significance": (
                    "Aims to degrade Houthi capabilities to threaten commercial maritime "
                    "transit in the Red Sea."
                ),
                "confidence_qualification": conf_qual,
            }
        )

    # 2. Rebels claim downing of military transport aircraft / army denies
    if (
        "rebels claim" in prompt_lower
        or "transport aircraft" in prompt_lower
        or "antonov" in prompt_lower
    ):
        return json.dumps(
            {
                "headline": (
                    "Rebels claim downing of military transport aircraft as army denies loss"
                ),
                "what_happened": (
                    "Insurgent commanders claimed they shot down an Antonov transport plane. "
                    "Military spokesmen formally denied the report, describing the claim "
                    "as unconfirmed propaganda."
                ),
                "key_details": {
                    "actors": ["Insurgent Commanders", "Military Spokesmen"],
                    "location": None,
                    "timing": (
                        "Date not specified in source"
                        if "Date not specified in source" in prompt
                        else "October 22, 2026"
                    ),
                    "action": "Contested military transport aircraft downing claim",
                },
                "significance": (
                    "Conflicting claims reflect active informational contestation "
                    "over battlefield attrition."
                ),
                "confidence_qualification": (
                    "Disputed reporting: conflicting accounts between insurgent claims "
                    "and military denials."
                ),
            }
        )

    # 3. Rostov oil depot drone strike
    if "rostov" in prompt_lower and (
        "oil depot" in prompt_lower or "fuel depot" in prompt_lower or "drone" in prompt_lower
    ):
        return json.dumps(
            {
                "headline": (
                    "Ukrainian drone strikes set ablaze oil depot in Russia's Rostov region"
                ),
                "what_happened": (
                    "A long-range Ukrainian drone attack struck a fuel depot in the Rostov "
                    "region, triggering a massive industrial fire according to Russian "
                    "regional authorities."
                ),
                "key_details": {
                    "actors": ["Ukrainian Armed Forces", "Russian Regional Authorities"],
                    "location": "Rostov, Russia",
                    "timing": (
                        "Date not specified in source"
                        if "Date not specified in source" in prompt
                        else "October 22, 2026"
                    ),
                    "action": "Long-range drone strike on industrial fuel depot",
                },
                "significance": (
                    "Targets Russian fuel supplies and rear-echelon logistics supporting "
                    "military operations."
                ),
                "confidence_qualification": conf_qual,
            }
        )

    # 4. Belgorod power substation shelling
    if "belgorod" in prompt_lower and (
        "shelling" in prompt_lower or "substation" in prompt_lower or "electrical" in prompt_lower
    ):
        return json.dumps(
            {
                "headline": "Cross-border shelling damages power substation in Belgorod",
                "what_happened": (
                    "Cross-border artillery shelling in the Russian border city of Belgorod "
                    "damaged an electrical substation, causing local power outages on "
                    "Thursday morning."
                ),
                "key_details": {
                    "actors": ["authorities"],
                    "location": "Belgorod",
                    "timing": (
                        "Date not specified in source"
                        if "Date not specified in source" in prompt
                        else "October 22, 2026"
                    ),
                    "action": "Cross-border artillery shelling of civilian power substation",
                },
                "significance": (
                    "Highlights the vulnerability of border electrical infrastructure to "
                    "cross-border artillery strikes."
                ),
                "confidence_qualification": conf_qual,
            }
        )

    # 5. UK MoD air defence contract / £400m
    if ("air defence" in prompt_lower or "air defense" in prompt_lower) and (
        "400m" in prompt_lower
        or "£400m" in prompt_lower
        or "british mod" in prompt_lower
        or bool(re.search(r"\bmod\b", prompt_lower))
    ):
        return json.dumps(
            {
                "headline": "UK Ministry of Defence signs £400m contract for air defence missiles",
                "what_happened": (
                    "The British Ministry of Defence announced a £400 million procurement "
                    "contract for next-generation short-range air defence missile systems "
                    "to boost British Army readiness."
                ),
                "key_details": {
                    "actors": ["UK Ministry of Defence", "British Army"],
                    "location": "United Kingdom",
                    "timing": (
                        "Date not specified in source"
                        if "Date not specified in source" in prompt
                        else "October 22, 2026"
                    ),
                    "action": "Procurement contract for short-range air defence systems",
                },
                "significance": (
                    "Enhances ground-based air defence coverage and replenishment for "
                    "the British Army."
                ),
                "confidence_qualification": "Official announcement from UK Ministry of Defence.",
            }
        )

    # 6. Premier League match postponed following drone security incident
    if "premier league" in prompt_lower or "arsenal" in prompt_lower or "stadium" in prompt_lower:
        return json.dumps(
            {
                "headline": "Premier League match postponed after drone strike near stadium",
                "what_happened": (
                    "The Premier League football match between Arsenal and Chelsea was delayed "
                    "following an isolated security incident involving a drone strike in the "
                    "city centre near the stadium."
                ),
                "key_details": {
                    "actors": ["Premier League", "Arsenal", "Chelsea"],
                    "location": None,
                    "timing": (
                        "Date not specified in source"
                        if "Date not specified in source" in prompt
                        else "October 22, 2026"
                    ),
                    "action": "Sporting match postponement following drone security incident",
                },
                "significance": (
                    "Reflects the heightened sensitivity and precautionary security measures "
                    "around civilian venues."
                ),
                "confidence_qualification": conf_qual,
            }
        )

    # Satellite proximity operations / space wars
    if "satellite" in prompt_lower and (
        bool(re.search(r"\btails\b", prompt_lower))
        or "space wars" in prompt_lower
        or "ukraine-linked" in prompt_lower
    ):
        return json.dumps(
            {
                "headline": "Washington accuses Moscow after Russian satellite tails US spacecraft",
                "what_happened": (
                    "US officials accused Russia of maneuvering a military satellite close to a "
                    "US-registered satellite in orbit last week, highlighting escalating proximity "
                    "operations in space."
                ),
                "key_details": {
                    "actors": ["Russian military", "US officials"],
                    "location": None,
                    "timing": "Last week",
                    "action": "Orbital proximity maneuver involving military satellite",
                },
                "significance": (
                    "Demonstrates contested orbital maneuvers and rising counter-space "
                    "operational risks."
                ),
                "confidence_qualification": conf_qual,
            }
        )

    # Biosphere Pentagon contract
    if "biosphere" in prompt_lower or (
        "pentagon contract" in prompt_lower and "chemicals" in prompt_lower
    ):
        return json.dumps(
            {
                "headline": (
                    "Startup Biosphere wins Pentagon contract to produce key defense chemicals"
                ),
                "what_happened": (
                    "The Pentagon awarded a contract to biotechnology startup Biosphere to "
                    "scale up a domestic biomanufacturing facility featuring 20,000-liter "
                    "vats producing critical materials ranging from bio-cement for helipads "
                    "to explosives precursors."
                ),
                "key_details": {
                    "actors": ["Pentagon", "Biosphere"],
                    "location": None,
                    "timing": "Within recent reporting period",
                    "action": "Contract award for chemical and biomanufacturing scale-up",
                },
                "significance": (
                    "Secures domestic defense supply chains for energetic materials and precursors."
                ),
                "confidence_qualification": (
                    "Official announcement reported by trade press; unverified independently."
                ),
            }
        )

    # F-47 procurement analysis
    if "f-47" in prompt_lower and ("f-15" in prompt_lower or "f-22" in prompt_lower):
        return json.dumps(
            {
                "headline": (
                    "Analysis examines F-47 procurement lessons from F-15 and F-22 programs"
                ),
                "what_happened": (
                    "An analytical study examines budgetary and production lessons from the "
                    "F-15 and F-22 fighter programs, arguing the Air Force must anticipate "
                    "difficult cost choices and fleet tradeoffs for the upcoming F-47 "
                    "acquisition."
                ),
                "key_details": {
                    "actors": ["Air Force"],
                    "location": None,
                    "timing": "Within recent reporting period",
                    "action": "Acquisition policy analysis for next-generation combat aircraft",
                },
                "significance": (
                    "Informs force structure planning and cost containment for "
                    "next-generation combat fleets."
                ),
                "confidence_qualification": (
                    "Single-source analytical commentary; reflects author assessment."
                ),
            }
        )

    # Pentagon procurement from Special Operations
    if "pentagon procurement" in prompt_lower and "special operations" in prompt_lower:
        return json.dumps(
            {
                "headline": (
                    "Pentagon examines SOCOM acquisition practices to accelerate unmanned systems"
                ),
                "what_happened": (
                    "US defense acquisition officials are examining commercial purchasing "
                    "models from the special operations community to buy commercial unmanned "
                    "systems rapidly and accelerate technology fielding across services."
                ),
                "key_details": {
                    "actors": ["Pentagon", "Special Operations Forces"],
                    "location": None,
                    "timing": "Within recent reporting period",
                    "action": "Procurement reform review for unmanned and autonomous systems",
                },
                "significance": (
                    "Highlights efforts to speed up acquisition cycles for autonomous systems."
                ),
                "confidence_qualification": (
                    "Single-source trade reporting; unverified independently."
                ),
            }
        )

    # Dynamic deterministic generation for prompts with source titles
    title_match = re.search(r"Title:\s*(.+)", prompt)
    if title_match:
        source_title = title_match.group(1).strip()
        snippet_match = re.search(r"Snippet:\s*(.+)", prompt)
        snippet_text = snippet_match.group(1).strip() if snippet_match else ""
        if snippet_text and snippet_text.lower() != "no snippet available.":
            what_text = f"{source_title}. {snippet_text}"
        else:
            what_text = f"{source_title}. Detailed reporting unconfirmed independently."

        title_lower = source_title.lower()
        if snippet_text and snippet_text.lower().startswith(title_lower[:25]):
            what_text = snippet_text
        elif snippet_text and snippet_text.lower() != "no snippet available.":
            what_text = f"{source_title}. {snippet_text}"
        else:
            what_text = f"{source_title}. Detailed reporting unconfirmed independently."

        if "pentagon" in title_lower:
            cand_actor = "Pentagon"
        elif "ministry of defence" in title_lower or "mod" in title_lower:
            cand_actor = "Ministry of Defence"
        elif "air force" in title_lower:
            cand_actor = "Air Force"
        elif "navy" in title_lower:
            cand_actor = "Navy"
        elif "army" in title_lower:
            cand_actor = "Army"
        elif "jordan" in title_lower:
            cand_actor = "military"
        elif "police" in title_lower:
            cand_actor = "police"
        elif "officials" in title_lower:
            cand_actor = "officials"
        elif "authorities" in title_lower:
            cand_actor = "authorities"
        else:
            cand_actor = "defence authorities"

        if "wins contract" in title_lower or "contract" in title_lower:
            cand_action = "Defense procurement contract award"
            cand_sig = (
                "Affects defence industrial capability, supply chain security, "
                "and inventory replenishment."
            )
        elif "airstrikes" in title_lower or "strikes" in title_lower or "shelling" in title_lower:
            cand_action = "Military strikes and operational combat action"
            cand_sig = (
                "Directly affects the tactical military balance and operational situation "
                "in the conflict sector."
            )
        elif "ceasefire" in title_lower or "truce" in title_lower:
            cand_action = "Ceasefire agreement and security arrangements"
            cand_sig = (
                "Directly impacts regional stability, civilian safety, and potential "
                "diplomatic resolution."
            )
        elif "talks" in title_lower or "summit" in title_lower or "call" in title_lower:
            cand_action = "Bilateral security talks and diplomatic dialogue"
            cand_sig = (
                "Signals diplomatic positioning and bilateral alignment among regional governments."
            )
        elif "procurement" in title_lower or "acquisition" in title_lower:
            cand_action = "Defense procurement and force structure planning"
            cand_sig = "Informs budgetary choices and modernization priorities for armed services."
        elif "satellite" in title_lower or "space" in title_lower:
            cand_action = "Military space and satellite operations"
            cand_sig = (
                "Reflects ongoing technological and operational developments in orbital "
                "defence capabilities."
            )
        elif "exercise" in title_lower or "drill" in title_lower:
            cand_action = "Military operational training exercise"
            cand_sig = "Maintains readiness and interoperability across deployed units."
        else:
            cand_action = "Reported security and defence development"
            cand_sig = "Carries operational implications for regional military and defence posture."

        return json.dumps(
            {
                "headline": source_title,
                "what_happened": what_text[:300],
                "key_details": {
                    "actors": [cand_actor],
                    "location": None,
                    "timing": (
                        "Date not specified in source"
                        if "Date not specified in source" in prompt
                        else "Within recent reporting period"
                    ),
                    "action": cand_action,
                },
                "significance": cand_sig,
                "confidence_qualification": conf_qual,
            }
        )

    # Default synthetic response clearly labeled as a test fixture
    return json.dumps(
        {
            "headline": "[FIXTURE] Synthetic Event Summary for Testing",
            "what_happened": (
                "Deterministic mock summary generated strictly for offline unit testing. "
                "No live intelligence content is asserted."
            ),
            "key_details": {
                "actors": ["Test Entity"],
                "location": "Test Sector",
                "timing": "Test Epoch",
                "action": "Synthetic testing procedure",
            },
            "significance": "Offline verification of digest assembly pipelines.",
            "confidence_qualification": conf_qual,
        }
    )


class GeminiClient(BaseLLMClient):
    """Production HTTP client for Google's Gemini API."""

    def __init__(
        self,
        api_key: str | SecretStr,
        model: str = "gemini-2.5-flash",
        base_url: str = "https://generativelanguage.googleapis.com",
        timeout: float = 30.0,
        max_retries: int = 2,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if isinstance(api_key, SecretStr):
            clean_key = api_key.get_secret_value().strip()
        else:
            clean_key = (api_key or "").strip()

        if not clean_key:
            raise GeminiConfigError("GEMINI_API_KEY must not be empty.")

        self._api_key = clean_key
        self._model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._transport = transport

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "gemini"

    def __repr__(self) -> str:
        return (
            f"GeminiClient(model='{self._model}', base_url='{self.base_url}', api_key='[REDACTED]')"
        )

    def is_available(self) -> bool:
        return bool(self._api_key)

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        format_type: str | None = "json",
        options: dict[str, Any] | None = None,
    ) -> str:
        url = f"{self.base_url}/v1beta/interactions"
        headers = {
            "x-goog-api-key": self._api_key,
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self._model,
            "input": prompt,
            "generation_config": {
                "temperature": 0.1,
                "max_output_tokens": 1024,
            },
        }
        if system:
            payload["system_instruction"] = system
        if format_type == "json":
            payload["response_format"] = [
                {
                    "type": "text",
                    "mime_type": "application/json",
                }
            ]

        timeout = httpx.Timeout(
            connect=10.0,
            read=self.timeout,
            write=10.0,
            pool=5.0,
        )

        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=timeout, transport=self._transport) as client:
                    res = client.post(url, headers=headers, json=payload)
                    status = res.status_code

                    if status == 200:
                        try:
                            data = res.json()
                        except Exception as exc:
                            raise GeminiResponseError(
                                f"Invalid JSON returned by Gemini API: {exc}"
                            ) from exc

                        return self._extract_text(data)

                    # Non-200 responses
                    error_data: dict[str, Any] = {}
                    try:
                        error_data = res.json().get("error", {})
                    except Exception:
                        pass
                    err_msg = error_data.get("message") or res.text
                    err_status = error_data.get("status")

                    if status in (401, 403) or (
                        status == 400
                        and ("API_KEY_INVALID" in err_msg or "not valid" in err_msg.lower())
                    ):
                        raise GeminiAuthError(
                            f"Gemini API authentication failed ({status}): {err_msg}"
                        )

                    if status == 404:
                        raise GeminiModelNotFoundError(
                            f"Gemini model '{self._model}' not found ({status}): {err_msg}"
                        )

                    if status == 429 or err_status == "RESOURCE_EXHAUSTED":
                        # Do not retry rate limits or quota exhaustion
                        raise GeminiQuotaError(
                            f"Gemini API quota or rate limit exhausted ({status}): {err_msg}"
                        )

                    if status in (500, 502, 503, 504):
                        if attempt < self.max_retries:
                            time.sleep(0.5 * (2**attempt))
                            continue
                        raise GeminiUnavailableError(
                            f"Gemini API server error ({status}) after "
                            f"{self.max_retries} retries: {err_msg}"
                        )

                    raise GeminiResponseError(f"Gemini API returned error ({status}): {err_msg}")

            except (httpx.ConnectError, httpx.NetworkError) as exc:
                if attempt < self.max_retries:
                    time.sleep(0.5 * (2**attempt))
                    continue
                raise GeminiUnavailableError(
                    f"Failed to connect to Gemini API after {self.max_retries} retries: {exc}"
                ) from exc

            except httpx.TimeoutException as exc:
                if attempt < self.max_retries:
                    time.sleep(0.5 * (2**attempt))
                    continue
                raise GeminiTimeoutError(
                    f"Gemini API request timed out after {self.timeout}s: {exc}"
                ) from exc

        raise GeminiUnavailableError("Gemini generation failed: retry limit exceeded")

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        """Extract generated text from an Interactions or GenerateContent API response."""
        # 1. Top-level output_text convenience property
        if "output_text" in data and isinstance(data["output_text"], str):
            return data["output_text"]

        # 2. Steps schema (modern Interactions API: steps[].content[].text)
        steps = data.get("steps")
        if isinstance(steps, list):
            for step in reversed(steps):
                if not isinstance(step, dict):
                    continue
                if step.get("type") == "model_output":
                    content = step.get("content")
                    if isinstance(content, list):
                        parts: list[str] = []
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                txt = item.get("text")
                                if txt is not None:
                                    parts.append(txt)
                        if parts:
                            return "".join(parts)
                    elif isinstance(content, str) and content:
                        return content
                    if "text" in step and isinstance(step["text"], str) and step["text"]:
                        return step["text"]

        # 3. Outputs schema (legacy Interactions API: outputs[].text)
        outputs = data.get("outputs")
        if isinstance(outputs, list):
            out_parts: list[str] = []
            for out in outputs:
                if isinstance(out, dict):
                    txt = out.get("text")
                    if txt is not None:
                        out_parts.append(txt)
            if out_parts:
                return "".join(out_parts)

        # 4. Candidates schema (backward-compatibility with generateContent fixtures)
        candidates = data.get("candidates")
        if isinstance(candidates, list) and candidates:
            first_candidate = candidates[0]
            if first_candidate.get("finishReason") == "SAFETY":
                raise GeminiResponseError(
                    "Gemini candidate generation was blocked by safety filters"
                )
            content = first_candidate.get("content") or {}
            parts_list = content.get("parts") or []
            if parts_list and isinstance(parts_list, list):
                raw_text = parts_list[0].get("text")
                if raw_text is not None:
                    return raw_text

        # 5. Check if interaction status explicitly indicates failure or block
        interaction_status = data.get("status")
        if interaction_status in ("failed", "cancelled"):
            error_obj = data.get("error") or {}
            msg = error_obj.get("message") if isinstance(error_obj, dict) else str(error_obj)
            raise GeminiResponseError(f"Gemini interaction {interaction_status}: {msg}")

        prompt_feedback = data.get("promptFeedback") or {}
        block_reason = prompt_feedback.get("blockReason")
        if block_reason:
            raise GeminiResponseError(f"Gemini content generation blocked: {block_reason}")

        raise GeminiResponseError("Gemini response missing text output")


class MockGeminiClient(BaseLLMClient):
    """In-memory mocked Gemini client for offline tests."""

    def __init__(
        self,
        model: str = "mock-gemini-2.5-flash",
        canned_response: str | None = None,
        should_timeout: bool = False,
        should_auth_fail: bool = False,
        should_quota_exhaust: bool = False,
        should_be_unavailable: bool = False,
        should_model_not_found: bool = False,
        should_return_malformed: bool = False,
        custom_generator: Callable[[str], str] | None = None,
    ) -> None:
        self._model = model
        self.canned_response = canned_response
        self.should_timeout = should_timeout
        self.should_auth_fail = should_auth_fail
        self.should_quota_exhaust = should_quota_exhaust
        self.should_be_unavailable = should_be_unavailable
        self.should_model_not_found = should_model_not_found
        self.should_return_malformed = should_return_malformed
        self.custom_generator = custom_generator
        self.calls: list[dict[str, Any]] = []
        self.last_metrics: dict[str, Any] = {}

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "gemini"

    def is_available(self) -> bool:
        return not (self.should_be_unavailable or self.should_auth_fail)

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        format_type: str | None = "json",
        options: dict[str, Any] | None = None,
    ) -> str:
        self.calls.append({"prompt": prompt, "system": system, "format": format_type})

        if self.should_auth_fail:
            raise GeminiAuthError("Mocked: Gemini API key invalid (403 Forbidden)")

        if self.should_quota_exhaust:
            raise GeminiQuotaError("Mocked: Gemini API quota exceeded (429 Resource Exhausted)")

        if self.should_be_unavailable:
            raise GeminiUnavailableError(
                "Mocked: Gemini server unavailable (503 Service Unavailable)"
            )

        if self.should_timeout:
            raise GeminiTimeoutError("Mocked: Gemini request timed out")

        if self.should_model_not_found:
            raise GeminiModelNotFoundError(f"Mocked: Model '{self._model}' not found (404)")

        if self.should_return_malformed:
            return "{ unparseable json: [incomplete "

        if self.custom_generator is not None:
            resp = self.custom_generator(prompt)
        elif self.canned_response is not None:
            resp = self.canned_response
        else:
            resp = _generate_mock_summary_json(prompt)

        self.last_metrics = {
            "model": self._model,
            "prompt_tokens": len(prompt.split()),
            "eval_tokens": len(resp.split()),
            "eval_tokens_per_second": 8.3,
            "elapsed_seconds": 0.05,
            "done_reason": "stop",
        }
        return resp


def resolve_summarization_clients(
    settings: "Settings | None" = None,
    provider: str | None = None,
    model: str | None = None,
    mock: bool = False,
) -> tuple[BaseLLMClient, BaseLLMClient | None]:
    """Resolve primary and optional fallback LLM clients according to configuration."""
    from geopulse.config import get_settings

    cfg = settings or get_settings()
    effective_provider = (provider or cfg.llm_provider).strip().lower()

    if effective_provider == "ollama":
        ollama_model = model or cfg.ollama_model
        if mock:
            return MockOllamaClient(model=f"mock-{ollama_model}"), None
        return (
            OllamaClient(
                base_url=cfg.ollama_base_url,
                model=ollama_model,
                read_timeout=getattr(cfg, "ollama_timeout", 120.0),
                request_timeout=getattr(cfg, "ollama_request_timeout", 180.0),
                max_retries=getattr(cfg, "ollama_max_retries", 1),
            ),
            None,
        )

    if effective_provider == "gemini":
        gemini_model = model or cfg.gemini_model
        if mock:
            return MockGeminiClient(model=f"mock-{gemini_model}"), None
        if not cfg.is_gemini_configured:
            raise GeminiConfigError(
                "GEMINI_API_KEY is not configured, but provider is set to 'gemini'."
            )
        assert cfg.gemini_api_key is not None
        gemini_key = cfg.gemini_api_key.get_secret_value()
        return (
            GeminiClient(
                api_key=gemini_key,
                model=gemini_model,
                base_url=cfg.gemini_base_url,
                timeout=cfg.gemini_timeout,
                max_retries=cfg.gemini_max_retries,
            ),
            None,
        )

    if effective_provider == "auto":
        gemini_model = model or cfg.gemini_model
        ollama_model = cfg.ollama_model
        if mock:
            return (
                MockGeminiClient(model=f"mock-{gemini_model}"),
                MockOllamaClient(model=f"mock-{ollama_model}"),
            )
        if not cfg.is_gemini_configured:
            # Gemini not configured in auto mode: smoothly use Ollama without fallback
            return (
                OllamaClient(
                    base_url=cfg.ollama_base_url,
                    model=ollama_model,
                    read_timeout=getattr(cfg, "ollama_timeout", 120.0),
                    request_timeout=getattr(cfg, "ollama_request_timeout", 180.0),
                    max_retries=getattr(cfg, "ollama_max_retries", 1),
                ),
                None,
            )
        assert cfg.gemini_api_key is not None
        gemini_key = cfg.gemini_api_key.get_secret_value()
        primary_client = GeminiClient(
            api_key=gemini_key,
            model=gemini_model,
            base_url=cfg.gemini_base_url,
            timeout=cfg.gemini_timeout,
            max_retries=cfg.gemini_max_retries,
        )
        fallback_client = OllamaClient(
            base_url=cfg.ollama_base_url,
            model=ollama_model,
            read_timeout=getattr(cfg, "ollama_timeout", 120.0),
            request_timeout=getattr(cfg, "ollama_request_timeout", 180.0),
            max_retries=getattr(cfg, "ollama_max_retries", 1),
        )
        return primary_client, fallback_client

    raise ValueError(f"Unknown LLM provider: {effective_provider}")
