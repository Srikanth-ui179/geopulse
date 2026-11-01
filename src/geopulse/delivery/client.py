"""Official Telegram Bot API client and test doubles."""

from abc import ABC, abstractmethod
from typing import Any

import httpx

from geopulse.delivery.exceptions import (
    TelegramConfigError,
    TelegramConflictError,
    TelegramNetworkError,
    TelegramPermanentError,
    TelegramRateLimitError,
    TelegramTimeoutError,
)
from geopulse.delivery.models import TelegramMessageResponse


class BaseTelegramClient(ABC):
    """Abstract interface for sending messages via Telegram Bot API."""

    @property
    @abstractmethod
    def chat_id(self) -> str:
        """The target Telegram chat identifier."""
        ...

    @abstractmethod
    def send_message(
        self,
        text: str,
        chat_id: str | int | None = None,
        parse_mode: str = "HTML",
        disable_web_page_preview: bool = True,
    ) -> TelegramMessageResponse:
        """Deliver a message to target Telegram chat (or default configured chat)."""
        ...

    @abstractmethod
    def get_updates(
        self,
        offset: int | None = None,
        limit: int = 100,
        timeout: int = 20,
    ) -> list[dict[str, Any]]:
        """Poll incoming Telegram updates via long polling."""
        ...


class TelegramClient(BaseTelegramClient):
    """Production HTTP client for the official Telegram Bot API."""

    def __init__(
        self,
        token: str,
        chat_id: str,
        base_url: str = "https://api.telegram.org",
        timeout: float = 15.0,
    ) -> None:
        clean_token = (token or "").strip()
        clean_chat_id = (chat_id or "").strip()
        if not clean_token:
            raise TelegramConfigError("Telegram bot token must not be empty.")
        if not clean_chat_id:
            raise TelegramConfigError("Telegram chat ID must not be empty.")

        self._token = clean_token
        self._chat_id = clean_chat_id
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @property
    def chat_id(self) -> str:
        return self._chat_id

    def __repr__(self) -> str:
        # Strictly redact token from object string representations
        return (
            f"TelegramClient(chat_id='{self._chat_id}', "
            f"base_url='{self.base_url}', token='[REDACTED]')"
        )

    def send_message(
        self,
        text: str,
        chat_id: str | int | None = None,
        parse_mode: str = "HTML",
        disable_web_page_preview: bool = True,
    ) -> TelegramMessageResponse:
        target_chat = str(chat_id).strip() if chat_id is not None else self._chat_id
        url = f"{self.base_url}/bot{self._token}/sendMessage"
        payload: dict[str, Any] = {
            "chat_id": target_chat,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": disable_web_page_preview,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(url, json=payload)
        except httpx.ConnectTimeout as exc:
            raise TelegramNetworkError(f"Connection to Telegram timed out: {exc}") from exc
        except httpx.ReadTimeout as exc:
            # Ambiguous: server may have already processed and sent the message
            raise TelegramTimeoutError(
                f"Read timeout waiting for Telegram API response: {exc}",
                is_ambiguous=True,
            ) from exc
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            raise TelegramNetworkError(f"Network error contacting Telegram: {exc}") from exc
        except Exception as exc:
            raise TelegramNetworkError(f"Unexpected transport error: {exc}") from exc

        # Parse response body safely
        try:
            data = res.json()
        except Exception as exc:
            if res.status_code >= 500:
                raise TelegramNetworkError(
                    f"Telegram server error ({res.status_code}) with unparseable body"
                ) from exc
            raise TelegramNetworkError(
                f"Telegram returned HTTP {res.status_code} with non-JSON body: {exc}"
            ) from exc

        # Handle rate limits
        if res.status_code == 429:
            retry_after = 1.0
            parameters = data.get("parameters")
            if isinstance(parameters, dict) and "retry_after" in parameters:
                try:
                    retry_after = float(parameters["retry_after"])
                except (ValueError, TypeError):
                    retry_after = 1.0
            description = data.get("description", "Too Many Requests")
            raise TelegramRateLimitError(
                message=f"Rate limit exceeded: {description}",
                retry_after=retry_after,
            )

        # Handle permanent client errors (4xx)
        if 400 <= res.status_code < 500:
            description = data.get("description", "Client Error")
            raise TelegramPermanentError(status_code=res.status_code, description=description)

        # Handle server errors (5xx)
        if res.status_code >= 500:
            description = data.get("description", "Internal Server Error")
            raise TelegramNetworkError(
                f"Telegram server returned HTTP {res.status_code}: {description}"
            )

        # Check Telegram ok status
        if not data.get("ok"):
            err_code = data.get("error_code", res.status_code)
            desc = data.get("description", "Unknown Telegram error")
            if err_code == 429:
                raise TelegramRateLimitError(message=desc)
            if 400 <= err_code < 500:
                raise TelegramPermanentError(status_code=err_code, description=desc)
            raise TelegramNetworkError(f"Telegram API reported error ({err_code}): {desc}")

        result_obj = data.get("result", {})
        return TelegramMessageResponse(
            message_id=result_obj.get("message_id", 0),
            chat_id=result_obj.get("chat", {}).get("id", target_chat),
            date=result_obj.get("date", 0),
            raw=data,
        )

    def get_updates(
        self,
        offset: int | None = None,
        limit: int = 100,
        timeout: int = 20,
    ) -> list[dict[str, Any]]:
        url = f"{self.base_url}/bot{self._token}/getUpdates"
        params: dict[str, Any] = {
            "timeout": timeout,
            "limit": limit,
        }
        if offset is not None:
            params["offset"] = offset

        request_timeout = max(self.timeout, float(timeout + 5))
        try:
            with httpx.Client(timeout=request_timeout) as client:
                res = client.get(url, params=params)
        except httpx.ConnectTimeout as exc:
            raise TelegramNetworkError(f"Connection to Telegram timed out: {exc}") from exc
        except httpx.ReadTimeout:
            # Normal on long polling when no updates arrive within timeout
            return []
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            raise TelegramNetworkError(f"Network error contacting Telegram: {exc}") from exc
        except Exception as exc:
            raise TelegramNetworkError(f"Unexpected transport error: {exc}") from exc

        # Handle polling conflicts (e.g. another instance or webhook active)
        if res.status_code == 409:
            raise TelegramConflictError(
                "Telegram polling conflict (HTTP 409): another instance or webhook is active."
            )

        # Handle rate limits
        if res.status_code == 429:
            retry_after = 1.0
            try:
                data = res.json()
                parameters = data.get("parameters", {})
                if isinstance(parameters, dict) and "retry_after" in parameters:
                    retry_after = float(parameters["retry_after"])
            except Exception:
                pass
            raise TelegramRateLimitError("Telegram rate limit encountered", retry_after=retry_after)

        # Handle client errors
        if 400 <= res.status_code < 500:
            raise TelegramPermanentError(status_code=res.status_code, description=res.text)

        # Handle server errors
        if res.status_code >= 500:
            raise TelegramNetworkError(f"Telegram server error (HTTP {res.status_code})")

        try:
            data = res.json()
        except Exception as exc:
            raise TelegramNetworkError(f"Telegram returned non-JSON response: {exc}") from exc

        result = data.get("result")
        if isinstance(result, list):
            return result
        return []


class MockTelegramClient(BaseTelegramClient):
    """In-memory test double for Telegram delivery without external network requests."""

    def __init__(
        self,
        chat_id: str = "mock-chat-123",
        fail_first_n_calls: int = 0,
        should_fail_rate_limit: float | None = None,
        should_fail_network: bool = False,
        should_timeout_ambiguous: bool = False,
        should_fail_permanent: tuple[int, str] | None = None,
    ) -> None:
        self._chat_id = chat_id
        self.fail_first_n_calls = fail_first_n_calls
        self.should_fail_rate_limit = should_fail_rate_limit
        self.should_fail_network = should_fail_network
        self.should_timeout_ambiguous = should_timeout_ambiguous
        self.should_fail_permanent = should_fail_permanent

        self.call_count: int = 0
        self.sent_messages: list[dict[str, Any]] = []
        self._mock_updates: list[dict[str, Any]] = []

    @property
    def chat_id(self) -> str:
        return self._chat_id

    def send_message(
        self,
        text: str,
        chat_id: str | int | None = None,
        parse_mode: str = "HTML",
        disable_web_page_preview: bool = True,
    ) -> TelegramMessageResponse:
        self.call_count += 1
        target_chat = str(chat_id).strip() if chat_id is not None else self._chat_id

        # Check transient failure counter
        if self.call_count <= self.fail_first_n_calls:
            if self.should_fail_rate_limit is not None:
                raise TelegramRateLimitError(
                    "Mocked rate limit",
                    retry_after=self.should_fail_rate_limit,
                )
            if self.should_fail_network:
                raise TelegramNetworkError("Mocked network error")
            if self.should_timeout_ambiguous:
                raise TelegramTimeoutError("Mocked ambiguous read timeout", is_ambiguous=True)

        # Permanent failures apply unconditionally if set
        if self.should_fail_permanent is not None:
            code, desc = self.should_fail_permanent
            raise TelegramPermanentError(status_code=code, description=desc)

        # Record sent message
        msg_id = 1000 + len(self.sent_messages) + 1
        record = {
            "message_id": msg_id,
            "chat_id": target_chat,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": disable_web_page_preview,
        }
        self.sent_messages.append(record)

        return TelegramMessageResponse(
            message_id=msg_id,
            chat_id=target_chat,
            date=1729680000,
            raw={"ok": True, "result": record},
        )

    def enqueue_update(
        self,
        update_id: int,
        chat_id: str | int = "mock-chat",
        user_id: str | int | None = None,
        text: str = "",
        from_id: str | int | None = None,
    ) -> None:
        """Enqueue a mock incoming user message for testing."""
        raw_uid = user_id if user_id is not None else (from_id if from_id is not None else 100)
        uid = int(raw_uid) if str(raw_uid).isdigit() else 100
        self._mock_updates.append(
            {
                "update_id": update_id,
                "message": {
                    "message_id": update_id * 10,
                    "from": {"id": uid, "first_name": "TestUser"},
                    "chat": {"id": chat_id, "type": "private"},
                    "date": 1729680000,
                    "text": text,
                },
            }
        )

    def get_updates(
        self,
        offset: int | None = None,
        limit: int = 100,
        timeout: int = 20,
    ) -> list[dict[str, Any]]:
        self.call_count += 1
        if self.should_fail_rate_limit is not None:
            raise TelegramRateLimitError(
                "Mocked rate limit",
                retry_after=self.should_fail_rate_limit,
            )
        if self.should_fail_network:
            raise TelegramNetworkError("Mocked network error")

        if offset is None:
            batch = list(self._mock_updates[:limit])
            self._mock_updates = self._mock_updates[limit:]
            return batch

        matching = [u for u in self._mock_updates if u.get("update_id", 0) >= offset]
        batch = matching[:limit]
        self._mock_updates = [u for u in self._mock_updates if u not in batch]
        return batch
