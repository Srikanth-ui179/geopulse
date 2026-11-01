"""Exception hierarchy for Telegram delivery and retry safety."""


class TelegramDeliveryError(Exception):
    """Base exception for all Telegram delivery errors."""


class TelegramConfigError(TelegramDeliveryError):
    """Raised when Telegram credentials or settings are missing or invalid."""


class TelegramNetworkError(TelegramDeliveryError):
    """Raised on network connection failures, DNS errors, or server-side 5xx errors."""


class TelegramTimeoutError(TelegramDeliveryError):
    """Raised when an HTTP request to the Telegram Bot API times out.

    Attributes:
        is_ambiguous: True if the timeout occurred while waiting for the response
                      (meaning the server might have processed and posted the message).
    """

    def __init__(self, message: str, is_ambiguous: bool = False) -> None:
        super().__init__(message)
        self.is_ambiguous = is_ambiguous


class TelegramRateLimitError(TelegramDeliveryError):
    """Raised when Telegram returns HTTP 429 Too Many Requests.

    Attributes:
        retry_after: Number of seconds to wait before retrying, as reported by Telegram.
    """

    def __init__(self, message: str, retry_after: float = 1.0) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class TelegramConflictError(TelegramDeliveryError):
    """Raised when Telegram returns HTTP 409 Conflict.

    Typically occurs when another polling instance or webhook is active.
    """

    def __init__(
        self,
        message: str = "Telegram polling conflict (HTTP 409): another instance is active.",
    ) -> None:
        super().__init__(message)


class TelegramPermanentError(TelegramDeliveryError):
    """Raised on non-retryable 4xx client errors (e.g. 400 Bad Request, 401 Unauthorized).

    Attributes:
        status_code: HTTP response status code.
        description: Error description provided by Telegram.
    """

    def __init__(self, status_code: int, description: str) -> None:
        super().__init__(f"Telegram API permanent error ({status_code}): {description}")
        self.status_code = status_code
        self.description = description


class TelegramDuplicateDeliveryError(TelegramDeliveryError):
    """Raised when attempting to deliver a briefing that was already successfully sent."""


class TelegramRetryExhaustedError(TelegramDeliveryError):
    """Raised when all retry attempts for a message chunk have failed."""


class CorruptDeliveryStateError(TelegramDeliveryError):
    """Raised when the delivery state file exists but is corrupted, empty, or unparseable.

    Fails closed to prevent accidental duplicate deliveries when prior delivery history
    cannot be reliably verified. Requires operator intervention to inspect, repair, or reset.
    """
