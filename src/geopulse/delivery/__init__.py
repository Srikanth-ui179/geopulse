"""Telegram delivery subsystem for formatting, sending, and retry safety."""

from geopulse.delivery.client import BaseTelegramClient, MockTelegramClient, TelegramClient
from geopulse.delivery.exceptions import (
    CorruptDeliveryStateError,
    TelegramConfigError,
    TelegramConflictError,
    TelegramDeliveryError,
    TelegramDuplicateDeliveryError,
    TelegramNetworkError,
    TelegramPermanentError,
    TelegramRateLimitError,
    TelegramRetryExhaustedError,
    TelegramTimeoutError,
)
from geopulse.delivery.formatter import (
    CATEGORY_LABELS,
    CONFIDENCE_BADGES,
    MAX_TELEGRAM_MESSAGE_LENGTH,
    SAFE_CHUNK_LIMIT,
    TelegramDigestFormatter,
    escape_html,
    format_telegram_digest,
)
from geopulse.delivery.models import BriefingRecord, DeliveryResult, TelegramMessageResponse
from geopulse.delivery.repository import (
    DeliveryRepository,
    InMemoryDeliveryRepository,
    JsonFileDeliveryRepository,
    SupabaseDeliveryRepository,
)
from geopulse.delivery.service import TelegramDeliveryService

__all__ = [
    "CATEGORY_LABELS",
    "CONFIDENCE_BADGES",
    "MAX_TELEGRAM_MESSAGE_LENGTH",
    "SAFE_CHUNK_LIMIT",
    "BaseTelegramClient",
    "BriefingRecord",
    "CorruptDeliveryStateError",
    "DeliveryRepository",
    "DeliveryResult",
    "InMemoryDeliveryRepository",
    "JsonFileDeliveryRepository",
    "MockTelegramClient",
    "SupabaseDeliveryRepository",
    "TelegramClient",
    "TelegramConfigError",
    "TelegramConflictError",
    "TelegramDeliveryError",
    "TelegramDeliveryService",
    "TelegramDigestFormatter",
    "TelegramDuplicateDeliveryError",
    "TelegramMessageResponse",
    "TelegramNetworkError",
    "TelegramPermanentError",
    "TelegramRateLimitError",
    "TelegramRetryExhaustedError",
    "TelegramTimeoutError",
    "escape_html",
    "format_telegram_digest",
]
