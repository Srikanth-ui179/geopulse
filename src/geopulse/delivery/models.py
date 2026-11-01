"""Data models for Telegram delivery and briefing lifecycle."""

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class TelegramMessageResponse(BaseModel):
    """Normalized response from Telegram Bot API sendMessage call."""

    model_config = ConfigDict(frozen=True)

    message_id: int
    chat_id: int | str
    date: int
    raw: dict[str, Any] = Field(default_factory=dict)


class BriefingRecord(BaseModel):
    """Persistence model representing a daily briefing delivery run."""

    model_config = ConfigDict(frozen=True)

    id: str
    briefing_date: date
    edition: str | None = None
    status: Literal["pending", "processing", "sent", "partial", "failed", "ambiguous"] = "pending"
    started_at: datetime
    completed_at: datetime | None = None
    telegram_message_count: int = 0
    error_message: str | None = None
    chunks_delivered: int = 0
    telegram_message_ids: tuple[int, ...] = Field(default_factory=tuple)
    digest_hash: str | None = None
    delivered_event_hashes: tuple[str, ...] = Field(default_factory=tuple)


class DeliveryResult(BaseModel):
    """Summary of the delivery execution returned by the delivery service."""

    model_config = ConfigDict(frozen=True)

    briefing_id: str
    briefing_date: date
    edition: str | None = None
    status: Literal["sent", "partial", "failed", "skipped_duplicate", "ambiguous"]
    chunks_total: int
    chunks_delivered: int
    message_ids: tuple[int, ...] = Field(default_factory=tuple)
    is_dry_run: bool = False
    error_message: str | None = None
    warning_message: str | None = None
    resumed_from_chunk: int = 0
