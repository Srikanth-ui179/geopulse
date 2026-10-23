"""Delivery service orchestrating bounded retries, deduplication, and state tracking."""

import hashlib
import logging
import time
from collections.abc import Callable
from datetime import UTC, date, datetime

from geopulse.config import Settings
from geopulse.delivery.client import BaseTelegramClient, TelegramClient
from geopulse.delivery.exceptions import (
    CorruptDeliveryStateError,
    TelegramNetworkError,
    TelegramPermanentError,
    TelegramRateLimitError,
    TelegramTimeoutError,
)
from geopulse.delivery.models import BriefingRecord, DeliveryResult
from geopulse.delivery.repository import (
    DeliveryRepository,
    JsonFileDeliveryRepository,
)

logger = logging.getLogger(__name__)


def compute_digest_hash(digest_chunks: tuple[str, ...] | list[str]) -> str:
    """Compute deterministic SHA256 hex digest across all chunks in order."""
    combined_text = "".join(digest_chunks)
    return hashlib.sha256(combined_text.encode("utf-8")).hexdigest()


class TelegramDeliveryService:
    """Orchestrates resilient Telegram digest delivery with bounded retries and deduplication."""

    def __init__(
        self,
        client: BaseTelegramClient | None = None,
        repository: DeliveryRepository | None = None,
        settings: Settings | None = None,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
        max_backoff: float = 30.0,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self.repository = repository or JsonFileDeliveryRepository()
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.max_backoff = max_backoff
        self.sleeper = sleeper

        if client is not None:
            self.client = client
        else:
            from geopulse.config import get_settings

            cfg = self.settings or get_settings()
            token, chat_id = cfg.require_telegram()
            self.client = TelegramClient(token=token, chat_id=chat_id)

    def deliver_digest(
        self,
        digest_chunks: tuple[str, ...],
        briefing_date: date | None = None,
        force: bool = False,
        dry_run: bool = False,
    ) -> DeliveryResult:
        """Deliver digest chunks to Telegram with deduplication, resumption, and retry safety."""
        target_date = briefing_date or date.today()
        total_chunks = len(digest_chunks)

        # Compute deterministic content digest hash
        digest_hash = compute_digest_hash(digest_chunks)

        # 1. State check and resumption resolution against repository
        try:
            existing = self.repository.get_briefing_by_date(target_date)
        except CorruptDeliveryStateError as exc:
            logger.error("State repository error: %s", exc)
            return DeliveryResult(
                briefing_id="corrupt-state",
                briefing_date=target_date,
                status="failed",
                chunks_total=total_chunks,
                chunks_delivered=0,
                is_dry_run=dry_run,
                error_message=str(exc),
            )
        resume_from_chunk = 0
        delivered_message_ids: list[int] = []
        warning_msg: str | None = None


        if existing:
            # Case A: Already fully sent
            if existing.status == "sent":
                if not force:
                    logger.info("Briefing for %s was already delivered. Skipping.", target_date)
                    delivered_count = (
                        existing.chunks_delivered or existing.telegram_message_count
                    )
                    return DeliveryResult(
                        briefing_id=existing.id,
                        briefing_date=target_date,
                        status="skipped_duplicate",
                        chunks_total=total_chunks,
                        chunks_delivered=delivered_count,
                        message_ids=existing.telegram_message_ids,
                        is_dry_run=dry_run,
                        error_message=(
                            f"Briefing for {target_date} has already been delivered. "
                            "Use --force to resend."
                        ),
                    )
                logger.info(
                    "Briefing for %s was previously sent, but --force requested. Restarting.",
                    target_date,
                )

            # Case B: Ambiguous timeout occurred in previous run
            elif existing.status == "ambiguous":
                if not force:
                    logger.warning(
                        "Previous delivery for %s had ambiguous timeout on chunk %d/%d. Halting.",
                        target_date,
                        existing.chunks_delivered + 1,
                        existing.telegram_message_count,
                    )
                    return DeliveryResult(
                        briefing_id=existing.id,
                        briefing_date=target_date,
                        status="ambiguous",
                        chunks_total=total_chunks,
                        chunks_delivered=existing.chunks_delivered,
                        message_ids=existing.telegram_message_ids,
                        is_dry_run=dry_run,
                        error_message=(
                            f"Previous delivery halted due to an ambiguous timeout on chunk "
                            f"{existing.chunks_delivered + 1}/{existing.telegram_message_count}. "
                            "Delivery outcome unconfirmed. To prevent duplicates or missing "
                            "chunks, explicit recovery is required (pass --force to re-send)."
                        ),
                    )
                logger.warning(
                    "WARNING: --force used after ambiguous delivery for %s. "
                    "Previously accepted chunks may be duplicated in the recipient's chat.",
                    target_date,
                )
                warning_msg = (
                    "Force re-delivery after an ambiguous timeout. "
                    "Previously accepted chunks may be duplicated in the chat."
                )


            # Case C: Partial or interrupted delivery exists
            elif existing.status in {"partial", "processing"}:
                if force:
                    logger.info(
                        "Prior partial delivery exists for %s, but --force requested. Restarting.",
                        target_date,
                    )
                elif existing.digest_hash and existing.digest_hash != digest_hash:
                    logger.warning(
                        "Digest content changed since partial delivery for %s.", target_date
                    )
                    return DeliveryResult(
                        briefing_id=existing.id,
                        briefing_date=target_date,
                        status="partial",
                        chunks_total=total_chunks,
                        chunks_delivered=existing.chunks_delivered,
                        message_ids=existing.telegram_message_ids,
                        is_dry_run=dry_run,
                        error_message=(
                            f"Digest content has changed since partial delivery for {target_date} "
                            f"({existing.chunks_delivered}/{existing.telegram_message_count} "
                            "chunks delivered). Resuming would produce mismatched content. "
                            "Pass --force to restart delivery."
                        ),
                    )
                elif 0 < existing.chunks_delivered < total_chunks:
                    # Safe to resume: chunks 1..existing.chunks_delivered were confirmed accepted
                    resume_from_chunk = existing.chunks_delivered
                    delivered_message_ids = list(existing.telegram_message_ids)
                    logger.info(
                        "Resuming delivery for %s: skipping %d already-delivered chunk(s), "
                        "delivering chunks %d..%d.",
                        target_date,
                        resume_from_chunk,
                        resume_from_chunk + 1,
                        total_chunks,
                    )


        # 2. Dry-run / preview mode: simulate successful delivery without sending
        if dry_run:
            logger.info("Dry-run delivery for %s (%d chunk(s)).", target_date, total_chunks)
            return DeliveryResult(
                briefing_id="dry-run",
                briefing_date=target_date,
                status="sent",
                chunks_total=total_chunks,
                chunks_delivered=total_chunks,
                is_dry_run=True,
            )

        # Acquire delivery lock to prevent concurrent executions
        if not self.repository.acquire_lock():
            logger.warning("Another delivery process is currently active. Aborting.")
            return DeliveryResult(
                briefing_id="lock-blocked",
                briefing_date=target_date,
                status="failed",
                chunks_total=total_chunks,
                chunks_delivered=0,
                is_dry_run=False,
                error_message=(
                    "Another delivery process is currently active on this state file "
                    "(delivery lock held). Aborting to prevent concurrent duplicate delivery."
                ),
            )

        try:
            # 3. Create or update briefing state record
            if existing and resume_from_chunk > 0:
                briefing_record = existing
            else:
                try:
                    briefing_record = self.repository.create_briefing(
                        briefing_date=target_date,
                        total_chunks=total_chunks,
                        digest_hash=digest_hash,
                    )
                except Exception as exc:
                    logger.error("Failed to create briefing state record: %s", exc)
                    return DeliveryResult(
                        briefing_id="create-failed",
                        briefing_date=target_date,
                        status="failed",
                        chunks_total=total_chunks,
                        chunks_delivered=0,
                        is_dry_run=False,
                        error_message=f"Failed to create briefing state record: {exc}",
                    )

            error_msg: str | None = None
            is_ambiguous_failure: bool = False

            # 4. Sequentially deliver each message chunk with resumption and retry safety
            for chunk_idx, chunk_text in enumerate(digest_chunks, start=1):
                if chunk_idx <= resume_from_chunk:
                    logger.debug(
                        "Chunk %d/%d was already delivered in a previous run. Skipping.",
                        chunk_idx,
                        total_chunks,
                    )
                    continue

                chunk_delivered = False
                last_error: Exception | None = None

                for attempt in range(1, self.max_retries + 1):
                    try:
                        logger.debug(
                            "Sending chunk %d/%d (attempt %d/%d)...",
                            chunk_idx,
                            total_chunks,
                            attempt,
                            self.max_retries,
                        )
                        resp = self.client.send_message(chunk_text)
                        delivered_message_ids.append(resp.message_id)
                        chunk_delivered = True

                        # Immediately record incremental progress to preserve state on crash
                        briefing_record = BriefingRecord(
                            id=briefing_record.id,
                            briefing_date=target_date,
                            status="processing",
                            started_at=briefing_record.started_at,
                            completed_at=None,
                            telegram_message_count=total_chunks,
                            chunks_delivered=len(delivered_message_ids),
                            telegram_message_ids=tuple(delivered_message_ids),
                            digest_hash=digest_hash,
                        )
                        self.repository.update_briefing(briefing_record)
                        break
                    except TelegramRateLimitError as exc:
                        last_error = exc
                        sleep_time = min(exc.retry_after, self.max_backoff)
                        logger.warning(
                            "Telegram rate limit on chunk %d/%d (attempt %d/%d). "
                            "Backing off for %.2fs.",
                            chunk_idx,
                            total_chunks,
                            attempt,
                            self.max_retries,
                            sleep_time,
                        )
                        if attempt < self.max_retries:
                            self.sleeper(sleep_time)
                    except TelegramNetworkError as exc:
                        last_error = exc
                        sleep_time = min(self.backoff_factor**attempt, self.max_backoff)
                        logger.warning(
                            "Network error delivering chunk %d/%d (attempt %d/%d): %s. "
                            "Retrying in %.2fs...",
                            chunk_idx,
                            total_chunks,
                            attempt,
                            self.max_retries,
                            exc,
                            sleep_time,
                        )
                        if attempt < self.max_retries:
                            self.sleeper(sleep_time)
                    except TelegramTimeoutError as exc:
                        last_error = exc
                        if exc.is_ambiguous:
                            # Ambiguous delivery outcome: response was lost after request was sent.
                            # Do not blindly retry to prevent duplicating the message in chat.
                            logger.error(
                                "Ambiguous read timeout delivering chunk %d/%d: %s. "
                                "Message may have been delivered by Telegram. "
                                "Halting retries and marking status as ambiguous.",
                                chunk_idx,
                                total_chunks,
                                exc,
                            )
                            is_ambiguous_failure = True
                            break
                        # If connect timeout (safe to retry), back off
                        sleep_time = min(self.backoff_factor**attempt, self.max_backoff)
                        if attempt < self.max_retries:
                            self.sleeper(sleep_time)
                    except TelegramPermanentError as exc:
                        # Non-retryable client errors (400, 401, 403)
                        logger.error(
                            "Permanent Telegram error on chunk %d/%d: %s. Aborting.",
                            chunk_idx,
                            total_chunks,
                            exc,
                        )
                        last_error = exc
                        break

                if not chunk_delivered:
                    # Chunk failed after retries or permanent error
                    if is_ambiguous_failure:
                        error_msg = (
                            f"Ambiguous timeout on chunk {chunk_idx}/{total_chunks}. "
                            "Halted to prevent duplicate sends. Pass --force to retry."
                        )
                    elif isinstance(last_error, TelegramPermanentError):
                        error_msg = str(last_error)
                    else:
                        error_msg = (
                            f"Failed delivering chunk {chunk_idx}/{total_chunks}: {last_error}"
                        )
                    break

            # 5. Finalize delivery state record
            is_complete = len(delivered_message_ids) == total_chunks
            if is_complete:
                final_status = "sent"
            elif is_ambiguous_failure:
                final_status = "ambiguous"
            elif delivered_message_ids:
                final_status = "partial"
            else:
                final_status = "failed"

            now_dt = datetime.now(UTC)
            updated_record = BriefingRecord(
                id=briefing_record.id,
                briefing_date=target_date,
                status=final_status,
                started_at=briefing_record.started_at,
                completed_at=now_dt if is_complete else None,
                telegram_message_count=total_chunks,
                chunks_delivered=len(delivered_message_ids),
                telegram_message_ids=tuple(delivered_message_ids),
                error_message=error_msg,
                digest_hash=digest_hash,
            )
            self.repository.update_briefing(updated_record)

            return DeliveryResult(
                briefing_id=briefing_record.id,
                briefing_date=target_date,
                status=final_status,
                chunks_total=total_chunks,
                chunks_delivered=len(delivered_message_ids),
                message_ids=tuple(delivered_message_ids),
                is_dry_run=False,
                error_message=error_msg,
                warning_message=warning_msg,
                resumed_from_chunk=resume_from_chunk,
            )
        except CorruptDeliveryStateError as exc:
            logger.error("State repository error during delivery: %s", exc)
            return DeliveryResult(
                briefing_id="corrupt-state",
                briefing_date=target_date,
                status="failed",
                chunks_total=total_chunks,
                chunks_delivered=len(delivered_message_ids),
                message_ids=tuple(delivered_message_ids),
                is_dry_run=False,
                error_message=str(exc),
            )
        finally:
            self.repository.release_lock()

