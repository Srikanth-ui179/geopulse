"""Continuous service runner orchestrating scheduled briefings and Telegram Q&A."""

import logging
import os
import signal
import sys
import threading
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from geopulse.bot.qa_service import QAService
from geopulse.config import Settings, get_settings
from geopulse.delivery.client import BaseTelegramClient, TelegramClient
from geopulse.delivery.exceptions import (
    TelegramConflictError,
    TelegramNetworkError,
    TelegramRateLimitError,
    TelegramTimeoutError,
)
from geopulse.delivery.repository import (
    DeliveryRepository,
    InMemoryDeliveryRepository,
    JsonFileDeliveryRepository,
)
from geopulse.delivery.service import TelegramDeliveryService
from geopulse.pipeline import BriefingPipeline, PipelineResult
from geopulse.summarization.client import OllamaClient
from geopulse.summarization.service import EventSummarizer

logger = logging.getLogger(__name__)

MORNING_HOUR = 8  # 08:00 AM
EVENING_HOUR = 20  # 08:00 PM


class ServiceAlreadyRunningError(RuntimeError):
    """Raised when another instance of the Geopulse service runner is already running."""


class ServiceRunner:
    """Manages scheduled morning/evening intelligence briefings and interactive Q&A."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: BaseTelegramClient | None = None,
        pipeline: BriefingPipeline | None = None,
        qa_service: QAService | None = None,
        delivery_repo: DeliveryRepository | None = None,
        mock_model: bool = False,
        persistent: bool = False,
        enable_scheduler: bool = True,
        enable_bot: bool = True,
        poll_timeout: int = 10,
        sleeper: Any = time.sleep,
        lock_instance: bool | None = None,
        lock_path: Path | str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.mock_model = mock_model
        self.persistent = persistent
        self.enable_scheduler = enable_scheduler
        self.enable_bot = enable_bot
        self.poll_timeout = poll_timeout
        self.sleeper = sleeper

        # State repository
        self.delivery_repo = delivery_repo or JsonFileDeliveryRepository(
            self.settings.delivery_state_file
        )

        if lock_instance is None:
            self.lock_instance = not isinstance(self.delivery_repo, InMemoryDeliveryRepository)
        else:
            self.lock_instance = lock_instance

        if lock_path is not None:
            self.lock_path = Path(lock_path)
        else:
            self.lock_path = self.settings.delivery_state_file.parent / "service.lock"
        self._lock_file: Any = None

        # Telegram transport client
        if client is not None:
            self.client = client
        else:
            token, chat_id = self.settings.require_telegram()
            self.client = TelegramClient(token=token, chat_id=chat_id)

        # Model client
        if mock_model:
            from geopulse.summarization.client import MockOllamaClient

            ollama: Any = MockOllamaClient()
        else:
            ollama = OllamaClient(
                base_url=self.settings.ollama_base_url,
                model=self.settings.ollama_model,
                read_timeout=self.settings.ollama_timeout,
            )

        # Pipeline
        if pipeline is not None:
            self.pipeline = pipeline
        else:
            deliv_svc = TelegramDeliveryService(
                client=self.client,
                repository=self.delivery_repo,
                settings=self.settings,
            )
            summarizer = (
                EventSummarizer(client=ollama, settings=self.settings) if mock_model else None
            )
            self.pipeline = BriefingPipeline(
                settings=self.settings,
                delivery_repo=self.delivery_repo,
                delivery_service=deliv_svc,
                summarizer=summarizer,
            )

        # Interactive Q&A service
        self.qa_service = qa_service or QAService(
            client=self.client,
            settings=self.settings,
            delivery_repo=self.delivery_repo,
            ollama_client=ollama,
        )

        self._stop_event = threading.Event()
        self._update_offset: int | None = None
        self._last_scheduled_sent: dict[tuple[date, str], bool] = {}
        self._start_time: datetime = datetime.now(UTC)
        self.last_schedule_status: dict[str, str] = {}
        self.last_bot_handled: int = 0

    @property
    def is_running(self) -> bool:
        return not self._stop_event.is_set()

    def stop(self) -> None:
        """Signal the service runner to shut down gracefully."""
        logger.info("Shutdown signal received. Stopping Geopulse service runner...")
        self._stop_event.set()

    def get_local_now(self) -> datetime:
        """Return the current localized datetime in the configured timezone."""
        return datetime.now(self.settings.tz)

    def should_trigger_briefing(self, now_dt: datetime, edition: str) -> bool:
        """Evaluate whether a briefing edition is due and not yet sent."""
        target_hour = MORNING_HOUR if edition == "morning" else EVENING_HOUR
        target_date = now_dt.date()

        # Check hour threshold
        if now_dt.hour < target_hour:
            return False

        # In-process idempotency check
        if self._last_scheduled_sent.get((target_date, edition)):
            return False

        # Persistent repository idempotency check
        try:
            existing = self.delivery_repo.get_briefing_by_date(target_date, edition=edition)
            if existing and existing.status == "sent":
                self._last_scheduled_sent[(target_date, edition)] = True
                return False
        except Exception as exc:
            logger.warning(
                "Could not check delivery state for %s (%s): %s", target_date, edition, exc
            )

        return True

    def run_scheduled_briefing(
        self, edition: str, target_date: date | None = None
    ) -> PipelineResult:
        """Execute a scheduled morning or evening briefing."""
        run_date = target_date or self.get_local_now().date()
        logger.info(
            "Starting scheduled %s briefing for %s (timezone: %s)...",
            edition.upper(),
            run_date,
            self.settings.timezone,
        )

        try:
            result = self.pipeline.run(
                briefing_date=run_date,
                edition=edition,
                persistent=self.persistent,
                send=True,
                deduplicate_consecutive=True,
            )

            is_success = result.delivery_result and result.delivery_result.status in {
                "sent",
                "skipped_duplicate",
            }
            if is_success:
                logger.info(
                    "Scheduled %s briefing for %s completed successfully (%d events briefed).",
                    edition.upper(),
                    run_date,
                    result.events_count,
                )
                self._last_scheduled_sent[(run_date, edition)] = True
            elif result.error_message:
                logger.error(
                    "Scheduled %s briefing for %s encountered an error: %s",
                    edition.upper(),
                    run_date,
                    result.error_message,
                )

            # Update Q&A cache with fresh events
            if result.events:
                self.qa_service.add_events(list(result.events))

            return result
        except Exception as exc:
            logger.exception("Unexpected exception in scheduled %s briefing: %s", edition, exc)
            return PipelineResult(
                briefing_date=run_date,
                is_preview=False,
                is_persistent=self.persistent,
                articles_count=0,
                new_articles_count=0,
                events_count=0,
                events=(),
                summaries=(),
                digest_chunks=(),
                edition=edition,
                error_message=str(exc),
            )

    def poll_bot_once(self) -> int:
        """Fetch and process pending Telegram updates. Returns count of updates handled."""
        try:
            updates = self.client.get_updates(offset=self._update_offset, timeout=self.poll_timeout)
        except TelegramConflictError as exc:
            logger.warning("Telegram polling conflict (HTTP 409): %s. Backing off for 10s...", exc)
            self.sleeper(10.0)
            return 0
        except TelegramRateLimitError as exc:
            logger.warning("Telegram rate limit (HTTP 429). Sleeping for %.1fs...", exc.retry_after)
            self.sleeper(min(exc.retry_after, 30.0))
            return 0
        except (TelegramNetworkError, TelegramTimeoutError) as exc:
            logger.debug("Network/timeout during polling: %s. Backing off...", exc)
            self.sleeper(2.0)
            return 0
        except Exception as exc:
            logger.error("Unexpected error fetching Telegram updates: %s", exc)
            self.sleeper(2.0)
            return 0

        handled = 0
        for upd in updates:
            upd_id = upd.get("update_id")
            if upd_id is not None:
                self._update_offset = max(self._update_offset or 0, upd_id + 1)

            msg = upd.get("message") or upd.get("edited_message")
            if not isinstance(msg, dict):
                continue

            text = msg.get("text")
            chat = msg.get("chat")
            from_user = msg.get("from")

            if not text or not isinstance(chat, dict):
                continue

            chat_id = chat.get("id")
            user_id = str(from_user.get("id", "")) if isinstance(from_user, dict) else ""

            if chat_id is not None:
                try:
                    self.qa_service.handle_message(
                        text=text,
                        user_id=user_id,
                        chat_id=chat_id,
                    )
                    handled += 1
                    logger.info(
                        "Bot: Handled update #%s from chat %s: '%s'",
                        upd_id,
                        chat_id,
                        text[:40],
                    )
                except Exception as exc:
                    logger.error("Error processing update %s: %s", upd_id, exc)

        if handled > 0:
            logger.info("Bot: Handled %d incoming update(s) successfully.", handled)
        else:
            logger.debug("Bot: 0 pending updates in Telegram queue.")

        return handled

    def check_and_run_schedules(self, now_dt: datetime) -> dict[str, str]:
        """Check morning and evening briefing triggers. Returns edition status map."""
        statuses: dict[str, str] = {}
        for edition in ("morning", "evening"):
            if self.should_trigger_briefing(now_dt, edition):
                res = self.run_scheduled_briefing(edition, now_dt.date())
                outcome = (
                    "sent"
                    if res.delivery_result and res.delivery_result.status == "sent"
                    else "completed"
                )
                statuses[edition] = outcome
            else:
                target_hour = MORNING_HOUR if edition == "morning" else EVENING_HOUR
                if now_dt.hour < target_hour:
                    statuses[edition] = f"not due (scheduled for {target_hour:02d}:00)"
                else:
                    statuses[edition] = f"already sent for {now_dt.date()}"
                logger.info("Scheduler [%s]: %s", edition.upper(), statuses[edition])
        return statuses

    def acquire_instance_lock(self) -> None:
        """Acquire an exclusive OS-level instance lock to prevent duplicate service processes."""
        if not self.lock_instance:
            return

        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            f = open(self.lock_path, "a+b")
            f.seek(0)
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

            f.seek(0)
            f.truncate()
            info = f"pid={os.getpid()}\nstarted={datetime.now(UTC).isoformat()}\n"
            f.write(info.encode("utf-8"))
            f.flush()
            self._lock_file = f
            logger.debug("Acquired exclusive service instance lock at %s", self.lock_path)
        except (OSError, PermissionError) as exc:
            logger.error("Failed to acquire service instance lock at %s: %s", self.lock_path, exc)
            raise ServiceAlreadyRunningError(
                f"Another Geopulse service instance is already running "
                f"(locked at {self.lock_path})."
            ) from exc

    def release_instance_lock(self) -> None:
        """Release the exclusive OS-level instance lock."""
        if self._lock_file is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt

                self._lock_file.seek(0)
                msvcrt.locking(self._lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
        except Exception as exc:
            logger.debug("Error unlocking service lock file: %s", exc)
        finally:
            try:
                self._lock_file.close()
            except Exception:
                pass
            self._lock_file = None

        try:
            if self.lock_path.exists():
                self.lock_path.unlink()
        except Exception:
            pass
        logger.debug("Released service instance lock at %s", self.lock_path)

    def run(self, once: bool = False) -> None:
        """Run the service continuously until stopped."""
        if self.lock_instance:
            self.acquire_instance_lock()

        try:
            # Install signal handlers if run in main thread
            try:
                signal.signal(signal.SIGINT, lambda _s, _f: self.stop())
                signal.signal(signal.SIGTERM, lambda _s, _f: self.stop())
            except (ValueError, AttributeError):
                pass

            logger.info(
                "Geopulse service runner started [Timezone: %s, Scheduler: %s, Bot: %s]",
                self.settings.timezone,
                "Enabled" if self.enable_scheduler else "Disabled",
                "Enabled" if self.enable_bot else "Disabled",
            )

            while not self._stop_event.is_set():
                now_dt = self.get_local_now()

                # 1. Evaluate scheduler triggers
                if self.enable_scheduler:
                    self.last_schedule_status = self.check_and_run_schedules(now_dt)

                # 2. Poll interactive bot messages
                if self.enable_bot:
                    self.last_bot_handled = self.poll_bot_once()

                if once:
                    logger.info("Geopulse single-pass run completed.")
                    break

                # Brief sleep between poll iterations if bot is disabled
                if not self.enable_bot:
                    self.sleeper(15.0)

            logger.info("Geopulse service runner has gracefully exited.")
        finally:
            if self.lock_instance:
                self.release_instance_lock()
