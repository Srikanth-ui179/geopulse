"""Repository layer for persisting briefing delivery state and deduplication."""

import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from geopulse.config import Settings
from geopulse.delivery.exceptions import CorruptDeliveryStateError
from geopulse.delivery.models import BriefingRecord

logger = logging.getLogger(__name__)


class DeliveryRepository(ABC):
    """Abstract interface for briefing delivery lifecycle tracking."""

    @abstractmethod
    def get_briefing_by_date(self, briefing_date: date) -> BriefingRecord | None:
        """Retrieve existing briefing record for a specific date, if any."""
        ...

    @abstractmethod
    def create_briefing(
        self,
        briefing_date: date,
        total_chunks: int,
        digest_hash: str | None = None,
    ) -> BriefingRecord:
        """Create a new briefing run record in 'processing' status."""
        ...

    @abstractmethod
    def update_briefing(self, record: BriefingRecord) -> BriefingRecord:
        """Update existing briefing status, chunk metrics, and error state."""
        ...

    def record_briefing_events(
        self,
        briefing_id: str,
        results: list[Any],
        article_url_to_id: dict[str, str] | None = None,
    ) -> tuple[int, int]:
        """Persist ranked briefing summaries and linked article evidence.

        Returns:
            tuple[int, int]: (briefing_events_count, evidence_records_count)
        """
        return 0, 0

    def acquire_lock(self) -> bool:
        """Acquire delivery lock to prevent concurrent executions."""
        return True

    def release_lock(self) -> None:
        """Release delivery lock."""
        pass


class InMemoryDeliveryRepository(DeliveryRepository):
    """In-memory delivery repository for testing and dry-run execution."""

    def __init__(self) -> None:
        self._records: dict[date, BriefingRecord] = {}
        self._briefing_events: dict[str, list[dict[str, Any]]] = {}
        self._briefing_evidence: dict[str, list[dict[str, Any]]] = {}
        self._is_locked = False

    def record_briefing_events(
        self,
        briefing_id: str,
        results: list[Any],
        article_url_to_id: dict[str, str] | None = None,
    ) -> tuple[int, int]:
        """Persist ranked briefing summaries and linked article evidence."""
        event_rows = []
        evidence_rows = []
        url_to_id = dict(article_url_to_id or {})
        for rank, res_item in enumerate(results, start=1):
            if hasattr(res_item, "event"):
                ev_id = getattr(res_item.event, "id", None)
            else:
                ev_id = getattr(res_item, "event_id", None)
            if not ev_id:
                continue

            summary_text = ""
            why_it_matters = None
            headline_text = None
            key_details_data = None
            conf_qual = None
            model_used_text = None

            if hasattr(res_item, "summary") and res_item.summary is not None:
                if isinstance(res_item.summary, str):
                    summary_text = res_item.summary
                    why_it_matters = getattr(res_item, "significance", None)
                else:
                    s = res_item.summary
                    summary_text = getattr(s, "what_happened", str(s))
                    why_it_matters = getattr(
                        s,
                        "significance",
                        getattr(res_item, "significance", None),
                    )
                    headline_text = getattr(s, "headline", None)
                    if hasattr(s, "key_details"):
                        if hasattr(s.key_details, "model_dump"):
                            key_details_data = s.key_details.model_dump()
                        elif isinstance(s.key_details, dict):
                            key_details_data = dict(s.key_details)
                    if key_details_data and "actors" in key_details_data:
                        key_details_data["actors"] = list(key_details_data["actors"])
                    conf_qual = getattr(s, "confidence_qualification", None)
                    model_used_text = getattr(s, "model_used", None)
            elif hasattr(res_item, "headline"):
                headline_text = getattr(res_item, "headline", None)
                summary_text = getattr(res_item, "headline", "")

            row: dict[str, Any] = {
                "briefing_id": briefing_id,
                "event_id": ev_id,
                "rank": rank,
                "summary": summary_text,
                "why_it_matters": why_it_matters,
            }
            if headline_text:
                row["headline"] = headline_text
            if key_details_data is not None:
                row["key_details"] = key_details_data
            if conf_qual:
                row["confidence_qualification"] = conf_qual
            if model_used_text:
                row["model_used"] = model_used_text

            event_rows.append(row)
            for a in getattr(res_item.event, "articles", ()):
                art_id = url_to_id.get(a.canonical_url)
                if art_id:
                    evidence_rows.append(
                        {"briefing_id": briefing_id, "event_id": ev_id, "article_id": art_id}
                    )
        self._briefing_events[briefing_id] = event_rows
        self._briefing_evidence[briefing_id] = evidence_rows
        return len(event_rows), len(evidence_rows)

    def acquire_lock(self) -> bool:
        if self._is_locked:
            return False
        self._is_locked = True
        return True

    def release_lock(self) -> None:
        self._is_locked = False


    def get_briefing_by_date(self, briefing_date: date) -> BriefingRecord | None:
        return self._records.get(briefing_date)

    def create_briefing(
        self,
        briefing_date: date,
        total_chunks: int,
        digest_hash: str | None = None,
    ) -> BriefingRecord:
        record_id = str(uuid.uuid4())
        rec = BriefingRecord(
            id=record_id,
            briefing_date=briefing_date,
            status="processing",
            started_at=datetime.now(UTC),
            telegram_message_count=total_chunks,
            digest_hash=digest_hash,
        )
        self._records[briefing_date] = rec
        return rec

    def update_briefing(self, record: BriefingRecord) -> BriefingRecord:
        self._records[record.briefing_date] = record
        return record

    def clear(self) -> None:
        self._records.clear()


class JsonFileDeliveryRepository(DeliveryRepository):
    """File-backed delivery repository persisting state to JSON across process restarts."""

    def __init__(
        self,
        file_path: Path | str | None = None,
    ) -> None:
        if file_path is None:
            from geopulse.config import get_settings

            try:
                cfg = get_settings()
                self.file_path = cfg.delivery_state_file
            except Exception:
                self.file_path = Path(".geopulse/delivery_state.json")
        else:
            self.file_path = Path(file_path)

        self.lock_path = self.file_path.with_suffix(".lock")
        self._lock_fd: int | None = None
        self._ensure_parent_dir()

    def acquire_lock(self) -> bool:
        """Acquire an exclusive OS-backed kernel lock to serialize delivery processes."""
        self._ensure_parent_dir()
        try:
            fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR)
        except OSError as exc:
            logger.warning("Could not open delivery lock file %s: %s", self.lock_path, exc)
            return False

        # Apply OS-level non-blocking lock to open handle
        try:
            if os.name == "nt":
                import msvcrt

                # Lock the first byte non-blocking
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                # Exclusive non-blocking lock
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            # Another process currently holds the OS lock
            os.close(fd)
            return False

        # Lock acquired: write owner metadata
        try:
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            meta = f"pid={os.getpid()}\nstarted_at={datetime.now(UTC).isoformat()}\n"
            os.write(fd, meta.encode("utf-8"))
        except OSError:
            pass

        self._lock_fd = fd
        return True

    def release_lock(self) -> None:
        """Release the OS-backed lock and close handle."""
        if self._lock_fd is not None:
            try:
                if os.name == "nt":
                    import msvcrt

                    os.lseek(self._lock_fd, 0, os.SEEK_SET)
                    msvcrt.locking(self._lock_fd, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            except OSError:
                pass

            try:
                os.close(self._lock_fd)
            except OSError:
                pass
            self._lock_fd = None

    def _ensure_parent_dir(self) -> None:
        if self.file_path.parent:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_data(self) -> dict[str, Any]:
        if not self.file_path.is_file():
            return {}
        try:
            content = self.file_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise CorruptDeliveryStateError(
                f"Delivery state file '{self.file_path}' cannot be read: {exc}. "
                "Failing closed to prevent accidental duplicate delivery. "
                "Operator intervention required."
            ) from exc

        if not content.strip():
            raise CorruptDeliveryStateError(
                f"Delivery state file '{self.file_path}' exists but is empty or truncated. "
                "Failing closed to prevent accidental duplicate delivery. "
                "Operator intervention required."
            )

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise CorruptDeliveryStateError(
                f"Delivery state file '{self.file_path}' contains invalid JSON: {exc}. "
                "Failing closed to prevent accidental duplicate delivery. "
                "Operator intervention required."
            ) from exc

        if not isinstance(parsed, dict):
            raise CorruptDeliveryStateError(
                f"Delivery state file '{self.file_path}' is corrupted: expected a JSON object, "
                f"found {type(parsed).__name__}. "
                "Failing closed to prevent accidental duplicate delivery. "
                "Operator intervention required."
            )

        return parsed

    def _save_data(self, data: dict[str, Any]) -> None:
        self._ensure_parent_dir()
        temp_file = self.file_path.with_suffix(f".tmp.{os.getpid()}.{uuid.uuid4().hex[:6]}")
        try:
            temp_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            temp_file.replace(self.file_path)
        except Exception as exc:
            logger.error("Failed to write delivery state file %s: %s", self.file_path, exc)
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass

    def get_briefing_by_date(self, briefing_date: date) -> BriefingRecord | None:
        data = self._load_data()
        key = briefing_date.isoformat()
        if key not in data:
            return None
        row = data[key]
        if not isinstance(row, dict):
            raise CorruptDeliveryStateError(
                f"Corrupted record for date '{key}' in state file '{self.file_path}': "
                f"expected an object, found {type(row).__name__}. "
                "Failing closed to prevent accidental duplicate delivery. "
                "Operator intervention required."
            )

        try:
            started_dt = (
                datetime.fromisoformat(row["started_at"])
                if row.get("started_at")
                else datetime.now(UTC)
            )
            completed_dt = (
                datetime.fromisoformat(row["completed_at"])
                if row.get("completed_at")
                else None
            )

            return BriefingRecord(
                id=str(row.get("id", uuid.uuid4())),
                briefing_date=briefing_date,
                status=row.get("status", "pending"),
                started_at=started_dt,
                completed_at=completed_dt,
                telegram_message_count=row.get("telegram_message_count", 0),
                error_message=row.get("error_message"),
                chunks_delivered=row.get("chunks_delivered", 0),
                telegram_message_ids=tuple(row.get("telegram_message_ids", ())),
                digest_hash=row.get("digest_hash"),
            )
        except Exception as exc:
            raise CorruptDeliveryStateError(
                f"Corrupted record for date '{key}' in state file '{self.file_path}': {exc}. "
                "Failing closed to prevent accidental duplicate delivery. "
                "Operator intervention required."
            ) from exc

    def create_briefing(
        self,
        briefing_date: date,
        total_chunks: int,
        digest_hash: str | None = None,
    ) -> BriefingRecord:
        data = self._load_data()
        record_id = str(uuid.uuid4())
        rec = BriefingRecord(
            id=record_id,
            briefing_date=briefing_date,
            status="processing",
            started_at=datetime.now(UTC),
            telegram_message_count=total_chunks,
            chunks_delivered=0,
            telegram_message_ids=(),
            digest_hash=digest_hash,
        )
        data[briefing_date.isoformat()] = {
            "id": rec.id,
            "briefing_date": rec.briefing_date.isoformat(),
            "status": rec.status,
            "started_at": rec.started_at.isoformat(),
            "completed_at": None,
            "telegram_message_count": rec.telegram_message_count,
            "chunks_delivered": 0,
            "telegram_message_ids": [],
            "digest_hash": rec.digest_hash,
            "error_message": None,
        }
        self._save_data(data)
        return rec

    def update_briefing(self, record: BriefingRecord) -> BriefingRecord:
        data = self._load_data()
        data[record.briefing_date.isoformat()] = {
            "id": record.id,
            "briefing_date": record.briefing_date.isoformat(),
            "status": record.status,
            "started_at": record.started_at.isoformat(),
            "completed_at": record.completed_at.isoformat() if record.completed_at else None,
            "telegram_message_count": record.telegram_message_count,
            "chunks_delivered": record.chunks_delivered,
            "telegram_message_ids": list(record.telegram_message_ids),
            "digest_hash": record.digest_hash,
            "error_message": record.error_message,
        }
        self._save_data(data)
        return record

    def clear(self) -> None:
        self.release_lock()
        if self.file_path.exists():
            try:
                self.file_path.unlink()
            except OSError:
                pass
        if self.lock_path.exists():
            try:
                self.lock_path.unlink()
            except OSError:
                pass



class SupabaseDeliveryRepository(DeliveryRepository):
    """PostgreSQL / Supabase persistence for briefing delivery state and audit logs."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: Any = None,
        chat_id: str | None = None,
        lock_path: Path | None = None,
    ) -> None:
        if client is not None:
            self.client = client
            self._chat_id = chat_id
        else:
            from geopulse.config import get_settings

            cfg = settings or get_settings()
            url, key = cfg.require_supabase()
            from supabase import Client, create_client

            self.client: Client = create_client(url, key)
            self._chat_id = chat_id or cfg.telegram_chat_id

        if lock_path is None:
            self.lock_path = Path(".geopulse/delivery_state.lock")
        else:
            self.lock_path = Path(lock_path)

        self._lock_fd: int | None = None
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)

    def acquire_lock(self) -> bool:
        """Acquire kernel lock to prevent concurrent briefing delivery processes."""
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR)
        except OSError as exc:
            logger.warning("Could not open lock file %s: %s", self.lock_path, exc)
            return False

        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError):
            try:
                os.close(fd)
            except OSError:
                pass
            return False

        # Lock acquired: record process metadata
        try:
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            meta = f"pid={os.getpid()}\nstarted_at={datetime.now(UTC).isoformat()}\n"
            os.write(fd, meta.encode("utf-8"))
        except OSError:
            pass

        self._lock_fd = fd
        return True

    def release_lock(self) -> None:
        """Release delivery lock safely."""
        if self._lock_fd is not None:
            try:
                if os.name == "nt":
                    import msvcrt

                    try:
                        os.lseek(self._lock_fd, 0, os.SEEK_SET)
                        msvcrt.locking(self._lock_fd, msvcrt.LK_UNLCK, 1)
                    except OSError:
                        pass
                else:
                    import fcntl

                    try:
                        fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                    except OSError:
                        pass
                os.close(self._lock_fd)
            except OSError:
                pass
            finally:
                self._lock_fd = None

    def get_briefing_by_date(self, briefing_date: date) -> BriefingRecord | None:
        res = (
            self.client.table("briefings")
            .select("*")
            .eq("briefing_date", briefing_date.isoformat())
            .limit(1)
            .execute()
        )
        if not res.data:
            return None

        row = res.data[0]
        started_dt = (
            datetime.fromisoformat(row["started_at"])
            if row.get("started_at")
            else datetime.now(UTC)
        )
        completed_dt = (
            datetime.fromisoformat(row["completed_at"]) if row.get("completed_at") else None
        )
        raw_ids = row.get("telegram_message_ids") or []
        msg_ids = tuple(int(x) for x in raw_ids if str(x).isdigit())

        return BriefingRecord(
            id=str(row["id"]),
            briefing_date=briefing_date,
            status=row.get("status", "pending"),
            started_at=started_dt,
            completed_at=completed_dt,
            telegram_message_count=row.get("telegram_message_count", 0),
            error_message=row.get("error_message"),
            chunks_delivered=row.get("chunks_delivered", 0),
            telegram_message_ids=msg_ids,
            digest_hash=row.get("digest_hash"),
        )

    def create_briefing(
        self,
        briefing_date: date,
        total_chunks: int,
        digest_hash: str | None = None,
    ) -> BriefingRecord:
        payload: dict[str, Any] = {
            "briefing_date": briefing_date.isoformat(),
            "status": "processing",
            "started_at": datetime.now(UTC).isoformat(),
            "telegram_message_count": total_chunks,
            "chunks_delivered": 0,
            "telegram_message_ids": [],
        }
        if digest_hash:
            payload["digest_hash"] = digest_hash
        if self._chat_id:
            payload["telegram_chat_id"] = self._chat_id

        res = self.client.table("briefings").upsert(
            payload,
            on_conflict="briefing_date",
        ).execute()

        row = res.data[0]
        return BriefingRecord(
            id=str(row["id"]),
            briefing_date=briefing_date,
            status="processing",
            started_at=datetime.fromisoformat(row["started_at"]),
            telegram_message_count=total_chunks,
            digest_hash=digest_hash,
            chunks_delivered=0,
            telegram_message_ids=(),
        )

    def update_briefing(self, record: BriefingRecord) -> BriefingRecord:
        payload: dict[str, Any] = {
            "status": record.status,
            "completed_at": record.completed_at.isoformat() if record.completed_at else None,
            "telegram_message_count": record.telegram_message_count,
            "error_message": record.error_message,
            "chunks_delivered": record.chunks_delivered,
            "telegram_message_ids": list(record.telegram_message_ids),
        }
        if record.digest_hash:
            payload["digest_hash"] = record.digest_hash
        self.client.table("briefings").update(payload).eq("id", record.id).execute()
        return record

    def record_briefing_events(
        self,
        briefing_id: str,
        results: list[Any],
        article_url_to_id: dict[str, str] | None = None,
    ) -> tuple[int, int]:
        """Persist ranked briefing summaries and linked article evidence."""
        if not results:
            return 0, 0

        url_to_id = dict(article_url_to_id or {})
        if not url_to_id:
            all_urls = [
                a.canonical_url
                for r in results
                if hasattr(r, "event")
                for a in r.event.articles
            ]
            if all_urls:
                res = (
                    self.client.table("articles")
                    .select("id, canonical_url")
                    .in_("canonical_url", all_urls)
                    .execute()
                )
                for row in res.data or []:
                    if "canonical_url" in row and "id" in row:
                        url_to_id[str(row["canonical_url"])] = str(row["id"])

        event_rows: list[dict[str, Any]] = []
        evidence_rows: list[dict[str, Any]] = []

        for rank, res_item in enumerate(results, start=1):
            if hasattr(res_item, "event"):
                ev_id = getattr(res_item.event, "id", None)
            else:
                ev_id = getattr(res_item, "event_id", None)
            if not ev_id:
                continue

            summary_text = ""
            why_it_matters = None
            headline_text = None
            key_details_data = None
            conf_qual = None
            model_used_text = None

            if hasattr(res_item, "summary") and res_item.summary is not None:
                if isinstance(res_item.summary, str):
                    summary_text = res_item.summary
                    why_it_matters = getattr(res_item, "significance", None)
                else:
                    s = res_item.summary
                    summary_text = getattr(
                        s, "what_happened", str(s)
                    )
                    why_it_matters = getattr(
                        s,
                        "significance",
                        getattr(res_item, "significance", None),
                    )
                    headline_text = getattr(s, "headline", None)
                    if hasattr(s, "key_details"):
                        if hasattr(s.key_details, "model_dump"):
                            key_details_data = s.key_details.model_dump()
                        elif isinstance(s.key_details, dict):
                            key_details_data = dict(s.key_details)
                    if key_details_data and "actors" in key_details_data:
                        key_details_data["actors"] = list(key_details_data["actors"])
                    conf_qual = getattr(s, "confidence_qualification", None)
                    model_used_text = getattr(s, "model_used", None)
            elif hasattr(res_item, "headline"):
                headline_text = getattr(res_item, "headline", None)
                summary_text = getattr(res_item, "headline", "")

            event_row: dict[str, Any] = {
                "briefing_id": briefing_id,
                "event_id": ev_id,
                "rank": rank,
                "summary": summary_text,
                "why_it_matters": why_it_matters,
            }
            if headline_text:
                event_row["headline"] = headline_text
            if key_details_data is not None:
                event_row["key_details"] = key_details_data
            if conf_qual:
                event_row["confidence_qualification"] = conf_qual
            if model_used_text:
                event_row["model_used"] = model_used_text

            event_rows.append(event_row)

            for a in getattr(res_item.event, "articles", ()):
                art_id = url_to_id.get(a.canonical_url)
                if art_id:
                    evidence_rows.append(
                        {
                            "briefing_id": briefing_id,
                            "event_id": ev_id,
                            "article_id": art_id,
                        }
                    )

        if event_rows:
            self.client.table("briefing_events").upsert(
                event_rows,
                on_conflict="briefing_id,event_id",
            ).execute()

        if evidence_rows:
            self.client.table("briefing_event_evidence").upsert(
                evidence_rows,
                on_conflict="briefing_id,event_id,article_id",
                ignore_duplicates=True,
            ).execute()

        return len(event_rows), len(evidence_rows)

