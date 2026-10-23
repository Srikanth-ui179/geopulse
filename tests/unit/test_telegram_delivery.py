"""Unit tests for Phase 5 Telegram delivery service, retry safety, and deduplication."""

from datetime import UTC, date, datetime

import pytest

from geopulse.delivery.client import MockTelegramClient
from geopulse.delivery.exceptions import CorruptDeliveryStateError
from geopulse.delivery.models import BriefingRecord
from geopulse.delivery.repository import InMemoryDeliveryRepository, JsonFileDeliveryRepository
from geopulse.delivery.service import TelegramDeliveryService


def test_delivery_service_successful_delivery():
    client = MockTelegramClient()
    repo = InMemoryDeliveryRepository()
    service = TelegramDeliveryService(client=client, repository=repo)

    chunks = ("<b>Chunk 1</b>", "<b>Chunk 2</b>")
    today = date(2026, 10, 23)

    result = service.deliver_digest(digest_chunks=chunks, briefing_date=today)

    assert result.status == "sent"
    assert result.chunks_total == 2
    assert result.chunks_delivered == 2
    assert len(result.message_ids) == 2
    assert len(client.sent_messages) == 2

    # Repository must reflect sent status
    stored = repo.get_briefing_by_date(today)
    assert stored is not None
    assert stored.status == "sent"
    assert stored.completed_at is not None
    assert stored.chunks_delivered == 2
    assert stored.telegram_message_ids == tuple(result.message_ids)


def test_delivery_service_deduplication_prevents_duplicate_sends():
    client = MockTelegramClient()
    repo = InMemoryDeliveryRepository()
    service = TelegramDeliveryService(client=client, repository=repo)

    chunks = ("<b>Chunk 1</b>",)
    today = date(2026, 10, 23)

    # First send succeeds
    res1 = service.deliver_digest(chunks, briefing_date=today)
    assert res1.status == "sent"
    assert len(client.sent_messages) == 1

    # Second send without force is skipped
    res2 = service.deliver_digest(chunks, briefing_date=today, force=False)
    assert res2.status == "skipped_duplicate"
    assert "already been delivered" in (res2.error_message or "")
    # No additional messages sent to Telegram
    assert len(client.sent_messages) == 1

    # Third send WITH force=True succeeds
    res3 = service.deliver_digest(chunks, briefing_date=today, force=True)
    assert res3.status == "sent"
    assert len(client.sent_messages) == 2


def test_delivery_service_dry_run_mode_does_not_contact_telegram():
    client = MockTelegramClient()
    repo = InMemoryDeliveryRepository()
    service = TelegramDeliveryService(client=client, repository=repo)

    chunks = ("<b>Preview Chunk 1</b>", "<b>Preview Chunk 2</b>")
    today = date(2026, 10, 23)

    result = service.deliver_digest(chunks, briefing_date=today, dry_run=True)

    assert result.status == "sent"
    assert result.is_dry_run is True
    assert result.chunks_total == 2
    assert result.chunks_delivered == 2
    # Zero messages sent to client
    assert len(client.sent_messages) == 0
    # Nothing recorded in repository as 'sent'
    assert repo.get_briefing_by_date(today) is None


def test_delivery_service_recovers_from_transient_rate_limit():
    sleep_calls: list[float] = []

    def mock_sleeper(seconds: float) -> None:
        sleep_calls.append(seconds)

    # Fails first call with 429, succeeds on retry
    client = MockTelegramClient(fail_first_n_calls=1, should_fail_rate_limit=2.5)
    repo = InMemoryDeliveryRepository()
    service = TelegramDeliveryService(
        client=client,
        repository=repo,
        sleeper=mock_sleeper,
    )

    chunks = ("<b>Rate limit test</b>",)
    result = service.deliver_digest(chunks, briefing_date=date(2026, 10, 23))

    assert result.status == "sent"
    assert len(client.sent_messages) == 1
    assert sleep_calls == [2.5]


def test_delivery_service_recovers_from_transient_network_error():
    sleep_calls: list[float] = []
    client = MockTelegramClient(fail_first_n_calls=2, should_fail_network=True)
    repo = InMemoryDeliveryRepository()
    service = TelegramDeliveryService(
        client=client,
        repository=repo,
        backoff_factor=1.5,
        sleeper=lambda s: sleep_calls.append(s),
    )

    chunks = ("<b>Network error test</b>",)
    result = service.deliver_digest(chunks, briefing_date=date(2026, 10, 23))

    assert result.status == "sent"
    assert len(client.sent_messages) == 1
    assert len(sleep_calls) == 2


def test_delivery_service_halts_on_ambiguous_read_timeout_to_prevent_duplicates():
    client = MockTelegramClient(fail_first_n_calls=5, should_timeout_ambiguous=True)
    repo = InMemoryDeliveryRepository()
    service = TelegramDeliveryService(client=client, repository=repo)

    chunks = ("<b>Ambiguous timeout test</b>",)
    today = date(2026, 10, 23)
    result = service.deliver_digest(chunks, briefing_date=today)

    assert result.status == "ambiguous"
    assert "Ambiguous timeout" in (result.error_message or "")
    assert "Halted to prevent duplicate sends" in (result.error_message or "")
    # Call count was halted immediately without blind retry
    assert client.call_count == 1


def test_delivery_service_permanent_error_fails_immediately_without_retrying():
    client = MockTelegramClient(should_fail_permanent=(400, "Bad Request: chat not found"))
    repo = InMemoryDeliveryRepository()
    service = TelegramDeliveryService(client=client, repository=repo)

    chunks = ("<b>Bad request test</b>",)
    today = date(2026, 10, 23)
    result = service.deliver_digest(chunks, briefing_date=today)

    assert result.status == "failed"
    assert "400" in (result.error_message or "")
    assert "chat not found" in (result.error_message or "")
    # Exactly one call made (no retry loop on 4xx)
    assert client.call_count == 1
    stored = repo.get_briefing_by_date(today)
    assert stored is not None
    assert stored.status == "failed"


def test_delivery_service_partial_delivery_state():
    # Succeeds on chunk 1, fails permanently on chunk 2
    class PartialFailClient(MockTelegramClient):
        def send_message(self, text, **kwargs):
            if "Chunk 2" in text:
                from geopulse.delivery.exceptions import TelegramPermanentError
                raise TelegramPermanentError(400, "Parse error in Chunk 2")
            return super().send_message(text, **kwargs)

    client = PartialFailClient()
    repo = InMemoryDeliveryRepository()
    service = TelegramDeliveryService(client=client, repository=repo)

    chunks = ("<b>Chunk 1</b>", "<b>Chunk 2</b>")
    today = date(2026, 10, 23)
    result = service.deliver_digest(chunks, briefing_date=today)

    assert result.status == "partial"
    assert result.chunks_delivered == 1
    assert result.chunks_total == 2
    stored = repo.get_briefing_by_date(today)
    assert stored is not None
    assert stored.status == "partial"
    assert stored.chunks_delivered == 1


def test_retry_after_partial_delivery_safely_resumes_without_duplicating_chunks():
    """Verify that a subsequent invocation after partial delivery resumes safely.

    Chunk 1 was already accepted (id: 1001), so resumption sends ONLY Chunk 2,
    avoiding duplicate delivery of Chunk 1 to the user.
    """
    import hashlib

    client = MockTelegramClient()
    repo = InMemoryDeliveryRepository()
    service = TelegramDeliveryService(client=client, repository=repo)

    chunks = ("<b>Chunk 1</b>", "<b>Chunk 2</b>")
    combined_hash = hashlib.sha256("".join(chunks).encode("utf-8")).hexdigest()
    today = date(2026, 10, 23)

    # Simulate existing partial delivery state in repository
    repo.update_briefing(
        BriefingRecord(
            id="prev-partial-run",
            briefing_date=today,
            status="partial",
            started_at=datetime.now(UTC),
            telegram_message_count=2,
            chunks_delivered=1,
            telegram_message_ids=(1001,),
            digest_hash=combined_hash,
        )
    )

    result = service.deliver_digest(chunks, briefing_date=today)

    # Resumed run succeeds and delivers remaining chunk
    assert result.status == "sent"
    assert result.chunks_delivered == 2
    assert result.resumed_from_chunk == 1
    # Only Chunk 2 was sent (1 call made, NOT 2)
    assert len(client.sent_messages) == 1
    assert "Chunk 2" in client.sent_messages[0]["text"]
    assert result.message_ids == (1001, client.sent_messages[0]["message_id"])


def test_partial_delivery_with_changed_digest_requires_force():
    """If digest content changed since partial delivery, refuse to resume mismatched chunks."""
    client = MockTelegramClient()
    repo = InMemoryDeliveryRepository()
    service = TelegramDeliveryService(client=client, repository=repo)

    today = date(2026, 10, 23)
    repo.update_briefing(
        BriefingRecord(
            id="prev-partial-run",
            briefing_date=today,
            status="partial",
            started_at=datetime.now(UTC),
            telegram_message_count=2,
            chunks_delivered=1,
            telegram_message_ids=(1001,),
            digest_hash="old-content-hash",
        )
    )

    new_chunks = ("<b>Brand New Chunk 1</b>", "<b>Brand New Chunk 2</b>")
    # Without force: stops with informative error
    result = service.deliver_digest(new_chunks, briefing_date=today, force=False)
    assert result.status == "partial"
    assert "Digest content has changed" in (result.error_message or "")
    assert len(client.sent_messages) == 0

    # With force: restarts from scratch with new content
    result_forced = service.deliver_digest(new_chunks, briefing_date=today, force=True)
    assert result_forced.status == "sent"
    assert len(client.sent_messages) == 2


def test_ambiguous_timeout_marks_ambiguous_and_blocks_unforced_retry():
    """An ambiguous timeout marks status='ambiguous' and blocks subsequent unforced delivery."""
    client = MockTelegramClient(fail_first_n_calls=5, should_timeout_ambiguous=True)
    repo = InMemoryDeliveryRepository()
    service = TelegramDeliveryService(client=client, repository=repo)

    chunks = ("<b>Chunk 1</b>",)
    today = date(2026, 10, 23)

    # First attempt: encounters ambiguous timeout
    res1 = service.deliver_digest(chunks, briefing_date=today)
    assert res1.status == "ambiguous"
    assert client.call_count == 1
    stored = repo.get_briefing_by_date(today)
    assert stored is not None
    assert stored.status == "ambiguous"

    # Second invocation without force: immediately halts without network calls
    res2 = service.deliver_digest(chunks, briefing_date=today, force=False)
    assert res2.status == "ambiguous"
    assert "Previous delivery halted due to an ambiguous timeout" in (res2.error_message or "")
    assert client.call_count == 1  # No new calls

    # Third invocation WITH force: overrides ambiguous block and retries from scratch
    client.fail_first_n_calls = 0  # Allow success
    client.should_timeout_ambiguous = False
    res3 = service.deliver_digest(chunks, briefing_date=today, force=True)
    assert res3.status == "sent"
    assert len(client.sent_messages) == 1
    assert res3.warning_message is not None
    assert "duplicated" in res3.warning_message
    assert "ambiguous timeout" in res3.warning_message


def test_json_file_repository_persists_across_instances(tmp_path):
    """Verify that JsonFileDeliveryRepository retains state across separate service instances."""
    state_file = tmp_path / "delivery_state.json"
    repo1 = JsonFileDeliveryRepository(file_path=state_file)
    client1 = MockTelegramClient()
    service1 = TelegramDeliveryService(client=client1, repository=repo1)

    chunks = ("<b>Persistent Chunk 1</b>",)
    today = date(2026, 10, 23)

    # First service instance delivers
    res1 = service1.deliver_digest(chunks, briefing_date=today)
    assert res1.status == "sent"
    assert state_file.is_file()

    # Second service instance (simulating process restart)
    repo2 = JsonFileDeliveryRepository(file_path=state_file)
    client2 = MockTelegramClient()
    service2 = TelegramDeliveryService(client=client2, repository=repo2)

    res2 = service2.deliver_digest(chunks, briefing_date=today, force=False)
    assert res2.status == "skipped_duplicate"
    assert len(client2.sent_messages) == 0


def test_interrupted_processing_state_with_delivered_chunks_resumes_cleanly():
    """If a process died while in 'processing' status, but chunk 1 was saved, resume cleanly."""
    import hashlib

    client = MockTelegramClient()
    repo = InMemoryDeliveryRepository()
    service = TelegramDeliveryService(client=client, repository=repo)

    chunks = ("<b>Chunk 1</b>", "<b>Chunk 2</b>")
    combined_hash = hashlib.sha256("".join(chunks).encode("utf-8")).hexdigest()
    today = date(2026, 10, 23)

    # State left in 'processing' with chunk 1 already confirmed and persisted
    repo.update_briefing(
        BriefingRecord(
            id="crashed-processing-run",
            briefing_date=today,
            status="processing",
            started_at=datetime.now(UTC),
            telegram_message_count=2,
            chunks_delivered=1,
            telegram_message_ids=(1001,),
            digest_hash=combined_hash,
        )
    )

    result = service.deliver_digest(chunks, briefing_date=today)
    assert result.status == "sent"
    assert result.chunks_delivered == 2
    assert result.resumed_from_chunk == 1
    # Only chunk 2 was sent to client (avoiding duplicate delivery of chunk 1)
    assert len(client.sent_messages) == 1
    assert "Chunk 2" in client.sent_messages[0]["text"]


def test_concurrent_delivery_processes_blocked_by_lock(tmp_path):
    """Two delivery processes sharing the same repository cannot run concurrently."""
    state_file = tmp_path / "concurrent_state.json"
    repo1 = JsonFileDeliveryRepository(file_path=state_file)
    repo2 = JsonFileDeliveryRepository(file_path=state_file)

    chunks = ("<b>Chunk 1</b>",)
    today = date(2026, 10, 23)

    # Manually hold lock on repo1
    assert repo1.acquire_lock() is True

    # Process 2 attempts to deliver while lock is held
    client2 = MockTelegramClient()
    service2 = TelegramDeliveryService(client=client2, repository=repo2)
    res2 = service2.deliver_digest(chunks, briefing_date=today)

    assert res2.status == "failed"
    assert "delivery lock held" in (res2.error_message or "")
    assert len(client2.sent_messages) == 0

    # Release lock on repo1
    repo1.release_lock()

    # Now process 2 succeeds
    res3 = service2.deliver_digest(chunks, briefing_date=today)
    assert res3.status == "sent"
    assert len(client2.sent_messages) == 1


def test_interrupted_processing_state_with_changed_digest_requires_force():
    """If an interrupted processing run had content that changed, require force to restart."""
    client = MockTelegramClient()
    repo = InMemoryDeliveryRepository()
    service = TelegramDeliveryService(client=client, repository=repo)

    today = date(2026, 10, 23)
    repo.update_briefing(
        BriefingRecord(
            id="processing-run-old-content",
            briefing_date=today,
            status="processing",
            started_at=datetime.now(UTC),
            telegram_message_count=2,
            chunks_delivered=1,
            telegram_message_ids=(1001,),
            digest_hash="old-hash-abc",
        )
    )

    new_chunks = ("<b>New Chunk A</b>", "<b>New Chunk B</b>")
    # Without force: stops with informative error
    result = service.deliver_digest(new_chunks, briefing_date=today, force=False)
    assert result.status == "partial"
    assert "Digest content has changed" in (result.error_message or "")
    assert len(client.sent_messages) == 0

    # With force: restarts from chunk 1
    result_forced = service.deliver_digest(new_chunks, briefing_date=today, force=True)
    assert result_forced.status == "sent"
    assert len(client.sent_messages) == 2


def test_json_file_repository_fails_closed_on_corrupt_state(tmp_path):
    """Missing state file is handled safely, but corrupt/incomplete files fail closed."""
    state_file = tmp_path / "corrupt_state.json"
    repo = JsonFileDeliveryRepository(file_path=state_file)
    today = date(2026, 10, 23)

    # 1. Non-existent file: handled safely (returns None, meaning no prior briefing)
    assert repo.get_briefing_by_date(today) is None

    # 2. Corrupt non-JSON file content: fails closed with CorruptDeliveryStateError
    state_file.write_text("{{malformed json content::", encoding="utf-8")
    with pytest.raises(CorruptDeliveryStateError, match="invalid JSON"):
        repo.get_briefing_by_date(today)

    # 3. Empty file: fails closed with CorruptDeliveryStateError
    state_file.write_text("   \n", encoding="utf-8")
    with pytest.raises(CorruptDeliveryStateError, match="empty or truncated"):
        repo.get_briefing_by_date(today)

    # 4. Valid JSON that is not an object/dict: fails closed
    state_file.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(CorruptDeliveryStateError, match="expected a JSON object"):
        repo.get_briefing_by_date(today)

    # 5. Valid JSON with malformed/missing row values: fails closed
    state_file.write_text('{"2026-10-23": {"started_at": "invalid-date"}}', encoding="utf-8")
    with pytest.raises(CorruptDeliveryStateError, match="Corrupted record for date"):
        repo.get_briefing_by_date(today)


def test_delivery_service_fails_closed_on_corrupt_state_file(tmp_path):
    """Delivery service fails closed and sends 0 messages when state file is corrupt."""
    state_file = tmp_path / "corrupt_state.json"
    state_file.write_text("{{corrupted-json}}", encoding="utf-8")

    client = MockTelegramClient()
    repo = JsonFileDeliveryRepository(file_path=state_file)
    service = TelegramDeliveryService(client=client, repository=repo)

    today = date(2026, 10, 23)
    chunks = ("<b>Chunk 1</b>", "<b>Chunk 2</b>")

    result = service.deliver_digest(digest_chunks=chunks, briefing_date=today)

    assert result.status == "failed"
    assert result.chunks_delivered == 0
    assert "invalid JSON" in (result.error_message or "")
    # Absolutely zero Telegram messages must be sent
    assert len(client.sent_messages) == 0





