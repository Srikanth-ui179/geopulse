"""Unit tests for Geopulse scheduled morning and evening briefings."""

from datetime import UTC, date, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from geopulse.classification.models import TopicCategory
from geopulse.clustering.models import EventCluster
from geopulse.config import Settings
from geopulse.delivery.client import MockTelegramClient
from geopulse.delivery.models import BriefingRecord
from geopulse.delivery.repository import InMemoryDeliveryRepository
from geopulse.delivery.service import TelegramDeliveryService
from geopulse.ingestion.models import ArticleRecord
from geopulse.pipeline import BriefingPipeline, compute_event_hash
from geopulse.service.runner import ServiceAlreadyRunningError, ServiceRunner
from geopulse.summarization.service import EventSummarizer


def _create_sample_article(
    title: str = "Missile Strike at Coastal Radar Base",
    url: str = "https://defensenews.com/strike-1",
    published_hour: int = 4,
) -> ArticleRecord:
    dt = datetime(2026, 10, 24, published_hour, 0, tzinfo=UTC)
    return ArticleRecord(
        source_name="Defense News",
        title=title,
        url=url,
        canonical_url=url,
        published_at=dt,
        fetched_at=dt,
        content_hash="hash-123",
        relevance_status="relevant",
        topic=TopicCategory.ARMED_CONFLICT,
    )


def _create_sample_event(
    title: str = "Missile Strike at Coastal Radar Base",
    articles: list[ArticleRecord] | None = None,
    is_analysis: bool = False,
) -> EventCluster:
    arts = articles or [_create_sample_article(title=title)]
    first_seen = arts[0].published_at or datetime.now(UTC)
    return EventCluster(
        id="evt-test-1",
        title=title,
        category=TopicCategory.ARMED_CONFLICT,
        region="Middle East",
        first_seen_at=first_seen,
        last_seen_at=first_seen,
        published_at=first_seen,
        is_analysis=is_analysis,
        confidence_label="corroborated",
        articles=tuple(arts),
    )


def test_timezone_resolution_asia_kolkata():
    """Verify that Asia/Kolkata timezone is properly resolved and maintains UTC+05:30 offset."""
    settings = Settings(timezone="Asia/Kolkata")
    tz = settings.tz
    now_local = datetime.now(tz)
    offset = now_local.utcoffset()
    assert offset == timedelta(hours=5, minutes=30)


def test_should_trigger_briefing_morning_at_8am():
    """Verify morning briefing triggers at 8:00 AM Kolkata and not before."""
    settings = Settings(timezone="Asia/Kolkata")
    kolkata_tz = settings.tz
    deliv_repo = InMemoryDeliveryRepository()
    runner = ServiceRunner(settings=settings, delivery_repo=deliv_repo, enable_bot=False)

    # 07:59 AM -> Should NOT trigger
    dt_759 = datetime(2026, 10, 24, 7, 59, tzinfo=kolkata_tz)
    assert runner.should_trigger_briefing(dt_759, "morning") is False

    # 08:00 AM -> Should trigger
    dt_800 = datetime(2026, 10, 24, 8, 0, tzinfo=kolkata_tz)
    assert runner.should_trigger_briefing(dt_800, "morning") is True

    # 08:30 AM -> Should trigger if not yet sent
    dt_830 = datetime(2026, 10, 24, 8, 30, tzinfo=kolkata_tz)
    assert runner.should_trigger_briefing(dt_830, "morning") is True


def test_should_trigger_briefing_evening_at_8pm():
    """Verify evening briefing triggers at 8:00 PM (20:00) Kolkata and not before."""
    settings = Settings(timezone="Asia/Kolkata")
    kolkata_tz = settings.tz
    deliv_repo = InMemoryDeliveryRepository()
    runner = ServiceRunner(settings=settings, delivery_repo=deliv_repo, enable_bot=False)

    # 19:59 PM (07:59 PM) -> Should NOT trigger
    dt_1959 = datetime(2026, 10, 24, 19, 59, tzinfo=kolkata_tz)
    assert runner.should_trigger_briefing(dt_1959, "evening") is False

    # 20:00 PM (08:00 PM) -> Should trigger
    dt_2000 = datetime(2026, 10, 24, 20, 0, tzinfo=kolkata_tz)
    assert runner.should_trigger_briefing(dt_2000, "evening") is True


def test_restart_recovery_prevents_duplicate_scheduled_sends():
    """Verify that restarting the service does not resend an already-delivered briefing."""
    settings = Settings(timezone="Asia/Kolkata")
    kolkata_tz = settings.tz
    target_date = date(2026, 10, 24)

    deliv_repo = InMemoryDeliveryRepository()
    # Pre-record that morning briefing was already successfully sent
    deliv_repo.create_briefing(
        briefing_date=target_date,
        total_chunks=1,
        digest_hash="abc",
        edition="morning",
    )
    rec = deliv_repo.get_briefing_by_date(target_date, edition="morning")
    assert rec is not None
    updated_rec = BriefingRecord(
        id=rec.id,
        briefing_date=target_date,
        edition="morning",
        status="sent",
        started_at=rec.started_at,
        completed_at=datetime.now(UTC),
        telegram_message_count=1,
        chunks_delivered=1,
    )
    deliv_repo.update_briefing(updated_rec)

    # Simulate fresh service start after restart
    runner = ServiceRunner(
        settings=settings,
        delivery_repo=deliv_repo,
        enable_bot=False,
    )

    # Current time is 09:00 AM on the same date
    dt_900 = datetime(2026, 10, 24, 9, 0, tzinfo=kolkata_tz)

    # Must NOT trigger because it was already sent
    assert runner.should_trigger_briefing(dt_900, "morning") is False

    # Evening briefing has not been sent yet, so at 20:00 it should still trigger
    dt_2000 = datetime(2026, 10, 24, 20, 0, tzinfo=kolkata_tz)
    assert runner.should_trigger_briefing(dt_2000, "evening") is True


def test_consecutive_briefing_deduplication():
    """Verify unchanged stories from previous briefing are suppressed in consecutive runs."""
    settings = Settings(timezone="Asia/Kolkata")
    today_date = date(2026, 10, 24)
    deliv_repo = InMemoryDeliveryRepository()

    art = _create_sample_article(title="Border Forces Deploy New Drone Surveillance")
    ev = _create_sample_event(title="Border Forces Deploy New Drone Surveillance", articles=[art])
    ev_hash = compute_event_hash(ev)

    # 1. Deliver morning briefing containing this event
    mock_tg = MockTelegramClient()
    deliv_svc = TelegramDeliveryService(client=mock_tg, repository=deliv_repo, settings=settings)

    mock_summarizer = MagicMock(spec=EventSummarizer)
    from geopulse.summarization.models import EventSummary, SummarizationResult

    summary = EventSummary(
        event_id=ev.id,
        headline=ev.title,
        what_happened="Surveillance drones deployed along northern border area.",
        confidence_qualification="Corroborated across source reporting.",
        model_used="test-model",
    )
    res = SummarizationResult(event_id=ev.id, status="success", summary=summary, event=ev)
    mock_summarizer.summarize_events.return_value = (res,)

    pipeline = BriefingPipeline(
        settings=settings,
        delivery_repo=deliv_repo,
        delivery_service=deliv_svc,
        summarizer=mock_summarizer,
    )

    # Run morning briefing
    pipeline_res_morning = pipeline.run(
        briefing_date=today_date,
        articles=[art],
        send=True,
        edition="morning",
    )
    assert pipeline_res_morning.delivery_result is not None
    assert pipeline_res_morning.delivery_result.status == "sent"

    # Verify latest briefing recorded the delivered event hash
    latest = deliv_repo.get_latest_briefing()
    assert latest is not None
    assert latest.edition == "morning"
    assert ev_hash in latest.delivered_event_hashes

    # 2. Run evening briefing with the exact same unchanged story
    pipeline_res_evening = pipeline.run(
        briefing_date=today_date,
        articles=[art],
        send=True,
        edition="evening",
        deduplicate_consecutive=True,
    )

    # Because story was unchanged, events were filtered out, producing honest low-activity update
    assert pipeline_res_evening.events_count == 0
    assert len(pipeline_res_evening.digest_chunks) == 1
    chunk = pipeline_res_evening.digest_chunks[0]

    # Verify honest low-activity bulletin for evening
    assert "GEOPULSE EVENING BRIEFING" in chunk
    assert "since the morning briefing" in chunk


def test_consecutive_deduplication_overridden_by_force():
    """Verify --force overrides consecutive deduplication and delivers all stories."""
    settings = Settings(timezone="Asia/Kolkata")
    today_date = date(2026, 10, 24)
    deliv_repo = InMemoryDeliveryRepository()

    art = _create_sample_article(title="Naval Maneuvers Underway in South China Sea")
    ev = _create_sample_event(title="Naval Maneuvers Underway in South China Sea", articles=[art])

    mock_tg = MockTelegramClient()
    deliv_svc = TelegramDeliveryService(client=mock_tg, repository=deliv_repo, settings=settings)

    from geopulse.summarization.models import EventSummary, SummarizationResult

    summary = EventSummary(
        event_id=ev.id,
        headline=ev.title,
        what_happened="Naval forces initiated carrier strike group exercises.",
        confidence_qualification="Single-source defence reporting.",
        model_used="test-model",
    )
    res = SummarizationResult(event_id=ev.id, status="success", summary=summary, event=ev)
    mock_summarizer = MagicMock(spec=EventSummarizer)
    mock_summarizer.summarize_events.return_value = (res,)

    pipeline = BriefingPipeline(
        settings=settings,
        delivery_repo=deliv_repo,
        delivery_service=deliv_svc,
        summarizer=mock_summarizer,
    )

    # Morning run
    pipeline.run(briefing_date=today_date, articles=[art], send=True, edition="morning")

    # Evening run with force=True
    pipeline_res_forced = pipeline.run(
        briefing_date=today_date,
        articles=[art],
        send=True,
        edition="evening",
        force=True,
        deduplicate_consecutive=True,
    )

    # Must contain the event because force bypassed deduplication
    assert pipeline_res_forced.events_count == 1
    assert "Naval Maneuvers Underway" in pipeline_res_forced.digest_chunks[0]


def test_cli_service_status_command(capsys):
    """Verify 'geopulse service status' prints schedule, timezone, and component status."""
    from geopulse.cli import main

    code = main(["service", "status"])
    captured = capsys.readouterr().out

    assert code == 0
    assert "Geopulse Continuous Service Status & Configuration" in captured
    assert "Morning Briefing:     08:00 AM" in captured
    assert "Evening Briefing:     08:00 PM" in captured
    assert "Asia/Kolkata" in captured


def test_cli_service_run_mock_once(capsys):
    """Verify 'geopulse service run --mock-telegram --mock-model' with --once exits cleanly."""
    from geopulse.cli import main

    code = main(["service", "run", "--mock-telegram", "--mock-model", "--no-scheduler", "--once"])
    captured = capsys.readouterr().out

    assert code == 0
    assert "Starting Geopulse continuous service..." in captured


def test_cli_service_run_smoke_test_exercises_scheduler_and_bot(monkeypatch, capsys):
    """Verify smoke-test mode runs both scheduler evaluation and Telegram Q&A polling."""
    from geopulse.cli import main
    from geopulse.delivery.models import DeliveryResult
    from geopulse.pipeline import PipelineResult

    def mock_pipeline_run(self, *args, **kwargs):
        edition = kwargs.get("edition", "morning")
        target_d = kwargs.get("briefing_date", date(2026, 10, 24))
        deliv_res = DeliveryResult(
            briefing_date=target_d,
            status="sent",
            telegram_message_ids=(1001,),
            chunks_delivered=1,
            digest_hash="test-hash",
        )
        return PipelineResult(
            briefing_date=target_d,
            is_preview=False,
            is_persistent=False,
            articles_count=1,
            new_articles_count=1,
            events_count=1,
            events=(),
            summaries=(),
            digest_chunks=("test digest",),
            edition=edition,
            delivery_result=deliv_res,
        )

    monkeypatch.setattr("geopulse.pipeline.BriefingPipeline.run", mock_pipeline_run)

    code = main(["service", "run", "--mock-telegram", "--mock-model", "--once"])
    captured = capsys.readouterr().out

    assert code == 0
    assert "Geopulse Service Smoke Test Summary (--once)" in captured
    assert "Scheduler Status:" in captured
    assert "Telegram Q&A:" in captured
    assert "1 update(s) processed" in captured
    assert "Mock Messages:" in captured
    assert "VERIFIED (Single pass executed cleanly)" in captured


def test_service_instance_lock_prevents_duplicate_process(tmp_path):
    """Verify that acquiring an instance lock prevents a second service from running."""
    settings = Settings(timezone="Asia/Kolkata")
    lock_file = tmp_path / "service.lock"
    deliv_repo = InMemoryDeliveryRepository()

    runner1 = ServiceRunner(
        settings=settings,
        delivery_repo=deliv_repo,
        enable_bot=False,
        enable_scheduler=False,
        lock_instance=True,
        lock_path=lock_file,
    )
    runner1.acquire_instance_lock()
    assert lock_file.exists()

    runner2 = ServiceRunner(
        settings=settings,
        delivery_repo=deliv_repo,
        enable_bot=False,
        enable_scheduler=False,
        lock_instance=True,
        lock_path=lock_file,
    )

    with pytest.raises(ServiceAlreadyRunningError) as exc_info:
        runner2.acquire_instance_lock()
    assert "already running" in str(exc_info.value).lower()

    # Releasing the lock from runner1 allows runner2 to acquire it
    runner1.release_instance_lock()
    runner2.acquire_instance_lock()
    runner2.release_instance_lock()


def test_cli_service_handles_service_already_running(monkeypatch, capsys):
    """Verify CLI service handles ServiceAlreadyRunningError gracefully with exit code 1."""
    from geopulse.cli import main

    def mock_run(self, once=False):
        raise ServiceAlreadyRunningError("Locked by another process.")

    monkeypatch.setattr("geopulse.service.runner.ServiceRunner.run", mock_run)

    code = main(["service", "run", "--mock-telegram", "--mock-model", "--once"])
    captured = capsys.readouterr().out
    assert code == 1
    assert "Locked by another process" in captured
