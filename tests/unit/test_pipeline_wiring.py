"""Offline tests for Phase 6B end-to-end persistent pipeline wiring."""

from datetime import UTC, date, datetime
from unittest.mock import MagicMock

from geopulse.classification.models import TopicCategory
from geopulse.cli import main
from geopulse.clustering.models import EventCluster
from geopulse.clustering.repository import InMemoryEventRepository
from geopulse.config import Settings
from geopulse.delivery.models import TelegramMessageResponse
from geopulse.delivery.repository import InMemoryDeliveryRepository
from geopulse.delivery.service import TelegramDeliveryService
from geopulse.ingestion.models import ArticleRecord
from geopulse.ingestion.repository import InMemoryArticleRepository
from geopulse.pipeline import BriefingPipeline, PipelineResult
from geopulse.summarization.models import (
    EventKeyDetails,
    EventSourceAttribution,
    EventSummary,
    SummarizationResult,
)
from geopulse.summarization.service import EventSummarizer


def _create_sample_articles() -> tuple[ArticleRecord, ArticleRecord]:
    now = datetime(2026, 10, 23, 10, 0, 0, tzinfo=UTC)
    art1 = ArticleRecord(
        source_name="Defense News",
        title="Missile Defense System Procured",
        url="https://defensenews.com/missile",
        canonical_url="https://defensenews.com/missile",
        published_at=now,
        fetched_at=now,
        snippet="Missile defense contract details signed by military forces.",
        content_hash="hash-1",
        relevance_status="relevant",
        topic="weapons_procurement",
    )
    art2 = ArticleRecord(
        source_name="Janes",
        title="Air Defense Procurement Finalized",
        url="https://janes.com/air-defense",
        canonical_url="https://janes.com/air-defense",
        published_at=now,
        fetched_at=now,
        snippet="Air defense procurement agreement finalized with contractors.",
        content_hash="hash-2",
        relevance_status="relevant",
        topic="weapons_procurement",
    )
    return art1, art2


def test_end_to_end_pipeline_orchestration_fake_repositories():
    """Verify that BriefingPipeline correctly orchestrates the entire persistent sequence."""
    art1, art2 = _create_sample_articles()
    today_date = date(2026, 10, 23)

    article_repo = InMemoryArticleRepository()
    event_repo = InMemoryEventRepository()
    delivery_repo = InMemoryDeliveryRepository()

    mock_summarizer = MagicMock(spec=EventSummarizer)
    summary_obj = EventSummary(
        event_id="dummy-id",
        headline="Missile Defense Procurement Finalized",
        what_happened="Military forces officially signed contracts for advanced air defense.",
        key_details=EventKeyDetails(actors=("Military",), action="Signed procurement contracts"),
        significance="Strengthens regional deterrence capabilities.",
        confidence_qualification="Corroborated by two independent defense outlets",
        sources=(
            EventSourceAttribution(
                source_name=art1.source_name, title=art1.title, url=art1.url
            ),
        ),
        model_used="mock-mistral:latest",
    )

    def fake_summarize(events):
        return tuple(
            SummarizationResult(
                event_id=ev.id,
                status="success",
                summary=summary_obj,
                event=ev,
            )
            for ev in events
        )

    mock_summarizer.summarize_events.side_effect = fake_summarize

    mock_tg = MagicMock()
    mock_tg.send_message.return_value = TelegramMessageResponse(
        message_id=901,
        chat_id="test-chat",
        date=1729684800,
    )
    delivery_service = TelegramDeliveryService(
        client=mock_tg,
        repository=delivery_repo,
        settings=Settings(),
    )

    pipeline = BriefingPipeline(
        article_repo=article_repo,
        event_repo=event_repo,
        delivery_repo=delivery_repo,
        summarizer=mock_summarizer,
        delivery_service=delivery_service,
    )

    res = pipeline.run(
        briefing_date=today_date,
        articles=[art1, art2],
        persistent=True,
        send=True,
    )

    # 1. Assert pipeline success metrics
    assert isinstance(res, PipelineResult)
    assert res.is_preview is False
    assert res.is_persistent is True
    assert res.articles_count == 2
    assert res.new_articles_count == 2
    assert res.events_count >= 1
    assert res.briefing_id is not None
    assert res.delivery_result is not None
    assert res.delivery_result.status == "sent"

    # 2. Assert articles persisted
    ids_map = article_repo.get_article_ids_by_urls([art1.canonical_url, art2.canonical_url])
    assert len(ids_map) == 2

    # 3. Assert events persisted in event_repo
    persisted_events = event_repo.get_events_by_date(today_date)
    assert len(persisted_events) >= 1

    # 4. Assert delivery repository has briefing_events and briefing_evidence
    assert res.persisted_briefing_events >= 1
    assert res.persisted_evidence_records >= 1
    assert res.briefing_id in delivery_repo._briefing_events
    assert res.briefing_id in delivery_repo._briefing_evidence


def test_correct_persistence_order_verified():
    """Verify that articles are persisted before events, and events before briefing evidence."""
    art1, art2 = _create_sample_articles()
    today_date = date(2026, 10, 23)

    call_order: list[str] = []

    class InstrumentedArticleRepo(InMemoryArticleRepository):
        def persist_articles(self, source_id, articles):
            call_order.append("article_repo.persist_articles")
            return super().persist_articles(source_id, articles)

        def get_article_ids_by_urls(self, canonical_urls):
            call_order.append("article_repo.get_article_ids_by_urls")
            return super().get_article_ids_by_urls(canonical_urls)

    class InstrumentedEventRepo(InMemoryEventRepository):
        def persist_events(self, events, article_url_to_id=None):
            call_order.append("event_repo.persist_events")
            return super().persist_events(events, article_url_to_id)

        def update_event_summaries(self, results):
            call_order.append("event_repo.update_event_summaries")
            return super().update_event_summaries(results)

    class InstrumentedDeliveryRepo(InMemoryDeliveryRepository):
        def create_briefing(self, briefing_date, total_chunks, digest_hash=None):
            call_order.append("delivery_repo.create_briefing")
            return super().create_briefing(briefing_date, total_chunks, digest_hash)

        def record_briefing_events(self, briefing_id, results, article_url_to_id=None):
            call_order.append("delivery_repo.record_briefing_events")
            return super().record_briefing_events(briefing_id, results, article_url_to_id)

    art_repo = InstrumentedArticleRepo()
    ev_repo = InstrumentedEventRepo()
    deliv_repo = InstrumentedDeliveryRepo()

    mock_tg = MagicMock()
    mock_tg.send_message.return_value = TelegramMessageResponse(
        message_id=902,
        chat_id="test-chat",
        date=1729684800,
    )
    delivery_service = TelegramDeliveryService(
        client=mock_tg,
        repository=deliv_repo,
        settings=Settings(),
    )

    mock_summarizer = MagicMock(spec=EventSummarizer)
    mock_summarizer.summarize_events.return_value = ()

    pipeline = BriefingPipeline(
        article_repo=art_repo,
        event_repo=ev_repo,
        delivery_repo=deliv_repo,
        summarizer=mock_summarizer,
        delivery_service=delivery_service,
    )

    pipeline.run(
        briefing_date=today_date,
        articles=[art1, art2],
        persistent=True,
        send=True,
    )

    # Assert exact sequential order
    idx_art_persist = call_order.index("article_repo.persist_articles")
    idx_art_resolve = call_order.index("article_repo.get_article_ids_by_urls")
    idx_ev_persist = call_order.index("event_repo.persist_events")
    idx_ev_summaries = call_order.index("event_repo.update_event_summaries")
    idx_deliv_create = call_order.index("delivery_repo.create_briefing")
    idx_deliv_evidence = call_order.index("delivery_repo.record_briefing_events")

    # Articles persisted before article ID resolution
    assert idx_art_persist < idx_art_resolve
    # Article IDs resolved before events persisted
    assert idx_art_resolve < idx_ev_persist
    # Events persisted before summaries updated
    assert idx_ev_persist < idx_ev_summaries
    # Briefing created before briefing events recorded
    assert idx_deliv_create < idx_deliv_evidence
    # Events persisted before briefing evidence recorded
    assert idx_ev_persist < idx_deliv_evidence


def test_repeat_run_idempotency_and_stable_event_ids():
    """Verify repeat runs produce identical event UUIDs and do not duplicate articles."""
    art1, art2 = _create_sample_articles()
    today_date = date(2026, 10, 23)

    art_repo = InMemoryArticleRepository()
    ev_repo = InMemoryEventRepository()
    deliv_repo = InMemoryDeliveryRepository()

    mock_summarizer = MagicMock(spec=EventSummarizer)
    mock_summarizer.summarize_events.return_value = ()

    pipeline = BriefingPipeline(
        article_repo=art_repo,
        event_repo=ev_repo,
        delivery_repo=deliv_repo,
        summarizer=mock_summarizer,
    )

    # Run 1
    res1 = pipeline.run(
        briefing_date=today_date,
        articles=[art1, art2],
        persistent=True,
        send=False,
    )
    assert res1.new_articles_count == 2
    event_ids_run1 = [e.id for e in res1.events]

    # Run 2 with identical articles
    res2 = pipeline.run(
        briefing_date=today_date,
        articles=[art1, art2],
        persistent=True,
        send=False,
    )
    # Second run reports 0 new articles (deduplicated against canonical_url)
    assert res2.new_articles_count == 0
    event_ids_run2 = [e.id for e in res2.events]

    # Event IDs are deterministic and identical across runs
    assert event_ids_run1 == event_ids_run2


def test_preview_mode_makes_zero_database_delivery_writes_and_zero_telegram_calls():
    """Verify that preview mode format digest but touches neither Telegram nor delivery records."""
    art1, _ = _create_sample_articles()
    today_date = date(2026, 10, 23)

    deliv_repo = InMemoryDeliveryRepository()
    mock_tg = MagicMock()
    delivery_service = TelegramDeliveryService(
        client=mock_tg,
        repository=deliv_repo,
        settings=Settings(),
    )

    mock_summarizer = MagicMock(spec=EventSummarizer)
    mock_summarizer.summarize_events.return_value = ()

    pipeline = BriefingPipeline(
        delivery_repo=deliv_repo,
        summarizer=mock_summarizer,
        delivery_service=delivery_service,
    )

    res = pipeline.run(
        briefing_date=today_date,
        articles=[art1],
        persistent=False,
        send=False,  # Preview mode
    )

    assert res.is_preview is True
    assert res.delivery_result is None
    assert res.briefing_id is None
    # Zero calls to Telegram
    assert mock_tg.send_message.call_count == 0
    # Zero records in delivery_repo
    assert len(deliv_repo._records) == 0
    assert len(res.digest_chunks) >= 1


def test_missing_credentials_fails_safely_without_network():
    """Verify pipeline fails safely with clear error if persistent mode has no credentials."""
    unconfigured_settings = Settings(supabase_url=None, supabase_key=None)

    pipeline = BriefingPipeline(settings=unconfigured_settings)
    res = pipeline.run(persistent=True, send=False)

    assert res.error_message is not None
    assert "requires SUPABASE_URL and SUPABASE_KEY" in res.error_message


def test_cli_cluster_persist_without_credentials_fails_safely(capsys, monkeypatch):
    """Verify CLI cluster --persist fails safely when credentials absent."""
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)

    ret = main(["cluster", "--persist"])
    captured = capsys.readouterr().out
    assert ret == 1
    assert "--persist requires SUPABASE_URL and SUPABASE_KEY" in captured


def test_cli_brief_persist_without_credentials_fails_safely(capsys, monkeypatch):
    """Verify CLI brief --persist fails safely when credentials absent."""
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)

    ret = main(["brief", "--persist"])
    captured = capsys.readouterr().out
    assert ret == 1
    assert "Persistence (--persist / --supabase-state) requires" in captured


def test_mode_1_default_preview_makes_zero_db_and_zero_telegram_calls():
    """Mode 1: Default preview has 0 DB writes, 0 Telegram calls, 0 briefing records."""
    art1, art2 = _create_sample_articles()
    today_date = date(2026, 10, 23)

    art_repo = MagicMock(spec=InMemoryArticleRepository)
    ev_repo = MagicMock(spec=InMemoryEventRepository)
    deliv_repo = MagicMock(spec=InMemoryDeliveryRepository)
    delivery_service = MagicMock(spec=TelegramDeliveryService)

    mock_summarizer = MagicMock(spec=EventSummarizer)
    mock_summarizer.summarize_events.return_value = ()

    pipeline = BriefingPipeline(
        article_repo=art_repo,
        event_repo=ev_repo,
        delivery_repo=deliv_repo,
        summarizer=mock_summarizer,
        delivery_service=delivery_service,
    )

    res = pipeline.run(
        briefing_date=today_date,
        articles=[art1, art2],
        persistent=False,
        send=False,
    )

    assert res.is_preview is True
    assert res.is_persistent is False
    assert res.delivery_result is None
    assert res.briefing_id is None
    # 0 DB writes
    art_repo.persist_articles.assert_not_called()
    ev_repo.persist_events.assert_not_called()
    ev_repo.update_event_summaries.assert_not_called()
    deliv_repo.create_briefing.assert_not_called()
    deliv_repo.record_briefing_events.assert_not_called()
    # 0 Telegram calls
    delivery_service.deliver_digest.assert_not_called()


def test_mode_2_persistent_preview_writes_db_but_makes_zero_telegram_calls():
    """Mode 2: Persistent preview writes articles/events/summaries, but 0 Telegram calls."""
    art1, art2 = _create_sample_articles()
    today_date = date(2026, 10, 23)

    art_repo = InMemoryArticleRepository()
    ev_repo = InMemoryEventRepository()
    deliv_repo = InMemoryDeliveryRepository()
    mock_tg = MagicMock()
    delivery_service = TelegramDeliveryService(
        client=mock_tg,
        repository=deliv_repo,
        settings=Settings(),
    )

    mock_summarizer = MagicMock(spec=EventSummarizer)
    summary_obj = EventSummary(
        event_id="dummy-id",
        headline="Missile Defense System Procured",
        what_happened="Military forces officially signed contracts for advanced air defense.",
        confidence_qualification="Corroborated by two independent defense outlets",
        model_used="mock-mistral:latest",
    )
    mock_summarizer.summarize_events.side_effect = lambda evs: tuple(
        SummarizationResult(event_id=ev.id, status="success", summary=summary_obj, event=ev)
        for ev in evs
    )

    pipeline = BriefingPipeline(
        article_repo=art_repo,
        event_repo=ev_repo,
        delivery_repo=deliv_repo,
        summarizer=mock_summarizer,
        delivery_service=delivery_service,
    )

    res = pipeline.run(
        briefing_date=today_date,
        articles=[art1, art2],
        persistent=True,
        send=False,  # Preview without send
    )

    assert res.is_preview is True
    assert res.is_persistent is True
    assert res.delivery_result is None
    assert res.briefing_id is None

    # Database writes occurred for articles, events, summaries
    assert len(art_repo.get_article_ids_by_urls([art1.canonical_url, art2.canonical_url])) == 2
    assert len(ev_repo.get_events_by_date(today_date)) >= 1
    assert len(ev_repo._summaries) >= 1

    # Zero Telegram calls and zero delivery / briefing records
    assert mock_tg.send_message.call_count == 0
    assert len(deliv_repo._records) == 0
    assert len(deliv_repo._briefing_events) == 0


def test_failure_in_article_persistence_fails_closed_and_prevents_telegram_send():
    """If article persistence fails, pipeline fails closed and never calls Telegram delivery."""
    art1, _ = _create_sample_articles()
    art_repo = MagicMock(spec=InMemoryArticleRepository)
    art_repo.persist_articles.side_effect = RuntimeError("PostgREST database error 500")

    mock_delivery = MagicMock(spec=TelegramDeliveryService)
    pipeline = BriefingPipeline(
        article_repo=art_repo,
        event_repo=InMemoryEventRepository(),
        delivery_repo=InMemoryDeliveryRepository(),
        delivery_service=mock_delivery,
    )

    res = pipeline.run(articles=[art1], persistent=True, send=True)

    assert res.error_message is not None
    assert "Failed to persist articles: PostgREST database error 500" in res.error_message
    assert res.delivery_result is None
    mock_delivery.deliver_digest.assert_not_called()


def test_failure_in_event_persistence_fails_closed_and_prevents_telegram_send():
    """If event persistence fails, pipeline fails closed and never calls Telegram delivery."""
    art1, _ = _create_sample_articles()
    ev_repo = MagicMock(spec=InMemoryEventRepository)
    ev_repo.persist_events.side_effect = RuntimeError(
        "Unique constraint violation in public.events"
    )

    mock_delivery = MagicMock(spec=TelegramDeliveryService)
    pipeline = BriefingPipeline(
        article_repo=InMemoryArticleRepository(),
        event_repo=ev_repo,
        delivery_repo=InMemoryDeliveryRepository(),
        delivery_service=mock_delivery,
    )

    res = pipeline.run(articles=[art1], persistent=True, send=True)

    assert res.error_message is not None
    assert "Failed to persist events: Unique constraint violation" in res.error_message
    assert res.delivery_result is None
    mock_delivery.deliver_digest.assert_not_called()


def test_failure_in_summary_persistence_fails_closed_and_prevents_telegram_send():
    """If event summary update fails, pipeline fails closed and never calls Telegram delivery."""
    art1, _ = _create_sample_articles()
    ev_repo = MagicMock(spec=InMemoryEventRepository)
    ev_repo.persist_events.return_value = (1, 1)
    ev_repo.update_event_summaries.side_effect = RuntimeError(
        "Failed to update public.events summary"
    )

    mock_delivery = MagicMock(spec=TelegramDeliveryService)
    pipeline = BriefingPipeline(
        article_repo=InMemoryArticleRepository(),
        event_repo=ev_repo,
        delivery_repo=InMemoryDeliveryRepository(),
        delivery_service=mock_delivery,
    )

    res = pipeline.run(articles=[art1], persistent=True, send=True)

    assert res.error_message is not None
    assert "Failed to persist event summaries: Failed to update" in res.error_message
    assert res.delivery_result is None
    mock_delivery.deliver_digest.assert_not_called()


def test_failure_in_briefing_creation_fails_closed():
    """All upstream persistence succeeds, but briefing creation fails; zero Telegram sends occur."""
    art1, art2 = _create_sample_articles()
    today_date = date(2026, 10, 23)

    art_repo = InMemoryArticleRepository()
    ev_repo = InMemoryEventRepository()
    deliv_repo = MagicMock(spec=InMemoryDeliveryRepository)
    deliv_repo.acquire_lock.return_value = True
    deliv_repo.get_briefing_by_date.return_value = None
    deliv_repo.create_briefing.side_effect = RuntimeError(
        "PostgREST briefing creation constraint failure"
    )

    mock_tg = MagicMock()
    delivery_service = TelegramDeliveryService(
        client=mock_tg,
        repository=deliv_repo,
        settings=Settings(),
    )

    mock_summarizer = MagicMock(spec=EventSummarizer)
    mock_summarizer.summarize_events.return_value = ()

    pipeline = BriefingPipeline(
        article_repo=art_repo,
        event_repo=ev_repo,
        delivery_repo=deliv_repo,
        summarizer=mock_summarizer,
        delivery_service=delivery_service,
    )

    res = pipeline.run(
        briefing_date=today_date,
        articles=[art1, art2],
        persistent=True,
        send=True,
    )

    assert res.delivery_result is not None
    assert res.delivery_result.status == "failed"
    assert "Failed to create briefing state record" in res.error_message
    # Telegram client was NEVER called
    mock_tg.send_message.assert_not_called()
    # Upstream articles and events remain persisted
    assert len(art_repo.get_article_ids_by_urls([art1.canonical_url, art2.canonical_url])) == 2
    assert len(ev_repo.get_events_by_date(today_date)) >= 1


def test_partial_delivery_failure_and_resumption_rerun():
    """Chunk 1 succeeds, Chunk 2 fails. On retry, chunk 1 is skipped and chunk 2 delivers."""
    from geopulse.delivery.exceptions import TelegramNetworkError

    art1, art2 = _create_sample_articles()
    today_date = date(2026, 10, 23)

    art_repo = InMemoryArticleRepository()
    ev_repo = InMemoryEventRepository()
    deliv_repo = InMemoryDeliveryRepository()

    # Formatter with 2 distinct chunks
    mock_formatter = MagicMock()
    mock_formatter.format_digest.return_value = ("Chunk 1 content", "Chunk 2 content")

    mock_tg = MagicMock()
    # First invocation: chunk 1 succeeds, chunk 2 fails
    mock_tg.send_message.side_effect = [
        TelegramMessageResponse(message_id=5001, chat_id="chat-1", date=1729684800),
        TelegramNetworkError("Connection reset by peer on chunk 2"),
        TelegramNetworkError("Connection reset by peer on chunk 2"),
        TelegramNetworkError("Connection reset by peer on chunk 2"),
        TelegramNetworkError("Connection reset by peer on chunk 2"),
    ]

    delivery_service = TelegramDeliveryService(
        client=mock_tg,
        repository=deliv_repo,
        settings=Settings(),
        max_retries=1,
        sleeper=lambda s: None,
    )

    mock_summarizer = MagicMock(spec=EventSummarizer)
    mock_summarizer.summarize_events.return_value = ()

    pipeline = BriefingPipeline(
        article_repo=art_repo,
        event_repo=ev_repo,
        delivery_repo=deliv_repo,
        summarizer=mock_summarizer,
        formatter=mock_formatter,
        delivery_service=delivery_service,
    )

    # Run 1: Fails partially
    res1 = pipeline.run(
        briefing_date=today_date,
        articles=[art1, art2],
        persistent=True,
        send=True,
    )

    assert res1.delivery_result is not None
    assert res1.delivery_result.status == "partial"
    assert res1.delivery_result.chunks_delivered == 1
    rec1 = deliv_repo.get_briefing_by_date(today_date)
    assert rec1 is not None
    assert rec1.status == "partial"
    assert rec1.chunks_delivered == 1

    # Run 2: Retry with successful Telegram response for chunk 2
    mock_tg.send_message.side_effect = [
        TelegramMessageResponse(message_id=5002, chat_id="chat-1", date=1729684810),
    ]

    res2 = pipeline.run(
        briefing_date=today_date,
        articles=[art1, art2],
        persistent=True,
        send=True,
    )

    assert res2.delivery_result is not None
    assert res2.delivery_result.status == "sent"
    assert res2.delivery_result.chunks_delivered == 2
    assert res2.delivery_result.resumed_from_chunk == 1
    assert res2.delivery_result.message_ids == (5001, 5002)


def test_retry_after_event_persistence_failure_heals_state():
    """Run 1 fails at event persistence; rerun succeeds and avoids duplicate articles."""
    art1, art2 = _create_sample_articles()
    today_date = date(2026, 10, 23)

    art_repo = InMemoryArticleRepository()
    ev_repo = InMemoryEventRepository()
    deliv_repo = InMemoryDeliveryRepository()

    mock_tg = MagicMock()
    mock_tg.send_message.return_value = TelegramMessageResponse(
        message_id=7001, chat_id="chat-1", date=1729684800
    )
    delivery_service = TelegramDeliveryService(
        client=mock_tg,
        repository=deliv_repo,
        settings=Settings(),
    )

    mock_summarizer = MagicMock(spec=EventSummarizer)
    mock_summarizer.summarize_events.return_value = ()

    # Fail event persistence on first run
    fail_ev_repo = MagicMock(spec=InMemoryEventRepository)
    fail_ev_repo.persist_events.side_effect = RuntimeError("Transient event constraint failure")

    pipeline_fail = BriefingPipeline(
        article_repo=art_repo,
        event_repo=fail_ev_repo,
        delivery_repo=deliv_repo,
        summarizer=mock_summarizer,
        delivery_service=delivery_service,
    )

    res1 = pipeline_fail.run(
        briefing_date=today_date,
        articles=[art1, art2],
        persistent=True,
        send=True,
    )

    assert res1.error_message is not None
    assert "Failed to persist events" in res1.error_message
    assert res1.delivery_result is None
    # Articles were persisted
    assert len(art_repo.get_article_ids_by_urls([art1.canonical_url, art2.canonical_url])) == 2

    # Run 2: Recover with working event repository
    pipeline_recover = BriefingPipeline(
        article_repo=art_repo,
        event_repo=ev_repo,
        delivery_repo=deliv_repo,
        summarizer=mock_summarizer,
        delivery_service=delivery_service,
    )

    res2 = pipeline_recover.run(
        briefing_date=today_date,
        articles=[art1, art2],
        persistent=True,
        send=True,
    )

    # No duplicate articles inserted (0 new articles)
    assert res2.new_articles_count == 0
    assert res2.delivery_result is not None
    assert res2.delivery_result.status == "sent"
    assert len(ev_repo.get_events_by_date(today_date)) >= 1


def test_retry_after_summary_update_failure_heals_state():
    """Run 1 fails at summary update; rerun updates summaries and delivers cleanly."""
    art1, art2 = _create_sample_articles()
    today_date = date(2026, 10, 23)

    art_repo = InMemoryArticleRepository()
    deliv_repo = InMemoryDeliveryRepository()

    mock_tg = MagicMock()
    mock_tg.send_message.return_value = TelegramMessageResponse(
        message_id=8001, chat_id="chat-1", date=1729684800
    )
    delivery_service = TelegramDeliveryService(
        client=mock_tg,
        repository=deliv_repo,
        settings=Settings(),
    )

    summary_obj = EventSummary(
        event_id="dummy-id",
        headline="Missile Defense System Procured",
        what_happened="Military forces officially signed contracts for advanced air defense.",
        confidence_qualification="Corroborated by independent defense outlets",
        model_used="mock-mistral:latest",
    )
    mock_summarizer = MagicMock(spec=EventSummarizer)
    mock_summarizer.summarize_events.side_effect = lambda evs: tuple(
        SummarizationResult(event_id=ev.id, status="success", summary=summary_obj, event=ev)
        for ev in evs
    )

    # First run: event repo raises on update_event_summaries
    class FlakyEventRepo(InMemoryEventRepository):
        def __init__(self):
            super().__init__()
            self.failed_once = False

        def update_event_summaries(self, results):
            if not self.failed_once:
                self.failed_once = True
                raise RuntimeError("PostgREST update timeout on public.events")
            return super().update_event_summaries(results)

    flaky_ev_repo = FlakyEventRepo()

    pipeline = BriefingPipeline(
        article_repo=art_repo,
        event_repo=flaky_ev_repo,
        delivery_repo=deliv_repo,
        summarizer=mock_summarizer,
        delivery_service=delivery_service,
    )

    # Run 1: fails closed
    res1 = pipeline.run(
        briefing_date=today_date,
        articles=[art1, art2],
        persistent=True,
        send=True,
    )
    assert res1.error_message is not None
    assert "Failed to persist event summaries" in res1.error_message
    assert res1.delivery_result is None
    assert mock_tg.send_message.call_count == 0

    # Run 2: retry succeeds
    res2 = pipeline.run(
        briefing_date=today_date,
        articles=[art1, art2],
        persistent=True,
        send=True,
    )
    assert res2.error_message is None
    assert res2.delivery_result is not None
    assert res2.delivery_result.status == "sent"
    assert len(flaky_ev_repo._summaries) >= 1
    assert mock_tg.send_message.call_count == 1


def test_structured_summary_persistence_and_reconstruction():
    """Verify SupabaseEventRepository updates migration 005 columns and reconstructs summary."""
    from geopulse.clustering.repository import SupabaseEventRepository, reconstruct_event_summary

    mock_client = MagicMock()
    mock_table = MagicMock()
    mock_client.table.return_value = mock_table
    mock_update = MagicMock()
    mock_table.update.return_value = mock_update
    mock_eq = MagicMock()
    mock_update.eq.return_value = mock_eq
    mock_eq.execute.return_value = MagicMock(data=[])

    repo = SupabaseEventRepository(client=mock_client)

    summary_obj = EventSummary(
        event_id="test-ev-123",
        headline="Russian Forces Strike Port Infrastructure in Odesa",
        what_happened="A Russian missile strike damaged grain storage facilities in Odesa port.",
        key_details=EventKeyDetails(
            actors=("Russian military",),
            location="Odesa, Ukraine",
            timing="Overnight",
            action="Missile strike on port grain silos",
        ),
        significance="Threatens Black Sea commercial grain corridor.",
        confidence_qualification="Corroborated by Ukrainian Navy and local port authority",
        model_used="deepseek-r1:8b",
    )

    sample_event = EventCluster(
        id="test-ev-123",
        title="Russian Forces Strike Port Infrastructure in Odesa",
        category=TopicCategory.ARMED_CONFLICT,
        first_seen_at=datetime(2026, 10, 23, 10, 0, tzinfo=UTC),
        last_seen_at=datetime(2026, 10, 23, 10, 0, tzinfo=UTC),
        articles=(),
    )

    res_item = SummarizationResult(
        event_id="test-ev-123",
        status="success",
        summary=summary_obj,
        event=sample_event,
    )

    repo.update_event_summaries([res_item])

    # Assert update was called with all migration 005 fields
    mock_client.table.assert_called_with("events")
    update_payload = mock_table.update.call_args[0][0]
    assert update_payload["summary"] == summary_obj.what_happened
    assert update_payload["significance"] == summary_obj.significance
    assert update_payload["headline"] == summary_obj.headline
    assert update_payload["key_details"] == {
        "actors": ["Russian military"],
        "location": "Odesa, Ukraine",
        "timing": "Overnight",
        "action": "Missile strike on port grain silos",
    }
    assert update_payload["confidence_qualification"] == summary_obj.confidence_qualification
    assert update_payload["model_used"] == "deepseek-r1:8b"

    # Now verify reconstruct_event_summary from a database row
    row_dict = {
        "id": "test-ev-123",
        "title": "Port Infrastructure Strike",
        "summary": update_payload["summary"],
        "significance": update_payload["significance"],
        "headline": update_payload["headline"],
        "key_details": update_payload["key_details"],
        "confidence_qualification": update_payload["confidence_qualification"],
        "model_used": update_payload["model_used"],
        "updated_at": "2026-10-23T12:00:00+00:00",
    }
    reconstructed = reconstruct_event_summary(row_dict)
    assert reconstructed is not None
    assert reconstructed.event_id == "test-ev-123"
    assert reconstructed.headline == "Russian Forces Strike Port Infrastructure in Odesa"
    assert (
        reconstructed.what_happened
        == "A Russian missile strike damaged grain storage facilities in Odesa port."
    )
    assert reconstructed.key_details.actors == ("Russian military",)
    assert reconstructed.key_details.location == "Odesa, Ukraine"
    assert reconstructed.significance == "Threatens Black Sea commercial grain corridor."
    assert reconstructed.model_used == "deepseek-r1:8b"


def test_reconstruct_event_summary_missing_data_returns_none():
    """Verify reconstruct_event_summary returns None if event_id or summary is absent."""
    from geopulse.clustering.repository import reconstruct_event_summary

    assert reconstruct_event_summary({}) is None
    assert reconstruct_event_summary({"id": "123"}) is None
    assert reconstruct_event_summary({"summary": "Valid summary text here"}) is None


def test_supabase_delivery_repository_records_migration_005_fields():
    """Verify SupabaseDeliveryRepository.record_briefing_events includes migration 005 columns."""
    from geopulse.delivery.repository import SupabaseDeliveryRepository

    mock_client = MagicMock()
    mock_table = MagicMock()
    mock_client.table.return_value = mock_table
    mock_table.upsert.return_value.execute.return_value = MagicMock(data=[])

    repo = SupabaseDeliveryRepository(client=mock_client)

    summary_obj = EventSummary(
        event_id="ev-456",
        headline="Strategic Baltic Defense Procurement Finalized",
        what_happened="Danish defense ministry confirmed procurement of air defense interceptors.",
        key_details=EventKeyDetails(
            actors=("Danish Ministry of Defence",),
            location="Copenhagen",
            timing="This morning",
            action="Signed procurement framework",
        ),
        significance="Strengthens NATO Baltic flank integration.",
        confidence_qualification="Official government announcement",
        model_used="mistral:latest",
    )

    sample_event_2 = EventCluster(
        id="ev-456",
        title="Strategic Baltic Defense Procurement Finalized",
        category=TopicCategory.WEAPONS_PROCUREMENT,
        first_seen_at=datetime(2026, 10, 23, 10, 0, tzinfo=UTC),
        last_seen_at=datetime(2026, 10, 23, 10, 0, tzinfo=UTC),
        articles=(),
    )

    res_item = SummarizationResult(
        event_id="ev-456",
        status="success",
        summary=summary_obj,
        event=sample_event_2,
    )

    ev_count, _ = repo.record_briefing_events(
        briefing_id="briefing-789",
        results=[res_item],
    )
    assert ev_count == 1

    # Check briefing_events upsert payload
    mock_client.table.assert_called_with("briefing_events")
    mock_table.upsert.assert_called_once()
    rows = mock_table.upsert.call_args_list[0][0][0]
    row = rows[0]
    assert row["briefing_id"] == "briefing-789"
    assert row["event_id"] == "ev-456"
    assert row["headline"] == "Strategic Baltic Defense Procurement Finalized"
    assert row["key_details"]["actors"] == ["Danish Ministry of Defence"]
    assert row["confidence_qualification"] == "Official government announcement"
    assert row["model_used"] == "mistral:latest"


def test_cli_brief_contradictory_flags_rejected_closed(capsys):
    """Verify CLI brief fails closed on mutually exclusive or contradictory flags."""
    # 1. --preview and --send
    ret = main(["brief", "--preview", "--send"])
    captured = capsys.readouterr().out
    assert ret == 1
    assert "Error: Cannot specify both --preview and --send." in captured

    # 2. --in-memory-state and --state-file
    ret = main(["brief", "--in-memory-state", "--state-file", "foo.json"])
    captured = capsys.readouterr().out
    assert ret == 1
    assert "Error: Cannot specify both --in-memory-state and --state-file." in captured

    # 3. --persist and --state-file
    ret = main(["brief", "--persist", "--state-file", "foo.json"])
    captured = capsys.readouterr().out
    assert ret == 1
    assert "Error: Cannot specify both persistent state and --state-file." in captured

    # 4. --supabase-state and --state-file
    ret = main(["brief", "--supabase-state", "--state-file", "foo.json"])
    captured = capsys.readouterr().out
    assert ret == 1
    assert "Error: Cannot specify both persistent state and --state-file." in captured
