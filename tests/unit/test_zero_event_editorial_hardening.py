"""Unit tests verifying the diagnosis and fixes for the zero-event evening briefing."""

from datetime import UTC, datetime

from geopulse.classification.models import TopicCategory
from geopulse.clustering.models import EventCluster
from geopulse.delivery.formatter import (
    DEFAULT_MIN_RELEVANCE_RANK,
    TelegramDigestFormatter,
    compute_event_priority_rank,
    partition_candidate_events,
)
from geopulse.ingestion.models import ArticleRecord
from geopulse.summarization.models import (
    ClaimEvidenceReference,
    EventKeyDetails,
    EventSourceAttribution,
    EventSummary,
    SummarizationResult,
    SummaryClaim,
)


def _create_article(
    title: str = "Test Military Operation",
    snippet: str = "Armed forces conducted operations in the region.",
    url: str = "https://example.test/article-1",
    source_name: str = "Defence News",
    published_at: datetime | None = None,
) -> ArticleRecord:
    dt = published_at or datetime(2026, 10, 29, 12, 0, tzinfo=UTC)
    return ArticleRecord(
        source_name=source_name,
        title=title,
        url=url,
        canonical_url=url,
        published_at=dt,
        snippet=snippet,
        content_hash=f"hash-{url}",
    )


def _create_event(
    title: str = "Test Military Operation",
    category: TopicCategory = TopicCategory.ARMED_CONFLICT,
    snippet: str = "Armed forces conducted operations in the region.",
    url: str = "https://example.test/article-1",
    source_name: str = "Defence News",
    is_analysis: bool = False,
    confidence_label: str = "single_source",
) -> EventCluster:
    art = _create_article(title=title, snippet=snippet, url=url, source_name=source_name)
    now = datetime(2026, 10, 29, 12, 0, tzinfo=UTC)
    return EventCluster(
        title=title,
        category=category,
        first_seen_at=now,
        last_seen_at=now,
        is_analysis=is_analysis,
        confidence_label=confidence_label,
        articles=(art,),
    )


def _create_summary_result(
    event: EventCluster,
    status: str = "success",
    model_used: str = "mock-ollama",
    headline: str | None = None,
    what_happened: str | None = None,
    claims: tuple[SummaryClaim, ...] = (),
    quarantine_reason: str | None = None,
) -> SummarizationResult:
    head = headline or event.title
    what = what_happened or f"Detailed development: {event.title} proceeded as planned."
    source_attrs = tuple(
        EventSourceAttribution(
            source_name=a.source_name,
            title=a.title,
            url=a.url,
            published_at=a.published_at,
        )
        for a in event.articles
    )
    summary = EventSummary(
        event_id=event.id,
        headline=head,
        what_happened=what,
        key_details=EventKeyDetails(actors=["defence forces"], action="military operation"),
        confidence_qualification="Single-source reporting; not independently verified.",
        sources=source_attrs,
        claims=claims,
        model_used=model_used,
    )
    return SummarizationResult(
        event_id=event.id,
        status=status,
        summary=summary if status != "failed" else None,
        quarantine_reason=quarantine_reason,
        event=event,
    )


# 1. High-priority event passes the default relevance filter
def test_high_priority_event_passes_default_relevance_filter():
    event = _create_event(category=TopicCategory.ARMED_CONFLICT)
    rank = compute_event_priority_rank(event)
    assert rank == 1.0
    assert rank <= DEFAULT_MIN_RELEVANCE_RANK

    selected, deferred = partition_candidate_events(
        items=[event],
        max_events=10,
        min_relevance_rank=DEFAULT_MIN_RELEVANCE_RANK,
        include_low_priority=False,
    )
    assert len(selected) == 1
    assert len(deferred) == 0
    assert selected[0].id == event.id


# 2. Event above the priority threshold is excluded with correct reason/rank
def test_event_above_priority_threshold_is_excluded_by_rank():
    # Historical commemoration or routine update has rank > 9.0
    event = _create_event(
        title="Remains of soldier identified after 100 years at Somme battlefield cemetery",
        category=TopicCategory.DEFENCE_POLICY,
    )
    rank = compute_event_priority_rank(event)
    assert rank == 10.5  # Historical commemoration
    assert rank > DEFAULT_MIN_RELEVANCE_RANK

    selected, deferred = partition_candidate_events(
        items=[event],
        max_events=10,
        min_relevance_rank=DEFAULT_MIN_RELEVANCE_RANK,
        include_low_priority=False,
    )
    assert len(selected) == 0
    assert len(deferred) == 0


# 3. Unknown event categories receive deliberate, tested handling
def test_unknown_event_categories_receive_deliberate_tested_handling():
    # String representation of a valid enum is resolved
    class DummyEvent:
        title = "Generic Operation"
        category = "armed_conflict"
        articles = ()
        is_analysis = False

    rank = compute_event_priority_rank(DummyEvent())
    assert rank == 1.0  # Resolved to ARMED_CONFLICT

    # Completely unknown category string defaults to deliberate UNCERTAIN rank 11.0
    class UnknownEvent:
        title = "Mystery Development"
        category = "completely_unknown_category"
        articles = ()
        is_analysis = False

    rank_unknown = compute_event_priority_rank(UnknownEvent())
    assert rank_unknown == 11.0

    # With default threshold 9.0, unknown categories are excluded
    selected, _ = partition_candidate_events(
        items=[UnknownEvent()],
        max_events=10,
        min_relevance_rank=DEFAULT_MIN_RELEVANCE_RANK,
        include_low_priority=False,
    )
    assert len(selected) == 0

    # With --include-low-priority, unknown categories can be considered
    selected_low, _ = partition_candidate_events(
        items=[UnknownEvent()],
        max_events=10,
        min_relevance_rank=DEFAULT_MIN_RELEVANCE_RANK,
        include_low_priority=True,
    )
    assert len(selected_low) == 1


# 4. No eligible candidates produces consistent, accurate counts
def test_no_eligible_candidates_produces_consistent_accurate_counts():
    formatter = TelegramDigestFormatter()
    # When no results exist at all
    chunks = formatter.format_digest(results=(), edition="evening")
    assert len(chunks) == 1
    assert "No significant new military or geopolitical developments identified" in chunks[0]

    selected = formatter.select_digest_events(results=())
    assert selected == ()


# 5. Omitted placeholders do not appear as published briefing events
def test_omitted_placeholders_do_not_appear_as_published_briefing_events():
    event = _create_event(category=TopicCategory.ARMED_CONFLICT)
    res = _create_summary_result(event, status="success", model_used="omitted")

    formatter = TelegramDigestFormatter()
    selected = formatter.select_digest_events(results=[res])
    assert len(selected) == 0

    chunks = formatter.format_digest(results=[res], edition="evening")
    assert len(chunks) == 1
    assert "No significant new military or geopolitical developments identified" in chunks[0]


# 6. Headline-equality filtering behaves as intended:
# Verified summaries with claims are preserved even if single claim
def test_headline_equality_filtering_preserves_verified_summaries():
    event = _create_event(
        title="Taiwan Conflict Window Remains Valid",
        category=TopicCategory.DEFENCE_POLICY,
    )
    claim = SummaryClaim(
        claim_id="c1",
        text="Army Pacific positions personnel to meet 2027 Taiwan benchmark.",
        evidence_refs=(
            ClaimEvidenceReference(
                source_url=event.articles[0].url,
                evidence_span="Army Pacific positions personnel",
                verification_status="accepted",
            ),
        ),
        classification="reported_fact",
        central=True,
        verification_status="accepted",
    )
    # A summary with identical headline and what_happened but holding verified claims
    # must NOT be discarded
    res = _create_summary_result(
        event,
        status="success",
        headline="Army Pacific positions personnel to meet 2027 Taiwan benchmark.",
        what_happened="Army Pacific positions personnel to meet 2027 Taiwan benchmark.",
        claims=(claim,),
        model_used="mock-ollama",
    )
    formatter = TelegramDigestFormatter()
    selected = formatter.select_digest_events(results=[res])
    assert len(selected) == 1
    assert selected[0].event_id == event.id


# 7. Failed and quarantined summaries are accounted for correctly
def test_failed_and_quarantined_summaries_are_accounted_correctly():
    event1 = _create_event(title="Event 1", category=TopicCategory.ARMED_CONFLICT)
    event2 = _create_event(title="Event 2", category=TopicCategory.MILITARY_OPERATIONS)

    res_quarantined = _create_summary_result(
        event1,
        status="quarantined",
        quarantine_reason="Central claim not verified by source",
    )
    res_failed = _create_summary_result(
        event2,
        status="failed",
    )

    formatter = TelegramDigestFormatter()
    selected = formatter.select_digest_events(results=[res_quarantined, res_failed])
    assert len(selected) == 0

    # Truthful message when events exist but were quarantined/failed
    chunks = formatter.format_digest(
        results=[res_quarantined, res_failed],
        edition="evening",
    )
    assert len(chunks) == 1
    expected_msg = (
        "No significant new military or geopolitical developments confirmed "
        "meeting briefing thresholds"
    )
    assert expected_msg in chunks[0]
    assert "Developing items remain under active monitoring" in chunks[0]


# 8. The event cap and bounded backfill work as intended
def test_event_cap_and_bounded_backfill():
    events = [
        _create_event(
            title=f"Operation {i}",
            category=TopicCategory.ARMED_CONFLICT,
            url=f"https://example.test/art-{i}",
        )
        for i in range(12)
    ]

    selected, deferred = partition_candidate_events(
        items=events,
        max_events=5,
        min_relevance_rank=DEFAULT_MIN_RELEVANCE_RANK,
        include_low_priority=False,
    )
    assert len(selected) == 5
    assert len(deferred) == 7


# 9. The displayed event count, omitted count, and digest contents agree
def test_displayed_event_count_omitted_count_and_digest_agree():
    events = [
        _create_event(
            title=f"Operation {i}",
            category=TopicCategory.ARMED_CONFLICT,
            url=f"https://example.test/art-{i}",
        )
        for i in range(6)
    ]
    results = [
        _create_summary_result(
            ev,
            status="success",
            model_used="mock-ollama",
            headline=f"Headline for {ev.title}",
            what_happened=f"Detailed narrative for {ev.title}",
        )
        for ev in events
    ]

    formatter = TelegramDigestFormatter()
    selected = formatter.select_digest_events(results=results, max_events=4)
    assert len(selected) == 4

    chunks = formatter.format_digest(
        results=results,
        max_events=4,
        edition="evening",
    )
    combined = "\n".join(chunks)
    assert "4 event(s) briefed" in combined
    assert "2 lower-priority event(s) omitted" in combined


# 10. A genuinely empty briefing produces a truthful message
def test_genuinely_empty_briefing_produces_truthful_message():
    formatter = TelegramDigestFormatter()
    # Genuinely no events ingested / identified
    chunks = formatter.format_digest(results=(), edition="evening")
    expected_empty = (
        "No significant new military or geopolitical developments identified "
        "since the morning briefing."
    )
    assert expected_empty in chunks[0]

    # Events ingested but all quarantined (not genuinely empty, pipeline rejection)
    ev = _create_event(category=TopicCategory.ARMED_CONFLICT)
    quarantined = _create_summary_result(
        ev, status="quarantined", quarantine_reason="Source outside event"
    )
    chunks_quarantined = formatter.format_digest(results=[quarantined], edition="evening")
    expected_quarantined_msg = (
        "No significant new military or geopolitical developments confirmed "
        "meeting briefing thresholds"
    )
    assert expected_quarantined_msg in chunks_quarantined[0]


# 11. Existing morning/evening behavior and low-priority opt-in behavior remain compatible
def test_morning_evening_and_low_priority_compatibility():
    formatter = TelegramDigestFormatter()

    ev_low = _create_event(
        title="Sanctions review meeting scheduled",
        category=TopicCategory.SANCTIONS,
    )
    # Rank for sanctions is 9.0
    rank = compute_event_priority_rank(ev_low)
    assert rank == 9.0

    # Excluded when min_relevance_rank is 8.0
    sel_strict, _ = partition_candidate_events(
        items=[ev_low],
        min_relevance_rank=8.0,
        include_low_priority=False,
    )
    assert len(sel_strict) == 0

    # Included when include_low_priority is True
    sel_opt_in, _ = partition_candidate_events(
        items=[ev_low],
        min_relevance_rank=8.0,
        include_low_priority=True,
    )
    assert len(sel_opt_in) == 1

    # Morning and evening badges render cleanly
    chunks_morning = formatter.format_digest(results=(), edition="morning")
    assert "GEOPULSE MORNING BRIEFING" in chunks_morning[0]

    chunks_evening = formatter.format_digest(results=(), edition="evening")
    assert "GEOPULSE EVENING BRIEFING" in chunks_evening[0]
