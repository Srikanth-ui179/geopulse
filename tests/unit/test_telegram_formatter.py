"""Unit tests for Phase 5 Telegram HTML digest formatting and message splitting."""

from datetime import UTC, date, datetime

from geopulse.classification.models import TopicCategory
from geopulse.clustering.models import EventCluster
from geopulse.delivery.formatter import (
    TelegramDigestFormatter,
    escape_html,
    format_telegram_digest,
)
from geopulse.ingestion.models import ArticleRecord
from geopulse.summarization.models import (
    EventKeyDetails,
    EventSourceAttribution,
    EventSummary,
    SummarizationResult,
)


def _make_dummy_result(
    headline: str,
    category: TopicCategory = TopicCategory.ARMED_CONFLICT,
    confidence_label: str = "corroborated",
    significance: str | None = "High strategic impact.",
    actors: tuple[str, ...] = ("Country A", "Country B"),
    location: str | None = "Region X",
    timing: str | None = "October 2026",
    action: str | None = "Missile strike",
) -> SummarizationResult:
    now_dt = datetime(2026, 10, 23, 10, 0, tzinfo=UTC)
    art1 = ArticleRecord(
        source_name="Defense News",
        title="Missile strike reported",
        url="https://defensenews.com/test-1",
        canonical_url="https://defensenews.com/test-1",
        published_at=now_dt,
        content_hash="h1",
    )
    art2 = ArticleRecord(
        source_name="BBC World News",
        title="Joint strike in Region X",
        url="https://bbc.com/test-2",
        canonical_url="https://bbc.com/test-2",
        published_at=now_dt,
        content_hash="h2",
    )
    evt = EventCluster(
        id="evt-1",
        title=headline,
        category=category,
        region=location,
        first_seen_at=now_dt,
        last_seen_at=now_dt,
        confidence_label=confidence_label,
        articles=(art1, art2),
    )
    summary = EventSummary(
        event_id=evt.id,
        headline=headline,
        what_happened="Military forces carried out targeted operations against strategic assets.",
        key_details=EventKeyDetails(
            actors=actors,
            location=location,
            timing=timing,
            action=action,
        ),
        significance=significance,
        confidence_qualification=(
            "Corroborated across 2 reporting outlets; military claims unconfirmed independently."
        ),
        sources=(
            EventSourceAttribution(
                source_name=art1.source_name,
                title=art1.title,
                url=art1.url,
            ),
            EventSourceAttribution(
                source_name=art2.source_name,
                title=art2.title,
                url=art2.url,
            ),
        ),
        model_used="test-model",
    )
    return SummarizationResult(
        event_id=evt.id,
        status="success",
        summary=summary,
        event=evt,
    )


def test_escape_html_handles_special_characters():
    assert escape_html("AT&T & Lockheed <Martin>") == "AT&amp;T &amp; Lockheed &lt;Martin&gt;"
    escaped_tomahawk = escape_html('Contract for "Tomahawk" missiles')
    assert escaped_tomahawk == "Contract for &quot;Tomahawk&quot; missiles"
    assert escape_html("") == ""
    assert escape_html(None) == ""


def test_format_event_block_includes_all_key_sections():
    res = _make_dummy_result(
        headline="US and UK Launch Strikes on Radar Sites",
        category=TopicCategory.ARMED_CONFLICT,
        confidence_label="corroborated",
        significance="Degrades regional missile capabilities.",
    )
    formatter = TelegramDigestFormatter()
    block = formatter.format_event_block(
        summary=res.summary,
        category=res.event.category,
        confidence_label=res.event.confidence_label,
        rank=1,
        total=1,
    )

    assert "<b>[#1/1] ⚔️ ARMED CONFLICT · [CORROBORATED]</b>" in block
    assert "<b>US and UK Launch Strikes on Radar Sites</b>" in block
    assert "Military forces carried out targeted operations" in block
    assert "• <b>Actors:</b> Country A, Country B" in block
    assert "• <b>Location:</b> Region X" in block
    assert "<b>Significance:</b> Degrades regional missile capabilities." in block
    assert "<b>Assessment:</b> Corroborated across 2 reporting outlets" in block
    expected_src_1 = (
        '• <a href="https://defensenews.com/test-1">Defense News</a>: Missile strike reported'
    )
    assert expected_src_1 in block
    expected_src_2 = (
        '• <a href="https://bbc.com/test-2">BBC World News</a>: Joint strike in Region X'
    )
    assert expected_src_2 in block


def test_format_event_block_omits_empty_or_unspecified_optional_fields():
    res = _make_dummy_result(
        headline="Unspecified Border Event",
        category=TopicCategory.SECURITY_INCIDENT,
        confidence_label="single_source",
        significance=None,
        actors=(),
        location="Unspecified in reporting",
        timing=None,
        action="Unspecified",
    )
    formatter = TelegramDigestFormatter()
    block = formatter.format_event_block(
        summary=res.summary,
        category=res.event.category,
        confidence_label=res.event.confidence_label,
        rank=1,
        total=5,
    )

    # Missing/unspecified fields must not appear
    assert "Significance:" not in block
    assert "Key Details:" not in block
    assert "Location:" not in block
    assert "Timing:" not in block
    assert "null" not in block
    assert "None" not in block


def test_format_digest_empty_results_returns_clean_monitoring_message():
    formatter = TelegramDigestFormatter()
    chunks = formatter.format_digest([], briefing_date=date(2026, 10, 23))

    assert len(chunks) == 1
    assert "GEOPULSE DAILY BRIEFING" in chunks[0]
    assert "October 23, 2026" in chunks[0]
    expected_sub = "No corroborated or verified geopolitical events met briefing thresholds today"
    assert expected_sub in chunks[0]


def test_format_digest_single_chunk_when_within_limit():
    res = _make_dummy_result(headline="Single Event Briefing")
    formatter = TelegramDigestFormatter(safe_limit=4000)
    chunks = formatter.format_digest([res], briefing_date=date(2026, 10, 23))

    assert len(chunks) == 1
    assert "<b>🌍 GEOPULSE DAILY INTELLIGENCE BRIEFING</b>" in chunks[0]
    assert "Part" not in chunks[0]
    assert "Geopulse Intelligence • Source-grounded briefing" in chunks[0]


def test_format_digest_splits_across_events_when_exceeding_limit():
    # Create multiple events that together exceed a tight limit of 800 characters
    events = [
        _make_dummy_result(headline=f"Major Geopolitical Event Number {i}")
        for i in range(1, 6)
    ]
    formatter = TelegramDigestFormatter(safe_limit=800)
    chunks = formatter.format_digest(events, briefing_date=date(2026, 10, 23))

    assert len(chunks) > 1
    # Check part headers
    assert "<b>🌍 GEOPULSE BRIEFING (Part 1/" in chunks[0]
    assert f"<b>🌍 GEOPULSE BRIEFING (Part {len(chunks)}/{len(chunks)})</b>" in chunks[-1]
    # Check that footer is attached only to the final chunk
    assert "Geopulse Intelligence • Source-grounded briefing" in chunks[-1]
    assert "Geopulse Intelligence • Source-grounded briefing" not in chunks[0]

    # Every chunk must have properly closed HTML tags
    for chunk in chunks:
        assert chunk.count("<b>") == chunk.count("</b>")
        assert chunk.count("<i>") == chunk.count("</i>")
        assert chunk.count("<a") == chunk.count("</a>")


def test_convenience_helper_format_telegram_digest():
    res = _make_dummy_result(headline="Convenience Helper Test")
    chunks = format_telegram_digest([res])
    assert len(chunks) == 1
    assert "Convenience Helper Test" in chunks[0]


def test_format_event_block_with_analysis_label_and_publication_date():
    res = _make_dummy_result(
        headline="Strategic Review of Microchip Industry Policy",
        category=TopicCategory.DEFENCE_POLICY,
        confidence_label="single_source",
    )
    formatter = TelegramDigestFormatter()
    block = formatter.format_event_block(
        summary=res.summary,
        category=res.event.category,
        confidence_label=res.event.confidence_label,
        rank=1,
        total=1,
        is_analysis=True,
        published_at=datetime(2026, 10, 22, 8, 0, tzinfo=UTC),
    )

    assert "<b>[#1/1] 🏛️ DEFENCE POLICY · [ANALYSIS] · [SINGLE SOURCE]</b>" in block
    assert "• <b>Published:</b> October 22, 2026" in block


def test_format_event_block_with_analysis_label_and_missing_publication_date():
    res = _make_dummy_result(
        headline="Strategic Review of Microchip Industry Policy",
        category=TopicCategory.DEFENCE_POLICY,
        confidence_label="single_source",
    )
    formatter = TelegramDigestFormatter()
    block = formatter.format_event_block(
        summary=res.summary,
        category=res.event.category,
        confidence_label=res.event.confidence_label,
        rank=1,
        total=1,
        is_analysis=True,
        published_at=None,
    )

    assert "<b>[#1/1] 🏛️ DEFENCE POLICY · [ANALYSIS] · [SINGLE SOURCE]</b>" in block
    assert "• <b>Published:</b> Date not specified in source" in block


def test_format_digest_deprioritizes_analysis_behind_operational_events():
    res_analysis = _make_dummy_result(
        headline="Strategic Review of Microchip Industry Policy",
        category=TopicCategory.DEFENCE_POLICY,
        confidence_label="single_source",
    )
    object.__setattr__(res_analysis.event, "is_analysis", True)

    res_operational = _make_dummy_result(
        headline="Missile Strikes Destroy Radar Installation",
        category=TopicCategory.ARMED_CONFLICT,
        confidence_label="corroborated",
    )
    object.__setattr__(res_operational.event, "is_analysis", False)

    formatter = TelegramDigestFormatter()
    # Pass analysis first in list
    chunks = formatter.format_digest(
        [res_analysis, res_operational], briefing_date=date(2026, 10, 23)
    )

    assert len(chunks) == 1
    assert "[#1/2] ⚔️ ARMED CONFLICT" in chunks[0]
    assert "Missile Strikes Destroy Radar Installation" in chunks[0]
    assert "[#2/2] 🏛️ DEFENCE POLICY · [ANALYSIS]" in chunks[0]
    assert "Strategic Review of Microchip Industry Policy" in chunks[0]

    pos_op = chunks[0].find("[#1/2]")
    pos_an = chunks[0].find("[#2/2]")
    assert pos_op < pos_an


