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
    ClaimEvidenceReference,
    EventKeyDetails,
    EventSourceAttribution,
    EventSummary,
    SummarizationResult,
    SummaryClaim,
    reconstruct_event_summary,
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
    is_analysis: bool = False,
    published_at: datetime | None = None,
    confidence_qualification: str | None = None,
    sources: tuple[EventSourceAttribution, ...] | None = None,
    corroboration_status: str | None = None,
) -> SummarizationResult:
    now_dt = published_at or datetime(2026, 10, 23, 10, 0, tzinfo=UTC)
    if corroboration_status is None:
        if confidence_label in ("corroborated", "independently_corroborated"):
            c_status = "independently_corroborated"
        elif confidence_label == "single_source":
            c_status = "single_source"
        elif confidence_label == "disputed":
            c_status = "conflicting_evidence"
        elif confidence_label in ("uncertain", "insufficient_information"):
            c_status = "insufficient_information"
        else:
            c_status = "unassessed"
    else:
        c_status = corroboration_status
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
        id=f"evt-{headline[:12].strip().replace(' ', '-').lower()}",
        title=headline,
        category=category,
        region=location,
        first_seen_at=now_dt,
        last_seen_at=now_dt,
        published_at=now_dt,
        is_analysis=is_analysis,
        confidence_label=confidence_label,
        articles=(art1, art2),
        corroboration_status=c_status,
        independent_chains_count=2
        if c_status == "independently_corroborated"
        else (1 if c_status == "single_source" else 0),
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
        confidence_qualification=confidence_qualification
        or ("Corroborated across 2 reporting outlets; military claims unconfirmed independently."),
        sources=sources
        if sources is not None
        else (
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
        corroboration_status=c_status,
        independent_chains_count=2
        if c_status == "independently_corroborated"
        else (1 if c_status == "single_source" else 0),
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
        _make_dummy_result(headline=f"Major Geopolitical Event Number {i}") for i in range(1, 6)
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


def test_format_digest_prioritizes_diplomacy_and_policy_over_weapons_procurement():
    """Consequential diplomacy and defence policy must rank ahead of weapons procurement."""
    res_procurement = _make_dummy_result(
        headline="Cyprus Acquires Patrol Vessel with Advanced Naval Systems",
        category=TopicCategory.WEAPONS_PROCUREMENT,
        confidence_label="single_source",
    )
    res_diplomacy = _make_dummy_result(
        headline="Zelenskyy Announces Peace Talks with Russia to Restart",
        category=TopicCategory.DIPLOMACY,
        confidence_label="single_source",
    )
    res_policy = _make_dummy_result(
        headline="E5 Defence Ministers Issue Joint Statement on European Security",
        category=TopicCategory.DEFENCE_POLICY,
        confidence_label="single_source",
    )

    formatter = TelegramDigestFormatter()
    # Pass procurement first in input list
    chunks = formatter.format_digest(
        [res_procurement, res_diplomacy, res_policy], briefing_date=date(2026, 10, 26)
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    # Diplomacy and Defence Policy must appear before Weapons Procurement
    pos_diplo = chunk.find("🕊️ DIPLOMACY")
    pos_policy = chunk.find("🏛️ DEFENCE POLICY")
    pos_proc = chunk.find("🚀 WEAPONS PROCUREMENT")

    assert pos_diplo != -1 and pos_policy != -1 and pos_proc != -1
    assert pos_diplo < pos_proc
    assert pos_policy < pos_proc


def test_format_digest_prioritizes_corroborated_over_single_source_in_same_tier():
    """Multi-source corroborated events must rank ahead of single-source reports in same tier."""
    res_single = _make_dummy_result(
        headline="Border Patrol Unit Encounters Surveillance Drone",
        category=TopicCategory.MILITARY_OPERATIONS,
        confidence_label="single_source",
    )
    res_corrob = _make_dummy_result(
        headline="Allied Forces Intercept Unmanned Aerial Vehicles",
        category=TopicCategory.MILITARY_OPERATIONS,
        confidence_label="corroborated",
    )

    formatter = TelegramDigestFormatter()
    chunks = formatter.format_digest([res_single, res_corrob], briefing_date=date(2026, 10, 26))

    assert len(chunks) == 1
    pos_corrob = chunks[0].find("[#1/2]")
    pos_single = chunks[0].find("[#2/2]")
    assert "Allied Forces Intercept" in chunks[0][pos_corrob:pos_single]
    assert "Border Patrol Unit" in chunks[0][pos_single:]


def test_format_event_block_separates_event_timing_from_publication_date():
    """Displays both event timing and publication date when event date precedes publication."""
    res = _make_dummy_result(
        headline="Joint Naval Maneuvers Conclude in the Baltic Sea",
        category=TopicCategory.MILITARY_OPERATIONS,
        confidence_label="corroborated",
        timing="October 24, 2026",
    )
    formatter = TelegramDigestFormatter()
    block = formatter.format_event_block(
        summary=res.summary,
        category=res.event.category,
        confidence_label=res.event.confidence_label,
        rank=1,
        total=1,
        is_analysis=False,
        published_at=datetime(2026, 10, 26, 12, 0, tzinfo=UTC),
    )

    assert "• <b>Timing:</b> October 24, 2026" in block
    assert "• <b>Published:</b> October 26, 2026" in block


def test_ranking_armed_conflict_vs_procurement_commentary():
    """Operational news leads over analysis; procurement operational news leads over conflict."""
    formatter = TelegramDigestFormatter()

    res_conflict_op = _make_dummy_result(
        headline="Precision strike damages coastal radar installation",
        category=TopicCategory.ARMED_CONFLICT,
        is_analysis=False,
    )
    res_procurement_analysis = _make_dummy_result(
        headline="Strategic assessment of fifth-generation fighter procurement delays",
        category=TopicCategory.WEAPONS_PROCUREMENT,
        is_analysis=True,
    )

    # Operational armed conflict must precede procurement analysis
    chunks = formatter.format_digest([res_procurement_analysis, res_conflict_op])
    pos_conflict = chunks[0].find("Precision strike damages")
    pos_proc = chunks[0].find("Strategic assessment of fifth-generation")
    assert pos_conflict != -1 and pos_proc != -1
    assert pos_conflict < pos_proc

    # Now reverse: procurement operational reporting vs armed conflict commentary
    res_proc_op = _make_dummy_result(
        headline="Air force signs contract for new radar system",
        category=TopicCategory.WEAPONS_PROCUREMENT,
        is_analysis=False,
    )
    res_conflict_analysis = _make_dummy_result(
        headline="Essay on changing doctrines in modern urban warfare",
        category=TopicCategory.ARMED_CONFLICT,
        is_analysis=True,
    )
    chunks2 = formatter.format_digest([res_conflict_analysis, res_proc_op])
    pos_proc2 = chunks2[0].find("Air force signs contract")
    pos_conflict2 = chunks2[0].find("Essay on changing doctrines")
    assert pos_proc2 != -1 and pos_conflict2 != -1
    assert pos_proc2 < pos_conflict2, "Operational news must precede commentary"


def test_ranking_consequential_diplomacy_vs_routine_procurement():
    """Consequential diplomacy ranks ahead of procurement; conflict ranks ahead of diplomacy."""
    formatter = TelegramDigestFormatter()

    res_conflict = _make_dummy_result(
        headline="Hostile artillery barrage targets border command post",
        category=TopicCategory.ARMED_CONFLICT,
        is_analysis=False,
    )
    res_diplomacy = _make_dummy_result(
        headline="Zelenskyy announces trilateral peace talks to resume",
        category=TopicCategory.DIPLOMACY,
        is_analysis=False,
    )
    res_procurement = _make_dummy_result(
        headline="Ministry of Defence awards ammunition contract",
        category=TopicCategory.WEAPONS_PROCUREMENT,
        is_analysis=False,
    )

    chunks = formatter.format_digest([res_procurement, res_diplomacy, res_conflict])
    pos_conflict = chunks[0].find("Hostile artillery barrage")
    pos_diplo = chunks[0].find("Zelenskyy announces trilateral")
    pos_proc = chunks[0].find("Ministry of Defence awards")

    assert pos_conflict != -1 and pos_diplo != -1 and pos_proc != -1
    # Armed conflict (#1) precedes diplomacy (#2), which precedes procurement (#3)
    assert pos_conflict < pos_diplo
    assert pos_diplo < pos_proc


def test_ranking_reporting_vs_opinion_when_categories_differ():
    """Operational news in lower categories always precedes analysis in higher categories."""
    formatter = TelegramDigestFormatter()

    res_sanctions_op = _make_dummy_result(
        headline="Treasury announces new sanctions on shadow tanker fleet",
        category=TopicCategory.SANCTIONS,
        is_analysis=False,
    )
    res_ops_opinion = _make_dummy_result(
        headline="Opinion: Modern military logistics requires deep reform",
        category=TopicCategory.MILITARY_OPERATIONS,
        is_analysis=True,
    )

    chunks = formatter.format_digest([res_ops_opinion, res_sanctions_op])
    pos_sanctions = chunks[0].find("Treasury announces new sanctions")
    pos_opinion = chunks[0].find("Opinion: Modern military logistics")

    assert pos_sanctions != -1 and pos_opinion != -1
    assert pos_sanctions < pos_opinion


def test_ranking_stable_deterministic_ordering_for_ties():
    """Ties across all priority fields resolve deterministically by title."""
    formatter = TelegramDigestFormatter()
    fixed_dt = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)

    res_alpha = _make_dummy_result(
        headline="Alpha forces execute reconnaissance patrol",
        category=TopicCategory.MILITARY_OPERATIONS,
        confidence_label="single_source",
        published_at=fixed_dt,
    )
    res_bravo = _make_dummy_result(
        headline="Bravo forces execute reconnaissance patrol",
        category=TopicCategory.MILITARY_OPERATIONS,
        confidence_label="single_source",
        published_at=fixed_dt,
    )

    # Pass in [bravo, alpha] order
    chunks1 = formatter.format_digest([res_bravo, res_alpha])
    pos_alpha1 = chunks1[0].find("Alpha forces")
    pos_bravo1 = chunks1[0].find("Bravo forces")

    # Pass in [alpha, bravo] order
    chunks2 = formatter.format_digest([res_alpha, res_bravo])
    pos_alpha2 = chunks2[0].find("Alpha forces")
    pos_bravo2 = chunks2[0].find("Bravo forces")

    # Both must produce identical deterministic ordering (Alpha < Bravo alphabetically)
    assert pos_alpha1 < pos_bravo1
    assert pos_alpha2 < pos_bravo2


def test_format_digest_with_max_events_cap():
    """When max_events cap is set, briefing includes top events and appends editorial note."""
    formatter = TelegramDigestFormatter()

    res1 = _make_dummy_result(headline="Event One", category=TopicCategory.ARMED_CONFLICT)
    res2 = _make_dummy_result(headline="Event Two", category=TopicCategory.MILITARY_OPERATIONS)
    res3 = _make_dummy_result(headline="Event Three", category=TopicCategory.WEAPONS_PROCUREMENT)

    chunks = formatter.format_digest([res1, res2, res3], max_events=2)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert "Event One" in chunk
    assert "Event Two" in chunk
    assert "Event Three" not in chunk
    assert "Editorial Note: Briefing capped at top 2 events" in chunk
    assert "1 lower-priority event(s) omitted" in chunk


def test_consequential_diplomacy_survives_cap_over_lower_priority_items():
    """Consequential diplomacy (peace talks, ceasefires) ranks Tier 1 and survives the cap."""
    formatter = TelegramDigestFormatter()
    fixed_dt = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)

    # Consequential diplomacy event
    res_dip = _make_dummy_result(
        headline="Envoys declare tripartite peace talks and ceasefire negotiations to restart",
        category=TopicCategory.DIPLOMACY,
        confidence_label="single_source",
        published_at=fixed_dt,
    )
    # Routine military operations and procurement events
    res_ops = _make_dummy_result(
        headline="Routine patrol conducted across northern border sector",
        category=TopicCategory.MILITARY_OPERATIONS,
        confidence_label="corroborated",
        published_at=fixed_dt,
    )
    res_proc = _make_dummy_result(
        headline="Military signs contract for radar components",
        category=TopicCategory.WEAPONS_PROCUREMENT,
        confidence_label="corroborated",
        published_at=fixed_dt,
    )
    res_policy = _make_dummy_result(
        headline="Ministry discusses internal organizational restructuring",
        category=TopicCategory.DEFENCE_POLICY,
        confidence_label="corroborated",
        published_at=fixed_dt,
    )

    # With cap=2, diplomacy and armed/ops should survive over procurement & internal policy
    chunks = formatter.format_digest([res_policy, res_proc, res_ops, res_dip], max_events=2)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert "peace talks and ceasefire" in chunk
    assert "contract for radar components" not in chunk


def test_historical_commemorations_ranked_below_modern_news():
    """Historical soldier burials and centenary commemorations rank below current defence news."""
    formatter = TelegramDigestFormatter()
    fixed_dt = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)

    res_modern_proc = _make_dummy_result(
        headline="Air force acquires advanced long-range interceptor missiles",
        category=TopicCategory.WEAPONS_PROCUREMENT,
        published_at=fixed_dt,
    )
    res_hist_burial = _make_dummy_result(
        headline="First World War soldier buried with military honours after remains identified",
        category=TopicCategory.OTHER_GEOPOLITICAL,
        published_at=fixed_dt,
    )

    chunks = formatter.format_digest(
        [res_hist_burial, res_modern_proc],
        max_events=2,
        include_low_priority=True,
    )
    pos_modern = chunks[0].find("advanced long-range interceptor")
    pos_hist = chunks[0].find("First World War soldier buried")

    assert pos_modern != -1 and pos_hist != -1
    assert pos_modern < pos_hist


def test_confidence_label_distinctions_and_no_dispute_conflation():
    """Verify single source, disputed, and corroborated badges and assessments are distinct."""
    formatter = TelegramDigestFormatter()

    # 1. Single source with vague LLM dispute wording should be normalized without dispute badge
    res_single = _make_dummy_result(
        headline="Single reconnaissance aircraft spotted over sea",
        confidence_label="single_source",
        confidence_qualification="Single source report; disputed by some.",
    )
    assert res_single.summary is not None
    block_single = formatter.format_event_block(
        summary=res_single.summary,
        category=TopicCategory.MILITARY_OPERATIONS,
        confidence_label="single_source",
        rank=1,
        total=1,
    )
    assert "[SINGLE SOURCE]" in block_single
    assert "[DISPUTED]" not in block_single
    assert "Single-source reporting; unverified independently." in block_single

    # 2. Genuine dispute: single-source assessment wording on disputed event is guardrailed
    res_disputed = _make_dummy_result(
        headline="Army claims victory while opposing rebels deny casualty reports",
        confidence_label="disputed",
        confidence_qualification="Single-source report.",
    )
    assert res_disputed.summary is not None
    block_disputed = formatter.format_event_block(
        summary=res_disputed.summary,
        category=TopicCategory.ARMED_CONFLICT,
        confidence_label="disputed",
        rank=1,
        total=1,
    )
    assert "[DISPUTED]" in block_disputed
    assert "Contested claims; opposing parties provide conflicting accounts or denials." in (
        block_disputed
    )


def test_nuclear_exercise_corroboration_distinguishes_official_announcements():
    """Corroboration involving official releases states announcement coverage, not ground truth."""
    formatter = TelegramDigestFormatter()

    summary = EventSummary(
        event_id="evt-nuclear-ex",
        headline="France and UK conduct joint nuclear deterrent drill",
        what_happened="Joint maritime deterrent exercises conducted in Atlantic waters.",
        key_details=EventKeyDetails(
            actors=("France", "UK"),
            location="Atlantic",
            timing="October 2026",
            action="Nuclear drill",
        ),
        confidence_qualification="Corroborated across two news sources.",
        sources=(
            EventSourceAttribution(
                source_name="UK Ministry of Defence",
                title="Joint deterrence exercise successfully completed",
                url="https://gov.uk/mod/exercise",
            ),
            EventSourceAttribution(
                source_name="Breaking Defense",
                title="UK and France exercise joint deterrence operations",
                url="https://breakingdefense.com/exercise",
            ),
        ),
        model_used="test-model",
        corroboration_status="multiple_reports_shared_origin",
        independent_chains_count=1,
    )
    block = formatter.format_event_block(
        summary=summary,
        category=TopicCategory.MILITARY_OPERATIONS,
        confidence_label="corroborated",
        rank=1,
        total=1,
    )
    assert "[OFFICIAL ANNOUNCEMENT]" in block
    assert "based on official announcement" in block
    assert "military details not independently verified" in block


def test_editorial_diversity_prevents_single_category_or_source_monopoly():
    """Secondary categories and sources are capped to ensure editorial variety."""
    formatter = TelegramDigestFormatter()
    fixed_dt = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)

    # 5 Defence Policy stories from Breaking Defense
    bd_results = []
    for i in range(5):
        bd_src = (
            EventSourceAttribution(
                source_name="Breaking Defense",
                title=f"Article {i + 1}",
                url=f"https://breakingdefense.com/art-{i + 1}",
            ),
        )
        res = _make_dummy_result(
            headline=f"Defence Policy update number {i + 1}",
            category=TopicCategory.DEFENCE_POLICY,
            confidence_label="corroborated",
            published_at=fixed_dt,
            sources=bd_src,
        )
        bd_results.append(res)

    # 1 Consequential diplomacy story from Al Jazeera
    dip_src = (
        EventSourceAttribution(
            source_name="Al Jazeera",
            title="Peace talks restart announcement",
            url="https://aljazeera.com/peace",
        ),
    )
    res_dip = _make_dummy_result(
        headline="Zelenskyy says peace talks with Russia and US to restart",
        category=TopicCategory.DIPLOMACY,
        confidence_label="single_source",
        published_at=fixed_dt,
        sources=dip_src,
    )

    # 2 Cyber items from Reuters
    cyber_results = []
    for i in range(2):
        cyber_src = (
            EventSourceAttribution(
                source_name="Reuters",
                title=f"Cyber article {i + 1}",
                url=f"https://reuters.com/cyber-{i + 1}",
            ),
        )
        res = _make_dummy_result(
            headline=f"Critical infrastructure cyber incident {i + 1}",
            category=TopicCategory.CYBER_HYBRID_WARFARE,
            confidence_label="corroborated",
            published_at=fixed_dt,
            sources=cyber_src,
        )
        cyber_results.append(res)

    all_candidates = bd_results + [res_dip] + cyber_results  # total 8 items
    chunks = formatter.format_digest(all_candidates, max_events=5)
    digest_text = chunks[0]

    # Consequential diplomacy is Tier 1 and MUST be included
    assert "peace talks with Russia and US" in digest_text
    # Both cyber items should be included because Breaking Defense was capped at 3
    assert "cyber incident 1" in digest_text
    assert "cyber incident 2" in digest_text
    # Exactly 3 lower-priority items omitted from the total 8
    assert "capped at top 5 events" in digest_text
    assert "3 lower-priority event(s) omitted" in digest_text


def test_briefing_with_fewer_than_ten_relevant_events_not_padded():
    """Briefing with fewer than 10 events delivers only suitable items without padding."""
    formatter = TelegramDigestFormatter()
    fixed_dt = datetime(2026, 10, 27, 8, 0, tzinfo=UTC)

    events = [
        _make_dummy_result(
            headline=f"Operational defence event number {i}",
            category=TopicCategory.MILITARY_OPERATIONS,
            published_at=fixed_dt,
        )
        for i in range(1, 7)
    ]

    chunks = formatter.format_digest(events, max_events=10, edition="morning")
    digest_text = "\n\n".join(chunks)

    # Verify exactly 6 events briefed and block tags reflect 6
    assert "[#1/6]" in digest_text
    assert "[#6/6]" in digest_text
    assert "[#7/" not in digest_text
    # No items were omitted since all 6 met relevance threshold and were within cap 10
    assert "omitted" not in digest_text


def test_low_relevance_historical_commemorations_excluded_by_default():
    """Historical soldier burials and centenary commemorations are excluded by default."""
    formatter = TelegramDigestFormatter()
    fixed_dt = datetime(2026, 10, 27, 8, 0, tzinfo=UTC)

    relevant_events = [
        _make_dummy_result(
            headline=f"Operational military news item {i}",
            category=TopicCategory.MILITARY_OPERATIONS,
            published_at=fixed_dt,
        )
        for i in range(1, 7)
    ]

    commem_events = [
        _make_dummy_result(
            headline="Somme battlefield burial of First World War soldier after remains identified",
            category=TopicCategory.OTHER_GEOPOLITICAL,
            published_at=fixed_dt,
        ),
        _make_dummy_result(
            headline="CWGC reinterment ceremony for Great War soldier on the Western Front",
            category=TopicCategory.OTHER_GEOPOLITICAL,
            published_at=fixed_dt,
        ),
    ]

    all_events = relevant_events + commem_events  # 6 relevant + 2 commemorations
    chunks = formatter.format_digest(all_events, max_events=10, edition="morning")
    digest_text = "\n\n".join(chunks)

    # Commemorations must not pad the briefing to 8
    assert "Somme battlefield burial" not in digest_text
    assert "CWGC reinterment ceremony" not in digest_text
    # Exactly 6 suitable events delivered
    assert "[#1/6]" in digest_text
    assert "[#6/6]" in digest_text
    assert "[#7/" not in digest_text
    # Omission note accounts for the 2 excluded low-priority events
    assert "2 lower-priority event(s) omitted" in digest_text


def test_explicit_override_allows_low_priority_items():
    """Explicit override --include-low-priority permits commemorations to backfill digest."""
    formatter = TelegramDigestFormatter()
    fixed_dt = datetime(2026, 10, 27, 8, 0, tzinfo=UTC)

    relevant_events = [
        _make_dummy_result(
            headline=f"Operational military news item {i}",
            category=TopicCategory.MILITARY_OPERATIONS,
            published_at=fixed_dt,
        )
        for i in range(1, 5)
    ]

    commem_events = [
        _make_dummy_result(
            headline="Somme battlefield burial of First World War soldier after remains identified",
            category=TopicCategory.OTHER_GEOPOLITICAL,
            published_at=fixed_dt,
        ),
        _make_dummy_result(
            headline="CWGC reinterment ceremony for Great War soldier on the Western Front",
            category=TopicCategory.OTHER_GEOPOLITICAL,
            published_at=fixed_dt,
        ),
    ]

    all_events = relevant_events + commem_events  # 4 relevant + 2 commemorations
    chunks = formatter.format_digest(
        all_events,
        max_events=10,
        include_low_priority=True,
        edition="morning",
    )
    digest_text = "\n\n".join(chunks)

    # With override, commemorations are permitted to backfill
    assert "Somme battlefield burial" in digest_text
    assert "CWGC reinterment ceremony" in digest_text
    assert "[#1/6]" in digest_text
    assert "[#6/6]" in digest_text


def test_strategic_analysis_with_historical_comparison_preserved():
    """Strategic analysis referencing past wars is preserved and not mistaken for commemoration."""
    formatter = TelegramDigestFormatter()
    fixed_dt = datetime(2026, 10, 27, 8, 0, tzinfo=UTC)

    analysis_event = _make_dummy_result(
        headline=(
            "Strategic analysis: Lessons of World War II naval doctrine for Pacific deterrence"
        ),
        category=TopicCategory.DEFENCE_POLICY,
        is_analysis=True,
        published_at=fixed_dt,
    )
    ops_event = _make_dummy_result(
        headline="Naval strike group conducts deterrence patrol in international waters",
        category=TopicCategory.MILITARY_OPERATIONS,
        published_at=fixed_dt,
    )

    chunks = formatter.format_digest([analysis_event, ops_event], max_events=10)
    digest_text = chunks[0]

    # Both must be present; analysis is not excluded by commemoration filter
    assert "Lessons of World War II naval doctrine" in digest_text
    assert "Naval strike group conducts deterrence patrol" in digest_text
    assert "2 event(s) briefed" in digest_text


def test_armed_conflict_not_crowded_out_by_procurement_and_policy():
    """Verify armed conflict developments are not crowded out by procurement/policy."""
    formatter = TelegramDigestFormatter()
    fixed_dt = datetime(2026, 10, 27, 8, 0, tzinfo=UTC)

    # 2 Armed conflict events (Tier 1)
    ac1 = _make_dummy_result(
        headline="Airstrikes target opposing artillery batteries across frontline",
        category=TopicCategory.ARMED_CONFLICT,
        published_at=fixed_dt,
        sources=(
            EventSourceAttribution(
                source_name="BBC", title="Airstrikes hit frontline", url="https://bbc.com/ac1"
            ),
        ),
    )
    ac2 = _make_dummy_result(
        headline="Artillery bombardment damages military command outpost",
        category=TopicCategory.ARMED_CONFLICT,
        published_at=fixed_dt,
        sources=(
            EventSourceAttribution(
                source_name="Al Jazeera", title="Artillery outpost hit", url="https://aj.com/ac2"
            ),
        ),
    )

    # 3 Procurement items
    proc_items = [
        _make_dummy_result(
            headline=f"Ministry awards procurement contract #{i}",
            category=TopicCategory.WEAPONS_PROCUREMENT,
            published_at=fixed_dt,
            sources=(
                EventSourceAttribution(
                    source_name=f"Defense News {i}", title=f"Proc {i}", url=f"https://dn.com/p{i}"
                ),
            ),
        )
        for i in range(1, 4)
    ]

    # 3 Defence policy items
    policy_items = [
        _make_dummy_result(
            headline=f"Government releases annual defence policy review #{i}",
            category=TopicCategory.DEFENCE_POLICY,
            published_at=fixed_dt,
            sources=(
                EventSourceAttribution(
                    source_name=f"Breaking Defense {i}",
                    title=f"Policy {i}",
                    url=f"https://bd.com/pol{i}",
                ),
            ),
        )
        for i in range(1, 4)
    ]

    all_candidates = proc_items + policy_items + [ac1, ac2]  # Total 8 items
    chunks = formatter.format_digest(all_candidates, max_events=5)
    digest_text = chunks[0]

    # Armed conflict developments MUST both be included in the capped digest
    assert "Airstrikes target opposing artillery" in digest_text
    assert "Artillery bombardment damages military" in digest_text
    # Exactly 5 events briefed
    assert "5 event(s) briefed" in digest_text
    assert "3 lower-priority event(s) omitted" in digest_text


def test_raw_slug_expanded_to_human_readable_assessment():
    """Raw machine confidence slug 'single_source' is expanded to a complete human sentence."""
    formatter = TelegramDigestFormatter()
    fixed_dt = datetime(2026, 10, 27, 8, 0, tzinfo=UTC)
    res = _make_dummy_result(
        headline="Cross-border artillery shelling strikes electrical substation",
        category=TopicCategory.ARMED_CONFLICT,
        published_at=fixed_dt,
        confidence_qualification="single_source",
        confidence_label="single_source",
    )
    rendered = formatter.format_event_block(
        summary=res.summary,
        category=res.event.category,
        confidence_label=res.event.confidence_label,
        rank=1,
        total=1,
    )
    assert "<b>Assessment:</b> Single-source reporting; unverified independently." in rendered
    assert "<b>Assessment:</b> single_source" not in rendered


def test_digest_omits_weak_candidates_rather_than_padding_to_ten():
    """Digest omits weak candidates when fewer than 10 qualify, avoiding artificial padding."""
    formatter = TelegramDigestFormatter()
    fixed_dt = datetime(2026, 10, 27, 8, 0, tzinfo=UTC)

    # 4 Armed conflict items (Tier 1)
    conflict_items = [
        _make_dummy_result(
            headline=f"Armed conflict action #{i}",
            category=TopicCategory.ARMED_CONFLICT,
            published_at=fixed_dt,
            sources=(
                EventSourceAttribution(
                    source_name=f"Outlet {i}", title=f"Title {i}", url=f"https://out.com/{i}"
                ),
            ),
        )
        for i in range(1, 5)
    ]

    # 6 Secondary policy items from same outlet (capped at 2 in pass 1; not forced in pass 2)
    policy_items = [
        _make_dummy_result(
            headline=f"Routine administrative defence policy memo #{i}",
            category=TopicCategory.DEFENCE_POLICY,
            published_at=fixed_dt,
            sources=(
                EventSourceAttribution(
                    source_name="Defense Outlet", title=f"Pol {i}", url=f"https://out.com/p{i}"
                ),
            ),
        )
        for i in range(1, 7)
    ]

    all_items = conflict_items + policy_items  # 10 items total
    chunks = formatter.format_digest(all_items, max_events=10, edition="morning")
    digest_text = "\n\n".join(chunks)

    # Exactly 6 events should be briefed (4 conflict + 2 policy cap), NOT padded to 10
    assert "6 event(s) briefed" in digest_text
    assert "4 lower-priority event(s) omitted" in digest_text


def test_official_announcement_detected_from_defense_department():
    """Multi-source report citing Defense Department is recognized as official announcement."""
    formatter = TelegramDigestFormatter()
    fixed_dt = datetime(2026, 10, 27, 8, 0, tzinfo=UTC)
    res = _make_dummy_result(
        headline="Defense Department selects Poland for new PAC-3 maintenance facility",
        category=TopicCategory.DEFENCE_POLICY,
        published_at=fixed_dt,
        confidence_label="corroborated",
        sources=(
            EventSourceAttribution(
                source_name="Breaking Defense",
                title="Defense Department selects Poland for new PAC-3 maintenance facility",
                url="https://bd.com/pac3",
            ),
            EventSourceAttribution(
                source_name="Defense News",
                title="US taps Poland to host European PAC-3 missile maintenance facility",
                url="https://dn.com/pac3",
            ),
        ),
    )
    rendered = formatter.format_event_block(
        summary=res.summary,
        category=res.event.category,
        confidence_label=res.event.confidence_label,
        rank=1,
        total=1,
    )
    assert "[OFFICIAL ANNOUNCEMENT]" in rendered
    assert "official announcement" in rendered.lower()


def test_retrospective_monthly_report_demoted_and_excluded_by_default():
    """Retrospective monthly reviews are assigned rank 10.5 and excluded by default."""
    from geopulse.delivery.formatter import compute_event_priority_rank, partition_candidate_events

    fixed_dt = datetime(2026, 10, 28, 8, 0, tzinfo=UTC)
    retrospective_event = _make_dummy_result(
        headline="How we protected the UK and space in September 2026",
        category=TopicCategory.DEFENCE_POLICY,
        published_at=fixed_dt,
    )
    operational_event = _make_dummy_result(
        headline="Air strikes conducted in southern border region",
        category=TopicCategory.ARMED_CONFLICT,
        published_at=fixed_dt,
    )

    rank = compute_event_priority_rank(retrospective_event)
    assert rank == 10.5

    selected, _ = partition_candidate_events(
        [operational_event, retrospective_event],
        max_events=10,
    )
    assert len(selected) == 1
    assert selected[0].summary.headline == "Air strikes conducted in southern border region"
    assert retrospective_event not in selected

    # With include_low_priority=True, it is admitted
    selected_override, _ = partition_candidate_events(
        [operational_event, retrospective_event],
        max_events=10,
        include_low_priority=True,
    )
    assert retrospective_event in selected_override


def test_routine_institutional_appointment_demoted_below_procurement():
    """Routine institutional / Director General appointments receive rank 8.5."""
    from geopulse.delivery.formatter import compute_event_priority_rank

    fixed_dt = datetime(2026, 10, 28, 8, 0, tzinfo=UTC)
    appointment_event = _make_dummy_result(
        headline="UK MOD appoints its first Director General Innovation",
        category=TopicCategory.DEFENCE_POLICY,
        published_at=fixed_dt,
    )
    rank = compute_event_priority_rank(appointment_event)
    assert rank == 8.5


def test_formatter_never_deletes_narrative_when_headline_equals_claim():
    """Single-claim summary where headline equals claim text retains body and links."""
    claim_text = "Canadian Armed Forces update Arctic defence deployment posture."
    claim = SummaryClaim(
        claim_id="c1",
        text=claim_text,
        evidence_refs=(
            ClaimEvidenceReference(
                source_url="https://example.com/arctic",
                evidence_span=claim_text,
                verification_status="accepted",
            ),
        ),
        classification="reported_fact",
        central=True,
        verification_status="accepted",
    )
    summary = EventSummary(
        event_id="evt-posture",
        headline=claim_text,
        what_happened=claim_text,
        confidence_qualification="Single-source reporting; unverified independently.",
        sources=(
            EventSourceAttribution(
                source_name="Defence Post",
                title="Canadian Armed Forces update Arctic defence deployment posture",
                url="https://example.com/arctic",
                published_at=datetime(2026, 10, 29, 12, 0, tzinfo=UTC),
            ),
        ),
        claims=(claim,),
        model_used="test-model",
    )
    formatter = TelegramDigestFormatter()
    rendered = formatter.format_event_block(
        summary=summary,
        category=TopicCategory.DEFENCE_POLICY,
        confidence_label="single_source",
        rank=1,
        total=1,
    )
    assert claim_text in rendered
    assert 'href="https://example.com/arctic"' in rendered
    assert "Defence Post" in rendered


def test_formatter_and_digest_selection_rejects_genuinely_invalid_empty_body():
    """An event with tag-only body without readable narrative is rejected from digest selection."""
    tag_only_summary = EventSummary(
        event_id="evt-empty",
        headline="Valid Headline Text",
        what_happened='<i>(<a href="https://example.com/link">Source</a>)</i>',
        confidence_qualification="Single-source",
        sources=(
            EventSourceAttribution(
                source_name="Source",
                title="Valid Headline Text",
                url="https://example.com/link",
            ),
        ),
        model_used="test-model",
    )
    event = EventCluster(
        title="Valid Headline Text",
        category=TopicCategory.DEFENCE_POLICY,
        first_seen_at=datetime(2026, 10, 29, 12, 0, tzinfo=UTC),
        last_seen_at=datetime(2026, 10, 29, 12, 0, tzinfo=UTC),
        confidence_label="single_source",
    )
    res = SummarizationResult(
        event_id="evt-empty",
        status="success",
        summary=tag_only_summary,
        event=event,
    )
    formatter = TelegramDigestFormatter()
    selected = formatter.select_digest_events([res])
    assert len(selected) == 0


# ===========================================================================
# Phase 3 Unassessed Status Integrity Regression Tests
# ===========================================================================
def test_unassessed_event_with_legacy_corroborated_and_five_publishers():
    """Unassessed event with legacy 'corroborated' label and 5 publishers must render
    [NOT ASSESSED]."""
    now = datetime(2026, 11, 4, 12, 0, tzinfo=UTC)
    sources = tuple(
        EventSourceAttribution(
            source_name=name,
            title=f"Report from {name}",
            url=f"https://example.com/{name.lower().replace(' ', '')}",
        )
        for name in ("BBC News", "CNN", "The Guardian", "Reuters", "Al Jazeera")
    )
    # Direct model construction
    summary = EventSummary(
        event_id="evt-unassessed-5",
        headline="Major Regional Diplomatic Summit Concludes",
        what_happened=(
            "Representatives discussed multilateral trade and maritime security protocols."
        ),
        confidence_qualification="unassessed",
        sources=sources,
        model_used="test-model",
        corroboration_status="unassessed",
        independent_chains_count=0,
    )
    event = EventCluster(
        title="Diplomatic Summit Concludes",
        category=TopicCategory.DIPLOMACY,
        confidence_label="corroborated",
        first_seen_at=now,
        last_seen_at=now,
        corroboration_status="unassessed",
        independent_chains_count=0,
    )
    formatter = TelegramDigestFormatter()
    block = formatter.format_event(
        summary=summary,
        event=event,
        category=event.category,
        confidence_label=event.confidence_label,
        rank=1,
        total=1,
    )
    assert "[NOT ASSESSED]" in block
    assert "[CORROBORATED]" not in block
    assert "[SINGLE SOURCE]" not in block
    assert "Unassessed; corroboration analysis not performed." in block

    # Database row reconstruction
    row = {
        "event_id": "evt-unassessed-5",
        "headline": "Major Regional Diplomatic Summit Concludes",
        "what_happened": "Representatives discussed multilateral trade protocols.",
        "confidence_qualification": "unassessed",
        "model_used": "legacy-model",
        "corroboration_status": "unassessed",
        "independent_chains_count": 0,
        "key_details": {
            "_evidence": {
                "sources": [
                    {"source_name": s.source_name, "title": s.title, "url": s.url} for s in sources
                ]
            }
        },
    }
    recon = reconstruct_event_summary(row)
    assert recon is not None
    assert recon.corroboration_status == "unassessed"
    assert recon.independent_chains_count == 0
    recon_block = formatter.format_event(
        summary=recon,
        event=event,
        category=event.category,
        confidence_label=event.confidence_label,
        rank=1,
        total=1,
    )
    assert "[NOT ASSESSED]" in recon_block
    assert "[CORROBORATED]" not in recon_block


def test_unassessed_event_with_legacy_single_source_and_one_publisher():
    """Unassessed event with legacy 'single_source' label and 1 publisher must render
    [NOT ASSESSED]."""
    now = datetime(2026, 11, 4, 12, 0, tzinfo=UTC)
    source = EventSourceAttribution(
        source_name="Financial Times",
        title="Sanctions Proposal Drafted",
        url="https://ft.com/sanctions-draft",
    )
    summary = EventSummary(
        event_id="evt-unassessed-1",
        headline="Sanctions Proposal Drafted for Industrial Chemicals",
        what_happened=(
            "Diplomats drafted an initial framework outlining potential export restrictions."
        ),
        confidence_qualification="unassessed",
        sources=(source,),
        model_used="test-model",
        corroboration_status="unassessed",
        independent_chains_count=0,
    )
    event = EventCluster(
        title="Sanctions Framework Drafted",
        category=TopicCategory.SANCTIONS,
        confidence_label="single_source",
        first_seen_at=now,
        last_seen_at=now,
        corroboration_status="unassessed",
        independent_chains_count=0,
    )
    formatter = TelegramDigestFormatter()
    block = formatter.format_event(
        summary=summary,
        event=event,
        category=event.category,
        confidence_label=event.confidence_label,
        rank=1,
        total=1,
    )
    assert "[NOT ASSESSED]" in block
    assert "[SINGLE SOURCE]" not in block

    # Database row reconstruction
    row = {
        "event_id": "evt-unassessed-1",
        "headline": "Sanctions Proposal Drafted for Industrial Chemicals",
        "what_happened": "Diplomats drafted an initial framework outlining restrictions.",
        "corroboration_status": "unassessed",
        "independent_chains_count": 0,
        "key_details": {
            "_evidence": {
                "sources": [
                    {"source_name": source.source_name, "title": source.title, "url": source.url}
                ]
            }
        },
    }
    recon = reconstruct_event_summary(row)
    assert recon is not None
    recon_block = formatter.format_event(
        summary=recon,
        event=event,
        category=event.category,
        confidence_label=event.confidence_label,
        rank=1,
        total=1,
    )
    assert "[NOT ASSESSED]" in recon_block
    assert "[SINGLE SOURCE]" not in recon_block


def test_unassessed_event_with_zero_publishers():
    """Unassessed event with zero publishers must render [NOT ASSESSED]."""
    now = datetime(2026, 11, 4, 12, 0, tzinfo=UTC)
    summary = EventSummary(
        event_id="evt-unassessed-0",
        headline="Border Patrol Observations Reported",
        what_happened=(
            "Regional authorities noted increased reconnaissance activity along the "
            "demarcation zone."
        ),
        confidence_qualification="unassessed",
        sources=(),
        model_used="test-model",
        corroboration_status="unassessed",
        independent_chains_count=0,
    )
    event = EventCluster(
        title="Border Patrol Observations",
        category=TopicCategory.MILITARY_OPERATIONS,
        confidence_label="corroborated",
        first_seen_at=now,
        last_seen_at=now,
        corroboration_status="unassessed",
        independent_chains_count=0,
    )
    formatter = TelegramDigestFormatter()
    block = formatter.format_event(
        summary=summary,
        event=event,
        category=event.category,
        confidence_label=event.confidence_label,
        rank=1,
        total=1,
    )
    assert "[NOT ASSESSED]" in block
    assert "[SINGLE SOURCE]" not in block
    assert "[CORROBORATED]" not in block


def test_missing_or_null_corroboration_status_handled_conservatively():
    """Missing or NULL corroboration status in models and DB rows defaults to unassessed
    and [NOT ASSESSED]."""
    now = datetime(2026, 11, 4, 12, 0, tzinfo=UTC)
    # Direct model construction with defaults
    ec_default = EventCluster(
        title="Default Event Cluster",
        category=TopicCategory.SECURITY_INCIDENT,
        first_seen_at=now,
        last_seen_at=now,
    )
    assert ec_default.corroboration_status == "unassessed"
    assert ec_default.independent_chains_count == 0

    es_default = EventSummary(
        event_id=ec_default.id,
        headline="Default Event Summary Headline",
        what_happened="Detailed explanation of the incident without explicit corroboration status.",
        confidence_qualification="uncertain",
        model_used="test-model",
    )
    assert es_default.corroboration_status == "unassessed"
    assert es_default.independent_chains_count == 0

    # Explicit None / null passed
    ec_none = EventCluster(
        title="Null Event Cluster",
        category=TopicCategory.SECURITY_INCIDENT,
        first_seen_at=now,
        last_seen_at=now,
        corroboration_status=None,
        independent_chains_count=None,
    )
    assert ec_none.corroboration_status == "unassessed"
    assert ec_none.independent_chains_count == 0

    es_none = EventSummary(
        event_id=ec_none.id,
        headline="Null Event Summary Headline",
        what_happened=(
            "Explanation of the incident with explicit None supplied to corroboration fields."
        ),
        confidence_qualification="uncertain",
        model_used="test-model",
        corroboration_status=None,
        independent_chains_count=None,
    )
    assert es_none.corroboration_status == "unassessed"
    assert es_none.independent_chains_count == 0

    formatter = TelegramDigestFormatter()
    block = formatter.format_event(
        summary=es_none,
        event=ec_none,
        category=ec_none.category,
        confidence_label=ec_none.confidence_label,
        rank=1,
        total=1,
    )
    assert "[NOT ASSESSED]" in block

    # DB row with missing / NULL columns
    row_null = {
        "event_id": ec_none.id,
        "headline": "Database Row With Null Corroboration",
        "what_happened": "Incident details from database record without corroboration fields.",
        "corroboration_status": None,
        "independent_chains_count": None,
    }
    recon = reconstruct_event_summary(row_null)
    assert recon is not None
    assert recon.corroboration_status == "unassessed"
    assert recon.independent_chains_count == 0

    recon_block = formatter.format_event(
        summary=recon,
        event=ec_none,
        category=ec_none.category,
        confidence_label=ec_none.confidence_label,
        rank=1,
        total=1,
    )
    assert "[NOT ASSESSED]" in recon_block


def test_explicitly_assessed_corroborated_event():
    """Explicitly assessed independently corroborated event renders [CORROBORATED]."""
    now = datetime(2026, 11, 4, 12, 0, tzinfo=UTC)
    s1 = EventSourceAttribution(
        source_name="The Washington Post", title="T1", url="https://wapo.com/1"
    )
    s2 = EventSourceAttribution(
        source_name="The Wall Street Journal", title="T2", url="https://wsj.com/2"
    )
    summary = EventSummary(
        event_id="evt-corrob",
        headline="Hardened Subterranean Fuel Storage Discovered",
        what_happened=(
            "Field records and geospatial satellite analysis independently confirmed fuel depots."
        ),
        confidence_qualification="Independently corroborated across 2 distinct reporting chains.",
        sources=(s1, s2),
        model_used="test-model",
        corroboration_status="independently_corroborated",
        independent_chains_count=2,
    )
    event = EventCluster(
        title="Fuel Depot Construction",
        category=TopicCategory.MILITARY_OPERATIONS,
        first_seen_at=now,
        last_seen_at=now,
        corroboration_status="independently_corroborated",
        independent_chains_count=2,
    )
    formatter = TelegramDigestFormatter()
    block = formatter.format_event(
        summary=summary,
        event=event,
        category=event.category,
        confidence_label="corroborated",
        rank=1,
        total=1,
    )
    assert "[CORROBORATED]" in block
    assert "[NOT ASSESSED]" not in block


def test_explicitly_assessed_single_source_event():
    """Explicitly assessed single-source event renders [SINGLE SOURCE]."""
    now = datetime(2026, 11, 4, 12, 0, tzinfo=UTC)
    s1 = EventSourceAttribution(
        source_name="Defense News", title="T1", url="https://defensenews.com/1"
    )
    summary = EventSummary(
        event_id="evt-single",
        headline="Procurement Tender Announced for Coastal Radar Systems",
        what_happened=(
            "Defense authorities released initial tender documentation for coastal \
            radar monitoring."
        ),
        confidence_qualification=(
            "Single-source reporting from Defense News; unverified independently."
        ),
        sources=(s1,),
        model_used="test-model",
        corroboration_status="single_source",
        independent_chains_count=1,
    )
    event = EventCluster(
        title="Radar Procurement Tender",
        category=TopicCategory.WEAPONS_PROCUREMENT,
        first_seen_at=now,
        last_seen_at=now,
        corroboration_status="single_source",
        independent_chains_count=1,
    )
    formatter = TelegramDigestFormatter()
    block = formatter.format_event(
        summary=summary,
        event=event,
        category=event.category,
        confidence_label="single_source",
        rank=1,
        total=1,
    )
    assert "[SINGLE SOURCE]" in block
    assert "[NOT ASSESSED]" not in block


def test_existing_conflicting_and_shared_origin_statuses():
    """Conflicting evidence, shared origin, and preliminary statuses retain correct badges."""
    now = datetime(2026, 11, 4, 12, 0, tzinfo=UTC)
    formatter = TelegramDigestFormatter()

    # Conflicting evidence -> [DISPUTED]
    sum_conflict = EventSummary(
        event_id="evt-dispute",
        headline="Contested Air Defense Interception Claims",
        what_happened=(
            "Opposing claims on whether drone attack was intercepted or destroyed facility."
        ),
        confidence_qualification="Contested claims; opposing parties provide conflicting accounts.",
        model_used="test-model",
        corroboration_status="conflicting_evidence",
        independent_chains_count=0,
    )
    ev_conflict = EventCluster(
        title="Substation Strike Claims",
        category=TopicCategory.ARMED_CONFLICT,
        first_seen_at=now,
        last_seen_at=now,
        corroboration_status="conflicting_evidence",
        independent_chains_count=0,
    )
    block_conflict = formatter.format_event(
        summary=sum_conflict,
        event=ev_conflict,
        category=ev_conflict.category,
        confidence_label="disputed",
        rank=1,
        total=1,
    )
    assert "[DISPUTED]" in block_conflict
    assert "[NOT ASSESSED]" not in block_conflict

    # Shared origin -> [SHARED ORIGIN]
    sum_shared = EventSummary(
        event_id="evt-shared",
        headline="Naval Drone Attack Reported Near Sevastopol",
        what_happened=(
            "Multiple outlets reported the patrol vessel incident citing a single Reuters dispatch."
        ),
        confidence_qualification=(
            "Reported across 5 publishers, but all rely on a single shared wire report."
        ),
        model_used="test-model",
        corroboration_status="multiple_reports_shared_origin",
        independent_chains_count=1,
    )
    ev_shared = EventCluster(
        title="Black Sea Naval Drone Attack",
        category=TopicCategory.ARMED_CONFLICT,
        first_seen_at=now,
        last_seen_at=now,
        corroboration_status="multiple_reports_shared_origin",
        independent_chains_count=1,
    )
    block_shared = formatter.format_event(
        summary=sum_shared,
        event=ev_shared,
        category=ev_shared.category,
        confidence_label="corroborated",
        rank=1,
        total=1,
    )
    assert "[SHARED ORIGIN]" in block_shared
    assert "[NOT ASSESSED]" not in block_shared

    # Shared origin with official announcement -> [OFFICIAL ANNOUNCEMENT]
    s_official = EventSourceAttribution(
        source_name="Ministry of Defense",
        title="Official Statement",
        url="https://mod.gov.example/press",
    )
    sum_official = EventSummary(
        event_id="evt-official",
        headline="Procurement of Three Air Defense Batteries Announced",
        what_happened=(
            "The defense ministry announced purchase of three mobile air defense batteries."
        ),
        confidence_qualification="Official announcement from Ministry of Defense.",
        sources=(s_official,),
        model_used="test-model",
        corroboration_status="multiple_reports_shared_origin",
        independent_chains_count=1,
    )
    block_official = formatter.format_event(
        summary=sum_official,
        event=ev_shared,
        category=ev_shared.category,
        confidence_label="corroborated",
        rank=1,
        total=1,
    )
    assert "[OFFICIAL ANNOUNCEMENT]" in block_official
    assert "[NOT ASSESSED]" not in block_official

    # Insufficient information -> [PRELIMINARY]
    sum_prelim = EventSummary(
        event_id="evt-prelim",
        headline="Unconfirmed Rumors Allege Hypersonic Battery Movement",
        what_happened=(
            "Online chatter claims missile units were relocated without official confirmation."
        ),
        confidence_qualification="Preliminary reporting; source relationships unverified.",
        model_used="test-model",
        corroboration_status="insufficient_information",
        independent_chains_count=0,
    )
    block_prelim = formatter.format_event(
        summary=sum_prelim,
        event=ev_conflict,
        category=ev_conflict.category,
        confidence_label="uncertain",
        rank=1,
        total=1,
    )
    assert "[PRELIMINARY]" in block_prelim
    assert "[NOT ASSESSED]" not in block_prelim
