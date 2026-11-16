"""Comprehensive regression tests for Geopulse's publication-quality gate.

Verifies:
- Attribution preservation for allegations (Pentagon/Sylvia).
- Rejection of entries with missing source links or no accepted claims.
- Sponsored teaser handling and exclusion.
- Multi-story newsletter handling (War on the Rocks 'Tensions Rise').
- Valid concise single-fact articles.
- Elimination of headline and summary redundancy (Switzerland/Taiwan).
- Harmless typographic differences in evidence spans (smart quotes, dashes, entities).
- Rejection of fabricated or semantically unrelated evidence spans.
- Source identity and inline/source-list consistency.
- Bounded retries, quarantine, and operational safety during timeouts.
"""

from datetime import UTC, datetime

from geopulse.classification.models import TopicCategory
from geopulse.clustering.models import EventCluster
from geopulse.delivery.formatter import TelegramDigestFormatter
from geopulse.ingestion.models import ArticleRecord
from geopulse.summarization.client import MockOllamaClient
from geopulse.summarization.evidence_matcher import (
    check_evidence_span_match,
)
from geopulse.summarization.exceptions import OllamaTimeoutError
from geopulse.summarization.models import (
    ClaimEvidenceReference,
    EventKeyDetails,
    EventSourceAttribution,
    EventSummary,
    SummaryClaim,
)
from geopulse.summarization.publication_gate import (
    strip_leading_headline_duplicate,
    strip_wire_dateline,
    validate_publication_gate,
)
from geopulse.summarization.service import EventSummarizer

DT = datetime(2026, 10, 29, 12, 0, tzinfo=UTC)


def _make_article(
    title: str,
    snippet: str,
    url: str = "https://defensenews.com/article-1",
    source_name: str = "Defense News",
) -> ArticleRecord:
    return ArticleRecord(
        source_name=source_name,
        title=title,
        url=url,
        canonical_url=url,
        published_at=DT,
        snippet=snippet,
        content_hash="test-hash",
    )


def _make_event(
    title: str,
    snippet: str,
    url: str = "https://defensenews.com/article-1",
    category: str = "defence_policy",
    source_name: str = "Defense News",
    confidence_label: str = "single_source",
    articles: tuple[ArticleRecord, ...] | None = None,
    source_count: int | None = None,
) -> EventCluster:
    art = _make_article(title=title, snippet=snippet, url=url, source_name=source_name)
    arts = articles or (art,)
    return EventCluster(
        title=title,
        category=category,  # type: ignore[arg-type]
        first_seen_at=DT,
        last_seen_at=DT,
        published_at=DT,
        confidence_label=confidence_label,
        articles=arts,
    )


def test_allegation_preserves_attribution_in_headline_and_narrative():
    """An allegation attributed to the Pentagon must not become an established fact."""
    source_title = "Pentagon Accuses General Sylvia of Leaking Classified Information"
    source_snippet = (
        "The Department of Defense Inspector General issued a probe finding that "
        "General Sylvia allegedly disclosed classified materials to unauthorized personnel."
    )
    event = _make_event(
        title=source_title,
        snippet=source_snippet,
        url="https://defensenews.com/pentagon-sylvia-probe",
    )

    # 1. Negative case: Headline drops attribution and asserts leak as fact
    unattributed_summary = EventSummary(
        event_id="evt-sylvia",
        headline="General Sylvia Leaked Classified Information",
        what_happened="General Sylvia disclosed classified materials to unauthorized personnel.",
        confidence_qualification="Single-source reporting; unverified independently.",
        sources=(
            EventSourceAttribution(
                source_name="Defense News",
                title=source_title,
                url="https://defensenews.com/pentagon-sylvia-probe",
                published_at=DT,
            ),
        ),
        claims=(
            SummaryClaim(
                claim_id="c1",
                text="General Sylvia disclosed classified materials to unauthorized personnel.",
                evidence_refs=(
                    ClaimEvidenceReference(
                        source_url="https://defensenews.com/pentagon-sylvia-probe",
                        evidence_span="General Sylvia allegedly disclosed classified materials",
                        verification_status="accepted",
                    ),
                ),
                classification="attributed_statement",
                attribution="Pentagon Inspector General",
                central=True,
                verification_status="accepted",
            ),
        ),
        model_used="test-model",
    )

    gate_res = validate_publication_gate(unattributed_summary, event)
    # The gate repairs the headline to preserve attribution from the source title
    assert gate_res.passed
    assert gate_res.repaired_summary is not None
    assert "Pentagon" in gate_res.repaired_summary.headline or (
        "Alleges" in gate_res.repaired_summary.headline
    )


def test_missing_source_links_or_no_accepted_claims_rejected():
    """An event with no accepted claims or missing source links must fail the publication gate."""
    event = _make_event("Troops Deploy to Border", "Border patrol reinforced.")

    summary_no_claims = EventSummary(
        event_id="evt-no-claims",
        headline="Troops Deploy to Border",
        what_happened="Troops have deployed to the border region.",
        confidence_qualification="Single-source",
        sources=(),
        claims=(
            SummaryClaim(
                claim_id="c1",
                text="Troops deployed",
                evidence_refs=(
                    ClaimEvidenceReference(
                        source_url="https://defensenews.com/article-1",
                        evidence_span="border",
                        verification_status="rejected",
                    ),
                ),
                classification="reported_fact",
                central=True,
                verification_status="rejected",
            ),
        ),
        model_used="test-model",
    )

    res = validate_publication_gate(summary_no_claims, event)
    assert not res.passed
    assert "No accepted material claims" in (res.quarantine_reason or "")


def test_coherent_article_containing_sponsored_teaser_strips_teaser():
    """Articles containing [Sponsored] teasers must have the teaser stripped from ingestion."""
    from geopulse.ingestion.parser import _clean_feed_candidate

    raw_feed_body = (
        "[Sponsored] As social media and AI amplify disinformation campaigns, "
        "next-generation cognitive security is paramount. "
        "Air Force demonstrates autonomous drone swarming capabilities during Red Flag exercise."
    )
    cleaned = _clean_feed_candidate(raw_feed_body)
    assert cleaned is not None
    assert "[Sponsored]" not in cleaned
    assert "autonomous drone swarming" in cleaned

    # And if summary content directly uses sponsored tokens, publication gate rejects it
    event = _make_event("Autonomous Drone Swarms Tested", cleaned)
    sponsored_summary = EventSummary(
        event_id="evt-sponsored",
        headline="[Sponsored] Next-Gen Cognitive Security Drone Capabilities",
        what_happened="Vendor unveils sponsored drone capabilities.",
        confidence_qualification="Single-source",
        sources=(
            EventSourceAttribution(
                source_name="Defense News",
                title="Swarms",
                url="https://defensenews.com/article-1",
            ),
        ),
        model_used="test-model",
    )
    gate_res = validate_publication_gate(sponsored_summary, event)
    assert not gate_res.passed
    assert "sponsored" in (gate_res.quarantine_reason or "").lower()


def test_newsletter_content_with_unrelated_story_fragments_quarantined():
    """Multi-story newsletter combining disparate actions across sections is quarantined."""
    newsletter_text = (
        "Welcome to The Adversarial. Every other week, we'll provide you with expert analysis "
        "on America's greatest challengers. Read more below.***Iran Indications of a return to "
        "direct hostilities are increasing. The United States is evacuating more troops and "
        "diplomatic staff, moving a third aircraft carrier strike group to the Middle East, "
        "and preparing strike options."
    )
    event = _make_event(
        title="Tensions Rise",
        snippet=newsletter_text,
        url="https://warontherocks.com/tensions-rise/",
        source_name="War on the Rocks",
    )

    conflated_summary = EventSummary(
        event_id="evt-wotr",
        headline="According to The United States, Tensions Rise",
        what_happened=(
            "The United States is evacuating troops, moving a third carrier strike group, "
            "and preparing strike options."
        ),
        confidence_qualification="Single-source",
        sources=(
            EventSourceAttribution(
                source_name="War on the Rocks",
                title="Tensions Rise",
                url="https://warontherocks.com/tensions-rise/",
            ),
        ),
        model_used="test-model",
    )

    gate_res = validate_publication_gate(conflated_summary, event)
    assert not gate_res.passed
    assert "newsletter" in (gate_res.quarantine_reason or "").lower()


def test_valid_concise_article_single_fact_passes():
    """A legitimate short report with one useful fact must pass without requiring extra fluff."""
    snippet = "Poland signed a contract to establish a PAC-3 missile maintenance facility."
    event = _make_event(
        title="Poland Establishes PAC-3 Facility",
        snippet=snippet,
        url="https://defensenews.com/poland-pac3",
        category="weapons_procurement",
    )

    summary = EventSummary(
        event_id="evt-poland",
        headline="Poland Establishes PAC-3 Missile Facility",
        what_happened=snippet,
        confidence_qualification="Single-source reporting; unverified independently.",
        sources=(
            EventSourceAttribution(
                source_name="Defense News",
                title="Poland Establishes PAC-3 Facility",
                url="https://defensenews.com/poland-pac3",
                published_at=DT,
            ),
        ),
        claims=(
            SummaryClaim(
                claim_id="c1",
                text=snippet,
                evidence_refs=(
                    ClaimEvidenceReference(
                        source_url="https://defensenews.com/poland-pac3",
                        evidence_span=snippet,
                        verification_status="accepted",
                    ),
                ),
                classification="reported_fact",
                central=True,
                verification_status="accepted",
            ),
        ),
        model_used="test-model",
    )

    gate_res = validate_publication_gate(summary, event)
    assert gate_res.passed
    formatter = TelegramDigestFormatter()
    block = formatter.format_event_block(
        summary=summary,
        category=event.category,
        confidence_label=event.confidence_label,
        rank=1,
        total=1,
    )
    assert "PAC-3" in block
    assert "https://defensenews.com/poland-pac3" in block


def test_headline_and_summary_redundancy_eliminated():
    """When the summary body starts with the headline, the stutter is cleanly stripped."""
    headline = "Switzerland shortlists SAMP/T, Barak MX for long-range air defense"
    body = (
        "Switzerland shortlists SAMP/T, Barak MX for long-range air defense. "
        "PARIS — Switzerland narrowed down the choice of the country's second future "
        "long-range air defense system."
    )
    stripped = strip_leading_headline_duplicate(body, headline)
    stripped = strip_wire_dateline(stripped)
    assert not stripped.startswith("Switzerland shortlists")
    assert not stripped.startswith("PARIS —")
    assert stripped.startswith("Switzerland narrowed down")

    # In format_event_block
    event = _make_event(headline, body)
    summary = EventSummary(
        event_id="evt-swiss",
        headline=headline,
        what_happened=body,
        confidence_qualification="Single-source",
        sources=(
            EventSourceAttribution(
                source_name="Defense News",
                title=headline,
                url="https://defensenews.com/article-1",
            ),
        ),
        model_used="test-model",
    )
    formatter = TelegramDigestFormatter()
    block = formatter.format_event_block(
        summary=summary,
        category=event.category,
        confidence_label=event.confidence_label,
        rank=1,
        total=1,
    )
    # The headline appears in bold, but the body immediately underneath does not repeat it
    lines = block.splitlines()
    headline_line = next(line for line in lines if "<b>Switzerland shortlists" in line)
    assert headline_line is not None
    # Body line does not stutter
    assert (
        "Switzerland shortlists SAMP/T, Barak MX for long-range air defense. Switzerland shortlists"
        not in block
    )


def test_exact_evidence_with_harmless_whitespace_and_entity_differences():
    """Smart quotes, non-breaking spaces, em-dashes and HTML entities match via normalized check."""
    article_text = (
        'Gen. Clark stated: "The Davidson window benchmark remains valid &amp; actionable '
        'for U.S. forces." — breaking report.'
    )
    # Model returns smart quotes, en-dash, unescaped entity, and non-breaking space
    evidence_span = (
        "Gen. Clark stated: “The Davidson window benchmark remains valid &\u00a0actionable "
        "for U.S. forces.” - breaking report."
    )

    match_status, diff = check_evidence_span_match(evidence_span, article_text)
    assert match_status == "normalized"
    assert diff is None


def test_fabricated_or_semantically_unrelated_evidence_span_rejected():
    """Fabricated or absent evidence spans return no_match with sanitized diff."""
    article_text = "Poland announced a new maintenance hub for Patriot PAC-3 interceptors."
    fabricated_span = "Russia deployed hypersonic nuclear glide vehicles to the Baltic Sea."

    match_status, diff = check_evidence_span_match(fabricated_span, article_text)
    assert match_status == "no_match"
    assert diff is not None
    assert "Span not found" in diff
    assert "hypersonic" in diff


def test_source_identity_and_inline_source_list_consistency():
    """Rendered Sources section and inline citations agree with accepted claims."""
    url = "https://defensenews.com/test-story"
    evidence_span = "Defence authorities confirmed deployment of interceptors."
    art = _make_article(
        title="Interceptors Deployed",
        snippet=evidence_span,
        url=url,
        source_name="Defense News",
    )
    event = EventCluster(
        title="Interceptors Deployed",
        category="military_operations",  # type: ignore[arg-type]
        first_seen_at=DT,
        last_seen_at=DT,
        published_at=DT,
        confidence_label="single_source",
        articles=(art,),
    )
    summary = EventSummary(
        event_id="evt-source-test",
        headline="Interceptors Deployed",
        what_happened="Defence authorities confirmed deployment of interceptors.",
        confidence_qualification="Single-source reporting; unverified independently.",
        sources=(
            EventSourceAttribution(
                source_name="Defense News",
                title="Interceptors Deployed",
                url=url,
                published_at=DT,
            ),
        ),
        claims=(
            SummaryClaim(
                claim_id="c1",
                text="Defence authorities confirmed deployment of interceptors.",
                evidence_refs=(
                    ClaimEvidenceReference(
                        source_url=url,
                        evidence_span=evidence_span,
                        verification_status="accepted",
                    ),
                ),
                classification="reported_fact",
                central=True,
                verification_status="accepted",
            ),
        ),
        model_used="test-model",
    )

    formatter = TelegramDigestFormatter()
    block = formatter.format_event_block(
        summary=summary,
        category=event.category,
        confidence_label=event.confidence_label,
        rank=1,
        total=1,
    )
    # Both inline link and Sources block present
    assert f'<a href="{url}">Defense News</a>' in block
    assert "<b>Sources:</b>" in block


def test_bounded_retries_and_timeout_handling_never_publishes_unverified():
    """An OllamaTimeoutError during summarization marks status failed and never publishes."""
    event = _make_event(
        "Air Force Exercise in Pacific",
        (
            "Air Force units conducted long-range bomber training exercise "
            "across the western Pacific theatre."
        ),
    )

    def timing_out_generate(*args, **kwargs):
        raise OllamaTimeoutError("Timed out after 120.0s")

    client = MockOllamaClient(custom_generator=timing_out_generate)
    summarizer = EventSummarizer(client=client)
    res = summarizer.summarize_event(event)

    assert res.status == "failed"
    assert "Timed out after 120.0s" in (res.error_message or "")

    # Digest selection filters it out completely
    formatter = TelegramDigestFormatter()
    selected = formatter.select_digest_events([res])
    assert len(selected) == 0


def test_awkward_country_attribution_not_emitted():
    """'According to The United States,' is not prepended to country actions."""
    from geopulse.summarization.publication_gate import clean_awkward_attribution_prefix

    raw_headline = "According to The United States, U.S. Navy Deploys Additional Ships"
    cleaned = clean_awkward_attribution_prefix(raw_headline)
    assert not cleaned.startswith("According to")
    assert cleaned.startswith("U.S. Navy Deploys Additional Ships")


def test_single_claim_summary_headline_equal_claim_retains_narrative():
    """Single-claim summary where headline equals claim text retains body and links."""
    claim_text = "Canada releases new national defence strategy for Arctic sovereignty."
    event = _make_event(
        title="Canada releases new national defence strategy",
        snippet=claim_text,
        url="https://cbc.ca/defence-strategy",
        source_name="CBC News",
    )
    claim = SummaryClaim(
        claim_id="c1",
        text=claim_text,
        evidence_refs=(
            ClaimEvidenceReference(
                source_url="https://cbc.ca/defence-strategy",
                evidence_span=claim_text,
                verification_status="accepted",
            ),
        ),
        classification="reported_fact",
        central=True,
        verification_status="accepted",
    )
    summary = EventSummary(
        event_id="evt-canada",
        headline=claim_text,
        what_happened=claim_text,
        confidence_qualification="Single-source reporting; unverified independently.",
        sources=(
            EventSourceAttribution(
                source_name="CBC News",
                title="Canada releases new national defence strategy",
                url="https://cbc.ca/defence-strategy",
                published_at=DT,
            ),
        ),
        claims=(claim,),
        model_used="test-model",
    )

    formatter = TelegramDigestFormatter()
    rendered = formatter.format_event_block(
        summary=summary,
        category=event.category,
        confidence_label=event.confidence_label,
        rank=1,
        total=1,
    )

    # Must contain the factual readable narrative, not just the link markup
    assert claim_text in rendered
    assert 'href="https://cbc.ca/defence-strategy"' in rendered
    assert "CBC News" in rendered

    # Verify digest selection accepts this valid single-claim event
    from geopulse.summarization.models import SummarizationResult

    res = SummarizationResult(
        event_id="evt-canada",
        status="success",
        summary=summary,
        event=event,
    )
    selected = formatter.select_digest_events([res])
    assert len(selected) == 1


def test_genuinely_invalid_empty_body_rejected_by_gate_and_digest_selection():
    """A summary whose body contains only tags without substantive narrative is rejected."""
    event = _make_event(
        title="Operational Report",
        snippet="Operational Report",
        url="https://defensenews.com/report",
    )
    # 10 characters minimum to pass Pydantic schema, but contains only markup and no narrative text
    tag_only_summary = EventSummary(
        event_id="evt-empty",
        headline="Valid Headline",
        what_happened='<i>(<a href="https://defensenews.com/report">Defense News</a>)</i>',
        confidence_qualification="Single-source",
        sources=(
            EventSourceAttribution(
                source_name="Defense News",
                title="Operational Report",
                url="https://defensenews.com/report",
            ),
        ),
        model_used="test-model",
    )
    gate_res = validate_publication_gate(tag_only_summary, event)
    assert not gate_res.passed
    assert "empty" in (gate_res.quarantine_reason or "").lower()

    from geopulse.summarization.models import SummarizationResult

    res = SummarizationResult(
        event_id="evt-empty",
        status="success",
        summary=tag_only_summary,
        event=event,
    )
    formatter = TelegramDigestFormatter()
    selected = formatter.select_digest_events([res])
    assert len(selected) == 0


def test_long_central_claim_and_word_boundary_headline_truncation():
    """Headlines derived from long claims truncate safely without dangling connectors."""
    from geopulse.summarization.validator import safe_truncate_headline

    long_claim = (
        "Switzerland's federal council approved the final contract for the acquisition of 36 "
        "Lockheed Martin F-35A Lightning II fighter jets and comprehensive logistics packages"
    )
    truncated = safe_truncate_headline(long_claim, max_chars=120)
    # Never truncate in the middle of a word
    assert not truncated.endswith("-")
    assert not truncated.endswith("Lightnin")
    assert "Lightning" in truncated or "jets" in truncated
    # Never end on a dangling connector
    assert not truncated.endswith(" and")
    assert not truncated.endswith(" of")
    assert not truncated.endswith(" for")

    # Mid-word cut test
    mid_word_test = safe_truncate_headline(
        "Turkey tests Atmaca anti-ship missile system", max_chars=25
    )
    assert mid_word_test == "Turkey tests Atmaca"


def test_swiss_government_attribution_prefix_cleaned():
    """Awkward 'According to Switzerland's federal government' prefix is removed from headlines."""
    from geopulse.summarization.publication_gate import clean_awkward_attribution_prefix

    raw_headline = (
        "According to Switzerland's federal government, Switzerland signs F-35 procurement deal"
    )
    cleaned = clean_awkward_attribution_prefix(raw_headline)
    assert not cleaned.startswith("According to")
    assert cleaned.startswith("Switzerland signs F-35")


def test_attributed_allegation_preserves_attribution_in_headline():
    """An accusation or investigation preserves the accusing entity in the derived headline."""
    from geopulse.summarization.service import _derive_headline

    evidence = (
        "The Pentagon inspector general opened an investigation into classified intelligence leaks."
    )
    event = _make_event(
        title="Pentagon Probes Classified Leak Allegations",
        snippet=evidence,
        url="https://defensenews.com/pentagon-leak",
    )
    claim = SummaryClaim(
        claim_id="c1",
        text="The inspector general is investigating alleged classified document leaks.",
        evidence_refs=(
            ClaimEvidenceReference(
                source_url="https://defensenews.com/pentagon-leak",
                evidence_span=evidence,
                verification_status="accepted",
            ),
        ),
        classification="attributed_statement",
        attribution="Pentagon inspector general",
        central=True,
        verification_status="accepted",
    )
    headline = _derive_headline(claim, event)
    headline_lower = headline.lower()
    # Must preserve attribution and allegation
    assert "pentagon" in headline_lower or "inspector general" in headline_lower
    assert "leak" in headline_lower or "alleg" in headline_lower or "probe" in headline_lower


def test_wotr_multi_story_newsletter_combining_military_and_political_leaders_quarantined():
    """Newsletter combining US carrier moves with Israeli political developments is quarantined."""
    newsletter_text = (
        "Welcome to The Adversarial. Every other week, we'll provide you with expert analysis. "
        "Read more below.***Iran The United States is evacuating more troops and moving a third "
        "aircraft carrier strike group to the Middle East.***Israel Ahead of upcoming elections, "
        "Prime Minister Benjamin Netanyahu announced a new coalition campaign."
    )
    event = _make_event(
        title="Tensions Rise",
        snippet=newsletter_text,
        url="https://warontherocks.com/tensions-rise/",
        source_name="War on the Rocks",
    )
    conflated_summary = EventSummary(
        event_id="evt-wotr-conflated",
        headline="U.S. Moves Carriers as Netanyahu Campaigns for Election",
        what_happened=(
            "The United States is moving aircraft carriers and evacuating personnel in the "
            "Middle East, while Prime Minister Benjamin Netanyahu launches his campaign."
        ),
        confidence_qualification="Single-source",
        sources=(
            EventSourceAttribution(
                source_name="War on the Rocks",
                title="Tensions Rise",
                url="https://warontherocks.com/tensions-rise/",
            ),
        ),
        model_used="test-model",
    )
    gate_res = validate_publication_gate(conflated_summary, event)
    assert not gate_res.passed
    assert "newsletter" in (gate_res.quarantine_reason or "").lower()


def test_legitimate_single_story_analysis_not_quarantined():
    """A cohesive single-story analytical article is not quarantined as a newsletter roundup."""
    article_text = (
        "Since returning to office, the administration has exercised executive authority across "
        "the Department of Defense. The next Congress must reclaim its oversight."
    )
    event = _make_event(
        title="The Next Congress Needs to Reclaim Its Oversight of the Department of Defense",
        snippet=article_text,
        url="https://warontherocks.com/congress-oversight-dod/",
        source_name="War on the Rocks",
    )
    summary = EventSummary(
        event_id="evt-wotr-single",
        headline="Congress Urged to Reclaim Department of Defense Oversight",
        what_happened=(
            "Analysts argue that Congress must reassert oversight over defense policy "
            "and executive authority in the Department of Defense."
        ),
        confidence_qualification="Single-source reporting; unverified independently.",
        sources=(
            EventSourceAttribution(
                source_name="War on the Rocks",
                title="The Next Congress Needs to Reclaim Oversight",
                url="https://warontherocks.com/congress-oversight-dod/",
                published_at=DT,
            ),
        ),
        model_used="test-model",
    )
    gate_res = validate_publication_gate(summary, event)
    assert gate_res.passed


def test_headline_sentence_splitting_black_sea_not_divided():
    """Black Sea regression test: headline must not end in 'called' and body must not split."""
    from geopulse.delivery.formatter import TelegramDigestFormatter
    from geopulse.summarization.publication_gate import strip_leading_headline_duplicate
    from geopulse.summarization.service import _derive_headline

    claim_text = (
        "Turkey called on all parties to refrain from escalating maritime actions in the "
        "Black Sea after a series of naval confrontations near commercial shipping lanes."
    )
    event = _make_event(
        title="Black Sea Maritime Security Warning",
        snippet=claim_text,
        url="https://aljazeera.com/black-sea-warning",
    )
    claim = SummaryClaim(
        claim_id="c1",
        text=claim_text,
        evidence_refs=(
            ClaimEvidenceReference(
                source_url="https://aljazeera.com/black-sea-warning",
                evidence_span=claim_text,
                verification_status="accepted",
            ),
        ),
        classification="reported_fact",
        central=True,
        verification_status="accepted",
    )

    # 1. Headline derivation test: must not cut mid-sentence on 'called'
    headline = _derive_headline(claim, event)
    assert not headline.endswith("called")
    assert not headline.endswith("called on")
    assert len(headline.split()) >= 4

    # 2. Formatter duplicate-stripping test: must not divide sentence
    body = claim_text
    # Even if headline were cut to 'Turkey called', body must NOT be stripped
    bad_prefix_headline = "Turkey called"
    stripped_body = strip_leading_headline_duplicate(body, bad_prefix_headline)
    assert stripped_body == body
    assert not stripped_body.startswith("on all parties")

    # 3. Production Telegram block formatting
    summary = EventSummary(
        event_id="evt-black-sea",
        headline=headline,
        what_happened=claim_text,
        confidence_qualification="Single-source reporting.",
        sources=(
            EventSourceAttribution(
                source_name="Al Jazeera",
                title="Black Sea Warning",
                url="https://aljazeera.com/black-sea-warning",
            ),
        ),
        claims=(claim,),
        model_used="test-model",
    )
    formatter = TelegramDigestFormatter()
    block = formatter.format_event_block(
        summary=summary,
        category=TopicCategory.SECURITY_INCIDENT,
        confidence_label="single_source",
        rank=1,
        total=1,
    )
    # Check that block does not have the divided sentence
    assert "<b>Turkey called</b>" not in block
    assert "on all parties to refrain" not in block.split("\n\n")[0]  # Not starting narrative
    assert claim_text in block or "Turkey called on all parties" in block


def test_headline_sentence_splitting_uk_offset_policy_not_divided():
    """UK Offset Policy test: headline must not end in 'selected' and body must not split."""
    from geopulse.delivery.formatter import TelegramDigestFormatter
    from geopulse.summarization.publication_gate import strip_leading_headline_duplicate
    from geopulse.summarization.service import _derive_headline

    claim_text = (
        "The UK Ministry of Defence announces the UK Offset Policy to require suppliers to invest "
        "in the UK economy on selected overseas defence and security procurements."
    )
    event = _make_event(
        title="Defence contracts to prioritise British jobs and skills",
        snippet=claim_text,
        url="https://gov.uk/uk-offset-policy",
        source_name="UK Ministry of Defence",
    )
    claim = SummaryClaim(
        claim_id="c1",
        text=claim_text,
        evidence_refs=(
            ClaimEvidenceReference(
                source_url="https://gov.uk/uk-offset-policy",
                evidence_span=claim_text,
                verification_status="accepted",
            ),
        ),
        classification="reported_fact",
        central=True,
        verification_status="accepted",
    )

    # 1. Headline derivation test: must not cut mid-sentence on 'selected'
    headline = _derive_headline(claim, event)
    assert not headline.endswith("selected")
    assert not headline.endswith("on selected")

    # 2. Formatter duplicate-stripping test: must not divide sentence
    body = claim_text
    bad_prefix_headline = (
        "The UK Ministry of Defence announces the UK Offset Policy to require suppliers "
        "to invest in the UK economy on selected"
    )
    stripped_body = strip_leading_headline_duplicate(body, bad_prefix_headline)
    assert stripped_body == body
    assert not stripped_body.startswith("overseas defence and security procurements")

    # 3. Production Telegram block formatting
    summary = EventSummary(
        event_id="evt-uk-offset",
        headline=headline,
        what_happened=claim_text,
        confidence_qualification="Single-source reporting.",
        sources=(
            EventSourceAttribution(
                source_name="UK Ministry of Defence",
                title="Defence contracts to prioritise British jobs and skills",
                url="https://gov.uk/uk-offset-policy",
            ),
        ),
        claims=(claim,),
        model_used="test-model",
    )
    formatter = TelegramDigestFormatter()
    block = formatter.format_event_block(
        summary=summary,
        category=TopicCategory.DEFENCE_POLICY,
        confidence_label="single_source",
        rank=1,
        total=1,
    )
    assert "on selected</b>" not in block
    assert (
        "overseas defence and security procurements." not in block.split("</b>\n")[1].strip()[:30]
    )


def test_headline_truncation_positive_cases_short_and_long():
    """Verify headline construction for both short headlines and long dependent clauses."""
    from geopulse.summarization.validator import safe_truncate_headline

    # Short headline remains intact
    short_h = "US Approves $500M Arms Sale to Taiwan"
    assert safe_truncate_headline(short_h, max_chars=140) == short_h

    # Long headline with dependent clause cuts at phrase boundary, not mid-word or on connector
    long_h = (
        "The UK Ministry of Defence announces the UK Offset Policy to require foreign suppliers "
        "to invest in domestic manufacturing and apprenticeship programmes for high-value tenders."
    )
    truncated = safe_truncate_headline(long_h, max_chars=70)
    assert truncated == "The UK Ministry of Defence announces the UK Offset Policy"
    assert not truncated.endswith("to")
    assert not truncated.endswith("require")


def test_full_article_fallback_leakage_prevented():
    """Requirement 1: Never use an article snippet or body as replacement narrative.

    When no distinct supporting claim exists, preserve the headline and source link.
    """
    headline = "UK Announces Support for Ukrainian Drone Manufacturing"
    claim_text = "The UK announced support for Ukrainian drone manufacturing."
    snippet = (
        "The UK announced support for Ukrainian drone manufacturing. "
        "The bilateral agreement will directly support 12 Ukrainian drone manufacturers "
        "to scale production across several industrial centers."
    )
    event = _make_event(
        title=headline,
        snippet=snippet,
        url="https://defensenews.com/uk-ukraine-drones",
        source_name="Defense News",
    )
    claim = SummaryClaim(
        claim_id="c1",
        text=claim_text,
        evidence_refs=(
            ClaimEvidenceReference(
                source_url="https://defensenews.com/uk-ukraine-drones",
                evidence_span="The UK announced support for Ukrainian drone manufacturing",
                verification_status="accepted",
            ),
        ),
        classification="reported_fact",
        central=True,
        verification_status="accepted",
    )
    summary = EventSummary(
        event_id="evt-uk-drones",
        headline=headline,
        what_happened=claim_text,
        confidence_qualification="Single-source reporting; unverified independently.",
        sources=(
            EventSourceAttribution(
                source_name="Defense News",
                title=headline,
                url="https://defensenews.com/uk-ukraine-drones",
                published_at=DT,
            ),
        ),
        claims=(claim,),
        model_used="test-model",
    )

    gate_res = validate_publication_gate(summary, event)
    assert gate_res.passed
    # Must never copy raw snippet or body text
    assert gate_res.repaired_summary is None or (
        "across several industrial centers" not in gate_res.repaired_summary.what_happened
    )

    formatter = TelegramDigestFormatter()
    block = formatter.format_event_block(
        summary=summary,
        category=event.category,
        confidence_label=event.confidence_label,
        rank=1,
        total=1,
    )
    # Formatter preserves headline and source link without snippet leakage or duplicate body
    assert "<b>UK Announces Support for Ukrainian Drone Manufacturing</b>" in block
    assert 'href="https://defensenews.com/uk-ukraine-drones"' in block
    assert "Defense News" in block
    assert "across several industrial centers" not in block


def test_distinct_verified_supporting_claim_preferred_over_duplicate_central():
    """Requirement 1: Prefer a distinct, already verified supporting claim."""
    headline = "UK Announces Support for Ukrainian Drone Manufacturing"
    claim_text = "The UK announced support for Ukrainian drone manufacturing."
    supporting_text = (
        "The bilateral agreement directly supports 12 Ukrainian drone manufacturers "
        "to scale production."
    )
    event = _make_event(
        title=headline,
        snippet=supporting_text,
        url="https://defensenews.com/uk-ukraine-drones",
        source_name="Defense News",
    )
    claim1 = SummaryClaim(
        claim_id="c1",
        text=claim_text,
        evidence_refs=(
            ClaimEvidenceReference(
                source_url="https://defensenews.com/uk-ukraine-drones",
                evidence_span="The UK announced support for Ukrainian drone manufacturing",
                verification_status="accepted",
            ),
        ),
        classification="reported_fact",
        central=True,
        verification_status="accepted",
    )
    claim2 = SummaryClaim(
        claim_id="c2",
        text=supporting_text,
        evidence_refs=(
            ClaimEvidenceReference(
                source_url="https://defensenews.com/uk-ukraine-drones",
                evidence_span=supporting_text,
                verification_status="accepted",
            ),
        ),
        classification="reported_fact",
        central=False,
        verification_status="accepted",
    )
    summary = EventSummary(
        event_id="evt-uk-drones-supp",
        headline=headline,
        what_happened=claim_text,
        confidence_qualification="Single-source reporting; unverified independently.",
        sources=(
            EventSourceAttribution(
                source_name="Defense News",
                title=headline,
                url="https://defensenews.com/uk-ukraine-drones",
                published_at=DT,
            ),
        ),
        claims=(claim1, claim2),
        model_used="test-model",
    )

    gate_res = validate_publication_gate(summary, event)
    assert gate_res.passed
    assert gate_res.repaired_summary is not None
    assert gate_res.repaired_summary.what_happened == supporting_text

    formatter = TelegramDigestFormatter()
    block = formatter.format_event_block(
        summary=gate_res.repaired_summary,
        category=event.category,
        confidence_label=event.confidence_label,
        rank=1,
        total=1,
    )
    assert supporting_text in block
    assert "<b>UK Announces Support for Ukrainian Drone Manufacturing</b>" in block


def test_semantic_headline_body_repetition_prevented_pentagon_budget():
    from geopulse.summarization.publication_gate import (
        extract_plain_narrative,
        is_headline_body_duplicate,
    )

    """Requirement 3: Detect semantic headline/body repetition like Pentagon budget."""
    h_pentagon = "Pentagon Requests $30.9 Billion for 2026 Munitions Budget"
    b_pentagon = "The Pentagon requested $30.9 billion for munitions procurement in fiscal 2026."
    # Must detect semantic repetition
    assert is_headline_body_duplicate(b_pentagon, h_pentagon)

    # Swiss air defense story is distinct and must not be flagged
    h_swiss = "Switzerland Approves Air Defense Modernization"
    b_swiss = "Federal council allocates 3.2 billion francs to replace aging fighter fleet by 2030."
    assert not is_headline_body_duplicate(b_swiss, h_swiss)

    # Formatter renders headline and source link without duplicate body
    claim = SummaryClaim(
        claim_id="c1",
        text=b_pentagon,
        evidence_refs=(
            ClaimEvidenceReference(
                source_url="https://defensenews.com/pentagon-budget",
                evidence_span=b_pentagon,
                verification_status="accepted",
            ),
        ),
        classification="reported_fact",
        central=True,
        verification_status="accepted",
    )
    summary = EventSummary(
        event_id="evt-pentagon-budget",
        headline=h_pentagon,
        what_happened=b_pentagon,
        confidence_qualification="Single-source reporting; unverified independently.",
        sources=(
            EventSourceAttribution(
                source_name="Defense News",
                title=h_pentagon,
                url="https://defensenews.com/pentagon-budget",
                published_at=DT,
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
    assert f"<b>{h_pentagon}</b>" in rendered
    assert 'href="https://defensenews.com/pentagon-budget"' in rendered
    assert "Defense News" in rendered
    # The duplicate claim sentence is omitted from the body
    plain_narrative = extract_plain_narrative(rendered)
    assert b_pentagon not in plain_narrative


def test_black_sea_story_headlines_underlying_agreement_or_attributed_stats():
    """Requirement 2: Black Sea story headlines agreement between Bulgaria, Romania and Turkey."""
    from geopulse.summarization.service import _derive_headline

    event = _make_event(
        title="Black Sea Naval Mine Action Task Group",
        snippet=(
            "Bulgaria, Romania and Turkey have established a joint naval task group "
            "to counter floating naval mines in the Black Sea."
        ),
        url="https://defensenews.com/black-sea-naval-agreement",
        source_name="Defense News",
    )
    claim = SummaryClaim(
        claim_id="c1",
        text=(
            "Bulgaria, Romania and Turkey signed an agreement to deploy joint naval "
            "countermeasure forces in the Black Sea."
        ),
        evidence_refs=(
            ClaimEvidenceReference(
                source_url="https://defensenews.com/black-sea-naval-agreement",
                evidence_span="Bulgaria, Romania and Turkey signed an agreement",
                verification_status="accepted",
            ),
        ),
        classification="reported_fact",
        central=True,
        verification_status="accepted",
    )
    headline = _derive_headline(claim, event)
    assert "Bulgaria, Romania and Turkey" in headline or "Black Sea" in headline
    assert "Defense News" not in headline
    assert len(headline) >= 15


def test_attribution_preserved_in_headlines_for_third_party_statistics():
    """Requirement 2: Headline preserves attribution when central claim has
    third-party statistics.
    """
    from geopulse.summarization.service import _derive_headline

    event = _make_event(
        title="Black Sea Attacks Hit 570 Vessels",
        snippet=(
            "According to data compiled by Oryx, more than 570 merchant vessels were hit "
            "and at least 40 seafarers were killed in Black Sea strikes."
        ),
        url="https://defensenews.com/black-sea-losses",
        source_name="Defense News",
    )
    claim = SummaryClaim(
        claim_id="c1",
        text="More than 570 merchant vessels were hit in Black Sea strikes.",
        evidence_refs=(
            ClaimEvidenceReference(
                source_url="https://defensenews.com/black-sea-losses",
                evidence_span=(
                    "According to data compiled by Oryx, more than 570 merchant vessels were hit"
                ),
                verification_status="accepted",
            ),
        ),
        classification="attributed_statement",
        attribution="Oryx",
        central=True,
        verification_status="accepted",
    )
    headline = _derive_headline(claim, event)
    # Attribution must be in headline
    assert "Oryx:" in headline or headline.startswith("Oryx")
    # Must not attribute to the reporting publisher Defense News
    assert "Defense News" not in headline


def test_attribution_chain_preserved_for_attributed_statistics():
    """Requirement 2: Preserve attribution chains for third-party statistics."""
    from geopulse.summarization.service import EventSummarizer

    snippet = (
        "According to data compiled by Oryx, Russian armed forces have lost over 3,000 tanks "
        "since the start of the conflict."
    )
    event = _make_event(
        title="Russian Tank Losses Exceed 3,000",
        snippet=snippet,
        url="https://defensenews.com/tank-losses",
        source_name="Defense News",
    )
    raw_claims = [
        {
            "claim_id": "c1",
            "text": "Russian armed forces have lost over 3,000 tanks",
            "src": 1,
            "evidence": (
                "According to data compiled by Oryx, Russian armed forces have lost over "
                "3,000 tanks"
            ),
            "classification": "reported_fact",
            "attribution": None,
            "central": True,
        }
    ]
    parsed, err = EventSummarizer._parse_compact_claims(raw_claims, event)
    assert err is None
    assert parsed is not None
    c1 = parsed[0]
    # Attribution must be retained and classified as attributed_statement
    assert c1.attribution == "Oryx"
    assert c1.classification == "attributed_statement"
    assert "According to Oryx" in c1.text


def test_html_stripping_and_entity_decoding_in_prompt_and_span_match():
    """Requirement 3: Article text is consistently HTML-stripped and entity-decoded."""
    from geopulse.summarization.evidence_matcher import clean_article_text
    from geopulse.summarization.prompts import build_event_prompt

    raw_snippet = (
        "<p>Defence Ministry signed a &pound;400m contract &amp; announced new interceptors.</p>"
    )
    cleaned = clean_article_text(raw_snippet)
    assert "<p>" not in cleaned
    assert "</p>" not in cleaned
    assert "&pound;" not in cleaned
    assert "£400m" in cleaned
    assert "&amp;" not in cleaned
    assert "&" in cleaned

    event = _make_event(
        title="<b>New Procurement Contract</b>",
        snippet=raw_snippet,
        url="https://defensenews.com/procurement",
        source_name="Defense News",
    )
    prompt = build_event_prompt(event)
    assert "<p>" not in prompt
    assert "</p>" not in prompt
    assert "<b>" not in prompt
    assert "&amp;" not in prompt

    # Span match with entity normalization
    evidence_span = "Defence Ministry signed a £400m contract & announced new interceptors."
    status, diff = check_evidence_span_match(evidence_span, raw_snippet)
    assert status in ("verbatim", "normalized")
    assert diff is None

    # Unverifiable / fabricated claims are still strictly quarantined
    fabricated_span = "France acquired nuclear aircraft carriers."
    status, diff = check_evidence_span_match(fabricated_span, raw_snippet)
    assert status == "no_match"
    assert diff is not None


def test_concrete_details_preserved_not_replaced_with_generic_prose():
    """Requirement 4: Concrete specifics (e.g. '12 Ukrainian manufacturers') are preserved."""
    from geopulse.summarization.validator import safe_truncate_headline

    headline_candidate = (
        "UK signs comprehensive agreement to support 12 Ukrainian drone manufacturers "
        "in expanding domestic production"
    )
    # When truncated, it must preserve the concrete number and entity instead of cutting
    truncated = safe_truncate_headline(headline_candidate, max_chars=120)
    assert "12 Ukrainian drone manufacturers" in truncated


def test_useful_narrative_restored_with_distinct_supporting_claim():
    """Requirement 1: Restore useful narratives by extracting a distinct supporting claim.

    Ensures stories are not left headline-only when c1 duplicates the headline.
    """
    from geopulse.summarization.models import EventKeyDetails
    from geopulse.summarization.service import _extract_supporting_claim_from_source

    snippet = (
        "The United Kingdom announced a major military assistance package for Ukraine today. "
        "The package includes 500 armoured vehicles and £200 million in air defence systems. "
        "Deliveries are scheduled to begin early next month."
    )
    art = _make_article(
        title="UK Announces Military Assistance Package for Ukraine",
        snippet=snippet,
        url="https://defensenews.com/uk-package",
    )
    event = _make_event(
        title=art.title,
        snippet=art.snippet,
        url=art.url,
    )

    c1_text = "The United Kingdom announced a military assistance package for Ukraine."
    c2 = _extract_supporting_claim_from_source(art, c1_text, event.title)
    assert c2 is not None
    assert c2.claim_id == "c2"
    assert "500 armoured vehicles" in c2.text or "£200 million" in c2.text
    assert c2.evidence_refs[0].source_url == art.url

    summary = EventSummary(
        event_id=event.id,
        headline=c1_text,
        what_happened=c2.text,
        key_details=EventKeyDetails(),
        confidence_qualification="Single-source reporting; not independently verified.",
        model_used="test-model",
        sources=(
            EventSourceAttribution(
                source_name=art.source_name,
                title=art.title,
                url=art.url,
                published_at=art.published_at,
            ),
        ),
        claims=(
            SummaryClaim(
                claim_id="c1",
                text=c1_text,
                evidence_refs=(
                    ClaimEvidenceReference(
                        source_url=art.url,
                        evidence_span="United Kingdom announced a major military assistance",
                        verification_status="accepted",
                    ),
                ),
                classification="reported_fact",
                central=True,
                verification_status="accepted",
            ),
            c2,
        ),
    )
    rendered = TelegramDigestFormatter().format_event_block(summary, event, "single_source", 1, 1)
    assert "500 armoured vehicles" in rendered or "£200 million" in rendered


def test_black_sea_ft_ambrey_imo_attribution_and_intel_sharing_headline():
    """Requirement 2: Preserve FT attribution / Ambrey data and Black Sea intel sharing headline."""
    from geopulse.summarization.service import _derive_headline

    # 1. Black Sea maritime intelligence sharing development
    art1 = _make_article(
        title="Bulgaria, Romania and Turkey Agree on Black Sea Maritime Intelligence Sharing",
        snippet=(
            "Bulgaria, Romania and Turkey agreed on Wednesday to establish a joint maritime "
            "intelligence sharing framework to monitor naval traffic in the Black Sea."
        ),
        url="https://reuters.com/black-sea-intel",
        source_name="Reuters",
    )
    event1 = _make_event(
        title=art1.title,
        snippet=art1.snippet,
        url=art1.url,
        source_name=art1.source_name,
        category="military_operations",
    )
    c1 = SummaryClaim(
        claim_id="c1",
        text="Bulgaria, Romania and Turkey agreed to share maritime intelligence in the Black Sea.",
        evidence_refs=(
            ClaimEvidenceReference(
                source_url=art1.url,
                evidence_span=(
                    "agreed on Wednesday to establish a joint maritime intelligence sharing"
                ),
                verification_status="accepted",
            ),
        ),
        classification="reported_fact",
        central=True,
        verification_status="accepted",
    )
    hl1 = _derive_headline(c1, event1)
    assert "Maritime Intelligence Sharing" in hl1
    assert "Bulgaria, Romania" in hl1

    # 2. Financial Times citing Ambrey / IMO data
    art2 = _make_article(
        title="Black Sea Shipping Faces Escalating Threats",
        snippet=(
            "According to the Financial Times, citing data from maritime security firm Ambrey, "
            "more than 40 commercial vessels were rerouted after drone strikes near Odesa."
        ),
        url="https://defensenews.com/black-sea-shipping",
        source_name="Defense News",
    )
    event2 = _make_event(
        title=art2.title,
        snippet=art2.snippet,
        url=art2.url,
        source_name=art2.source_name,
        category="security_incident",
    )
    c2 = SummaryClaim(
        claim_id="c1",
        text="More than 40 commercial vessels were rerouted after drone strikes near Odesa.",
        evidence_refs=(
            ClaimEvidenceReference(
                source_url=art2.url,
                evidence_span="Financial Times, citing data from maritime security firm Ambrey",
                verification_status="accepted",
            ),
        ),
        classification="attributed_statement",
        attribution="Financial Times / Ambrey",
        central=True,
        verification_status="accepted",
    )
    hl2 = _derive_headline(c2, event2)
    assert "Financial Times" in hl2
    assert "Ambrey" in hl2


def test_uk_offset_policy_distinguishes_contract_threshold_from_benefits():
    """Requirement 3: Distinguish £50m contract threshold from required industrial benefits."""
    from geopulse.summarization.service import _normalize_uk_offset_claim

    confused_text = (
        "The new UK defence offset policy requires foreign contractors to provide £50 million "
        "in industrial benefits."
    )
    normalized = _normalize_uk_offset_claim(confused_text)
    assert "contracts valued at £50 million or more" in normalized
    assert "provide £50 million in industrial benefits" not in normalized

    confused_2 = "UK MoD mandates £50m of economic offsets for overseas suppliers."
    normalized_2 = _normalize_uk_offset_claim(confused_2)
    assert "contracts valued at £50 million or more" in normalized_2
    assert "mandates £50m of economic offsets" not in normalized_2


def test_pentagon_budget_preserves_both_mip_and_nip_context():
    """Requirement 4: Pentagon budget preserves verified context of both MIP and NIP."""
    from geopulse.summarization.service import _extract_supporting_claim_from_source

    snippet = (
        "The Pentagon announced a $30.9 billion budget request for the "
        "Military Intelligence Program. "
        "Separately, the National Intelligence Program request totals $78.1 billion for FY2027. "
        "Total intelligence spending requested reaches $109 billion."
    )
    art = _make_article(
        title="Pentagon Requests $30.9B for Military Intelligence in $109B US Intel Budget",
        snippet=snippet,
        url="https://breakingdefense.com/mip-nip-request",
        source_name="Breaking Defense",
    )
    c1_text = "The Pentagon requested $30.9 billion for the Military Intelligence Program."
    supp = _extract_supporting_claim_from_source(art, c1_text, art.title)
    assert supp is not None
    assert "78.1 billion" in supp.text or "National Intelligence Program" in supp.text
    assert supp.evidence_refs[0].source_url == art.url


def test_hormuz_story_source_association_prevents_misattribution_to_al_jazeera():
    """Requirement 5: Check Hormuz claim against linked article and prevent misattribution."""
    from geopulse.summarization.service import _find_best_article_for_claim

    bbc_art = _make_article(
        title="UK Navy Reports Tanker Boarded in Strait of Hormuz",
        snippet=(
            "Armed men boarded a commercial oil tanker in the Strait of Hormuz on Tuesday, "
            "maritime security monitors reported."
        ),
        url="https://bbc.com/news/hormuz-tanker",
        source_name="BBC News",
    )
    aljazeera_art = _make_article(
        title="GCC Leaders Meet in Doha for Regional Security Summit",
        snippet=(
            "Gulf Cooperation Council leaders gathered in Doha on Tuesday to discuss "
            "diplomatic cooperation and regional security."
        ),
        url="https://aljazeera.com/news/doha-summit",
        source_name="Al Jazeera",
    )

    all_articles = (bbc_art, aljazeera_art)
    claim_text = "Armed men boarded a commercial oil tanker in the Strait of Hormuz."
    evidence_span = "Armed men boarded a commercial oil tanker in the Strait of Hormuz"

    # Even if model paired the claim with Al Jazeera (candidate_article = aljazeera_art):
    best_art = _find_best_article_for_claim(
        claim_text=claim_text,
        evidence_span=evidence_span,
        candidate_article=aljazeera_art,
        all_articles=all_articles,
    )
    assert best_art.url == bbc_art.url
    assert best_art.source_name == "BBC News"
    assert best_art.url != aljazeera_art.url


def test_hanwha_lsm_award_source_supported_terminology():
    """Regression test: Hanwha $269M LSM award terminology.

    - Ensures LSM is not expanded as LSTH or Landing Ship, Tank, Helicopter.
    - Preserves source-supported Landing Ship Medium (LSM) when source provides expansion.
    - Preserves bare LSM without expansion when source does not provide an expansion.
    - Validates that ungrounded LSTH / Landing Ship, Tank, Helicopter is quarantined.
    """
    from geopulse.summarization.models import EventKeyDetails
    from geopulse.summarization.service import _derive_headline
    from geopulse.summarization.validator import (
        sanitize_technical_acronyms,
        validate_summary_grounding,
    )

    # 1. Source explicitly expands LSM as Landing Ship Medium
    art_with_expansion = _make_article(
        title="Hanwha Ocean Wins $269M Award to Build Landing Ship Medium (LSM)",
        snippet=(
            "South Korea's Hanwha Ocean has secured a $269 million contract to build "
            "Landing Ship Medium (LSM) vessels for the U.S. Navy."
        ),
        url="https://breakingdefense.com/hanwha-lsm",
        source_name="Breaking Defense",
    )
    event_with_expansion = _make_event(
        title=art_with_expansion.title,
        snippet=art_with_expansion.snippet,
        url=art_with_expansion.url,
        source_name=art_with_expansion.source_name,
    )

    # If model hallucinates LSTH (Landing Ship, Tank, Helicopter):
    hallucinated_claim = (
        "Hanwha Ocean won a $269 million award to build "
        "LSTH (Landing Ship, Tank, Helicopter) vessels."
    )
    source_text = f"{art_with_expansion.title} {art_with_expansion.snippet}"
    sanitized = sanitize_technical_acronyms(hallucinated_claim, source_text)
    assert "LSTH" not in sanitized
    assert "Landing Ship, Tank, Helicopter" not in sanitized
    assert "LSM" in sanitized
    assert "Landing Ship Medium" in sanitized

    # 2. Source mentions LSM but does NOT provide an expansion
    art_no_expansion = _make_article(
        title="Hanwha Ocean Secures $269M Award for Navy LSM Program",
        snippet=(
            "Hanwha Ocean was awarded a $269 million contract to construct "
            "LSM vessels for the U.S. Navy."
        ),
        url="https://breakingdefense.com/hanwha-lsm-award",
        source_name="Breaking Defense",
    )
    event_no_expansion = _make_event(
        title=art_no_expansion.title,
        snippet=art_no_expansion.snippet,
        url=art_no_expansion.url,
        source_name=art_no_expansion.source_name,
    )
    source_no_expansion = f"{art_no_expansion.title} {art_no_expansion.snippet}"
    sanitized_no_exp = sanitize_technical_acronyms(hallucinated_claim, source_no_expansion)
    assert "LSTH" not in sanitized_no_exp
    assert "Landing Ship, Tank, Helicopter" not in sanitized_no_exp
    assert "Landing Ship Medium" not in sanitized_no_exp
    assert "LSM" in sanitized_no_exp

    # 3. Grounding validator strictly quarantines invented LSTH expansion
    bad_summary = EventSummary(
        event_id=event_with_expansion.id,
        headline="Hanwha Wins $269M Award for LSTH",
        what_happened="Hanwha won a $269M award to build LSTH (Landing Ship, Tank, Helicopter).",
        key_details=EventKeyDetails(),
        confidence_qualification="Single-source reporting; not independently verified.",
        model_used="test-model",
        sources=(
            EventSourceAttribution(
                source_name=art_with_expansion.source_name,
                title=art_with_expansion.title,
                url=art_with_expansion.url,
                published_at=art_with_expansion.published_at,
            ),
        ),
    )
    val_res = validate_summary_grounding(bad_summary, event_with_expansion)
    assert not val_res.is_valid
    assert any(
        "lsth" in str(e).lower() or "landing ship, tank, helicopter" in str(e).lower()
        for e in (*val_res.unsupported_entities, *val_res.unsupported_claims)
    )

    # 4. Headline derivation preserves correct LSM terminology
    claim_correct = SummaryClaim(
        claim_id="c1",
        text=(
            "Hanwha Ocean was awarded a $269 million contract for "
            "Landing Ship Medium (LSM) vessels."
        ),
        evidence_refs=(
            ClaimEvidenceReference(
                source_url=art_with_expansion.url,
                evidence_span="Hanwha Ocean has secured a $269 million contract",
                verification_status="accepted",
            ),
        ),
        classification="reported_fact",
        central=True,
        verification_status="accepted",
    )
    hl = _derive_headline(claim_correct, event_with_expansion)
    assert "LSTH" not in hl
    assert "LSM" in hl

    hl_no_exp = _derive_headline(claim_correct, event_no_expansion)
    assert "LSTH" not in hl_no_exp
    assert "LSM" in hl_no_exp


def test_single_source_versus_independently_corroborated_claims():
    """Requirement 1: Single-source reporting vs corroborated reporting.

    Corroboration requires multiple genuinely independent sources.
    Single-source reporting must never imply independent verification.
    """
    from geopulse.delivery.formatter import TelegramDigestFormatter
    from geopulse.summarization.validator import (
        are_sources_genuinely_independent,
        count_genuinely_independent_sources,
    )

    art1 = _make_article(
        title="Bulgaria and Romania Ratify Black Sea Intelligence Pact",
        snippet=(
            "Bulgaria and Romania formally signed a joint maritime surveillance agreement today."
        ),
        url="https://reuters.com/world/black-sea-pact",
        source_name="Reuters",
    )
    art2 = _make_article(
        title="Naval Intelligence Sharing Framework Formed in Black Sea",
        snippet="Naval forces in the Black Sea established a tripartite intelligence center.",
        url="https://defensenews.com/naval/black-sea-intel",
        source_name="Defense News",
    )

    # 1. Independent sources: genuinely corroborated
    assert are_sources_genuinely_independent([art1, art2])
    assert count_genuinely_independent_sources([art1, art2]) == 2

    c_corrob = SummaryClaim(
        claim_id="c1",
        text="Bulgaria and Romania signed a joint maritime surveillance agreement.",
        evidence_refs=(
            ClaimEvidenceReference(
                source_url=art1.url,
                evidence_span=(
                    "Bulgaria and Romania formally signed a joint maritime surveillance agreement"
                ),
                verification_status="accepted",
            ),
            ClaimEvidenceReference(
                source_url=art2.url,
                evidence_span=(
                    "Naval forces in the Black Sea established a tripartite intelligence center"
                ),
                verification_status="accepted",
            ),
        ),
        classification="reported_fact",
        central=True,
        verification_status="accepted",
    )
    summary_corrob = EventSummary(
        event_id="e1",
        headline="Bulgaria and Romania Sign Maritime Intelligence Agreement",
        what_happened="Bulgaria and Romania established a joint maritime surveillance framework.",
        confidence_qualification=(
            "Corroborated across 2 independent reporting outlets; "
            "claims remain subject to verification."
        ),
        sources=(
            EventSourceAttribution(source_name=art1.source_name, title=art1.title, url=art1.url),
            EventSourceAttribution(source_name=art2.source_name, title=art2.title, url=art2.url),
        ),
        claims=(c_corrob,),
        corroboration_status="independently_corroborated",
        independent_chains_count=2,
        model_used="test-model",
    )
    event_corrob = _make_event(
        title=art1.title,
        snippet=art1.snippet,
        url=art1.url,
        confidence_label="corroborated",
        source_count=2,
    )
    rendered_corrob = TelegramDigestFormatter().format_event_block(
        summary_corrob, event_corrob, "corroborated", 1, 1
    )
    assert "[CORROBORATED]" in rendered_corrob

    # 2. Single source: never implies independent verification
    assert not are_sources_genuinely_independent([art1])
    assert count_genuinely_independent_sources([art1]) == 1

    summary_single = summary_corrob.model_copy(
        update={
            "sources": (
                EventSourceAttribution(
                    source_name=art1.source_name, title=art1.title, url=art1.url
                ),
            ),
            "claims": (
                c_corrob.model_copy(update={"evidence_refs": (c_corrob.evidence_refs[0],)}),
            ),
            "confidence_qualification": "Single-source reporting; not independently verified.",
            "corroboration_status": "single_source",
            "independent_chains_count": 1,
        }
    )
    event_single = _make_event(
        title=art1.title,
        snippet=art1.snippet,
        url=art1.url,
        confidence_label="single_source",
        source_count=1,
    )
    rendered_single = TelegramDigestFormatter().format_event_block(
        summary_single, event_single, "single_source", 1, 1
    )
    assert "[CORROBORATED]" not in rendered_single
    assert "[SINGLE SOURCE]" in rendered_single or "[REPORT]" in rendered_single
    assert (
        "Single-source reporting" in rendered_single
        or "unverified independently" in rendered_single
    )


def test_syndicated_articles_not_counted_as_independent_sources():
    """Requirement 1 & 5: Syndicated copies or shared origins are not independent corroboration.

    Do not count syndicated copies, duplicated articles, or multiple URLs from the same
    underlying report as independent corroboration.
    """
    from geopulse.delivery.formatter import TelegramDigestFormatter
    from geopulse.summarization.validator import (
        are_sources_genuinely_independent,
        count_genuinely_independent_sources,
    )

    # Case A: AP wire story syndicated by Yahoo News with wire dateline
    ap_art = _make_article(
        title="Pentagon Requests $30.9B for Military Intelligence",
        snippet=(
            "WASHINGTON (AP) — The Pentagon requested $30.9 billion for the "
            "Military Intelligence Program."
        ),
        url="https://apnews.com/article/pentagon-intel-budget",
        source_name="Associated Press",
    )
    yahoo_art = _make_article(
        title="Pentagon Requests $30.9B for Military Intelligence",
        snippet=(
            "(AP) — The Pentagon requested $30.9 billion for the Military Intelligence Program."
        ),
        url="https://finance.yahoo.com/news/pentagon-intel-budget",
        source_name="Yahoo News",
    )
    assert not are_sources_genuinely_independent([ap_art, yahoo_art])
    assert count_genuinely_independent_sources([ap_art, yahoo_art]) == 1

    # Case B: Multiple URLs from the same registered domain
    reuters1 = _make_article(
        title="Diplomatic Talks Resume in Geneva",
        snippet="Diplomats convened in Geneva to resume ceasefire discussions.",
        url="https://reuters.com/world/europe/talks-resume-1",
        source_name="Reuters",
    )
    reuters2 = _make_article(
        title="Geneva Ceasefire Negotiations Underway",
        snippet="Negotiators met in Geneva on Tuesday for bilateral security talks.",
        url="https://reuters.com/world/europe/talks-resume-2",
        source_name="Reuters UK",
    )
    assert not are_sources_genuinely_independent([reuters1, reuters2])
    assert count_genuinely_independent_sources([reuters1, reuters2]) == 1

    # Case C: Secondary outlet explicitly citing primary outlet
    def_art = _make_article(
        title="Maritime Monitoring Near Odesa",
        snippet=(
            "According to the Financial Times, citing Ambrey data, "
            "commercial vessels were rerouted."
        ),
        url="https://defensenews.com/black-sea-report",
        source_name="Defense News",
    )
    ft_art = _make_article(
        title="Black Sea Shipping Faces Drone Threat",
        snippet="Commercial shipping traffic was rerouted in the Black Sea after drone strikes.",
        url="https://ft.com/content/black-sea-shipping",
        source_name="Financial Times",
    )
    assert not are_sources_genuinely_independent([def_art, ft_art])
    assert count_genuinely_independent_sources([def_art, ft_art]) == 1

    # Ensure formatter downgrades syndicated claim from 'corroborated' to 'single_source'
    c_synd = SummaryClaim(
        claim_id="c1",
        text="The Pentagon requested $30.9 billion for military intelligence.",
        evidence_refs=(
            ClaimEvidenceReference(
                source_url=ap_art.url,
                evidence_span="Pentagon requested $30.9 billion",
                verification_status="accepted",
            ),
            ClaimEvidenceReference(
                source_url=yahoo_art.url,
                evidence_span="Pentagon requested $30.9 billion",
                verification_status="accepted",
            ),
        ),
        classification="reported_fact",
        central=True,
        verification_status="accepted",
    )
    summary_synd = EventSummary(
        event_id="e_synd",
        headline="Pentagon Requests $30.9B for Military Intelligence",
        what_happened="The Pentagon announced a $30.9 billion budget request.",
        confidence_qualification=(
            "Single-origin reporting across outlets; unverified independently."
        ),
        sources=(
            EventSourceAttribution(
                source_name=ap_art.source_name, title=ap_art.title, url=ap_art.url
            ),
            EventSourceAttribution(
                source_name=yahoo_art.source_name, title=yahoo_art.title, url=yahoo_art.url
            ),
        ),
        claims=(c_synd,),
        corroboration_status="multiple_reports_shared_origin",
        independent_chains_count=1,
        model_used="test-model",
    )
    event_synd = _make_event(
        title=ap_art.title,
        snippet=ap_art.snippet,
        url=ap_art.url,
        confidence_label="corroborated",  # Clustering suggested corroborated based on raw count
        source_count=2,
    )
    # Formatter must catch the syndication and downgrade badge to [REPORT] or [SINGLE SOURCE]
    rendered = TelegramDigestFormatter().format_event_block(
        summary_synd, event_synd, "corroborated", 1, 1
    )
    assert "[CORROBORATED]" not in rendered
    assert (
        "[OFFICIAL ANNOUNCEMENT]" in rendered
        or "[SHARED ORIGIN]" in rendered
        or "[SINGLE SOURCE]" in rendered
        or "[REPORT]" in rendered
    )
    assert "Single-origin reporting across outlets" in rendered


def test_attribution_of_government_and_military_statements():
    """Requirement 1 & 5: Attribution of statements by governments and military organisations.

    Never present belligerent statements or military claims as independently established facts.
    Preserve attribution in both narrative and headline.
    """
    from geopulse.summarization.service import _derive_headline

    art = _make_article(
        title="Air Defences Active in Isfahan Region",
        snippet=(
            "Iranian state media reported that air defences intercepted three drones "
            "near an air base."
        ),
        url="https://aljazeera.com/news/iran-air-defences",
        source_name="Al Jazeera",
    )
    event = _make_event(title=art.title, snippet=art.snippet, url=art.url)

    # 1. Attributed military claim
    c1 = SummaryClaim(
        claim_id="c1",
        text="Air defences intercepted three unmanned aerial vehicles near an air base.",
        evidence_refs=(
            ClaimEvidenceReference(
                source_url=art.url,
                evidence_span=(
                    "Iranian state media reported that air defences intercepted three drones"
                ),
                verification_status="accepted",
            ),
        ),
        classification="attributed_statement",
        attribution="Iranian military",
        central=True,
        verification_status="accepted",
    )

    hl = _derive_headline(c1, event)
    assert "Iranian military" in hl or "Iran" in hl

    # 2. Foreign ministry statement
    c2 = SummaryClaim(
        claim_id="c1",
        text="Russian forces established complete control over a settlement in Donetsk.",
        evidence_refs=(
            ClaimEvidenceReference(
                source_url=art.url,
                evidence_span="air defences intercepted three drones",
                verification_status="accepted",
            ),
        ),
        classification="attributed_statement",
        attribution="Russian Defence Ministry",
        central=True,
        verification_status="accepted",
    )
    hl2 = _derive_headline(c2, event)
    assert "Russian Defence Ministry" in hl2 or "Russia" in hl2


def test_explicitly_unknown_details_versus_missing_information():
    """Requirement 1 & 5: Explicitly unknown details vs. missing information.

    Report unknown or unconfirmed details only when the source explicitly establishes
    that the detail is unknown, unclear or unverified. Absence of evidence must not be
    converted into an assertion that something is unknown.
    """
    from geopulse.summarization.validator import validate_summary_grounding

    # Case A: Source explicitly states uncertainty
    art_with_uncertainty = _make_article(
        title="Fire Breaks Out at Central Gaza Shelter",
        snippet=(
            "A fire broke out at a displacement shelter in Deir al-Balah, killing "
            "three people. The cause of the fire was unknown."
        ),
        url="https://bbc.com/news/gaza-shelter-fire",
        source_name="BBC News",
    )
    event_with_unc = _make_event(
        title=art_with_uncertainty.title,
        snippet=art_with_uncertainty.snippet,
        url=art_with_uncertainty.url,
    )
    summary_valid = EventSummary(
        event_id=event_with_unc.id,
        headline="Fire at Deir al-Balah Shelter Kills Three",
        what_happened=(
            "A fire broke out at a shelter in Deir al-Balah, killing three people, "
            "while the cause remains unknown."
        ),
        confidence_qualification="Single-source reporting; not independently verified.",
        sources=(
            EventSourceAttribution(
                source_name=art_with_uncertainty.source_name,
                title=art_with_uncertainty.title,
                url=art_with_uncertainty.url,
            ),
        ),
        model_used="test-model",
    )
    res_valid = validate_summary_grounding(summary_valid, event_with_unc)
    assert res_valid.is_valid

    # Case B: Source merely omits who carried out an attack; model falsely asserts it is unknown
    art_no_uncertainty = _make_article(
        title="Depot Struck Near Kharkiv on Tuesday",
        snippet="An industrial depot was damaged in an early morning missile strike near Kharkiv.",
        url="https://reuters.com/world/kharkiv-strike",
        source_name="Reuters",
    )
    event_no_unc = _make_event(
        title=art_no_uncertainty.title,
        snippet=art_no_uncertainty.snippet,
        url=art_no_uncertainty.url,
    )
    summary_invented_unc = EventSummary(
        event_id=event_no_unc.id,
        headline="Industrial Depot Damaged Near Kharkiv",
        what_happened="An industrial depot was damaged near Kharkiv; the perpetrator is unknown.",
        confidence_qualification="Single-source reporting; not independently verified.",
        sources=(
            EventSourceAttribution(
                source_name=art_no_uncertainty.source_name,
                title=art_no_uncertainty.title,
                url=art_no_uncertainty.url,
            ),
        ),
        model_used="test-model",
    )
    res_invalid = validate_summary_grounding(summary_invented_unc, event_no_unc)
    assert not res_valid.is_valid or not res_invalid.is_valid
    assert any("unknown or unclear" in str(c).lower() for c in res_invalid.unsupported_claims)


def test_multiple_casualty_incidents_represented_accurately():
    """Requirement 3: Gaza reporting regression scenario.

    Preserve distinction between different reported casualties and their circumstances.
    Do not imply that all casualties resulted from the same attack.
    Preserve explicitly stated uncertainty (e.g. unknown cause of a fire).
    """
    from geopulse.summarization.service import _extract_supporting_claim_from_source

    snippet = (
        "Medical officials reported that an Israeli airstrike killed 14 people in "
        "Jabalia in northern Gaza. "
        "Separately, eight Palestinians were killed in an artillery strike in Khan Younis. "
        "In central Gaza, three people died in a shelter fire of unknown cause."
    )
    art = _make_article(
        title="Strikes in Gaza Kill 22; Three Dead in Shelter Fire",
        snippet=snippet,
        url="https://aljazeera.com/news/gaza-fatalities",
        source_name="Al Jazeera",
    )

    c1_text = "An Israeli airstrike killed 14 people in Jabalia in northern Gaza."
    supp = _extract_supporting_claim_from_source(art, c1_text, art.title)
    assert supp is not None
    # Supporting claim must capture the distinct incident (Khan Younis or shelter fire) with figures
    assert any(term in supp.text for term in ("Khan Younis", "eight", "fire", "unknown cause"))
    # Must not claim 25 died in a single strike
    assert "25 killed in an airstrike" not in supp.text


def test_separation_of_related_but_distinct_diplomatic_and_defence_agreements():
    """Requirement 3: Qatar, Turkey and regional defence cooperation scenario.

    Distinguish domestic defence-industrial developments, drone capabilities,
    and separate diplomatic agreements.
    Do not conflate the Mecca Pact with the Badr defence-industrial cooperation arrangement.
    """
    from geopulse.summarization.service import _extract_supporting_claim_from_source

    snippet = (
        "Turkey and Qatar signed the Mecca Pact to formalize diplomatic "
        "coordination across the Gulf. "
        "Under a separate agreement, the two nations established the Badr "
        "cooperation arrangement to co-produce drones. "
        "Turkish aerospace manufacturers also expanded domestic production of unmanned systems."
    )
    art = _make_article(
        title="Turkey and Qatar Sign Diplomatic and Defense Agreements",
        snippet=snippet,
        url="https://breakingdefense.com/turkey-qatar-pacts",
        source_name="Breaking Defense",
    )

    c1_text = "Turkey and Qatar signed the Mecca Pact to formalize diplomatic coordination."
    supp = _extract_supporting_claim_from_source(art, c1_text, art.title)
    assert supp is not None
    # Must preserve the distinct Badr drone co-production arrangement
    assert "Badr" in supp.text or "drones" in supp.text
    # Must not falsely assign drone co-production to the Mecca Pact
    assert "Mecca Pact to co-produce drones" not in supp.text


def test_concise_narrative_with_distinct_verified_supporting_details():
    """Requirement 2: Narratives must prefer concrete operational details over generic filler.

    Avoid generic commentary like 'countries are bolstering their capabilities'.
    Prefer concrete figures, entities, and operational mechanisms.
    """
    from geopulse.summarization.service import _extract_supporting_claim_from_source

    snippet = (
        "The bilateral defense treaty was ratified by parliament today. "
        "Countries are bolstering their capabilities amid growing regional tensions. "
        "The agreement allocates $450 million for 15 joint air defense radar installations."
    )
    art = _make_article(
        title="Parliament Ratifies $450M Air Defense Radar Pact",
        snippet=snippet,
        url="https://defensenews.com/radar-pact",
        source_name="Defense News",
    )

    c1_text = "The bilateral defense treaty was ratified by parliament today."
    supp = _extract_supporting_claim_from_source(art, c1_text, art.title)
    assert supp is not None
    # Must select the concrete financial and operational details ($450 million, radar installations)
    assert "$450 million" in supp.text or "radar installations" in supp.text
    # Must NOT select the generic commentary sentence
    assert "bolstering their capabilities" not in supp.text


def test_prevention_of_headline_body_repetition_and_article_text_leakage():
    """Requirement 2: Prevent exact headline/body duplication and article snippet leakage."""
    from geopulse.summarization.publication_gate import is_headline_body_duplicate

    headline = "UK Signs £50 Million Defense Agreement With Ukraine"
    exact_body = "UK signs £50 million defense agreement with Ukraine"
    assert is_headline_body_duplicate(exact_body, headline)

    semantic_body = "The UK has signed a defense agreement valued at £50 million with Ukraine."
    assert is_headline_body_duplicate(semantic_body, headline)

    distinct_body = (
        "Under the agreement, 12 Ukrainian drone manufacturers will receive "
        "direct industrial assistance."
    )
    assert not is_headline_body_duplicate(distinct_body, headline)


def test_attribution_rendering_normalizes_dict_objects():
    """Observed issue 1: Dict attributions must be normalized and never leaked to Telegram."""
    from geopulse.delivery.formatter import TelegramDigestFormatter
    from geopulse.summarization.publication_gate import (
        normalize_attribution_text,
        validate_publication_gate,
    )

    raw_dict = {
        "name": "Dale Marks, Assistant Secretary of the Air Force for Acquisition",
        "organization": "Pentagon",
    }
    normalized = normalize_attribution_text(raw_dict)
    assert normalized == (
        "Dale Marks, Assistant Secretary of the Air Force for Acquisition (Pentagon)"
    )

    str_dict = (
        "{'name': 'Dale Marks, Assistant Secretary of the Air Force for Acquisition', "
        "'organization': 'Pentagon'}"
    )
    normalized_str = normalize_attribution_text(str_dict)
    assert "Dale Marks" in normalized_str
    assert "Pentagon" in normalized_str
    assert "{" not in normalized_str
    assert "}" not in normalized_str

    event = _make_event(
        title="Pentagon Outlines Next-Generation Acquisition Priorities",
        snippet="Dale Marks announced new procurement reforms at the defense symposium.",
        url="https://defensenews.com/pentagon-acquisition",
    )
    summary = EventSummary(
        event_id="evt-attr-dict",
        headline="Pentagon Outlines Next-Generation Acquisition Priorities",
        what_happened=(
            "Procurement procedures will undergo accelerated prototyping under new guidelines."
        ),
        confidence_qualification="Single-source reporting; unverified independently.",
        sources=(
            EventSourceAttribution(
                source_name="Defense News",
                title="Pentagon Outlines Next-Generation Acquisition Priorities",
                url="https://defensenews.com/pentagon-acquisition",
                published_at=DT,
            ),
        ),
        claims=(
            SummaryClaim(
                claim_id="c1",
                text=(
                    "Procurement procedures will undergo accelerated prototyping under guidelines."
                ),
                attribution=str_dict,
                classification="reported_fact",
                central=True,
                verification_status="accepted",
                evidence_refs=(
                    ClaimEvidenceReference(
                        source_url="https://defensenews.com/pentagon-acquisition",
                        evidence_span="Dale Marks announced new procurement reforms",
                        verification_status="accepted",
                    ),
                ),
            ),
        ),
        model_used="test-model",
    )

    gate_res = validate_publication_gate(summary, event)
    assert gate_res.passed
    clean_summary = gate_res.repaired_summary or summary
    assert "{" not in clean_summary.claims[0].attribution
    assert "}" not in clean_summary.claims[0].attribution

    formatter = TelegramDigestFormatter()
    rendered_block = formatter.format_event_block(
        clean_summary,
        category=event.category,
        confidence_label=event.confidence_label,
        rank=1,
        total=1,
    )
    assert "{'name':" not in rendered_block
    assert "'organization':" not in rendered_block
    assert "Dale Marks" in rendered_block or "Pentagon" in rendered_block

    from geopulse.summarization.models import SummarizationResult

    res = SummarizationResult(
        event_id=event.id,
        status="success",
        summary=clean_summary,
        event=event,
    )
    rendered_chunks = formatter.format_digest(results=[res])
    digest_text = "\n".join(rendered_chunks)
    assert "{'name':" not in digest_text
    assert "'organization':" not in digest_text
    assert "Dale Marks" in digest_text or "Pentagon" in digest_text


def test_hanwha_tote_services_lsm_contract_relationship():
    """Observed issue 2: Must preserve roles of Navy, TOTE Services, and Hanwha Defense USA."""
    from geopulse.summarization.validator import validate_summary_grounding

    source_text = (
        "TOTE Services selected Hanwha Defense USA to build auxiliary vessels for the U.S. Navy "
        "and Marine Corps Landing Ship Medium (LSM) programme under an initial subcontract."
    )
    event = _make_event(
        title="TOTE Services Selects Hanwha Defense USA for Navy LSM Programme",
        snippet=source_text,
        url="https://breakingdefense.com/hanwha-lsm",
        source_name="Breaking Defense",
    )

    # Inaccurate: claims US Navy directly awarded Hanwha the contract
    inaccurate_summary = EventSummary(
        event_id="evt-lsm-inaccurate",
        headline="US Navy Awards Hanwha Defense Contract for Landing Ships",
        what_happened="The US Navy awarded Hanwha Defense USA a contract to build LSM vessels.",
        confidence_qualification="Single-source reporting; unverified independently.",
        sources=(
            EventSourceAttribution(
                source_name="Breaking Defense",
                title=event.title,
                url="https://breakingdefense.com/hanwha-lsm",
                published_at=DT,
            ),
        ),
        claims=(
            SummaryClaim(
                claim_id="c1",
                text="The US Navy awarded Hanwha Defense USA a contract to build LSM vessels.",
                classification="reported_fact",
                central=True,
                evidence_refs=(
                    ClaimEvidenceReference(
                        source_url="https://breakingdefense.com/hanwha-lsm",
                        evidence_span="TOTE Services selected Hanwha Defense USA to build",
                        verification_status="accepted",
                    ),
                ),
            ),
        ),
        model_used="test-model",
    )

    val_res = validate_summary_grounding(inaccurate_summary, event)
    assert not val_res.is_valid
    assert any("TOTE Services" in err for err in val_res.unsupported_claims)

    # Accurate: preserves TOTE Services selecting Hanwha for the Navy LSM program
    accurate_summary = EventSummary(
        event_id="evt-lsm-accurate",
        headline="TOTE Services Selects Hanwha Defense USA for Navy LSM Programme",
        what_happened=(
            "TOTE Services selected Hanwha Defense USA for the Navy and Marine Corps LSM programme."
        ),
        confidence_qualification="Single-source reporting; unverified independently.",
        sources=(
            EventSourceAttribution(
                source_name="Breaking Defense",
                title=event.title,
                url="https://breakingdefense.com/hanwha-lsm",
                published_at=DT,
            ),
        ),
        claims=(
            SummaryClaim(
                claim_id="c1",
                text=(
                    "TOTE Services selected Hanwha Defense USA for the Navy and "
                    "Marine Corps LSM programme."
                ),
                classification="reported_fact",
                central=True,
                evidence_refs=(
                    ClaimEvidenceReference(
                        source_url="https://breakingdefense.com/hanwha-lsm",
                        evidence_span="TOTE Services selected Hanwha Defense USA to build",
                        verification_status="accepted",
                    ),
                ),
            ),
        ),
        model_used="test-model",
    )

    val_res_ok = validate_summary_grounding(accurate_summary, event)
    assert val_res_ok.is_valid


def test_patriot_fms_approval_distinguished_from_delivery():
    """Observed issue 3: Distinguish State Department FMS approval from completed delivery."""
    from geopulse.summarization.validator import validate_summary_grounding

    source_text = (
        "The State Department has approved a possible Foreign Military Sale of 500 Patriot PAC-3 "
        "missiles and related equipment to European allies with allied funding."
    )
    event = _make_event(
        title="State Department Approves Potential $5B Patriot Missile Sale",
        snippet=source_text,
        url="https://defensenews.com/patriot-fms",
        source_name="Defense News",
    )

    # Inaccurate: states missiles were delivered or supplied
    inaccurate_summary = EventSummary(
        event_id="evt-patriot-delivered",
        headline="US Delivers 500 Patriot Missiles to European Allies",
        what_happened="The United States has completed delivery of 500 Patriot missiles to allies.",
        confidence_qualification="Single-source reporting; unverified independently.",
        sources=(
            EventSourceAttribution(
                source_name="Defense News",
                title=event.title,
                url="https://defensenews.com/patriot-fms",
                published_at=DT,
            ),
        ),
        claims=(
            SummaryClaim(
                claim_id="c1",
                text="The United States has completed delivery of 500 Patriot missiles to allies.",
                classification="reported_fact",
                central=True,
                evidence_refs=(
                    ClaimEvidenceReference(
                        source_url="https://defensenews.com/patriot-fms",
                        evidence_span="State Department has approved a possible Foreign Military",
                        verification_status="accepted",
                    ),
                ),
            ),
        ),
        model_used="test-model",
    )

    val_res = validate_summary_grounding(inaccurate_summary, event)
    assert not val_res.is_valid
    assert any("FMS approval" in err or "delivery" in err for err in val_res.unsupported_claims)

    # Accurate: preserves FMS approval status with 500 missiles and allied funding
    accurate_summary = EventSummary(
        event_id="evt-patriot-approved",
        headline="State Department Approves Potential Sale of 500 Patriot Missiles",
        what_happened=(
            "The State Department approved a potential Foreign Military Sale of 500 Patriot "
            "missiles to European allies funded through allied contributions."
        ),
        confidence_qualification="Single-source reporting; unverified independently.",
        sources=(
            EventSourceAttribution(
                source_name="Defense News",
                title=event.title,
                url="https://defensenews.com/patriot-fms",
                published_at=DT,
            ),
        ),
        claims=(
            SummaryClaim(
                claim_id="c1",
                text=(
                    "The State Department approved a potential Foreign Military Sale of 500 "
                    "Patriot missiles to European allies funded through allied contributions."
                ),
                classification="reported_fact",
                central=True,
                evidence_refs=(
                    ClaimEvidenceReference(
                        source_url="https://defensenews.com/patriot-fms",
                        evidence_span="State Department has approved a possible Foreign Military",
                        verification_status="accepted",
                    ),
                ),
            ),
        ),
        model_used="test-model",
    )

    val_res_ok = validate_summary_grounding(accurate_summary, event)
    assert val_res_ok.is_valid


def test_black_sea_narrative_prefers_maritime_agreement_over_incidental_statistics():
    """Observed issue 4: Prefer Bulgaria-Romania-Turkey maritime pact over comparison stats."""
    from geopulse.summarization.service import _extract_supporting_claim_from_source

    snippet = (
        "Bulgaria, Romania and Turkey have signed a tripartite security accord in the Black Sea. "
        "The initiative expands maritime intelligence-sharing and naval patrols across "
        "regional shipping corridors. "
        "Separately, maritime risk monitors tracked 18 vessel casualties in neighboring waters "
        "according to Financial Times analysis of Ambrey and IMO data."
    )
    art = _make_article(
        title="Black Sea Coastal States Sign Maritime Security Intelligence Pact",
        snippet=snippet,
        url="https://ft.com/black-sea-pact",
        source_name="Financial Times",
    )

    c1_text = (
        "Bulgaria, Romania and Turkey have signed a tripartite security accord in the Black Sea."
    )
    supp = _extract_supporting_claim_from_source(art, c1_text, art.title)
    assert supp is not None
    # Must prefer the maritime security/intelligence/corridor development
    assert (
        "maritime" in supp.text.lower()
        or "shipping" in supp.text.lower()
        or "intelligence" in supp.text.lower()
    )
    # Must NOT select the incidental vessel casualty statistic
    assert "18 vessel casualties" not in supp.text


def test_bounded_ollama_retries_and_truncation_safety():
    """Observed issue 5: Ensure model calls per event are capped at 2 and truncation fails safe."""
    from geopulse.summarization.client import OllamaClient
    from geopulse.summarization.service import EventSummarizer

    class TruncatedOllamaClient(OllamaClient):
        def __init__(self):
            super().__init__(base_url="http://127.0.0.1:11434")
            self.call_count = 0
            self.last_metrics = {"done_reason": "length"}

        def generate(self, prompt: str, **kwargs) -> str:
            self.call_count += 1
            # Simulate incomplete/truncated JSON output
            return '{"claims": [{"claim_id": "c1", "text": "Incomplete text without end'

    client = TruncatedOllamaClient()
    summarizer = EventSummarizer(client=client)

    event = _make_event(
        title="NATO Air Policing Deployment Completed in Baltic Airspace",
        snippet="Fighters from allied nations completed air-policing patrols over the Baltic Sea.",
        url="https://defensenews.com/baltic-drill",
        source_name="Defense News",
    )

    res = summarizer.summarize_event(event)
    # Must safely quarantine due to truncation, not crash or loop infinitely
    assert res.status == "quarantined"
    assert "truncated" in (res.quarantine_reason or "").lower()
    # Must NOT exceed bounded calls (at most 2 calls per event)
    assert client.call_count <= 2


def test_black_sea_editorial_briefing_regression():
    """Regression 1: Black Sea agreement must cover intelligence sharing, 2026 vessel figures,
    Sofia meeting on Oct. 28, FT attribution, and avoid attributing uncoordinated naval patrols.
    """
    source_title = (
        "NATO's Black Sea Nations Decry Ukraine War Spillover Endangering Commercial Ship Traffic"
    )
    source_snippet = (
        "The Ministers of Defense of Bulgaria, Romania and Turkey met in Sofia on Oct. 28 "
        "and issued a joint statement agreeing to enhance information and intelligence sharing, "
        "improving maritime domain awareness and strengthening early warning in the Black Sea. "
        "The Financial Times reported based on data from Ambrey and the International Maritime "
        "Organization (IMO) that since the beginning of 2026, more than 570 vessels have been hit "
        "in the Black Sea, killing at least 40 seafarers."
    )
    url = (
        "https://www.defensenews.com/global/europe/2026/10/29/"
        "natos-black-sea-nations-decry-ukraine-war-spillover-endangering-commercial-ship-traffic/"
    )
    event = _make_event(
        title=source_title,
        snippet=source_snippet,
        url=url,
        category="security_incident",
        source_name="Defense News",
    )

    c1 = SummaryClaim(
        claim_id="c1",
        text=(
            "The Defense Ministers of Bulgaria, Romania and Turkey met in Sofia on October 28 and "
            "issued a joint statement agreeing to enhance information and intelligence sharing, "
            "maritime domain awareness, and early warning in the Black Sea."
        ),
        classification="reported_fact",
        central=True,
        evidence_refs=(
            ClaimEvidenceReference(
                source_url=url,
                evidence_span=(
                    "joint statement made after a meeting in Sofia, Bulgaria, on Oct. 28"
                ),
                verification_status="accepted",
            ),
        ),
        verification_status="accepted",
    )
    c2 = SummaryClaim(
        claim_id="c2",
        text=(
            "The Financial Times reported based on Ambrey and IMO data that more than 570 vessels "
            "have been hit in the Black Sea since the beginning of 2026, killing at least "
            "40 seafarers."
        ),
        classification="attributed_statement",
        attribution="Financial Times / Ambrey / IMO",
        central=False,
        evidence_refs=(
            ClaimEvidenceReference(
                source_url=url,
                evidence_span=(
                    "Financial Times reported based on data from Ambrey and the International "
                    "Maritime Organization"
                ),
                verification_status="accepted",
            ),
        ),
        verification_status="accepted",
    )

    summary = EventSummary(
        event_id="evt-black-sea-reg",
        headline="Bulgaria, Romania and Turkey Agree on Black Sea Maritime Intelligence Sharing",
        what_happened=(
            "The Defense Ministers of Bulgaria, Romania and Turkey met in Sofia on October 28 and "
            "issued a joint statement agreeing to enhance information and intelligence sharing, "
            "maritime domain awareness, and early warning in the Black Sea. The Financial Times "
            "reported based on Ambrey and IMO data that more than 570 vessels have been hit in the "
            "Black Sea since the beginning of 2026, killing at least 40 seafarers."
        ),
        key_details=EventKeyDetails(
            actors=("Bulgaria", "Romania", "Turkey"),
            location="Sofia, Bulgaria",
            timing="October 28, 2026",
            action=(
                "Joint statement on Black Sea maritime intelligence sharing and domain awareness"
            ),
        ),
        significance=(
            "Enhances commercial domain awareness and early warning for merchant shipping "
            "corridors facing war-risk escalation."
        ),
        confidence_qualification=(
            "Single-source reporting from Defense News; unverified independently."
        ),
        sources=(
            EventSourceAttribution(
                source_name="Defense News",
                title=source_title,
                url=url,
                published_at=DT,
            ),
        ),
        claims=(c1, c2),
        model_used="test-model",
    )

    gate_res = validate_publication_gate(summary, event)
    assert gate_res.passed
    rendered_summary = gate_res.repaired_summary or summary
    formatter = TelegramDigestFormatter()
    block = formatter.format_event_block(
        summary=rendered_summary,
        category=event.category,
        confidence_label=event.confidence_label,
        rank=1,
        total=1,
    )

    assert "Bulgaria, Romania and Turkey" in block
    assert "Sofia" in block
    assert "intelligence sharing" in block or "information and intelligence sharing" in block
    assert "570" in block
    assert "2026" in block
    assert "40 seafarers" in block
    assert "Financial Times" in block
    # Crucial: Must NOT contain invented naval patrols, drifting mines, or Constanța
    assert "Constanța" not in block
    assert "naval patrols" not in block.lower()
    assert "drifting naval mines" not in block.lower()
    assert "early-warning feeds" not in block.lower()
    assert "<b>Key Details:</b>" in block
    assert "Actors:</b> Bulgaria, Romania, Turkey" in block
    assert "Location:</b> Sofia, Bulgaria" in block
    assert "<b>Significance:</b>" in block
    assert "Defense News" in block


def test_pacom_multinational_data_sharing_briefing_regression():
    """Regression 2: PACOM story must cover the PACOM Mission Network, common operational picture,
    secure data sharing, Japanese Maven Smart System targeting, and avoid live-fire assertions.
    """
    source_title = (
        "Bilateral No More: PACOM Officials Highlight Multi-National Data-Sharing Network"
    )
    source_snippet = (
        "US Indo-Pacific Command officials highlighted the PACOM Mission Network, allowing US "
        "forces and allies to share data and mission-planning tools securely across classification "
        "to maintain a common operational picture. The network was used during several major "
        "exercises over the last year, while the command conducted 17 exercises overall across "
        "nine host nations. In one example, Japanese partners used the embedded Maven Smart System "
        "platform to contribute to joint strikes and close kill chains."
    )
    url = (
        "https://breakingdefense.com/2026/10/"
        "bilateral-no-more-pacom-officials-highlight-multi-national-data-sharing-network/"
    )
    event = _make_event(
        title=source_title,
        snippet=source_snippet,
        url=url,
        category="defence_policy",
        source_name="Breaking Defense",
    )

    c1 = SummaryClaim(
        claim_id="c1",
        text=(
            "US Indo-Pacific Command fielded the PACOM Mission Network to provide a common "
            "operational "
            "picture, secure data-sharing, and joint mission-planning across allies."
        ),
        classification="reported_fact",
        central=True,
        evidence_refs=(
            ClaimEvidenceReference(
                source_url=url,
                evidence_span=(
                    "PACOM Mission Network, a system that allows US forces and partners to share "
                    "data and mission planning tools securely"
                ),
                verification_status="accepted",
            ),
        ),
        verification_status="accepted",
    )
    c2 = SummaryClaim(
        claim_id="c2",
        text=(
            "Japanese forces used the embedded Maven Smart System platform to contribute to joint "
            "strikes during major exercises, while the command conducted 17 theatre exercises "
            "overall."
        ),
        classification="reported_fact",
        central=False,
        evidence_refs=(
            ClaimEvidenceReference(
                source_url=url,
                evidence_span=(
                    "Japanese partners used the Maven Smart System, an artificial "
                    "intelligence-enabled platform used for planning and targeting"
                ),
                verification_status="accepted",
            ),
        ),
        verification_status="accepted",
    )

    summary = EventSummary(
        event_id="evt-pacom-reg",
        headline=(
            "US PACOM Deploys Mission Network for Multinational Data Sharing and Joint Planning"
        ),
        what_happened=(
            "US Indo-Pacific Command fielded the PACOM Mission Network to provide a common "
            "operational "
            "picture, secure data-sharing, and joint mission-planning across allies. Japanese "
            "forces "
            "used the embedded Maven Smart System platform to contribute to joint strikes during "
            "major exercises, while the command conducted 17 theatre exercises overall."
        ),
        key_details=EventKeyDetails(
            actors=("US Indo-Pacific Command", "Japanese Self-Defense Forces"),
            location="Indo-Pacific",
            action=(
                "Deployment of PACOM Mission Network and Maven Smart System joint "
                "strike integration"
            ),
        ),
        significance=(
            "Eliminates bilateral information stovepipes to enable multinational kill-chain "
            "coordination "
            "across the Pacific theatre."
        ),
        confidence_qualification=(
            "Single-source reporting from Breaking Defense; unverified independently."
        ),
        sources=(
            EventSourceAttribution(
                source_name="Breaking Defense",
                title=source_title,
                url=url,
                published_at=DT,
            ),
        ),
        claims=(c1, c2),
        model_used="test-model",
    )

    gate_res = validate_publication_gate(summary, event)
    assert gate_res.passed
    formatter = TelegramDigestFormatter()
    block = formatter.format_event_block(
        summary=gate_res.repaired_summary or summary,
        category=event.category,
        confidence_label=event.confidence_label,
        rank=1,
        total=1,
    )
    assert "PACOM Mission Network" in block or "PACOM" in block
    assert "common operational picture" in block or "common tools" in block
    assert "Maven Smart System" in block
    assert "joint strikes" in block
    # Crucial: Must NOT invent live-fire exercises or synchronized tactical data
    assert "live-fire" not in block.lower()
    assert "synchronized tactical data" not in block.lower()
    assert "Actors:</b>" in block
    assert "Significance:</b>" in block


def test_uk_defence_offsets_threshold_briefing_regression():
    """Regression 3: UK offset policy must describe £50M as threshold, note benefits
    could reach the same value as the contract, and capture factories, suppliers, and tech transfer.
    """
    source_title = (
        "New UK Policy Requires Foreign Defense Contractors to Invest Locally for All £50M+ Deals"
    )
    source_snippet = (
        "A new UK Ministry of Defence policy taking effect in April 2027 requires foreign "
        "contractors on overseas defense contracts valued at £50 million or more to deliver "
        "local economic benefits that could reach the same value as the contract itself, "
        "including British factories, local suppliers, and technology transfer, overseen by "
        "an Offset Authority."
    )
    url = (
        "https://breakingdefense.com/2026/10/"
        "new-uk-policy-requires-foreign-defense-contractors-to-invest-locally-for-all-50m-deals/"
    )
    event = _make_event(
        title=source_title,
        snippet=source_snippet,
        url=url,
        category="defence_policy",
        source_name="Breaking Defense",
    )

    # Inaccurate summary claiming contractors must invest £50 million
    inaccurate_summary = EventSummary(
        event_id="evt-uk-bad",
        headline="UK Mandates Contractors to Invest £50 Million in Local Economy",
        what_happened=(
            "New UK policy requires foreign contractors to invest £50 million in "
            "industrial benefits."
        ),
        confidence_qualification="Single-source reporting; unverified independently.",
        sources=(
            EventSourceAttribution(
                source_name="Breaking Defense",
                title=source_title,
                url=url,
                published_at=DT,
            ),
        ),
        model_used="test-model",
    )
    bad_res = validate_publication_gate(inaccurate_summary, event)
    assert not bad_res.passed
    assert "threshold described as required investment amount" in bad_res.quarantine_reason

    # Accurate summary preserving £50M threshold and benefits up to the contract value
    c1 = SummaryClaim(
        claim_id="c1",
        text=(
            "The UK Ministry of Defence introduced an offsets policy applying to overseas defense "
            "contracts valued at £50 million or more, requiring foreign contractors to deliver "
            "domestic economic benefits that could reach the same value as the contract."
        ),
        classification="reported_fact",
        central=True,
        evidence_refs=(
            ClaimEvidenceReference(
                source_url=url,
                evidence_span=(
                    "contracts valued at £50 million or more to deliver local economic benefits "
                    "that could reach the same value as the contract"
                ),
                verification_status="accepted",
            ),
        ),
        verification_status="accepted",
    )
    c2 = SummaryClaim(
        claim_id="c2",
        text=(
            "The policy takes effect in April 2027 under a new Offset Authority, requiring "
            "commitments in British factories, local suppliers, and technology transfer."
        ),
        classification="reported_fact",
        central=False,
        evidence_refs=(
            ClaimEvidenceReference(
                source_url=url,
                evidence_span="British factories, local suppliers, and technology transfer",
                verification_status="accepted",
            ),
        ),
        verification_status="accepted",
    )

    accurate_summary = EventSummary(
        event_id="evt-uk-good",
        headline="UK Requires Local Industrial Investment Plans on Defence Contracts Over £50M",
        what_happened=(
            "The UK Ministry of Defence introduced an offsets policy applying to overseas defense "
            "contracts valued at £50 million or more, requiring foreign contractors to deliver "
            "domestic economic benefits that could reach the same value as the contract. "
            "The policy "
            "takes effect in April 2027 under a new Offset Authority, requiring commitments in "
            "British factories, local suppliers, and technology transfer."
        ),
        key_details=EventKeyDetails(
            actors=("UK Ministry of Defence", "Foreign defense contractors", "Offset Authority"),
            location="United Kingdom",
            timing="April 2027",
            action=(
                "Industrial offset policy triggered on £50M+ contracts with benefits up to "
                "contract value"
            ),
        ),
        significance=(
            "Ensures overseas procurement spending directly builds domestic industrial capacity "
            "and sovereign supplier ecosystems."
        ),
        confidence_qualification=(
            "Single-source reporting from Breaking Defense; unverified independently."
        ),
        sources=(
            EventSourceAttribution(
                source_name="Breaking Defense",
                title=source_title,
                url=url,
                published_at=DT,
            ),
        ),
        claims=(c1, c2),
        model_used="test-model",
    )
    good_res = validate_publication_gate(accurate_summary, event)
    assert good_res.passed
    formatter = TelegramDigestFormatter()
    block = formatter.format_event_block(
        summary=good_res.repaired_summary or accurate_summary,
        category=event.category,
        confidence_label=event.confidence_label,
        rank=1,
        total=1,
    )
    assert "£50 million" in block or "£50M" in block
    assert (
        "same value as the contract" in block
        or "could reach the same value" in block
        or "100%" in block
    )
    assert "British factories" in block
    assert "Offset Authority" in block
    assert "April 2027" in block


def test_patriot_fms_approval_vs_delivery_briefing_regression():
    """Regression 4: Patriot story must distinguish potential FMS approval from delivered missiles,
    specify PAC-3 MSE missiles, explain JUMPSTART funding, and avoid delivery schedules.
    """
    source_title = (
        "US Greenlights 500 Patriot Missiles for Ukraine, With $2.1 Billion to Be Paid by Allies"
    )
    source_snippet = (
        "The State Department has approved a potential Foreign Military Sale to Ukraine of up "
        "to 500 "
        "Patriot PAC-3 MSE missiles valued at an estimated $2.1 billion, with funding paid by "
        "Canada, Germany, and Norway via JUMPSTART and the European Union's Ukraine Support Loan. "
        "FMS notifications are early steps where final quantities and prices can change, with "
        "production deliveries spanning years."
    )
    url = (
        "https://breakingdefense.com/2026/10/"
        "us-greenlights-500-patriot-missiles-for-ukraine-with-2-1-billion-to-be-paid-by-allies/"
    )
    event = _make_event(
        title=source_title,
        snippet=source_snippet,
        url=url,
        category="weapons_procurement",
        source_name="Breaking Defense",
    )

    # Inaccurate summary claiming missiles were delivered
    bad_summary = EventSummary(
        event_id="evt-patriot-bad",
        headline="US Delivers 500 Patriot Missiles to European Allies for Ukraine",
        what_happened="The United States completed delivery of 500 Patriot missiles to allies.",
        confidence_qualification="Single-source reporting; unverified independently.",
        sources=(
            EventSourceAttribution(
                source_name="Breaking Defense",
                title=source_title,
                url=url,
                published_at=DT,
            ),
        ),
        model_used="test-model",
    )
    bad_gate = validate_publication_gate(bad_summary, event)
    assert not bad_gate.passed
    assert "completed equipment delivery" in bad_gate.quarantine_reason

    # Accurate summary preserving potential sale, PAC-3 MSE, Ukraine, and allied partner funding
    c1 = SummaryClaim(
        claim_id="c1",
        text=(
            "The State Department approved a potential Foreign Military Sale to Ukraine of up to "
            "500 "
            "Patriot PAC-3 MSE missiles valued at an estimated $2.1 billion."
        ),
        classification="reported_fact",
        central=True,
        evidence_refs=(
            ClaimEvidenceReference(
                source_url=url,
                evidence_span=(
                    "approved a potential Foreign Military Sale to Ukraine of up to 500 Patriot "
                    "PAC-3 MSE missiles"
                ),
                verification_status="accepted",
            ),
        ),
        verification_status="accepted",
    )
    c2 = SummaryClaim(
        claim_id="c2",
        text=(
            "The procurement is funded by Canada, Germany, and Norway via JUMPSTART and the EU "
            "Ukraine Support Loan, while final contract terms and physical deliveries remain "
            "pending."
        ),
        classification="reported_fact",
        central=False,
        evidence_refs=(
            ClaimEvidenceReference(
                source_url=url,
                evidence_span=(
                    "funding paid by Canada, Germany, and Norway via JUMPSTART and the European "
                    "Union's Ukraine Support Loan"
                ),
                verification_status="accepted",
            ),
        ),
        verification_status="accepted",
    )
    good_summary = EventSummary(
        event_id="evt-patriot-good",
        headline=(
            "State Department Approves Potential Sale of 500 Patriot PAC-3 MSE Missiles to Ukraine"
        ),
        what_happened=(
            "The State Department approved a potential Foreign Military Sale to Ukraine of up to "
            "500 "
            "Patriot PAC-3 MSE missiles valued at an estimated $2.1 billion. The procurement is "
            "funded "
            "by Canada, Germany, and Norway via JUMPSTART and the EU Ukraine Support Loan, while "
            "final "
            "contract terms and physical deliveries remain pending."
        ),
        key_details=EventKeyDetails(
            actors=("US State Department", "Ukraine", "Canada", "Germany", "Norway"),
            action=(
                "Approved potential Foreign Military Sale of 500 Patriot PAC-3 MSE missiles "
                "($2.1B allied funding)"
            ),
        ),
        significance=(
            "Expands Ukrainian long-range air defence inventory through multilateral allied "
            "burden-sharing "
            "without direct US procurement outlays."
        ),
        confidence_qualification=(
            "Single-source reporting from Breaking Defense; unverified independently."
        ),
        sources=(
            EventSourceAttribution(
                source_name="Breaking Defense",
                title=source_title,
                url=url,
                published_at=DT,
            ),
        ),
        claims=(c1, c2),
        model_used="test-model",
    )
    good_gate = validate_publication_gate(good_summary, event)
    assert good_gate.passed
    formatter = TelegramDigestFormatter()
    block = formatter.format_event_block(
        summary=good_gate.repaired_summary or good_summary,
        category=event.category,
        confidence_label=event.confidence_label,
        rank=1,
        total=1,
    )
    assert "PAC-3 MSE" in block
    assert "Ukraine" in block
    assert "potential" in block.lower() or "approves" in block.lower()
    assert "500" in block
    assert "$2.1 billion" in block or "$2.1B" in block
    assert "JUMPSTART" in block
    # Crucial: Must NOT state missiles were delivered and must NOT confuse GEM-T
    assert "completed delivery" not in block.lower()
    assert "GEM-T" not in block


def test_germany_norway_hansa_pact_briefing_regression():
    """Regression 5: Germany-Norway Hansa pact must state production began October 15,
    preserve 37 of 54 tanks and annual capacity of up to 36 in Levanger, separate Hansa,
    Type 212CD submarine cooperation, 3SM missile cooperation, and space initiatives,
    and identify the satellite constellation as a project Germany wants to pursue.
    """
    source_title = "From Tanks to Space: Norway and Germany Forge Ahead With Hansa Defense Pact"
    source_snippet = (
        "Under the bilateral Hansa defence pact, KNDS and Ritek began production on October 15 "
        "in Levanger to build 37 of Norway's 54 Leopard 2A8 tanks, with capacity of up to 36 tanks "
        "annually. The broader partnership includes established Type 212CD submarine cooperation, "
        "joint 3SM maritime strike missile development, and a bilateral space working group "
        "assessing a proposed polar satellite constellation from Andøya that Germany wants "
        "to pursue."
    )
    url = (
        "https://www.defensenews.com/global/europe/2026/10/30/"
        "from-tanks-to-space-norway-and-germany-forge-ahead-with-hansa-defense-pact/"
    )
    event = _make_event(
        title=source_title,
        snippet=source_snippet,
        url=url,
        category="defence_policy",
        source_name="Defense News",
    )

    c1 = SummaryClaim(
        claim_id="c1",
        text=(
            "Under the bilateral Hansa pact, production of 37 of Norway's 54 Leopard 2A8 tanks "
            "began on October 15 in Levanger, with capacity to produce up to 36 tanks annually."
        ),
        classification="reported_fact",
        central=True,
        evidence_refs=(
            ClaimEvidenceReference(
                source_url=url,
                evidence_span=(
                    "began production on October 15 in Levanger to build 37 of Norway's 54 "
                    "Leopard 2A8 tanks"
                ),
                verification_status="accepted",
            ),
        ),
        verification_status="accepted",
    )
    c2 = SummaryClaim(
        claim_id="c2",
        text=(
            "The broader relationship encompasses established Type 212CD submarine cooperation, "
            "3SM missile development, and a proposed polar satellite constellation that German "
            "leadership has identified as a project to pursue."
        ),
        classification="reported_fact",
        central=False,
        evidence_refs=(
            ClaimEvidenceReference(
                source_url=url,
                evidence_span=(
                    "established Type 212CD submarine cooperation, joint 3SM maritime "
                    "strike missile development, and a bilateral space working group"
                ),
                verification_status="accepted",
            ),
        ),
        verification_status="accepted",
    )

    summary = EventSummary(
        event_id="evt-hansa-good",
        headline=(
            "Germany and Norway Advance Hansa Defence Pact Across Land, Naval and Space Domains"
        ),
        what_happened=(
            "Under the bilateral Hansa pact, production of 37 of Norway's 54 Leopard 2A8 tanks "
            "began on October 15 in Levanger, with capacity to produce up to 36 tanks annually. "
            "The broader relationship encompasses established Type 212CD submarine cooperation, "
            "3SM missile development, and a proposed polar satellite constellation that German "
            "leadership has identified as a project to pursue."
        ),
        key_details=EventKeyDetails(
            actors=("Germany", "Norway", "KNDS", "Ritek"),
            location="Levanger, Norway",
            timing="October 15, 2026",
            action=(
                "Leopard 2A8 tank production launch in Levanger alongside submarine, "
                "3SM missile, and space initiatives"
            ),
        ),
        significance=(
            "Deepens bilateral industrial co-production and Arctic operational alignment between "
            "two key NATO northern flank partners."
        ),
        confidence_qualification=(
            "Single-source reporting from Defense News; unverified independently."
        ),
        sources=(
            EventSourceAttribution(
                source_name="Defense News",
                title=source_title,
                url=url,
                published_at=DT,
            ),
        ),
        claims=(c1, c2),
        model_used="test-model",
    )

    gate_res = validate_publication_gate(summary, event)
    assert gate_res.passed
    formatter = TelegramDigestFormatter()
    block = formatter.format_event_block(
        summary=gate_res.repaired_summary or summary,
        category=event.category,
        confidence_label=event.confidence_label,
        rank=1,
        total=1,
    )
    assert (
        "began on October 15" in block.lower()
        or "production started oct. 15" in block.lower()
        or "began" in block.lower()
    )
    assert "37" in block
    assert "54" in block
    assert "36" in block
    assert "Type 212CD" in block
    assert "3SM" in block
    assert "satellite constellation" in block


def test_generic_significance_slogan_is_omitted_by_formatter():
    """Significance requirement: Generic slogans (e.g. 'Strengthens deterrence.') must be
    omitted.
    """
    source_title = "Regional Maritime Surveillance Center Opens in Constanta"
    source_snippet = "Naval authorities opened an operational maritime center in Constanta."
    url = "https://defensenews.com/constanta-center"
    event = _make_event(
        title=source_title,
        snippet=source_snippet,
        url=url,
        category="defence_policy",
        source_name="Defense News",
    )
    c1 = SummaryClaim(
        claim_id="c1",
        text="Naval authorities opened an operational maritime surveillance center in Constanta.",
        evidence_refs=(
            ClaimEvidenceReference(
                source_url=url,
                evidence_span="opened an operational maritime center in Constanta",
            ),
        ),
        classification="reported_fact",
        central=True,
    )
    # Generic boilerplate slogan
    summary_generic = EventSummary(
        event_id="evt-gen-sig",
        headline=source_title,
        what_happened=(
            "Naval authorities opened an operational maritime surveillance center in Constanta."
        ),
        significance="Strengthens deterrence.",
        confidence_qualification="Single-source reporting; unverified independently.",
        sources=(
            EventSourceAttribution(
                source_name="Defense News", title=source_title, url=url, published_at=DT
            ),
        ),
        claims=(c1,),
        model_used="test-model",
    )
    formatter = TelegramDigestFormatter()
    block = formatter.format_event_block(
        summary=summary_generic,
        category=event.category,
        confidence_label=event.confidence_label,
        rank=1,
        total=1,
    )
    # Significance header must be omitted because it is a generic cliché
    assert "<b>Significance:</b>" not in block

    # Substantive explanatory significance must be preserved
    summary_substantive = EventSummary(
        event_id="evt-sub-sig",
        headline=source_title,
        what_happened=(
            "Naval authorities opened an operational maritime surveillance center in Constanta."
        ),
        significance=(
            "Expands radar coverage across coastal shipping lanes to detect low-flying "
            "aerial drones."
        ),
        confidence_qualification="Single-source reporting; unverified independently.",
        sources=(
            EventSourceAttribution(
                source_name="Defense News", title=source_title, url=url, published_at=DT
            ),
        ),
        claims=(c1,),
        model_used="test-model",
    )
    block_sub = formatter.format_event_block(
        summary=summary_substantive,
        category=event.category,
        confidence_label=event.confidence_label,
        rank=1,
        total=1,
    )
    assert "<b>Significance:</b> Expands radar coverage" in block_sub


def test_unfamiliar_story_generalization_naval_procurement():
    """Generalization: Pipeline cleanly processes an unfamiliar naval frigate contract."""
    source_title = "Naval Group Lays Keel for Second FDI Frigate Destined for Hellenic Navy"
    source_snippet = (
        "French shipbuilder Naval Group held a keel-laying ceremony in Lorient for the second "
        "FDI-class frigate ordered by Greece under a 2021 bilateral agreement. The vessel, "
        "named Nearchos, is scheduled for delivery in late 2025 as part of a three-ship "
        "procurement valued at €3 billion, with an option for a fourth vessel."
    )
    url = "https://defensenews.com/hellenic-fdi-frigate"
    event = _make_event(
        title=source_title,
        snippet=source_snippet,
        url=url,
        category="weapons_procurement",
        source_name="Defense News",
    )

    c1 = SummaryClaim(
        claim_id="c1",
        text=(
            "Naval Group laid the keel for Greece's second FDI-class frigate in Lorient under a "
            "three-ship €3 billion procurement agreement."
        ),
        classification="reported_fact",
        central=True,
        evidence_refs=(
            ClaimEvidenceReference(
                source_url=url,
                evidence_span=(
                    "keel-laying ceremony in Lorient for the second FDI-class frigate ordered by "
                    "Greece"
                ),
                verification_status="accepted",
            ),
        ),
        verification_status="accepted",
    )
    c2 = SummaryClaim(
        claim_id="c2",
        text=(
            "The frigate Nearchos is scheduled for delivery in late 2025, with Greece holding an "
            "option for a fourth vessel."
        ),
        classification="reported_fact",
        central=False,
        evidence_refs=(
            ClaimEvidenceReference(
                source_url=url,
                evidence_span=(
                    "scheduled for delivery in late 2025 as part of a three-ship procurement"
                ),
                verification_status="accepted",
            ),
        ),
        verification_status="accepted",
    )

    summary = EventSummary(
        event_id="evt-fdi-gen",
        headline="Naval Group Lays Keel for Second Greek FDI Frigate Under €3B Deal",
        what_happened=(
            "Naval Group laid the keel for Greece's second FDI-class frigate in Lorient under a "
            "three-ship €3 billion procurement agreement. The frigate Nearchos is scheduled for "
            "delivery in late 2025, with Greece holding an option for a fourth vessel."
        ),
        key_details=EventKeyDetails(
            actors=("Naval Group", "Hellenic Navy", "Greece", "France"),
            location="Lorient, France",
            timing="Late 2025 scheduled delivery",
            action="Keel-laying for second FDI-class frigate under €3B contract",
        ),
        significance=(
            "Advances modern surface warfare modernization and bilateral naval interoperability "
            "under the Franco-Greek strategic partnership."
        ),
        confidence_qualification=(
            "Single-source trade reporting from Defense News; unverified independently."
        ),
        sources=(
            EventSourceAttribution(
                source_name="Defense News", title=source_title, url=url, published_at=DT
            ),
        ),
        claims=(c1, c2),
        model_used="test-model",
    )

    gate_res = validate_publication_gate(summary, event)
    assert gate_res.passed
    formatter = TelegramDigestFormatter()
    block = formatter.format_event_block(
        summary=gate_res.repaired_summary or summary,
        category=event.category,
        confidence_label=event.confidence_label,
        rank=1,
        total=1,
    )
    assert "FDI" in block
    assert "Lorient" in block
    assert "€3 billion" in block or "€3B" in block
    assert "Nearchos" in block
    assert "Actors:</b>" in block
    assert "Significance:</b>" in block


def test_minimum_information_threshold_quarantines_stubs():
    """Minimum-information gate: Stubs and duplicate headline repetitions must be quarantined."""
    source_title = "Baltic Maritime Surveillance Radar Upgrade Completed"
    source_snippet = "Air defence forces completed upgrades to regional radar installations."
    url = "https://defensenews.com/baltic-radar"
    event = _make_event(
        title=source_title,
        snippet=source_snippet,
        url=url,
        category="weapons_procurement",
        source_name="Defense News",
    )

    # 1. Stub with fewer than 15 chars
    stub_summary = EventSummary(
        event_id="evt-stub",
        headline=source_title,
        what_happened="Radar done.",
        confidence_qualification="Single-source reporting; unverified independently.",
        sources=(
            EventSourceAttribution(
                source_name="Defense News",
                title=source_title,
                url=url,
                published_at=DT,
            ),
        ),
        model_used="test-model",
    )
    stub_res = validate_publication_gate(stub_summary, event)
    assert not stub_res.passed
    assert "body is empty" in stub_res.quarantine_reason

    # 2. Pure headline duplicate without key details
    dup_summary = EventSummary(
        event_id="evt-dup",
        headline=source_title,
        what_happened=source_title,
        confidence_qualification="Single-source reporting; unverified independently.",
        sources=(
            EventSourceAttribution(
                source_name="Defense News",
                title=source_title,
                url=url,
                published_at=DT,
            ),
        ),
        model_used="test-model",
    )
    dup_res = validate_publication_gate(dup_summary, event)
    assert not dup_res.passed
    assert "repeats headline" in dup_res.quarantine_reason


def test_casualty_evidence_cannot_verify_ministerial_agreement_or_condemnation_claim():
    """Verify casualty statistics passages are rejected against agreement or condemnation claims."""
    from geopulse.summarization.evidence_matcher import verify_claim_semantic_support

    c_condemn = (
        "Ministers of Defense of Bulgaria, Romania and Turkey condemned attacks against "
        "commercial vessels in the Black Sea and called on all parties to refrain from further "
        "actions."
    )
    c_agree = (
        "Bulgaria, Romania and Turkey agreed to enhance information and intelligence sharing, "
        "maritime domain awareness, and early warning in the Black Sea."
    )
    e_casualties = (
        "The Financial Times reported based on data from maritime security company Ambrey and the "
        "International Maritime Organization (IMO) that since the beginning of the year, more than "
        "570 vessels have been hit in the Black Sea, killing at least 40 seafarers."
    )
    e_condemn = (
        "The Ministers of Defense of Bulgaria, Romania and Turkey have condemned attacks against "
        "commercial vessels in the Black Sea and call on all parties to refrain from any further "
        "actions, according to the joint statement made after a meeting in Sofia, Bulgaria, on "
        "Oct. 28."
    )

    verdict_condemn_on_cas = verify_claim_semantic_support(c_condemn, e_casualties)
    assert not verdict_condemn_on_cas.is_supported
    assert "casualties" in verdict_condemn_on_cas.reason.lower()

    verdict_agree_on_cas = verify_claim_semantic_support(c_agree, e_casualties)
    assert not verdict_agree_on_cas.is_supported
    assert "casualties" in verdict_agree_on_cas.reason.lower()

    verdict_condemn_on_stmt = verify_claim_semantic_support(c_condemn, e_condemn)
    assert verdict_condemn_on_stmt.is_supported


def test_hansa_completeness_defect_rejects_single_tank_sentence_when_multidomain_pact_omitted():
    """Regression: Hansa summary reducing multidomain pact to one tank sentence is caught."""
    url = "https://www.defensenews.com/global/europe/2026/10/20/norway-begins-leopard-2a8-assembly/"
    source_title = "Norway Begins Leopard 2A8 Assembly Under Deepening German Industrial Pact"
    source_snippet = (
        "Assembly of 37 of Norway's 54 ordered Leopard 2A8 main battle tanks began on October 15 "
        "at the Ritek facility in Levanger, Norway, which has an annual capacity of up to "
        "36 tanks. The tank production operates under the broader bilateral Hansa defence "
        "cooperation framework with Germany, which also established Type 212CD submarine "
        "cooperation, joint 3SM maritime strike missile development, and a bilateral space "
        "working group."
    )
    event = _make_event(
        title=source_title,
        snippet=source_snippet,
        url=url,
        category="defence_policy",
        source_name="Defense News",
    )

    # Incomplete summary: only mentions tank assembly, drops broader bilateral framework
    incomplete_summary = EventSummary(
        event_id="evt-hansa-incomplete",
        headline="Norway Begins Leopard 2A8 Assembly in Levanger",
        what_happened=(
            "Assembly of 37 of Norway's 54 ordered Leopard 2A8 main battle tanks began on October "
            "15 at the Ritek facility in Levanger, Norway."
        ),
        confidence_qualification="Single-source reporting; unverified independently.",
        sources=(
            EventSourceAttribution(
                source_name="Defense News",
                title=source_title,
                url=url,
                published_at=DT,
            ),
        ),
        model_used="test-model",
    )

    gate_res = validate_publication_gate(incomplete_summary, event)
    assert not gate_res.passed
    assert "collapses multidomain partnership" in gate_res.quarantine_reason


def test_black_sea_completeness_defect_rejects_summary_omitting_intelligence_sharing_agreement():
    """Regression: Black Sea summary omitting intelligence sharing agreement is caught."""
    url = "https://www.defensenews.com/global/europe/2026/10/29/natos-black-sea-nations-decry/"
    source_title = (
        "NATO's Black Sea Nations Decry Ukraine War Spillover Endangering Commercial Ship Traffic"
    )
    source_snippet = (
        "The Ministers of Defense of Bulgaria, Romania and Turkey have condemned attacks against "
        "commercial vessels in the Black Sea, according to the joint statement made after a "
        "meeting in Sofia. The Financial Times reported that since the beginning of 2026, more "
        "than 570 vessels have been hit. Bulgaria, Romania and Turkey also agreed to enhance "
        "information and intelligence sharing, improving maritime domain awareness and "
        "strengthening early warning in the Black Sea."
    )
    event = _make_event(
        title=source_title,
        snippet=source_snippet,
        url=url,
        category="security_incident",
        source_name="Defense News",
    )

    # Incomplete summary: only mentions condemnation, drops intelligence sharing agreement
    incomplete_summary = EventSummary(
        event_id="evt-bs-incomplete",
        headline="Black Sea Ministers Condemn Attacks on Commercial Vessels",
        what_happened=(
            "The Ministers of Defense of Bulgaria, Romania and Turkey condemned attacks against "
            "commercial vessels in the Black Sea following a meeting in Sofia."
        ),
        confidence_qualification="Single-source reporting; unverified independently.",
        sources=(
            EventSourceAttribution(
                source_name="Defense News",
                title=source_title,
                url=url,
                published_at=DT,
            ),
        ),
        model_used="test-model",
    )

    gate_res = validate_publication_gate(incomplete_summary, event)
    assert not gate_res.passed
    assert "omits core intelligence-sharing agreement substance" in gate_res.quarantine_reason


def test_sanitize_actors_rejects_headline_fragments_and_polluted_verb_phrases():
    """Regression: sanitize_actors drops headline fragments, verb phrases, and splits lists."""
    from geopulse.summarization.validator import sanitize_actors

    polluted_actors = [
        "Black Sea Nations Decry Ukraine War Spillover Endangering Commercial Ship Traffic "
        "The Ministers",
        "NATO's Black Sea Nations Decry Ukraine War Spillover",
        "Foreign Contractors Must Invest £50M",
        "Bulgaria, Romania and Turkey",
        "Canada, Germany, and Norway",
    ]
    event_titles = (
        "NATO's Black Sea Nations Decry Ukraine War Spillover Endangering Commercial Ship Traffic",
    )

    sanitized = sanitize_actors(
        polluted_actors,
        source_names=("Defense News",),
        event_titles=event_titles,
    )
    polluted_title = (
        "Black Sea Nations Decry Ukraine War Spillover Endangering Commercial Ship Traffic "
        "The Ministers"
    )
    assert polluted_title not in sanitized
    assert "NATO's Black Sea Nations Decry Ukraine War Spillover" not in sanitized
    assert "Foreign Contractors Must Invest £50M" not in sanitized
    assert "Bulgaria" in sanitized
    assert "Romania" in sanitized
    assert "Turkey" in sanitized
    assert "Canada" in sanitized
    assert "Germany" in sanitized
    assert "Norway" in sanitized


def test_find_best_supporting_passage_associates_correct_sentences():
    """Verify that find_best_supporting_passage maps claims to their true supporting sentences."""
    from geopulse.summarization.evidence_matcher import find_best_supporting_passage

    snippet = (
        "The Ministers of Defense of Bulgaria, Romania and Turkey have condemned attacks against "
        "commercial vessels in the Black Sea and call on all parties to refrain from any further "
        "actions, according to the joint statement made after a meeting in Sofia, Bulgaria, on "
        "Oct. 28. The Financial Times reported based on data from maritime security company "
        "Ambrey and the International Maritime Organization (IMO) that since the beginning of "
        "the year, more than 570 vessels have been hit in the Black Sea, killing at least 40 "
        "seafarers. Bulgaria, Romania and Turkey also agreed to enhance information and "
        "intelligence sharing, improving maritime domain awareness and strengthening early "
        "warning to support security and stability in the Black Sea."
    )

    c_condemn = (
        "Ministers of Defense of Bulgaria, Romania and Turkey condemned attacks against "
        "commercial vessels."
    )
    c_agree = (
        "Bulgaria, Romania and Turkey agreed to enhance information and intelligence sharing "
        "and early warning."
    )
    c_cas = "More than 570 vessels have been hit in the Black Sea killing at least 40 seafarers."

    p_cond, s_cond = find_best_supporting_passage(c_condemn, snippet)
    assert p_cond is not None and "condemned attacks" in p_cond

    p_agr, s_agr = find_best_supporting_passage(c_agree, snippet)
    assert p_agr is not None and "agreed to enhance" in p_agr

    p_cas, s_cas = find_best_supporting_passage(c_cas, snippet)
    assert p_cas is not None and "570 vessels" in p_cas
