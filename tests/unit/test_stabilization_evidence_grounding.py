"""Unit tests for evidence grounding verification, placeholder rejection, and headline fallback."""

from datetime import UTC, datetime

from geopulse.classification.models import TopicCategory
from geopulse.clustering.models import EventCluster
from geopulse.ingestion.models import ArticleRecord
from geopulse.summarization.evidence_matcher import check_evidence_span_match
from geopulse.summarization.models import ClaimEvidenceReference, SummaryClaim
from geopulse.summarization.service import _derive_headline
from geopulse.summarization.validator import clean_headline_grammar, safe_truncate_headline


def test_reject_placeholder_evidence_spans() -> None:
    """Ensure placeholder, empty, and synthetic evidence spans are rejected."""
    available_text = (
        "The Ukrainian military reported overnight drone strikes damaging energy infrastructure "
        "across several districts in Kyiv."
    )

    placeholders = [
        "<short exact verbatim quote under 25 words>",
        "<exact verbatim quote>",
        "[verbatim quote]",
        "[exact quote]",
        "None",
        "N/A",
        "placeholder",
        "quote",
        "...",
        "<b></b>",
        "   ",
        "",
    ]

    for p in placeholders:
        status, diff_msg = check_evidence_span_match(p, available_text)
        assert status == "no_match", f"Expected 'no_match' for placeholder '{p}', got '{status}'"
        assert diff_msg is not None


def test_accept_genuine_verbatim_and_normalized_evidence() -> None:
    """Ensure genuine article quotes match as verbatim or normalized."""
    available_text = (
        "The Ukrainian military reported overnight drone strikes damaging energy infrastructure "
        "across several districts in Kyiv, according to officials."
    )

    # Exact verbatim match
    v_span = "overnight drone strikes damaging energy infrastructure"
    v_status, v_diff = check_evidence_span_match(v_span, available_text)
    assert v_status == "verbatim"
    assert v_diff is None

    # Normalized match with smart quotes and spacing
    n_span = "“overnight drone strikes damaging energy infrastructure”"
    n_status, n_diff = check_evidence_span_match(n_span, available_text)
    assert n_status in {"verbatim", "normalized"}


def test_dangling_headline_falls_back_to_publisher_title() -> None:
    """A candidate headline ending in a dangling demonym falls back to source title."""
    article = ArticleRecord(
        url="https://example.com/report1",
        canonical_url="https://example.com/report1",
        content_hash="h1",
        source_name="Defense News",
        title="Commercial tanker damaged in the Red Sea following missile strike",
        published_at=datetime.now(UTC),
        relevance_status="relevant",
    )
    event = EventCluster(
        id="evt-test-1",
        title="Commercial tanker damaged in the Red Sea following missile strike",
        category=TopicCategory.ARMED_CONFLICT,
        region="middle_east",
        first_seen_at=datetime.now(UTC),
        last_seen_at=datetime.now(UTC),
        confidence_label="single_source",
        articles=(article,),
    )

    # Case A: Truncated headline ending in dangling demonym "Iranian"
    dangling_candidate = (
        "Commercial vessel transiting international waters off Yemen damaged by Iranian"
    )
    truncated = safe_truncate_headline(dangling_candidate, max_chars=140)
    cleaned = clean_headline_grammar(truncated, source_title=event.title)
    assert not cleaned.endswith("Iranian")

    # Case B: _derive_headline with a dangling ending falls back to clean event.title
    claim = SummaryClaim(
        claim_id="c1",
        text="Commercial vessel transiting international waters off Yemen damaged by Iranian",
        classification="reported_fact",
        attribution=None,
        central=True,
        evidence_refs=(
            ClaimEvidenceReference(
                source_url=article.url,
                evidence_span="damaged by",
                provenance="full_article",
            ),
        ),
    )
    derived = _derive_headline(claim, event)
    assert derived == event.title or "damaged in the Red Sea" in derived
    assert not derived.endswith(("by", "Iranian", "in", "of", "to"))


def test_extremely_short_headline_falls_back_to_publisher_title() -> None:
    """A claim that produces < 4 words headline must fall back to the publisher's clean title."""
    article = ArticleRecord(
        url="https://example.com/report2",
        canonical_url="https://example.com/report2",
        content_hash="h2",
        source_name="Breaking Defense",
        title="France orders additional Caesar howitzers under new defense budget",
        published_at=datetime.now(UTC),
        relevance_status="relevant",
    )
    event = EventCluster(
        id="evt-test-2",
        title="France orders additional Caesar howitzers under new defense budget",
        category=TopicCategory.WEAPONS_PROCUREMENT,
        region="france",
        first_seen_at=datetime.now(UTC),
        last_seen_at=datetime.now(UTC),
        confidence_label="single_source",
        articles=(article,),
    )
    claim = SummaryClaim(
        claim_id="c1",
        text="122,951,031 copies",
        classification="reported_fact",
        attribution=None,
        central=True,
        evidence_refs=(
            ClaimEvidenceReference(
                source_url=article.url,
                evidence_span="122,951,031 copies",
                provenance="full_article",
            ),
        ),
    )
    derived = _derive_headline(claim, event)
    assert derived == event.title
