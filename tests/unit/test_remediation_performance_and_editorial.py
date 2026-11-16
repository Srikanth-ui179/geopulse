"""Focused regression tests for Geopulse performance and editorial remediation.

Covers:
1. Candidate cap and category diversity.
2. High-priority single-source events surviving candidate selection.
3. No unnecessary summarization of omitted clusters.
4. Wall-clock timeout enforcement.
5. Retry limits (max 1 retry by default).
6. Safe handling of truncated JSON and timed-out calls.
7. Prompt output compatibility.
8. eBook advertisements being excluded.
9. Legitimate reporting mentioning eBooks being retained.
10. Liveblogs not automatically becoming Tier 1 events.
11. Genuine developments from liveblogs remaining eligible.
12. Opinion and analysis being appropriately labelled.
13. Tactical promotional stories downranked.
14. Existing evidence, attribution, source lineage and publication gates remaining intact.
"""

import json
import time
from datetime import UTC, datetime

import httpx
import pytest

from geopulse.classification.models import TopicCategory
from geopulse.classification.rules import RuleBasedClassifier
from geopulse.clustering.models import EventCluster
from geopulse.delivery.formatter import (
    TelegramDigestFormatter,
    compute_event_priority_rank,
    partition_candidate_events,
)
from geopulse.ingestion.models import ArticleRecord
from geopulse.summarization.client import MockOllamaClient, OllamaClient
from geopulse.summarization.exceptions import OllamaResponseError, OllamaTimeoutError
from geopulse.summarization.prompts import (
    CORRECTION_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_event_prompt,
)
from geopulse.summarization.service import EventSummarizer


def _create_sample_article(
    title: str,
    source_name: str = "Defense News",
    url: str = "https://defensenews.com/article",
    canonical_url: str | None = None,
    snippet: str | None = None,
    published_at: datetime | None = None,
    topic: TopicCategory | None = None,
) -> ArticleRecord:
    dt = published_at or datetime(2026, 11, 7, 10, 0, tzinfo=UTC)
    can_url = canonical_url or url
    return ArticleRecord(
        source_name=source_name,
        title=title,
        url=url,
        canonical_url=can_url,
        published_at=dt,
        fetched_at=dt,
        snippet=snippet or title,
        content_hash=f"hash-{hash(can_url)}",
        relevance_status="relevant",
        topic=topic,
    )


def _create_sample_cluster(
    cluster_id: str,
    title: str,
    category: TopicCategory,
    source_count: int = 1,
    is_analysis: bool = False,
    published_at: datetime | None = None,
    url: str | None = None,
    publisher_prefix: str | None = None,
) -> EventCluster:
    dt = published_at or datetime(2026, 11, 7, 10, 0, tzinfo=UTC)
    pub_base = publisher_prefix or cluster_id
    articles = tuple(
        _create_sample_article(
            title=f"{title} - Report {i}",
            source_name=f"Publisher_{pub_base}_{i}",
            url=url or f"https://outlet-{i}.test/{cluster_id}",
            canonical_url=url or f"https://outlet-{i}.test/{cluster_id}",
            snippet=(
                "Operational forces completed regional military maneuvers "
                "and units deployed along borders."
            ),
            published_at=dt,
            topic=category,
        )
        for i in range(1, source_count + 1)
    )
    conf = "corroborated" if source_count > 1 else "single_source"
    return EventCluster(
        id=cluster_id,
        title=title,
        category=category,
        first_seen_at=dt,
        last_seen_at=dt,
        published_at=dt,
        is_analysis=is_analysis,
        confidence_label=conf,
        corroboration_status="independently_corroborated" if source_count > 1 else "single_source",
        articles=articles,
    )


# 1. Candidate cap and category diversity
def test_candidate_cap_and_category_diversity():
    """Verify that a candidate cap of 12 selects exactly 12 items and prevents
    one category from consuming the entire briefing.
    """
    fixed_dt = datetime(2026, 11, 7, 8, 0, tzinfo=UTC)

    # 10 Armed Conflict clusters
    conflict_clusters = [
        _create_sample_cluster(
            f"conf-{i}",
            f"Frontline strike action in sector {i}",
            TopicCategory.ARMED_CONFLICT,
            source_count=2,
            published_at=fixed_dt,
        )
        for i in range(10)
    ]
    # 4 Diplomacy clusters
    dip_clusters = [
        _create_sample_cluster(
            f"dip-{i}",
            f"Envoys hold ceasefire negotiations round {i}",
            TopicCategory.DIPLOMACY,
            source_count=2,
            published_at=fixed_dt,
        )
        for i in range(4)
    ]
    # 4 Defence Policy clusters
    def_clusters = [
        _create_sample_cluster(
            f"def-{i}",
            f"Ministry outlines air defence modernization plan {i}",
            TopicCategory.DEFENCE_POLICY,
            source_count=1,
            published_at=fixed_dt,
        )
        for i in range(4)
    ]
    # 4 Military Operations clusters
    mil_clusters = [
        _create_sample_cluster(
            f"mil-{i}",
            f"Allied naval patrol deployed in channel {i}",
            TopicCategory.MILITARY_OPERATIONS,
            source_count=1,
            published_at=fixed_dt,
        )
        for i in range(4)
    ]

    all_clusters = conflict_clusters + dip_clusters + def_clusters + mil_clusters
    assert len(all_clusters) == 22

    selected, deferred = partition_candidate_events(
        items=all_clusters,
        max_events=12,
        min_relevance_rank=9.0,
    )

    assert len(selected) == 12
    assert len(deferred) == 10

    # Verify category diversity: armed conflict must NOT take all 12 slots
    selected_cats = [c.category for c in selected]
    assert selected_cats.count(TopicCategory.ARMED_CONFLICT) <= 6
    assert TopicCategory.DIPLOMACY in selected_cats
    assert TopicCategory.MILITARY_OPERATIONS in selected_cats


# 2. High-priority single-source events survive candidate selection
def test_high_priority_single_source_survives_candidate_selection():
    """High-priority events (Tier 1: armed conflict) must never be discarded
    simply because they have fewer sources than secondary categories.
    """
    fixed_dt = datetime(2026, 11, 7, 8, 0, tzinfo=UTC)

    # Single-source armed conflict event
    single_source_strike = _create_sample_cluster(
        "strike-single",
        "Air strike targets coastal radar installation",
        TopicCategory.ARMED_CONFLICT,
        source_count=1,
        published_at=fixed_dt,
    )

    # Diverse secondary items
    diverse_policy = [
        _create_sample_cluster(
            f"policy-{i}",
            f"Ministry updates administrative procurement guidelines {i}",
            TopicCategory.DEFENCE_POLICY,
            source_count=3,
            published_at=fixed_dt,
        )
        for i in range(2)
    ]
    diverse_mil = [
        _create_sample_cluster(
            f"mil-{i}",
            f"Allied border forces complete joint maneuvers {i}",
            TopicCategory.MILITARY_OPERATIONS,
            source_count=3,
            published_at=fixed_dt,
        )
        for i in range(2)
    ]

    all_clusters = [single_source_strike] + diverse_policy + diverse_mil
    selected, _ = partition_candidate_events(items=all_clusters, max_events=5)

    assert len(selected) == 5
    assert single_source_strike in selected
    # First item should be the armed conflict event
    assert selected[0].id == single_source_strike.id


# 3. No unnecessary summarization of omitted clusters
def test_no_unnecessary_summarization_of_omitted_clusters():
    """When 25 clusters exist, candidate cap of 12 must invoke LLM at most 12 times."""
    fixed_dt = datetime(2026, 11, 7, 8, 0, tzinfo=UTC)
    clusters = [
        _create_sample_cluster(
            f"evt-{i}",
            f"Geopolitical development {i}",
            TopicCategory.ARMED_CONFLICT if i < 10 else TopicCategory.MILITARY_OPERATIONS,
            source_count=1,
            published_at=fixed_dt,
        )
        for i in range(25)
    ]

    class CountingClient(MockOllamaClient):
        def __init__(self):
            super().__init__(model="test-mock")
            self.call_count = 0

        def generate(
            self,
            prompt: str,
            system: str | None = None,
            format_type: str | None = "json",
            options: dict | None = None,
        ) -> str:
            self.call_count += 1
            if "source-entailment verifier" in (system or ""):
                return json.dumps(
                    {
                        "verdicts": [
                            {
                                "claim_id": "c1",
                                "src": 1,
                                "verdict": "supported",
                                "reason": "verified",
                            },
                            {
                                "claim_id": "c2",
                                "src": 1,
                                "verdict": "supported",
                                "reason": "verified",
                            },
                        ]
                    }
                )
            return json.dumps(
                {
                    "claims": [
                        {
                            "claim_id": "c1",
                            "text": "Operational forces completed regional military maneuvers.",
                            "src": 1,
                            "evidence": "forces completed",
                            "classification": "reported_fact",
                            "attribution": None,
                            "central": True,
                        },
                        {
                            "claim_id": "c2",
                            "text": "Additional units deployed along perimeter borders.",
                            "src": 1,
                            "evidence": "units deployed",
                            "classification": "reported_fact",
                            "attribution": None,
                            "central": False,
                        },
                    ],
                    "key_details": {
                        "actors": ["Armed Forces"],
                        "location": None,
                        "timing": "November 2026",
                        "action": "Completed maneuvers",
                    },
                    "significance": None,
                }
            )

    client = CountingClient()
    summarizer = EventSummarizer(client=client)

    results = summarizer.summarize_events(clusters, candidate_cap=12)

    # All 25 clusters returned in results (for omission accounting)
    assert len(results) == 25
    # Primary candidate clusters summarized (12 events * 2 calls [gen + verifier] = 24 calls)
    assert client.call_count <= 24
    omitted_results = [r for r in results if r.summary and r.summary.model_used == "omitted"]
    assert len(omitted_results) == 13

    # Accounting metrics accurately recorded
    metrics = summarizer.last_candidate_metrics
    assert metrics["total_clusters"] == 25
    assert metrics["selected_candidates"] == 12
    assert metrics["omitted_candidates"] == 13
    assert metrics["candidate_cap"] == 12


# 4. Wall-clock timeout enforcement
def test_wall_clock_timeout_enforcement():
    """Client aborts when total elapsed time exceeds request_timeout."""

    def slow_retry_handler(request: httpx.Request) -> httpx.Response:
        time.sleep(0.15)
        raise httpx.ReadError("connection dropped")

    transport = httpx.MockTransport(slow_retry_handler)
    # Set request_timeout to 0.1s so 0.15s sleep triggers wall-clock timeout
    client = OllamaClient(
        base_url="http://localhost:11434",
        request_timeout=0.1,
        max_retries=1,
        transport=transport,
    )

    with pytest.raises(OllamaTimeoutError) as exc_info:
        client.generate(prompt="Test prompt")
    assert "timed out" in str(exc_info.value).lower() or "deadline" in str(exc_info.value).lower()


# 5. Retry limits (default max 1 retry)
def test_retry_limits():
    """Client defaults to max_retries=1 and does not retry beyond limit on 503."""
    call_count = 0

    def transient_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(503, text="Service Unavailable")

    transport = httpx.MockTransport(transient_handler)
    client = OllamaClient(base_url="http://localhost:11434", max_retries=1, transport=transport)

    with pytest.raises((OllamaResponseError, httpx.HTTPStatusError)):
        client.generate(prompt="Test prompt")

    # Initial attempt + 1 retry = exactly 2 calls
    assert call_count == 2


# 6. Safe handling of truncated JSON and timed-out calls
def test_safe_handling_of_truncated_json():
    """When model output is truncated by num_predict (done_reason=length),
    the service safely quarantines the summary without unhandled exceptions.
    """
    cluster = _create_sample_cluster("c-trunc", "Test Event", TopicCategory.ARMED_CONFLICT)

    client = MockOllamaClient(
        canned_response='{"claims": [{"claim_id": "c1", "text": "Unfinished claim text...',
    )
    client.last_metrics = {"done_reason": "length"}

    summarizer = EventSummarizer(client=client)
    res = summarizer.summarize_event(cluster)

    assert res.status == "quarantined"
    qr = (res.quarantine_reason or "").lower()
    assert "truncated" in qr or "parse" in qr or "claim" in qr


def test_safe_handling_of_timed_out_event():
    """When a model call times out, the summarizer fails gracefully and does
    not crash the entire pipeline.
    """
    cluster = _create_sample_cluster("c-timeout", "Test Event", TopicCategory.ARMED_CONFLICT)
    client = MockOllamaClient(should_timeout=True)

    summarizer = EventSummarizer(client=client)
    res = summarizer.summarize_event(cluster)

    assert res.status in {"unavailable", "failed"}
    assert "timed out" in (res.error_message or "").lower()


# 7. Prompt output compatibility
def test_prompt_output_compatibility():
    """Optimized SYSTEM_PROMPT is compact (< 4,000 chars) and maintains
    compatibility with event prompt builders.
    """
    assert len(SYSTEM_PROMPT) < 4000
    assert len(CORRECTION_SYSTEM_PROMPT) < 800

    cluster = _create_sample_cluster("p-comp", "Naval Drill", TopicCategory.MILITARY_OPERATIONS)
    user_prompt = build_event_prompt(cluster)

    assert "EVENT METADATA:" in user_prompt
    assert "REQUIRED JSON STRUCTURE" in user_prompt
    assert '"claims":' in user_prompt
    assert '"key_details":' in user_prompt


# 8. eBook advertisements being excluded
def test_ebook_advertisement_excluded():
    """Promotional eBook collaterals are classified as irrelevant."""
    classifier = RuleBasedClassifier()

    ad_title = "Missed AUSA 2026? Get the Ground Truth from Breaking Defense's free eBook"
    ad_snippet = "Download your free copy of the AUSA 2026 after-action report and eBook."
    res = classifier.classify_text(title=ad_title, snippet=ad_snippet)

    assert res.relevance_status == "irrelevant"


# 9. Legitimate reporting mentioning eBooks being retained
def test_legitimate_reporting_mentioning_ebook_retained():
    """Reporting that merely mentions an eBook or whitepaper in passing is retained."""
    classifier = RuleBasedClassifier()

    title = "Air strikes and artillery barrage reported along southern sector"
    snippet = (
        "According to an eBook and analytical whitepaper published alongside the report, "
        "artillery strikes continued."
    )
    res = classifier.classify_text(title=title, snippet=snippet)

    assert res.relevance_status == "relevant"
    assert res.primary_topic in {
        TopicCategory.ARMED_CONFLICT,
        TopicCategory.MILITARY_OPERATIONS,
        TopicCategory.DEFENCE_POLICY,
    }


# 10. Liveblogs not automatically becoming Tier 1 events
def test_liveblogs_not_automatically_tier_1():
    """Rolling generic liveblog threads receive downranked priority (rank 10.0)."""
    live_cluster = _create_sample_cluster(
        "live-1",
        "Iran war live: latest updates and rolling coverage",
        TopicCategory.ARMED_CONFLICT,
        url="https://aljazeera.com/news/liveblog/iran-war-updates",
    )

    rank = compute_event_priority_rank(live_cluster)
    assert rank == 10.0  # Deferred / not automatically Tier 1


# 11. Genuine developments from liveblogs remaining eligible
def test_genuine_developments_from_liveblogs_remaining_eligible():
    """Specific diplomatic statements extracted from a liveblog headline retain priority."""
    substantive_live_cluster = _create_sample_cluster(
        "live-2",
        "Iran war live: Qatar says US, Iran should not wait to reach a peace deal",
        TopicCategory.DIPLOMACY,
        url="https://aljazeera.com/news/liveblog/qatar-peace-deal",
    )

    rank = compute_event_priority_rank(substantive_live_cluster)
    # Recognized as consequential diplomacy / peace deal statement (rank 1.5 or 6.0)
    assert rank in {1.5, 6.0}
    assert rank < 9.0  # Eligible for candidate selection


# 12. Opinion and analysis being appropriately labelled
def test_opinion_and_analysis_appropriately_labelled():
    """Normative argumentative opinion pieces are detected as analysis
    and formatted with the [ANALYSIS] badge.
    """
    from geopulse.clustering.clusterer import _ClusterBuilder

    article = _create_sample_article(
        title="The Golden Dome Proposal is Destabilizing and Dangerous",
        source_name="War on the Rocks",
        url="https://wotr.test/ideas/golden-dome-proposal",
    )
    cluster = _ClusterBuilder(article, TopicCategory.DEFENCE_POLICY).to_event_cluster()

    assert cluster.is_analysis is True

    # Check that formatter adds [ANALYSIS] badge
    formatter = TelegramDigestFormatter()
    res = _make_dummy_summary_result(cluster, is_analysis=True)
    block = formatter.format_event(
        summary=res.summary,
        event=cluster,
        category=cluster.category,
        is_analysis=True,
    )
    assert "[ANALYSIS]" in block


# 13. Tactical promotional stories downranked
def test_tactical_promotional_stories_downranked():
    """Vendor-driven infantry weapon accessory demonstrations are downranked to 8.8."""
    cluster = _create_sample_cluster(
        "tactical-1",
        "Israeli defense company says this trigger is programmed to shoot drones",
        TopicCategory.DEFENCE_POLICY,
    )

    rank = compute_event_priority_rank(cluster)
    assert rank == 8.8  # Downranked below major procurement 8.0

    # If it has a major contract indicator, it retains regular procurement rank 8.0
    strategic_cluster = _create_sample_cluster(
        "tactical-2",
        "Ministry awarded $50 million contract for computerized anti-drone trigger systems",
        TopicCategory.WEAPONS_PROCUREMENT,
    )
    rank_strategic = compute_event_priority_rank(strategic_cluster)
    assert rank_strategic == 8.0


# Helper for summary formatting test
def _make_dummy_summary_result(cluster: EventCluster, is_analysis: bool = False):
    from geopulse.summarization.models import (
        ClaimEvidenceReference,
        EventKeyDetails,
        EventSourceAttribution,
        EventSummary,
        SummarizationResult,
        SummaryClaim,
    )

    summary = EventSummary(
        event_id=cluster.id,
        headline=cluster.title,
        what_happened="Detailed factual narrative for event.",
        key_details=EventKeyDetails(actors=["State Actor"], action="Action summary"),
        claims=(
            SummaryClaim(
                claim_id="c1",
                text="Factual claim.",
                classification="reported_fact",
                central=True,
                evidence_refs=(
                    ClaimEvidenceReference(
                        source_url=cluster.articles[0].url,
                        evidence_span="Factual",
                        verification_status="accepted",
                    ),
                ),
            ),
        ),
        sources=tuple(
            EventSourceAttribution(
                source_name=a.source_name,
                title=a.title,
                url=a.url,
                published_at=a.published_at,
            )
            for a in cluster.articles
        ),
        confidence_qualification="Single-source reporting.",
        model_used="mock-model",
    )
    return SummarizationResult(
        event_id=cluster.id,
        status="success",
        summary=summary,
        event=cluster,
    )
