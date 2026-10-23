"""Unit tests for Phase 3 Part B: Geopolitical event clustering."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from geopulse.clustering import (
    EventClusterer,
    cluster_articles,
)
from geopulse.clustering.similarity import (
    check_location_conflict,
    compute_article_similarity,
    extract_locations,
    has_dispute_indicators,
)
from geopulse.ingestion.models import ArticleRecord

FIXTURES_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "sample_clustering_articles.json"


def load_fixture_articles() -> list[ArticleRecord]:
    """Helper to load parsed ArticleRecords from fixture JSON."""
    raw = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))
    records: list[ArticleRecord] = []
    for item in raw:
        published_dt = (
            datetime.fromisoformat(item["published_at"])
            if item.get("published_at")
            else None
        )
        records.append(
            ArticleRecord(
                source_name=item["source_name"],
                title=item["title"],
                url=item["url"],
                canonical_url=item["canonical_url"],
                published_at=published_dt,
                snippet=item.get("snippet"),
                content_hash=f"hash-{item['id']}",
            )
        )
    return records


def test_multiple_sources_covering_same_event_are_merged_and_corroborated():
    fixtures = load_fixture_articles()
    # art-01 (BBC) and art-02 (Al Jazeera) cover US/UK strikes against Houthis in Yemen
    houthi_articles = [fixtures[0], fixtures[1]]

    summary = cluster_articles(houthi_articles)

    assert summary.total_clusters == 1
    cluster = summary.clusters[0]
    assert cluster.confidence_label == "corroborated"
    assert cluster.source_count == 2
    assert cluster.article_count == 2
    sources = {a.source_name for a in cluster.articles}
    assert sources == {"BBC World News", "Al Jazeera English"}


def test_different_events_in_same_conflict_remain_separate():
    fixtures = load_fixture_articles()
    # art-03 is Rostov oil depot drone strike
    # art-04 is Belgorod power substation shelling
    ukraine_articles = [fixtures[2], fixtures[3]]

    summary = cluster_articles(ukraine_articles)

    # Must NOT merge into one event merely because both involve Russia/Ukraine war
    assert summary.total_clusters == 2
    for c in summary.clusters:
        assert c.article_count == 1
        assert c.confidence_label == "single_source"


def test_location_conflict_detection_prevents_merging_different_cities():
    locs_rostov = extract_locations("Drone strike hits Rostov oil depot")
    locs_belgorod = extract_locations("Shelling damages power line in Belgorod")
    assert "rostov" in locs_rostov
    assert "belgorod" in locs_belgorod
    assert check_location_conflict(locs_rostov, locs_belgorod) is True

    # Same location must not conflict
    locs_rostov_2 = extract_locations("Blaze continues at Rostov fuel site")
    assert check_location_conflict(locs_rostov, locs_rostov_2) is False


def test_temporal_gating_prevents_merging_identical_headlines_on_different_dates():
    base_time = datetime(2026, 10, 1, 10, 0, 0, tzinfo=UTC)
    later_time = base_time + timedelta(days=15)

    art1 = ArticleRecord(
        source_name="BBC World News",
        title="North Korea tests ballistic missile into Sea of Japan",
        url="https://bbc.com/nk-test-1",
        canonical_url="https://bbc.com/nk-test-1",
        published_at=base_time,
        content_hash="h1",
    )
    art2 = ArticleRecord(
        source_name="Al Jazeera English",
        title="North Korea tests ballistic missile into Sea of Japan",
        url="https://aljazeera.com/nk-test-2",
        canonical_url="https://aljazeera.com/nk-test-2",
        published_at=later_time,
        content_hash="h2",
    )

    similarity = compute_article_similarity(art1, art2, max_time_delta=timedelta(hours=48))
    assert similarity == 0.0

    summary = cluster_articles([art1, art2])
    assert summary.total_clusters == 2


def test_disputed_event_detection():
    fixtures = load_fixture_articles()
    # art-06: "Rebels claim downing of military transport aircraft; army denies loss"
    disputed_article = fixtures[5]

    assert has_dispute_indicators(disputed_article.title) is True

    summary = cluster_articles([disputed_article])
    assert summary.total_clusters == 1
    assert summary.clusters[0].confidence_label == "disputed"


def test_single_source_and_ambiguous_reports():
    fixtures = load_fixture_articles()
    # art-05 is UK MoD procurement contract reported only by Defense News
    procurement_article = fixtures[4]

    summary = cluster_articles([procurement_article])
    assert summary.total_clusters == 1
    cluster = summary.clusters[0]
    assert cluster.confidence_label == "single_source"
    assert cluster.source_count == 1
    assert cluster.article_count == 1


def test_overlapping_keywords_with_different_meanings_do_not_merge():
    fixtures = load_fixture_articles()
    # art-03: Ukrainian drone strikes set ablaze oil depot in Russia's Rostov region
    # art-07: Premier League match postponed after drone strike near stadium
    # art-08: Hollywood box office hits record high
    articles = [fixtures[2], fixtures[6], fixtures[7]]

    summary = cluster_articles(articles)
    # The entertainment article is filtered out as irrelevant;
    # the drone strike in Rostov and Premier League incident do NOT merge
    titles = [c.title for c in summary.clusters]
    assert any("Rostov" in t for t in titles)
    assert any("Premier League" in t for t in titles)
    for c in summary.clusters:
        assert c.article_count == 1


def test_repeated_clustering_is_deterministic_and_idempotent():
    fixtures = load_fixture_articles()
    # Run 1
    summary1 = cluster_articles(fixtures)
    # Run 2 on shuffled copy
    shuffled = list(reversed(fixtures))
    summary2 = cluster_articles(shuffled)

    assert summary1.total_clusters == summary2.total_clusters
    assert summary1.corroborated_clusters == summary2.corroborated_clusters
    assert summary1.single_source_clusters == summary2.single_source_clusters
    assert summary1.disputed_clusters == summary2.disputed_clusters

    # Compare cluster IDs and article counts deterministically
    clusters1_by_id = {c.id: c for c in summary1.clusters}
    clusters2_by_id = {c.id: c for c in summary2.clusters}

    assert set(clusters1_by_id.keys()) == set(clusters2_by_id.keys())
    for cid, c1 in clusters1_by_id.items():
        c2 = clusters2_by_id[cid]
        assert c1.title == c2.title
        assert c1.confidence_label == c2.confidence_label
        assert c1.article_count == c2.article_count


def test_all_source_articles_and_provenance_preserved():
    fixtures = load_fixture_articles()
    # Test with the corroborated Houthi airstrikes event
    houthi_articles = [fixtures[0], fixtures[1]]

    summary = cluster_articles(houthi_articles)
    cluster = summary.clusters[0]

    # Verify every article's original metadata is untouched
    for orig in houthi_articles:
        matched = next((a for a in cluster.articles if a.canonical_url == orig.canonical_url), None)
        assert matched is not None
        assert matched.url == orig.url
        assert matched.source_name == orig.source_name
        assert matched.published_at == orig.published_at
        assert matched.snippet == orig.snippet
        assert matched.content_hash == orig.content_hash


def test_empty_articles_returns_empty_summary():
    clusterer = EventClusterer()
    summary = clusterer.cluster_articles([])
    assert summary.total_articles == 0
    assert summary.total_clusters == 0
    assert summary.clusters == ()


def test_denmark_missile_procurement_articles_cluster_and_corroborate():
    base_time = datetime(2026, 10, 22, 12, 0, 0, tzinfo=UTC)
    art1 = ArticleRecord(
        source_name="Breaking Defense",
        title="Denmark orders long-range missiles for new strike brigade",
        url="https://breakingdefense.com/2026/10/denmark-orders-long-range-missiles",
        canonical_url="https://breakingdefense.com/2026/10/denmark-orders-long-range-missiles",
        published_at=base_time,
        snippet="Copenhagen signs defence procurement deal for precision strike missile systems.",
        content_hash="hash-denmark-1",
    )
    art2 = ArticleRecord(
        source_name="Defense News",
        title="Denmark to buy long-range missiles for new heavy brigade",
        url="https://defensenews.com/land/2026/10/denmark-buy-long-range-missiles",
        canonical_url="https://defensenews.com/land/2026/10/denmark-buy-long-range-missiles",
        published_at=base_time + timedelta(hours=3),
        snippet=(
            "Denmark will purchase precision strike missiles to equip its primary heavy brigade."
        ),
        content_hash="hash-denmark-2",
    )

    summary = cluster_articles([art1, art2])
    assert summary.total_clusters == 1
    cluster = summary.clusters[0]
    assert cluster.confidence_label == "corroborated"
    assert cluster.source_count == 2
    assert cluster.article_count == 2
    assert {a.source_name for a in cluster.articles} == {"Breaking Defense", "Defense News"}


def test_different_events_in_same_country_do_not_merge():
    base_time = datetime(2026, 10, 22, 12, 0, 0, tzinfo=UTC)
    art1 = ArticleRecord(
        source_name="Breaking Defense",
        title="Denmark orders long-range missiles for new strike brigade",
        url="https://breakingdefense.com/2026/10/denmark-orders-long-range-missiles",
        canonical_url="https://breakingdefense.com/2026/10/denmark-orders-long-range-missiles",
        published_at=base_time,
        snippet="Copenhagen signs defence procurement deal for precision strike missile systems.",
        content_hash="hash-denmark-1",
    )
    art2 = ArticleRecord(
        source_name="Defense News",
        title="Denmark commissions new naval patrol vessel in Baltic Sea",
        url="https://defensenews.com/naval/2026/10/denmark-commissions-patrol-vessel",
        canonical_url="https://defensenews.com/naval/2026/10/denmark-commissions-patrol-vessel",
        published_at=base_time + timedelta(hours=3),
        snippet="Danish navy accepts new patrol frigate for territorial surveillance duties.",
        content_hash="hash-denmark-naval",
    )

    summary = cluster_articles([art1, art2])
    # Different topics and actions in the same country must remain distinct
    assert summary.total_clusters == 2
    for c in summary.clusters:
        assert c.article_count == 1


def test_older_analytical_article_preserves_published_date_and_analysis_flag():
    old_pub = datetime(2024, 5, 10, 12, 0, 0, tzinfo=UTC)
    fetch_time = datetime(2026, 10, 23, 10, 0, 0, tzinfo=UTC)

    art = ArticleRecord(
        source_name="War on the Rocks",
        title="Strategic Review: Nuclear Deterrence and Second-Strike Capabilities",
        url="https://warontherocks.com/strategic-review-nuclear-deterrence",
        canonical_url="https://warontherocks.com/strategic-review-nuclear-deterrence",
        published_at=old_pub,
        fetched_at=fetch_time,
        snippet="An in-depth analysis of strategic stability, deterrence theory, and posture.",
        content_hash="hash-old-analysis",
    )

    summary = cluster_articles([art])
    assert summary.total_clusters == 1
    cluster = summary.clusters[0]

    # Must be marked as analysis
    assert cluster.is_analysis is True
    # Must preserve the exact 2024 publication date, NOT the 2026 fetch time
    assert cluster.published_at == old_pub
    assert cluster.published_at != cluster.first_seen_at or cluster.first_seen_at == old_pub
    assert cluster.published_at.year == 2024


def test_missing_publication_date_is_not_silently_replaced_by_fetch_time():
    fetch_time = datetime(2026, 10, 23, 10, 0, 0, tzinfo=UTC)

    art = ArticleRecord(
        source_name="BBC World News",
        title="Unspecified border post reports military troop movement",
        url="https://bbc.com/undated-border-report",
        canonical_url="https://bbc.com/undated-border-report",
        published_at=None,
        fetched_at=fetch_time,
        snippet="Border guards noted increased vehicle traffic near border crossing.",
        content_hash="hash-undated",
    )

    summary = cluster_articles([art])
    assert summary.total_clusters == 1
    cluster = summary.clusters[0]

    # Cluster published_at must be strictly None, not silently set to fetch_time
    assert cluster.published_at is None
    # first_seen_at tracks the retrieval boundary
    assert cluster.first_seen_at == fetch_time


def test_denmark_procurement_two_source_clustering_when_both_present():
    art1 = ArticleRecord(
        source_name="Defense News",
        title="Denmark to buy Kongsberg, Covenant missiles for new long-range strike option",
        url="https://www.defensenews.com/global/europe/2026/10/22/denmark-to-buy-kongsberg-covenant-missiles-for-new-long-range-strike-option",
        canonical_url="https://www.defensenews.com/global/europe/2026/10/22/denmark-to-buy-kongsberg-covenant-missiles-for-new-long-range-strike-option",
        published_at=datetime(2026, 10, 22, 19, 3, tzinfo=UTC),
        snippet="Denmark will buy missiles from Norway and the United States...",
        content_hash="hash-denmark-1",
        relevance_status="relevant",
    )
    art2 = ArticleRecord(
        source_name="Breaking Defense",
        title="‘Ability to strike back’: Denmark commits to long-range missile orders",
        url="https://breakingdefense.com/2026/10/ability-to-strike-back-denmark-commits-to-long-range-missile-orders",
        canonical_url="https://breakingdefense.com/2026/10/ability-to-strike-back-denmark-commits-to-long-range-missile-orders",
        published_at=datetime(2026, 10, 22, 17, 18, tzinfo=UTC),
        snippet="Denmark announces long-range missile procurement plan...",
        content_hash="hash-denmark-2",
        relevance_status="relevant",
    )

    summary = cluster_articles([art1, art2])
    assert summary.total_clusters == 1
    cluster = summary.clusters[0]
    assert cluster.confidence_label == "corroborated"
    assert cluster.source_count == 2
    assert {a.source_name for a in cluster.articles} == {"Defense News", "Breaking Defense"}


def test_unclassified_uncertain_general_news_excluded_from_clusters():
    art = ArticleRecord(
        source_name="BBC World News",
        title="Central bank announces new interest rate policy amid rising inflation",
        url="https://bbc.com/interest-rates",
        canonical_url="https://bbc.com/interest-rates",
        published_at=datetime(2026, 10, 23, 8, 0, tzinfo=UTC),
        snippet="Economists express cautious optimism regarding financial market stability.",
        content_hash="hash-econ",
    )

    summary = cluster_articles([art])
    assert summary.total_clusters == 0
    assert summary.clusters == ()


def test_multiple_outlets_citing_same_wire_agency_remain_single_source():
    """Two outlets echoing the same state wire must NOT be upgraded to corroborated."""
    base_time = datetime(2026, 10, 23, 10, 0, 0, tzinfo=UTC)
    art1 = ArticleRecord(
        source_name="BBC World News",
        title="Ukrainian drone strikes set ablaze oil depot in Rostov",
        url="https://bbc.com/rostov-fire-1",
        canonical_url="https://bbc.com/rostov-fire-1",
        published_at=base_time,
        snippet="According to Russian state news agency TASS, firefighters tackle depot blaze...",
        content_hash="hash-tass-1",
        relevance_status="relevant",
    )
    art2 = ArticleRecord(
        source_name="Defense News",
        title="Oil depot in Rostov set ablaze after Ukrainian drone strikes",
        url="https://defensenews.com/rostov-fire-2",
        canonical_url="https://defensenews.com/rostov-fire-2",
        published_at=base_time + timedelta(hours=1),
        snippet="Citing TASS, fuel tanks were damaged in the overnight raid.",
        content_hash="hash-tass-2",
        relevance_status="relevant",
    )

    summary = cluster_articles([art1, art2])
    assert summary.total_clusters == 1
    cluster = summary.clusters[0]
    assert cluster.confidence_label == "single_source"
    assert cluster.source_count == 2


def test_outlet_citing_another_outlets_press_release_remains_single_source():
    """Trade outlet citing MoD press release is derived reporting, not independent corroboration."""
    base_time = datetime(2026, 10, 23, 10, 0, 0, tzinfo=UTC)
    art1 = ArticleRecord(
        source_name="UK Ministry of Defence",
        title="Chief of Defence hosts tenth Dragon Group meeting in London",
        url="https://gov.uk/dragon-group",
        canonical_url="https://gov.uk/dragon-group",
        published_at=base_time,
        snippet="Air Chief Marshal Sir Richard Knighton hosted defence leaders in London.",
        content_hash="hash-mod-1",
        relevance_status="relevant",
    )
    art2 = ArticleRecord(
        source_name="Defense News",
        title="Tenth Dragon Group meeting hosted by Chief of Defence in London",
        url="https://defensenews.com/dragon-group",
        canonical_url="https://defensenews.com/dragon-group",
        published_at=base_time + timedelta(hours=2),
        snippet="According to UK Ministry of Defence, military leaders met at Trinity House.",
        content_hash="hash-mod-2",
        relevance_status="relevant",
    )

    summary = cluster_articles([art1, art2])
    assert summary.total_clusters == 1
    cluster = summary.clusters[0]
    assert cluster.confidence_label == "single_source"
    assert cluster.source_count == 2


def test_multiple_articles_from_single_source_remain_single_source():
    """Multiple updates from the same outlet must remain single_source, not uncertain."""
    base_time = datetime(2026, 10, 23, 10, 0, 0, tzinfo=UTC)
    art1 = ArticleRecord(
        source_name="Breaking Defense",
        title="Air Force conducts experimental hypersonic glider flight test",
        url="https://breakingdefense.com/hypersonic-1",
        canonical_url="https://breakingdefense.com/hypersonic-1",
        published_at=base_time,
        snippet="Initial flight test conducted over Pacific test range.",
        content_hash="hash-hypersonic-1",
        relevance_status="relevant",
    )
    art2 = ArticleRecord(
        source_name="Breaking Defense",
        title="Air Force experimental hypersonic glider flight test achieves Mach 5",
        url="https://breakingdefense.com/hypersonic-2",
        canonical_url="https://breakingdefense.com/hypersonic-2",
        published_at=base_time + timedelta(hours=3),
        snippet="Pentagon confirms telemetry data recorded from Pacific glide test.",
        content_hash="hash-hypersonic-2",
        relevance_status="relevant",
    )

    summary = cluster_articles([art1, art2])
    assert summary.total_clusters == 1
    cluster = summary.clusters[0]
    assert cluster.confidence_label == "single_source"
    assert cluster.source_count == 1
    assert cluster.article_count == 2


def test_operational_events_precede_analytical_essays_in_cluster_ordering():
    """Analytical essays must be de-prioritized to the end of the briefing sequence."""
    base_time = datetime(2026, 10, 23, 10, 0, 0, tzinfo=UTC)
    art_analysis = ArticleRecord(
        source_name="War on the Rocks",
        title="The Endless Chip War",
        url="https://warontherocks.com/the-endless-chip-war",
        canonical_url="https://warontherocks.com/the-endless-chip-war",
        published_at=base_time - timedelta(days=2),
        snippet="Strategic review and essay on global semiconductor competition.",
        content_hash="hash-analysis-1",
        relevance_status="relevant",
    )
    art_breaking = ArticleRecord(
        source_name="Defense News",
        title="Missile strike damages coastal naval radar installation",
        url="https://defensenews.com/radar-strike",
        canonical_url="https://defensenews.com/radar-strike",
        published_at=base_time,
        snippet="Naval forces reported precision strike against radar facility.",
        content_hash="hash-breaking-1",
        relevance_status="relevant",
    )

    summary = cluster_articles([art_analysis, art_breaking])
    assert summary.total_clusters == 2
    assert summary.clusters[0].is_analysis is False
    assert "radar" in summary.clusters[0].title.lower()
    assert summary.clusters[1].is_analysis is True
    assert "chip war" in summary.clusters[1].title.lower()




