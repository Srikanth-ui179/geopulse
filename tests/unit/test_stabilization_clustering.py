"""Regression and stabilization tests for cross-publisher event clustering."""

from datetime import UTC, datetime

from geopulse.classification.models import TopicCategory
from geopulse.clustering import cluster_articles
from geopulse.clustering.similarity import compute_article_similarity
from geopulse.ingestion.models import ArticleRecord


def test_iwi_arbel_cross_publisher_clustering() -> None:
    """IWI and Israel Weapons Industry coverage of ARBEL counter-drone system must cluster."""
    a1 = ArticleRecord(
        url="https://breakingdefense.com/2026/10/iwi-arbel-system-counter-drone",
        canonical_url="https://breakingdefense.com/2026/10/iwi-arbel-system-counter-drone",
        content_hash="hash-iwi-bd",
        source_name="Breaking Defense",
        title="IWI's computerized ARBEL system offers counter-drone edge",
        snippet=(
            "Israeli small arms manufacturer unveils ARBEL computerised small arms system "
            "aimed at downing drones and improving infantry lethality."
        ),
        published_at=datetime(2026, 10, 24, 12, 0, tzinfo=UTC),
        relevance_status="relevant",
    )
    a2 = ArticleRecord(
        url="https://www.defensenews.com/land/2026/10/24/israel-weapons-industry-arbel-counter-drone",
        canonical_url="https://www.defensenews.com/land/2026/10/24/israel-weapons-industry-arbel-counter-drone",
        content_hash="hash-iwi-dn",
        source_name="Defense News",
        title="Israel Weapons Industry introduces Arbel system with counter-drone capability",
        snippet=(
            "Israel Weapons Industry has developed a new computerized weapon system called "
            "Arbel for counter-drone operations and squad defense."
        ),
        published_at=datetime(2026, 10, 24, 14, 30, tzinfo=UTC),
        relevance_status="relevant",
    )

    sim = compute_article_similarity(
        a1,
        a2,
        topic_a=TopicCategory.WEAPONS_PROCUREMENT,
        topic_b=TopicCategory.WEAPONS_PROCUREMENT,
    )
    assert sim >= 0.50, f"Expected similarity >= 0.50 for IWI Arbel coverage, got {sim:.3f}"

    summary = cluster_articles([a1, a2])
    assert summary.total_clusters == 1, (
        f"Expected 1 unified cluster for IWI Arbel, got {summary.total_clusters}"
    )
    assert summary.clusters[0].confidence_label == "corroborated"


def test_euronaval_same_system_clustering() -> None:
    """Distinct outlets reporting on Naval Group's UUV at Euronaval must cluster."""
    en1 = ArticleRecord(
        url="https://breakingdefense.com/2026/11/naval-group-uuv-euronaval",
        canonical_url="https://breakingdefense.com/2026/11/naval-group-uuv-euronaval",
        content_hash="hash-en-1",
        source_name="Breaking Defense",
        title="At Euronaval, Naval Group pitches oceanic autonomous underwater drone",
        snippet=(
            "Naval Group unveiled its large autonomous underwater vehicle (UUV) demonstrator "
            "at the Euronaval exhibition in Paris."
        ),
        published_at=datetime(2026, 11, 4, 10, 0, tzinfo=UTC),
        relevance_status="relevant",
    )
    en2 = ArticleRecord(
        url="https://www.defensenews.com/naval/2026/11/04/naval-group-unmanned-sub-euronaval",
        canonical_url="https://www.defensenews.com/naval/2026/11/04/naval-group-unmanned-sub-euronaval",
        content_hash="hash-en-2",
        source_name="Defense News",
        title="Naval Group displays large unmanned underwater vessel at Euronaval",
        snippet=(
            "French shipbuilder Naval Group showcased its long-range combat UUV demonstrator "
            "during the Euronaval conference in Paris."
        ),
        published_at=datetime(2026, 11, 4, 11, 15, tzinfo=UTC),
        relevance_status="relevant",
    )

    sim = compute_article_similarity(
        en1,
        en2,
        topic_a=TopicCategory.WEAPONS_PROCUREMENT,
        topic_b=TopicCategory.WEAPONS_PROCUREMENT,
    )
    assert sim >= 0.50, f"Expected similarity >= 0.50 for Euronaval UUV, got {sim:.3f}"

    summary = cluster_articles([en1, en2])
    assert summary.total_clusters == 1


def test_euronaval_distinct_sub_events_preserved() -> None:
    """Articles sharing Euronaval venue but describing different contracts must NOT merge."""
    en_uuv = ArticleRecord(
        url="https://breakingdefense.com/2026/11/naval-group-uuv-euronaval",
        canonical_url="https://breakingdefense.com/2026/11/naval-group-uuv-euronaval",
        content_hash="hash-en-uuv",
        source_name="Breaking Defense",
        title="At Euronaval, Naval Group pitches oceanic autonomous underwater drone",
        snippet=(
            "Naval Group unveiled its large autonomous underwater vehicle (UUV) demonstrator "
            "at the Euronaval exhibition in Paris."
        ),
        published_at=datetime(2026, 11, 4, 10, 0, tzinfo=UTC),
        relevance_status="relevant",
    )
    en_frigate = ArticleRecord(
        url="https://www.defensenews.com/naval/2026/11/04/france-greece-frigate-pact-euronaval",
        canonical_url="https://www.defensenews.com/naval/2026/11/04/france-greece-frigate-pact-euronaval",
        content_hash="hash-en-frigate",
        source_name="Defense News",
        title="At Euronaval, French and Greek navies sign frigate support pact",
        snippet=(
            "France and Greece finalized a bilateral naval maintenance agreement for FDI frigates "
            "at the Euronaval defense exhibition."
        ),
        published_at=datetime(2026, 11, 4, 10, 30, tzinfo=UTC),
        relevance_status="relevant",
    )

    sim = compute_article_similarity(
        en_uuv,
        en_frigate,
        topic_a=TopicCategory.WEAPONS_PROCUREMENT,
        topic_b=TopicCategory.WEAPONS_PROCUREMENT,
    )
    assert sim < 0.50, (
        f"Expected similarity < 0.50 for different Euronaval contracts, got {sim:.3f}"
    )

    summary = cluster_articles([en_uuv, en_frigate])
    assert summary.total_clusters == 2, (
        f"Expected 2 separate clusters for different Euronaval events, got {summary.total_clusters}"
    )
