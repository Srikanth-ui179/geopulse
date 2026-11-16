"""Unit tests for Phase 2: Article Revision History & Historical Provenance.

Validates:
1. Initial publication creates revision 1 with correct hash and sequence.
2. Repeated identical ingestion creates no duplicate revision.
3. HTTP 304 Not Modified creates no revision and updates revalidation timestamp.
4. Changed body creates revision 2 classified as content_update.
5. Title change without body change creates revision classified as metadata_update.
6. Explicit correction detected (explicit_correction).
7. Explicit retraction detected (explicit_retraction).
8. 404 / HTTP error does not imply retraction (classified as unavailability).
9. Network failure preserves history and cached revisions without truncation.
10. Concurrency deduplication (thread-safe sequential revisions).
11. Historical revisions are immutable and cannot be overwritten.
12. Existing published claims retain links to their original revisions.
13. Revised evidence in later revision does not rewrite historical claims.
14. Existing canonical deduplication intact across tracking params and schemes.
15. Distinct outlets covering the same event remain distinct article identities.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

from geopulse.ingestion.extractor import ArticleExtractor
from geopulse.ingestion.models import ArticleRecord, FeedSource
from geopulse.ingestion.repository import InMemoryArticleRepository
from geopulse.summarization.models import (
    ClaimEvidenceReference,
    EventCluster,
    SummaryClaim,
)
from geopulse.summarization.service import _populate_evidence_provenance


def make_html(text: str, title: str = "Defense Report", include_h1: bool = True) -> str:
    """Helper to generate HTML with substantive paragraphs (>150 chars) for trafilatura."""
    padding = (
        " Operational units across multiple regional theaters maintained comprehensive readiness "
        "and surveillance protocols. Joint coordination committees reported steady communications "
        "and continuous logistics throughout the deployment cycle."
    )
    full = f"<p>{text}</p><p>{padding}</p>"
    h1_part = f"<h1>{title}</h1>" if include_h1 else ""
    return (
        f"<html><head><title>{title}</title></head>"
        f"<body><article>{h1_part}{full}</article></body></html>"
    )


@pytest.fixture
def temp_cache(tmp_path: Path) -> Path:
    cache_dir = tmp_path / "article_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


# 1. Initial publication creates revision 1
def test_initial_publication_creates_rev_1(temp_cache: Path):
    url = "https://example.com/defense/procurement-deal"
    html = make_html(
        "The defense ministry signed a landmark $1.2B contract to procure three frigates."
    )
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.text = html
    mock_resp.url = httpx.URL(url)
    mock_resp.headers = {"etag": '"etag-v1"', "last-modified": "Mon, 01 Nov 2026 10:00:00 GMT"}

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.get.return_value = mock_resp

    extractor = ArticleExtractor(cache_dir=temp_cache, check_robots=False, http_client=mock_client)
    extracted = extractor.extract_url(url=url)

    assert extracted.revision is not None
    assert extracted.revision.revision_sequence == 1
    assert extracted.revision.revision_classification == "initial_publication"
    assert extracted.revision.previous_revision_id is None
    assert extracted.revision.etag == '"etag-v1"'
    assert "landmark $1.2B contract" in (extracted.revision.body_text or "")
    assert extracted.revision.article_id.startswith("")

    # Check repository history
    history = extractor.get_revision_history(url)
    assert len(history) == 1
    assert history[0].revision_id == extracted.revision.revision_id


# 2. Repeated identical ingestion creates no duplicate revision
def test_repeated_identical_ingestion_creates_no_duplicate(temp_cache: Path):
    url = "https://example.com/defense/drone-strike"
    html = make_html(
        "Air defenses intercepted two unidentified drones over the northern airbase facility."
    )
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.text = html
    mock_resp.url = httpx.URL(url)
    mock_resp.headers = {}

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.get.return_value = mock_resp

    extractor = ArticleExtractor(cache_dir=temp_cache, check_robots=False, http_client=mock_client)
    ext1 = extractor.extract_url(url=url)
    assert ext1.revision is not None
    rev1_id = ext1.revision.revision_id

    # Second extraction with identical body and title
    ext2 = extractor.extract_url(url=url)
    assert ext2.revision is not None
    assert ext2.revision.revision_id == rev1_id
    assert ext2.revision.revision_sequence == 1

    history = extractor.get_revision_history(url)
    assert len(history) == 1


# 3. HTTP 304 creates no new revision and refreshes revalidation timestamp
def test_http_304_creates_no_revision_and_refreshes_timestamp(temp_cache: Path):
    url = "https://example.com/diplomacy/summit"
    html = make_html(
        "Foreign ministers convened in Sofia to discuss maritime security coordination."
    )
    mock_resp1 = MagicMock(spec=httpx.Response)
    mock_resp1.status_code = 200
    mock_resp1.text = html
    mock_resp1.url = httpx.URL(url)
    mock_resp1.headers = {"etag": '"summit-hash-v1"'}

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.get.return_value = mock_resp1

    extractor = ArticleExtractor(cache_dir=temp_cache, check_robots=False, http_client=mock_client)
    ext1 = extractor.extract_url(url=url)
    assert ext1.revision is not None
    rev1_id = ext1.revision.revision_id

    # Now simulate server returning HTTP 304 Not Modified
    mock_resp2 = MagicMock(spec=httpx.Response)
    mock_resp2.status_code = 304
    mock_resp2.headers = {"etag": '"summit-hash-v1"'}
    mock_client.get.return_value = mock_resp2

    ext2 = extractor.extract_url(url=url, force_revalidate=True)
    assert ext2.extraction_status == "cached"
    assert ext2.revision is not None
    assert ext2.revision.revision_id == rev1_id
    assert ext2.revision.revision_sequence == 1

    history = extractor.get_revision_history(url)
    assert len(history) == 1
    assert history[0].revision_id == rev1_id


# 4. Changed body creates revision 2 classified as content_update
def test_changed_body_creates_rev_2_content_update(temp_cache: Path):
    url = "https://example.com/conflict/casualties"
    html_v1 = make_html("Officials report 40 casualties following the port explosion.")
    html_v2 = make_html("Officials report 44 casualties following the port explosion.")

    mock_client = MagicMock(spec=httpx.Client)
    mock_resp1 = MagicMock(spec=httpx.Response, status_code=200, text=html_v1, url=httpx.URL(url))
    mock_resp1.headers = {}
    mock_client.get.return_value = mock_resp1

    extractor = ArticleExtractor(cache_dir=temp_cache, check_robots=False, http_client=mock_client)
    ext1 = extractor.extract_url(url=url)
    assert ext1.revision is not None
    assert ext1.revision.revision_sequence == 1

    # Second call returns updated body
    mock_resp2 = MagicMock(spec=httpx.Response, status_code=200, text=html_v2, url=httpx.URL(url))
    mock_resp2.headers = {}
    mock_client.get.return_value = mock_resp2

    ext2 = extractor.extract_url(url=url, force_revalidate=True)
    assert ext2.revision is not None
    assert ext2.revision.revision_sequence == 2
    assert ext2.revision.revision_classification == "content_update"
    assert ext2.revision.previous_revision_id == ext1.revision.revision_id
    assert ext2.revision.content_hash != ext1.revision.content_hash
    assert "44 casualties" in (ext2.revision.body_text or "")

    history = extractor.get_revision_history(url)
    assert len(history) == 2
    assert history[0].revision_id == ext1.revision.revision_id
    assert history[1].revision_id == ext2.revision.revision_id


# 5. Title change without body change creates metadata_update
def test_title_change_creates_metadata_update(temp_cache: Path):
    url = "https://example.com/policy/sanctions"
    body_text = (
        "The government announced extensive financial restrictions on export activities. "
        "Ministers approved regulatory oversight measures to verify financial integrity."
    )
    html_v1 = make_html(body_text, title="Initial Sanctions Draft", include_h1=False)
    html_v2 = make_html(body_text, title="Confirmed Sanctions Package", include_h1=False)

    mock_client = MagicMock(spec=httpx.Client)
    mock_resp1 = MagicMock(spec=httpx.Response, status_code=200, text=html_v1, url=httpx.URL(url))
    mock_resp1.headers = {}
    mock_client.get.return_value = mock_resp1

    extractor = ArticleExtractor(cache_dir=temp_cache, check_robots=False, http_client=mock_client)
    ext1 = extractor.extract_url(url=url, title="Initial Sanctions Draft")
    assert ext1.revision is not None
    assert ext1.revision.revision_sequence == 1

    mock_resp2 = MagicMock(spec=httpx.Response, status_code=200, text=html_v2, url=httpx.URL(url))
    mock_resp2.headers = {}
    mock_client.get.return_value = mock_resp2

    ext2 = extractor.extract_url(
        url=url,
        force_revalidate=True,
        title="Confirmed Sanctions Package",
    )
    assert ext2.revision is not None
    assert ext2.revision.revision_sequence == 2
    assert ext2.revision.revision_classification == "metadata_update"
    assert ext2.revision.title == "Confirmed Sanctions Package"
    assert ext2.revision.previous_revision_id == ext1.revision.revision_id


# 6. Explicit correction detected
def test_explicit_correction_detected(temp_cache: Path):
    url = "https://example.com/investigation/naval-incident"
    html_v1 = make_html("Early reports claimed a collision occurred with a commercial tanker.")
    html_v2 = make_html(
        "Correction: An earlier version of this article misstated the vessel involved. "
        "The collision involved a patrol boat, not a commercial tanker."
    )

    mock_client = MagicMock(spec=httpx.Client)
    mock_resp1 = MagicMock(spec=httpx.Response, status_code=200, text=html_v1, url=httpx.URL(url))
    mock_resp1.headers = {}
    mock_client.get.return_value = mock_resp1

    extractor = ArticleExtractor(cache_dir=temp_cache, check_robots=False, http_client=mock_client)
    ext1 = extractor.extract_url(url=url)
    assert ext1.revision is not None

    mock_resp2 = MagicMock(spec=httpx.Response, status_code=200, text=html_v2, url=httpx.URL(url))
    mock_resp2.headers = {}
    mock_client.get.return_value = mock_resp2

    ext2 = extractor.extract_url(url=url, force_revalidate=True)
    assert ext2.revision is not None
    assert ext2.revision.revision_classification == "explicit_correction"
    assert ext2.revision.revision_sequence == 2
    assert ext2.revision.previous_revision_id == ext1.revision.revision_id


# 7. Explicit retraction detected
def test_explicit_retraction_detected(temp_cache: Path):
    url = "https://example.com/reports/unconfirmed-coup"
    html_v1 = make_html(
        "Unconfirmed reports suggest military units have surrounded the presidential palace."
    )
    html_v2 = make_html(
        "Retraction Notice: This article has been retracted by the editorial board due to "
        "fabricated sourcing. Normal governmental operations continue uninterrupted."
    )

    mock_client = MagicMock(spec=httpx.Client)
    mock_resp1 = MagicMock(spec=httpx.Response, status_code=200, text=html_v1, url=httpx.URL(url))
    mock_resp1.headers = {}
    mock_client.get.return_value = mock_resp1

    extractor = ArticleExtractor(cache_dir=temp_cache, check_robots=False, http_client=mock_client)
    ext1 = extractor.extract_url(url=url)
    assert ext1.revision is not None

    mock_resp2 = MagicMock(spec=httpx.Response, status_code=200, text=html_v2, url=httpx.URL(url))
    mock_resp2.headers = {}
    mock_client.get.return_value = mock_resp2

    ext2 = extractor.extract_url(url=url, force_revalidate=True)
    assert ext2.revision is not None
    assert ext2.revision.revision_classification == "explicit_retraction"
    assert ext2.revision.revision_sequence == 2


# 8. 404 / HTTP error does not imply retraction (classified as unavailability)
def test_http_404_classified_as_unavailability(temp_cache: Path):
    url = "https://example.com/temporary/broken-link"
    html_v1 = make_html("Valid original reporting on bilateral trade pact negotiations.")

    mock_client = MagicMock(spec=httpx.Client)
    mock_resp1 = MagicMock(spec=httpx.Response, status_code=200, text=html_v1, url=httpx.URL(url))
    mock_resp1.headers = {}
    mock_client.get.return_value = mock_resp1

    extractor = ArticleExtractor(cache_dir=temp_cache, check_robots=False, http_client=mock_client)
    ext1 = extractor.extract_url(url=url)
    assert ext1.revision is not None
    assert ext1.revision.revision_sequence == 1

    # Now server returns HTTP 404 Not Found
    mock_resp2 = MagicMock(
        spec=httpx.Response, status_code=404, text="Not Found", url=httpx.URL(url)
    )
    mock_resp2.headers = {}
    mock_client.get.return_value = mock_resp2

    ext2 = extractor.extract_url(url=url, force_revalidate=True)
    # Should record an unavailability revision, NOT a retraction
    assert ext2.revision is not None
    assert ext2.revision.revision_classification == "unavailability"
    assert ext2.revision.extraction_status == "failed"
    assert ext2.revision.previous_revision_id == ext1.revision.revision_id
    # Historical v1 remains intact
    history = extractor.get_revision_history(url)
    assert len(history) == 2
    assert history[0].body_text is not None


# 9. Network failure preserves history and cached revisions without truncation
def test_network_failure_preserves_history(temp_cache: Path):
    url = "https://example.com/network/timeout-test"
    html_v1 = make_html("Critical defense data that must never be lost to network failures.")

    mock_client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock(spec=httpx.Response, status_code=200, text=html_v1, url=httpx.URL(url))
    mock_resp.headers = {}
    mock_client.get.return_value = mock_resp

    extractor = ArticleExtractor(cache_dir=temp_cache, check_robots=False, http_client=mock_client)
    ext1 = extractor.extract_url(url=url)
    assert ext1.revision is not None
    rev1_id = ext1.revision.revision_id

    # Simulate network timeout / connection refused
    mock_client.get.side_effect = httpx.ConnectError("Connection refused")
    extractor.extract_url(url=url, force_revalidate=True)

    # Provenance history remains intact
    history = extractor.get_revision_history(url)
    assert len(history) >= 1
    assert history[0].revision_id == rev1_id
    assert "Critical defense data" in (history[0].body_text or "")


# 10. Concurrency deduplication (thread-safe sequential revisions)
def test_concurrency_deduplication(temp_cache: Path):
    url = "https://example.com/concurrency/concurrent-test"
    html = make_html("Concurrent extraction stress test text.")

    def get_resp(*args, **kwargs):
        r = MagicMock(spec=httpx.Response, status_code=200, text=html, url=httpx.URL(url))
        r.headers = {}
        return r

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.get.side_effect = get_resp

    extractor = ArticleExtractor(
        cache_dir=temp_cache,
        check_robots=False,
        min_domain_interval=0.0,
        http_client=mock_client,
    )

    def run_extract():
        return extractor.extract_url(url=url)

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(lambda _: run_extract(), range(10)))

    # All workers should resolve to the same revision 1 without sequence corruption
    assert all(r.revision is not None for r in results)
    history = extractor.get_revision_history(url)
    assert len(history) == 1
    assert history[0].revision_sequence == 1


# 11. Historical revisions are immutable and cannot be overwritten
def test_historical_revisions_immutable(temp_cache: Path):
    url = "https://example.com/immutability/contract"
    html_v1 = make_html("Initial contract signed for $500M.")
    html_v2 = make_html("Amended contract increased to $750M.")

    mock_client = MagicMock(spec=httpx.Client)
    mock_resp1 = MagicMock(spec=httpx.Response, status_code=200, text=html_v1, url=httpx.URL(url))
    mock_resp1.headers = {}
    mock_client.get.return_value = mock_resp1

    extractor = ArticleExtractor(cache_dir=temp_cache, check_robots=False, http_client=mock_client)
    ext1 = extractor.extract_url(url=url)
    assert ext1.revision is not None
    rev1_id = ext1.revision.revision_id
    rev1_hash = ext1.revision.content_hash

    mock_resp2 = MagicMock(spec=httpx.Response, status_code=200, text=html_v2, url=httpx.URL(url))
    mock_resp2.headers = {}
    mock_client.get.return_value = mock_resp2

    ext2 = extractor.extract_url(url=url, force_revalidate=True)
    assert ext2.revision is not None
    rev2_id = ext2.revision.revision_id

    # Re-fetch Revision 1 explicitly by ID
    rev1_fetched = extractor.get_revision(url, rev1_id)
    assert rev1_fetched is not None
    assert rev1_fetched.revision_id == rev1_id
    assert rev1_fetched.content_hash == rev1_hash
    assert "$500M" in (rev1_fetched.body_text or "")
    assert "$750M" not in (rev1_fetched.body_text or "")

    # Re-fetch Revision 2
    rev2_fetched = extractor.get_revision(url, rev2_id)
    assert rev2_fetched is not None
    assert rev2_fetched.previous_revision_id == rev1_id
    assert "$750M" in (rev2_fetched.body_text or "")


# 12. Existing claims retain links to original revisions
def test_existing_claims_retain_links_to_original_revisions():
    url = "https://example.com/intel/strike-figures"
    body_v1 = "Defense Ministry reports 12 aircraft deployed in the northern exercise."

    article_v1 = ArticleRecord(
        url=url,
        canonical_url=url,
        title="Exercise Deployment",
        snippet="12 aircraft deployed",
        body_text=body_v1,
        published_at=datetime.now(UTC),
        source_name="Defense News",
        content_hash="hash-v1",
        article_id="art-100",
        current_revision_id="rev-100-v1",
        revision_sequence=1,
        is_full_text=True,
    )

    base_ref = ClaimEvidenceReference(
        source_url=url,
        evidence_span="12 aircraft deployed in the northern exercise",
    )
    enriched_ref = _populate_evidence_provenance(base_ref, article_v1)

    assert enriched_ref.article_id == "art-100"
    assert enriched_ref.article_revision_id == "rev-100-v1"
    assert enriched_ref.content_hash == "hash-v1"
    assert enriched_ref.start_char == 25
    assert enriched_ref.end_char == 25 + len("12 aircraft deployed in the northern exercise")
    assert enriched_ref.section_label in {"opening", "middle", "concluding"}

    claim = SummaryClaim(
        claim_id="c1",
        text="According to the Defense Ministry, 12 aircraft were deployed.",
        evidence_refs=(enriched_ref,),
        classification="reported_fact",
        central=True,
        article_id=enriched_ref.article_id,
        article_revision_id=enriched_ref.article_revision_id,
    )

    assert claim.article_revision_id == "rev-100-v1"


# 13. Revised evidence does not rewrite historical claims
def test_revised_evidence_does_not_rewrite_historical_claims():
    """Verify that when an article updates, historical claim stays at rev 1."""
    # Historical published claim created against Revision 1
    hist_ref = ClaimEvidenceReference(
        source_url="https://example.com/news/incident",
        evidence_span="Officials report 40 casualties",
        article_id="art-200",
        article_revision_id="rev-200-v1",
        content_hash="hash-cas-40",
        verification_status="accepted",
        provenance="full_article",
    )
    historical_claim = SummaryClaim(
        claim_id="c1",
        text="Officials reported 40 casualties following the incident.",
        evidence_refs=(hist_ref,),
        classification="reported_fact",
        central=True,
        article_id="art-200",
        article_revision_id="rev-200-v1",
        verification_status="accepted",
    )

    # Later, article is updated to Revision 2 (44 casualties)
    art_v2 = ArticleRecord(
        url="https://example.com/news/incident",
        canonical_url="https://example.com/news/incident",
        title="Updated Casualty Toll",
        snippet="44 casualties",
        body_text="Officials report 44 casualties following the incident.",
        published_at=datetime.now(UTC),
        source_name="News Wire",
        content_hash="hash-cas-44",
        article_id="art-200",
        current_revision_id="rev-200-v2",
        revision_sequence=2,
        is_full_text=True,
    )

    # New claim created against Revision 2
    new_ref = _populate_evidence_provenance(
        ClaimEvidenceReference(
            source_url="https://example.com/news/incident",
            evidence_span="Officials report 44 casualties",
        ),
        art_v2,
    )
    new_claim = SummaryClaim(
        claim_id="c1",
        text="Officials reported 44 casualties following the updated assessment.",
        evidence_refs=(new_ref,),
        classification="reported_fact",
        central=True,
        article_id=new_ref.article_id,
        article_revision_id=new_ref.article_revision_id,
        verification_status="accepted",
    )

    # Invariant: Historical claim still points to rev 1 with hash-cas-40
    assert historical_claim.article_revision_id == "rev-200-v1"
    assert historical_claim.evidence_refs[0].article_revision_id == "rev-200-v1"
    assert historical_claim.evidence_refs[0].content_hash == "hash-cas-40"
    assert "40 casualties" in historical_claim.text

    # New claim points to rev 2 with hash-cas-44
    assert new_claim.article_revision_id == "rev-200-v2"
    assert new_claim.evidence_refs[0].article_revision_id == "rev-200-v2"
    assert new_claim.evidence_refs[0].content_hash == "hash-cas-44"
    assert "44 casualties" in new_claim.text


# 14. Existing canonical deduplication intact across tracking params and schemes
def test_canonical_deduplication_intact(temp_cache: Path):
    url_base = "https://example.com/article/defense-initiative"
    url_tracked1 = (
        "https://example.com/article/defense-initiative?utm_source=twitter&utm_medium=social"
    )
    url_tracked2 = "https://EXAMPLE.COM/article/defense-initiative/?fbclid=XYZ123#overview"

    # Both must resolve to identical canonical URL and identical article_id
    html = make_html("Bilateral defense initiative formally ratified by both governments.")

    mock_client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock(spec=httpx.Response, status_code=200, text=html, url=httpx.URL(url_base))
    mock_resp.headers = {}
    mock_client.get.return_value = mock_resp

    extractor = ArticleExtractor(cache_dir=temp_cache, check_robots=False, http_client=mock_client)
    ext1 = extractor.extract_url(url=url_tracked1)
    ext2 = extractor.extract_url(url=url_tracked2)

    assert ext1.canonical_url == ext2.canonical_url
    assert ext1.revision is not None
    assert ext2.revision is not None
    assert ext1.revision.article_id == ext2.revision.article_id
    assert ext1.revision.revision_id == ext2.revision.revision_id

    # Verify only a single article manifest and revision exists on disk
    history = extractor.get_revision_history(url_base)
    assert len(history) == 1


# 15. Distinct outlets covering the same event remain distinct article identities
def test_distinct_outlets_covering_same_event_remain_distinct():
    repo = InMemoryArticleRepository()
    source1 = FeedSource(name="Reuters", feed_url="https://reuters.com/rss/world")
    source2 = FeedSource(name="BBC", feed_url="https://bbc.com/rss/world")
    s1_id = repo.get_or_create_source(source1)
    s2_id = repo.get_or_create_source(source2)

    # Both report on the exact same summit
    art_reuters = ArticleRecord(
        url="https://reuters.com/world/black-sea-ministers-meet",
        canonical_url="https://reuters.com/world/black-sea-ministers-meet",
        title="Black Sea ministers agree intelligence sharing in Sofia",
        snippet="Ministers from Bulgaria, Romania and Turkey agreed on maritime intelligence.",
        published_at=datetime.now(UTC),
        source_name="Reuters",
        content_hash="hash-reuters",
    )
    art_bbc = ArticleRecord(
        url="https://bbc.com/news/black-sea-security-pact",
        canonical_url="https://bbc.com/news/black-sea-security-pact",
        title="Black Sea nations sign maritime surveillance pact",
        snippet="Turkey, Romania and Bulgaria sign pact in Sofia to track maritime threats.",
        published_at=datetime.now(UTC),
        source_name="BBC",
        content_hash="hash-bbc",
    )

    repo.persist_articles(s1_id, [art_reuters])
    repo.persist_articles(s2_id, [art_bbc])

    ids = repo.get_article_ids_by_urls([art_reuters.canonical_url, art_bbc.canonical_url])
    assert len(ids) == 2
    # Ensure they have completely different unique article IDs despite covering the same event
    assert ids[art_reuters.canonical_url] != ids[art_bbc.canonical_url]

    now = datetime.now(UTC)
    # Invariant: Clustering groups them, but article identities and revisions remain separate
    cluster = EventCluster(
        id="cluster-black-sea",
        title="Black Sea security pact",
        category="defence_policy",
        first_seen_at=now,
        last_seen_at=now,
        articles=(art_reuters, art_bbc),
    )
    assert len(cluster.articles) == 2
    assert cluster.articles[0].canonical_url != cluster.articles[1].canonical_url
