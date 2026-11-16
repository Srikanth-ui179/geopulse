"""Unit tests for the respectful, cached full-article extraction layer."""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx

from geopulse.ingestion.extractor import (
    ArticleExtractor,
    ExtractedContent,
)
from geopulse.ingestion.models import ArticleRecord

SAMPLE_HTML = """
<!DOCTYPE html>
<html>
<head><title>Black Sea Security Summit Concludes</title></head>
<body>
<nav><a href="/">Home</a></nav>
<article>
<h1>Black Sea Security Summit Concludes in Sofia</h1>
<p>
Defense ministers from Bulgaria, Romania, and Turkey convened in Sofia on October 28
to address maritime security challenges in the Black Sea.
</p>
<p>
The ministers agreed to establish an enhanced intelligence-sharing framework
and expand continuous maritime domain awareness monitoring across the western Black Sea.
</p>
<p>
The joint statement emphasized coordination among littoral NATO allies to ensure
freedom of navigation and detect floating naval mines.
</p>
</article>
<footer>Copyright 2026 Example News Agency</footer>
</body>
</html>
"""

SAMPLE_PAYWALL_HTML = """
<!DOCTYPE html>
<html>
<head><title>Exclusive Analysis</title></head>
<body>
<h1>Naval Modernization Trends in Northern Europe</h1>
<p>Subscribe to continue reading this premium intelligence report.</p>
<p>Already a subscriber? Log in to access the full article.</p>
</body>
</html>
"""

SAMPLE_SHORT_HTML = """
<!DOCTYPE html>
<html>
<body>
<p>Short note.</p>
</body>
</html>
"""


def test_extractor_extracts_clean_body_and_caches_to_disk(tmp_path: Path):
    """Test successful article body extraction and deterministic caching."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.text = SAMPLE_HTML
    mock_resp.url = httpx.URL("https://example.com/news/black-sea-summit")

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.get.return_value = mock_resp

    extractor = ArticleExtractor(
        cache_dir=tmp_path / "cache",
        check_robots=False,
        http_client=mock_client,
    )

    result = extractor.extract_url(
        url="https://example.com/news/black-sea-summit?ref=rss",
        original_snippet="Brief RSS snippet",
    )

    assert result.extraction_status == "success"
    assert result.is_full_text is True
    assert result.body_text is not None
    assert "Sofia on October 28" in result.body_text
    assert "intelligence-sharing framework" in result.body_text
    assert result.word_count > 30
    assert result.content_hash is not None

    # Verify cache file exists
    cache_path = extractor._get_cache_path("https://example.com/news/black-sea-summit")
    assert cache_path.exists()

    # Second extraction should read from cache without network GET
    mock_client.get.reset_mock()
    cached_result = extractor.extract_url(
        url="https://example.com/news/black-sea-summit?ref=rss",
    )
    assert cached_result.extraction_status == "cached"
    assert cached_result.is_full_text is True
    assert cached_result.body_text == result.body_text
    mock_client.get.assert_not_called()


def test_extractor_detects_paywalls_and_falls_back_to_snippet(tmp_path: Path):
    """Test that paywalled articles are flagged without raising exceptions."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.text = SAMPLE_PAYWALL_HTML
    mock_resp.url = httpx.URL("https://example.com/premium/analysis")

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.get.return_value = mock_resp

    extractor = ArticleExtractor(
        cache_dir=tmp_path / "cache",
        check_robots=False,
        http_client=mock_client,
    )

    snippet = "Norway and Germany advance naval defense procurement under bilateral framework."
    result = extractor.extract_url(
        url="https://example.com/premium/analysis",
        original_snippet=snippet,
    )

    assert result.extraction_status == "paywall_detected"
    assert result.is_full_text is False
    assert result.body_text == snippet


def test_extractor_detects_http_401_403_as_paywall(tmp_path: Path):
    """HTTP 401 or 403 responses are safely handled as paywall/auth required."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 403

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.get.return_value = mock_resp

    extractor = ArticleExtractor(
        cache_dir=tmp_path / "cache",
        check_robots=False,
        http_client=mock_client,
    )

    result = extractor.extract_url(
        url="https://example.com/paywalled-story",
        original_snippet="Snippet fallback text",
    )

    assert result.extraction_status == "paywall_detected"
    assert result.is_full_text is False
    assert result.body_text == "Snippet fallback text"


def test_extractor_handles_empty_or_too_short_html(tmp_path: Path):
    """Extracting an HTML stub lacking substantive paragraphs falls back cleanly."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.text = SAMPLE_SHORT_HTML
    mock_resp.url = httpx.URL("https://example.com/short")

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.get.return_value = mock_resp

    extractor = ArticleExtractor(
        cache_dir=tmp_path / "cache",
        check_robots=False,
        http_client=mock_client,
    )

    result = extractor.extract_url(
        url="https://example.com/short",
        original_snippet="Original RSS snippet text",
    )

    assert result.extraction_status == "empty"
    assert result.is_full_text is False
    assert result.body_text == "Original RSS snippet text"


def test_extractor_respects_robots_txt(tmp_path: Path):
    """Test that robots.txt disallow rules are respected."""
    extractor = ArticleExtractor(
        cache_dir=tmp_path / "cache",
        check_robots=True,
    )

    with patch.object(extractor, "_is_allowed_by_robots", return_value=False):
        result = extractor.extract_url(
            url="https://example.com/disallowed-path/story",
            original_snippet="Snippet text",
        )

    assert result.extraction_status == "robots_denied"
    assert result.is_full_text is False
    assert result.body_text == "Snippet text"


def test_enrich_article_record(tmp_path: Path):
    """Test that ArticleRecord is properly enriched with full body text and provenance."""
    extractor = ArticleExtractor(
        cache_dir=tmp_path / "cache",
        check_robots=False,
    )

    record = ArticleRecord(
        url="https://example.com/story-123",
        canonical_url="https://example.com/story-123",
        title="Test Article Title",
        snippet="Short RSS summary",
        published_at=datetime.now(UTC),
        source_name="Defense Outlet",
        content_hash="abc12345",
    )

    mock_content = ExtractedContent(
        url=record.url,
        canonical_url=record.canonical_url,
        body_text="Comprehensive extracted article text containing substantive details.",
        extraction_status="success",
        status_reason=None,
        extracted_at=datetime.now(UTC),
        content_hash="def67890",
        word_count=8,
        is_full_text=True,
    )

    with patch.object(extractor, "extract_url", return_value=mock_content):
        enriched = extractor.enrich_article_record(record)

    assert enriched.is_full_text is True
    assert enriched.extraction_status == "success"
    assert "Comprehensive extracted article" in (enriched.body_text or "")
    assert enriched.title == record.title
    assert enriched.url == record.url


def test_enrich_article_record_skips_if_already_full_text(tmp_path: Path):
    """Articles already populated with full text do not trigger extraction."""
    extractor = ArticleExtractor(cache_dir=tmp_path / "cache")

    already_full = ArticleRecord(
        url="https://example.com/already-full",
        canonical_url="https://example.com/already-full",
        title="Already Extracted",
        snippet="Snippet",
        body_text="Already full body text with lots of details.",
        published_at=datetime.now(UTC),
        source_name="Outlet",
        content_hash="11122233",
        is_full_text=True,
    )

    with patch.object(extractor, "extract_url") as mock_extract:
        enriched = extractor.enrich_article_record(already_full)
        mock_extract.assert_not_called()

    assert enriched == already_full


def test_select_bounded_article_passages_covers_opening_middle_concluding():
    """Verify bounded passage selection captures opening, middle, and conclusion."""
    from geopulse.ingestion.extractor import select_bounded_article_passages

    long_article = (
        "The UK Ministry of Defence unveiled a new industrial offset policy in London.\n\n"
        "Under landmark rules, foreign contractors winning defence contracts valued at "
        "£50 million or more must provide domestic economic benefits valued at up to 100% "
        "of the contract value. The newly established Defence Offset Authority will oversee "
        "compliance, investments in British factories, and technology transfer starting "
        "in April 2027.\n\n"
        "Officials noted routine administrative meetings took place with trade delegates.\n\n"
        "Separate working groups discussed general naval safety protocols in ports.\n\n"
        "The mandatory offset policy formally takes effect in April 2027, with foreign suppliers "
        "subject to biennial compliance reviews."
    )

    formatted, passages = select_bounded_article_passages(long_article, max_chars=550)

    assert len(formatted) <= 550
    assert len(passages) >= 3

    # Check coverage of opening lead
    assert "Ministry of Defence unveiled a new industrial offset policy" in passages[0].text
    assert passages[0].start_char == 0

    # Check coverage of substantive middle with financial thresholds & mechanisms
    middle_texts = " ".join(p.text for p in passages[1:-1])
    assert "£50 million" in middle_texts or "Defence Offset Authority" in middle_texts

    # Check coverage of concluding horizon
    assert "April 2027" in passages[-1].text or "compliance reviews" in passages[-1].text


def test_etag_last_modified_conditional_revalidation_304(tmp_path: Path):
    """Verify HTTP 304 Not Modified revalidation retains cached content."""
    from geopulse.ingestion.extractor import ExtractedContent

    extractor = ArticleExtractor(
        cache_dir=tmp_path / "cache",
        check_robots=False,
    )

    norm_url = "https://example.com/live-story"
    old_content = ExtractedContent(
        url=norm_url,
        canonical_url=norm_url,
        body_text="Stable article text with sufficient substance and detail for testing.",
        extraction_status="success",
        status_reason=None,
        extracted_at=datetime.now(UTC),
        content_hash="hash123",
        word_count=10,
        is_full_text=True,
        etag='"etag-abc"',
        last_modified="Wed, 21 Oct 2026 07:28:00 GMT",
    )
    extractor._write_to_cache(old_content)

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 304
    mock_resp.headers = {"etag": '"etag-abc"'}

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.get.return_value = mock_resp
    extractor._external_client = mock_client

    result = extractor.extract_url(
        url=norm_url,
        force_revalidate=True,
    )

    assert result.extraction_status == "cached"
    assert result.body_text == old_content.body_text
    mock_client.get.assert_called_once()
    called_headers = mock_client.get.call_args[1]["headers"]
    assert called_headers.get("If-None-Match") == '"etag-abc"'
    assert called_headers.get("If-Modified-Since") == "Wed, 21 Oct 2026 07:28:00 GMT"


def test_article_revision_detected_when_content_changes(tmp_path: Path):
    """Verify that updated article content is captured as a distinct revision."""
    from geopulse.ingestion.extractor import ExtractedContent

    extractor = ArticleExtractor(
        cache_dir=tmp_path / "cache",
        check_robots=False,
    )

    norm_url = "https://example.com/developing-story"
    old_content = ExtractedContent(
        url=norm_url,
        canonical_url=norm_url,
        body_text="Initial report: 18 vessels were impacted in maritime security operations.",
        extraction_status="success",
        status_reason=None,
        extracted_at=datetime.now(UTC),
        content_hash="oldhash111",
        word_count=10,
        is_full_text=True,
        etag='"v1"',
    )
    extractor._write_to_cache(old_content)

    updated_html = """
    <html><body><article>
    <h1>Developing Maritime Story</h1>
    <p>Updated report: 25 vessels have been hit according to the latest maritime risk analysis.</p>
    <p>Naval authorities confirmed expanded patrols across the entire western corridor today.</p>
    <p>Littoral NATO allies are coordinating joint reconnaissance flights in response.</p>
    </article></body></html>
    """

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.text = updated_html
    mock_resp.headers = {"etag": '"v2"'}

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.get.return_value = mock_resp
    extractor._external_client = mock_client

    result = extractor.extract_url(
        url=norm_url,
        force_revalidate=True,
    )

    assert result.extraction_status == "updated"
    assert result.is_full_text is True
    assert "25 vessels have been hit" in (result.body_text or "")
    assert "oldhash111" in result.previous_content_hashes
    assert result.content_hash != "oldhash111"
