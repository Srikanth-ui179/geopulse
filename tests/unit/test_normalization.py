"""Unit tests for URL normalization, HTML cleaning, and content hashing."""

import pytest

from geopulse.ingestion.normalization import clean_text, compute_content_hash, normalize_url


def test_normalize_url_strips_tracking_parameters():
    raw = (
        "https://www.defensenews.com/global/2026/10/22/radar-upgrade/?"
        "utm_source=twitter&utm_medium=social&utm_campaign=daily&fbclid=XYZ123&id=456"
    )
    normalized = normalize_url(raw)
    assert "utm_source" not in normalized
    assert "utm_medium" not in normalized
    assert "utm_campaign" not in normalized
    assert "fbclid" not in normalized
    assert "id=456" in normalized


def test_normalize_url_sorts_query_parameters_deterministically():
    url1 = "https://example.com/article?b=2&a=1"
    url2 = "https://example.com/article?a=1&b=2"
    assert normalize_url(url1) == normalize_url(url2)
    assert normalize_url(url1) == "https://example.com/article?a=1&b=2"


def test_normalize_url_removes_fragment_and_default_ports():
    raw_http = "http://example.com:80/news/story/#section-comments"
    assert normalize_url(raw_http) == "http://example.com/news/story"

    raw_https = "https://example.com:443/news/story/#top"
    assert normalize_url(raw_https) == "https://example.com/news/story"


def test_normalize_url_collapses_slashes():
    raw = "https://example.com//world///conflict//item/"
    assert normalize_url(raw) == "https://example.com/world/conflict/item"


def test_normalize_url_rejects_invalid_scheme():
    with pytest.raises(ValueError, match="Invalid URL scheme"):
        normalize_url("ftp://example.com/resource.xml")


def test_normalize_url_rejects_empty():
    with pytest.raises(ValueError, match="URL cannot be empty"):
        normalize_url("   ")


def test_clean_text_strips_html_tags_and_unescapes_entities():
    raw_html = "<p>Military officials announced a <b>ceasefire</b> &amp; peace talks.</p>"
    cleaned = clean_text(raw_html)
    assert cleaned == "Military officials announced a ceasefire & peace talks."


def test_clean_text_handles_empty_and_whitespace():
    assert clean_text("") is None
    assert clean_text("   ") is None
    assert clean_text(None) is None


def test_compute_content_hash_deterministic():
    url = "https://example.com/article"
    title = "Troops Deployed"
    snippet = "Forces crossed the border."

    h1 = compute_content_hash(url, title, snippet)
    h2 = compute_content_hash(url, title, snippet)
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex digest length


def test_compute_content_hash_differs_on_content_change():
    url = "https://example.com/article"
    h1 = compute_content_hash(url, "Troops Deployed", "Forces crossed.")
    h2 = compute_content_hash(url, "Troops Withdrawn", "Forces crossed.")
    assert h1 != h2
