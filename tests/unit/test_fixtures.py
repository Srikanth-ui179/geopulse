"""Unit tests verifying test fixtures."""

import json
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

FIXTURES_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "sample_articles.json"


def test_sample_articles_fixture_exists_and_is_valid():
    assert FIXTURES_PATH.is_file()
    data = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) >= 1

    required_fields = {"title", "url", "source", "published_at", "snippet"}
    for item in data:
        assert isinstance(item, dict)
        missing = required_fields - item.keys()
        assert not missing, f"Missing required fields: {missing}"

        # Validate URL format
        parsed_url = urlparse(item["url"])
        assert parsed_url.scheme in {"http", "https"}, f"Invalid URL scheme: {item['url']}"
        assert parsed_url.netloc, f"Invalid URL netloc: {item['url']}"

        # Validate timestamp format
        dt = datetime.fromisoformat(item["published_at"])
        assert dt is not None
