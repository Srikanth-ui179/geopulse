"""Unit tests for timestamp parsing from feed entries."""

import time
from datetime import UTC, datetime

from geopulse.ingestion.timestamps import parse_entry_timestamp


def test_parse_timestamp_from_published_parsed():
    # 2026-10-22 08:30:00 UTC
    struct_time = time.gmtime(1792657800)
    entry = {"published_parsed": struct_time}

    dt = parse_entry_timestamp(entry)
    assert dt is not None
    assert dt.tzinfo == UTC
    assert dt.year == 2026


def test_parse_timestamp_from_rfc2822_string():
    entry = {"published": "Thu, 22 Oct 2026 09:15:00 GMT"}
    dt = parse_entry_timestamp(entry)
    assert dt is not None
    assert dt.tzinfo == UTC
    assert dt == datetime(2026, 10, 22, 9, 15, 0, tzinfo=UTC)


def test_parse_timestamp_from_iso8601_string():
    entry = {"updated": "2026-10-22T14:45:00Z"}
    dt = parse_entry_timestamp(entry)
    assert dt is not None
    assert dt.tzinfo == UTC
    assert dt == datetime(2026, 10, 22, 14, 45, 0, tzinfo=UTC)


def test_parse_timestamp_falls_back_through_fields():
    entry = {
        "published": "invalid-date",
        "updated": "Thu, 22 Oct 2026 12:00:00 +0000",
    }
    dt = parse_entry_timestamp(entry)
    assert dt is not None
    assert dt == datetime(2026, 10, 22, 12, 0, 0, tzinfo=UTC)


def test_parse_timestamp_returns_none_on_missing_or_invalid():
    assert parse_entry_timestamp({}) is None
    assert parse_entry_timestamp({"published": "not-a-valid-date"}) is None
    assert parse_entry_timestamp({"published": ""}) is None
