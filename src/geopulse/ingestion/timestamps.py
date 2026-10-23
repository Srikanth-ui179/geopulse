"""Robust publication timestamp parsing into UTC."""

import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any


def parse_entry_timestamp(entry: dict[str, Any]) -> datetime | None:
    """Extract and parse publication timestamp from a feedparser entry as UTC datetime.

    Checks published_parsed, updated_parsed, published, updated, and pubDate fields.
    Returns None if no valid timestamp can be parsed.
    """
    # 1. Prefer parsed struct_time provided directly by feedparser
    for struct_key in ("published_parsed", "updated_parsed", "created_parsed"):
        struct_val = entry.get(struct_key)
        if isinstance(struct_val, time.struct_time):
            try:
                return datetime(*struct_val[:6], tzinfo=UTC)
            except (ValueError, OverflowError):
                continue

    # 2. Fall back to raw date strings (RFC 2822 / ISO 8601)
    for str_key in ("published", "updated", "pubDate", "dc:date"):
        raw_val = entry.get(str_key)
        if isinstance(raw_val, str) and raw_val.strip():
            cleaned = raw_val.strip()
            # Try RFC 2822 (e.g., "Thu, 22 Oct 2026 08:00:00 GMT")
            try:
                dt = parsedate_to_datetime(cleaned)
                if dt.tzinfo is None:
                    return dt.replace(tzinfo=UTC)
                return dt.astimezone(UTC)
            except (ValueError, TypeError, IndexError):
                pass

            # Try ISO 8601 format
            try:
                iso_cleaned = cleaned.replace("Z", "+00:00")
                dt = datetime.fromisoformat(iso_cleaned)
                if dt.tzinfo is None:
                    return dt.replace(tzinfo=UTC)
                return dt.astimezone(UTC)
            except (ValueError, TypeError):
                pass

    return None
