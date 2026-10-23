"""Isolated network fetcher for RSS and Atom feeds."""

import urllib.request
from urllib.error import HTTPError, URLError

DEFAULT_USER_AGENT = "Geopulse/0.1.0 (+https://github.com/geopulse-bot; contact@geopulse.local)"
DEFAULT_TIMEOUT_SECONDS = 15.0


class FeedFetchError(Exception):
    """Raised when an RSS feed cannot be retrieved over HTTP."""

    def __init__(self, feed_url: str, reason: str):
        super().__init__(f"Failed to fetch feed at {feed_url}: {reason}")
        self.feed_url = feed_url
        self.reason = reason


def fetch_feed_data(
    feed_url: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    user_agent: str = DEFAULT_USER_AGENT,
) -> str:
    """Fetch raw feed XML/Atom text from the network with timeout and error handling.

    This function isolates network I/O so unit tests can mock it directly.
    """
    req = urllib.request.Request(
        feed_url,
        headers={
            "User-Agent": user_agent,
            "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml;q=0.9",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            raw_bytes = response.read()
            return raw_bytes.decode(charset, errors="replace")
    except HTTPError as exc:
        raise FeedFetchError(feed_url, f"HTTP Error {exc.code}: {exc.reason}") from exc
    except URLError as exc:
        raise FeedFetchError(feed_url, f"URL Error: {exc.reason}") from exc
    except TimeoutError as exc:
        raise FeedFetchError(feed_url, f"Timed out after {timeout} seconds") from exc
    except Exception as exc:
        raise FeedFetchError(feed_url, f"Unexpected error: {exc}") from exc
