"""Isolated network fetcher for RSS and Atom feeds."""

import logging
import time

import httpx

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "Geopulse/0.1.0 (+https://github.com/geopulse-bot; contact@geopulse.local)"
DEFAULT_CONNECT_TIMEOUT_SECONDS = 5.0
DEFAULT_READ_TIMEOUT_SECONDS = 10.0
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_RETRIES = 1
DEFAULT_RETRY_BACKOFF_SECONDS = 0.5


class FeedFetchError(Exception):
    """Raised when an RSS feed cannot be retrieved over HTTP."""

    def __init__(self, feed_url: str, reason: str):
        super().__init__(f"Failed to fetch feed at {feed_url}: {reason}")
        self.feed_url = feed_url
        self.reason = reason


def fetch_feed_data(
    feed_url: str,
    timeout: float | httpx.Timeout = DEFAULT_TIMEOUT_SECONDS,
    user_agent: str = DEFAULT_USER_AGENT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF_SECONDS,
) -> str:
    """Fetch raw feed XML/Atom text from the network with timeout, retry, and error handling.

    This function isolates network I/O so unit tests can mock it directly.
    """
    if isinstance(timeout, httpx.Timeout):
        http_timeout = timeout
    else:
        conn_timeout = min(DEFAULT_CONNECT_TIMEOUT_SECONDS, float(timeout))
        read_timeout = float(timeout)
        http_timeout = httpx.Timeout(
            connect=conn_timeout,
            read=read_timeout,
            write=5.0,
            pool=5.0,
        )

    headers = {
        "User-Agent": user_agent,
        "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml;q=0.9",
    }

    attempts = 0
    last_exc: Exception | None = None

    while attempts <= max_retries:
        attempts += 1
        try:
            with httpx.Client(timeout=http_timeout, follow_redirects=True) as client:
                response = client.get(feed_url, headers=headers)
                response.raise_for_status()
                encoding = response.encoding or "utf-8"
                return response.content.decode(encoding, errors="replace")
        except httpx.HTTPStatusError as exc:
            raise FeedFetchError(
                feed_url, f"HTTP Error {exc.response.status_code}: {exc.response.reason_phrase}"
            ) from exc
        except (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
        ) as exc:
            last_exc = exc
            if attempts <= max_retries:
                logger.debug(
                    "Retrying feed fetch for %s after transient error (attempt %d/%d): %s",
                    feed_url,
                    attempts,
                    max_retries + 1,
                    exc,
                )
                time.sleep(retry_backoff)
                continue
            timeout_desc = (
                f"Timed out after {timeout} seconds"
                if isinstance(exc, (httpx.ConnectTimeout, httpx.ReadTimeout))
                else f"Network error: {exc}"
            )
            raise FeedFetchError(feed_url, timeout_desc) from exc
        except httpx.RequestError as exc:
            raise FeedFetchError(feed_url, f"Request error: {exc}") from exc
        except Exception as exc:
            raise FeedFetchError(feed_url, f"Unexpected error: {exc}") from exc

    raise FeedFetchError(feed_url, f"Failed after {attempts} attempts: {last_exc}")
