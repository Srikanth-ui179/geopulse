"""Curated and verified RSS feeds for geopolitical and defence reporting."""

from geopulse.ingestion.models import FeedSource

# Verified, accessible, and reputable defence and geopolitical feeds.
DEFAULT_FEEDS: tuple[FeedSource, ...] = (
    FeedSource(
        name="Defense News",
        feed_url="https://www.defensenews.com/arc/outboundfeeds/rss/category/global/?outputType=xml",
        enabled=True,
    ),
    FeedSource(
        name="Breaking Defense",
        feed_url="https://breakingdefense.com/feed/",
        enabled=True,
    ),
    FeedSource(
        name="War on the Rocks",
        feed_url="https://warontherocks.com/feed/",
        enabled=True,
    ),
    FeedSource(
        name="UK Ministry of Defence",
        feed_url="https://www.gov.uk/government/organisations/ministry-of-defence.atom",
        enabled=True,
    ),
    FeedSource(
        name="BBC World News",
        feed_url="http://feeds.bbci.co.uk/news/world/rss.xml",
        enabled=True,
    ),
    FeedSource(
        name="Al Jazeera English",
        feed_url="https://www.aljazeera.com/xml/rss/all.xml",
        enabled=True,
    ),
)


def get_default_feeds() -> list[FeedSource]:
    """Return a list of enabled default feed sources."""
    return [feed for feed in DEFAULT_FEEDS if feed.enabled]


def find_feed(query: str, feeds: list[FeedSource] | None = None) -> FeedSource | None:
    """Find a feed by exact or case-insensitive name or URL substring match."""
    target_feeds = feeds if feeds is not None else list(DEFAULT_FEEDS)
    clean_query = query.strip().lower()
    for feed in target_feeds:
        if clean_query == feed.name.lower() or clean_query == feed.feed_url.lower():
            return feed
    for feed in target_feeds:
        if clean_query in feed.name.lower() or clean_query in feed.feed_url.lower():
            return feed
    return None
