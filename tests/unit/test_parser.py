"""Unit tests for feedparser-based RSS/Atom content parsing."""

from geopulse.ingestion.parser import parse_feed_content

SAMPLE_RSS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Defense News</title>
    <link>https://www.defensenews.com</link>
    <description>Military News</description>
    <item>
      <title>Air Force Tests Hypersonic Missile Prototype</title>
      <link>https://www.defensenews.com/air/2026/10/22/hypersonic-test/?utm_source=rss</link>
      <description>&lt;p&gt;Officials confirmed a test over Pacific.&lt;/p&gt;</description>
      <author>Jane Reporter</author>
      <pubDate>Thu, 22 Oct 2026 10:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Navy Commissions New Destroyer</title>
      <link>https://www.defensenews.com/naval/2026/10/22/new-destroyer/</link>
      <description>Ceremony held at Norfolk naval station.</description>
      <pubDate>Thu, 22 Oct 2026 11:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""

SAMPLE_ATOM_XML = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>UK Ministry of Defence</title>
  <link href="https://www.gov.uk/government/organisations/ministry-of-defence"/>
  <entry>
    <title>Defence Secretary visits joint training operation</title>
    <link rel="alternate" href="https://www.gov.uk/government/news/defence-secretary-visit"/>
    <id>urn:uuid:12345</id>
    <updated>2026-10-22T09:30:00Z</updated>
    <summary>Joint NATO training exercise concludes in Poland.</summary>
    <author><name>MoD Press Office</name></author>
  </entry>
</feed>
"""

SAMPLE_MALFORMED_XML = """<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <title>Broken Feed</title>
    <!-- Missing title -->
    <item>
      <link>https://example.com/no-title</link>
      <description>Has no title</description>
    </item>
    <!-- Missing link -->
    <item>
      <title>Valid Title But No Link</title>
      <description>Has no link</description>
    </item>
    <!-- Valid entry -->
    <item>
      <title>Troops mobilize for border security exercise</title>
      <link>https://example.com/valid-story</link>
      <description>Security forces stationed at borders.</description>
      <pubDate>Thu, 22 Oct 2026 13:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""


def test_parse_valid_rss_feed():
    articles = parse_feed_content(SAMPLE_RSS_XML, source_name="Defense News")
    assert len(articles) == 2

    first = articles[0]
    assert first.source_name == "Defense News"
    assert first.title == "Air Force Tests Hypersonic Missile Prototype"
    assert "utm_source" not in first.canonical_url
    assert first.canonical_url == "https://www.defensenews.com/air/2026/10/22/hypersonic-test"
    assert first.author == "Jane Reporter"
    assert first.snippet == "Officials confirmed a test over Pacific."
    assert first.relevance_status == "relevant"
    assert first.published_at is not None


def test_parse_valid_atom_feed():
    articles = parse_feed_content(SAMPLE_ATOM_XML, source_name="UK Ministry of Defence")
    assert len(articles) == 1

    entry = articles[0]
    assert entry.title == "Defence Secretary visits joint training operation"
    assert entry.canonical_url == "https://www.gov.uk/government/news/defence-secretary-visit"
    assert entry.author == "MoD Press Office"
    assert "Joint NATO training" in (entry.snippet or "")
    assert entry.relevance_status == "relevant"


def test_parse_skips_entries_missing_title_or_link():
    articles = parse_feed_content(SAMPLE_MALFORMED_XML, source_name="General News")
    assert len(articles) == 1
    assert articles[0].title == "Troops mobilize for border security exercise"


def test_parse_deduplicates_identical_canonical_urls_in_batch():
    duplicate_xml = """<?xml version="1.0"?>
    <rss version="2.0">
      <channel>
        <title>Dup Feed</title>
        <item>
          <title>Conflict escalation reported</title>
          <link>https://example.com/story?utm_source=twitter</link>
        </item>
        <item>
          <title>Conflict escalation reported (updated)</title>
          <link>https://example.com/story?utm_source=facebook</link>
        </item>
      </channel>
    </rss>
    """
    articles = parse_feed_content(duplicate_xml, source_name="News")
    # Both URLs normalize to https://example.com/story, so only 1 should be retained
    assert len(articles) == 1
    assert articles[0].canonical_url == "https://example.com/story"
