"""Unit tests for feedparser-based RSS/Atom content parsing."""

from datetime import UTC, datetime

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


def test_parse_prefers_longer_feed_supplied_article_body_for_grounding():
    """Use RSS content:encoded/Atom content when it carries more article evidence."""
    xml = """<?xml version="1.0"?>
    <rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/">
      <channel><title>Defense News</title>
        <item>
          <title>AUKUS exercise tests autonomous systems</title>
          <link>https://example.com/aukus</link>
          <description>Partners tested uncrewed systems.</description>
          <content:encoded><![CDATA[<p>Partners tested more than 40 uncrewed technologies,
          including an anti-submarine warfare component.</p>]]></content:encoded>
        </item>
      </channel>
    </rss>"""
    articles = parse_feed_content(xml, source_name="Defense News")
    assert len(articles) == 1
    assert "more than 40 uncrewed technologies" in (articles[0].snippet or "")
    assert "anti-submarine warfare" in (articles[0].snippet or "")


def test_parse_feed_content_fields_ignore_trailing_boilerplate_and_deduplicate():
    """Content and summary duplicates plus feed chrome resolve to substantive text."""
    atom = """<?xml version="1.0" encoding="utf-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <title>Defense News</title>
      <entry>
        <title>AUKUS exercise tests maritime systems</title>
        <link rel="alternate" href="https://example.com/aukus-fields"/>
        <summary>AUKUS partners tested maritime systems during the exercise.</summary>
        <content type="html"><![CDATA[<p>AUKUS partners tested maritime systems during
          the exercise.</p>
          <p>Subscribe to our newsletter for updates. Related stories: unrelated background.</p>]]>
        </content>
      </entry>
    </feed>"""
    fixed_time = datetime(2026, 10, 29, 8, 0, tzinfo=UTC)
    first = parse_feed_content(atom, "Defense News", fetched_at=fixed_time)[0]
    second = parse_feed_content(atom, "Defense News", fetched_at=fixed_time)[0]
    assert first.snippet == "AUKUS partners tested maritime systems during the exercise."
    assert first.snippet == second.snippet
    assert first.content_hash == second.content_hash


def test_parse_ignores_long_unrelated_content_when_excerpt_matches_headline():
    """Longer but unrelated content metadata does not replace a relevant excerpt."""
    xml = """<?xml version="1.0"?>
    <rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/">
      <channel><title>Defense News</title><item>
        <title>Ethiopia and Tigray leaders discuss renewed tensions</title>
        <link>https://example.com/tigray</link>
        <description>Ethiopian federal authorities and Tigrayan leaders discussed
          renewed tensions.</description>
        <content:encoded><![CDATA[<p>A long unrelated profile covers
        naval procurement, aircraft maintenance, shipyards, budget policy, and technology
        programs.</p>]]></content:encoded>
      </item></channel>
    </rss>"""
    parsed = parse_feed_content(xml, "Defense News")[0]
    assert parsed.snippet == (
        "Ethiopian federal authorities and Tigrayan leaders discussed renewed tensions."
    )


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
