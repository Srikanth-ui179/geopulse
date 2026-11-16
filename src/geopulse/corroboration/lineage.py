"""Source lineage extraction and classification for news articles."""

import re
from collections.abc import Sequence

from geopulse.corroboration.models import ArticleLineage, SourceOriginType
from geopulse.ingestion.models import ArticleRecord
from geopulse.summarization.validator import canonicalize_publisher_name, extract_registered_domain

# 1. Recognized international news wire services
WIRE_SERVICE_CANONICAL: dict[str, str] = {
    "reuters": "Reuters",
    "associated press": "Associated Press",
    "ap": "Associated Press",
    "afp": "Agence France-Presse",
    "agence france-presse": "Agence France-Presse",
    "bloomberg": "Bloomberg",
    "tass": "TASS",
    "ria novosti": "RIA Novosti",
    "interfax": "Interfax",
    "xinhua": "Xinhua",
    "kyodo": "Kyodo News",
    "kyodo news": "Kyodo News",
    "anadolu": "Anadolu Agency",
    "anadolu agency": "Anadolu Agency",
    "dpa": "Deutsche Presse-Agentur",
    "upi": "United Press International",
    "efe": "Agencia EFE",
}

WIRE_NAME_PATTERN = re.compile(
    r"\b(Reuters|Associated Press|AP|AFP|Agence France-Presse|Bloomberg|TASS|"
    r"RIA Novosti|Interfax|Xinhua|Kyodo News|Kyodo|Anadolu|DPA|UPI)\b",
    re.I,
)

# 2. Wire attribution in datelines, credits, or leading parentheticals
WIRE_SERVICE_NAMES_RE = (
    r"Reuters|Associated Press|AP|AFP|Agence France-Presse|Bloomberg|TASS|"
    r"RIA Novosti|Interfax|Xinhua|Kyodo News|Kyodo|Anadolu|DPA|UPI"
)


def detect_wire_service(text: str) -> tuple[str, str] | None:
    """Detect wire service attribution or datelines in text."""
    # Dateline pattern: (Reuters) - or Sevastopol (Reuters) -
    m = re.search(
        rf"^\s*(?:[A-Za-z\s,]+)?\(({WIRE_SERVICE_NAMES_RE})\)\s*[-—–]",
        text,
        re.I,
    )
    if m:
        matched = m.group(1)
        canon = WIRE_SERVICE_CANONICAL.get(matched.lower(), matched.title())
        return (canon, m.group(0).strip())

    # Citing/according to wire: "According to Reuters", "Citing a report by Reuters", "Per Reuters"
    m = re.search(
        rf"\b(?:according to|citing|quoted by|reported by|as reported by|per|via|credit:?)\b"
        rf".{{0,40}}\b(?:the\s+)?({WIRE_SERVICE_NAMES_RE})\b",
        text,
        re.I,
    )
    if m:
        matched = m.group(1)
        canon = WIRE_SERVICE_CANONICAL.get(matched.lower(), matched.title())
        return (canon, m.group(0).strip())

    # Wire reported/said: "Reuters reported that..."
    m = re.search(
        rf"\b({WIRE_SERVICE_NAMES_RE})\b.{{0,30}}\b(?:reported|said|announced|stated|confirmed)\b",
        text,
        re.I,
    )
    if m:
        matched = m.group(1)
        canon = WIRE_SERVICE_CANONICAL.get(matched.lower(), matched.title())
        return (canon, m.group(0).strip())

    return None


# 3. Official government, ministry, and military announcement patterns
OFFICIAL_ENTITIES_CANONICAL: dict[str, str] = {
    "ministry of defense": "Ministry of Defense",
    "ministry of defence": "Ministry of Defense",
    "defence ministry": "Ministry of Defense",
    "defense ministry": "Ministry of Defense",
    "department of defense": "Department of Defense",
    "defense department": "Department of Defense",
    "pentagon": "Department of Defense",
    "mod": "Ministry of Defense",
    "dod": "Department of Defense",
    "idf": "Israel Defense Forces",
    "israel defense forces": "Israel Defense Forces",
    "white house": "White House",
    "kremlin": "Kremlin",
    "state department": "U.S. State Department",
    "foreign ministry": "Foreign Ministry",
    "ministry of foreign affairs": "Foreign Ministry",
    "general staff": "General Staff",
    "armed forces command": "Armed Forces Command",
}

OFFICIAL_ENTITIES_RE = (
    r"ministry of defen[cs]e|defence ministry|defense ministry|department of defen[cs]e|"
    r"defense department|pentagon|white house|kremlin|state department|foreign ministry|"
    r"ministry of foreign affairs|idf|israel defense forces|general staff|armed forces command|"
    r"military spokesperson|government spokesperson"
)


def detect_official_statement(text: str) -> tuple[str, str] | None:
    """Detect official government/military statement attribution in text."""
    # Official statement or announcement attribution
    m = re.search(
        rf"\b(?:in an? (?:official\s+)?(?:statement|press release|briefing|communique)"
        rf"(?:\s+on\s+[A-Za-z]+)?|"
        rf"according to (?:a\s+)?(?:statement|press release)(?:\s+issued)?\s+by|"
        rf"citing (?:a\s+)?(?:press release|statement)(?:\s+by)?|"
        rf"in a statement\s+(?:issued\s+)?by|"
        rf"according to|citing|quoted by|per)\b.{{0,40}}\b(?:the\s+)?({OFFICIAL_ENTITIES_RE})\b",
        text,
        re.I,
    )
    if m:
        matched = m.group(1)
        canon = OFFICIAL_ENTITIES_CANONICAL.get(matched.strip().lower(), matched.strip().title())
        return (canon, m.group(0).strip())

    # Entity followed by announcement or press release
    m = re.search(
        rf"\b(?:the\s+)?({OFFICIAL_ENTITIES_RE})\b.{{0,40}}\b"
        rf"(?:announced(?:\s+in\s+a\s+(?:statement|press release|briefing))?|"
        rf"said\s+in\s+a\s+(?:statement|press release|briefing)|issued a statement|"
        rf"released a statement|confirmed\s+in\s+a\s+statement)\b",
        text,
        re.I,
    )
    if m:
        matched = m.group(1)
        canon = OFFICIAL_ENTITIES_CANONICAL.get(matched.strip().lower(), matched.strip().title())
        return (canon, m.group(0).strip())

    return None


# 4. Explicit syndication and republishing patterns
SYNDICATION_PATTERN = re.compile(
    r"\b(?:republished with permission|originally published (?:at|by|in)|"
    r"this article (?:first )?appeared (?:at|on|in)|syndicated from)\s+"
    r"(?:the\s+)?([A-Za-z0-9\s\.-]+?)(?:\.|\s+and|\s*$)",
    re.I,
)

# 5. Derivative secondary reporting patterns (citing another commercial news outlet)
DERIVATIVE_CITATION_PATTERN = re.compile(
    r"\b(?:according to|citing|as first reported by|as reported by|quoted by)\s+"
    r"(?:the\s+)?([A-Z][A-Za-z0-9\s\.-]{2,30}?)(?:,\s|\s+reported|\s+said|\s+noted|\s+first|\.|$)",
)

# 6. Unverified / unknown source indicator patterns
UNKNOWN_SOURCE_PATTERNS = re.compile(
    r"\b(?:unconfirmed reports|social media reports|rumors circulating|"
    r"anonymous sources|unverified claims|telegram channels|online chatter|"
    r"activists said|local sources said)\b",
    re.I,
)

# 7. Positive indicators of original first-hand / investigative reporting
POSITIVE_ORIGINAL_REPORTING_PATTERNS = re.compile(
    r"\b(?:"
    r"(?:our|staff|senior|foreign|special|defence|defense|security|investigative|diplomatic|political|military)\s+"
    r"(?:reporters?|correspondents?|writers?|team)|"
    r"by\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}(?:\s*,\s*[A-Za-z\s]+)?|"
    r"our\s+(?:investigation|reporting|analysis|correspondent|reporters?)|"
    r"exclusive(?::|\s+reporting|\s+investigation)?|"
    r"on the ground(?:\s+in)?|field (?:dispatch|report|investigation)|"
    r"interviewed (?:by|with)|spoke to (?:reporters|journalists)|"
    r"told [A-Za-z0-9\s\.-]+ in an interview|"
    r"confirmed to [A-Za-z0-9\s\.-]+|obtained by [A-Za-z0-9\s\.-]+|"
    r"reviewed by [A-Za-z0-9\s\.-]+|"
    r"(?:satellite|commercial|geospatial)\s+(?:imagery|photos|analysis)\s+(?:analyzed|revealed|obtained|shows?)|"
    r"inspected (?:construction|field|logistics|documents|manifests|records|sites)|"
    r"first-hand (?:account|reporting|investigation)|"
    r"eyewitness(?:es)?|witnesses (?:told|reported to)"
    r")\b",
    re.I,
)

AGGREGATOR_DOMAINS: frozenset[str] = frozenset(
    {
        "yahoo.com",
        "msn.com",
        "google.com",
        "feedproxy.google.com",
        "news.google.com",
        "smartnews.com",
    }
)

EYEWITNESS_PATTERNS = re.compile(
    r"\b(eyewitness(?:es)?|witness(?:es)?\s+(?:said|reported|told|described)|"
    r"local\s+residents?\s+(?:said|reported|described)|bystanders?|"
    r"on-the-ground\s+(?:video|footage|observers?|accounts?)|"
    r"civilians?\s+on\s+the\s+scene|geolocated\s+(?:footage|video|imagery))\b",
    re.I,
)


def determine_origin_type(
    lineage_classification: str,
    text: str,
) -> SourceOriginType:
    """Determine transparent source origin type: official, eyewitness, independent, or wire."""
    if EYEWITNESS_PATTERNS.search(text):
        return "eyewitness_account"
    if lineage_classification == "official_statement":
        return "official_statement"
    if lineage_classification in ("wire_service_citation", "syndicated"):
        return "wire_syndication"
    if lineage_classification == "original_reporting":
        return "independent_reporting"
    return "unknown"


def _extract_article_lineage_raw(
    article: ArticleRecord,
    known_publishers: Sequence[str] | None = None,
) -> ArticleLineage:
    """Analyze an article's provenance, attribution, and reporting lineage."""
    pub_raw = (article.source_name or "Unknown Publisher").strip()
    pub_canon = canonicalize_publisher_name(pub_raw)
    url = article.url or ""
    canonical_url = article.canonical_url or url
    domain = extract_registered_domain(url)
    canon_domain = extract_registered_domain(canonical_url)

    text_parts = [
        article.title or "",
        article.snippet or "",
        getattr(article, "body_text", "") or "",
    ]
    full_text = " ".join(p for p in text_parts if p).strip()

    # Case 0: Empty or extremely sparse text -> UNKNOWN
    if len(full_text) < 40 and not article.title:
        return ArticleLineage(
            article_url=canonical_url,
            publisher=pub_raw,
            original_source="Unknown",
            lineage_classification="unknown",
            evidence_text="Insufficient text content to determine reporting source.",
        )

    # Case 1: Publisher itself is a wire service
    if pub_canon in WIRE_SERVICE_CANONICAL:
        wire_name = WIRE_SERVICE_CANONICAL[pub_canon]
        return ArticleLineage(
            article_url=canonical_url,
            publisher=pub_raw,
            original_source=wire_name,
            lineage_classification="original_reporting",
            citation_source=None,
            evidence_text=f"Direct wire reporting published under {pub_raw} masthead.",
        )

    # Case 2: Syndication via canonical domain divergence or explicit syndication notice
    if canon_domain and domain and canon_domain != domain:
        if domain in AGGREGATOR_DOMAINS or "news" in domain:
            origin_pub = canon_domain.split(".")[0].title()
            return ArticleLineage(
                article_url=canonical_url,
                publisher=pub_raw,
                original_source=origin_pub,
                lineage_classification="syndicated",
                citation_source=origin_pub,
                evidence_text=f"Aggregated or syndicated from canonical domain: {canon_domain}",
                canonical_origin_url=canonical_url,
            )

    synd_match = SYNDICATION_PATTERN.search(full_text)
    if synd_match:
        origin_source = synd_match.group(1).strip()
        return ArticleLineage(
            article_url=canonical_url,
            publisher=pub_raw,
            original_source=origin_source,
            lineage_classification="syndicated",
            citation_source=origin_source,
            evidence_text=synd_match.group(0).strip(),
            canonical_origin_url=canonical_url,
        )

    # Case 3: Wire service citation / dateline
    wire_detected = detect_wire_service(full_text)
    if wire_detected:
        canon_wire, wire_ev = wire_detected
        return ArticleLineage(
            article_url=canonical_url,
            publisher=pub_raw,
            original_source=canon_wire,
            lineage_classification="wire_service_citation",
            citation_source=canon_wire,
            evidence_text=wire_ev,
        )

    # Case 4: Official statement / press release dependency
    official_detected = detect_official_statement(full_text)
    if official_detected:
        canon_entity, official_ev = official_detected
        return ArticleLineage(
            article_url=canonical_url,
            publisher=pub_raw,
            original_source=canon_entity,
            lineage_classification="official_statement",
            citation_source=canon_entity,
            evidence_text=official_ev,
        )

    # Case 5: Derivative reporting citing another commercial news outlet
    if known_publishers:
        for kp in known_publishers:
            kp_canon = canonicalize_publisher_name(kp)
            if kp_canon and kp_canon != pub_canon and len(kp_canon) >= 4:
                ref_pat = (
                    rf"\b(?:according to|citing|quoted by|reported by|per)\s+"
                    rf"(?:the\s+)?{re.escape(kp)}\b"
                )
                if re.search(ref_pat, full_text, re.I):
                    return ArticleLineage(
                        article_url=canonical_url,
                        publisher=pub_raw,
                        original_source=kp,
                        lineage_classification="derivative",
                        citation_source=kp,
                        evidence_text=f"Explicitly cites {kp}",
                    )

    deriv_match = DERIVATIVE_CITATION_PATTERN.search(full_text)
    if deriv_match:
        cited_candidate = deriv_match.group(1).strip()
        canon_candidate = canonicalize_publisher_name(cited_candidate)
        # Avoid treating common nouns or generic words as publishers
        if (
            len(cited_candidate) >= 4
            and canon_candidate != pub_canon
            and not canon_candidate.startswith("ministry")
            and not canon_candidate.startswith("official")
            and not canon_candidate.startswith("statement")
        ):
            return ArticleLineage(
                article_url=canonical_url,
                publisher=pub_raw,
                original_source=cited_candidate,
                lineage_classification="derivative",
                citation_source=cited_candidate,
                evidence_text=deriv_match.group(0).strip(),
            )

    # Case 6: Purely unverified / unknown attribution
    unknown_match = UNKNOWN_SOURCE_PATTERNS.search(full_text)
    if unknown_match and (
        "unconfirmed" in full_text.lower() or "social media" in full_text.lower()
    ):
        return ArticleLineage(
            article_url=canonical_url,
            publisher=pub_raw,
            original_source="Unknown",
            lineage_classification="unknown",
            evidence_text=unknown_match.group(0).strip(),
        )

    # Case 7: Positive evidence of original reporting
    # Requires positive evidence (designated author byline, staff correspondent,
    # investigative indicators, field reporting, or primary wire masthead).
    has_valid_author = bool(
        article.author
        and len(article.author.strip()) >= 3
        and article.author.strip().lower()
        not in {"admin", "editor", "editorial", "unknown", "staff", "reuters", "ap"}
    )
    orig_match = POSITIVE_ORIGINAL_REPORTING_PATTERNS.search(full_text)

    if has_valid_author or orig_match:
        matched_span = (
            orig_match.group(0).strip()
            if orig_match
            else f"Bylined author: {article.author.strip()}"
        )
        return ArticleLineage(
            article_url=canonical_url,
            publisher=pub_raw,
            original_source=pub_raw,
            lineage_classification="original_reporting",
            evidence_text=f"Positive evidence of original reporting ({matched_span}).",
        )

    # Case 8: Absence of secondary attribution without positive evidence of original reporting
    # Conservative rule: absence of attribution does NOT establish original reporting.
    return ArticleLineage(
        article_url=canonical_url,
        publisher=pub_raw,
        original_source="Unknown",
        lineage_classification="unknown",
        evidence_text="Absence of attribution without positive evidence of original reporting.",
    )


def extract_article_lineage(
    article: ArticleRecord,
    known_publishers: Sequence[str] | None = None,
) -> ArticleLineage:
    """Analyze an article's provenance, attribution, and reporting lineage with origin type."""
    raw = _extract_article_lineage_raw(article, known_publishers=known_publishers)
    text_parts = [
        article.title or "",
        article.snippet or "",
        getattr(article, "body_text", "") or "",
    ]
    full_text = " ".join(p for p in text_parts if p).strip()
    origin_type = determine_origin_type(raw.lineage_classification, full_text)
    return raw.model_copy(update={"origin_type": origin_type})


def build_cluster_lineages(articles: Sequence[ArticleRecord]) -> dict[str, ArticleLineage]:
    """Build a mapping of canonical URL to ArticleLineage for all articles in a cluster."""
    known_publishers = [a.source_name for a in articles if a.source_name]
    lineages: dict[str, ArticleLineage] = {}
    for art in articles:
        lineage = extract_article_lineage(art, known_publishers=known_publishers)
        url_key = art.canonical_url or art.url
        lineages[url_key] = lineage
        if art.url and art.url != url_key:
            lineages[art.url] = lineage
    return lineages
