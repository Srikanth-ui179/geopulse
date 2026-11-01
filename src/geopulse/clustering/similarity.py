"""Lightweight similarity and entity matching for geopolitical event clustering."""

import re
from collections.abc import Sequence
from datetime import timedelta

from geopulse.classification.models import TopicCategory
from geopulse.ingestion.models import ArticleRecord

# Standard English and news-filler stopwords to ignore during similarity scoring
STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "along",
        "also",
        "am",
        "an",
        "and",
        "another",
        "any",
        "are",
        "around",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "day",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "first",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "latest",
        "me",
        "more",
        "most",
        "my",
        "myself",
        "new",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "one",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "reports",
        "s",
        "said",
        "same",
        "says",
        "second",
        "she",
        "should",
        "so",
        "some",
        "such",
        "t",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "today",
        "too",
        "two",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "week",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "year",
        "yesterday",
    }
)

# Geographic entities mapped to parent regions for entity overlap and conflict detection
LOCATION_HIERARCHY: dict[str, str] = {
    # Eastern Europe
    "kyiv": "ukraine",
    "kharkiv": "ukraine",
    "odesa": "ukraine",
    "zaporizhzhia": "ukraine",
    "kherson": "ukraine",
    "donetsk": "ukraine",
    "luhansk": "ukraine",
    "crimea": "ukraine",
    "ukraine": "ukraine",
    "moscow": "russia",
    "belgorod": "russia",
    "kursk": "russia",
    "rostov": "russia",
    "voronezh": "russia",
    "russia": "russia",
    "black sea": "eastern_europe",
    # Middle East
    "gaza": "middle_east",
    "rafah": "middle_east",
    "khan younis": "middle_east",
    "west bank": "middle_east",
    "jerusalem": "middle_east",
    "tel aviv": "middle_east",
    "israel": "middle_east",
    "beirut": "lebanon",
    "lebanon": "lebanon",
    "damascus": "syria",
    "syria": "syria",
    "tehran": "iran",
    "iran": "iran",
    "yemen": "yemen",
    "sanaa": "yemen",
    "houthi": "yemen",
    "red sea": "middle_east",
    "strait of hormuz": "middle_east",
    "baghdad": "iraq",
    "iraq": "iraq",
    # Indo-Pacific
    "taiwan": "indo_pacific",
    "taipei": "indo_pacific",
    "beijing": "china",
    "china": "china",
    "south china sea": "indo_pacific",
    "pyongyang": "korea",
    "north korea": "korea",
    "seoul": "korea",
    "south korea": "korea",
    "tokyo": "japan",
    "japan": "japan",
    "manila": "philippines",
    "philippines": "indo_pacific",
    # Africa
    "khartoum": "sudan",
    "darfur": "sudan",
    "sudan": "sudan",
    "somalia": "somalia",
    "mogadishu": "somalia",
    # South Asia
    "kashmir": "south_asia",
    "pakistan": "south_asia",
    "islamabad": "south_asia",
    # Western & Northern Europe / NATO
    "copenhagen": "denmark",
    "denmark": "denmark",
    "stockholm": "sweden",
    "sweden": "sweden",
    "oslo": "norway",
    "norway": "norway",
    "helsinki": "finland",
    "finland": "finland",
    "warsaw": "poland",
    "poland": "poland",
    "berlin": "germany",
    "germany": "germany",
    "paris": "france",
    "france": "france",
    "london": "uk",
    "uk": "uk",
    "baltic": "baltic",
    "estonia": "estonia",
    "latvia": "latvia",
    "lithuania": "lithuania",
    "washington": "us",
    "us": "us",
    "united states": "us",
}

# Normalized actor and term aliases to resolve journalistic naming variations
ACTOR_ALIASES: dict[str, str] = {
    "british": "uk",
    "britain": "uk",
    "american": "us",
    "americans": "us",
    "russian": "russia",
    "russians": "russia",
    "ukrainian": "ukraine",
    "ukrainians": "ukraine",
    "israeli": "israel",
    "israelis": "israel",
    "palestinian": "palestine",
    "palestinians": "palestine",
    "lebanese": "lebanon",
    "syrian": "syria",
    "iranian": "iran",
    "iranians": "iran",
    "houthis": "houthi",
    "copenhagen": "denmark",
    "stockholm": "sweden",
    "oslo": "norway",
    "helsinki": "finland",
    "warsaw": "poland",
    "berlin": "germany",
    "paris": "france",
    "london": "uk",
    "washington": "us",
    "kyiv": "ukraine",
    "moscow": "russia",
    "beijing": "china",
    "tokyo": "japan",
    "seoul": "korea",
    "pyongyang": "korea",
    "tehran": "iran",
    "danish": "denmark",
    "danes": "denmark",
    "swedish": "sweden",
    "norwegian": "norway",
    "finnish": "finland",
    "polish": "poland",
    "german": "germany",
    "french": "france",
    "airstrikes": "strike",
    "airstrike": "strike",
    "strikes": "strike",
    "missiles": "missile",
    "drones": "drone",
    "orders": "procure",
    "order": "procure",
    "ordering": "procure",
    "buy": "procure",
    "buys": "procure",
    "buying": "procure",
    "purchase": "procure",
    "purchases": "procure",
    "purchased": "procure",
    "purchasing": "procure",
    "acquire": "procure",
    "acquires": "procure",
    "acquiring": "procure",
    "acquisition": "procure",
    "procures": "procure",
    "procuring": "procure",
    "procurement": "procure",
    "commits": "commit",
    "committed": "commit",
    "capabilities": "capability",
    "f-35s": "f-35",
    "airborne": "air",
}

# Words signaling contested claims or disputes (explicit denials, conflicting accounts)
DISPUTE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(denies|denied|denying|rebuts|rebutted|rejects|rejected|refutes|refuted)\b",
        re.I,
    ),
    re.compile(
        r"\b(conflicting claims|contradictory accounts|claims without evidence|"
        r"mutually exclusive claims|opposing claims)\b",
        re.I,
    ),
    re.compile(
        r"\b(disputed claim|disputed claims|disputed account|disputed accounts|"
        r"disputed by|disputed the claim|disputed the account|dispute between)\b",
        re.I,
    ),
)

_GEOGRAPHICAL_DISPUTE_PATTERN: re.Pattern[str] = re.compile(
    r"\bdisputed\s+(?:[\w-]+\s+){0,3}"
    r"(?:waters|seas?|borders?|territor\w*|islands?|regions?|zones?|areas?|boundaries|land|corridors?|airspace)\b",
    re.I,
)

SHORT_GEOPOLITICAL_ENTITIES: frozenset[str] = frozenset({"us", "uk", "un", "eu"})

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)?")


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase words, filtering stopwords and normalizing aliases."""
    if not text:
        return []
    words = _TOKEN_RE.findall(text.lower())
    tokens: list[str] = []
    for w in words:
        # Split non-numeric hyphenated compound terms (e.g. 'uk-france', 'nuclear-strike')
        # while preserving military alphanumerics (e.g. 'f-35')
        if "-" in w and not re.search(r"\d", w):
            parts = w.split("-")
            for part in parts:
                if (len(part) > 2 or part in SHORT_GEOPOLITICAL_ENTITIES) and part not in STOPWORDS:
                    tokens.append(ACTOR_ALIASES.get(part, part))
        else:
            if (len(w) > 2 or w in SHORT_GEOPOLITICAL_ENTITIES) and w not in STOPWORDS:
                tokens.append(ACTOR_ALIASES.get(w, w))
    return tokens


def extract_bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    """Extract adjacent and short-window token pairs for phrase-level matching."""
    if len(tokens) < 2:
        return set()
    pairs: set[tuple[str, str]] = set()
    for i in range(len(tokens) - 1):
        # Order-independent adjacent pair
        pairs.add((min(tokens[i], tokens[i + 1]), max(tokens[i], tokens[i + 1])))
        # Skip-pair within window of 2
        if i + 2 < len(tokens):
            pairs.add((min(tokens[i], tokens[i + 2]), max(tokens[i], tokens[i + 2])))
    return pairs


# Distinctive policy, strategy, and defence industrial initiative patterns
_POLICY_IDENTIFIER_PATTERN: re.Pattern[str] = re.compile(
    r"\b((?:[A-Za-z]+(?:\s+[A-Za-z]+){0,2}\s+)?"
    r"(?:offset\s+policy|offset\s+guidance|defence\s+offsets?|defense\s+offsets?|"
    r"procurement\s+offsets?|offset\s+rules?|industrial\s+strategy|"
    r"defence\s+industrial\s+strategy|strategic\s+defence\s+review|"
    r"national\s+defence\s+strategy|national\s+security\s+strategy|"
    r"defence\s+procurement\s+policy))\b|"
    r"\b(british\s+jobs\s+and\s+skills)\b",
    re.I,
)


def extract_policy_identifiers(text: str) -> set[str]:
    """Extract recognized defence policy and procurement initiative identifiers."""
    if not text:
        return set()
    found: set[str] = set()
    for m in _POLICY_IDENTIFIER_PATTERN.finditer(text):
        raw = m.group(0).lower().strip()
        if "offset" in raw or "british jobs and skills" in raw:
            found.add("offset_policy")
        elif "defence industrial strategy" in raw or "industrial strategy" in raw:
            found.add("defence_industrial_strategy")
        elif "strategic defence review" in raw:
            found.add("strategic_defence_review")
        elif "national defence strategy" in raw:
            found.add("national_defence_strategy")
        else:
            found.add(raw)
    return found


def extract_locations(text: str) -> set[str]:
    """Extract recognized geographic locations from text."""
    if not text:
        return set()
    lower_text = text.lower()
    found: set[str] = set()
    for loc in LOCATION_HIERARCHY:
        # Use boundary check for accurate location extraction
        pattern = rf"\b{re.escape(loc)}\b"
        if re.search(pattern, lower_text):
            found.add(loc)
    return found


def has_dispute_indicators(text: str) -> bool:
    """Check if the text contains explicit dispute or contested claim keywords."""
    if not text:
        return False
    # Strip geographical descriptors like 'disputed waters' or 'disputed Asian seas'
    # so sovereignty disputes over geographic areas are not conflated with factual event disputes.
    clean_text = _GEOGRAPHICAL_DISPUTE_PATTERN.sub("", text)
    return any(pat.search(clean_text) for pat in DISPUTE_PATTERNS)


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute standard Jaccard set similarity."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def check_location_conflict(locs_a: set[str], locs_b: set[str]) -> bool:
    """Determine if two location sets have mutually exclusive specific locations.

    If both articles mention distinct specific cities/regions that belong to different
    parent regions or distinct subregions without overlap, they conflict.
    """
    if not locs_a or not locs_b:
        # No location data to conflict
        return False

    # Check if there is any exact location match
    if locs_a & locs_b:
        return False

    # Check parent region overlaps
    parents_a = {LOCATION_HIERARCHY[loc] for loc in locs_a}
    parents_b = {LOCATION_HIERARCHY[loc] for loc in locs_b}

    # If parents have zero overlap, it's a conflict
    if not (parents_a & parents_b):
        return True

    # If within the same parent country (e.g. Russia: Belgorod vs Voronezh),
    # check if both mention distinct specific cities (not just country name)
    specific_a = {loc for loc in locs_a if loc != LOCATION_HIERARCHY[loc]}
    specific_b = {loc for loc in locs_b if loc != LOCATION_HIERARCHY[loc]}
    if specific_a and specific_b and not (specific_a & specific_b):
        return True

    return False


def compute_article_similarity(
    article_a: ArticleRecord,
    article_b: ArticleRecord,
    max_time_delta: timedelta = timedelta(hours=48),
    topic_a: TopicCategory | None = None,
    topic_b: TopicCategory | None = None,
) -> float:
    """Compute pairwise similarity between two articles for event clustering.

    Returns a score between 0.0 (completely distinct) and 1.0 (exact same event).
    """
    # 1. Temporal gating: articles occurring > max_time_delta apart cannot be the same event
    time_a = article_a.published_at or article_a.fetched_at
    time_b = article_b.published_at or article_b.fetched_at
    time_diff = abs(time_a - time_b)
    if time_diff > max_time_delta:
        return 0.0

    # 2. Topic compatibility: wildly incompatible topics cannot be the same event
    if topic_a and topic_b:
        incompatible_pairs = {
            (TopicCategory.WEAPONS_PROCUREMENT, TopicCategory.ARMED_CONFLICT),
            (TopicCategory.DEFENCE_POLICY, TopicCategory.ARMED_CONFLICT),
            (TopicCategory.SANCTIONS, TopicCategory.TERRORISM_INSURGENCY),
        }
        if (topic_a, topic_b) in incompatible_pairs or (topic_b, topic_a) in incompatible_pairs:
            return 0.0

    # 3. Location extraction and conflict detection
    text_a = f"{article_a.title} {article_a.snippet or ''}"
    text_b = f"{article_b.title} {article_b.snippet or ''}"
    locs_a = extract_locations(text_a)
    locs_b = extract_locations(text_b)

    if check_location_conflict(locs_a, locs_b):
        return 0.0

    # 4. Token and Bigram Jaccard similarity of titles
    tokens_title_a = tokenize(article_a.title)
    tokens_title_b = tokenize(article_b.title)

    set_title_a = set(tokens_title_a)
    set_title_b = set(tokens_title_b)
    title_jaccard = jaccard_similarity(set_title_a, set_title_b)

    bigrams_a = extract_bigrams(tokens_title_a)
    bigrams_b = extract_bigrams(tokens_title_b)
    bigram_jaccard = jaccard_similarity(bigrams_a, bigrams_b)

    # 5. Full text tokens and cross-containment (utilizing snippets when available)
    tokens_full_a = set(tokens_title_a + tokenize(article_a.snippet or ""))
    tokens_full_b = set(tokens_title_b + tokenize(article_b.snippet or ""))
    full_text_jaccard = jaccard_similarity(tokens_full_a, tokens_full_b)

    cross_a = len(set_title_a & tokens_full_b) / len(set_title_a) if set_title_a else 0.0
    cross_b = len(set_title_b & tokens_full_a) / len(set_title_b) if set_title_b else 0.0
    cross_containment = (cross_a + cross_b) / 2.0

    # 6. Location and topic alignment bonus
    location_bonus = 0.0
    if locs_a and locs_b and (locs_a & locs_b):
        matching_locs = locs_a & locs_b
        # Enhanced bonus when multiple specific partner nations align (e.g. bilateral exercise)
        location_bonus = 0.15 + (0.05 if len(matching_locs) >= 2 else 0.0)

    topic_bonus = 0.0
    if topic_a and topic_b and topic_a == topic_b and topic_a != TopicCategory.UNCERTAIN:
        topic_bonus = 0.05

    # 7. Policy and procurement initiative co-reference bonus
    policy_bonus = 0.0
    policies_a = extract_policy_identifiers(text_a)
    policies_b = extract_policy_identifiers(text_b)
    if policies_a and policies_b and (policies_a & policies_b):
        policy_topics = {
            TopicCategory.DEFENCE_POLICY,
            TopicCategory.WEAPONS_PROCUREMENT,
            TopicCategory.UNCERTAIN,
            None,
        }
        if topic_a in policy_topics and topic_b in policy_topics:
            has_shared_entity = bool(locs_a & locs_b) or (
                article_a.source_name.strip().lower() == article_b.source_name.strip().lower()
            )
            if has_shared_entity:
                policy_bonus = 0.25

    # 8. Composite score combining title, bigram, full-text, cross-containment, and location
    composite = (
        (title_jaccard * 0.30)
        + (bigram_jaccard * 0.15)
        + (full_text_jaccard * 0.25)
        + (cross_containment * 0.20)
        + location_bonus
        + topic_bonus
        + policy_bonus
    )

    # Slight penalty for greater time separation within the 48-hour window
    time_decay = 1.0 - (time_diff.total_seconds() / max_time_delta.total_seconds()) * 0.15

    return min(1.0, composite * time_decay)


_ATTRIBUTION_PATTERN = re.compile(
    r"\b(?:citing|according to|quoted by|quoting|reported by|reporting by|reported in|via)\s+"
    r"(?:[\w-]+\s+){0,4}"
    r"(tass|ria novosti|interfax|sana|state media|state news agency|kremlin|"
    r"reuters|associated press|\bap\b|afp|agence france-presse|bloomberg|dpa|upi|"
    r"pentagon|defence department|defense department|white house|"
    r"press release|ministry statement|official statement|joint statement|"
    r"spokesperson|spokesman)\b",
    re.I,
)

_CLAIM_ORIGIN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "rebel_claim",
        re.compile(
            r"\b(?:rebels?|insurgents?|militants?|separatists?)\s+(?:claim|claimed|claims|say|said|announce|announced)\b",
            re.I,
        ),
    ),
    (
        "military_spokesperson",
        re.compile(
            r"\b(?:military\s+spokesm[ae]n|defence\s+spokesm[ae]n|army\s+spokesm[ae]n|pentagon\s+spokesm[ae]n|"
            r"spokesperson\s+said|spokesman\s+said|spokesman\s+stated|spokesperson\s+stated)\b",
            re.I,
        ),
    ),
    (
        "ministry_announcement",
        re.compile(
            r"\b(?:ministry\s+of\s+defen[cs]e|defence\s+ministry|defense\s+ministry|mod|dod)\s+"
            r"(?:announced|said|stated|confirmed|statement|claims?|claimed)\b",
            re.I,
        ),
    ),
    (
        "government_announcement",
        re.compile(
            r"\b(?:pentagon|defence\s+department|defense\s+department|white\s+house|"
            r"state\s+department|foreign\s+ministry)\s+"
            r"(?:announced|selects?|selected|awards?|awarded|confirms?|confirmed|stated|said)\b",
            re.I,
        ),
    ),
    (
        "company_claim",
        re.compile(
            r"\b(?:the\s+company\s+(?:claims?|announced|stated|said)|manufacturer\s+(?:claims?|stated))\b",
            re.I,
        ),
    ),
)


def has_single_origin_dependency(
    articles: Sequence[ArticleRecord],
) -> tuple[bool, str | None]:
    """Check if multiple articles derive from the same original report or single source.

    Returns:
        tuple[bool, str | None]: (is_single_origin, origin_name)
    """
    if len(articles) <= 1:
        return False, None

    # 1. Check if any article explicitly cites another source present in the cluster
    source_names = {a.source_name.strip().lower() for a in articles}
    for art in articles:
        text = f"{art.title} {art.snippet or ''}".lower()
        for other_source in source_names:
            if other_source != art.source_name.strip().lower() and len(other_source) > 3:
                attrib_patterns = [
                    f"according to {other_source}",
                    f"citing {other_source}",
                    f"quoted by {other_source}",
                    f"{other_source} reported",
                    f"{other_source} announced",
                    f"{other_source} statement",
                    f"{other_source} said",
                    f"{other_source} confirmed",
                    f"released by {other_source}",
                    f"press release from {other_source}",
                    f"statement from {other_source}",
                    f"statement by {other_source}",
                ]
                if any(pat in text for pat in attrib_patterns):
                    return True, other_source

    # 2. Check if all articles in the cluster repeat the exact same combatant or official claim
    all_texts = [f"{a.title} {a.snippet or ''}" for a in articles]
    for origin_tag, pat in _CLAIM_ORIGIN_PATTERNS:
        if all(pat.search(t) for t in all_texts):
            return True, origin_tag

    # 3. Check if all articles cite the exact same originating wire or agency
    found_entities: list[str] = []
    found_entities: list[str] = []
    for t in all_texts:
        m = _ATTRIBUTION_PATTERN.search(t)
        if m:
            found_entities.append(m.group(1).lower())

    if len(found_entities) == len(all_texts):
        first_ent = found_entities[0]
        state_wire_synonyms = {
            "state media",
            "state news agency",
            "tass",
            "ria novosti",
            "interfax",
        }
        western_wire_synonyms = {
            "reuters",
            "associated press",
            "ap",
            "afp",
            "agence france-presse",
            "bloomberg",
        }
        official_announcements = {
            "press release",
            "ministry statement",
            "official statement",
            "joint statement",
            "spokesperson",
        }

        if all(
            ent == first_ent
            or (ent in state_wire_synonyms and first_ent in state_wire_synonyms)
            or (ent in western_wire_synonyms and first_ent in western_wire_synonyms)
            or (ent in official_announcements and first_ent in official_announcements)
            for ent in found_entities
        ):
            return True, first_ent

    return False, None
