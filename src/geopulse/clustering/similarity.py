"""Lightweight similarity and entity matching for geopolitical event clustering."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from datetime import timedelta

from geopulse.classification.models import TopicCategory
from geopulse.clustering.models import EventCluster
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
    for raw in words:
        # Strip trailing possessive 's (e.g. "iwi's" -> "iwi", "russia's" -> "russia")
        w = re.sub(r"'s$", "", raw)
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


# Specific defence corporate entities, military systems, and programs
DEFENSE_ENTITIES_AND_SYSTEMS: dict[str, str] = {
    "israel weapons industry": "iwi",
    "iwi": "iwi",
    "naval group": "naval_group",
    "mbda": "mbda",
    "saildrone": "saildrone",
    "rheinmetall": "rheinmetall",
    "lockheed martin": "lockheed",
    "lockheed": "lockheed",
    "rtx": "raytheon",
    "raytheon": "raytheon",
    "bae systems": "bae",
    "bae": "bae",
    "general dynamics": "general_dynamics",
    "northrop grumman": "northrop",
    "northrop": "northrop",
    "kongsberg": "kongsberg",
    "elbit systems": "elbit",
    "elbit": "elbit",
    "rafael": "rafael",
    "hanwha": "hanwha",
    "anduril": "anduril",
    "thales": "thales",
    "saab": "saab",
    "arbel": "arbel_system",
    "scx 26": "scx_26",
    "scx-26": "scx_26",
    "counter-drone": "tech:counter_drone",
    "counter drone": "tech:counter_drone",
    "c-uas": "tech:counter_drone",
    "cuas": "tech:counter_drone",
    "uuv": "tech:uuv",
    "unmanned underwater vessel": "tech:uuv",
    "autonomous underwater vehicle": "tech:uuv",
    "drone sub": "tech:uuv",
    "usv": "tech:usv",
    "unmanned surface vessel": "tech:usv",
}


def extract_defense_entities_and_platforms(text: str) -> set[str]:
    """Extract recognized defense companies, platforms, and technological systems."""
    if not text:
        return set()
    lower = text.lower()
    found = set()
    for phrase, canonical in DEFENSE_ENTITIES_AND_SYSTEMS.items():
        if re.search(rf"\b{re.escape(phrase)}\b", lower):
            found.add(canonical)
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
            (TopicCategory.INTELLIGENCE_ESPIONAGE, TopicCategory.ARMED_CONFLICT),
            (TopicCategory.MILITARY_TECHNOLOGY, TopicCategory.ARMED_CONFLICT),
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
    if topic_a and topic_b:
        if topic_a == topic_b and topic_a != TopicCategory.UNCERTAIN:
            topic_bonus = 0.05
        elif {topic_a, topic_b} == {
            TopicCategory.DEFENCE_POLICY,
            TopicCategory.WEAPONS_PROCUREMENT,
        }:
            topic_bonus = 0.03

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

    # 8. Defense entity and system co-reference bonus
    entity_bonus = 0.0
    entities_a = extract_defense_entities_and_platforms(text_a)
    entities_b = extract_defense_entities_and_platforms(text_b)
    shared_entities = entities_a & entities_b
    if shared_entities:
        has_named_system = any(e in {"arbel_system", "scx_26"} for e in shared_entities)
        if has_named_system and len(shared_entities) >= 2:
            entity_bonus = 0.25
        elif has_named_system or len(shared_entities) >= 2:
            entity_bonus = 0.20
        else:
            entity_bonus = 0.10

    # 9. Composite score combining title, bigram, full-text, cross-containment, and location
    composite = (
        (title_jaccard * 0.30)
        + (bigram_jaccard * 0.15)
        + (full_text_jaccard * 0.25)
        + (cross_containment * 0.20)
        + location_bonus
        + topic_bonus
        + policy_bonus
        + entity_bonus
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


# ==============================================================================
# Cross-Day Event Deduplication & Event Identity Infrastructure
# ==============================================================================

_CORE_FACILITY_PATTERNS = re.compile(
    r"\b((?:RAF|Fort|Camp)\s+[A-Z][a-z]+|(?:Air Base|Naval Base)\s+[A-Z][a-z]+)\b",
)

_MEANINGFUL_DEVELOPMENT_PATTERNS = re.compile(
    r"\b(?:"
    r"charged|indicted|court\s+appearance|appeared\s+in\s+court|arraigned|"
    r"pleaded\s+guilty|sentenced|convicted|bail\s+denied|extradited|"
    r"death\s+toll\s+(?:rose|rises|climbed|reaches|reached)|"
    r"casualties\s+(?:rose|increased|confirmed)|bodies\s+recovered|"
    r"signed\s+(?:a\s+)?(?:ceasefire|treaty|accord|deal)|"
    r"ceasefire\s+(?:takes\s+effect|took\s+effect|enters?\s+into\s+force)|"
    r"hostages?\s+released|prisoner\s+exchange\s+completed|"
    r"sanctions\s+imposed|sanctions\s+enacted|"
    r"(?<!\bno\s)new\s+(?:charges|evidence|arrests|suspects)"
    r")\b",
    re.I,
)


def extract_event_core_facility(text: str) -> str | None:
    """Extract explicit military base, air station, or specific facility from text."""
    m = _CORE_FACILITY_PATTERNS.search(text)
    if m:
        return m.group(1).strip()
    return None


def compute_event_hash(event: EventCluster) -> str:
    """Compute deterministic content hash for an event cluster based on title and source URLs."""
    urls = sorted(a.canonical_url for a in getattr(event, "articles", ()))
    key = f"{event.title.strip().lower()}|{'|'.join(urls)}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def compute_event_semantic_signature(event: EventCluster) -> str:
    """Compute semantic event signature for cross-day deduplication and persistent identity."""
    combined_text = f"{event.title} " + " ".join(
        f"{a.title} {a.snippet or ''}" for a in getattr(event, "articles", ())
    )
    facility = extract_event_core_facility(combined_text)
    fac_slug = re.sub(r"\W+", "_", facility.lower()).strip("_") if facility else ""

    suspect_m = re.search(
        r"\b(?:Sergeant|Sgt\.?|Soldier|Captain|Major|Colonel|Analyst|Contractor|Officer)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b",
        combined_text,
    )
    suspect_slug = re.sub(r"\W+", "_", suspect_m.group(1).lower()).strip("_") if suspect_m else ""

    theme = "general"
    if re.search(
        r"\b(sabotage|plot|activist|break-in|protest|terror|terrorist|security\s+threat)\b",
        combined_text,
        re.I,
    ):
        theme = "sabotage_plot"
    elif re.search(
        r"\b(espionage|treason|selling\s+secrets|classified\s+documents?)\b", combined_text, re.I
    ):
        theme = "espionage"
    elif re.search(r"\b(airstrike|drone\s+strike|missile\s+strike|bombing)\b", combined_text, re.I):
        theme = "strike"
    elif re.search(r"\b(procurement|contract|arms\s+sale)\b", combined_text, re.I):
        theme = "procurement"
    elif re.search(r"\b(ceasefire|peace\s+talks?|treaty)\b", combined_text, re.I):
        theme = "diplomacy"

    cat_slug = getattr(event.category, "value", str(event.category)).lower()
    return f"sig:{cat_slug}:{fac_slug}:{suspect_slug}:{theme}"


def is_same_underlying_event(
    new_event: EventCluster,
    prev_event_or_sig: EventCluster | str,
) -> bool:
    """Determine whether new_event describes the same underlying real-world event across days."""
    if not isinstance(prev_event_or_sig, (str, EventCluster)):
        return False
    new_combined = f"{new_event.title} " + " ".join(
        f"{a.title} {a.snippet or ''}" for a in getattr(new_event, "articles", ())
    )
    new_facility = extract_event_core_facility(new_combined)

    if isinstance(prev_event_or_sig, str):
        if not prev_event_or_sig.startswith("sig:"):
            return False
        new_sig = compute_event_semantic_signature(new_event)
        if new_sig == prev_event_or_sig:
            return True
        parts_prev = prev_event_or_sig.split(":")
        parts_new = new_sig.split(":")
        if len(parts_prev) >= 5 and len(parts_new) >= 5:
            if parts_prev[2] and parts_prev[2] == parts_new[2]:
                if parts_prev[4] == parts_new[4] and parts_prev[4] != "general":
                    return True
            if parts_prev[3] and parts_prev[3] == parts_new[3]:
                return True
        return False

    prev_event = prev_event_or_sig
    prev_combined = f"{prev_event.title} " + " ".join(
        f"{a.title} {a.snippet or ''}" for a in getattr(prev_event, "articles", ())
    )
    prev_facility = extract_event_core_facility(prev_combined)

    if new_facility and prev_facility and new_facility.lower() == prev_facility.lower():
        new_theme = re.search(
            r"\b(plot|sabotage|break-in|protest|security|investigation|activist|charged)\b",
            new_combined,
            re.I,
        )
        prev_theme = re.search(
            r"\b(plot|sabotage|break-in|protest|security|investigation|activist|charged)\b",
            prev_combined,
            re.I,
        )
        if new_theme and prev_theme:
            return True
        new_deploy = re.search(r"\b(deploy|exercise|bomber|flight|drill)\b", new_combined, re.I)
        prev_deploy = re.search(r"\b(deploy|exercise|bomber|flight|drill)\b", prev_combined, re.I)
        if bool(new_deploy) != bool(prev_deploy):
            return False

    new_suspect = re.search(
        r"\b(?:Sergeant|Sgt\.?|Soldier|Captain|Major|Colonel|Analyst|Contractor|Officer)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b",
        new_combined,
    )
    prev_suspect = re.search(
        r"\b(?:Sergeant|Sgt\.?|Soldier|Captain|Major|Colonel|Analyst|Contractor|Officer)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b",
        prev_combined,
    )
    if (
        new_suspect
        and prev_suspect
        and new_suspect.group(1).lower() == prev_suspect.group(1).lower()
    ):
        return True

    return False


def has_meaningful_development(event: EventCluster) -> bool:
    """Check if an event cluster contains a material progression/update rather than status quo."""
    combined_text = f"{event.title} " + " ".join(
        f"{a.title} {a.snippet or ''}" for a in getattr(event, "articles", ())
    )
    return bool(_MEANINGFUL_DEVELOPMENT_PATTERNS.search(combined_text))


def mark_event_as_update(event: EventCluster) -> EventCluster:
    """Return a copy of the event cluster marked as an update."""
    title = event.title
    if not title.startswith("[UPDATE]"):
        title = f"[UPDATE] {title}"
    return event.model_copy(update={"is_update": True, "title": title})
