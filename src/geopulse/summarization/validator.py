"""Deterministic factual grounding and entity verification for event summaries.

Validates that LLM-generated summaries do not introduce unsupported actors,
fabricated modern alliances, false historical claims, or unwarranted corroboration.
"""

import html
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from geopulse.clustering.models import EventCluster
from geopulse.ingestion.normalization import normalize_url
from geopulse.summarization.models import EventSummary

_DEMONYM_SUBJECT_MAP: dict[str, str] = {
    "turkish": "Turkey",
    "british": "UK",
    "spanish": "Spain",
    "german": "Germany",
    "russian": "Russia",
    "ukrainian": "Ukraine",
    "french": "France",
    "polish": "Poland",
    "chinese": "China",
    "israeli": "Israel",
    "american": "US",
    "italian": "Italy",
    "swedish": "Sweden",
    "finnish": "Finland",
    "greek": "Greece",
    "japanese": "Japan",
    "iranian": "Iran",
}

_VERB_AFTER_SUBJECT_PATTERN = re.compile(
    r"^(Turkish|British|Spanish|German|Russian|Ukrainian|French|Polish|Chinese|Israeli|"
    r"American|Italian|Swedish|Finnish|Greek|Japanese|Iranian)\s+"
    r"(unveils?|launches?|deploys?|procures?|approves?|announces?|signs?|demos?|tests?|"
    r"acquires?|commits?|orders?|reveals?|develops?|completes?|agrees?|confirms?|warns?|"
    r"urges?|demands?|reports?|targets?|strikes?)\b",
    re.I,
)


_LEAKAGE_PATTERN = re.compile(
    r"\[?\b(?:unspecified(?:\s+in\s+reporting)?|unknown\s+in\s+reporting|"
    r"not\s+specified(?:\s+in\s+reporting|\s+in\s+source)?)\b\]?",
    re.I,
)


def _distinct_displayed_sources(summary: EventSummary, event: EventCluster) -> int:
    """Count distinct, event-attached publisher/link pairs shown to readers."""

    def identities(items: Sequence[object], *, is_article: bool) -> set[tuple[str, str]]:
        result: set[tuple[str, str]] = set()
        for item in items:
            name = getattr(item, "source_name", "")
            url = getattr(item, "canonical_url", "") if is_article else getattr(item, "url", "")
            if (
                not isinstance(name, str)
                or not name.strip()
                or not isinstance(url, str)
                or not url.strip()
            ):
                continue
            try:
                canonical_url = normalize_url(url)
            except ValueError:
                canonical_url = url.strip().casefold()
            result.add((name.strip().casefold(), canonical_url))
        return result

    attached = identities(summary.sources, is_article=False)
    event_sources = identities(event.articles, is_article=True)
    common = attached & event_sources
    distinct_publishers = {name for name, _ in common}
    distinct_urls = {url for _, url in common}
    return min(len(distinct_publishers), len(distinct_urls))


_METONYM_CAPITALS: frozenset[str] = frozenset(
    {
        "moscow",
        "washington",
        "beijing",
        "london",
        "paris",
        "berlin",
        "tokyo",
        "tehran",
        "kyiv",
        "ankara",
        "brussels",
        "tel aviv",
    }
)

_SPACE_MARITIME_CYBER_TERMS: re.Pattern[str] = re.compile(
    r"\b(space\s+wars?|space\s+operations?|military\s+satellites?|satellite\s+maneuvers?|"
    r"orbital\s+maneuvers?|satellite|orbit|orbital|rendezvous|spacecraft|cislunar|"
    r"high\s+seas|international\s+waters|naval\s+patrol|carrier\s+voyages?|carrier\s+strike\s+group|"
    r"disputed\s+asian\s+seas|taiwan\s+strait|south\s+china\s+sea|philippine\s+sea|"
    r"cyberattack|cyber\s+warfare|malware|gps\s+jamming)\b",
    re.I,
)


def sanitize_placeholder_text(text: str) -> str:
    """Strip 'Unspecified in reporting', '[Unspecified]', and placeholder leakage from text."""
    if not text:
        return ""
    # Strip phrases like "in the Ukrainian city of [Unspecified in reporting]"
    cleaned = re.sub(
        r"\b(?:in\s+)?(?:the\s+)?(?:[\w-]+\s+)?city\s+of\s+\[?(?:unspecified(?:\s+in\s+reporting)?|unknown)\]?",
        "",
        text,
        flags=re.I,
    )
    # Strip standalone placeholder leakage
    cleaned = re.sub(
        r"\s*\[?(?:unspecified(?:\s+in\s+reporting)?|not\s+specified\s+in\s+reporting|"
        r"unknown\s+in\s+reporting|date\s+not\s+specified\s+in\s+source)\]?",
        "",
        cleaned,
        flags=re.I,
    )
    # Collapse multiple spaces and clean dangling punctuation
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\s+([,.:;])", r"\1", cleaned)
    return cleaned


def clean_headline_grammar(headline: str, source_title: str | None = None) -> str:
    """Clean grammatical anomalies, HTML entities, and improper demonym subjects in headlines."""
    if not headline:
        return source_title or ""

    # 1. Unescape HTML entities (&amp;, &#8216;, etc.)
    cleaned = html.unescape(headline).strip()

    # 2. Strip placeholder leakage
    cleaned = sanitize_placeholder_text(cleaned)

    # 3. Strip surrounding quotes or bullets
    cleaned = re.sub(r"^[\"\'\u201c\u201d\u2018\u2019\u2022\-\s]+", "", cleaned)
    cleaned = re.sub(r"[\"\'\u201c\u201d\u2018\u2019\s]+$", "", cleaned)

    # 4. Correct known typos / malformed words (e.g., 'Intercceptors', 'inteceptors')
    def _fix_case(match: re.Match[str], replacement: str) -> str:
        orig = match.group(0)
        if orig.isupper():
            return replacement.upper()
        if orig[0].isupper():
            return replacement.capitalize()
        return replacement.lower()

    cleaned = re.sub(
        r"\bintercceptors\b",
        lambda m: _fix_case(m, "interceptors"),
        cleaned,
        flags=re.I,
    )
    cleaned = re.sub(
        r"\bintercceptor\b",
        lambda m: _fix_case(m, "interceptor"),
        cleaned,
        flags=re.I,
    )
    cleaned = re.sub(
        r"\binteceptors\b",
        lambda m: _fix_case(m, "interceptors"),
        cleaned,
        flags=re.I,
    )
    cleaned = re.sub(
        r"\binteceptor\b",
        lambda m: _fix_case(m, "interceptor"),
        cleaned,
        flags=re.I,
    )
    cleaned = re.sub(
        r"\bsattelites\b",
        lambda m: _fix_case(m, "satellites"),
        cleaned,
        flags=re.I,
    )
    cleaned = re.sub(
        r"\bsattelite\b",
        lambda m: _fix_case(m, "satellite"),
        cleaned,
        flags=re.I,
    )

    # 5. Fix awkward constructions like 'Forms Patterns in...'
    cleaned = re.sub(
        r"\b(?:forms?|forming)\s+patterns?\s+(?:of\s+voyages\s+)?in\b",
        "Conducts Pattern of Voyages in",
        cleaned,
        flags=re.I,
    )
    cleaned = re.sub(r"\bforms?\s+patterns?\b", "Conducts Voyages", cleaned, flags=re.I)

    # 6. Correct third-party possessive ceasefire in headline (e.g. 'with X's Ceasefire')
    cleaned = re.sub(
        r"\bwith\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?'s\s+(?:Ceasefire|Truce)\b",
        "for Humanitarian Aid",
        cleaned,
        flags=re.I,
    )

    # 7. Fix demonym adjective erroneously used as a singular subject before a verb
    # e.g., "Turkish Unveils Ateş Serbest Drone" -> "Turkey Unveils Ateş Serbest Drone"
    match = _VERB_AFTER_SUBJECT_PATTERN.match(cleaned)
    if match:
        demonym = match.group(1).lower()
        replacement_noun = _DEMONYM_SUBJECT_MAP.get(demonym)
        if replacement_noun:
            verb = match.group(2)
            rest = cleaned[match.end() :]
            cleaned = f"{replacement_noun} {verb}{rest}"

    # 8. Restore truncated military equipment / aircraft designations if present in source_title
    # e.g., source has "F-47", "Su-57", but headline dropped prefix to "47’s" or "47"
    if source_title:
        designation_pat = re.compile(r"\b([A-Za-z]{1,4})-(\d{1,4})\b")
        for m in designation_pat.finditer(source_title):
            prefix, num = m.group(1), m.group(2)
            pattern = re.compile(rf"\b(?<![A-Za-z0-9\-])(?<!-){re.escape(num)}(['’]s)?\b")
            cleaned = pattern.sub(rf"{prefix}-{num}\1", cleaned)

    # 9. Collapse multiple spaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # 10. Fallback to clean source title if headline remains malformed or excessively short
    if source_title and (
        re.search(r"\b[a-zA-Z]*[c]{3,}[a-zA-Z]*\b", cleaned) or len(cleaned.split()) < 3
    ):
        return html.unescape(source_title).strip()

    return cleaned


_DANGLING_ENDINGS = re.compile(
    r"\b("
    # Conjunctions, prepositions, and relative connectors
    r"and|or|but|nor|so|for|yet|in|on|at|to|from|by|with|about|into|through|"
    r"during|before|after|above|below|between|under|over|that|which|who|whom|whose|"
    r"because|as|since|although|though|while|if|unless|until|as\s+well\s+as|"
    r"according\s+to|regarding|concerning|amid|per|versus|vs|via|"
    # Incomplete transitive, reporting, directive, or action verbs and participles
    r"called|urged|warned|asks?|asked|demands?|demanded|sought|orders?|ordered|"
    r"directs?|directed|selects?|selected|announces?|announced|declares?|declared|"
    r"requires?|required|aims?|aimed|plans?|planned|seeks?|decides?|decided|"
    r"pledges?|pledged|vows?|vowed|agrees?|agreed|promises?|promised|expects?|expected|"
    r"considers?|considered|begins?|began|starts?|started|continues?|continued|"
    r"reveals?|revealed|unveils?|unveiled|identifies?|identified|approves?|approved|"
    r"authorizes?|authorized|signs?|signed|targets?|targeted|citing|quotes?|quoted|"
    r"asserts?|asserted|says|said|stated|reported|argued|claimed|"
    # Participles and modifiers lacking objects
    r"designated|operating|following|involving|including|"
    # Determiners and articles
    r"a|an|the|this|that|these|those|its|their|his|her|our|some|any|each|every|all|both|"
    # Auxiliary and modal verbs
    r"is|are|was|were|be|been|being|have|has|had|do|does|did|will|would|shall|should|may|might|must|can|could"
    r")\s*$",
    re.I,
)

_CLAUSE_BOUNDARY_PATTERNS = re.compile(
    r"\s+(?:to\s+(?:require|prioritise|prioritize|invest|support|boost|protect|provide|expand|counter|deter|strengthen|conduct|secure|ensure)|"
    r"after|amid|following|where|while|which|that|in\s+order\s+to|aimed\s+at)\b",
    re.I,
)


def safe_truncate_headline(text: str, max_chars: int = 140) -> str:
    """Safely truncate a headline at a complete sentence or meaningful phrase boundary.

    Guarantees no mid-word cuts and no dangling conjunctions, prepositions, or incomplete verbs.
    """
    if not text:
        return ""
    cleaned = text.strip().rstrip(" ,;:-\t")
    if len(cleaned) <= max_chars:
        # Strip any trailing dangling connectors even if within max_chars
        for _ in range(3):
            cleaned = cleaned.rstrip(" ,;:-\t")
            match = _DANGLING_ENDINGS.search(cleaned)
            if match:
                cleaned = cleaned[: match.start()].strip()
            else:
                break
        return cleaned.rstrip(" ,;:-\t")

    # 1. Try first complete sentence if within limits
    sentence_match = re.search(r"^([^.!?]+[.!?])", cleaned)
    if sentence_match:
        first_sentence = sentence_match.group(1).strip()
        if 20 <= len(first_sentence) <= max_chars:
            cand = first_sentence.rstrip(" ,;:-\t")
            for _ in range(3):
                cand = cand.rstrip(" ,;:-\t")
                m = _DANGLING_ENDINGS.search(cand)
                if m:
                    cand = cand[: m.start()].strip()
                else:
                    break
            if len(cand) >= 20:
                return cand.rstrip(" ,;:-\t")

    # 2. Try complete clause before colon, semicolon, dash, or em-dash
    clause_match = re.search(r"^([^:;—–]+[:;—–])", cleaned)
    if clause_match:
        first_clause = clause_match.group(1).strip().rstrip(":;—– ")
        if 20 <= len(first_clause) <= max_chars:
            cand = first_clause
            for _ in range(3):
                cand = cand.rstrip(" ,;:-\t")
                m = _DANGLING_ENDINGS.search(cand)
                if m:
                    cand = cand[: m.start()].strip()
                else:
                    break
            if len(cand) >= 20:
                return cand.rstrip(" ,;:-\t")

    # 3. Look for natural clause/phrase boundaries before max_chars
    # Find all dependent clause boundaries within the character budget
    truncated_region = cleaned[:max_chars]
    clause_splits = list(_CLAUSE_BOUNDARY_PATTERNS.finditer(truncated_region))
    if clause_splits:
        # Choose the longest meaningful phrase boundary that yields >= 25 chars
        for split in reversed(clause_splits):
            cand = truncated_region[: split.start()].strip().rstrip(" ,;:-\t")
            for _ in range(3):
                cand = cand.rstrip(" ,;:-\t")
                m = _DANGLING_ENDINGS.search(cand)
                if m:
                    cand = cand[: m.start()].strip()
                else:
                    break
            if len(cand) >= 25:
                # If discarding the rest of truncated_region would strip a concrete numeric
                # detail (e.g. \b\d+\s+[A-Za-z]+) that would otherwise fit in max_chars, check if
                # a longer candidate or sentence ending can be preserved
                discarded = truncated_region[split.start() :]
                if re.search(r"\b\d+\s+[A-Za-z]+", discarded):
                    next_punct = re.search(r"[,;:.!?]", discarded)
                    if next_punct:
                        longer_cand = (
                            truncated_region[: split.start() + next_punct.start()]
                            .strip()
                            .rstrip(" ,;:-\t")
                        )
                        if len(longer_cand) <= max_chars and len(longer_cand) >= 25:
                            return longer_cand
                return cand.rstrip(" ,;:-\t")

    # 4. Truncate at last whitespace before max_chars
    truncated = cleaned[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space >= 10:
        candidate = truncated[:last_space].strip()
    else:
        candidate = truncated.strip()

    # 5. Strip dangling prepositions, conjunctions, verbs, and connectors
    for _ in range(4):
        candidate = candidate.rstrip(" ,;:-\t")
        match = _DANGLING_ENDINGS.search(candidate)
        if match:
            candidate = candidate[: match.start()].strip()
        else:
            break

    return candidate.rstrip(" ,;:-\t")


_KNOWN_PUBLISHER_NAMES: frozenset[str] = frozenset(
    {
        "al jazeera",
        "al jazeera english",
        "al jazeera arabic",
        "bbc",
        "bbc news",
        "bbc world news",
        "bbc world service",
        "reuters",
        "associated press",
        "ap news",
        "ap",
        "afp",
        "agence france-presse",
        "defense news",
        "breaking defense",
        "war on the rocks",
        "the war zone",
        "twz",
        "ukrinform",
        "tass",
        "ria novosti",
        "interfax",
        "financial times",
        "the guardian",
        "the new york times",
        "the washington post",
        "the wall street journal",
        "bloomberg",
        "politico",
        "cnn",
        "fox news",
        "msnbc",
        "cbs news",
        "abc news",
        "nbc news",
        "dw",
        "deutsche welle",
        "france 24",
        "rfi",
        "euronews",
        "al arabiya",
        "al mayadeen",
        "kyiv independent",
        "kyiv post",
        "moscow times",
        "the moscow times",
        "naval news",
        "air force times",
        "army times",
        "defense one",
        "usni news",
    }
)


def is_publisher_or_outlet(name: str, source_names: Sequence[str] = ()) -> bool:
    """Check if an entity name is a publisher, news outlet, or source metadata entry."""
    if not name:
        return False
    clean = name.strip().lower()
    # Official government, ministry, or military bodies are not publishers
    if re.search(
        r"\b(ministry|department|government|house|pentagon|kremlin|embassy|consulate|"
        r"parliament|congress|senate|armed forces|defence forces|defense forces)\b",
        clean,
    ):
        return False
    if clean in _KNOWN_PUBLISHER_NAMES:
        return True
    for s in source_names:
        if s and clean == s.strip().lower():
            return True
    clean_no_the = re.sub(r"^the\s+", "", clean)
    if clean_no_the in _KNOWN_PUBLISHER_NAMES:
        return True
    for s in source_names:
        if s and clean_no_the == re.sub(r"^the\s+", "", s.strip().lower()):
            return True
    # Common news outlet suffix/pattern check
    if re.search(
        r"\b(news|times|journal|post|herald|tribune|broadcasting|gazette|daily|wire|agency|press)\b",
        clean,
    ):
        return True
    return False


_KNOWN_ACTOR_ACRONYMS: dict[str, str] = {
    "usfk": "US Forces Korea",
    "dod": "Department of Defense",
    "mod": "Ministry of Defence",
    "idf": "Israel Defense Forces",
    "irgc": "Islamic Revolutionary Guard Corps",
    "pla": "People's Liberation Army",
    "plan": "People's Liberation Army Navy",
    "nato": "North Atlantic Treaty Organization",
    "iaea": "International Atomic Energy Agency",
    "usaf": "US Air Force",
    "usn": "US Navy",
    "uk": "United Kingdom",
    "us": "United States",
    "eu": "European Union",
    "un": "United Nations",
}


def deduplicate_actor_acronyms(actors: Sequence[str]) -> list[str]:
    """Deduplicate actor lists when both an acronym and its expanded name are present.

    For example, ['US Forces Korea', 'USFK'] -> ['US Forces Korea'].
    """
    if len(actors) <= 1:
        return list(actors)

    lower_map = {a.lower().strip(): a for a in actors}
    to_remove: set[str] = set()

    for a in actors:
        a_clean = a.strip()
        a_lower = a_clean.lower()

        # 1. Known dictionary lookup
        if a_lower in _KNOWN_ACTOR_ACRONYMS:
            expanded = _KNOWN_ACTOR_ACRONYMS[a_lower].lower()
            if any(expanded in other_lower for other_lower in lower_map if other_lower != a_lower):
                to_remove.add(a_clean)
                continue

        # 2. Dynamic acronym check against longer actors
        # e.g., 'USFK' vs 'US Forces Korea' or 'DoD' vs 'Department of Defense'
        if 2 <= len(a_clean) <= 6 and a_clean.isupper():
            for other in actors:
                if other == a or len(other) <= len(a):
                    continue
                words = [
                    w
                    for w in re.split(r"[\s-]+", other)
                    if w and w.lower() not in {"of", "the", "and", "for", "in", "to"}
                ]
                initials = "".join(w[0] for w in words).upper()
                compound_initials = "".join(
                    w if w.upper() in {"US", "UK", "EU", "UN"} else w[0] for w in words
                ).upper()

                if a_clean in {initials, compound_initials}:
                    to_remove.add(a_clean)
                    break

    return [a for a in actors if a not in to_remove]


def sanitize_actors(
    actors: Sequence[str],
    source_names: Sequence[str] = (),
) -> list[str]:
    """Deduplicate, clean, and normalize actor names, filtering publishers and placeholders."""
    seen_lower: set[str] = set()
    sanitized: list[str] = []

    for raw in actors:
        clean = html.unescape(str(raw)).strip()
        # Strip outer quotes or punctuation
        clean = clean.strip("\"'“”‘’.,;: ")
        # Strip possessive suffix if mistakenly parsed (e.g., "Turkey's" -> "Turkey")
        clean = re.sub(r"['’]s$", "", clean).strip()

        if not clean or len(clean) < 2:
            continue
        clean_lower = clean.lower()
        if (
            clean_lower
            in {
                "unspecified",
                "unspecified in reporting",
                "none",
                "null",
                "unknown",
                "single source",
            }
            or bool(_LEAKAGE_PATTERN.search(clean_lower))
            or is_publisher_or_outlet(clean, source_names)
        ):
            continue
        if clean_lower in seen_lower:
            continue

        seen_lower.add(clean_lower)
        sanitized.append(clean)

    return deduplicate_actor_acronyms(sanitized)


def ground_event_location(
    raw_location: str | None,
    event: EventCluster,
    source_text: str | None = None,
) -> str | None:
    """Ground and validate an event's location, distinguishing capitals from event scenes.

    If the event occurred in space/orbit, high seas, or cyber, national capitals (e.g. Moscow,
    Washington) represent seats of government/actors rather than event locations.
    If the location is unsupported by source text or a placeholder, it is cleanly omitted (None).
    """
    if not raw_location:
        return None

    cleaned = html.unescape(str(raw_location)).strip().strip("\"'“”‘’.,;: ")
    if not cleaned:
        return None

    cleaned_lower = cleaned.lower()
    if _LEAKAGE_PATTERN.search(cleaned_lower) or cleaned_lower in {
        "unspecified",
        "unspecified in reporting",
        "unknown",
        "none",
        "null",
        "n/a",
        "reported conflict zone",
        "conflict zone",
    }:
        return None

    if source_text is None:
        src_parts = []
        for art in event.articles:
            src_parts.append(art.title)
            if art.snippet:
                src_parts.append(art.snippet)
        source_text = " ".join(src_parts)

    src_lower = source_text.lower()

    # 1. Distinguish space / orbital / high seas / cyber metonym capitals from event locations
    if cleaned_lower in _METONYM_CAPITALS:
        if _SPACE_MARITIME_CYBER_TERMS.search(source_text):
            # In space/high-seas/cyber events, capital is an actor/headquarters, not event scene
            return None
        attribution_pat = re.compile(
            rf"\b(?:blamed|blames|warned|warns|says|said|urged|demanded|announced|reported\s+by)\s+{re.escape(cleaned_lower)}\b|"
            rf"\b{re.escape(cleaned_lower)}\s+(?:blamed|blames|warned|warns|says|said|urged|demanded|announced)\b"
        )
        scene_pat = re.compile(
            rf"\b(?:in|at|outside|near|across)\s+(?:the\s+city\s+of\s+)?{re.escape(cleaned_lower)}\b"
        )
        if attribution_pat.search(src_lower) and not scene_pat.search(src_lower):
            return None

    # 2. Check if the location (or its demonym/synonyms) appears in source text
    if cleaned_lower in _DEMONYM_SYNONYMS:
        has_syn = any(
            syn in src_lower or bool(re.search(rf"\b{re.escape(syn)}\b", src_lower))
            for syn in _DEMONYM_SYNONYMS[cleaned_lower]
        )
        if has_syn:
            return cleaned

    loc_tokens = [
        tok.lower()
        for tok in re.findall(r"\b[a-zA-Z0-9_\u00C0-\u017F-]+\b", cleaned)
        if len(tok) >= 3 and tok.lower() not in _STOP_WORDS
    ]
    if loc_tokens:
        if len(loc_tokens) >= 2:
            phrase_clean = re.sub(r"\s+", " ", cleaned_lower)
            has_phrase = bool(re.search(rf"\b{re.escape(phrase_clean)}\b", src_lower))
            all_tokens = all(
                tok in src_lower
                or (
                    tok in _DEMONYM_SYNONYMS
                    and any(syn in src_lower for syn in _DEMONYM_SYNONYMS[tok])
                )
                for tok in loc_tokens
            )
            if not (has_phrase or all_tokens):
                return None
        else:
            tok = loc_tokens[0]
            has_tok = tok in src_lower or (
                tok in _DEMONYM_SYNONYMS and any(syn in src_lower for syn in _DEMONYM_SYNONYMS[tok])
            )
            if not has_tok:
                return None

    # 3. Reject location if it only appears in introductory boilerplate or list of challengers
    boilerplate_pat = re.compile(
        r"\b(?:challengers?|threats?|adversaries|competitors):\s*[^.]*\b"
        + re.escape(cleaned_lower)
        + r"\b",
        re.I,
    )
    if boilerplate_pat.search(src_lower):
        outside_boilerplate = boilerplate_pat.sub("", src_lower)
        if not re.search(rf"\b{re.escape(cleaned_lower)}\b", outside_boilerplate):
            return None

    return cleaned


# Words and titles commonly prefixed to person names
_PREFIX_TITLES: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^(?:(?:Ret\.|Retired)\s+)?(?:Gen\.|General|Adm\.|Admiral|Sec\.|Secretary|"
        r"President|Prime Minister|Minister|Colonel|Col\.|Capt\.|Captain|Lt\.|Lieutenant|"
        r"Pte|Private|Dr\.|Director|CEO|Chief Executive|Spokesman|Spokesperson)\s+",
        re.I,
    ),
)

# Common generic categories, institutions, or collective nouns
_GENERIC_ACTOR_WORDS: frozenset[str] = frozenset(
    {
        "military",
        "forces",
        "armed forces",
        "regional military forces",
        "army",
        "navy",
        "air force",
        "air forces",
        "land forces",
        "naval forces",
        "maritime forces",
        "troops",
        "soldiers",
        "fighters",
        "pilots",
        "special forces",
        "special operations forces",
        "insurgents",
        "rebels",
        "militants",
        "combatants",
        "officials",
        "authorities",
        "government",
        "leaders",
        "commanders",
        "insurgent commanders",
        "rebel commanders",
        "spokesperson",
        "spokesman",
        "spokeswoman",
        "military spokesmen",
        "military spokespersons",
        "military authorities",
        "military officials",
        "defence officials",
        "defense officials",
        "allies",
        "allied forces",
        "allied units",
        "allied nations",
        "allied commanders",
        "coalition",
        "coalition forces",
        "investigators",
        "citizens",
        "civilians",
        "resident",
        "residents",
        "local resident",
        "local residents",
        "reporting resident",
        "reporting residents",
        "witness",
        "witnesses",
        "experts",
        "analysts",
        "commentators",
        "authors",
        "defence policy makers",
        "defense policy makers",
        "semiconductor manufacturers",
        "contractors",
        "defence industry",
        "industry leaders",
        "single source",
        "reporting outlets",
        "sources",
        "unknown",
        "unspecified",
        "unspecified in reporting",
        "pentagon",
        "mod",
        "dod",
        "department of defense",
        "ministry of defence",
        "royal air force",
        "us",
        "uk",
        "united states",
        "united kingdom",
        "russia",
        "china",
        "france",
        "turkey",
        "germany",
        "ukraine",
        "poland",
    }
)

_DEMONYM_SYNONYMS: dict[str, tuple[str, ...]] = {
    "polish": ("poland", "polish"),
    "poland": ("poland", "polish"),
    "french": ("france", "french"),
    "france": ("france", "french"),
    "british": ("britain", "british", "uk", "united kingdom"),
    "britain": ("britain", "british", "uk", "united kingdom"),
    "uk": ("britain", "british", "uk", "united kingdom"),
    "united kingdom": ("britain", "british", "uk", "united kingdom"),
    "american": ("us", "usa", "america", "american", "united states"),
    "us": ("us", "usa", "america", "american", "united states"),
    "usa": ("us", "usa", "america", "american", "united states"),
    "united states": ("us", "usa", "america", "american", "united states"),
    "russian": ("russia", "russian"),
    "russia": ("russia", "russian"),
    "ukrainian": ("ukraine", "ukrainian"),
    "ukraine": ("ukraine", "ukrainian"),
    "german": ("germany", "german"),
    "germany": ("germany", "german"),
    "raf": ("raf", "royal air force", "uk", "british", "britain"),
    "royal air force": ("raf", "royal air force", "uk", "british", "britain"),
    "allied": ("allies", "allied", "alliance", "coalition"),
    "allies": ("allies", "allied", "alliance", "coalition"),
    "coalition": ("coalition", "allies", "allied"),
}

_GENERIC_BRANCH_DESCRIPTORS: frozenset[str] = frozenset(
    {
        "army",
        "navy",
        "air",
        "force",
        "forces",
        "land",
        "maritime",
        "military",
        "command",
        "committee",
        "corps",
        "defence",
        "defense",
        "ministry",
        "department",
        "officials",
        "authorities",
        "commanders",
        "spokesmen",
        "spokespersons",
        "units",
        "regiment",
        "troops",
        "fighters",
        "pilots",
    }
)

_STOP_WORDS: frozenset[str] = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "from",
        "into",
        "over",
        "under",
        "about",
        "new",
        "u.s.",
        "u.k.",
    }
)

# Acronyms that require explicit source attestation when generated in headlines or claims
_PROTECTED_ACRONYMS: frozenset[str] = frozenset(
    {
        "NATO",
        "AUKUS",
        "NORAD",
        "CENTCOM",
        "EUCOM",
        "INDOPACOM",
        "BRICS",
        "IAEA",
        "OPEC",
        "FSB",
        "GRU",
        "KGB",
        "Mossad",
        "MI6",
        "CIA",
    }
)

_KNOWN_VESSEL_ACRONYMS: dict[str, str] = {
    "lsm": "Landing Ship Medium",
    "lst": "Landing Ship, Tank",
    "lsth": "Landing Ship, Tank, Helicopter",
    "lsd": "Landing Ship Dock",
    "lpd": "Landing Platform Dock",
    "lha": "Landing Helicopter Assault",
    "lhd": "Landing Helicopter Dock",
    "usv": "Unmanned Surface Vessel",
    "uuv": "Unmanned Underwater Vehicle",
    "uav": "Unmanned Aerial Vehicle",
}


def _find_source_expansion_for_acronym(acronym: str, source_text: str) -> str | None:
    """Find source-supported expansion for an acronym if present in source text."""
    if not source_text or not acronym:
        return None
    known = _KNOWN_VESSEL_ACRONYMS.get(acronym.lower())
    if known and re.search(rf"\b{re.escape(known)}\b", source_text, re.I):
        return known
    m1 = re.search(
        rf"\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*){{1,4}})\s*\(\s*{re.escape(acronym)}\s*\)",
        source_text,
    )
    if m1:
        return m1.group(1).strip()
    m2 = re.search(
        rf"\b{re.escape(acronym)}\s*\(\s*([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*){{1,4}})\s*\)",
        source_text,
    )
    if m2:
        return m2.group(1).strip()
    return None


def sanitize_technical_acronyms(text: str, source_text: str) -> str:
    """Ensure technical acronyms/expansions are strictly taken from source terminology.

    Never invent an expansion based on familiar military acronyms or historical classifications
    (e.g., never expand LSM as 'LSTH' or 'Landing Ship, Tank, Helicopter').
    When the source does not provide an expansion, preserves the acronym without expanding it.
    """
    if not text or not source_text:
        return text

    source_lower = source_text.lower()

    # 1. Check parenthetical acronym expansions: ACRONYM (Expansion)
    def _replace_acronym_paren(m: re.Match) -> str:
        acronym = m.group(1).strip()
        expansion = m.group(2).strip()
        exp_words = [
            w.lower()
            for w in re.findall(r"\b[A-Za-z]{3,}\b", expansion)
            if w.lower() not in _STOP_WORDS
        ]
        if not exp_words:
            return m.group(0)
        is_supported = all(w in source_lower for w in exp_words)
        if is_supported and acronym.lower() in source_lower:
            return m.group(0)

        true_acronym = acronym
        if acronym.lower() not in source_lower:
            for k in _KNOWN_VESSEL_ACRONYMS:
                if re.search(rf"\b{k}\b", source_lower, re.I):
                    true_acronym = k.upper()
                    break

        src_exp = _find_source_expansion_for_acronym(true_acronym, source_text)
        if src_exp:
            return f"{true_acronym} ({src_exp})"
        return true_acronym

    text = re.sub(r"\b([A-Z0-9]{2,6})\s*\(([^)]+)\)", _replace_acronym_paren, text)

    # 2. Check reverse parenthetical: Expansion (ACRONYM)
    def _replace_expansion_paren(m: re.Match) -> str:
        expansion = m.group(1).strip()
        acronym = m.group(2).strip()
        exp_words = [
            w.lower()
            for w in re.findall(r"\b[A-Za-z]{3,}\b", expansion)
            if w.lower() not in _STOP_WORDS
        ]
        if not exp_words:
            return m.group(0)
        is_supported = all(w in source_lower for w in exp_words)
        if is_supported and acronym.lower() in source_lower:
            return m.group(0)

        true_acronym = acronym
        if acronym.lower() not in source_lower:
            for k in _KNOWN_VESSEL_ACRONYMS:
                if re.search(rf"\b{k}\b", source_lower, re.I):
                    true_acronym = k.upper()
                    break

        src_exp = _find_source_expansion_for_acronym(true_acronym, source_text)
        if src_exp:
            return f"{src_exp} ({true_acronym})"
        return true_acronym

    text = re.sub(
        r"((?:[A-Z][a-zA-Z]+(?:,\s*|\s+)){1,5}[A-Z][a-zA-Z]+)\s*\(([A-Z0-9]{2,6})\)",
        _replace_expansion_paren,
        text,
    )

    # 3. Check bare invented expansions:
    # e.g. 'Landing Ship, Tank, Helicopter' -> 'Landing Ship Medium' or 'LSM'
    invented_vessel_expansions = [
        (re.compile(r"\bLanding Ship,?\s+Tank,?\s+Helicopter\b", re.I), "lsth"),
        (re.compile(r"\bLanding Ship,?\s+Tank\b", re.I), "lst"),
    ]
    for pat, wrong_acro in invented_vessel_expansions:
        if pat.search(text) and not pat.search(source_text):
            target_acro = None
            for k in _KNOWN_VESSEL_ACRONYMS:
                if k != wrong_acro and re.search(rf"\b{k}\b", source_lower, re.I):
                    target_acro = k.upper()
                    break
            if target_acro:
                src_exp = _find_source_expansion_for_acronym(target_acro, source_text)
                repl = src_exp if src_exp else target_acro
                text = pat.sub(lambda m, r=repl: r, text)

    # 4. Check bare hallucinated acronym e.g. \bLSTH\b when source has LSM
    if re.search(r"\bLSTH\b", text) and not re.search(r"\bLSTH\b", source_text, re.I):
        if re.search(r"\bLSM\b", source_text, re.I):
            text = re.sub(r"\bLSTH\b", "LSM", text)

    return text


_TITLED_PERSON_PATTERN = re.compile(
    r"\b(?:(?:Ret\.|Retired)\s+)?(?:Gen\.|General|Adm\.|Admiral|Sec\.|Secretary|"
    r"President|Prime Minister|Minister|Colonel|Col\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"
)

_HISTORICAL_MARKERS = re.compile(
    r"\b(?:battle of\s+[\w\s'-]+?|bail[eé]n|waterloo|trafalgar|gettysburg|antietam|"
    r"verdun|somme|passchendaele|gallipoli|stalingrad|normandy landing|"
    r"crimean war|boer war|peloponnesian war|punic war|crusades|ancient warfare|"
    r"roman (?:legion|army|empire)|medieval (?:warfare|knights|siege)|"
    r"napoleon(?:ic)?(?:\s+wars?|\s+era)?|"
    r"in\s+(?:1[4-9]\d\d|1[0-3]\d\d)|"
    r"on the morning of [a-z]+ \d{1,2},\s*(?:1[4-9]\d\d|1[0-3]\d\d))\b",
    re.I,
)

_CONTEMPORARY_HOOK_MARKERS = re.compile(
    r"\b(?:dna (?:testing|analysis|match|identification)|"
    r"remains? (?:identified|laid to rest|unearthed|discovered|recovered|found)|"
    r"laid to rest|unearthed|archaeolog\w*|ceremony|memorial service|commemorat\w*|"
    r"newly declassified|unsealed (?:records|files)|inquest|inquiry (?:finds|reveals|concludes)|"
    r"new evidence uncovered|posthumous|apology|pardon|museum exhibit)\b",
    re.I,
)


def extract_registered_domain(url: str) -> str:
    """Extract registered domain from URL (e.g. 'reuters.com', 'bbc.co.uk')."""
    if not url:
        return ""
    try:
        netloc = urlparse(url).netloc.lower().split(":")[0]
    except Exception:
        return ""
    netloc = re.sub(r"^(?:www\d*|m)\.", "", netloc)
    parts = netloc.split(".")
    if len(parts) >= 3 and parts[-2] in {
        "co",
        "com",
        "gov",
        "org",
        "net",
        "edu",
        "ac",
        "mil",
    }:
        return ".".join(parts[-3:])
    elif len(parts) >= 2:
        return ".".join(parts[-2:])
    return netloc


_CANONICAL_PUBLISHER_MAP: dict[str, str] = {
    "ap": "associated press",
    "associated press": "associated press",
    "ap news": "associated press",
    "reuters": "reuters",
    "reuters uk": "reuters",
    "reuters world": "reuters",
    "afp": "afp",
    "agence france-presse": "afp",
    "bloomberg": "bloomberg",
    "bloomberg news": "bloomberg",
    "tass": "tass",
    "ria novosti": "ria novosti",
    "interfax": "interfax",
    "bbc": "bbc",
    "bbc news": "bbc",
    "bbc world news": "bbc",
    "al jazeera": "al jazeera",
    "al jazeera english": "al jazeera",
    "financial times": "financial times",
    "ft": "financial times",
    "defense news": "defense news",
    "breaking defense": "breaking defense",
    "the guardian": "the guardian",
    "the new york times": "the new york times",
    "nyt": "the new york times",
    "the washington post": "the washington post",
    "wapo": "the washington post",
    "politico": "politico",
}


def canonicalize_publisher_name(name: str | None) -> str:
    """Normalize publisher / outlet names for independence checking."""
    if not name:
        return ""
    cleaned = re.sub(r"[^a-zA-Z0-9\s]+", " ", name).strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return _CANONICAL_PUBLISHER_MAP.get(cleaned, cleaned)


def _extract_wire_service(text: str) -> str | None:
    """Detect if an article text is a wire service reproduction."""
    if not text:
        return None
    text_lower = text.lower()
    if re.search(
        r"(?:(?:^|[^\w])\(?ap\)?\s*[-—]|associated\s+press\b|\baccording\s+to\s+(?:the\s+)?ap\b)",
        text_lower,
    ):
        return "associated press"
    if re.search(
        r"(?:(?:^|[^\w])\(?reuters\)?\s*[-—]|by\s+reuters\b|\breuters\s+reported\b|\baccording\s+to\s+reuters\b)",
        text_lower,
    ):
        return "reuters"
    if re.search(
        r"(?:(?:^|[^\w])\(?afp\)?\s*[-—]|agence\s+france-presse\b|\baccording\s+to\s+afp\b)",
        text_lower,
    ):
        return "afp"
    if re.search(
        r"(?:(?:^|[^\w])\(?bloomberg\)?\s*[-—]|\bbloomberg\s+reported\b|\baccording\s+to\s+bloomberg\b)",
        text_lower,
    ):
        return "bloomberg"
    if re.search(
        r"(?:(?:^|[^\w])\(?tass\)?\s*[-—]|\btass\s+reported\b|\baccording\s+to\s+tass\b)",
        text_lower,
    ):
        return "tass"
    return None


_SYNTHETIC_FIXTURE_DOMAINS: frozenset[str] = frozenset(
    {
        "example.com",
        "example.org",
        "example.net",
        "example.test",
        "test.com",
        "test.org",
        "localhost",
    }
)


def count_genuinely_independent_sources(sources: Sequence[Any]) -> int:
    """Count the number of genuinely independent sources supporting an event or claim.

    Does not count syndicated copies, duplicated articles, or multiple URLs
    from the same publisher/domain as independent corroboration.
    """
    if not sources:
        return 0
    if len(sources) == 1:
        return 1

    items: list[dict[str, Any]] = []
    for s in sources:
        url = getattr(s, "url", None) or (s.get("url") if isinstance(s, dict) else "")
        source_name = getattr(s, "source_name", None) or (
            s.get("source_name") if isinstance(s, dict) else ""
        )
        title = getattr(s, "title", None) or (s.get("title") if isinstance(s, dict) else "")
        snippet = getattr(s, "snippet", None) or (s.get("snippet") if isinstance(s, dict) else "")
        text = f"{title or ''} {snippet or ''}".strip()
        domain = extract_registered_domain(url or "")
        pub = canonicalize_publisher_name(source_name)
        wire = _extract_wire_service(text) or (
            pub if pub in {"associated press", "reuters", "afp", "bloomberg", "tass"} else None
        )
        words = set(re.findall(r"\b[A-Za-z]{4,}\b", text.lower()))
        items.append(
            {
                "url": url,
                "source_name": source_name,
                "domain": domain,
                "pub": pub,
                "wire": wire,
                "text": text.lower(),
                "words": words,
            }
        )

    parent = list(range(len(items)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            item_i = items[i]
            item_j = items[j]

            # 0. Identical URL (same article/link)
            if item_i["url"] and item_j["url"]:
                try:
                    if normalize_url(item_i["url"]) == normalize_url(item_j["url"]):
                        union(i, j)
                        continue
                except ValueError:
                    if item_i["url"] == item_j["url"]:
                        union(i, j)
                        continue

            # 1. Same registered domain (ignoring synthetic test/fixture domains)
            if (
                item_i["domain"]
                and item_i["domain"] == item_j["domain"]
                and item_i["domain"] not in _SYNTHETIC_FIXTURE_DOMAINS
            ):
                union(i, j)
                continue

            # 2. Same canonical publisher name
            if item_i["pub"] and item_i["pub"] == item_j["pub"]:
                union(i, j)
                continue

            # 3. Both share the same wire service origin
            if item_i["wire"] and item_i["wire"] == item_j["wire"]:
                union(i, j)
                continue

            # 4. One carries wire credit for the other
            if item_i["wire"] and item_i["wire"] == item_j["pub"]:
                union(i, j)
                continue
            if item_j["wire"] and item_j["wire"] == item_i["pub"]:
                union(i, j)
                continue

            # 5. One explicitly cites the other
            pub_i = item_i["pub"]
            pub_j = item_j["pub"]
            if pub_i and len(pub_i) >= 4:
                pat_i = (
                    rf"\b(?:according to|citing|quoted by|reported by|statement (?:from|by)|"
                    rf"per|data from|figures from)\s+(?:the\s+)?{re.escape(pub_i)}\b|"
                    rf"\b{re.escape(pub_i)}\s+(?:reported|said|announced|confirmed|noted)\b"
                )
                if re.search(pat_i, item_j["text"]):
                    union(i, j)
                    continue
            if pub_j and len(pub_j) >= 4:
                pat_j = (
                    rf"\b(?:according to|citing|quoted by|reported by|statement (?:from|by)|"
                    rf"per|data from|figures from)\s+(?:the\s+)?{re.escape(pub_j)}\b|"
                    rf"\b{re.escape(pub_j)}\s+(?:reported|said|announced|confirmed|noted)\b"
                )
                if re.search(pat_j, item_i["text"]):
                    union(i, j)
                    continue

            # 6. High textual duplication (for uncredited syndication or scraped copies)
            if item_i["words"] and item_j["words"]:
                if pub_i and pub_j and pub_i != pub_j:
                    # Distinct publishers: only union if substantial article text is near-identical
                    if len(item_i["text"]) >= 200 and len(item_j["text"]) >= 200:
                        overlap = len(item_i["words"] & item_j["words"])
                        union_words = len(item_i["words"] | item_j["words"])
                        if union_words > 0 and (overlap / union_words) >= 0.85:
                            union(i, j)
                            continue
                else:
                    overlap = len(item_i["words"] & item_j["words"])
                    union_words = len(item_i["words"] | item_j["words"])
                    if union_words > 0 and (overlap / union_words) >= 0.75:
                        union(i, j)
                        continue

    distinct_roots = {find(k) for k in range(len(items))}
    return len(distinct_roots)


def are_sources_genuinely_independent(sources: Sequence[Any]) -> bool:
    """Return True if sources contain at least two genuinely independent reporting outlets."""
    return count_genuinely_independent_sources(sources) >= 2


@dataclass(frozen=True)
class GroundingValidationResult:
    """Outcome of deterministic factual grounding validation."""

    is_valid: bool
    unsupported_entities: tuple[str, ...] = ()
    unsupported_claims: tuple[str, ...] = ()
    quarantine_reason: str | None = None


def is_evidence_sufficient(event: EventCluster) -> tuple[bool, str | None]:
    """Check whether an event cluster has sufficient informative text for factual summarization."""
    if not event.articles:
        return False, "No source articles attached to event cluster"

    corpus_parts: list[str] = []
    has_substantive_body = False

    for art in event.articles:
        t = (art.title or "").strip()
        s = (art.snippet or "").strip()
        if t:
            corpus_parts.append(t)
        if s:
            corpus_parts.append(s)
            s_clean = re.sub(
                r"\b(?:no snippet available|read more|click here|subscribe|photo by|"
                r"all rights reserved)\b",
                "",
                s,
                flags=re.I,
            ).strip()
            # Check if snippet provides substantive tokens beyond repeating the title
            t_words = set(re.findall(r"\b[a-zA-Z0-9_\u00C0-\u017F-]+\b", t.lower()))
            s_words = set(re.findall(r"\b[a-zA-Z0-9_\u00C0-\u017F-]+\b", s_clean.lower()))
            new_words = s_words - t_words
            if len(s_clean) >= 20 and len(new_words) >= 3:
                has_substantive_body = True

    combined_text = " ".join(corpus_parts).strip()
    # Remove standard non-informative boilerplate
    cleaned = re.sub(
        r"\b(?:no snippet available|read more|click here|subscribe|photo by|all rights reserved)\b",
        "",
        combined_text,
        flags=re.I,
    ).strip()

    if len(cleaned) < 40:
        return (
            False,
            f"Source text contains insufficient evidence ({len(cleaned)} chars)",
        )

    # Check for document labels with tautological snippets (e.g. 'E5 Statement' with snippet 'E5')
    _DOCUMENT_LABEL_PAT = re.compile(
        r"\b(statement|remarks|speech|notice|listings|readout|meeting|agenda|bulletin)\b",
        re.I,
    )
    title_text = (event.articles[0].title or "").strip() if event.articles else ""
    if not has_substantive_body and len(event.articles) == 1:
        if _DOCUMENT_LABEL_PAT.search(title_text):
            return (
                False,
                "Source text contains insufficient evidence: "
                "document label without substantive reporting body",
            )

    # Check for administrative guidance / reference documents that lack breaking event substance
    _ADMIN_GUIDANCE_PAT = re.compile(
        r"\b(guidance(?:\s*:)?|defence\s+safety\s+management\s+system|"
        r"jsp\s*\d+|joint\s+service\s+publication|standard\s+operating\s+procedures?|"
        r"code\s+of\s+practice|reference\s+table|pension\s+scheme)\b",
        re.I,
    )
    if _ADMIN_GUIDANCE_PAT.search(title_text):
        if not re.search(
            r"\b(overhaul|reform|inquiry|investigation|scandal|halted|suspended|grounded|"
            r"probe|charges|unveils|unveiled|crisis|disaster|fatal|crash)\b",
            combined_text,
            re.I,
        ):
            return (
                False,
                "Source text contains insufficient evidence: "
                "administrative guidance or reference document without breaking news event",
            )

    return True, None


def _validate_quantitative_claims(
    text: str,
    source_lower: str,
    source_text: str,
) -> list[str]:
    """Validate that quantitative assertions (percentages, times, quantities, magnitudes)

    are grounded in the source text. Quarantines unsupported numbers while preserving
    valid source-supported claims.
    """
    unsupported: list[str] = []

    # 1. Percentages (e.g., 48 percent, 97%, 48%)
    pct_matches = re.finditer(r"\b(\d+(?:\.\d+)?)\s*(?:%|percent|per cent)\b", text, re.I)
    for m in pct_matches:
        pct_val = m.group(1)
        if pct_val not in source_text:
            unsupported.append(f"Percentage claim '{m.group(0)}' not substantiated by source text")

    # 2. Time quantities (e.g. 20 minutes, 19 minutes, 45 minutes)
    time_matches = re.finditer(
        r"\b(?:nearly|roughly|about|approximately|under|over)?\s*(\d+)\s*"
        r"(minutes?|hours?|seconds?|days?)\b",
        text,
        re.I,
    )
    for m in time_matches:
        num_val = int(m.group(1))
        unit = m.group(2).lower().rstrip("s")
        allowed_nums = {str(num_val), str(num_val - 1), str(num_val + 1)}
        if not any(n in source_text for n in allowed_nums) or unit not in source_lower:
            unsupported.append(
                f"Time quantity '{m.group(0).strip()}' not substantiated by source text"
            )

    # 3. Multi-word magnitudes (e.g. hundreds or thousands, hundreds of thousands)
    mag_matches = re.finditer(
        r"\b(hundreds\s+or\s+thousands|hundreds\s+of\s+thousands|millions\s+of|billions\s+of)\b",
        text,
        re.I,
    )
    for m in mag_matches:
        mag_phrase = m.group(1).lower()
        first_word = mag_phrase.split()[0]
        last_word = mag_phrase.split()[-1]
        if first_word not in source_lower or last_word not in source_lower:
            unsupported.append(f"Magnitude claim '{m.group(0)}' not substantiated by source text")

    # 4. Large counts or quantities of weapons/assets (e.g. 50 drones, 100 missiles)
    asset_matches = re.finditer(
        r"\b(\d{2,})\s+(drones?|missiles?|aircraft|satellites?|ships?|vessels?|"
        r"tanks?|warheads?|seafarers?)\b",
        text,
        re.I,
    )
    for m in asset_matches:
        asset_num = m.group(1)
        asset_name = m.group(2).lower().rstrip("s")
        source_has_same_count_and_asset = any(
            re.search(rf"(?<!\d){re.escape(asset_num)}(?!\d)", sentence)
            and re.search(rf"\b{re.escape(asset_name)}s?\b", sentence, re.I)
            for sentence in re.split(r"(?<=[.!?])\s+", source_text)
        )
        if not source_has_same_count_and_asset:
            unsupported.append(
                f"Asset quantity claim '{m.group(0)}' not substantiated by source text"
            )

    # 5. Prices, currencies, and contract amounts (e.g. $100,000, $50 million, €20M, £5 billion)
    currency_matches = re.finditer(
        r"(?:[\$€£]|(?:USD|EUR|GBP)\s*)(\d+(?:,\d{3})*(?:\.\d+)?)\s*(billion|million|thousand)?\b",
        text,
        re.I,
    )
    for m in currency_matches:
        raw_num = m.group(1).replace(",", "")
        full_match = m.group(0).strip()
        raw_with_commas = m.group(1)
        found = (
            bool(re.search(r"(?<!\d)" + re.escape(raw_with_commas) + r"(?!\d)", source_text))
            or bool(re.search(r"(?<!\d)" + re.escape(raw_num) + r"(?!\d)", source_text))
            or full_match.lower() in source_lower
        )
        if not found:
            unsupported.append(
                f"Price or financial claim '{full_match}' not substantiated by source text"
            )

    # 6. Technical specifications (e.g. Mach 5, 500 km, 155mm)
    spec_matches = re.finditer(
        r"\b(mach\s*\d+(?:\.\d+)?|\d+\s*(?:km|kilometers?|miles?|mm|caliber))\b",
        text,
        re.I,
    )
    for m in spec_matches:
        spec_phrase = m.group(1).strip()
        num_m = re.search(r"\d+(?:\.\d+)?", spec_phrase)
        if num_m:
            num_part = num_m.group(0)
            if not re.search(r"(?<!\d)" + re.escape(num_part) + r"(?!\d)", source_text):
                unsupported.append(
                    f"Technical specification claim '{spec_phrase}' "
                    "not substantiated by source text"
                )

    # 7. Volumes, weights, and commodity capacities (e.g. 1.7 billion cubic metres, 390,000 barrels)
    vol_matches = re.finditer(
        r"\b(\d+(?:,\d{3})*(?:\.\d+)?)\s*(billion|million|thousand)?\s*"
        r"(cubic\s+(?:met(?:ers?|res?)|feet)|bcm|m3|barrels?(?:\s+a\s+day|\s*/\s*day)?|bpd|tons?|tonnes?)\b",
        text,
        re.I,
    )
    for m in vol_matches:
        full_vol = m.group(0).strip()
        raw_num = m.group(1).replace(",", "")
        raw_commas = m.group(1)
        found = (
            bool(re.search(r"(?<!\d)" + re.escape(raw_commas) + r"(?!\d)", source_text))
            or bool(re.search(r"(?<!\d)" + re.escape(raw_num) + r"(?!\d)", source_text))
            or full_vol.lower() in source_lower
        )
        if not found:
            unsupported.append(
                f"Volume or capacity claim '{full_vol}' not substantiated by source text"
            )

    # 8. Extended timeframes and multi-year durations (e.g. two decades, 20 years, 5 months)
    timeframe_matches = re.finditer(
        r"\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
        r"(decades?|centuries?|years?|months?)\b",
        text,
        re.I,
    )
    word_to_num = {
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    for m in timeframe_matches:
        full_tf = m.group(0).strip()
        num_str = m.group(1).lower()
        digit_str = word_to_num.get(num_str, num_str)
        unit = m.group(2).lower()
        unit_singular = unit.rstrip("s")
        found = full_tf.lower() in source_lower or (
            unit_singular in source_lower
            and (
                bool(re.search(r"(?<!\d)" + re.escape(digit_str) + r"(?!\d)", source_text))
                or num_str in source_lower
            )
        )
        if not found:
            unsupported.append(
                f"Timeframe or duration claim '{full_tf}' not substantiated by source text"
            )

    # 9. General ungrounded decimal quantities (e.g. 1.7 billion, 2.5 million)
    decimal_matches = re.finditer(
        r"\b(\d+\.\d+)\s*(billion|million|trillion|thousand)?\b",
        text,
        re.I,
    )
    for m in decimal_matches:
        dec_num = m.group(1)
        full_match = m.group(0).strip()
        if not re.search(r"(?<!\d)" + re.escape(dec_num) + r"(?!\d)", source_text):
            unsupported.append(
                f"Decimal quantitative claim '{full_match}' not substantiated by source text"
            )

    return unsupported


_MALFORMED_CLAUSE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # 1. Subordinate clause with relative pronoun ending abruptly without main predicate
    # e.g., ", as residents who survived on weeds during the conflict."
    re.compile(
        r",\s*(?:as|while|where|although|because|since)\s+"
        r"(?:the\s+)?[\w\s]{2,50}?\s+(?:who|which|that)\s+[^,\.!?]+[\.!?]$",
        re.I,
    ),
    # 2. Dangling conjunction or preposition at sentence boundary
    re.compile(
        r"\b(?:as|while|where|which|who|that|and|but|or|with|for|during|at|in|on|of|to)\s*[\.!?]$",
        re.I,
    ),
    # 3. Fragment starting with "as residents/civilians who" without a main predicate verb
    re.compile(
        r"\b(?:as|while)\s+(?:residents|civilians|citizens|people|refugees|families|forces)\s+"
        r"who\b[^,\.!?]+[\.!?]$",
        re.I,
    ),
    # 4. Incomplete comparative or attribution clause at sentence boundary
    re.compile(
        r",\s*as\s+(?:reported|noted|stated|described)\s+by\s*[\.!?]$",
        re.I,
    ),
)


def detect_malformed_clauses(text: str) -> list[str]:
    """Detect incomplete clauses and grammatically broken sentences in summary text."""
    if not text:
        return []
    issues: list[str] = []
    # Split by sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    for s in sentences:
        s_clean = s.strip()
        for pat in _MALFORMED_CLAUSE_PATTERNS:
            m = pat.search(s_clean)
            if m:
                matched_fragment = m.group(0).strip()
                issues.append(f"Incomplete/dangling clause detected: '{matched_fragment}'")
                break
    return issues


def check_cross_theatre_conflation(
    summary: EventSummary,
    source_text: str,
) -> list[str]:
    """Detect when a summary conflates actors, locations, or actions across theatres.

    Validates that if an operational military action (e.g. carrier deployment,
    troop evacuation, strike preparations) is reported in the source text for
    Theatre A (e.g. Middle East), the summary does not falsely attribute that action
    to Theatre B (e.g. South China Sea / China).
    Preserves legitimate multi-theatre articles where separate actions are correctly
    attributed to their respective passages.
    """
    issues: list[str] = []
    headline_and_body = f"{summary.headline} {summary.what_happened}"
    headline_and_body_lower = headline_and_body.lower()

    action_patterns: dict[str, re.Pattern[str]] = {
        "aircraft_carrier": re.compile(
            r"\b(aircraft\s+carrier(?:\s+strike\s+group)?|carrier\s+strike\s+group|carrier\s+deployment|"
            r"deployed\s+(?:a\s+)?(?:new\s+)?carrier)\b",
            re.I,
        ),
        "troop_evacuation": re.compile(
            r"\b(evacuat\w*\s+(?:more\s+)?(?:troops|diplomats|diplomatic\s+staff|civilians|citizens))\b",
            re.I,
        ),
        "strike_options": re.compile(
            r"\b(preparing\s+strike\s+options|strike\s+options\s+against)\b",
            re.I,
        ),
    }

    theatres: dict[str, dict[str, re.Pattern[str]]] = {
        "middle_east": {
            "terms": re.compile(
                r"\b(middle\s+east|persian\s+gulf|strait\s+of\s+hormuz|red\s+sea|iran|israel|gaza|"
                r"lebanon|syria|yemen|iraq|tehran|tel\s+aviv|netanyahu)\b",
                re.I,
            ),
            "actors": re.compile(
                r"\b(iran|israel|netanyahu|hezbollah|houthis?|hamas|irgc|idf)\b",
                re.I,
            ),
        },
        "indo_pacific": {
            "terms": re.compile(
                r"\b(south\s+china\s+sea|taiwan\s+strait|east\s+china\s+sea|philippine\s+sea|taiwan|"
                r"philippines|manila|taipei|indo-pacific)\b",
                re.I,
            ),
            "actors": re.compile(
                r"\b(china|chinese|beijing|pla|plan|taiwan|philippines)\b",
                re.I,
            ),
        },
        "eastern_europe": {
            "terms": re.compile(
                r"\b(ukraine|black\s+sea|sea\s+of\s+azov|crimea|donbas|kharkiv|kherson|kursk|belgorod|"
                r"rostov|dnipro|kyiv|moscow)\b",
                re.I,
            ),
            "actors": re.compile(
                r"\b(russia|russian|ukraine|ukrainian|zelenskyy|putin)\b",
                re.I,
            ),
        },
        "korean_peninsula": {
            "terms": re.compile(
                r"\b(korean\s+peninsula|north\s+korea|south\s+korea|pyongyang|seoul|dmz)\b",
                re.I,
            ),
            "actors": re.compile(
                r"\b(north\s+korea|south\s+korea|dprk|rok|pyongyang|seoul)\b",
                re.I,
            ),
        },
    }

    theatre_names = {
        "middle_east": "Middle East",
        "indo_pacific": "Indo-Pacific / South China Sea",
        "eastern_europe": "Eastern Europe",
        "korean_peninsula": "Korean Peninsula",
    }

    passages = [
        p.strip()
        for p in re.split(r"(?:\n\s*\n|\*\*\*|\.\s+(?=[A-Z]))", source_text)
        if len(p.strip()) > 20
    ]

    for action_key, action_pat in action_patterns.items():
        if not action_pat.search(headline_and_body_lower):
            continue

        source_action_theatres: set[str] = set()
        for p in passages:
            if action_pat.search(p):
                for th_key, th_data in theatres.items():
                    if th_data["terms"].search(p):
                        source_action_theatres.add(th_key)

        if not source_action_theatres:
            continue

        summary_action_theatres: set[str] = set()
        for th_key, th_data in theatres.items():
            in_loc = bool(
                summary.key_details.location
                and th_data["terms"].search(summary.key_details.location)
            )
            in_body_near_action = False
            for s in re.split(r"(?<=[.!?])\s+", headline_and_body):
                if action_pat.search(s) and th_data["terms"].search(s):
                    in_body_near_action = True
                    break

            actor_performing = False
            for a in summary.key_details.actors:
                if th_data["actors"].search(a):
                    actor_performing = True
                    break

            if (
                in_loc
                or in_body_near_action
                or (actor_performing and th_key not in source_action_theatres)
            ):
                summary_action_theatres.add(th_key)

        mismatched_theatres = summary_action_theatres - source_action_theatres
        if mismatched_theatres:
            for bad_th in mismatched_theatres:
                src_names = ", ".join(theatre_names[t] for t in source_action_theatres)
                action_name = action_key.replace("_", " ")
                issues.append(
                    f"Cross-theatre conflation: action '{action_name}' reported in source "
                    f"for {src_names} was incorrectly attributed to {theatre_names[bad_th]}"
                )

    return issues


_DISTINCT_COMMODITIES: dict[str, re.Pattern[str]] = {
    "LNG": re.compile(r"\b(lng|liquefied\s+natural\s+gas)\b", re.I),
    "Natural Gas": re.compile(r"\b(natural\s+gas|pipeline\s+gas)\b", re.I),
    "Diesel": re.compile(r"\b(diesel|diesel\s+fuel)\b", re.I),
    "Crude Oil": re.compile(r"\b(crude\s+oil|petroleum)\b", re.I),
    "Coal": re.compile(r"\b(coal|thermal\s+coal|coking\s+coal)\b", re.I),
    "Uranium": re.compile(r"\b(uranium|nuclear\s+fuel)\b", re.I),
    "Grain": re.compile(r"\b(grain|wheat|corn|barley)\b", re.I),
}


def check_commodity_integrity(
    summary: EventSummary,
    event: EventCluster,
    source_lower: str,
) -> list[str]:
    """Validate that commodities asserted in summary match the reporting and source text.

    Detects commodity substitutions (e.g. asserting LNG when reporting is about diesel),
    unsupported commodities absent from source, and headline/body commodity contradictions.
    """
    issues: list[str] = []
    headline_and_body = f"{summary.headline} {summary.what_happened}"
    if summary.key_details.action:
        headline_and_body += f" {summary.key_details.action}"

    # Identify which commodities are asserted in summary
    asserted_commodities: dict[str, str] = {}
    for comm_name, pat in _DISTINCT_COMMODITIES.items():
        m = pat.search(headline_and_body)
        if m:
            asserted_commodities[comm_name] = m.group(0)

    # Identify which commodities are present in event title / cluster title
    title_text = " ".join([event.title] + [art.title for art in event.articles]).lower()
    title_commodities: set[str] = set()
    for comm_name, pat in _DISTINCT_COMMODITIES.items():
        if pat.search(title_text):
            title_commodities.add(comm_name)

    for comm_name, matched_text in asserted_commodities.items():
        pat = _DISTINCT_COMMODITIES[comm_name]
        in_source = bool(pat.search(source_lower))
        if not in_source:
            issues.append(
                f"Commodity '{matched_text}' asserted in summary is not present in source text"
            )
            # If the title was about another distinct commodity, flag explicit mismatch
            mismatched_titles = [tc for tc in title_commodities if tc != comm_name]
            if mismatched_titles:
                for tc in mismatched_titles:
                    issues.append(
                        f"Commodity mismatch: headline/source reporting concerns '{tc}' "
                        f"but summary asserts '{comm_name}'"
                    )

    return issues


def validate_summary_grounding(
    summary: EventSummary,
    event: EventCluster,
) -> GroundingValidationResult:
    """Deterministically validate summary against supplied source articles."""
    # 1. Assemble source evidence corpus
    corpus_items: list[str] = []
    article_text_items: list[str] = []
    publication_years: set[str] = set()
    for art in event.articles:
        corpus_items.append(art.title)
        article_text_items.append(art.title)
        if art.snippet:
            corpus_items.append(art.snippet)
            article_text_items.append(art.snippet)
        corpus_items.append(art.source_name)
        if art.author:
            corpus_items.append(art.author)
        if art.published_at:
            publication_years.add(str(art.published_at.year))
        # Include URL slug components for named entities in URLs
        corpus_items.append(art.canonical_url.replace("-", " ").replace("/", " ").replace("_", " "))

    source_text = " ".join(corpus_items)
    article_text = " ".join(article_text_items)
    source_lower = source_text.lower()
    source_tokens = set(re.findall(r"\b[a-zA-Z0-9_\u00C0-\u017F-]+\b", source_lower))

    unsupported_entities: list[str] = []
    unsupported_claims: list[str] = []

    # 2. Check explicit insufficient evidence output
    what_lower = summary.what_happened.lower()
    headline_lower = summary.headline.lower()
    if (
        "insufficient evidence" in what_lower
        or "insufficient reporting" in what_lower
        or "not enough information" in what_lower
        or "no information provided" in what_lower
        or "no information in the snippet" in what_lower
        or "no information in" in what_lower
        or "no headline provided" in headline_lower
        or headline_lower.startswith("no headline")
    ):
        return GroundingValidationResult(
            is_valid=False,
            quarantine_reason="Source reporting contains insufficient evidence for factual summary",
        )

    # 2b. Check if what_happened merely echoes the headline without substantive reporting
    norm_head = re.sub(r"[^a-zA-Z0-9]+", " ", summary.headline).strip().lower()
    norm_what = re.sub(r"[^a-zA-Z0-9]+", " ", summary.what_happened).strip().lower()
    if not summary.claims and (
        norm_what == norm_head
        or (norm_what.startswith(norm_head) and len(norm_what) < len(norm_head) + 25)
    ):
        return GroundingValidationResult(
            is_valid=False,
            quarantine_reason=(
                "Summary simply repeats headline without substantive factual narrative"
            ),
        )

    # 2c. Check for empty, placeholder, or trivially brief narrative (< 3 words)
    what_words = [
        w for w in re.findall(r"\b[a-zA-Z0-9_\u00C0-\u017F-]+\b", norm_what) if len(w) >= 2
    ]
    if len(what_words) < 3:
        return GroundingValidationResult(
            is_valid=False,
            quarantine_reason=(
                f"Summary text contains insufficient narrative detail ({len(what_words)} words)"
            ),
        )

    # 2d. Check for generic strategic commentary without specific event facts
    if (
        not summary.key_details.actors
        and not summary.key_details.action
        and not summary.key_details.location
    ):
        generic_commentary_cues = (
            "strategic analysis examines",
            "commentary discusses",
            "analysis explores",
            "lessons should inform",
            "highlights the evolving nature",
        )
        if any(phrase in norm_what for phrase in generic_commentary_cues):
            return GroundingValidationResult(
                is_valid=False,
                quarantine_reason=(
                    "Generic strategic commentary without specific event actors, "
                    "actions, or locations"
                ),
            )

    # Every rendered summary field can carry a factual claim.
    headline_and_body = " ".join(
        part
        for part in (
            summary.headline,
            summary.what_happened,
            summary.significance or "",
            *summary.key_details.actors,
        )
        if part
    )
    significance_text = summary.significance or ""
    significance_concepts = (
        (
            "grain",
            re.compile(r"\b(?:grain|wheat|corn|cereal)\b", re.I),
            re.compile(r"\b(?:grain|wheat|corn|cereal)\b", re.I),
        ),
        (
            "fertilizer supply",
            re.compile(r"\bfertili[sz]ers?\b", re.I),
            re.compile(r"\bfertili[sz]ers?\b", re.I),
        ),
        (
            "global trade",
            re.compile(r"\b(?:global|international|world(?:wide)?)\s+(?:trade|commerce)\b", re.I),
            re.compile(
                r"\b(?:global|international|world(?:wide)?)\s+(?:trade|commerce)\b|"
                r"\btrade routes?\b|\bmaritime trade\b",
                re.I,
            ),
        ),
    )
    for claim_name, summary_pattern, source_pattern in significance_concepts:
        if summary_pattern.search(significance_text) and not source_pattern.search(source_text):
            unsupported_claims.append(
                f"Unsupported material significance: source does not substantiate {claim_name}"
            )
    # Officeholder titles must be attached to the named person in the source itself.
    # Mentioning a person elsewhere in the article is not evidence for their office.
    officeholder_roles = (
        r"chancellor|prime minister|president|vice president|foreign secretary|"
        r"defen[cs]e secretary|interior minister|finance minister|minister of finance|"
        r"foreign minister|home secretary|secretary of state|secretary of defense|"
        r"secretary of defence|chief of defense|chief of defence|governor"
    )
    role_claim = re.compile(
        rf"(?i:\b(?P<role>{officeholder_roles})\b)\s+"
        r"(?P<name>[A-Z][a-z][\w.'’\-]*(?:\s+[A-Z][a-z][\w.'’\-]*){0,2})\b"
    )
    for role_match in role_claim.finditer(headline_and_body):
        person = role_match.group("name").casefold()
        role = role_match.group("role")
        source_supports_role = any(
            re.search(rf"\b{re.escape(role)}\b", sentence, re.I)
            and re.search(rf"\b{re.escape(person)}\b", sentence, re.I)
            for sentence in re.split(r"(?<=[.!?])\s+", source_text)
        )
        if not source_supports_role:
            unsupported_claims.append(
                f"Unsupported officeholder attribution: source does not identify {person.title()} "
                f"as {role}"
            )
    event_source_names = [art.source_name for art in event.articles if art.source_name]
    for raw_actor in summary.key_details.actors:
        actor_clean = raw_actor.strip()
        if not actor_clean:
            continue

        # Publishers and news outlets must never be listed as event actors
        if is_publisher_or_outlet(actor_clean, event_source_names):
            unsupported_entities.append(actor_clean)
            continue

        # Strip prefixes
        stripped = actor_clean
        for prefix_pat in _PREFIX_TITLES:
            stripped = prefix_pat.sub("", stripped).strip()

        stripped_lower = stripped.lower()
        if stripped_lower in _GENERIC_ACTOR_WORDS or stripped_lower in source_lower:
            # Check if actor is an adversary/topic from an intro list but absent from narrative
            if stripped_lower not in _GENERIC_ACTOR_WORDS:
                intro_list_pat = re.compile(
                    r"\b(?:challengers?|threats?|adversaries|competitors|topics?):\s*[^.]*\b"
                    + re.escape(stripped_lower)
                    + r"\b",
                    re.I,
                )
                if intro_list_pat.search(source_lower):
                    outside_intro = intro_list_pat.sub("", source_lower)
                    if not re.search(rf"\b{re.escape(stripped_lower)}\b", outside_intro):
                        unsupported_entities.append(actor_clean)
                        unsupported_claims.append(
                            f"Actor mismatch: '{actor_clean}' only appears in introductory "
                            f"overview topic list and does not participate in the described event"
                        )
                        continue
            if stripped_lower in _GENERIC_ACTOR_WORDS or stripped_lower in source_lower:
                continue

        if stripped_lower in _DEMONYM_SYNONYMS:
            if any(
                syn in source_lower or syn in source_tokens
                for syn in _DEMONYM_SYNONYMS[stripped_lower]
            ):
                continue

        # Tokenize actor into substantive words
        actor_tokens = [
            tok.lower()
            for tok in re.findall(r"\b[a-zA-Z0-9_\u00C0-\u017F-]+\b", stripped)
            if len(tok) >= 3 and tok.lower() not in _STOP_WORDS
        ]

        if not actor_tokens:
            continue

        # If all tokens are generic collective or branch words, accept
        if all(
            tok in _GENERIC_ACTOR_WORDS or tok in _GENERIC_BRANCH_DESCRIPTORS
            for tok in actor_tokens
        ):
            continue

        # Extract substantive entity tokens (excluding generic branch words)
        entity_tokens = [tok for tok in actor_tokens if tok not in _GENERIC_BRANCH_DESCRIPTORS]
        if not entity_tokens:
            continue

        # Verify whether the substantive entity tokens appear in source (including demonyms)
        entity_grounded = any(
            tok in source_tokens
            or tok in source_lower
            or (
                tok in _DEMONYM_SYNONYMS
                and any(
                    syn in source_tokens or syn in source_lower for syn in _DEMONYM_SYNONYMS[tok]
                )
            )
            for tok in entity_tokens
        )

        if not entity_grounded:
            unsupported_entities.append(actor_clean)
            continue

        # If actor pairs entity with military branch, verify source provides military context
        if any(tok in {"military", "army", "navy", "forces", "strike"} for tok in actor_tokens):
            is_entity_in_military_context = False
            for art in event.articles:
                snip_lower = (art.snippet or "").lower()
                for e_tok in entity_tokens:
                    syns = _DEMONYM_SYNONYMS.get(e_tok, (e_tok,))
                    if any(s in snip_lower for s in syns):
                        if re.search(
                            r"\b(military|forces|army|navy|air|troops|soldiers|fighters|jets|"
                            r"defence|defense|exercise|drill|combat|strike|strikes|attack|attacks|"
                            r"war|battle|weapons|logistics|missile|drone|patrol|maneuver|maneuvers)\b",
                            snip_lower,
                        ):
                            is_entity_in_military_context = True
                            break
                    elif any(s in art.title.lower() for s in syns):
                        is_entity_in_military_context = True
                        break
                if is_entity_in_military_context:
                    break

            if not is_entity_in_military_context:
                unsupported_entities.append(actor_clean)
                continue

    # 4. Check for high-profile acronyms in headline and summary
    headline_and_body = f"{summary.headline} {summary.what_happened}"
    for acronym in _PROTECTED_ACRONYMS:
        acronym_pat = re.compile(rf"\b{re.escape(acronym)}\b")
        if acronym_pat.search(headline_and_body):
            if acronym.lower() not in source_tokens and acronym.lower() not in source_lower:
                if acronym not in unsupported_entities:
                    unsupported_entities.append(acronym)

    # 4a. Check for ungrounded technical / vessel acronym expansions in headline and body
    for m in re.finditer(r"\b([A-Z0-9]{2,6})\s*\(([^)]+)\)", headline_and_body):
        acro = m.group(1).strip()
        exp = m.group(2).strip()
        exp_words = [
            w.lower() for w in re.findall(r"\b[A-Za-z]{3,}\b", exp) if w.lower() not in _STOP_WORDS
        ]
        if exp_words and not all(w in source_lower for w in exp_words):
            unsupported_claims.append(
                f"Invented technical acronym expansion '{exp}' for '{acro}' "
                f"not supported by source text"
            )

    for m in re.finditer(
        r"((?:[A-Z][a-zA-Z]+(?:,\s*|\s+)){1,5}[A-Z][a-zA-Z]+)\s*\(([A-Z0-9]{2,6})\)",
        headline_and_body,
    ):
        exp = m.group(1).strip()
        acro = m.group(2).strip()
        exp_words = [
            w.lower() for w in re.findall(r"\b[A-Za-z]{3,}\b", exp) if w.lower() not in _STOP_WORDS
        ]
        if exp_words and not all(w in source_lower for w in exp_words):
            unsupported_claims.append(
                f"Invented technical acronym expansion '{exp}' for '{acro}' "
                f"not supported by source text"
            )

    # 4b. Check for hallucinated vessel acronyms (e.g. LSTH when source has LSM)
    for vessel_acro in ("LSTH", "LST"):
        if re.search(rf"\b{vessel_acro}\b", headline_and_body):
            if not re.search(rf"\b{vessel_acro}\b", source_text, re.I):
                if vessel_acro not in unsupported_entities:
                    unsupported_entities.append(vessel_acro)

    # 5. Check for titled person names in headline and body
    for match in _TITLED_PERSON_PATTERN.finditer(headline_and_body):
        person_name = match.group(1).strip()
        name_tokens = [
            t.lower() for t in person_name.split() if len(t) >= 3 and t.lower() not in _STOP_WORDS
        ]
        if name_tokens and not any(t in source_tokens or t in source_lower for t in name_tokens):
            if person_name not in unsupported_entities:
                unsupported_entities.append(person_name)

    # 5a. Check for fabricated kinetic military actions / strikes absent from source
    _KINETIC_ACTION_PATTERN = re.compile(
        r"\b(airstrike|air strike|airstrikes|air strikes|bombing|bombings|shelling|"
        r"artillery barrage|missile strike|missile strikes|air assault|ground offensive|"
        r"counteroffensive|suicide bombing|drone strike|drone strikes)\b",
        re.I,
    )
    _KINETIC_SOURCE_TERMS = re.compile(
        r"\b(strike|strikes|struck|attack|attacks|attacked|bomb|bombs|bombed|bombing|"
        r"missile|missiles|shell|shells|shelled|shelling|assault|combat|offensive|drone|"
        r"killed|wounded|casualties|shot down|downed|warhead|air force)\b",
        re.I,
    )
    if _KINETIC_ACTION_PATTERN.search(headline_and_body):
        if not _KINETIC_SOURCE_TERMS.search(source_text):
            unsupported_claims.append(
                "Unsupported kinetic military action or strike not substantiated by source text"
            )

    # 5b. Check for vague disagreement phrases without specific attribution in source
    _VAGUE_DISAGREEMENT_PATTERN = re.compile(
        r"\b(disputed by some|critics disagree|"
        r"some (?:critics|observers|analysts) (?:disagree|dispute|question)|"
        r"this is disputed by some|contested by some|some dispute)\b",
        re.I,
    )
    if _VAGUE_DISAGREEMENT_PATTERN.search(headline_and_body):
        dispute_in_source = bool(
            re.search(
                r"\b(dispute|disputed|critics|criticism|disagree|controversy|contested|opponents)\b",
                source_lower,
            )
        )
        if not dispute_in_source:
            unsupported_claims.append(
                "Vague disagreement phrase ('disputed by some' / 'critics disagree') "
                "without named disagreement in source text"
            )

    # 5b-2. Check for assertions of unknown details without source evidence of uncertainty
    # Absence of evidence must not be converted into an assertion that something is unknown.
    _UNKNOWN_ASSERTION_PATTERN = re.compile(
        r"\b(?:remains?\s+unknown|is\s+unknown|was\s+unknown|"
        r"cause\s+(?:is|was|remains)\s+unknown|unclear\s+(?:who|how|why|what|when)|"
        r"not\s+(?:immediately\s+)?known|unknown\s+(?:cause|origin|identity|perpetrator|actor)|"
        r"it\s+is\s+unknown\b)\b",
        re.I,
    )
    _SOURCE_UNCERTAINTY_CUES = re.compile(
        r"\b(?:unknown|unclear|unconfirmed|unverified|investigating|under\s+investigation|"
        r"not\s+(?:immediately\s+)?(?:known|clear|verified|confirmed)|undetermined|"
        r"yet\s+to\s+be\s+determined)\b",
        re.I,
    )
    if _UNKNOWN_ASSERTION_PATTERN.search(headline_and_body):
        if not _SOURCE_UNCERTAINTY_CUES.search(source_text):
            unsupported_claims.append(
                "Asserts a detail is unknown or unclear without explicit source evidence "
                "of uncertainty"
            )

    # 5c. Check for unwarranted claims of operational deployment or active service
    _DEPLOYMENT_CLAIMS = re.compile(
        r"\b(operationally deployed|entered service|in active service|deployed to the front|"
        r"combat-ready deployment|now deployed|fielded in combat)\b",
        re.I,
    )
    _DEV_OR_UNVEIL_TERMS = re.compile(
        r"\b(unveil\w*|reveal\w*|test\w*|demo\w*|prototyp\w*|concept\w*|develop\w*)\b",
        re.I,
    )
    if _DEPLOYMENT_CLAIMS.search(headline_and_body):
        has_deployment_terms = bool(
            re.search(
                r"\b(operationally deployed|entered service|active service|"
                r"in service|combat service)\b",
                source_lower,
            )
        )
        if _DEV_OR_UNVEIL_TERMS.search(source_text) and not has_deployment_terms:
            unsupported_claims.append(
                "Unwarranted claim of operational deployment or active service "
                "not supported by source text"
            )

    # 5c2. Check for unwarranted claims of system unveiling, reveal, rollout, or debut
    _UNVEIL_SUMMARY_PATTERN = re.compile(
        r"\b(unveils?|unveiled|unveiling|rolls?\s+out|rolled\s+out|reveals?|revealed|debuts?|debuted)\b",
        re.I,
    )
    _UNVEIL_SOURCE_TERMS = re.compile(
        r"\b(unveil\w*|reveal\w*|roll\w*\s+out|debut\w*|demonstrat\w*|showcas\w*|present\w*)\b",
        re.I,
    )
    if _UNVEIL_SUMMARY_PATTERN.search(headline_and_body):
        if not _UNVEIL_SOURCE_TERMS.search(source_lower):
            unsupported_claims.append(
                "Claim of system unveiling, rollout, or reveal not substantiated by source text"
            )

    # 5c3. Check for analysis/commentary misrepresented as official military decisions
    _OFFICIAL_ANNOUNCEMENT_TERMS = re.compile(
        r"\b(?:announc\w*|unveiled\s+plans?|formally\s+plans?|official\s+decision|"
        r"statement|press\s+release|ordered)\b",
        re.I,
    )
    _ANNOUNCEMENT_SUMMARY_PAT = re.compile(
        r"\b(?:announced?(?:\s+plans?)?|ordered|awarded contract|unveiled|approved)\b",
        re.I,
    )
    _ANALYSIS_DISCLAIMER_TERMS = re.compile(
        r"\b(?:analyst\s+argues|one\s+analyst|op-ed|commentary|opinion|"
        r"lessons?\s+should\s+inform|lessons?\s+from)\b",
        re.I,
    )
    if _ANALYSIS_DISCLAIMER_TERMS.search(source_lower) and not _OFFICIAL_ANNOUNCEMENT_TERMS.search(
        source_lower
    ):
        if _ANNOUNCEMENT_SUMMARY_PAT.search(headline_and_body):
            if not re.search(
                r"\b(?:analyst|essay|commentary|op-ed|article|argument|author|opinion)\b",
                headline_and_body,
                re.I,
            ):
                unsupported_claims.append(
                    "Analytical commentary or opinion piece misrepresented as official "
                    "military decision or announcement"
                )

    # 5d. Check for invented denials, rebuttals, or refutations absent from source
    _DENIAL_SUMMARY_PATTERN = re.compile(
        r"\b(denies|denied|denying|denial|rebuts|rebutted|rebuttal|rejects|rejected|refutes|refuted)\b",
        re.I,
    )
    _DENIAL_SOURCE_TERMS = re.compile(
        r"\b(denies|denied|denying|denial|rebuts|rebutted|rebuttal|rejects|rejected|refutes|refuted|"
        r"dismisses|dismissed|dismissing|unfounded|false claim|no truth|denounce|denounced)\b",
        re.I,
    )
    if _DENIAL_SUMMARY_PATTERN.search(headline_and_body):
        if not _DENIAL_SOURCE_TERMS.search(source_lower):
            unsupported_claims.append(
                "Denial or rebuttal asserted in summary without denial evidence in source text"
            )

    # 5e. Check for ungrounded nuclear capability, weapons, or policy claims absent from source
    _NUCLEAR_SUMMARY_PATTERN = re.compile(
        r"\b(nuclear|atomic|warhead|warheads|enrichment)\b",
        re.I,
    )
    _NUCLEAR_SOURCE_TERMS = re.compile(
        r"\b(nuclear|atomic|warhead|warheads|enrichment)\b",
        re.I,
    )
    if _NUCLEAR_SUMMARY_PATTERN.search(headline_and_body):
        if not _NUCLEAR_SOURCE_TERMS.search(source_lower):
            unsupported_claims.append(
                "Nuclear capability, weapon, or policy claim asserted "
                "without nuclear reference in source text"
            )

    # 5e2. Check for ungrounded nuclear weapon upgrade announcements or plans
    _NUCLEAR_ANNOUNCEMENT_SUMMARY_PAT = re.compile(
        r"\b(?:announces?(?:\s+plans?)?|announced|plans?\s+to\s+upgrade|announces\s+upgrade)\b.*\b(?:nuclear|weapons?)\b|"
        r"\b(?:nuclear(?:\s+weapons?)?\s+(?:upgrade|modernization)\s+plans?)\b|"
        r"\b(?:plans\s+to\s+upgrade\s+nuclear)\b",
        re.I,
    )
    _OFFICIAL_ANNOUNCEMENT_SOURCE_TERMS = re.compile(
        r"\b(?:announc\w*|unveiled\s+plans?|formally\s+plans?|official\s+decision|statement|press\s+release|ordered)\b",
        re.I,
    )
    if _NUCLEAR_ANNOUNCEMENT_SUMMARY_PAT.search(headline_and_body):
        if not _OFFICIAL_ANNOUNCEMENT_SOURCE_TERMS.search(source_lower):
            unsupported_claims.append(
                "Nuclear weapons upgrade plan or announcement asserted "
                "without announcement evidence in source text"
            )

    # 5e3. Check for invented silence, non-response, or refusal to comment absent from source
    _SILENCE_SUMMARY_PATTERN = re.compile(
        r"\b(?:silence|remained\s+silent|maintains?\s+silence|did\s+not\s+comment|"
        r"refused\s+to\s+comment|declined\s+to\s+comment|no\s+comment\s+from|"
        r"has\s+not\s+commented|has\s+not\s+responded|without\s+response|"
        r"unresponsive|no\s+public\s+statement|silent\s+on\s+the\s+matter)\b",
        re.I,
    )
    _SILENCE_SOURCE_TERMS = re.compile(
        r"\b(?:silence|silent|no\s+comment|declined\s+to\s+comment|refused\s+to\s+comment|"
        r"unresponsive|not\s+responded|did\s+not\s+respond|not\s+commented|"
        r"did\s+not\s+comment|without\s+response)\b",
        re.I,
    )
    if _SILENCE_SUMMARY_PATTERN.search(headline_and_body):
        if not _SILENCE_SOURCE_TERMS.search(source_lower):
            unsupported_claims.append(
                "Claim of official silence or absence of comment not substantiated by source text"
            )

    # 5e4. Check for invented reactions, concern, alarm, or international outcry absent from source
    _REACTION_SUMMARY_PATTERN = re.compile(
        r"\b(?:sparking|prompting|raising|triggering|causing|fueling|stoking|amid)\s+"
        r"(?:growing\s+|widespread\s+|heightened\s+|deep\s+)?"
        r"(?:western|international|allied|regional|global|domestic)?\s*"
        r"(?:concern|alarm|fears?|outcry|anxiety|unease|condemnation)\b|"
        r"\b(?:western|international|allied|regional|global)\s+"
        r"(?:concern|alarm|outcry|anxiety|unease|condemnation)\b",
        re.I,
    )
    _REACTION_SOURCE_TERMS = re.compile(
        r"\b(?:concern|alarm|fear|fears|anxiety|unease|outcry|condemn|condemnation|condemned|"
        r"critic|critics|backlash|condemns)\b",
        re.I,
    )
    if _REACTION_SUMMARY_PATTERN.search(headline_and_body):
        if not _REACTION_SOURCE_TERMS.search(source_lower):
            unsupported_claims.append(
                "Claim of Western or international concern, alarm, or reaction "
                "not substantiated by source text"
            )

    # 5e5. Check for contract role misattribution (TOTE Services vs direct Navy award to Hanwha)
    if "tote" in source_lower and "hanwha" in source_lower:
        direct_navy_award_pat = re.compile(
            r"\b(?:u\.?s\.?\s+)?navy\s+(?:directly\s+)?(?:awarded|awards|contracted)\s+hanwha\b|"
            r"\bhanwha\s+(?:received|won|secured)\s+(?:a\s+)?contract\s+from\s+(?:the\s+)?(?:u\.?s\.?\s+)?navy\b|"
            r"\bcontract\s+awarded\s+by\s+(?:the\s+)?(?:u\.?s\.?\s+)?navy\s+to\s+hanwha\b|"
            r"\bnavy\s+(?:awarded|awards)\s+(?:a\s+)?(?:\$?\d+[\d,.]*(?:\s*[bm]illion)?\s+)?contract\s+to\s+hanwha\b",
            re.I,
        )
        has_direct_award = (
            direct_navy_award_pat.search(headline_and_body)
            and "tote" not in headline_and_body.lower()
        )
        if has_direct_award:
            unsupported_claims.append(
                "Contract role misattribution: TOTE Services selected Hanwha Defense USA "
                "for the Navy and Marine Corps LSM programme; do not describe the Navy as "
                "directly awarding Hanwha the contract"
            )

    # 5e6. Check for premature missile delivery on Foreign Military Sale approvals
    if (
        "state department" in source_lower
        or "foreign military sale" in source_lower
        or "potential sale" in source_lower
        or "dsca" in source_lower
    ) and "patriot" in source_lower:
        source_has_delivery = bool(
            re.search(
                r"\b(?:missiles?\s+(?:have\s+been\s+)?delivered|delivered\s+\d+|"
                r"delivery\s+completed|arrived\s+in|received\s+\d+\s+patriot)\b",
                source_lower,
            )
        )
        if not source_has_delivery:
            delivery_assertion_pat = re.compile(
                r"\b(?:delivered\s+(?:\d+\s+)?patriot|patriot\s+missiles?\s+delivered|"
                r"completed\s+(?:the\s+)?sale\s+of\s+500\s+patriot|supplies\s+500\s+patriot|"
                r"us\s+delivers\s+500\s+patriot|transferred\s+500\s+patriot|delivers\s+patriot)\b",
                re.I,
            )
            if delivery_assertion_pat.search(headline_and_body):
                unsupported_claims.append(
                    "Unsupported delivery/transaction status: State Department approved "
                    "a potential Foreign Military Sale of Patriot missiles; missiles have "
                    "not been delivered or the transaction completed"
                )

    # 5e7. Check Black Sea vessel casualty data provenance (Ambrey / IMO / Financial Times)
    if "black sea" in source_lower and re.search(
        r"\b(?:vessel\s+casualties|merchant\s+ships?\s+(?:hit|damaged|attacked))\b",
        headline_and_body,
        re.I,
    ):
        if ("ambrey" in source_lower or "imo" in source_lower) and not re.search(
            r"\b(?:ambrey|imo|financial\s+times|ft)\b",
            headline_and_body,
            re.I,
        ):
            unsupported_claims.append(
                "Missing data provenance: Black Sea vessel casualty statistics must retain "
                "Financial Times attribution and Ambrey/IMO data provenance"
            )

    # 5f. Check placeholder leakage in rendered headline and body
    if _LEAKAGE_PATTERN.search(headline_and_body):
        unsupported_claims.append(
            "Placeholder leakage ('Unspecified in reporting') detected in summary text"
        )

    # 5g. Check location grounding in key_details.location
    if summary.key_details.location:
        grounded_loc = ground_event_location(summary.key_details.location, event, source_text)
        if grounded_loc is None and summary.key_details.location.strip().lower() not in {
            "unspecified",
            "unspecified in reporting",
            "unknown",
            "none",
            "null",
        }:
            unsupported_claims.append(
                f"Location '{summary.key_details.location}' is not a supported event location"
            )
        else:
            loc_clean = summary.key_details.location.strip().lower()
            intro_list_pat = re.compile(
                r"\b(?:challengers?|threats?|adversaries|competitors):\s*[^.]*\b"
                + re.escape(loc_clean)
                + r"\b",
                re.I,
            )
            if intro_list_pat.search(source_lower):
                outside_intro = intro_list_pat.sub("", source_lower)
                if not re.search(rf"\b{re.escape(loc_clean)}\b", outside_intro):
                    unsupported_claims.append(
                        f"Location mismatch: '{summary.key_details.location}' does not match "
                        f"the event theater described in the summary"
                    )

    # 5h. High-consequence diplomacy verification (truce / ceasefire / peace deals)
    _DIPLOMACY_TRUCE_PATTERN = re.compile(
        r"\b(truce|ceasefire|peace\s+talks?|peace\s+deal|peace\s+accord|peace\s+agreement|armistice)\b",
        re.I,
    )
    if _DIPLOMACY_TRUCE_PATTERN.search(headline_and_body):
        if not _DIPLOMACY_TRUCE_PATTERN.search(source_lower):
            unsupported_claims.append(
                "Truce or ceasefire asserted without source text substantiation"
            )
        else:
            truce_city_match = re.search(
                r"\b(?:truce|ceasefire)\s+(?:begins?|takes? effect|announced|held|declared)?\s*"
                r"(?:in|for|at)\s+(?:the\s+)?(?:wartorn\s+)?(?:city\s+of\s+|town\s+of\s+)?([A-Z][a-z]+)\b",
                headline_and_body,
            )
            if truce_city_match:
                city_cand = truce_city_match.group(1)
                if city_cand.lower() not in _STOP_WORDS and city_cand.lower() not in source_lower:
                    unsupported_claims.append(
                        f"Truce city '{city_cand}' not substantiated by source text"
                    )

            # Check ungrounded negotiation or enforcement roles
            _CEASEFIRE_ROLE_PAT = re.compile(
                r"\b(?:negotiated\s+the\s+(?:ceasefire|truce)|"
                r"enforcing\s+the\s+(?:ceasefire|truce)|"
                r"implemented\s+the\s+(?:ceasefire|truce)|"
                r"[A-Z][a-z]+'s\s+(?:ceasefire|truce))\b",
                re.I,
            )
            if _CEASEFIRE_ROLE_PAT.search(headline_and_body):
                if not re.search(r"\b(negotiat\w*|enforc\w*|implement\w*)\b", source_lower):
                    unsupported_claims.append(
                        "Ceasefire role (negotiation, enforcement, or ownership) "
                        "unsupported by source text"
                    )

    # 5i. Check casualty and fatality assertions
    _CASUALTY_SUMMARY_PAT = re.compile(
        r"\b(\d+\s+(?:killed|dead|fatalities|casualties)|heavy casualties|mass casualties)\b",
        re.I,
    )
    _CASUALTY_SOURCE_PAT = re.compile(
        r"\b(killed|dead|fatal\w*|casualt\w*|death\w*|wounded|injur\w*)\b",
        re.I,
    )
    if _CASUALTY_SUMMARY_PAT.search(headline_and_body):
        if not _CASUALTY_SOURCE_PAT.search(source_lower):
            unsupported_claims.append(
                "Casualty or fatality figures asserted without source text substantiation"
            )

    # 5j. Contradictory attribution & allegation verification
    _SELF_BLAME_PAT = re.compile(
        r"\b(russia|russian(?:\s+officials?|\s+forces?|\s+government)?)\s+"
        r"(?:blamed|blames|accused|accuses|criticized|criticizes)\s+(?:moscow|russia)\b|"
        r"\b(united\s+states|us|american(?:\s+officials?|\s+government)?)\s+"
        r"(?:blamed|blames|accused|accuses|criticized|criticizes)\s+"
        r"(?:washington|united\s+states|us)\b|"
        r"\b(china|chinese(?:\s+officials?|\s+government)?)\s+"
        r"(?:blamed|blames|accused|accuses|criticized|criticizes)\s+(?:beijing|china)\b|"
        r"\b(ukraine|ukrainian(?:\s+officials?|\s+government)?)\s+"
        r"(?:blamed|blames|accused|accuses|criticized|criticizes)\s+(?:kyiv|ukraine)\b",
        re.I,
    )
    if _SELF_BLAME_PAT.search(headline_and_body):
        unsupported_claims.append(
            "Contradictory attribution: entity or state actor reported blaming itself "
            "or its own capital"
        )
    _BLAME_PAIR_PAT = re.compile(
        r"\b(Washington|Moscow|Russia|US|China|Beijing|Ukraine|Kyiv)\s+"
        r"(?:blamed|blames|accused|accuses)\s+"
        r"(Moscow|Washington|Russia|US|China|Beijing|Ukraine|Kyiv)\b",
        re.I,
    )
    for m in _BLAME_PAIR_PAT.finditer(headline_and_body):
        accuser = m.group(1).lower()
        accused = m.group(2).lower()
        blame_in_src = re.search(
            rf"\b([a-zA-Z]+)\s+(?:blamed|blames|accused|accuses)\s+{re.escape(accused)}\b",
            source_lower,
        )
        if blame_in_src:
            actual_accuser = blame_in_src.group(1).lower()
            accuser_syns = _DEMONYM_SYNONYMS.get(accuser, (accuser,))
            if actual_accuser not in accuser_syns and actual_accuser != accuser:
                unsupported_claims.append(
                    f"Attribution error: '{m.group(0)}' contradicts source text "
                    f"where '{actual_accuser}' made the allegation"
                )

    # 5k. Check ungrounded testing, demonstration, or operational interception claims
    _TESTING_DEMO_SUMMARY_PAT = re.compile(
        r"\b(?:demonstrated|tested\s+in\s+combat|proven\s+in\s+testing|live-fire\s+tested|"
        r"successfully\s+intercepted|passed\s+field\s+trials)\b",
        re.I,
    )
    _TESTING_DEMO_SOURCE_PAT = re.compile(
        r"\b(?:demonstrat\w*|tested|trials?|live-fire|intercepted|field\s+test\w*)\b",
        re.I,
    )
    if _TESTING_DEMO_SUMMARY_PAT.search(headline_and_body):
        if not _TESTING_DEMO_SOURCE_PAT.search(source_lower):
            unsupported_claims.append(
                "Testing, demonstration, or operational interception asserted "
                "without substantiation in source text"
            )

    # 5l. Validate quantitative claims (percentages, times, counts, magnitudes)
    quant_unsupported = _validate_quantitative_claims(
        headline_and_body,
        source_lower,
        source_text,
    )
    if quant_unsupported:
        unsupported_claims.extend(quant_unsupported)

    # 5l.1. Reject specific years/months asserted as event timing unless the article
    # itself contains them. Publication metadata may support its own year only.
    timing_text = summary.key_details.timing or ""
    timeline_text = " ".join(
        (
            summary.headline,
            summary.what_happened,
            significance_text,
            timing_text,
            summary.key_details.action or "",
        )
    )
    claimed_years = set(re.findall(r"\b(?:19|20)\d{2}\b", timeline_text))
    # URL slugs and publication metadata can contain years unrelated to an event's
    # chronology, so only title and article-text evidence can ground event years.
    supported_years = set(re.findall(r"\b(?:19|20)\d{2}\b", article_text))
    for year in sorted(claimed_years - supported_years):
        # An article's publication year is an allowed reporting-date fallback in
        # the dedicated timing field. It cannot substantiate chronology elsewhere.
        if (
            year in publication_years
            and year in re.findall(r"\b(?:19|20)\d{2}\b", timing_text)
            and year
            not in re.findall(
                r"\b(?:19|20)\d{2}\b",
                f"{summary.headline} {summary.what_happened} {summary.key_details.action or ''}",
            )
        ):
            continue
        unsupported_claims.append(
            f"Event timeline year '{year}' is not grounded in the article text"
        )

    # A weekday attached to an election is an event date claim, not something that
    # can be inferred from the briefing date or the article's publication weekday.
    weekday_names = "Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday"
    election_terms = r"(?:national\s+)?(?:election|vote|polls?)"
    weekday_election_claim = re.compile(
        rf"\b(?P<day>{weekday_names})\b[^.!?]{{0,50}}\b{election_terms}\b|"
        rf"\b{election_terms}\b[^.!?]{{0,50}}\b(?P<day_after>{weekday_names})\b",
        re.I,
    )
    for sentence in re.split(r"(?<=[.!?])\s+", timeline_text):
        match = weekday_election_claim.search(sentence)
        if not match:
            continue
        weekday = (match.group("day") or match.group("day_after") or "").lower()
        source_has_weekday_election = any(
            re.search(rf"\b{weekday}\b", source_sentence, re.I)
            and re.search(rf"\b{election_terms}\b", source_sentence, re.I)
            for source_sentence in re.split(r"(?<=[.!?])\s+", article_text)
        )
        if not source_has_weekday_election:
            unsupported_claims.append(
                f"Relative election date '{weekday.title()}' is not grounded in the article text"
            )

    # Preserve two high-signal exercise details when they are explicitly reported.
    # These checks apply only when the source names the fact, and do not require
    # generic summaries to reproduce incidental details.
    summary_text = (
        f"{summary.headline} {summary.what_happened} {significance_text} "
        f"{summary.key_details.action or ''}"
    )
    count_40_plus = r"(?:\b40\s*\+|\b40[- ]plus\b|\bmore than 40\b|\bover 40\b|\bforty[- ]plus\b)"
    uncrewed_terms = r"\b(?:uncrewed|unmanned|autonomous)\b"
    source_has_40_plus = bool(re.search(count_40_plus, source_lower, re.I))
    summary_has_40_plus = bool(re.search(count_40_plus, summary_text, re.I))
    source_counts_uncrewed = bool(
        re.search(
            rf"{count_40_plus}[^.!?]{{0,24}}{uncrewed_terms}\s+technolog\w*|"
            rf"{uncrewed_terms}\s+technolog\w*[^.!?]{{0,24}}{count_40_plus}",
            source_text,
            re.I,
        )
    )
    summary_counts_uncrewed = bool(
        re.search(
            rf"{count_40_plus}[^.!?]{{0,24}}{uncrewed_terms}\s+technolog\w*|"
            rf"{uncrewed_terms}\s+technolog\w*[^.!?]{{0,24}}{count_40_plus}",
            summary_text,
            re.I,
        )
    )
    if source_has_40_plus and not summary_has_40_plus:
        unsupported_claims.append(
            "Material omission: source reports more than 40 exercise technologies"
        )
    elif summary_has_40_plus and not source_has_40_plus:
        unsupported_claims.append("Unsupported detail: 40+ technologies are not stated in source")
    if summary_counts_uncrewed and not source_counts_uncrewed:
        unsupported_claims.append(
            "Unsupported detail: source does not state that 40+ technologies were all uncrewed"
        )
    source_has_asw = bool(re.search(r"\banti[- ]submarine\s+warfare\b|\bASW\b", source_text, re.I))
    summary_has_asw = bool(
        re.search(r"\banti[- ]submarine\s+warfare\b|\bASW\b", summary_text, re.I)
    )
    if source_has_asw and not summary_has_asw:
        unsupported_claims.append(
            "Material omission: source identifies an anti-submarine warfare component"
        )
    elif summary_has_asw and not source_has_asw:
        unsupported_claims.append(
            "Unsupported detail: anti-submarine warfare is not stated in source"
        )

    # A reference to Ukraine elsewhere in an article does not ground a causal
    # connection between Ukraine and a Russian diesel deal. Require the source to
    # state that relationship in a sentence that includes the relevant subject.
    causal_claim = re.compile(
        r"\b(?:diesel|fuel|imports?|deal|agreement|revenue|earnings?)\b.*?"
        r"\b(?:fund\w*|financ\w*|bankroll\w*|support\w*|benefit\w*|"
        r"help\w*|link\w*|tie\w*|connect\w*|cost\w*|erod\w*|weaken\w*|"
        r"remove\w*|strip\w*|because of|due to|in response to)\b.*?"
        r"\b(?:ukraine|ukrainian|kyiv)\b|"
        r"\b(?:ukraine|ukrainian|kyiv)\b.*?"
        r"\b(?:fund\w*|financ\w*|bankroll\w*|support\w*|benefit\w*|"
        r"help\w*|link\w*|tie\w*|connect\w*|cost\w*|erod\w*|weaken\w*|"
        r"remove\w*|strip\w*|because of|due to|in response to)\b.*?"
        r"\b(?:diesel|fuel|imports?|deal|agreement|revenue|earnings?)\b",
        re.I,
    )
    source_has_ukraine_causal_link = any(
        causal_claim.search(sentence) for sentence in re.split(r"(?<=[.!?])\s+", source_text)
    )
    summary_has_ukraine_causal_link = any(
        causal_claim.search(sentence) for sentence in re.split(r"(?<=[.!?])\s+", summary_text)
    )
    if summary_has_ukraine_causal_link and not source_has_ukraine_causal_link:
        unsupported_claims.append(
            "Unsupported causal claim: source does not link Ukraine to the Russian diesel deal"
        )
    if summary_has_ukraine_causal_link:
        source_analysis_attribution = any(
            causal_claim.search(sentence)
            and re.search(
                r"\b(?:according to|analysts?|experts?|researchers?|report from|study by)\b",
                sentence,
                re.I,
            )
            for sentence in re.split(r"(?<=[.!?])\s+", source_text)
        )
        summary_preserves_attribution = bool(
            re.search(
                r"\b(?:according to|analysts?|experts?|researchers?|assessment by|"
                r"report from|study by|ISW|Carnegie)\b",
                summary_text,
                re.I,
            )
        )
        if source_analysis_attribution and not summary_preserves_attribution:
            unsupported_claims.append(
                "Analytical consequence omits the source's expert attribution"
            )

    # A bilateral agreement is a stronger claim than one party easing restrictions.
    # Require the source to describe an agreement between both parties, in one sentence.
    diesel_agreement_claim = re.compile(
        r"\b(?:us|united states|washington)\b[^.!?]{0,100}"
        r"\b(?:russia|russian|moscow)\b[^.!?]{0,60}"
        r"\b(?:agree[ds]?|agreement|deal)\b|"
        r"\b(?:russia|russian|moscow)\b[^.!?]{0,100}"
        r"\b(?:us|united states|washington)\b[^.!?]{0,60}"
        r"\b(?:agree[ds]?|agreement|deal)\b",
        re.I,
    )
    source_diesel_agreement = any(
        diesel_agreement_claim.search(sentence)
        for sentence in re.split(r"(?<=[.!?])\s+", source_text)
    )
    summary_diesel_agreement = any(
        re.search(r"\b(?:us|united states|washington)\b", sentence, re.I)
        and re.search(r"\b(?:russia|russian|moscow)\b", sentence, re.I)
        and re.search(r"\b(?:agree[ds]?|agreement)\b", sentence, re.I)
        for sentence in re.split(r"(?<=[.!?])\s+", summary_text)
    )
    if summary_diesel_agreement and not source_diesel_agreement:
        unsupported_claims.append(
            "Unsupported transaction: source does not establish a bilateral US-Russia agreement"
        )

    # Normalize the same physical flow expressed as exporter→importer or
    # importer←exporter, and reject a summary that reverses the source direction.
    flow_patterns = {
        "russia_to_us": re.compile(
            r"(?:\brussia\b[^.!?]{0,50}\bexport\w*\b[^.!?]{0,50}\b(?:us|united states)\b|"
            r"\b(?:us|united states)\b[^.!?]{0,50}\bimport\w*\b[^.!?]{0,50}\brussia\b|"
            r"\b(?:us|united states)\b[^.!?]{0,50}\bimport\w*\b[^.!?]{0,50}\brussian\s+diesel\b|"
            r"\brussian\b[^.!?]{0,50}\bdiesel\b[^.!?]{0,50}\bimports?\b[^.!?]{0,30}\b(?:to|into)\s+(?:the\s+)?us\b)",
            re.I,
        ),
        "us_to_russia": re.compile(
            r"(?:\b(?:us|united states)\b[^.!?]{0,50}\bexport\w*\b[^.!?]{0,50}\brussia\b|"
            r"\brussia\b[^.!?]{0,50}\bimport\w*\b[^.!?]{0,50}\b(?:us|united states)\b|"
            r"\brussia\b[^.!?]{0,50}\bimport\w*\b[^.!?]{0,50}\bus\s+diesel\b|"
            r"\b(?:us|united states)\b[^.!?]{0,50}\b(?:sell|supply)\w*\b"
            r"[^.!?]{0,50}\bdiesel\b[^.!?]{0,30}\bto\s+russia\b)",
            re.I,
        ),
    }
    summary_flow = next(
        (direction for direction, pattern in flow_patterns.items() if pattern.search(summary_text)),
        None,
    )
    source_flows = {
        direction for direction, pattern in flow_patterns.items() if pattern.search(source_text)
    }
    if summary_flow and source_flows and summary_flow not in source_flows:
        unsupported_claims.append(
            "Reversed trade direction: summary changes the US-Russia diesel "
            "import/export relationship"
        )
    elif summary_flow and not source_flows:
        unsupported_claims.append(
            "Unsupported trade direction: source does not establish the stated "
            "US-Russia diesel flow"
        )

    # 5m. Check for malformed or incomplete clauses in what_happened
    malformed_clauses = detect_malformed_clauses(summary.what_happened)
    if malformed_clauses:
        unsupported_claims.extend(malformed_clauses)

    # 5n. Unattributed negative claims (e.g. "no breaches occurred", "no violations took place")
    _UNATTRIBUTED_NEGATIVE_PAT = re.compile(
        r"\b(?:there\s+have\s+been\s+no\s+breaches|no\s+breaches\s+(?:have\s+occurred|occurred|took\s+place)|"
        r"no\s+violations\s+(?:have\s+occurred|occurred|took\s+place))\b",
        re.I,
    )
    if _UNATTRIBUTED_NEGATIVE_PAT.search(summary.what_happened):
        if not re.search(
            r"\b(?:reported|stated|confirmed|according\s+to|said|announced|claimed|monitors?|authorities|officials)\b",
            summary.what_happened,
            re.I,
        ):
            unsupported_claims.append(
                "Negative assertion ('no breaches occurred') presented as independent fact "
                "without attribution to reporting authority"
            )

    # 5o. Major geographic theatre and maritime region grounding
    _MAJOR_GEOGRAPHIC_THEATRES = (
        "south china sea",
        "taiwan strait",
        "east china sea",
        "philippine sea",
        "yellow sea",
        "black sea",
        "baltic sea",
        "red sea",
        "persian gulf",
        "strait of hormuz",
        "gulf of oman",
        "korean peninsula",
        "dnipro river",
        "gaza strip",
    )
    for theatre in _MAJOR_GEOGRAPHIC_THEATRES:
        if re.search(rf"\b{re.escape(theatre)}\b", headline_and_body, re.I):
            if not re.search(rf"\b{re.escape(theatre)}\b", source_lower):
                unsupported_claims.append(
                    f"Geographic theatre or body of water '{theatre.title()}' asserted in summary "
                    f"is not present in source text"
                )

    # 5p. Cross-theatre military action conflation check
    theatre_conflation_issues = check_cross_theatre_conflation(summary, source_text)
    if theatre_conflation_issues:
        unsupported_claims.extend(theatre_conflation_issues)

    # 5q. Validate legislative and presidential action claims
    _LEGISLATIVE_ACTION_CHECKS: tuple[tuple[re.Pattern[str], re.Pattern[str], str], ...] = (
        (
            re.compile(r"\b(?:issued\s+)?executive\s+orders?\b", re.I),
            re.compile(r"\bexecutive\s+orders?\b", re.I),
            "Assertion of 'executive order(s)' is not substantiated by source text",
        ),
        (
            re.compile(r"\bpresidential\s+directives?\b", re.I),
            re.compile(r"\bpresidential\s+directives?\b", re.I),
            "Assertion of 'presidential directive(s)' is not substantiated by source text",
        ),
        (
            re.compile(r"\bsigned\s+(?:into\s+law|the\s+bill|legislation)\b", re.I),
            re.compile(
                r"\b(?:signed\s+(?:into\s+law|the\s+bill|legislation)|signs\s+(?:into\s+law|bill))\b",
                re.I,
            ),
            "Claim that legislation was 'signed into law' is not substantiated by source text",
        ),
        (
            re.compile(r"\bratified\s+(?:the\s+)?(?:treaty|accord|pact|agreement)\b", re.I),
            re.compile(r"\bratif\w+\b", re.I),
            "Claim of treaty ratification is not substantiated by source text",
        ),
        (
            re.compile(r"\bvetoed\s+(?:the\s+)?(?:bill|legislation|measure)\b", re.I),
            re.compile(r"\bveto\w*\b", re.I),
            "Claim of legislative veto is not substantiated by source text",
        ),
        (
            re.compile(r"\bpassed\s+(?:a\s+)?(?:new\s+)?sanctions\s+bill\b", re.I),
            re.compile(r"\b(?:passed\s+(?:a\s+)?(?:new\s+)?sanctions|sanctions\s+bill)\b", re.I),
            "Claim of sanctions bill passage is not substantiated by source text",
        ),
    )
    for claim_pat, src_pat, err_msg in _LEGISLATIVE_ACTION_CHECKS:
        if claim_pat.search(headline_and_body):
            if not src_pat.search(source_text):
                unsupported_claims.append(err_msg)

    # 5r. Validate unsubstantiated strategic intent and motive claims
    _UNSUBSTANTIATED_INTENT_PAT = re.compile(
        r"\b(intended\s+to\s+provoke|aiming\s+to\s+topple|aimed\s+at\s+destabilizing|"
        r"designed\s+to\s+undermine|in\s+retaliation\s+for|as\s+a\s+direct\s+response\s+to|"
        r"to\s+test\s+(?:the\s+)?resolve\s+of)\b",
        re.I,
    )
    for m in _UNSUBSTANTIATED_INTENT_PAT.finditer(headline_and_body):
        intent_phrase = m.group(1).lower()
        first_w = intent_phrase.split()[0]
        last_w = intent_phrase.split()[-1]
        if first_w not in source_lower or last_w not in source_lower:
            unsupported_claims.append(
                f"Unsubstantiated strategic intent or motive claim '{m.group(0)}' "
                "not substantiated by source text"
            )

    # 5s. Check duration vs endurance mischaracterization (e.g. 48 hours of jamming)
    _ENDURANCE_MISCHARACTERIZATION_PAT = re.compile(
        r"\b(?:48\s*hours?\s+of\s+jamming|jamm(?:ing|ed)\s+(?:for\s+)?48\s*hours?|"
        r"48[- ]hour\s+jamming|jamming\s+lasting\s+48\s*hours?|"
        r"endur(?:ed|ing)\s+48\s*hours?\s+of\s+jamming)\b",
        re.I,
    )
    if _ENDURANCE_MISCHARACTERIZATION_PAT.search(headline_and_body):
        unsupported_claims.append(
            "Incorrect paraphrase of endurance: '48 hours' represents capability survival "
            "threshold under degraded conditions, not the duration of jamming"
        )
    if (
        summary.key_details.timing
        and re.search(r"\b48\s*hours?\b", summary.key_details.timing, re.I)
        and summary.key_details.action
        and re.search(r"\bjam(?:ming)?\b", summary.key_details.action, re.I)
        and "daily" in source_lower
    ):
        unsupported_claims.append(
            "Timing mismatch: '48 hours' represents capability survival threshold, "
            "not the duration or timing of jamming"
        )

    # The quoted 48-hour limit applies to communications-dependent capabilities,
    # not to the physical survival of US forces. Catch subject substitution even
    # when the summary says "under daily jamming" instead of "48 hours of jamming".
    force_survival_claim = re.compile(
        r"\b(?:us\s+forces(?:\s+korea)?|american\s+forces|forces\s+on\s+(?:the\s+)?"
        r"(?:korean\s+)?peninsula|troops|soldiers|service\s+members)\b"
        r"[^.!?]{0,100}\b(?:would|will|could|might|cannot|can't|won't)\s+not\s+survive\b"
        r"[^.!?]{0,35}\b48\s*hours?\b",
        re.I,
    )
    source_capability_threshold = re.compile(
        r"\b(?:capabilit(?:y|ies)|systems?|equipment|technology)\b"
        r"[^.!?]{0,240}\b(?:will|would|could|might)\s+not\s+survive\b"
        r"[^.!?]{0,35}\b48\s*hours?\b",
        re.I,
    )
    if force_survival_claim.search(headline_and_body) and source_capability_threshold.search(
        source_text
    ):
        unsupported_claims.append(
            "Subject mismatch: source says communications-dependent capabilities will not "
            "survive 48 hours; it does not say US forces will not survive"
        )

    # 5t. Check inverted trade flows (e.g. US selling to Russia when source describes purchasing)
    _INVERTED_TRADE_PAT = re.compile(
        r"\b(?:(?:us|united states|white house|washington)\s+"
        r"(?:agreed\s+to\s+sell|agrees\s+to\s+sell|selling|sells)\s+(?:to\s+)?(?:russia|moscow))\b",
        re.I,
    )
    if _INVERTED_TRADE_PAT.search(headline_and_body):
        if not re.search(
            r"\b(?:sell\s+to\s+russia|selling\s+to\s+russia|sales\s+to\s+russia)\b",
            source_lower,
        ):
            unsupported_claims.append(
                "Inverted transaction: US reported selling to Russia when source describes "
                "US purchasing or importing Russian energy"
            )

    # 5u. Validate commodity integrity
    commodity_issues = check_commodity_integrity(summary, event, source_lower)
    if commodity_issues:
        unsupported_claims.extend(commodity_issues)

    # 6. Check historical vs modern temporal contradiction
    is_historical = bool(_HISTORICAL_MARKERS.search(source_text))
    has_hook = bool(_CONTEMPORARY_HOOK_MARKERS.search(source_text))
    if is_historical and not has_hook:
        # If source is historical without a hook, check if summary invents modern claims
        modern_pat = re.compile(
            r"\b(future military exercises|modern autonomous war|"
            r"nato exercises|contemporary exercises)\b",
            re.I,
        )
        if modern_pat.search(headline_and_body):
            unsupported_claims.append(
                "Historical battle text framed with unsupported modern military claims"
            )

    # 7. Corroboration requires distinct source links attached to this event and
    # shown to readers; a source's citations to other institutions are not attached
    # reports and cannot establish multi-source coverage for this briefing item.
    displayed_source_count = _distinct_displayed_sources(summary, event)
    combined_lower = f"{headline_and_body} {summary.confidence_qualification}".lower()
    corroboration_claim = re.compile(
        r"\bcorroborat\w*\b|\b(?:multiple|distinct|independent)\s+"
        r"(?:independent\s+)?(?:sources?|reports?|outlets?|institutions?)\b|"
        r"\bconfirmed by both\b|\bcross-verified\b",
        re.I,
    )
    has_corroboration_claim = False
    for sentence in re.split(r"(?<=[.!?])\s+", combined_lower):
        if not corroboration_claim.search(sentence):
            continue
        if re.search(
            r"\b(?:no|not|without|never|unverified|unconfirmed)\b[^.!?]{0,30}"
            r"\bcorroborat\w*\b|\bcorroborat\w*\b[^.!?]{0,24}"
            r"\b(?:not|absent|unavailable|unconfirmed)\b",
            sentence,
        ):
            continue
        has_corroboration_claim = True
        break
    if has_corroboration_claim and displayed_source_count < 2:
        unsupported_claims.append(
            "Single-source event improperly claims multi-source corroboration; fewer than "
            "two distinct, attached source links are available"
        )
    if event.confidence_label == "corroborated" and displayed_source_count < 2:
        unsupported_claims.append(
            "Corroborated confidence label lacks two distinct, attached source links"
        )

    if displayed_source_count < 2:
        has_verified_claim = bool(
            re.search(
                r"\b(independently verified|independently confirmed|verified by multiple)\b",
                combined_lower,
            )
        )
        if has_verified_claim and not re.search(
            r"\b(?:unverified independently|not independently verified|"
            r"unconfirmed independently|unconfirmed)\b",
            combined_lower,
        ):
            unsupported_claims.append(
                "Single-source event improperly asserts independent verification"
            )

    # 8. Compile outcome
    if unsupported_entities or unsupported_claims:
        reasons: list[str] = []
        if unsupported_entities:
            reasons.append(
                f"unsupported entities {list(unsupported_entities)} not present in source text"
            )
        if unsupported_claims:
            reasons.append("; ".join(unsupported_claims))
        return GroundingValidationResult(
            is_valid=False,
            unsupported_entities=tuple(unsupported_entities),
            unsupported_claims=tuple(unsupported_claims),
            quarantine_reason="; ".join(reasons),
        )

    return GroundingValidationResult(is_valid=True)
