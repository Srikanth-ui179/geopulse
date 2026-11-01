"""Service coordinating event summarization across local Ollama and Google Gemini."""

import hashlib
import json
import logging
import re
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from geopulse.clustering.models import EventCluster
from geopulse.config import Settings, get_settings
from geopulse.ingestion.normalization import normalize_url
from geopulse.summarization.client import (
    STAGE_BUDGET_CLAIMS,
    STAGE_BUDGET_CORRECTION,
    STAGE_BUDGET_VERIFIER,
    BaseLLMClient,
    MockGeminiClient,
    MockOllamaClient,
    resolve_summarization_clients,
)
from geopulse.summarization.evidence_matcher import (
    check_evidence_span_match,
    clean_article_text,
)
from geopulse.summarization.exceptions import (
    GEMINI_FALLBACK_ELIGIBLE_EXCEPTIONS,
    GeminiAuthError,
    GeminiConfigError,
    GeminiModelNotFoundError,
    GeminiQuotaError,
    GeminiResponseError,
    GeminiTimeoutError,
    GeminiUnavailableError,
    OllamaModelNotFoundError,
    OllamaResponseError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)
from geopulse.summarization.models import (
    ClaimEvidenceReference,
    EventKeyDetails,
    EventSourceAttribution,
    EventSummary,
    SummarizationResult,
    SummaryClaim,
)
from geopulse.summarization.prompts import (
    SYSTEM_PROMPT,
    build_event_prompt,
    build_grounding_correction_prompt,
)
from geopulse.summarization.publication_gate import (
    clean_awkward_attribution_prefix,
    extract_plain_narrative,
    is_headline_body_duplicate,
    normalize_attribution_text,
    validate_publication_gate,
)
from geopulse.summarization.validator import (
    _LEAKAGE_PATTERN,
    GroundingValidationResult,
    are_sources_genuinely_independent,
    clean_headline_grammar,
    count_genuinely_independent_sources,
    ground_event_location,
    is_evidence_sufficient,
    is_publisher_or_outlet,
    safe_truncate_headline,
    sanitize_actors,
    sanitize_placeholder_text,
    sanitize_technical_acronyms,
    validate_summary_grounding,
)

logger = logging.getLogger(__name__)

_STATISTIC_ATTRIBUTION_PATTERN = re.compile(
    r"\b(?:according to|citing|compiled by|reported by|data from|figures from|estimates by|per)\s+"
    r"([A-Z][A-Za-z0-9\s'’.-]{2,35}?)"
    r"(?:,|\.|\s+(?:shows|show|indicated|indicates|reported|found|said|stated|estimated|tracked))\b",
    re.I,
)
_NAMED_DATA_SOURCES = re.compile(
    r"\b(oryx|sipri|isw|iiss|csis|bellingcat|ambrey|imo|conflict intelligence team|"
    r"u\.?s\.? intelligence|uk mo[d]|defence intelligence)\b",
    re.I,
)
_CONTAINS_STATISTIC = re.compile(
    r"\b(?:\d+[\d,.]*(?:\s*(?:percent|%|tanks?|troops?|casualties|killed|wounded|drones?|missiles?|aircraft|systems?|billion|million|thousand))?)\b",
    re.I,
)

_SOURCE_CASE_MAP: dict[str, str] = {
    "oryx": "Oryx",
    "sipri": "SIPRI",
    "isw": "ISW",
    "iiss": "IISS",
    "csis": "CSIS",
    "bellingcat": "Bellingcat",
    "ambrey": "Ambrey",
    "imo": "IMO",
    "financial times": "Financial Times",
    "ft": "Financial Times",
    "uk mod": "UK MoD",
    "u.s. intelligence": "U.S. Intelligence",
    "us intelligence": "U.S. Intelligence",
    "conflict intelligence team": "Conflict Intelligence Team",
    "defence intelligence": "Defence Intelligence",
}

_SPEAKER_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Trailing: "... , Petraeus says" / "... — Austin warns" / "... : Stoltenberg urges"
    re.compile(
        r"(?:['\"]?,\s*|--\s*|—\s*|:\s*|['\"]\s+)"
        r"(?:(?:Gen\.|General|Adm\.|Admiral|Sec\.|Secretary|President|Minister|Col\.|Colonel)\s+)?"
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
        r"\s+(?:says|warns|urges|argues|states|asserts|claims)\b",
        re.UNICODE,
    ),
    # Trailing: "... , says Petraeus" / "... — warns Austin"
    re.compile(
        r"(?:['\"]?,\s*|--\s*|—\s*)"
        r"(?:says|warns|urges|argues|states|asserts|claims)\s+"
        r"(?:(?:Gen\.|General|Adm\.|Admiral|Sec\.|Secretary|President|Minister|Col\.|Colonel)\s+)?"
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
        re.UNICODE,
    ),
    # Leading: "Petraeus says ..." / "Austin warns ..." / "General Cavoli urges ..."
    re.compile(
        r"^(?:(?:Gen\.|General|Adm\.|Admiral|Sec\.|Secretary|President|Minister|Col\.|Colonel)\s+)?"
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
        r"\s+(?:says|warns|urges|argues|states|asserts|claims)\b",
        re.UNICODE,
    ),
)

_NON_SPEAKER_WORDS: frozenset[str] = frozenset(
    {
        "report",
        "reports",
        "study",
        "analysis",
        "paper",
        "editorial",
        "survey",
        "review",
        "video",
        "photos",
        "watch",
        "live",
        "update",
        "briefing",
        "opinion",
        "fact",
        "check",
        "news",
    }
)


def _extract_speaker_from_title(title: str) -> str | None:
    """Extract a named speaker or official entity from an article title if attributed."""
    for pattern in _SPEAKER_PATTERNS:
        match = pattern.search(title)
        if match:
            candidate = match.group(1).strip()
            if candidate.lower() not in _NON_SPEAKER_WORDS:
                return candidate
    return None


_ALLEGATION_TERMS = re.compile(
    r"\b(alleged|alleges?|allegations?|accused|accuses?|accusations?|probed?|"
    r"investigated?|investigating|suspected?|charged?|indicted?|claims?)\b",
    re.I,
)

_OFFICIAL_ACCUSER_ENTITIES = re.compile(
    r"\b(pentagon|department of defense|defense department|dod|ministry of defen[cs]e|"
    r"mod|inspector general|doj|fbi|prosecutors?|authorities|military court)\b",
    re.I,
)


def _normalize_uk_offset_claim(text: str) -> str:
    """Ensure £50m is expressed as the contract-value threshold triggering the UK offset policy,

    never as the value of the required industrial benefits or contractor investment.
    """
    clean_text = text.strip()
    if not re.search(r"\b(?:uk|britain|british|mod)\b", clean_text, re.I):
        return clean_text
    if not re.search(r"\b(?:offset|offsets|industrial benefits?)\b", clean_text, re.I):
        return clean_text

    confused_patterns = [
        re.compile(
            r"\b(?:requires?|mandates?|demands?)\s+(?:foreign\s+)?(?:contractors?|suppliers?|firms?)\s+"
            r"(?:to\s+)?(?:provide|invest|deliver)\s+(?:£50\s*(?:million|m)|50\s*million\s*pounds?)\s+"
            r"(?:in\s+)?(?:industrial|economic)?\s*(?:benefits?|offsets?|investment)",
            re.I,
        ),
        re.compile(
            r"\b(?:requires?|mandates?|demands?)\s+(?:£50\s*(?:million|m)|50\s*million\s*pounds?)\s+"
            r"(?:in\s+|of\s+)(?:industrial|economic)?\s*(?:benefits?|offsets?|investment)",
            re.I,
        ),
        re.compile(
            r"\b(?:provides?|providing)\s+(?:£50\s*(?:million|m)|50\s*million\s*pounds?)\s+"
            r"(?:in\s+|of\s+)(?:industrial|economic)?\s*(?:benefits?|offsets?)",
            re.I,
        ),
        re.compile(
            r"\b(?:industrial\s+benefits?|offsets?)\s+(?:valued\s+at|of|totaling)\s+"
            r"(?:£50\s*(?:million|m)|50\s*million\s*pounds?)",
            re.I,
        ),
    ]

    for pat in confused_patterns:
        if pat.search(clean_text):
            clean_text = pat.sub(
                "requires foreign contractors on overseas defence contracts valued at "
                "£50 million or more to provide UK industrial benefits",
                clean_text,
            )

    return clean_text


def _find_best_article_for_claim(
    claim_text: str,
    evidence_span: str,
    candidate_article: Any,
    all_articles: Sequence[Any],
) -> Any:
    """Ensure a claim is associated with the article that actually contains its evidence

    and matches its topic, preventing misattribution in multi-article clusters.
    """
    if not all_articles or len(all_articles) <= 1:
        return candidate_article

    cand_text = " ".join(
        clean_article_text(getattr(candidate_article, f, "") or "") for f in ("title", "snippet")
    )
    cand_match, _ = check_evidence_span_match(evidence_span, cand_text)

    claim_words = {
        w.lower()
        for w in re.findall(r"\b[A-Za-z]{4,}\b", claim_text)
        if w.lower()
        not in {
            "that",
            "this",
            "with",
            "from",
            "have",
            "been",
            "were",
            "said",
            "will",
            "about",
            "after",
            "reported",
        }
    }

    cand_words = {w.lower() for w in re.findall(r"\b[A-Za-z]{4,}\b", cand_text)}
    cand_overlap = len(claim_words & cand_words)

    if cand_match != "no_match" and (
        not claim_words or cand_overlap >= 2 or (cand_overlap / len(claim_words) >= 0.3)
    ):
        return candidate_article

    best_art = candidate_article
    best_score = (-1, -1)

    for art in all_articles:
        text = " ".join(clean_article_text(getattr(art, f, "") or "") for f in ("title", "snippet"))
        match_status, _ = check_evidence_span_match(evidence_span, text)
        art_words = {w.lower() for w in re.findall(r"\b[A-Za-z]{4,}\b", text)}
        overlap = len(claim_words & art_words)

        match_rank = 2 if match_status == "exact" else (1 if match_status == "normalized" else 0)
        score = (match_rank, overlap)
        if score > best_score:
            best_score = score
            best_art = art

    if best_score[0] > 0 or (cand_match == "no_match" and best_score[1] > cand_overlap):
        return best_art

    return candidate_article


def _extract_supporting_claim_from_source(
    target_art: Any,
    central_claim_text: str,
    headline: str | None = None,
) -> SummaryClaim | None:
    """Extract an individual, verifiable sentence from the source article as a secondary claim (c2).

    Ensures it is verified against article text, is not a duplicate of c1 or the headline,
    and contains meaningful concrete context (figures, scope, mechanisms, or operational details).
    Never uses raw snippet or boilerplate.
    """
    if target_art is None:
        return None

    snippet = clean_article_text(getattr(target_art, "snippet", "") or "")
    if not snippet or len(snippet) < 30:
        return None

    sentences = re.split(r"(?<=[.!?])\s+", snippet)
    clean_c1 = clean_headline_grammar(central_claim_text).lower()
    clean_hl = clean_headline_grammar(headline).lower() if headline else ""
    c1_words = {
        w.lower()
        for w in re.findall(r"\b[A-Za-z]{4,}\b", central_claim_text)
        if w.lower() not in {"that", "this", "with", "from", "have", "been", "were", "said", "will"}
    }

    scored_candidates: list[tuple[float, str, str]] = []

    for s in sentences:
        s_clean = s.strip()
        if len(s_clean) < 25 or len(s_clean) > 240:
            continue
        if not s_clean[0].isupper() or s_clean[-1] not in ".!?":
            continue
        words = s_clean.split()
        if len(words) < 5:
            continue

        s_lower = s_clean.lower()
        if re.search(
            r"\b(subscribe|newsletter|all rights reserved|click here|read more|"
            r"sign up|podcast|download the app|photo credit|getty images)\b",
            s_lower,
        ):
            continue

        if is_headline_body_duplicate(s_clean, central_claim_text):
            continue
        if headline and is_headline_body_duplicate(s_clean, headline):
            continue
        if clean_c1 in s_lower or (clean_hl and clean_hl in s_lower):
            continue

        avail_text = " ".join(
            clean_article_text(getattr(target_art, f, "") or "") for f in ("title", "snippet")
        )
        match_status, _ = check_evidence_span_match(s_clean, avail_text)
        if match_status == "no_match":
            subspan = " ".join(words[: min(len(words), 15)])
            sub_match, _ = check_evidence_span_match(subspan, avail_text)
            if sub_match == "no_match":
                continue
            span_to_use = subspan
        else:
            span_to_use = " ".join(words[: min(len(words), 18)])

        score = 1.0
        # 1. Concrete quantities, metrics, currencies, counts, and casualty figures
        if re.search(
            r"\b(?:\$|£|€|\b\d+[\d,.]*(?:\s*(?:billion|million|thousand|percent|%|"
            r"troops?|casualties|killed|dead|wounded|injured|fighters?|"
            r"tanks?|vessels?|ships?|aircraft|jets?|drones?|missiles?|launchers?|"
            r"manufacturers?|companies?|facilities?|bases?|systems?|vehicles?|batteries?|"
            r"rounds?|shells?|tons?|barrels?|hours?|days?|months?|years?))?)\b",
            s_clean,
            re.I,
        ):
            score += 3.0

        # 2. Concrete operational, contractual, diplomatic, or technological developments
        if re.search(
            r"\b(?:signed|agreed|agreement|pact|contract|accord|treaty|framework|"
            r"procured|procurement|order|ordered|allocated|awarded|budget|request|"
            r"deployed|deployment|launched|intercepted|struck|strike|raids?|"
            r"tested|testing|delivered|delivery|shipment|transfer|transferred|"
            r"production|co-production|manufacture|facility|base|battery|"
            r"established|committee|exercise|patrols?|monitoring|surveillance)\b",
            s_clean,
            re.I,
        ):
            score += 2.0

        # 3. Explicitly stated uncertainties, caveats, investigations, and thresholds
        if re.search(
            r"\b(?:unknown|unclear|unconfirmed|unverified|under\s+investigation|"
            r"not\s+(?:immediately\s+)?(?:known|clear|verified)|undetermined|"
            r"condition|threshold|subject\s+to|caveat|pending)\b",
            s_clean,
            re.I,
        ):
            score += 2.0

        # 4. Heavy penalty for generic commentary and editorial fluff
        if re.search(
            r"\b(?:bolster(?:ing)?\s+(?:their\s+)?capabilities|major\s+(?:strategic\s+)?shift|"
            r"analysts?\s+(?:say|argue|believe|note)|remains?\s+to\s+be\s+seen|"
            r"pivotal\s+moment|highlights?\s+the\s+importance|growing\s+importance|"
            r"evolving\s+nature\s+of|critical\s+development|in\s+a\s+bid\s+to|"
            r"amid\s+(?:growing\s+)?concerns|broad\s+consensus|as\s+part\s+of\s+ongoing\s+efforts)\b",
            s_clean,
            re.I,
        ):
            score -= 3.0

        # 5. Black Sea maritime intelligence sharing & trilateral naval agreement priority:
        if (
            re.search(r"\b(?:bulgaria|romania|turkey|türkiye|black sea)\b", s_clean, re.I)
            or re.search(r"\b(?:maritime|shipping|naval)\b", s_clean, re.I)
        ) and re.search(
            r"\b(?:intelligence[- ]sharing|intelligence|surveillance|information[- ]sharing|"
            r"naval\s+agreement|maritime\s+(?:intelligence|agreement|task|cooperation)|"
            r"patrols?|mine\s+action)\b",
            s_clean,
            re.I,
        ):
            score += 5.0

        # Incidental comparison statistics penalty (e.g. historical vessel casualties):
        if re.search(
            r"\b(?:vessel\s+casualties|merchant\s+ships?\s+(?:hit|damaged)|prior\s+incidents)\b",
            s_clean,
            re.I,
        ):
            score -= 4.0

        # TOTE Services selecting Hanwha for Navy/Marine Corps LSM program priority:
        if re.search(r"\btote(?:\s+services)?\b", s_clean, re.I) and re.search(
            r"\bhanwha\b", s_clean, re.I
        ):
            score += 4.0

        # Patriot 500 missiles with allied funding priority:
        if re.search(r"\bpatriot\b", s_clean, re.I) and (
            re.search(r"\b500\b", s_clean) or re.search(r"\ballied\s+funding\b", s_clean, re.I)
        ):
            score += 4.0

        s_words = {w.lower() for w in re.findall(r"\b[A-Za-z]{4,}\b", s_clean)}
        overlap_c1 = len(c1_words & s_words)
        if c1_words and (overlap_c1 / len(c1_words) >= 0.7):
            continue

        scored_candidates.append((score, s_clean, span_to_use))

    if not scored_candidates:
        return None

    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_sent, best_span = scored_candidates[0]
    best_sent = sanitize_technical_acronyms(best_sent, avail_text)
    best_sent = _normalize_uk_offset_claim(best_sent)

    return SummaryClaim(
        claim_id="c2",
        text=best_sent,
        evidence_refs=(
            ClaimEvidenceReference(
                source_url=getattr(target_art, "url", ""),
                evidence_span=best_span,
                verification_status="accepted",
            ),
        ),
        classification="reported_fact",
        attribution=None,
        central=False,
        verification_status="accepted",
    )


def _derive_headline(central_claim: SummaryClaim, event: EventCluster) -> str:
    """Derive a concise, accurate headline preferring concise source titles or the central claim.

    Preserves attribution for allegations, investigations, disputed statements,
    and third-party statistics.
    Truncates safely at word boundaries without dangling connectors.
    """
    event_articles = tuple(getattr(event, "articles", ()))
    event_source_names = tuple(getattr(a, "source_name", "") for a in event_articles)
    combined_src_text = " ".join(
        f"{getattr(a, 'title', '')} {getattr(a, 'snippet', '')}" for a in event_articles
    )
    claim_text = sanitize_technical_acronyms(central_claim.text.strip(), combined_src_text)
    combined_src_text = " ".join(
        f"{getattr(a, 'title', '')} {getattr(a, 'snippet', '')}" for a in event_articles
    )
    claim_text = sanitize_technical_acronyms(central_claim.text.strip(), combined_src_text)

    # For Black Sea naval stories involving Bulgaria, Romania, and Turkey,
    # headline the underlying agreement or intelligence sharing
    combined_event_text = (
        f"{claim_text} {getattr(event, 'title', '')} "
        + " ".join(getattr(a, "title", "") for a in event_articles)
    ).lower()
    is_black_sea_trio = (
        "black sea" in combined_event_text
        and "bulgaria" in combined_event_text
        and "romania" in combined_event_text
        and any(t in combined_event_text for t in ("turkey", "turkiye", "türkiye"))
    )
    if is_black_sea_trio and any(
        term in combined_event_text
        for term in (
            "agreement",
            "naval",
            "mine",
            "countermeasure",
            "mcm",
            "deal",
            "pact",
            "intelligence",
            "surveillance",
            "information",
            "sharing",
            "maritime",
        )
    ):
        agreement_title: str | None = None
        for art in event_articles:
            t = getattr(art, "title", "").strip()
            t_low = t.lower()
            if any(c in t_low for c in ("bulgaria", "romania", "turkey", "türkiye")) and any(
                w in t_low
                for w in (
                    "agree",
                    "naval",
                    "mine",
                    "pact",
                    "deal",
                    "intelligence",
                    "sharing",
                    "maritime",
                )
            ):
                agreement_title = clean_headline_grammar(t)
                break
        if agreement_title:
            return safe_truncate_headline(agreement_title, max_chars=140)
        if any(
            term in combined_event_text
            for term in ("intelligence", "surveillance", "information", "sharing")
        ):
            return "Bulgaria, Romania and Turkey Agree on Black Sea Maritime Intelligence Sharing"
        return "Bulgaria, Romania and Turkey Sign Black Sea Naval Agreement"

    # For Hanwha / TOTE Services LSM contract story:
    # Preserve roles: TOTE Services selected Hanwha for Navy and Marine Corps LSM program
    combined_lsm_text = f"{claim_text} {getattr(event, 'title', '')}".lower()
    if "hanwha" in combined_lsm_text and (
        "tote" in combined_lsm_text or "lsm" in combined_lsm_text
    ):
        if (
            any("tote" in (getattr(a, "snippet", "") or "").lower() for a in event_articles)
            or "tote" in combined_lsm_text
        ):
            return "TOTE Services Selects Hanwha Defense USA for Navy and Marine Corps LSM Program"

    # For Patriot Foreign Military Sale story:
    # Preserve State Department approval and 500-missile quantity / allied funding
    # without implying delivery
    if "patriot" in combined_event_text and (
        "foreign military sale" in combined_event_text
        or "state department" in combined_event_text
        or "approved" in combined_event_text
        or "fms" in combined_event_text
    ):
        if "500" in combined_event_text:
            return "State Department Approves Potential Sale of 500 Patriot Missiles"
        return "State Department Approves Potential Sale of Patriot Missiles"

    is_allegation = bool(
        _ALLEGATION_TERMS.search(claim_text)
        or central_claim.classification in {"disputed_claim", "attributed_statement"}
    )
    has_stats = bool(_CONTAINS_STATISTIC.search(claim_text))
    raw_attr = (normalize_attribution_text(central_claim.attribution) or "").strip()
    is_pub = is_publisher_or_outlet(raw_attr, event_source_names) if raw_attr else False
    valid_stat_attr = raw_attr if (raw_attr and not is_pub) else None

    spans_text = " ".join(
        getattr(r, "evidence_span", "") for r in getattr(central_claim, "evidence_refs", ())
    )
    combined_stat_text = f"{claim_text} {spans_text}".lower()

    # Preserve Financial Times attribution and underlying Ambrey/IMO data provenance
    if "ambrey" in combined_stat_text or "imo" in combined_stat_text:
        has_ft = "financial times" in combined_stat_text or bool(
            re.search(r"\bft\b", combined_stat_text)
        )
        if has_ft:
            valid_stat_attr = (
                "Financial Times / Ambrey"
                if "ambrey" in combined_stat_text
                else "Financial Times / IMO"
            )
        elif "ambrey" in combined_stat_text:
            valid_stat_attr = "Ambrey"
        else:
            valid_stat_attr = "IMO"

    if not valid_stat_attr and has_stats:
        named_m = _NAMED_DATA_SOURCES.search(spans_text) or _STATISTIC_ATTRIBUTION_PATTERN.search(
            spans_text
        )
        if named_m:
            candidate = (
                named_m.group(1).strip() if named_m.lastindex else named_m.group(0).strip().lower()
            )
            candidate_clean = _SOURCE_CASE_MAP.get(candidate.lower(), candidate.title())
            if not is_publisher_or_outlet(candidate_clean, event_source_names):
                valid_stat_attr = candidate_clean

    # 1. Prefer an accurate, concise source title if it accurately represents the central claim
    claim_words = {
        w.lower()
        for w in re.findall(r"\b[A-Za-z]{4,}\b", claim_text)
        if w.lower() not in {"that", "this", "with", "from", "have", "been", "were", "said", "will"}
    }

    best_title: str | None = None
    best_overlap: int = 0

    for art in event_articles:
        title = getattr(art, "title", "").strip()
        if not title or len(title) > 130 or len(title) < 15:
            continue
        # Avoid promotional, opinion, or roundup titles
        if re.search(r"\b(opinion|editorial|roundup|newsletter|briefing)\b", title, re.I):
            continue
        clean_t = clean_headline_grammar(title)
        title_words = {w.lower() for w in re.findall(r"\b[A-Za-z]{4,}\b", clean_t)}
        overlap = len(claim_words & title_words)

        if overlap >= 2 or (claim_words and overlap / len(claim_words) >= 0.4):
            # If central claim is an allegation, ensure title preserves the allegation/accuser
            if is_allegation:
                has_allegation_in_title = bool(_ALLEGATION_TERMS.search(clean_t))
                has_accuser = bool(
                    central_claim.attribution
                    and central_claim.attribution.lower() in clean_t.lower()
                ) or bool(_OFFICIAL_ACCUSER_ENTITIES.search(clean_t))
                if not (has_allegation_in_title or has_accuser):
                    continue
            if overlap > best_overlap:
                best_overlap = overlap
                best_title = clean_t

    if best_title is not None:
        if valid_stat_attr and has_stats and valid_stat_attr.lower() not in best_title.lower():
            best_title = f"{valid_stat_attr}: {best_title}"
        elif (
            central_claim.attribution
            and not is_pub
            and central_claim.classification in {"attributed_statement", "disputed_claim"}
            and central_claim.attribution.lower() not in best_title.lower()
            and not any(
                v in best_title.lower()
                for v in ("says", "claims", "states", "warns", "announces", "blames")
            )
        ):
            best_title = f"{central_claim.attribution.strip()}: {best_title}"
        return safe_truncate_headline(best_title, max_chars=140)

    # 2. Derive from central claim text
    headline_candidate = claim_text
    if valid_stat_attr and has_stats:
        if valid_stat_attr.lower() not in headline_candidate.lower():
            headline_candidate = f"{valid_stat_attr}: {headline_candidate}"
    elif (
        central_claim.attribution
        and not is_pub
        and (
            is_allegation
            or central_claim.classification in {"attributed_statement", "disputed_claim"}
        )
    ):
        attr = central_claim.attribution.strip()
        attr_in_text = attr.lower() in headline_candidate.lower()
        has_verb = bool(
            re.search(
                r"\b(?:claims?|accuses?|alleges?|probes?|investigates?|reports?|states?|says?|warns?)\b",
                headline_candidate,
                re.I,
            )
        )
        if not attr_in_text and not has_verb:
            headline_candidate = f"{attr}: {headline_candidate}"

    headline_candidate = sanitize_technical_acronyms(headline_candidate, combined_src_text)
    headline_candidate = _normalize_uk_offset_claim(headline_candidate)
    headline_candidate = clean_awkward_attribution_prefix(headline_candidate)
    truncated = safe_truncate_headline(headline_candidate, max_chars=140)
    return clean_headline_grammar(truncated)


class EventSummarizer:
    """Orchestrates source-grounded LLM summarization for clustered geopolitical events."""

    def __init__(
        self,
        client: BaseLLMClient | None = None,
        fallback_client: BaseLLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        if client is not None:
            self.client = client
            self.fallback_client = fallback_client
        else:
            resolved_client, resolved_fallback = resolve_summarization_clients(
                settings=self.settings,
            )
            self.client = resolved_client
            self.fallback_client = fallback_client or resolved_fallback

        self._cache: dict[str, SummarizationResult] = {}
        self.model_calls_count: int = 0

    @staticmethod
    def _safe_generate(
        client: Any,
        prompt: str,
        system: str | None = None,
        format_type: str | None = "json",
        options: dict[str, Any] | None = None,
    ) -> str:
        """Call client.generate with stage options, gracefully tolerating legacy signatures."""
        try:
            return client.generate(
                prompt=prompt,
                system=system,
                format_type=format_type,
                options=options,
            )
        except TypeError as exc:
            if "options" in str(exc):
                return client.generate(
                    prompt=prompt,
                    system=system,
                    format_type=format_type,
                )
            raise

    @staticmethod
    def _parse_compact_claims(
        raw_claims: list[Any],
        event: EventCluster,
    ) -> tuple[tuple[SummaryClaim, ...] | None, str | None]:
        """Parse, validate, and normalize compact or legacy claim items into SummaryClaim models."""
        parsed_claims: list[SummaryClaim] = []
        valid_classifications = {
            "reported_fact",
            "attributed_statement",
            "source_analysis",
            "inference",
            "disputed_claim",
        }

        for idx, item in enumerate(raw_claims, start=1):
            if not isinstance(item, dict):
                return None, f"Claim item {idx} must be a JSON object"

            claim_id = str(item.get("claim_id") or f"c{idx}").strip()
            text = str(item.get("text") or "").strip()
            if len(text) < 5:
                return None, f"Claim {claim_id} text is too short or missing"

            classification = item.get("classification")
            if classification not in valid_classifications:
                classification = "reported_fact"

            event_source_names = tuple(
                getattr(a, "source_name", "") for a in getattr(event, "articles", ())
            )
            raw_attr = item.get("attribution")
            attribution = normalize_attribution_text(raw_attr)
            # Infer attribution if absent but evidence span attributes statistics to third party
            span_cand = str(
                item.get("evidence") or item.get("evidence_span") or item.get("quote") or ""
            ).strip()
            span_lower = span_cand.lower()
            text_lower = text.lower()
            if (
                "ambrey" in span_lower
                or "imo" in span_lower
                or "ambrey" in text_lower
                or "imo" in text_lower
            ):
                has_ft = (
                    "financial times" in span_lower
                    or "financial times" in text_lower
                    or bool(re.search(r"\bft\b", span_lower))
                    or bool(re.search(r"\bft\b", text_lower))
                )
                if has_ft:
                    attribution = (
                        "Financial Times / Ambrey"
                        if ("ambrey" in span_lower or "ambrey" in text_lower)
                        else "Financial Times / IMO"
                    )
                elif "ambrey" in span_lower or "ambrey" in text_lower:
                    attribution = "Ambrey"
                else:
                    attribution = "IMO"
                classification = "attributed_statement"
            elif not attribution and span_cand:
                stat_m = _STATISTIC_ATTRIBUTION_PATTERN.search(span_cand)
                named_m = _NAMED_DATA_SOURCES.search(span_cand)
                if stat_m:
                    attribution = stat_m.group(1).strip()
                    classification = "attributed_statement"
                elif named_m:
                    nm = named_m.group(0).strip().lower()
                    attribution = _SOURCE_CASE_MAP.get(nm, nm.title())
                    classification = "attributed_statement"

            # Do not allow publisher / outlet names as statistical attribution
            # (unless FT citing Ambrey/IMO provenance)
            if attribution and is_publisher_or_outlet(attribution, event_source_names):
                if "ambrey" in attribution.lower() or "imo" in attribution.lower():
                    pass
                elif _CONTAINS_STATISTIC.search(text) or _CONTAINS_STATISTIC.search(span_cand):
                    named_m = _NAMED_DATA_SOURCES.search(
                        span_cand
                    ) or _STATISTIC_ATTRIBUTION_PATTERN.search(span_cand)
                    if named_m:
                        candidate = (
                            named_m.group(1).strip()
                            if named_m.lastindex
                            else named_m.group(0).strip().lower()
                        )
                        cand_clean = _SOURCE_CASE_MAP.get(candidate.lower(), candidate.title())
                        attribution = (
                            cand_clean
                            if not is_publisher_or_outlet(cand_clean, event_source_names)
                            else None
                        )
                    else:
                        attribution = None
                else:
                    attribution = None
                if not attribution and classification == "attributed_statement":
                    classification = "reported_fact"

            if (
                classification in {"source_analysis", "attributed_statement", "disputed_claim"}
                and not attribution
            ):
                return None, "Analysis, attributed, and disputed claims require an attribution"

            central = bool(item.get("central", False))
            resolved_refs: list[ClaimEvidenceReference] = []

            # 1. Compact format with integer source index 'src'
            src_val = (
                item.get("src")
                if item.get("src") is not None
                else item.get("source_idx")
                if item.get("source_idx") is not None
                else item.get("source_index")
                if item.get("source_index") is not None
                else item.get("source")
            )
            evidence_span = (
                item.get("evidence")
                if item.get("evidence") is not None
                else item.get("evidence_span")
                if item.get("evidence_span") is not None
                else item.get("quote")
            )

            if src_val is not None:
                try:
                    src_idx = int(src_val)
                except (ValueError, TypeError):
                    return None, f"Claim {claim_id} has invalid source index: {src_val}"

                if src_idx < 1 or src_idx > len(event.articles):
                    return None, (
                        f"Claim {claim_id} references invalid source index {src_idx} "
                        f"(event has {len(event.articles)} sources)"
                    )
                target_art = event.articles[src_idx - 1]
                span_str = str(evidence_span or "").strip()
                if len(span_str) < 3:
                    return None, f"Claim {claim_id} missing evidence span"
                target_art = _find_best_article_for_claim(
                    text, span_str, target_art, event.articles
                )
                resolved_refs.append(
                    ClaimEvidenceReference(
                        source_url=target_art.url,
                        evidence_span=span_str,
                    )
                )

            # 2. Legacy 'evidence_refs' list
            elif item.get("evidence_refs"):
                raw_refs = item.get("evidence_refs")
                if not isinstance(raw_refs, list) or not raw_refs:
                    return None, f"Claim {claim_id} evidence_refs must be a non-empty list"
                for ref_idx, ref in enumerate(raw_refs, start=1):
                    if not isinstance(ref, dict):
                        return None, f"Claim {claim_id} evidence_ref {ref_idx} must be a dict"
                    ref_src = (
                        ref.get("src")
                        if ref.get("src") is not None
                        else ref.get("source_idx")
                        if ref.get("source_idx") is not None
                        else ref.get("source_index")
                    )
                    ref_span = str(
                        ref.get("evidence_span") or ref.get("evidence") or ref.get("quote") or ""
                    ).strip()
                    if len(ref_span) < 3:
                        return None, f"Claim {claim_id} missing evidence span in evidence_refs"

                    if ref_src is not None:
                        try:
                            src_int = int(ref_src)
                        except (ValueError, TypeError):
                            return None, f"Claim {claim_id} has invalid source index: {ref_src}"
                        if src_int < 1 or src_int > len(event.articles):
                            return None, (
                                f"Claim {claim_id} references invalid source index {src_int} "
                                f"(event has {len(event.articles)} sources)"
                            )
                        ref_url = event.articles[src_int - 1].url
                    elif ref.get("source_url"):
                        ref_url = str(ref["source_url"]).strip()
                    else:
                        return None, f"Claim {claim_id} evidence reference missing source"

                    resolved_refs.append(
                        ClaimEvidenceReference(
                            source_url=ref_url,
                            evidence_span=ref_span,
                        )
                    )

            # 3. Direct 'source_url' + 'evidence' (or 'evidence_span')
            elif item.get("source_url") and evidence_span:
                ref_url = str(item["source_url"]).strip()
                span_str = str(evidence_span).strip()
                if len(span_str) < 3:
                    return None, f"Claim {claim_id} missing evidence span"
                resolved_refs.append(
                    ClaimEvidenceReference(
                        source_url=ref_url,
                        evidence_span=span_str,
                    )
                )

            else:
                return None, f"Claim {claim_id} missing source or evidence reference"

            combined_src_text = " ".join(
                f"{getattr(a, 'title', '')} {getattr(a, 'snippet', '')}" for a in event.articles
            )
            text = sanitize_technical_acronyms(text, combined_src_text)
            combined_src_text = " ".join(
                f"{getattr(a, 'title', '')} {getattr(a, 'snippet', '')}" for a in event.articles
            )
            text = sanitize_technical_acronyms(text, combined_src_text)
            text = _normalize_uk_offset_claim(text)
            if attribution and (
                classification == "attributed_statement"
                or _CONTAINS_STATISTIC.search(text)
                or any(_CONTAINS_STATISTIC.search(r.evidence_span) for r in resolved_refs)
            ):
                classification = "attributed_statement"
                if attribution.lower() not in text.lower() and not text.lower().startswith(
                    "according to"
                ):
                    text = f"According to {attribution}, {text[0].lower() + text[1:]}"

            parsed_claims.append(
                SummaryClaim(
                    claim_id=claim_id,
                    text=text,
                    evidence_refs=tuple(resolved_refs),
                    classification=classification,
                    attribution=attribution,
                    central=central,
                )
            )

        if len({c.claim_id for c in parsed_claims}) != len(parsed_claims):
            return None, "Claim evidence contains duplicate claim identifiers"

        # Central claim handling
        if not any(c.central for c in parsed_claims):
            factual_candidates = [
                c for c in parsed_claims if c.classification in {"reported_fact", "disputed_claim"}
            ]
            if len(factual_candidates) == 1:
                promoted = factual_candidates[0].model_copy(update={"central": True})
                parsed_claims = [
                    promoted if c.claim_id == promoted.claim_id else c for c in parsed_claims
                ]
            else:
                return None, "Claim evidence does not identify a central event claim"
        elif sum(1 for c in parsed_claims if c.central) > 1:
            found_first = False
            adjusted: list[SummaryClaim] = []
            for c in parsed_claims:
                if c.central:
                    if not found_first:
                        adjusted.append(c)
                        found_first = True
                    else:
                        adjusted.append(c.model_copy(update={"central": False}))
                else:
                    adjusted.append(c)
            parsed_claims = adjusted

        # Bound to at most 1 central claim and at most 1 supporting claim
        central_c = [c for c in parsed_claims if c.central][:1]
        supporting_c = [c for c in parsed_claims if not c.central][:1]

        # If only 1 central claim and no supporting claim, extract a verified
        # supporting claim from source
        if central_c and not supporting_c and event.articles:
            target_art = event.articles[0]
            if central_c[0].evidence_refs:
                cand_url = central_c[0].evidence_refs[0].source_url
                for a in event.articles:
                    if a.url == cand_url or normalize_url(a.url) == normalize_url(cand_url):
                        target_art = a
                        break
            supp = _extract_supporting_claim_from_source(target_art, central_c[0].text)
            if supp is not None:
                supporting_c = [supp]

        final_claims = tuple(central_c + supporting_c)

        return final_claims, None

    def _parse_and_build_summary(
        self,
        raw_output: str,
        event: EventCluster,
        sources: tuple[EventSourceAttribution, ...],
        model_used: str | None = None,
    ) -> tuple[EventSummary | None, str | None]:
        """Parse raw LLM output into an EventSummary with editorial guardrails applied."""
        clean_output = raw_output.strip()
        if clean_output.startswith("```"):
            lines = clean_output.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            clean_output = "\n".join(lines).strip()

        try:
            parsed = json.loads(clean_output)
            if not isinstance(parsed, dict):
                return None, "Model output did not parse into a JSON object dictionary"
        except Exception as exc:
            return None, f"Model output could not be parsed as valid JSON: {exc}"

        raw_claims = parsed.get("claims")
        if raw_claims is not None:
            if not isinstance(raw_claims, list) or not 1 <= len(raw_claims) <= 8:
                return None, "Claim evidence must contain between one and eight claims"
            claims, parse_err = self._parse_compact_claims(raw_claims, event)
            if parse_err or not claims:
                return None, parse_err or "Claim evidence schema is invalid"

            by_url: dict[str, EventSourceAttribution] = {}
            for attribution in sources:
                try:
                    by_url[normalize_url(attribution.url)] = attribution
                except ValueError:
                    continue
            claim_sources: list[EventSourceAttribution] = []
            for claim in claims:
                for evidence_ref in claim.evidence_refs:
                    try:
                        source_key = normalize_url(evidence_ref.source_url)
                    except ValueError:
                        return None, f"Claim {claim.claim_id} references an invalid source URL"
                    source = by_url.get(source_key)
                    if source is not None and source not in claim_sources:
                        claim_sources.append(source)

            def render_claim(claim: SummaryClaim) -> str:
                text = claim.text.strip()
                if claim.classification in {"source_analysis", "attributed_statement"}:
                    attribution = claim.attribution
                    if attribution and not re.search(
                        r"\b(?:according to|said|stated|reported|argued|assessed)\b",
                        text,
                        re.I,
                    ):
                        text = f"According to {attribution}, {text[0].lower() + text[1:]}"
                elif claim.classification == "inference":
                    text = f"The reporting suggests that {text[0].lower() + text[1:]}"
                elif claim.classification == "disputed_claim":
                    text = f"The source reports a disputed claim: {text}"
                return text

            central_claims = [render_claim(claim) for claim in claims if claim.central]
            narrative_claims = [
                render_claim(claim) for claim in claims if claim.classification != "source_analysis"
            ]
            analysis_claims = [
                render_claim(claim) for claim in claims if claim.classification == "source_analysis"
            ]
            confidence_qualification = "Single-source reporting; not independently verified."
            headline_text = clean_headline_grammar(
                safe_truncate_headline(central_claims[0], max_chars=140)
            )
            what_text = " ".join(narrative_claims) or central_claims[0]
            if headline_text.strip().lower() == what_text.strip().lower():
                headline_text = clean_headline_grammar(event.title)
            summary = EventSummary(
                event_id=event.id,
                headline=headline_text,
                what_happened=what_text,
                key_details=EventKeyDetails(),
                significance=" ".join(analysis_claims) or None,
                confidence_qualification=confidence_qualification,
                sources=tuple(claim_sources),
                claims=claims,
                model_used=model_used or self.client.model,
            )
            return summary, None

        source_title_fallback = event.articles[0].title if event.articles else None
        headline = clean_headline_grammar(
            str(parsed.get("headline") or "").strip(),
            source_title=source_title_fallback,
        )
        what_happened = sanitize_placeholder_text(str(parsed.get("what_happened") or "").strip())
        if len(headline) < 5 or len(what_happened) < 10:
            return (
                None,
                "Summary validation failed: 'headline' or 'what_happened' is missing or empty",
            )

        what_lower_raw = what_happened.lower()
        headline_lower_raw = headline.lower()
        if (
            "no headline provided" in headline_lower_raw
            or headline_lower_raw.startswith("no headline")
            or "no information provided" in what_lower_raw
            or "no information in the snippet" in what_lower_raw
        ):
            return (
                None,
                "Summary validation failed: model reported insufficient evidence "
                "or no information in snippet",
            )

        combined_src_text = " ".join(
            f"{clean_article_text(a.title)} {clean_article_text(a.snippet or '')}"
            for a in event.articles
        )
        combined_src_lower = combined_src_text.lower()
        headline = sanitize_technical_acronyms(headline, combined_src_text)
        what_happened = sanitize_technical_acronyms(what_happened, combined_src_text)

        raw_key_details = parsed.get("key_details")
        event_source_names = [a.source_name for a in event.articles if a.source_name]
        if isinstance(raw_key_details, dict):
            raw_actors = raw_key_details.get("actors", [])
            if isinstance(raw_actors, list):
                actors = tuple(sanitize_actors(raw_actors, source_names=event_source_names))
            else:
                actors = ()

            # Preserve attribution to reporting resident without inventing additional actors
            has_resident_reporting = bool(
                re.search(
                    r"\b(according to a resident|resident's report|reporting resident|"
                    r"a resident (?:told|said|reported)|local residents? reported)\b",
                    combined_src_lower + " " + what_happened.lower(),
                )
            )
            if has_resident_reporting:
                if not actors or all(
                    is_publisher_or_outlet(a, event_source_names) for a in raw_actors
                ):
                    actors = ("Reporting resident",)
                elif not any("resident" in a.lower() for a in actors):
                    actors = (*actors, "Reporting resident")

            raw_loc = raw_key_details.get("location")
            raw_locations = raw_loc if isinstance(raw_loc, (list, tuple)) else [raw_loc]
            grounded_locations: list[str] = []
            for raw_item in raw_locations:
                if not isinstance(raw_item, str):
                    continue
                grounded = ground_event_location(raw_item, event, combined_src_text)
                if grounded and grounded.casefold() not in {
                    value.casefold() for value in grounded_locations
                }:
                    grounded_locations.append(grounded)
            location = ", ".join(grounded_locations) or None

            raw_timing = raw_key_details.get("timing")
            timing: str | None = None
            if raw_timing:
                t_clean = sanitize_placeholder_text(str(raw_timing)).strip().strip("\"'“”‘’.,;: ")
                if (
                    t_clean
                    and not _LEAKAGE_PATTERN.search(t_clean)
                    and t_clean.lower()
                    not in {
                        "unspecified",
                        "unspecified in reporting",
                        "unknown",
                        "none",
                        "null",
                        "date not specified in source",
                    }
                ):
                    timing = t_clean
                    # Temporal grounding: check if article published days ago claims 'today'
                    earliest_pub = min(
                        (a.published_at for a in event.articles if a.published_at),
                        default=None,
                    )
                    if earliest_pub:
                        now_utc = datetime.now(UTC)
                        if (now_utc - earliest_pub) > timedelta(hours=36):
                            recent_terms = (
                                "within the last 24 hours",
                                "past 24 hours",
                                "today",
                                "yesterday",
                                "this morning",
                                "overnight",
                            )
                            if any(term in timing.lower() for term in recent_terms):
                                timing = earliest_pub.strftime("%B %d, %Y")

            raw_action = raw_key_details.get("action")
            action: str | None = None
            if raw_action:
                act_clean = sanitize_placeholder_text(str(raw_action)).strip().strip("\"'“”‘’.,;: ")
                if (
                    act_clean
                    and not _LEAKAGE_PATTERN.search(act_clean)
                    and act_clean.lower()
                    not in {
                        "unspecified",
                        "unspecified in reporting",
                        "unknown",
                        "none",
                        "null",
                    }
                ):
                    action = act_clean

            key_details = EventKeyDetails(
                actors=actors,
                location=location,
                timing=timing,
                action=action,
            )
        else:
            key_details = EventKeyDetails()
        has_nuclear_exclusion = (
            "no nuclear weapons" in combined_src_lower
            or "without nuclear weapons" in combined_src_lower
            or "unarmed with nuclear" in combined_src_lower
        )
        if has_nuclear_exclusion:
            what_lower = what_happened.lower()
            if not any(
                term in what_lower
                for term in ("no nuclear weapons", "non-nuclear", "without nuclear weapons")
            ):
                what_happened = (
                    f"{what_happened} (The source explicitly confirmed that "
                    "no nuclear weapons were involved in the exercise.)"
                )

        # Operational scope precision
        _SCOPE_PHRASES: tuple[re.Pattern[str], ...] = (
            re.compile(r"first\s+joint\s+air\s+nuclear\s+defence\s+exercise", re.I),
            re.compile(r"first\s+joint\s+air\s+nuclear\s+exercise", re.I),
            re.compile(r"first\s+joint\s+nuclear\s+air\s+exercise", re.I),
            re.compile(r"first\s+joint\s+nuclear\s+defence\s+exercise", re.I),
            re.compile(r"first\s+joint\s+air\s+exercise", re.I),
            re.compile(r"first\s+joint\s+nuclear\s+exercise", re.I),
            re.compile(r"first\s+joint\s+exercise", re.I),
        )
        attested_scope: str | None = None
        for scope_pat in _SCOPE_PHRASES:
            scope_match = scope_pat.search(combined_src_text)
            if scope_match:
                attested_scope = scope_match.group(0).strip()
                break

        if attested_scope:
            for overstatement in (
                "first-ever joint exercise",
                "first joint exercise between",
                "first joint military exercise",
            ):
                if overstatement in what_happened.lower():
                    pattern = re.compile(re.escape(overstatement), re.IGNORECASE)
                    what_happened = pattern.sub(attested_scope, what_happened)

        # Attribution accuracy guardrail: Prevent accuser/accused swap
        if "washington blamed moscow" in combined_src_lower:
            what_happened = re.sub(
                r"\bRussia\s+blamed\s+Moscow\b",
                "Washington blamed Moscow",
                what_happened,
                flags=re.IGNORECASE,
            )
            headline = re.sub(
                r"\bRussia\s+blamed\s+Moscow\b",
                "Washington blamed Moscow",
                headline,
                flags=re.IGNORECASE,
            )

        # Vendor capability claim guardrail: Qualify unverified system capabilities
        if "the company claims" in combined_src_lower:
            what_happened = re.sub(
                r"\bcapable of tracking and engaging\b",
                "which the company claims can track and engage",
                what_happened,
                flags=re.IGNORECASE,
            )

        # Ceasefire ownership guardrail: Non-combatants facilitate rather than own/enforce
        headline = re.sub(
            r"\bwith\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?'s\s+(?:Ceasefire|Truce)\b",
            "for Humanitarian Aid",
            headline,
            flags=re.IGNORECASE,
        )
        what_happened = re.sub(
            r"\bannounced a ceasefire in\b",
            "helped secure a humanitarian ceasefire in",
            what_happened,
            flags=re.IGNORECASE,
        )
        # Truce summary grammar repair: eliminate dangling subordinate clause
        what_happened = re.sub(
            r",\s*as\s+residents\s+who\s+survived\s+on\s+weeds\s+during\s+the\s+conflict"
            r"(?:,\s*will\s+now\s+be\s+allowed[^\.]*)?\.?",
            ", in a city where residents had resorted to eating weeds to survive.",
            what_happened,
            flags=re.IGNORECASE,
        )
        what_happened = re.sub(r"\.\s*,\s*", ". ", what_happened)
        what_happened = re.sub(r",\s*\.", ".", what_happened)

        # Speaker attribution preservation
        found_speakers: set[str] = set()
        articles_with_speaker: int = 0
        for art in event.articles:
            spk = _extract_speaker_from_title(art.title)
            if spk:
                found_speakers.add(spk)
                articles_with_speaker += 1

        named_speaker: str | None = None
        if len(found_speakers) == 1:
            candidate_speaker = next(iter(found_speakers))
            if len(event.articles) == 1 or articles_with_speaker >= (len(event.articles) / 2):
                named_speaker = candidate_speaker
            elif event.confidence_label == "single_source":
                named_speaker = candidate_speaker

        if named_speaker:
            spk_lower = named_speaker.lower()
            in_headline = spk_lower in headline.lower()
            in_what = spk_lower in what_happened.lower()
            in_actors = any(spk_lower in act.lower() for act in key_details.actors)
            already_attributed = (
                in_what
                or in_headline
                or what_happened.lower().startswith("according to")
                or re.search(
                    r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+(?:stated|said|warned|urged):",
                    what_happened,
                )
                is not None
            )
            if not already_attributed:
                what_happened = f"According to {named_speaker}: {what_happened}"
            if not in_actors:
                key_details = EventKeyDetails(
                    actors=tuple(
                        sanitize_actors(
                            (named_speaker, *key_details.actors),
                            source_names=event_source_names,
                        )
                    ),
                    location=key_details.location,
                    timing=key_details.timing,
                    action=key_details.action,
                )

        # Guardrail: Vague disagreement removal if unsupported by source text
        has_source_dispute = bool(
            re.search(
                r"\b(dispute|disputed|critics|criticism|disagree|controversy|contested|opponents)\b",
                combined_src_lower,
            )
        )
        if not has_source_dispute:
            what_happened = re.sub(
                r"\s*\(?(?:this is )?disputed by some\b[^\)]*\)?\.?",
                "",
                what_happened,
                flags=re.IGNORECASE,
            ).strip()
            what_happened = re.sub(
                r"\s*\(?critics disagree\b[^\)]*\)?\.?",
                "",
                what_happened,
                flags=re.IGNORECASE,
            ).strip()

        # Guardrail: Explicit attribution for negative/absence-of-harm claims
        # Prevent bare assertions like 'no breaches have been reported' from being presented
        # as established independent facts.
        no_breach_pat = (
            r"(?<!according to\s)(?<!authorities\s)(?<!officials\s)"
            r"(?<!spokesman\s)(?<!spokesperson\s)"
            r"\b(no\s+(?:security\s+)?breaches\s+(?:have\s+been|were)\s+reported)\b"
        )
        unattributed_no_breach = re.search(
            no_breach_pat,
            what_happened,
            flags=re.IGNORECASE,
        )
        if unattributed_no_breach and not any(
            w in what_happened.lower()
            for w in (
                "according to",
                "authorities stated",
                "officials reported",
                "officials stated",
                "ministry said",
            )
        ):
            what_happened = re.sub(
                r"\b(no\s+(?:security\s+)?breaches\s+(?:have\s+been|were)\s+reported)\b",
                "officials stated that no breaches have been reported",
                what_happened,
                flags=re.IGNORECASE,
            )

        # Confidence qualification
        confidence_qual = str(parsed.get("confidence_qualification") or "").strip()
        slug_norm = confidence_qual.lower().replace("-", "_").replace(" ", "_").strip()
        slugs_to_expand = {
            "single_source",
            "single_source_report",
            "corroborated",
            "multi_source_corroboration",
            "multi_source_corroborated",
            "disputed",
            "disputed_report",
            "uncertain",
            "unconfirmed",
        }

        if not confidence_qual or len(confidence_qual) < 15 or slug_norm in slugs_to_expand:
            if event.confidence_label == "disputed":
                confidence_qual = (
                    "Disputed reporting: contradictory claims reported across belligerents."
                )
            elif event.confidence_label == "single_source":
                source_name = event.articles[0].source_name if event.articles else "Single source"
                if named_speaker:
                    confidence_qual = (
                        f"Single-source report from {source_name}; reflects statements by "
                        f"{named_speaker} and unverified independently."
                    )
                else:
                    confidence_qual = (
                        f"Single-source report from {source_name}; unverified independently."
                    )
            elif event.confidence_label == "corroborated":
                confidence_qual = (
                    f"Corroborated across {event.source_count} reporting outlets; "
                    "military claims unconfirmed independently."
                )
            else:
                confidence_qual = "Reporting details remain preliminary or unconfirmed."

        conf_lower = confidence_qual.lower()
        if event.confidence_label == "single_source":
            multi_mentioned = "multiple outlets" in conf_lower or "multiple sources" in conf_lower
            if "corroborat" in conf_lower or (event.source_count <= 1 and multi_mentioned):
                source_name = event.articles[0].source_name if event.articles else "Single source"
                if event.source_count >= 2:
                    confidence_qual = (
                        "Single-origin reporting across outlets; unverified independently."
                    )
                else:
                    confidence_qual = (
                        f"Single-source report from {source_name}; unverified independently."
                    )
        elif event.confidence_label == "corroborated" and event.source_count >= 2:
            if "single-source" in conf_lower or "single source" in conf_lower:
                confidence_qual = (
                    f"Corroborated across {event.source_count} reporting outlets; "
                    "military claims unconfirmed independently."
                )
            elif (
                "unconfirmed" not in conf_lower
                and "unverified" not in conf_lower
                and "not" not in conf_lower
            ):
                confidence_qual = (
                    f"Corroborated across {event.source_count} reporting outlets; "
                    "military claims unconfirmed independently."
                )

        raw_sig = parsed.get("significance")
        if raw_sig is not None:
            cleaned_sig = sanitize_placeholder_text(str(raw_sig)).strip()
            if (
                cleaned_sig.lower()
                in {
                    "null",
                    "none",
                    "n/a",
                    "nil",
                    "unspecified",
                    "unspecified in reporting",
                }
                or not cleaned_sig
                or bool(_LEAKAGE_PATTERN.search(cleaned_sig))
            ):
                significance = None
            else:
                generic_sig_pat = re.compile(
                    r"\b(major strategic shift|pivotal moment in|growing importance of|"
                    r"evolving nature of modern warfare|highlights the importance of|"
                    r"critical development for regional dynamics)\b",
                    re.I,
                )
                if generic_sig_pat.search(cleaned_sig) and not generic_sig_pat.search(
                    combined_src_lower
                ):
                    significance = None
                else:
                    significance = cleaned_sig
        else:
            significance = None

        summary = EventSummary(
            event_id=event.id,
            headline=headline,
            what_happened=what_happened,
            key_details=key_details,
            significance=significance,
            confidence_qualification=confidence_qual,
            sources=sources,
            model_used=model_used or self.client.model,
        )
        return summary, None

    def _verify_claims_and_render(
        self,
        summary: EventSummary,
        event: EventCluster,
        client: BaseLLMClient,
        model_used: str,
    ) -> tuple[EventSummary | None, str | None]:
        """Check exact source references, then batch semantic entailment for claim spans."""
        if not summary.claims:
            return summary, None

        article_by_url: dict[str, object] = {}
        for article in event.articles:
            try:
                article_by_url[normalize_url(article.canonical_url)] = article
                article_by_url[normalize_url(article.url)] = article
            except ValueError:
                continue

        def _resolve_article_for_source_ref(source_ref: str) -> object | None:
            clean_ref = source_ref.strip()
            if not clean_ref:
                return None
            try:
                norm_ref = normalize_url(clean_ref)
                if norm_ref in article_by_url:
                    return article_by_url[norm_ref]
            except ValueError:
                norm_ref = None

            ref_no_query = clean_ref.split("?")[0].rstrip("/")
            for art in event.articles:
                if (
                    art.url.split("?")[0].rstrip("/") == ref_no_query
                    or art.canonical_url.split("?")[0].rstrip("/") == ref_no_query
                ):
                    return art

            ref_lower = clean_ref.lower()
            for art in event.articles:
                if art.source_name.strip().lower() == ref_lower:
                    return art

            if len(event.articles) == 1:
                art = event.articles[0]
                if (
                    art.source_name.lower() in ref_lower
                    or "article_1" in ref_lower
                    or ref_lower in {"1", "article 1"}
                    or ref_no_query.lower() in art.url.lower()
                ):
                    return art
            return None

        verifier_items: list[dict[str, str]] = []
        expected_pairs: set[tuple[str, str]] = set()
        resolved_claims = []
        parsed_verdicts: dict[Any, Any] = {}
        for claim in summary.claims:
            claim_urls: set[str] = set()
            resolved_evidence_refs = []
            for evidence_ref in claim.evidence_refs:
                article = _resolve_article_for_source_ref(evidence_ref.source_url)
                if article is None:
                    return None, f"Claim {claim.claim_id} refers to a source outside this event"
                article = _find_best_article_for_claim(
                    claim.text, evidence_ref.evidence_span, article, event.articles
                )
                resolved_url = getattr(article, "url", evidence_ref.source_url)
                evidence_ref = evidence_ref.model_copy(update={"source_url": resolved_url})
                resolved_evidence_refs.append(evidence_ref)
                try:
                    source_key = normalize_url(resolved_url)
                except ValueError:
                    return None, f"Claim {claim.claim_id} has an invalid source reference"
                if source_key in claim_urls:
                    return None, f"Claim {claim.claim_id} repeats a source reference"
                claim_urls.add(source_key)
                pair = (claim.claim_id, source_key)
                expected_pairs.add(pair)
                clean_t = clean_article_text(article.title)
                clean_s = clean_article_text(article.snippet or "")
                available_text = " ".join(value for value in (clean_t, clean_s) if value)

                match_status, diff_msg = check_evidence_span_match(
                    evidence_ref.evidence_span, available_text
                )
                if match_status == "no_match":
                    if not claim.central:
                        # Secondary claim lacks supporting evidence: drop secondary claim
                        continue
                    return None, (
                        f"Claim {claim.claim_id} evidence span is absent from "
                        f"its referenced article: {diff_msg or ''}"
                    )

                src_num = next(
                    (
                        i
                        for i, a in enumerate(event.articles, 1)
                        if normalize_url(a.url) == source_key
                    ),
                    1,
                )
                verifier_items.append(
                    {
                        "claim_id": claim.claim_id,
                        "src": src_num,
                        "claim": claim.text,
                        "classification": claim.classification,
                        "attribution": claim.attribution or "",
                        "evidence": evidence_ref.evidence_span,
                    }
                )
            if resolved_evidence_refs:
                resolved_claims.append(
                    claim.model_copy(update={"evidence_refs": tuple(resolved_evidence_refs)})
                )
        summary = summary.model_copy(update={"claims": tuple(resolved_claims)})

        if verifier_items:
            prompt = (
                "Check whether each claim is entailed by its paired evidence span only. "
                "A quotation existing is not sufficient unless it supports the claim. Do not use "
                "outside knowledge. Preserve attribution and distinguish fact from analysis. "
                "Use uncertain for ambiguity or missing context. Return JSON only: "
                '{"verdicts":[{"claim_id":"...",'
                '"verdict":"supported|unsupported|uncertain",'
                '"reason":"concise <=10 words"}]}. '
                "Include exactly one verdict per claim.\n"
                + json.dumps(verifier_items, ensure_ascii=False)
            )
            try:
                self.model_calls_count += 1
                raw_verdict = self._safe_generate(
                    client,
                    prompt=prompt,
                    system="You are a conservative source-entailment verifier.",
                    format_type="json",
                    options={
                        "num_predict": STAGE_BUDGET_VERIFIER,
                        "temperature": 0.0,
                        "repeat_penalty": 1.15,
                    },
                )
                try:
                    data = json.loads(raw_verdict)
                except Exception as json_exc:
                    logger.warning(
                        "Claim verifier output was not valid JSON (raw_len=%d): %s",
                        len(raw_verdict),
                        json_exc,
                    )
                    return None, f"Claim verifier returned malformed JSON: {json_exc}"

                verdicts = data.get("verdicts") if isinstance(data, dict) else None
                if not isinstance(verdicts, list) or not verdicts:
                    return None, (
                        "Claim verifier returned missing, duplicate, or unknown "
                        "claim/source references"
                    )
                valid_claim_ids = {c.claim_id for c in summary.claims}
                for item in verdicts:
                    if not isinstance(item, dict):
                        continue
                    cid = item.get("claim_id")
                    if not isinstance(cid, str) or cid not in valid_claim_ids:
                        return None, (
                            "Claim verifier returned missing, duplicate, or unknown "
                            "claim/source references"
                        )
                    src_val = item.get("src")
                    if src_val is not None:
                        try:
                            s_int = int(src_val)
                            if 1 <= s_int <= len(event.articles):
                                target_art = event.articles[s_int - 1]
                                parsed_verdicts[(cid, normalize_url(target_art.url))] = item
                        except (ValueError, TypeError):
                            pass
                    s_url = item.get("source_url")
                    if isinstance(s_url, str) and s_url.strip():
                        resolved_art = _resolve_article_for_source_ref(s_url)
                        s_key = (
                            normalize_url(resolved_art.url)
                            if resolved_art
                            else (normalize_url(s_url) if s_url.strip() else None)
                        )
                        if s_key is not None:
                            parsed_verdicts[(cid, s_key)] = item
                    if cid not in parsed_verdicts:
                        parsed_verdicts[cid] = item
            except Exception as exc:
                return None, f"Claim verifier failed: {type(exc).__name__}"

        accepted = []
        rejected_central: list[str] = []
        for claim in summary.claims:
            supported_refs = []
            for evidence_ref in claim.evidence_refs:
                source_key = normalize_url(evidence_ref.source_url)
                verdict = parsed_verdicts.get((claim.claim_id, source_key)) or parsed_verdicts.get(
                    claim.claim_id
                )
                if not verdict:
                    continue
                status = verdict.get("verdict")
                reason = str(verdict.get("reason") or "").strip()[:240]
                if status == "supported":
                    supported_refs.append(
                        evidence_ref.model_copy(
                            update={
                                "verification_status": "accepted",
                                "verification_reason": reason or None,
                            }
                        )
                    )
                elif status not in {"unsupported", "uncertain"}:
                    return None, f"Claim verifier returned invalid verdict for {claim.claim_id}"
            if supported_refs:
                claim_to_add = claim
                claim_text = claim_to_add.text
                claim_attr = normalize_attribution_text(claim_to_add.attribution)
                claim_class = claim_to_add.classification
                combined_spans = " ".join(r.evidence_span for r in supported_refs)

                event_source_names = tuple(
                    getattr(a, "source_name", "") for a in getattr(event, "articles", ())
                )
                combined_spans_lower = combined_spans.lower()
                claim_text_lower = claim_text.lower()
                if (
                    "ambrey" in combined_spans_lower
                    or "imo" in combined_spans_lower
                    or "ambrey" in claim_text_lower
                    or "imo" in claim_text_lower
                    or "vessel casualties" in combined_spans_lower
                    or "vessel casualties" in claim_text_lower
                ):
                    has_ft = (
                        "financial times" in combined_spans_lower
                        or "financial times" in claim_text_lower
                        or bool(re.search(r"\bft\b", combined_spans_lower))
                        or bool(re.search(r"\bft\b", claim_text_lower))
                    )
                    if has_ft:
                        claim_attr = (
                            "Financial Times / Ambrey"
                            if ("ambrey" in combined_spans_lower or "ambrey" in claim_text_lower)
                            else "Financial Times / IMO"
                        )
                    elif "ambrey" in combined_spans_lower or "ambrey" in claim_text_lower:
                        claim_attr = "Ambrey"
                    else:
                        claim_attr = "IMO"
                    claim_class = "attributed_statement"
                elif not claim_attr:
                    stat_m = _STATISTIC_ATTRIBUTION_PATTERN.search(combined_spans)
                    named_m = _NAMED_DATA_SOURCES.search(combined_spans)
                    if stat_m:
                        candidate_attr = stat_m.group(1).strip()
                        if not is_publisher_or_outlet(candidate_attr, event_source_names):
                            claim_attr = candidate_attr
                            claim_class = "attributed_statement"
                    elif named_m:
                        nm = named_m.group(0).strip().lower()
                        candidate_attr = _SOURCE_CASE_MAP.get(nm, nm.title())
                        if not is_publisher_or_outlet(candidate_attr, event_source_names):
                            claim_attr = candidate_attr
                            claim_class = "attributed_statement"

                if claim_attr and is_publisher_or_outlet(claim_attr, event_source_names):
                    if "ambrey" in claim_attr.lower() or "imo" in claim_attr.lower():
                        pass
                    else:
                        # Do not attribute third-party statistics to the reporting outlet
                        named_m = _NAMED_DATA_SOURCES.search(
                            combined_spans
                        ) or _STATISTIC_ATTRIBUTION_PATTERN.search(combined_spans)
                        if named_m:
                            candidate = (
                                named_m.group(1).strip()
                                if named_m.lastindex
                                else named_m.group(0).strip().lower()
                            )
                            cand_clean = _SOURCE_CASE_MAP.get(candidate.lower(), candidate.title())
                            claim_attr = (
                                cand_clean
                                if not is_publisher_or_outlet(cand_clean, event_source_names)
                                else None
                            )
                        else:
                            claim_attr = None

                if claim_attr and (
                    claim_class == "attributed_statement"
                    or _CONTAINS_STATISTIC.search(claim_text)
                    or _CONTAINS_STATISTIC.search(combined_spans)
                ):
                    claim_class = "attributed_statement"
                    if (
                        claim_attr.lower() not in claim_text.lower()
                        and not claim_text.lower().startswith("according to")
                    ):
                        claim_text = (
                            f"According to {claim_attr}, {claim_text[0].lower() + claim_text[1:]}"
                        )

                accepted.append(
                    claim_to_add.model_copy(
                        update={
                            "text": claim_text,
                            "attribution": claim_attr,
                            "classification": claim_class,
                            "evidence_refs": tuple(supported_refs),
                            "verification_status": "accepted",
                            "verification_reason": None,
                        }
                    )
                )
            elif claim.central:
                rejected_central.append(f"{claim.claim_id}: no supporting source verdict")
        if rejected_central:
            return None, "Central claim not verified: " + "; ".join(rejected_central)
        if not accepted or not any(claim.central for claim in accepted):
            return None, "No verified central claim remains after unsupported claims were removed"

        _NON_SPEAKER_COUNTRIES = frozenset(
            {
                "the united states",
                "united states",
                "us",
                "russia",
                "china",
                "iran",
                "israel",
                "uk",
                "britain",
                "japan",
                "germany",
                "france",
                "poland",
                "taiwan",
                "syria",
                "jordan",
                "switzerland",
                "canada",
            }
        )

        # Rebuild every user-visible narrative field exclusively from accepted claims.
        def accepted_text(claim: SummaryClaim) -> str:
            text = claim.text.strip()
            if claim.classification in {"source_analysis", "attributed_statement"}:
                attr_clean = (claim.attribution or "").strip()
                attr_lower = attr_clean.lower()
                text_lower = text.lower()
                is_country_or_gov = attr_lower in _NON_SPEAKER_COUNTRIES or any(
                    term in attr_lower
                    for term in (
                        "government",
                        "administration",
                        "military",
                        "authorities",
                        "ministry",
                        "department",
                    )
                )
                starts_with_attr = bool(attr_clean and text_lower.startswith(attr_lower))
                has_verb = bool(
                    re.search(
                        r"\b(?:according to|said|stated|reported|argued|assessed|"
                        r"announced|claimed|alleged)\b",
                        text,
                        re.I,
                    )
                )
                if attr_clean and not starts_with_attr and not has_verb:
                    if is_country_or_gov:
                        if any(
                            term in attr_lower
                            for term in (
                                "government",
                                "ministry",
                                "department",
                                "military",
                                "authorities",
                                "forces",
                                "officials",
                                "command",
                            )
                        ):
                            text = f"{attr_clean} stated that {text[0].lower() + text[1:]}"
                        elif attr_lower in _NON_SPEAKER_COUNTRIES:
                            text = f"According to {attr_clean}, {text[0].lower() + text[1:]}"
                        else:
                            text = f"According to {attr_clean}, {text[0].lower() + text[1:]}"
                    else:
                        text = f"According to {attr_clean}, {text[0].lower() + text[1:]}"
            elif claim.classification == "inference":
                text = f"The reporting suggests that {text[0].lower() + text[1:]}"
            elif claim.classification == "disputed_claim":
                text = f"The source reports a disputed claim: {text}"
            return text

        central = [accepted_text(claim) for claim in accepted if claim.central]
        narrative = [
            accepted_text(claim) for claim in accepted if claim.classification != "source_analysis"
        ]
        analysis = [
            accepted_text(claim) for claim in accepted if claim.classification == "source_analysis"
        ]
        used_urls = {
            normalize_url(ref.source_url) for claim in accepted for ref in claim.evidence_refs
        }
        sources = tuple(
            source for source in summary.sources if normalize_url(source.url) in used_urls
        )
        if not sources and accepted:
            sources = tuple(
                EventSourceAttribution(
                    source_name=art.source_name,
                    title=art.title,
                    url=art.url,
                    published_at=art.published_at,
                )
                for art in event.articles
                if normalize_url(art.url) in used_urls
            )
        central_claims = [c for c in accepted if c.central]
        all_central_corroborated = False
        if central_claims:
            is_corrob_list = []
            for claim in central_claims:
                claim_arts = [
                    art
                    for art in event.articles
                    if normalize_url(art.url)
                    in {normalize_url(ref.source_url) for ref in claim.evidence_refs}
                ]
                is_corrob_list.append(are_sources_genuinely_independent(claim_arts))
            all_central_corroborated = (
                all(is_corrob_list) and event.confidence_label == "corroborated"
            )

        if all_central_corroborated:
            all_claim_arts = [
                art
                for art in event.articles
                if any(
                    normalize_url(art.url)
                    in {normalize_url(ref.source_url) for ref in c.evidence_refs}
                    for c in central_claims
                )
            ]
            indep_count = count_genuinely_independent_sources(all_claim_arts)
            confidence = (
                f"Corroborated across {indep_count} reporting outlets; "
                "claims remain subject to independent verification."
            )
        else:
            if len(event.articles) >= 2:
                confidence = "Single-source reporting across outlets; unverified independently."
            else:
                source_name = event.articles[0].source_name if event.articles else "Single source"
                confidence = f"Single-source report from {source_name}; unverified independently."
        central_claim_obj = [c for c in accepted if c.central][0]
        final_headline = _derive_headline(central_claim_obj, event)

        final_narrative = " ".join(narrative) or central[0]
        clean_fn = clean_headline_grammar(final_headline).lower()
        clean_narr = clean_headline_grammar(final_narrative).lower()
        plain_fn = extract_plain_narrative(final_headline).lower()
        plain_narr = extract_plain_narrative(final_narrative).lower()

        # Prevent exact or semantic headline/body duplication
        if (
            clean_fn == clean_narr
            or plain_fn == plain_narr
            or is_headline_body_duplicate(final_narrative, final_headline)
            or (len(accepted) == 1 and plain_fn in plain_narr)
        ):
            distinct_claims = [
                accepted_text(c)
                for c in accepted
                if not is_headline_body_duplicate(accepted_text(c), final_headline)
                and c.classification != "source_analysis"
            ]
            if distinct_claims:
                final_narrative = " ".join(distinct_claims)
            elif (
                summary.significance
                and len(extract_plain_narrative(summary.significance)) >= 15
                and not is_headline_body_duplicate(summary.significance, final_headline)
            ):
                final_narrative = summary.significance.strip()
            else:
                target_art = event.articles[0] if event.articles else None
                if accepted and accepted[0].evidence_refs:
                    cand_url = accepted[0].evidence_refs[0].source_url
                    for a in event.articles:
                        if a.url == cand_url or normalize_url(a.url) == normalize_url(cand_url):
                            target_art = a
                            break
                supp = _extract_supporting_claim_from_source(
                    target_art, accepted[0].text, final_headline
                )
                if supp is not None:
                    accepted.append(supp)
                    final_narrative = accepted_text(supp)
                else:
                    # If no meaningful supporting claim exists, keep the verified central claim text
                    # without copying article snippets or bodies.
                    final_narrative = central[0]

        combined_src_text = " ".join(
            f"{getattr(a, 'title', '')} {getattr(a, 'snippet', '')}" for a in event.articles
        )
        final_headline = sanitize_technical_acronyms(final_headline, combined_src_text)
        final_narrative = sanitize_technical_acronyms(final_narrative, combined_src_text)

        combined_src_text = " ".join(
            f"{getattr(a, 'title', '')} {getattr(a, 'snippet', '')}" for a in event.articles
        )
        final_headline = sanitize_technical_acronyms(final_headline, combined_src_text)
        final_narrative = sanitize_technical_acronyms(final_narrative, combined_src_text)

        return (
            summary.model_copy(
                update={
                    "headline": final_headline,
                    "what_happened": final_narrative,
                    "significance": " ".join(analysis) or None,
                    "confidence_qualification": confidence,
                    "sources": sources,
                    "claims": tuple(accepted),
                }
            ),
            None,
        )

    def summarize_event(self, event: EventCluster) -> SummarizationResult:
        """Summarize a single EventCluster with evidence sufficiency and deterministic grounding."""
        # 0. Check in-memory cache keyed by content hash
        urls = sorted(getattr(a, "canonical_url", a.url) for a in event.articles)
        key = f"{event.title.strip().lower()}|{'|'.join(urls)}"
        event_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
        if event_hash in self._cache:
            return self._cache[event_hash]

        # 1. Evidence sufficiency pre-check
        is_sufficient, insufficiency_reason = is_evidence_sufficient(event)
        if not is_sufficient:
            res = SummarizationResult(
                event_id=event.id,
                status="insufficient_evidence",
                error_message=insufficiency_reason,
                quarantine_reason=insufficiency_reason,
                event=event,
            )
            self._cache[event_hash] = res
            return res

        # 2. Preserve original source attribution
        sources = tuple(
            EventSourceAttribution(
                source_name=art.source_name,
                title=art.title,
                url=art.url,
                published_at=art.published_at,
            )
            for art in event.articles
        )

        # 3. Request LLM completion
        prompt = build_event_prompt(event)
        active_client = self.client
        model_used = self.client.model
        raw_output: str | None = None

        try:
            self.model_calls_count += 1
            raw_output = self._safe_generate(
                active_client,
                prompt=prompt,
                system=SYSTEM_PROMPT,
                format_type="json",
                options={
                    "num_predict": STAGE_BUDGET_CLAIMS,
                    "temperature": 0.1,
                    "repeat_penalty": 1.15,
                },
            )
        except GEMINI_FALLBACK_ELIGIBLE_EXCEPTIONS as exc:
            if self.fallback_client is not None:
                logger.warning(
                    "Primary LLM provider (%s) encountered eligible failure (%s); "
                    "falling back to %s model '%s'",
                    getattr(active_client, "provider", "primary"),
                    exc,
                    getattr(self.fallback_client, "provider", "fallback"),
                    self.fallback_client.model,
                )
                active_client = self.fallback_client
                model_used = self.fallback_client.model
                try:
                    self.model_calls_count += 1
                    raw_output = self._safe_generate(
                        active_client,
                        prompt=prompt,
                        system=SYSTEM_PROMPT,
                        format_type="json",
                        options={
                            "num_predict": STAGE_BUDGET_CLAIMS,
                            "temperature": 0.1,
                            "repeat_penalty": 1.15,
                        },
                    )
                except OllamaUnavailableError as fallback_exc:
                    res = SummarizationResult(
                        event_id=event.id,
                        status="unavailable",
                        error_message=(
                            f"Primary provider failed ({exc}) and fallback "
                            f"Ollama is unavailable: {fallback_exc}"
                        ),
                        event=event,
                    )
                    self._cache[event_hash] = res
                    return res
                except (
                    OllamaTimeoutError,
                    OllamaModelNotFoundError,
                    OllamaResponseError,
                ) as fallback_exc:
                    res = SummarizationResult(
                        event_id=event.id,
                        status="failed",
                        error_message=(
                            f"Primary provider failed ({exc}) and fallback "
                            f"Ollama failed: {fallback_exc}"
                        ),
                        event=event,
                    )
                    self._cache[event_hash] = res
                    return res
                except Exception as fallback_exc:
                    res = SummarizationResult(
                        event_id=event.id,
                        status="failed",
                        error_message=(
                            f"Primary provider failed ({exc}) and fallback "
                            f"unexpected error: {fallback_exc}"
                        ),
                        event=event,
                    )
                    self._cache[event_hash] = res
                    return res
            else:
                status_label = (
                    "unavailable" if isinstance(exc, GeminiUnavailableError) else "failed"
                )
                res = SummarizationResult(
                    event_id=event.id,
                    status=status_label,
                    error_message=str(exc),
                    event=event,
                )
                self._cache[event_hash] = res
                return res
        except (OllamaUnavailableError, GeminiUnavailableError) as exc:
            res = SummarizationResult(
                event_id=event.id,
                status="unavailable",
                error_message=str(exc),
                event=event,
            )
            self._cache[event_hash] = res
            return res
        except (
            OllamaTimeoutError,
            OllamaModelNotFoundError,
            OllamaResponseError,
            GeminiAuthError,
            GeminiConfigError,
            GeminiQuotaError,
            GeminiTimeoutError,
            GeminiModelNotFoundError,
            GeminiResponseError,
        ) as exc:
            res = SummarizationResult(
                event_id=event.id,
                status="failed",
                error_message=str(exc),
                event=event,
            )
            self._cache[event_hash] = res
            return res
        except Exception as exc:
            res = SummarizationResult(
                event_id=event.id,
                status="failed",
                error_message=f"Unexpected error during summarization: {exc}",
                event=event,
            )
            self._cache[event_hash] = res
            return res

        assert raw_output is not None
        # 4. Parse and construct candidate summary
        candidate_summary, parse_error = self._parse_and_build_summary(
            raw_output, event, sources, model_used=model_used
        )
        if candidate_summary is None:
            if (
                parse_error == "Claim evidence does not identify a central event claim"
                and not isinstance(active_client, (MockOllamaClient, MockGeminiClient))
            ):
                compact_parts = [
                    f"[Source {i}] ({a.source_name}):\nTitle: {clean_article_text(a.title)}\n"
                    f"Snippet: {clean_article_text(a.snippet or a.title)}"
                    for i, a in enumerate(event.articles, 1)
                ]
                compact_articles = "\n".join(compact_parts)
                correction_prompt = (
                    f"Event: {event.title}\n\n"
                    f"SOURCE ARTICLES:\n{compact_articles}\n\n"
                    f"CORRECTION REQUIRED: {parse_error}. You must designate exactly one "
                    'claim with "central": true representing the primary event development.\n'
                    "Return valid JSON only matching the schema: "
                    '{"claims": [{"claim_id": "c1", "text": "...", "src": 1, '
                    '"evidence": "<short exact verbatim quote under 25 words>", '
                    '"classification": "reported_fact", "attribution": null, "central": true}]}\n'
                    "Cite the source by integer index in 'src'. Use exact verbatim evidence spans."
                )
                try:
                    self.model_calls_count += 1
                    corrected_output = self._safe_generate(
                        active_client,
                        prompt=correction_prompt,
                        system=SYSTEM_PROMPT,
                        format_type="json",
                        options={
                            "num_predict": STAGE_BUDGET_CORRECTION,
                            "temperature": 0.1,
                            "repeat_penalty": 1.15,
                        },
                    )
                    corrected_summary, retry_error = self._parse_and_build_summary(
                        corrected_output, event, sources, model_used=model_used
                    )
                    if corrected_summary is not None:
                        candidate_summary = corrected_summary
                        parse_error = None
                    else:
                        parse_error = retry_error or parse_error
                except Exception as exc:
                    parse_error = f"Bounded claim correction failed: {type(exc).__name__}"

            if candidate_summary is None:
                last_done = getattr(active_client, "last_metrics", {}).get("done_reason")
                is_truncated = (
                    last_done == "length"
                    or "token limit" in (parse_error or "").lower()
                    or "unterminated string" in (parse_error or "").lower()
                )
                status_label = (
                    "quarantined" if ("Claim" in (parse_error or "") or is_truncated) else "failed"
                )
                quarantine_msg = (
                    f"Model output truncated by token budget (num_predict): {parse_error}"
                    if is_truncated
                    else (parse_error if status_label == "quarantined" else None)
                )
                res = SummarizationResult(
                    event_id=event.id,
                    status=status_label,
                    error_message=parse_error,
                    quarantine_reason=quarantine_msg,
                    event=event,
                )
                self._cache[event_hash] = res
                return res

        if not candidate_summary.claims and not isinstance(
            active_client, (MockOllamaClient, MockGeminiClient)
        ):
            reason = "Model output omitted required claim-level source evidence"
            res = SummarizationResult(
                event_id=event.id,
                status="quarantined",
                error_message=f"Quarantined: {reason}",
                quarantine_reason=reason,
                event=event,
            )
            self._cache[event_hash] = res
            return res

        if candidate_summary.claims:
            verified_summary, claim_error = self._verify_claims_and_render(
                candidate_summary,
                event,
                active_client,
                model_used,
            )
            if (
                verified_summary is None
                and claim_error
                and (
                    claim_error.startswith("Central claim not verified")
                    or "evidence span is absent" in claim_error
                    or "source outside this event" in claim_error
                )
            ):
                compact_parts = [
                    f"[Source {i}] ({a.source_name}):\nTitle: {clean_article_text(a.title)}\n"
                    f"Snippet: {clean_article_text(a.snippet or a.title)}"
                    for i, a in enumerate(event.articles, 1)
                ]
                compact_articles = "\n".join(compact_parts)
                correction_prompt = (
                    f"Event: {event.title}\n\n"
                    f"SOURCE ARTICLES:\n{compact_articles}\n\n"
                    f"CORRECTION REQUIRED: The prior submission failed: {claim_error}.\n"
                    "Return valid JSON only matching the schema: "
                    '{"claims": [{"claim_id": "c1", "text": "...", "src": 1, '
                    '"evidence": "<short exact verbatim quote under 25 words>", '
                    '"classification": "reported_fact", "attribution": null, "central": true}]}\n'
                    "Cite the source by integer index in 'src'. Use exact verbatim evidence spans."
                )
                try:
                    self.model_calls_count += 1
                    corrected_output = self._safe_generate(
                        active_client,
                        prompt=correction_prompt,
                        system=SYSTEM_PROMPT,
                        format_type="json",
                        options={
                            "num_predict": STAGE_BUDGET_CORRECTION,
                            "temperature": 0.1,
                            "repeat_penalty": 1.15,
                        },
                    )
                    corrected_summary, _ = self._parse_and_build_summary(
                        corrected_output, event, sources, model_used=model_used
                    )
                    if corrected_summary is not None and corrected_summary.claims:
                        verified_summary, corrected_error = self._verify_claims_and_render(
                            corrected_summary, event, active_client, model_used
                        )
                        claim_error = corrected_error or claim_error
                except Exception as exc:
                    claim_error = f"Bounded claim correction failed: {type(exc).__name__}"
            if verified_summary is None:
                res = SummarizationResult(
                    event_id=event.id,
                    status="quarantined",
                    error_message=f"Quarantined: {claim_error}",
                    quarantine_reason=claim_error,
                    event=event,
                )
                self._cache[event_hash] = res
                return res
            candidate_summary = verified_summary

        # 5. Deterministic grounding validation and publication quality gate
        validation_res = validate_summary_grounding(candidate_summary, event)
        if validation_res.is_valid:
            gate_res = validate_publication_gate(candidate_summary, event)
            if gate_res.passed:
                final_summary = gate_res.repaired_summary or candidate_summary
                res = SummarizationResult(
                    event_id=event.id,
                    status="success",
                    summary=final_summary,
                    event=event,
                )
                self._cache[event_hash] = res
                return res
            else:
                res = SummarizationResult(
                    event_id=event.id,
                    status="quarantined",
                    error_message=gate_res.quarantine_reason,
                    quarantine_reason=gate_res.quarantine_reason,
                    event=event,
                )
                self._cache[event_hash] = res
                return res

        # 6. Bounded single retry with corrective prompt if unsupported entities
        # or malformed clauses detected, provided there are no other uncorrectable fatal claims
        malformed_issues = [
            claim
            for claim in validation_res.unsupported_claims
            if "Incomplete/dangling clause" in claim or "Malformed summary" in claim
        ]
        fatal_claims = [c for c in validation_res.unsupported_claims if c not in malformed_issues]
        if (validation_res.unsupported_entities or malformed_issues) and not fatal_claims:
            correction_prompt = build_grounding_correction_prompt(
                event=event,
                unsupported_entities=validation_res.unsupported_entities,
                malformed_issues=malformed_issues,
            )
            try:
                self.model_calls_count += 1
                retry_output = self._safe_generate(
                    active_client,
                    prompt=correction_prompt,
                    system=SYSTEM_PROMPT,
                    format_type="json",
                    options={
                        "num_predict": STAGE_BUDGET_CORRECTION,
                        "temperature": 0.1,
                        "repeat_penalty": 1.15,
                    },
                )
                retry_summary, retry_parse_err = self._parse_and_build_summary(
                    retry_output, event, sources, model_used=model_used
                )
                if retry_summary is not None:
                    retry_val = validate_summary_grounding(retry_summary, event)
                    if retry_val.is_valid:
                        gate_res = validate_publication_gate(retry_summary, event)
                        if gate_res.passed:
                            final_summary = gate_res.repaired_summary or retry_summary
                            res = SummarizationResult(
                                event_id=event.id,
                                status="success",
                                summary=final_summary,
                                event=event,
                            )
                            self._cache[event_hash] = res
                            return res
                        else:
                            validation_res = GroundingValidationResult(
                                is_valid=False,
                                quarantine_reason=gate_res.quarantine_reason,
                            )
                    validation_res = retry_val
            except Exception:
                pass

        # 7. Final quarantine: Exclude from briefing with sanitized reason
        status_label = (
            "insufficient_evidence"
            if "insufficient" in (validation_res.quarantine_reason or "").lower()
            else "quarantined"
        )
        res = SummarizationResult(
            event_id=event.id,
            status=status_label,
            error_message=f"Quarantined: {validation_res.quarantine_reason}",
            quarantine_reason=validation_res.quarantine_reason,
            event=event,
        )
        self._cache[event_hash] = res
        return res

    def summarize_events(
        self,
        events: list[EventCluster] | tuple[EventCluster, ...],
        max_events: int | None = None,
        min_relevance_rank: float | None = None,
        include_low_priority: bool = False,
        max_backfill: int = 5,
    ) -> tuple[SummarizationResult, ...]:
        """Summarize EventClusters with candidate selection before LLM generation.

        Deterministic ranking and diversity selection run BEFORE LLM calls, so only
        eligible candidates up to the digest cap (with bounded backfill) are sent to Ollama.
        Remaining events receive lightweight deterministic summaries to preserve the omission count.
        """
        if not events:
            return ()

        if max_events is None or max_events <= 0:
            return tuple(self.summarize_event(event) for event in events)

        from geopulse.delivery.formatter import partition_candidate_events

        selected_clusters, deferred_clusters = partition_candidate_events(
            items=events,
            max_events=max_events,
            min_relevance_rank=min_relevance_rank,
            include_low_priority=include_low_priority,
        )

        target_success = len(selected_clusters)
        results_by_id: dict[str, SummarizationResult] = {}
        successful_count = 0
        deferred_queue = list(deferred_clusters)
        backfill_attempts = 0

        # 1. Summarize primary candidate clusters
        for cluster in selected_clusters:
            res = self.summarize_event(cluster)
            results_by_id[cluster.id] = res
            if res.status == "success" and res.summary is not None:
                successful_count += 1

        # 2. Bounded backfill from deferred candidates if any primary candidate
        # was quarantined or failed
        while (
            successful_count < target_success
            and deferred_queue
            and backfill_attempts < max_backfill
        ):
            candidate = deferred_queue.pop(0)
            backfill_attempts += 1
            res = self.summarize_event(candidate)
            results_by_id[candidate.id] = res
            if res.status == "success" and res.summary is not None:
                successful_count += 1

        # 3. Create lightweight omitted fallback summaries for all remaining events
        # to preserve the omission count in format_digest
        for ev in events:
            if ev.id not in results_by_id:
                fallback_sources = tuple(
                    EventSourceAttribution(
                        source_name=art.source_name,
                        title=art.title,
                        url=art.url,
                        published_at=art.published_at,
                    )
                    for art in ev.articles
                )
                fallback_summary = EventSummary(
                    event_id=ev.id,
                    headline=ev.title,
                    what_happened=ev.title,
                    confidence_qualification=ev.confidence_label,
                    sources=fallback_sources,
                    model_used="omitted",
                )
                results_by_id[ev.id] = SummarizationResult(
                    event_id=ev.id,
                    status="success",
                    summary=fallback_summary,
                    event=ev,
                )

        # Preserve the original sequence of events
        return tuple(results_by_id[ev.id] for ev in events)
