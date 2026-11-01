"""Dedicated Telegram digest formatter producing source-grounded HTML briefings."""

import html
import re
from collections.abc import Sequence
from datetime import date, datetime
from typing import Any

from geopulse.classification.models import TopicCategory
from geopulse.clustering.models import ConfidenceLabel, EventCluster
from geopulse.ingestion.normalization import normalize_url
from geopulse.summarization.models import EventSummary, SummarizationResult
from geopulse.summarization.publication_gate import normalize_attribution_text
from geopulse.summarization.validator import (
    are_sources_genuinely_independent,
    deduplicate_actor_acronyms,
)

# Maximum character length per Telegram message
MAX_TELEGRAM_MESSAGE_LENGTH: int = 4096
# Target safe limit per chunk to leave headroom for Telegram metadata
SAFE_CHUNK_LIMIT: int = 4000

CATEGORY_LABELS: dict[TopicCategory, str] = {
    TopicCategory.ARMED_CONFLICT: "⚔️ ARMED CONFLICT",
    TopicCategory.MILITARY_OPERATIONS: "🛡️ MILITARY OPERATIONS",
    TopicCategory.DEFENCE_POLICY: "🏛️ DEFENCE POLICY",
    TopicCategory.WEAPONS_PROCUREMENT: "🚀 WEAPONS PROCUREMENT",
    TopicCategory.DIPLOMACY: "🕊️ DIPLOMACY",
    TopicCategory.SANCTIONS: "⚖️ SANCTIONS",
    TopicCategory.TERRORISM_INSURGENCY: "⚠️ TERRORISM & INSURGENCY",
    TopicCategory.CYBER_HYBRID_WARFARE: "💻 CYBER & HYBRID WARFARE",
    TopicCategory.SECURITY_INCIDENT: "🚨 SECURITY INCIDENT",
    TopicCategory.OTHER_GEOPOLITICAL: "🌐 GEOPOLITICAL",
    TopicCategory.UNCERTAIN: "❓ UNCERTAIN",
}

CONFIDENCE_BADGES: dict[ConfidenceLabel | str, str] = {
    "corroborated": "[CORROBORATED]",
    "official_announcement": "[OFFICIAL ANNOUNCEMENT]",
    "single_source": "[SINGLE SOURCE]",
    "disputed": "[DISPUTED]",
    "uncertain": "[PRELIMINARY]",
}

CATEGORY_PRIORITY: dict[TopicCategory, int] = {
    # Tier 1: Armed conflict, active military operations, critical security incidents
    TopicCategory.ARMED_CONFLICT: 1,
    TopicCategory.SECURITY_INCIDENT: 2,
    TopicCategory.TERRORISM_INSURGENCY: 3,
    TopicCategory.MILITARY_OPERATIONS: 4,
    TopicCategory.CYBER_HYBRID_WARFARE: 5,
    # Tier 2: Consequential diplomatic developments & strategic defence policy
    TopicCategory.DIPLOMACY: 6,
    TopicCategory.DEFENCE_POLICY: 7,
    # Tier 3: Weapons procurement & economic sanctions
    TopicCategory.WEAPONS_PROCUREMENT: 8,
    TopicCategory.SANCTIONS: 9,
    # Tier 4: General geopolitical & uncertain
    TopicCategory.OTHER_GEOPOLITICAL: 10,
    TopicCategory.UNCERTAIN: 11,
}

# Consequential diplomacy patterns (peace talks, ceasefires, summits, prisoner exchanges)
CONSEQUENTIAL_DIPLOMACY_TERMS: re.Pattern[str] = re.compile(
    r"\b(peace talks?|ceasefire|peace negotiations?|truce|armistice|peace deal|peace accord|"
    r"peace treaty|peace summit|cessation of hostilities|hostage release|prisoner exchange|"
    r"diplomatic breakthrough)\b",
    re.I,
)

# Routine historical military commemoration patterns
HISTORICAL_COMMEMORATION_TERMS: re.Pattern[str] = re.compile(
    r"\b(first world war|second world war|wwi|wwii|great war|war graves commission|"
    r"cwgc|reburial|reinterment|buried with (?:full )?military honours?|remains of soldier|"
    r"burial of soldier|fallen soldier.{0,30}identified|identified after \d+ years|"
    r"centenary of)\b",
    re.I,
)

CEREMONIAL_ACTION_PAT: re.Pattern[str] = re.compile(
    r"\b(laid to rest|burial|reburied|re-interred|reinterment|"
    r"buried with (?:full )?military honours?|remains identified|"
    r"remains of soldier|burial of soldier|fallen soldier.{0,30}identified|"
    r"identified after \d+ years|cwgc|commonwealth war graves|"
    r"unmarked grave|memorial dedication|centenary of|centenary commemoration|"
    r"ceremonial service)\b",
    re.I,
)

HISTORICAL_BATTLEFIELD_PAT: re.Pattern[str] = re.compile(
    r"\b(somme|western front|ypres|passchendaele|gallipoli|normandy|"
    r"first world war|second world war|world war (i|ii|one|two)|wwi\b|wwii\b|"
    r"great war|century-old|historical remains)\b",
    re.I,
)

DEFAULT_MIN_RELEVANCE_RANK: float = 9.0


EVERGREEN_GUIDANCE_PATTERNS: re.Pattern[str] = re.compile(
    r"\b(guidance(?:\s*:)?|defence\s+safety\s+management\s+system|jsp\s*\d+|"
    r"joint\s+service\s+publication|standard\s+operating\s+procedures?|"
    r"code\s+of\s+practice|armed\s+forces\s+pensions?|reference\s+table|"
    r"low[- ]flying\s+(?:training\s+)?timetable|mod[- ]sponsored\s+(?:air\s+)?exercises?)\b",
    re.I,
)


def is_evergreen_guidance(category: TopicCategory | None, text: str) -> bool:
    """Identify administrative guidance, manuals, standard procedures, and routine timetables."""
    if category in {TopicCategory.ARMED_CONFLICT, TopicCategory.SECURITY_INCIDENT}:
        return False
    return bool(EVERGREEN_GUIDANCE_PATTERNS.search(text))


def is_historical_commemoration(category: TopicCategory | None, text: str) -> bool:
    """Identify historical soldier burials, centenary commemorations, and routine ceremonies.

    Preserves modern operational news, active conflicts, diplomacy, and analytical
    commentary, even when historical references (e.g. World War II comparisons)
    are present.
    """
    if category in {TopicCategory.ARMED_CONFLICT, TopicCategory.SECURITY_INCIDENT}:
        return False
    if HISTORICAL_COMMEMORATION_TERMS.search(text):
        return True
    return bool(CEREMONIAL_ACTION_PAT.search(text) and HISTORICAL_BATTLEFIELD_PAT.search(text))


_MONTH_NAMES_PAT = (
    r"(?:january|february|march|april|may|june|july|august|september|october|november|december)"
)
RETROSPECTIVE_REPORT_PATTERNS: re.Pattern[str] = re.compile(
    r"\b(how we protected (?:the uk|space|the realm)|"
    r"(?:monthly|quarterly|annual|biannual)\s+(?:report|review|update|summary|bulletin|overview)|"
    rf"review of {_MONTH_NAMES_PAT}|"
    rf"(?:activities|operations|highlights)\s+in\s+{_MONTH_NAMES_PAT}\s+\d{4}|"
    rf"how we (?:supported|protected|served)\b[^.!?]*\bin {_MONTH_NAMES_PAT}\s+\d{4})\b",
    re.I,
)


def is_retrospective_report(category: TopicCategory | None, text: str) -> bool:
    """Identify retrospective monthly/quarterly summaries, institutional recaps, and reviews."""
    if category in {TopicCategory.ARMED_CONFLICT, TopicCategory.SECURITY_INCIDENT}:
        return False
    return bool(RETROSPECTIVE_REPORT_PATTERNS.search(text))


ROUTINE_INSTITUTIONAL_PATTERNS: re.Pattern[str] = re.compile(
    r"\b(appoints?(?:\s+(?:its\s+)?first|\s+new)?\s+(?:director\s+general|director|head\s+of|chief\s+of\s+staff|secretary)|"
    r"names\s+(?:new\s+)?director\s+general|internal\s+reorganization|bureaucratic\s+appointment)\b",
    re.I,
)


def is_routine_institutional_update(category: TopicCategory | None, text: str) -> bool:
    """Identify routine personnel, bureaucratic, or internal institutional appointments."""
    if category in {TopicCategory.ARMED_CONFLICT, TopicCategory.SECURITY_INCIDENT}:
        return False
    return bool(ROUTINE_INSTITUTIONAL_PATTERNS.search(text))


def compute_event_priority_rank(r: SummarizationResult | EventCluster) -> float:
    """Compute deterministic numeric priority rank for an event or summary.

    Lower values represent higher editorial priority:
    - 1.0: Active armed conflict
    - 1.5: Consequential diplomacy (peace talks, ceasefires, summits)
    - 2.0: Security incidents
    - 3.0: Terrorism & insurgency
    - 4.0: Military operations
    - 5.0: Cyber & hybrid warfare
    - 6.0: General diplomacy
    - 7.0: Defence policy developments
    - 8.0: Weapons procurement
    - 9.0: Sanctions
    - 10.0: Peripheral / general geopolitical without defence/conflict dimension
    - 10.5: Historical commemorations and ceremonial burials
    - 11.0: Uncertain / unclassified
    """
    event = getattr(r, "event", None) or r
    summary = getattr(r, "summary", None)
    cat = getattr(event, "category", None)
    if isinstance(cat, str):
        try:
            cat = TopicCategory(cat.lower())
        except ValueError:
            try:
                cat = TopicCategory[cat.upper()]
            except KeyError:
                cat = None

    text_parts = [getattr(event, "title", "")]
    if summary is not None:
        if summary.headline:
            text_parts.append(summary.headline)
        if summary.what_happened:
            text_parts.append(summary.what_happened)
    elif event is not None and getattr(event, "articles", None):
        for art in event.articles:
            if art.title:
                text_parts.append(art.title)
            if art.snippet:
                text_parts.append(art.snippet)

    text_to_check = " ".join(text_parts)

    if cat == TopicCategory.DIPLOMACY and CONSEQUENTIAL_DIPLOMACY_TERMS.search(text_to_check):
        return 1.5

    if (
        is_historical_commemoration(cat, text_to_check)
        or is_evergreen_guidance(cat, text_to_check)
        or is_retrospective_report(cat, text_to_check)
    ):
        return 10.5

    if is_routine_institutional_update(cat, text_to_check):
        return 8.5

    if cat is None or cat not in CATEGORY_PRIORITY:
        return 11.0
    return float(CATEGORY_PRIORITY.get(cat, 11.0))


def _central_claims_are_corroborated(summary: EventSummary) -> bool | None:
    """Return claim-level confidence when claim evidence is present."""
    if not summary.claims:
        return None
    central_claims = [claim for claim in summary.claims if claim.central]
    if not central_claims:
        return False
    for claim in central_claims:
        sources = []
        for evidence_ref in claim.evidence_refs:
            if evidence_ref.verification_status != "accepted":
                continue
            try:
                ref_url = normalize_url(evidence_ref.source_url)
            except ValueError:
                continue
            for source in summary.sources:
                try:
                    source_url = normalize_url(source.url)
                except ValueError:
                    continue
                if source_url == ref_url:
                    sources.append(source)
                    break
        if not are_sources_genuinely_independent(sources):
            return False
    return True


def _sort_key(r: SummarizationResult | EventCluster) -> tuple[int, float, int, float, str]:
    """Sort key prioritizing operational breaking news over analytical commentary."""
    event = r if isinstance(r, EventCluster) else getattr(r, "event", None)
    summary = getattr(r, "summary", None) if not isinstance(r, EventCluster) else None

    is_analysis = 1 if getattr(event, "is_analysis", False) else 0
    cat_rank = compute_event_priority_rank(r)
    claim_confidence = _central_claims_are_corroborated(summary) if summary else None
    if claim_confidence and summary and "single-source" in summary.confidence_qualification.lower():
        claim_confidence = False
    is_corroborated = (
        claim_confidence
        if claim_confidence is not None
        else getattr(event, "confidence_label", None) == "corroborated"
    )
    corrob_rank = 0 if is_corroborated else 1
    last_dt = getattr(event, "last_seen_at", None) or getattr(event, "published_at", None)
    ts = -last_dt.timestamp() if last_dt else 0.0
    tie_breaker = getattr(event, "title", "") or (summary.headline if summary else "") or ""
    return (is_analysis, cat_rank, corrob_rank, ts, tie_breaker)


# Official government/ministry source indicators
OFFICIAL_SOURCE_PATTERNS: re.Pattern[str] = re.compile(
    r"\b(ministry of defen[cs]e|defence ministry|defense ministry|"
    r"department of defen[cs]e|defen[cs]e department|pentagon|mod|dod|idf|"
    r"government|white house|kremlin|press release|official announcement|official statement)\b",
    re.I,
)


_LEAKAGE_PATTERN: re.Pattern[str] = re.compile(
    r"\[?\b(?:unspecified(?:\s+in\s+reporting)?|unknown\s+in\s+reporting|"
    r"not\s+specified(?:\s+in\s+reporting|\s+in\s+source)?)\b\]?",
    re.I,
)


def _clean_rendered_text(text: str | None) -> str:
    """Sanitize leaked placeholder tokens, typos, and awkward phrasing from rendered output."""
    if not text:
        return ""
    cleaned = re.sub(
        r"\b(?:in\s+)?(?:the\s+)?(?:[\w-]+\s+)?city\s+of\s+\[?(?:unspecified(?:\s+in\s+reporting)?|unknown)\]?",
        "",
        text,
        flags=re.I,
    )
    cleaned = _LEAKAGE_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"\bintercceptors\b", "interceptors", cleaned, flags=re.I)
    cleaned = re.sub(r"\bintercceptor\b", "interceptor", cleaned, flags=re.I)
    cleaned = re.sub(r"\binteceptors\b", "interceptors", cleaned, flags=re.I)
    cleaned = re.sub(r"\binteceptor\b", "interceptor", cleaned, flags=re.I)
    cleaned = re.sub(
        r"\b(?:forms?|forming)\s+patterns?\s+(?:of\s+voyages\s+)?in\b",
        "Conducts Pattern of Voyages in",
        cleaned,
        flags=re.I,
    )
    cleaned = re.sub(r"\bforms?\s+patterns?\b", "Conducts Voyages", cleaned, flags=re.I)
    cleaned = re.sub(
        r"\{['\"][a-zA-Z_-]+['\"]\s*:.*?\}(?:\s*\(.*?\))?",
        lambda m: normalize_attribution_text(m.group(0)) or "",
        cleaned,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\s+([,.:;])", r"\1", cleaned)
    return cleaned


def escape_html(text: str | None) -> str:
    """Safely escape plain text for Telegram HTML parse mode."""
    if not text:
        return ""
    return html.escape(_clean_rendered_text(str(text)), quote=True)


def partition_candidate_events(
    items: Sequence[SummarizationResult | EventCluster],
    max_events: int | None = None,
    min_relevance_rank: float | None = DEFAULT_MIN_RELEVANCE_RANK,
    include_low_priority: bool = False,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Partition items into (selected, deferred) candidates using editorial priority & diversity.

    Items can be either EventCluster or SummarizationResult instances.
    """
    if not items:
        return ((), ())

    # 1. Minimum relevance filtering: exclude clearly low-relevance items by default
    effective_min_rank = None if include_low_priority else min_relevance_rank
    if effective_min_rank is not None:
        eligible_candidates = [
            r for r in items if compute_event_priority_rank(r) <= effective_min_rank
        ]
    else:
        eligible_candidates = list(items)

    if not eligible_candidates:
        return ((), ())

    # Sort eligible candidates by editorial priority
    ordered_candidates = sorted(eligible_candidates, key=_sort_key)

    effective_cap = (
        max_events if (max_events is not None and max_events > 0) else len(ordered_candidates)
    )

    selected: list[Any] = []
    deferred: list[Any] = []
    category_counts: dict[TopicCategory | None, int] = {}
    source_counts: dict[str, int] = {}
    max_per_sec_category = 2
    max_per_source = 3

    # Pass 1: Select diverse high-priority items
    for r in ordered_candidates:
        if len(selected) >= effective_cap:
            deferred.append(r)
            continue

        event = r if isinstance(r, EventCluster) else getattr(r, "event", None)
        summary = getattr(r, "summary", None) if not isinstance(r, EventCluster) else None
        cat = getattr(event, "category", None)

        text_parts = [getattr(event, "title", "")]
        if summary is not None and summary.headline:
            text_parts.append(summary.headline)
        elif event is not None and getattr(event, "articles", None):
            for art in event.articles:
                if art.title:
                    text_parts.append(art.title)
        text_to_check = " ".join(text_parts)

        is_consequential_dip = (
            cat == TopicCategory.DIPLOMACY
            and CONSEQUENTIAL_DIPLOMACY_TERMS.search(text_to_check) is not None
        )
        is_tier_one = (
            cat
            in {
                TopicCategory.ARMED_CONFLICT,
                TopicCategory.SECURITY_INCIDENT,
                TopicCategory.TERRORISM_INSURGENCY,
                TopicCategory.MILITARY_OPERATIONS,
            }
            or is_consequential_dip
        )

        if summary is not None and summary.sources:
            primary_source = summary.sources[0].source_name
        elif event is not None and getattr(event, "articles", None) and len(event.articles) > 0:
            primary_source = event.articles[0].source_name
        else:
            primary_source = "unknown"

        cat_count = category_counts.get(cat, 0)
        src_count = source_counts.get(primary_source, 0)

        # Tier 1 events are protected and never suppressed by diversity caps
        if is_tier_one:
            selected.append(r)
            category_counts[cat] = cat_count + 1
            source_counts[primary_source] = src_count + 1
        else:
            # Secondary categories & sources capped during pass 1
            if cat_count < max_per_sec_category and src_count < max_per_source:
                selected.append(r)
                category_counts[cat] = cat_count + 1
                source_counts[primary_source] = src_count + 1
            else:
                deferred.append(r)

    # Pass 2: Selective backfill of strong, diverse developments only.
    # Do NOT force-pad up to effective_cap if remaining candidates are redundant,
    # analytical, or low-value secondary policy/procurement items.
    if len(selected) < effective_cap and deferred:
        for r in list(deferred):
            if len(selected) >= effective_cap:
                break
            event = r if isinstance(r, EventCluster) else getattr(r, "event", None)
            if getattr(event, "is_analysis", False):
                continue
            cat = getattr(event, "category", None)
            rank = compute_event_priority_rank(r)
            if not include_low_priority and rank > 8.0:
                continue
            cat_count = category_counts.get(cat, 0)
            if cat in {
                TopicCategory.WEAPONS_PROCUREMENT,
                TopicCategory.DEFENCE_POLICY,
                TopicCategory.SANCTIONS,
            }:
                if cat_count >= max_per_sec_category:
                    continue
            if r not in selected:
                selected.append(r)
                category_counts[cat] = cat_count + 1
                deferred.remove(r)

    return (tuple(sorted(selected, key=_sort_key)), tuple(deferred))


class TelegramDigestFormatter:
    """Formats event summarization results into clean, length-bounded Telegram HTML messages."""

    def __init__(self, safe_limit: int = SAFE_CHUNK_LIMIT) -> None:
        self.safe_limit = safe_limit

    def format_event_block(
        self,
        summary: EventSummary,
        category: TopicCategory,
        confidence_label: ConfidenceLabel,
        rank: int,
        total: int,
        is_analysis: bool = False,
        published_at: datetime | None = None,
    ) -> str:
        """Format a single event summary into a structured HTML block."""
        if confidence_label == "corroborated":
            conf_qual_lower = (summary.confidence_qualification or "").lower()
            if "single-source" in conf_qual_lower or "single-origin" in conf_qual_lower:
                confidence_label = "single_source"
            elif summary.claims:
                claim_independence = []
                for claim in summary.claims:
                    if not claim.central:
                        continue
                    c_sources = []
                    for s in summary.sources or ():
                        refs = [
                            r
                            for r in claim.evidence_refs
                            if normalize_url(r.source_url) == normalize_url(s.url)
                        ]
                        if refs:
                            spans = " ".join(r.evidence_span for r in refs if r.evidence_span)
                            c_sources.append(
                                {
                                    "url": s.url,
                                    "source_name": s.source_name,
                                    "title": s.title,
                                    "snippet": spans,
                                }
                            )
                    claim_independence.append(are_sources_genuinely_independent(c_sources))
                if not claim_independence or not all(claim_independence):
                    confidence_label = "single_source"
            elif not are_sources_genuinely_independent(summary.sources or ()):
                confidence_label = "single_source"
        cat_badge = CATEGORY_LABELS.get(category, "🌐 GEOPOLITICAL")
        has_official = any(
            OFFICIAL_SOURCE_PATTERNS.search(s.source_name)
            or OFFICIAL_SOURCE_PATTERNS.search(s.title)
            for s in (summary.sources or [])
        )
        if confidence_label == "corroborated" and has_official:
            conf_badge = "[OFFICIAL ANNOUNCEMENT]"
        else:
            conf_badge = CONFIDENCE_BADGES.get(confidence_label, "[REPORT]")

        if is_analysis:
            header_badge = f"<b>[#{rank}/{total}] {cat_badge} · [ANALYSIS] · {conf_badge}</b>"
        else:
            header_badge = f"<b>[#{rank}/{total}] {cat_badge} · {conf_badge}</b>"

        cleaned_headline = _clean_rendered_text(summary.headline)
        cleaned_what = _clean_rendered_text(summary.what_happened)

        def render_claim_with_sources(claim: Any) -> str:
            cited: list[str] = []
            cited_urls: set[str] = set()
            for evidence_ref in claim.evidence_refs:
                if evidence_ref.verification_status != "accepted":
                    continue
                try:
                    evidence_url = normalize_url(evidence_ref.source_url)
                except ValueError:
                    continue
                for source in summary.sources:
                    try:
                        source_url = normalize_url(source.url)
                    except ValueError:
                        continue
                    if source_url == evidence_url and source_url not in cited_urls:
                        cited.append(source)
                        cited_urls.add(source_url)
                        break
            if not cited:
                return ""
            raw_text = _clean_rendered_text(claim.text)
            attr = normalize_attribution_text(getattr(claim, "attribution", None))
            attr = (attr or "").strip()
            if attr and attr.lower() not in raw_text.lower():
                if any(
                    verb in raw_text.lower()
                    for verb in (
                        "said",
                        "stated",
                        "reported",
                        "alleged",
                        "claimed",
                        "according to",
                        "assessed",
                        "argued",
                    )
                ):
                    body_text = (
                        raw_text
                        if raw_text.lower().startswith("according to")
                        else f"According to {attr}, {raw_text[0].lower() + raw_text[1:]}"
                    )
                else:
                    body_text = f"According to {attr}, {raw_text[0].lower() + raw_text[1:]}"
            else:
                body_text = raw_text
            body = escape_html(body_text)
            links = ", ".join(
                f'<a href="{escape_html(source.url)}">{escape_html(source.source_name)}</a>'
                for source in cited
            )
            return f"{body} <i>({links})</i>"

        from geopulse.summarization.publication_gate import (
            extract_plain_narrative,
            is_headline_body_duplicate,
            strip_leading_headline_duplicate,
            strip_wire_dateline,
        )

        if summary.claims:
            body_claims = [
                render_claim_with_sources(claim)
                for claim in summary.claims
                if claim.verification_status == "accepted"
                and claim.classification != "source_analysis"
            ]
            rendered_body = "\n".join(text for text in body_claims if text)
            if not rendered_body.strip():
                rendered_body = escape_html(cleaned_what)
        else:
            rendered_body = escape_html(cleaned_what)

        rendered_body = strip_leading_headline_duplicate(rendered_body, cleaned_headline)
        rendered_body = strip_wire_dateline(rendered_body)

        # Check for exact or semantic headline/body duplication
        if is_headline_body_duplicate(rendered_body, cleaned_headline):
            plain_what = extract_plain_narrative(cleaned_what)
            all_links = ", ".join(
                f'<a href="{escape_html(s.url)}">{escape_html(s.source_name)}</a>'
                for s in summary.sources
                if getattr(s, "url", None) and getattr(s, "source_name", None)
            )
            links_suffix = f"<i>({all_links})</i>" if all_links else ""
            if len(plain_what) >= 10 and not is_headline_body_duplicate(
                cleaned_what, cleaned_headline
            ):
                rendered_body = f"{escape_html(cleaned_what)} {links_suffix}".strip()
            elif (
                summary.significance
                and len(extract_plain_narrative(summary.significance)) >= 10
                and not is_headline_body_duplicate(summary.significance, cleaned_headline)
            ):
                sig_html = escape_html(summary.significance.strip())
                rendered_body = f"{sig_html} {links_suffix}".strip()
            else:
                # If no meaningful supporting claim exists, preserve the headline and source link
                # rather than copying the article or repeating the claim.
                rendered_body = links_suffix.strip()

        # Invariant: Never strip the only readable factual narrative from an event
        if len(extract_plain_narrative(rendered_body)) < 10:
            if not is_headline_body_duplicate(cleaned_what, cleaned_headline):
                if len(extract_plain_narrative(cleaned_what)) >= 10:
                    rendered_body = escape_html(cleaned_what)
                elif summary.claims:
                    accepted_factual = [
                        c
                        for c in summary.claims
                        if c.verification_status == "accepted"
                        and c.classification != "source_analysis"
                    ]
                    if accepted_factual:
                        rendered_body = render_claim_with_sources(accepted_factual[0])

        # Invariant: Never divide a sentence across headline and body
        plain_first = extract_plain_narrative(rendered_body)
        if plain_first and plain_first[0].islower():
            if len(extract_plain_narrative(cleaned_what)) >= 10:
                rendered_body = escape_html(cleaned_what)

        lines: list[str] = [
            header_badge,
            f"<b>{escape_html(cleaned_headline)}</b>\n",
        ]
        if rendered_body:
            lines.append(rendered_body)

        # Key details (omit absent, unspecified, or placeholder fields)
        kd = summary.key_details
        kd_bullets: list[str] = []
        if kd.actors:
            clean_actors = [
                escape_html(a)
                for a in kd.actors
                if a
                and not _LEAKAGE_PATTERN.search(a)
                and a.lower() not in {"unspecified", "unknown", "none"}
            ]
            clean_actors = deduplicate_actor_acronyms(clean_actors)
            if clean_actors:
                kd_bullets.append(f"• <b>Actors:</b> {', '.join(clean_actors)}")

        if (
            kd.location
            and not _LEAKAGE_PATTERN.search(kd.location)
            and kd.location.lower()
            not in {
                "unspecified",
                "unspecified in reporting",
                "unknown",
                "none",
                "null",
                "reported conflict zone",
            }
        ):
            kd_bullets.append(f"• <b>Location:</b> {escape_html(kd.location)}")

        # Timing and publication dates
        if is_analysis:
            if published_at is not None:
                kd_bullets.append(f"• <b>Published:</b> {published_at.strftime('%B %d, %Y')}")
            else:
                kd_bullets.append("• <b>Published:</b> Date not specified in source")
        else:
            pub_date_str = published_at.strftime("%B %d, %Y") if published_at is not None else None
            effective_timing: str | None = None

            if (
                kd.timing
                and not _LEAKAGE_PATTERN.search(kd.timing)
                and kd.timing.lower()
                not in {
                    "unspecified",
                    "unspecified in reporting",
                    "unknown",
                    "none",
                    "null",
                    "date not specified in source",
                }
            ):
                effective_timing = kd.timing.strip()
                # Distinguish historical reporting vs current events:
                # If article was published days earlier, do not claim event happened "today"
                if published_at is not None:
                    now_utc = datetime.now(published_at.tzinfo or None)
                    if (now_utc - published_at).total_seconds() > 36 * 3600:
                        recent_buzzwords = (
                            "within the last 24 hours",
                            "past 24 hours",
                            "today",
                            "yesterday",
                            "this morning",
                            "overnight",
                        )
                        if any(bw in effective_timing.lower() for bw in recent_buzzwords):
                            effective_timing = pub_date_str

            if effective_timing:
                kd_bullets.append(f"• <b>Timing:</b> {escape_html(effective_timing)}")

            if pub_date_str is not None:
                if not effective_timing or effective_timing.strip() != pub_date_str:
                    kd_bullets.append(f"• <b>Published:</b> {pub_date_str}")

        if (
            kd.action
            and not _LEAKAGE_PATTERN.search(kd.action)
            and kd.action.lower()
            not in {"unspecified", "unspecified in reporting", "unknown", "none", "null"}
        ):
            kd_bullets.append(f"• <b>Action:</b> {escape_html(kd.action)}")

        if kd_bullets:
            lines.append("\n<b>Key Details:</b>\n" + "\n".join(kd_bullets))

        # Strategic significance (omit if null or unspecified)
        if summary.significance:
            sig_clean = summary.significance.strip()
            if sig_clean.lower() not in {
                "null",
                "none",
                "n/a",
                "nil",
                "unspecified",
                "unspecified in reporting",
            }:
                analysis_claims = [
                    render_claim_with_sources(claim)
                    for claim in summary.claims
                    if claim.verification_status == "accepted"
                    and claim.classification == "source_analysis"
                ]
                cited_analysis = "\n".join(text for text in analysis_claims if text)
                significance_html = cited_analysis if cited_analysis else escape_html(sig_clean)
                lines.append(f"\n<b>Significance:</b> {significance_html}")

        # Confidence and uncertainty assessment (enforcing badge and assessment consistency)
        assessment_text = ""
        if summary.confidence_qualification:
            assessment_text = summary.confidence_qualification.strip()

        # Expand raw machine slugs, abbreviations, or short tags into human-readable text
        raw_slugs = {
            "single_source": "Single-source reporting; unverified independently.",
            "single_source_report": "Single-source reporting; unverified independently.",
            "corroborated": (
                "Reported across multiple outlets based on official announcement; "
                "military details not independently verified."
                if has_official
                else (
                    "Independent reporting from distinct sources; "
                    "operational details not independently verified."
                )
            ),
            "multi_source_corroborated": (
                "Reported across multiple outlets based on official announcement; "
                "military details not independently verified."
                if has_official
                else (
                    "Independent reporting from distinct sources; "
                    "operational details not independently verified."
                )
            ),
            "disputed": (
                "Contested claims; opposing parties provide conflicting accounts or denials."
            ),
            "disputed_report": (
                "Contested claims; opposing parties provide conflicting accounts or denials."
            ),
            "uncertain": "Preliminary reporting; subject to ongoing verification.",
            "unconfirmed": "Preliminary reporting; subject to ongoing verification.",
        }
        slug_norm = assessment_text.lower().replace("-", "_").strip()
        if slug_norm in raw_slugs:
            assessment_text = raw_slugs[slug_norm]

        # Enforce consistency between badge and assessment text
        if confidence_label == "single_source":
            if (
                not assessment_text
                or len(assessment_text) < 15
                or "corroborat" in assessment_text.lower()
                or "disput" in assessment_text.lower()
            ):
                assessment_text = "Single-source reporting; unverified independently."
        elif confidence_label == "disputed":
            if (
                not assessment_text
                or len(assessment_text) < 15
                or (
                    "single-source" in assessment_text.lower()
                    and "disput" not in assessment_text.lower()
                )
            ):
                assessment_text = (
                    "Contested claims; opposing parties provide conflicting accounts or denials."
                )
        elif confidence_label == "corroborated":
            if has_official:
                assessment_text = (
                    "Reported across multiple outlets based on official announcement; "
                    "military details not independently verified."
                )
            elif assessment_text and (
                "independently verified operational details" in assessment_text.lower()
                or "operational details independently verified" in assessment_text.lower()
            ):
                assessment_text = "Independently verified operational details."
            elif (
                not assessment_text
                or len(assessment_text) < 15
                or "single-source" in assessment_text.lower()
                or "corroborated across multiple" in assessment_text.lower()
            ):
                assessment_text = (
                    "Independent reporting from distinct sources; "
                    "operational details not independently verified."
                )
        elif confidence_label == "uncertain":
            if (
                not assessment_text
                or len(assessment_text) < 15
                or "corroborat" in assessment_text.lower()
            ):
                assessment_text = "Preliminary reporting; subject to ongoing verification."

        if assessment_text:
            lines.append(f"\n<b>Assessment:</b> {escape_html(assessment_text)}")

        # Source attribution with clickable hyperlinks
        if summary.sources:
            lines.append("\n<b>Sources:</b>")
            for src in summary.sources:
                escaped_url = escape_html(src.url)
                escaped_outlet = escape_html(src.source_name)
                escaped_title = escape_html(src.title)
                lines.append(f'• <a href="{escaped_url}">{escaped_outlet}</a>: {escaped_title}')

        return "\n".join(lines)

    def select_digest_events(
        self,
        results: Sequence[SummarizationResult],
        max_events: int | None = None,
        min_relevance_rank: float | None = DEFAULT_MIN_RELEVANCE_RANK,
        include_low_priority: bool = False,
    ) -> tuple[SummarizationResult, ...]:
        """Select, filter, and prioritize events for the briefing digest.

        Filters low-relevance events (historical commemorations, routine ceremonies,
        and peripheral items) unless include_low_priority is True.
        Applies source and category diversity constraints up to max_events without
        padding the digest with low-value items.
        """
        from geopulse.summarization.publication_gate import validate_publication_gate

        passed_results: list[SummarizationResult] = []
        for r in results:
            if (
                r.status == "success"
                and r.summary is not None
                and getattr(r.summary, "model_used", "") != "omitted"
            ):
                event = getattr(r, "event", None)
                if event is not None:
                    gate_res = validate_publication_gate(r.summary, event)
                    if not gate_res.passed:
                        continue
                    if gate_res.repaired_summary:
                        r = r.model_copy(update={"summary": gate_res.repaired_summary})
                elif (
                    r.summary.what_happened.strip().lower() == r.summary.headline.strip().lower()
                    and not getattr(r.summary, "claims", None)
                ):
                    continue
                # Reject items that contain no substantive readable narrative text
                from geopulse.summarization.publication_gate import extract_plain_narrative

                if len(extract_plain_narrative(r.summary.what_happened)) < 10:
                    continue
                passed_results.append(r)

        if not passed_results:
            return ()
        selected, _ = partition_candidate_events(
            items=passed_results,
            max_events=max_events,
            min_relevance_rank=min_relevance_rank,
            include_low_priority=include_low_priority,
        )
        return selected

    def format_digest(
        self,
        results: Sequence[SummarizationResult],
        briefing_date: date | None = None,
        edition: str | None = None,
        max_events: int | None = None,
        min_relevance_rank: float | None = DEFAULT_MIN_RELEVANCE_RANK,
        include_low_priority: bool = False,
    ) -> tuple[str, ...]:
        """Format multiple summarization results into one or more Telegram message chunks.

        Splits across message boundaries cleanly between events, ensuring no message
        exceeds safe character limits and no HTML tags are broken.
        """
        successful = [
            r
            for r in results
            if r.status == "success"
            and r.summary is not None
            and getattr(r.summary, "model_used", "") != "omitted"
            and not (
                r.summary.what_happened.strip().lower() == r.summary.headline.strip().lower()
                and not getattr(r.summary, "claims", None)
            )
        ]
        ordered_successful = self.select_digest_events(
            results=successful,
            max_events=max_events,
            min_relevance_rank=min_relevance_rank,
            include_low_priority=include_low_priority,
        )

        candidate_count = sum(
            1
            for r in results
            if (
                (r.status == "success" and r.summary is not None)
                or (r.summary is not None and getattr(r.summary, "model_used", "") == "omitted")
            )
            and r.status != "quarantined"
        )
        omitted_note = ""
        omitted_count = (
            max(0, candidate_count - len(ordered_successful)) if len(ordered_successful) > 0 else 0
        )
        if omitted_count > 0 and len(ordered_successful) > 0:
            if max_events is not None and max_events > 0:
                if len(ordered_successful) >= max_events:
                    omitted_note = (
                        f"\n\n<i>[Editorial Note: Briefing capped at top {max_events} events. "
                        f"{omitted_count} lower-priority event(s) omitted to maintain "
                        "digest length.]</i>"
                    )
                else:
                    omitted_note = (
                        f"\n\n<i>[Editorial Note: {len(ordered_successful)} event(s) briefed "
                        f"(configured limit: {max_events}). {omitted_count} lower-priority "
                        "event(s) omitted to maintain digest length.]</i>"
                    )
            else:
                omitted_note = (
                    f"\n\n<i>[Editorial Note: {omitted_count} lower-priority event(s) "
                    "omitted to maintain digest length.]</i>"
                )

        target_date = briefing_date or date.today()
        date_str = target_date.strftime("%B %d, %Y")
        total_events = len(ordered_successful)

        clean_edition = (edition or "").strip().lower()
        has_quarantined_or_failed = any(
            getattr(r, "status", "") in {"quarantined", "failed", "insufficient_evidence"}
            for r in results
        )
        if clean_edition == "morning":
            badge = "🌅 GEOPULSE MORNING BRIEFING"
            single_header = (
                f"<b>{badge}</b>\n<i>{date_str} • {total_events} event(s) briefed</i>\n\n"
            )
            part_prefix = f"<b>{badge} (Part "
            if has_quarantined_or_failed:
                empty_msg = (
                    f"<b>{badge}</b> — {date_str}\n\n"
                    "<i>No significant new military or geopolitical developments confirmed "
                    "meeting briefing thresholds in this edition. "
                    "Developing items remain under active monitoring "
                    "and source verification.</i>\n\n"
                    "<i>Geopulse Intelligence • Source-grounded briefing</i>"
                )
            else:
                empty_msg = (
                    f"<b>{badge}</b> — {date_str}\n\n"
                    "<i>No significant new military or geopolitical developments identified "
                    "since the evening briefing. Monitoring continues across all configured "
                    "defence feeds.</i>\n\n"
                    "<i>Geopulse Intelligence • Source-grounded briefing</i>"
                )
        elif clean_edition == "evening":
            badge = "🌆 GEOPULSE EVENING BRIEFING"
            single_header = (
                f"<b>{badge}</b>\n<i>{date_str} • {total_events} event(s) briefed</i>\n\n"
            )
            part_prefix = f"<b>{badge} (Part "
            if has_quarantined_or_failed:
                empty_msg = (
                    f"<b>{badge}</b> — {date_str}\n\n"
                    "<i>No significant new military or geopolitical developments confirmed "
                    "meeting briefing thresholds in this edition. "
                    "Developing items remain under active monitoring "
                    "and source verification.</i>\n\n"
                    "<i>Geopulse Intelligence • Source-grounded briefing</i>"
                )
            else:
                empty_msg = (
                    f"<b>{badge}</b> — {date_str}\n\n"
                    "<i>No significant new military or geopolitical developments identified "
                    "since the morning briefing. Monitoring continues across all configured "
                    "defence feeds.</i>\n\n"
                    "<i>Geopulse Intelligence • Source-grounded briefing</i>"
                )
        else:
            single_header = (
                f"<b>🌍 GEOPULSE DAILY INTELLIGENCE BRIEFING</b>\n"
                f"<i>{date_str} • {total_events} event(s) briefed</i>\n\n"
            )
            part_prefix = "<b>🌍 GEOPULSE BRIEFING (Part "
            if has_quarantined_or_failed:
                empty_msg = (
                    f"<b>🌍 GEOPULSE DAILY BRIEFING</b> — {date_str}\n\n"
                    "<i>No corroborated or verified geopolitical events met briefing "
                    "thresholds today. Developing items remain under active monitoring "
                    "and source verification.</i>\n\n"
                    "<i>Geopulse Intelligence • Source-grounded briefing</i>"
                )
            else:
                empty_msg = (
                    f"<b>🌍 GEOPULSE DAILY BRIEFING</b> — {date_str}\n\n"
                    "<i>No corroborated or verified geopolitical events met briefing "
                    "thresholds today. "
                    "Monitoring continues across all configured defence feeds.</i>\n\n"
                    "<i>Geopulse Intelligence • Source-grounded briefing</i>"
                )

        if not ordered_successful:
            return (empty_msg,)

        event_blocks: list[str] = []
        for idx, res in enumerate(ordered_successful, start=1):
            assert res.summary is not None
            evt = res.event
            block = self.format_event_block(
                summary=res.summary,
                category=evt.category,
                confidence_label=evt.confidence_label,
                rank=idx,
                total=total_events,
                is_analysis=getattr(evt, "is_analysis", False),
                published_at=getattr(evt, "published_at", None),
            )
            event_blocks.append(block)

        # Build chunks respecting self.safe_limit
        return self._pack_chunks(
            event_blocks=event_blocks,
            date_str=date_str,
            single_header=single_header,
            part_prefix=part_prefix,
            omitted_note=omitted_note,
        )

    def _pack_chunks(
        self,
        event_blocks: list[str],
        date_str: str,
        single_header: str,
        part_prefix: str,
        omitted_note: str = "",
    ) -> tuple[str, ...]:
        """Pack event blocks into message chunks bounded by character limits."""
        footer = f"\n\n<i>Geopulse Intelligence • Source-grounded briefing</i>{omitted_note}"
        full_single = single_header + "\n\n".join(event_blocks) + footer

        if len(full_single) <= self.safe_limit:
            return (full_single,)

        # Multi-part packing: assign event blocks into chunk groups
        chunks_events: list[list[str]] = []
        current_group: list[str] = []

        for block in event_blocks:
            current_group.append(block)
            # Estimate chunk size with conservative header/footer overhead
            estimated_size = (
                len(f"{part_prefix}99/99) — {date_str}</b>\n\n")
                + sum(len(b) + 2 for b in current_group)
                + len(footer)
            )
            if estimated_size > self.safe_limit and len(current_group) > 1:
                # Pop the block that caused overflow and start next group
                current_group.pop()
                chunks_events.append(list(current_group))
                current_group = [block]

        if current_group:
            chunks_events.append(current_group)

        total_parts = len(chunks_events)
        result_chunks: list[str] = []

        for part_idx, group in enumerate(chunks_events, start=1):
            if total_parts > 1:
                header = f"{part_prefix}{part_idx}/{total_parts})</b>\n<i>{date_str}</i>\n\n"
            else:
                header = single_header

            is_last = part_idx == total_parts
            body = "\n\n".join(group)
            part_msg = header + body + (footer if is_last else "")
            result_chunks.append(part_msg)

        return tuple(result_chunks)


def format_telegram_digest(
    results: Sequence[SummarizationResult],
    briefing_date: date | None = None,
    safe_limit: int = SAFE_CHUNK_LIMIT,
    edition: str | None = None,
    max_events: int | None = None,
) -> tuple[str, ...]:
    """Convenience helper to format summarization results into Telegram HTML chunks."""
    formatter = TelegramDigestFormatter(safe_limit=safe_limit)
    return formatter.format_digest(
        results=results,
        briefing_date=briefing_date,
        edition=edition,
        max_events=max_events,
    )
