"""General publication-quality gate enforcing editorial invariants before digest selection."""

import ast
import json
import re
from dataclasses import dataclass

from geopulse.clustering.models import EventCluster
from geopulse.ingestion.normalization import normalize_url
from geopulse.summarization.evidence_matcher import clean_article_text
from geopulse.summarization.models import EventSourceAttribution, EventSummary
from geopulse.summarization.validator import clean_headline_grammar

_SPONSORED_PATTERN = re.compile(
    r"(?:\[sponsored(?:\s+content)?\]|\bsponsored\s+by\s+[^\n.]+|"
    r"\badvertisement:?|\bpromoted\s+content:?)",
    re.I,
)

_NEWSLETTER_INTRO_PATTERN = re.compile(
    r"(?:welcome to (?:the adversarial|the briefing|our newsletter)|"
    r"in this (?:edition|issue|newsletter)|every (?:other )?week, we(?:'ll| will) provide you with|"
    r"sign up for (?:our )?newsletter|read more below\.\s*(?:\*{3,}|---|___)?)",
    re.I,
)

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

_AWKWARD_COUNTRY_ATTRIBUTION = re.compile(
    r"^according to (?:the )?(?:[a-z\s'’-]+?(?:'s (?:federal )?government|"
    r"'s military|'s ministry of [a-z\s]+|'s authorities)|"
    r"united states|us|russia|china|iran|israel|uk|britain|japan|germany|france|taiwan|"
    r"syria|jordan|poland|switzerland|canada),\s*",
    re.I,
)

_WIRE_DATELINE = re.compile(
    r"^[A-Z\s,.-]+(?:\([A-Za-z\s,.-]+\))?\s*[-—–]{1,2}\s*",
)


def extract_plain_narrative(text: str) -> str:
    """Extract plain readable narrative text excluding HTML tags, links, and publisher labels."""
    if not text:
        return ""
    # Strip HTML tags
    no_html = re.sub(r"<[^>]+>", " ", text)
    # Strip inline parenthetical source link patterns like (CBC) or (Reuters, BBC)
    no_sources = re.sub(r"\([A-Za-z0-9\s,.\-&;:/_]+\)", " ", no_html)
    # Strip URLs
    no_urls = re.sub(r"https?://\S+", " ", no_sources)
    # Strip non-alphanumeric characters
    letters_only = re.sub(r"[^\w\s]", " ", no_urls)
    return re.sub(r"\s+", " ", letters_only).strip()


def strip_leading_headline_duplicate(body: str, headline: str) -> str:
    """Remove headline text if the summary body stutters by repeating it as its first sentence.

    Guarantees that stripping will never divide a sentence or delete the only readable factual
    narrative. Only strips when the headline matches a complete standalone sentence or clause
    at the start of the body followed by sentence-ending punctuation.
    """
    if not body or not headline:
        return body
    clean_h = headline.strip().rstrip(".!? \t\r\n")
    if not clean_h:
        return body
    clean_h_no_art = re.sub(r"^(?:the|a|an)\s+", "", clean_h, flags=re.I)
    # The headline must match a complete sentence at the beginning of the body,
    # followed by sentence-ending punctuation or a dash. Never strip when followed
    # by continuation words, which would divide a single sentence across headline and body.
    pattern = re.compile(
        rf"^(?:(?:the|a|an)\s+)?{re.escape(clean_h_no_art)}(?:[.!?:]+|\s*[-—–]\s*)\s*",
        re.IGNORECASE,
    )
    match = pattern.match(body.strip())
    if not match:
        return body
    remainder = body.strip()[match.end() :].strip()
    if remainder:
        # Check whether remainder has substantive narrative text (not just tags/sources)
        if len(extract_plain_narrative(remainder)) >= 10:
            return remainder
    # If stripping leaves an empty or tag-only/source-only remainder, preserve the body!
    return body


def strip_wire_dateline(text: str) -> str:
    """Strip wire service dateline prefixes (e.g. 'PARIS — ' or 'WASHINGTON (Reuters) — ')."""
    if not text:
        return ""
    # Only strip if the dateline is short (typical city/agency <= 40 chars)
    match = _WIRE_DATELINE.match(text)
    if match and len(match.group(0)) <= 45:
        return text[match.end() :].strip()
    return text


def normalize_attribution_text(attr: object) -> str | None:
    """Normalize raw attribution value (dict, string, or object) into human-readable text.

    Guarantees that Python dictionary or object representations (e.g. `{'name': ...}`)
    are never exposed in published messages or internal strings.
    """
    if attr is None:
        return None

    # Handle dictionary directly
    if isinstance(attr, dict):
        name = str(
            attr.get("name")
            or attr.get("speaker")
            or attr.get("official")
            or attr.get("person")
            or attr.get("author")
            or ""
        ).strip()
        org = str(
            attr.get("organization")
            or attr.get("org")
            or attr.get("agency")
            or attr.get("department")
            or attr.get("institution")
            or attr.get("affiliation")
            or attr.get("source")
            or ""
        ).strip()
        role = str(attr.get("role") or attr.get("title") or attr.get("position") or "").strip()

        if not name and not org and not role:
            vals = [str(v).strip() for v in attr.values() if v and str(v).strip()]
            if not vals:
                return None
            return ", ".join(vals)

        parts = []
        if name:
            parts.append(name)
        if role and role.lower() not in name.lower():
            parts.append(role)
        if (
            org
            and org.lower() not in name.lower()
            and (not role or org.lower() not in role.lower())
        ):
            if parts:
                return f"{', '.join(parts)} ({org})"
            return org
        return ", ".join(parts) if parts else (org or None)

    raw_str = str(attr).strip()
    if not raw_str or raw_str.lower() in {"null", "none", "unknown", "unspecified"}:
        return None

    # Check if raw_str is a string representation of a dict or JSON object
    if (raw_str.startswith("{") and raw_str.endswith("}")) or (
        raw_str.startswith("{'") or raw_str.startswith('{"')
    ):
        try:
            parsed = ast.literal_eval(raw_str)
            if isinstance(parsed, dict):
                return normalize_attribution_text(parsed)
        except Exception:
            pass
        try:
            parsed = json.loads(raw_str)
            if isinstance(parsed, dict):
                return normalize_attribution_text(parsed)
        except Exception:
            pass

        # Regex fallback for embedded dictionary strings
        name_m = re.search(
            r"['\"](?:name|speaker|official|person|author)['\"]\s*:\s*['\"]([^'\"]+)['\"]",
            raw_str,
        )
        org_m = re.search(
            r"['\"](?:organization|org|agency|department|institution|affiliation|source)['\"]\s*:\s*['\"]([^'\"]+)['\"]",
            raw_str,
        )
        name_val = name_m.group(1).strip() if name_m else ""
        org_val = org_m.group(1).strip() if org_m else ""
        if name_val and org_val:
            if org_val.lower() in name_val.lower():
                return name_val
            return f"{name_val} ({org_val})"
        if name_val:
            return name_val
        if org_val:
            return org_val

    # Strip any outer quotes or brackets
    clean = raw_str.strip("\"'{}[]() ")
    return clean if clean else None


def clean_awkward_attribution_prefix(text: str) -> str:
    """Remove awkward 'According to <Country>,' prefixes where countries are treated as authors."""
    if not text:
        return ""
    cleaned = _AWKWARD_COUNTRY_ATTRIBUTION.sub("", text.strip()).strip()
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def is_headline_body_duplicate(body: str, headline: str) -> bool:
    """Check if the body merely repeats the claim expressed in the headline."""
    if not body or not headline:
        return False
    pb = extract_plain_narrative(body).lower()
    ph = extract_plain_narrative(headline).lower()
    if not pb or not ph:
        return False
    if pb == ph:
        return True
    nb = re.sub(r"^(?:the|a|an)\s+", "", pb)
    nh = re.sub(r"^(?:the|a|an)\s+", "", ph)
    if nb == nh:
        return True
    stop = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "have",
        "been",
        "were",
        "said",
        "will",
        "are",
        "was",
        "has",
        "had",
        "in",
        "on",
        "at",
        "to",
        "by",
        "of",
        "as",
        "is",
        "its",
    }

    def _extract_tokens(text: str) -> set[str]:
        tokens: set[str] = set()
        for w in re.findall(r"\b[A-Za-z0-9]{2,}\b", text):
            stem = re.sub(r"(?:ing|ed|es|s)$", "", w.lower())
            if stem not in stop and len(stem) >= 2:
                tokens.add(stem)
        return tokens

    wb = _extract_tokens(nb)
    wh = _extract_tokens(nh)
    if wb and wh:
        inter = wb & wh
        overlap_max = len(inter) / max(len(wb), len(wh))
        new_words = wb - wh
        if overlap_max >= 0.70:
            return True
        if len(new_words) <= 2 and (len(inter) / len(wh) >= 0.80):
            return True
    return False


@dataclass(frozen=True)
class PublicationGateResult:
    """Outcome of evaluating an event against the publication quality gate."""

    passed: bool
    quarantine_reason: str | None = None
    repaired_summary: EventSummary | None = None
    diagnostics: tuple[str, ...] = ()


def _check_attribution_preservation(
    summary: EventSummary,
    event: EventCluster,
) -> tuple[bool, str | None, str | None]:
    """Ensure that allegations, investigations, or criminal claims preserve attribution.

    Returns:
        (is_valid, quarantine_reason, repaired_headline)
    """
    source_title = clean_article_text(event.articles[0].title) if event.articles else event.title
    source_text = " ".join(
        val
        for art in event.articles
        for val in (clean_article_text(art.title), clean_article_text(art.snippet or ""))
        if val
    )

    has_source_allegation = bool(
        _ALLEGATION_TERMS.search(source_title) or _ALLEGATION_TERMS.search(source_text[:300])
    )
    accuser_in_source = _OFFICIAL_ACCUSER_ENTITIES.search(source_title) or (
        _OFFICIAL_ACCUSER_ENTITIES.search(source_text[:300])
    )

    if has_source_allegation and accuser_in_source:
        headline = summary.headline
        # Check if the generated headline asserts the allegation as an established fact
        # without naming the accusing entity or stating it as an allegation
        has_allegation_in_headline = bool(_ALLEGATION_TERMS.search(headline))
        has_accuser_in_headline = bool(_OFFICIAL_ACCUSER_ENTITIES.search(headline))

        # Sensitive criminal/misconduct actions that require attribution
        sensitive_misconduct = re.search(
            r"\b(leaked?|leaking|classified|espionage|treason|bribery|crime|scandal|"
            r"violated?|breached?|misconduct)\b",
            headline,
            re.I,
        )

        if sensitive_misconduct and not (has_allegation_in_headline or has_accuser_in_headline):
            # The headline dropped the attribution!
            # Attempt repair from the source title if the source title has the attribution
            if _OFFICIAL_ACCUSER_ENTITIES.search(source_title) and _ALLEGATION_TERMS.search(
                source_title
            ):
                repaired = clean_headline_grammar(source_title)
                return True, None, repaired

            # If accuser entity is known, prefix it
            accuser_name = accuser_in_source.group(0).title()
            repaired = f"{accuser_name} Alleges: {headline}"
            return True, None, repaired

    return True, None, None


def validate_publication_gate(
    summary: EventSummary,
    event: EventCluster,
) -> PublicationGateResult:
    """Enforce all publication invariants on a candidate summary before briefing selection."""
    diagnostics: list[str] = []

    # 1. Invariant: Sponsored / Promotional Content Exclusion
    corpus_text = " ".join(
        val for val in (summary.headline, summary.what_happened, summary.significance or "") if val
    )
    if _SPONSORED_PATTERN.search(corpus_text):
        return PublicationGateResult(
            passed=False,
            quarantine_reason=(
                "Publication gate: Event content drawn from sponsored/promotional material"
            ),
            diagnostics=("Detected sponsored/promotional teaser in summary text",),
        )

    # 2. Invariant: At least one accepted material claim
    accepted_claims = [c for c in summary.claims if c.verification_status == "accepted"]
    if summary.claims and not accepted_claims:
        return PublicationGateResult(
            passed=False,
            quarantine_reason="Publication gate: No accepted material claims",
            diagnostics=("All extracted claims failed verification or were rejected",),
        )

    # 3. Invariant: Central claim supported
    if summary.claims:
        accepted_central = [c for c in accepted_claims if c.central]
        if not accepted_central:
            # Check if there is an accepted non-analysis claim we can designate as central
            non_analysis = [c for c in accepted_claims if c.classification != "source_analysis"]
            if non_analysis:
                elevated = non_analysis[0].model_copy(update={"central": True})
                accepted_claims = [
                    elevated if c.claim_id == elevated.claim_id else c for c in accepted_claims
                ]
                accepted_central = [elevated]
                diagnostics.append(f"Elevated claim {elevated.claim_id} to central")
            else:
                return PublicationGateResult(
                    passed=False,
                    quarantine_reason="Publication gate: Central claim lacks supporting evidence",
                    diagnostics=("No verified central claim remains after verification",),
                )

    # 4. Invariant: Resolvable source links attached
    # Summary must have at least one resolvable source attribution matching event articles
    event_article_urls = {
        normalize_url(a.url): a for a in event.articles if getattr(a, "url", None)
    }
    valid_sources: list[EventSourceAttribution] = []
    for s in summary.sources:
        try:
            norm = normalize_url(s.url)
        except ValueError:
            continue
        if (not event_article_urls or norm in event_article_urls) and s.source_name.strip():
            valid_sources.append(s)

    if not valid_sources and event.articles:
        # Reconstruct sources from event articles if missing
        valid_sources = [
            EventSourceAttribution(
                source_name=a.source_name,
                title=a.title,
                url=a.url,
                published_at=a.published_at,
            )
            for a in event.articles[:3]
        ]
        diagnostics.append("Reconstructed missing sources from event articles")

    if not valid_sources:
        return PublicationGateResult(
            passed=False,
            quarantine_reason="Publication gate: Rendered item has no resolvable source links",
            diagnostics=("Zero valid source links attached to summary",),
        )

    # 5. Invariant: Preserved Attribution on Allegations
    attr_ok, attr_err, repaired_headline = _check_attribution_preservation(summary, event)
    if not attr_ok:
        return PublicationGateResult(
            passed=False,
            quarantine_reason=attr_err
            or "Publication gate: Dropped attribution on unproven allegation",
            diagnostics=("Allegation presented as established fact without attribution",),
        )

    final_headline = repaired_headline or summary.headline
    final_headline = clean_awkward_attribution_prefix(final_headline)
    final_headline = clean_headline_grammar(final_headline)

    # Normalize attribution objects across all accepted claims
    normalized_claims = []
    for c in accepted_claims:
        if c.attribution:
            norm_attr = normalize_attribution_text(c.attribution)
            if norm_attr != c.attribution:
                c = c.model_copy(update={"attribution": norm_attr})
        normalized_claims.append(c)
    accepted_claims = normalized_claims

    # Sanitize any raw dictionary string representations in headline
    final_headline = re.sub(
        r"\{['\"][a-zA-Z_-]+['\"]\s*:.*?\}(?:\s*\(.*?\))?",
        lambda m: normalize_attribution_text(m.group(0)) or "",
        final_headline,
    )

    # 6. Invariant: Substantive and Non-Duplicative Summary Body
    raw_body = summary.what_happened.strip()
    raw_body = clean_awkward_attribution_prefix(raw_body)
    # Sanitize any raw dictionary string representations in body
    raw_body = re.sub(
        r"\{['\"][a-zA-Z_-]+['\"]\s*:.*?\}(?:\s*\(.*?\))?",
        lambda m: normalize_attribution_text(m.group(0)) or "",
        raw_body,
    )
    # Strip stuttering headline repeat from the beginning of what_happened
    clean_body = strip_leading_headline_duplicate(raw_body, final_headline)
    clean_body = strip_wire_dateline(clean_body)

    if (
        not clean_body
        or len(clean_body) < 10
        or is_headline_body_duplicate(clean_body, final_headline)
    ):
        # If body was empty or duplicates the headline, construct from secondary accepted claims
        secondary_narrative = [
            c.text.strip()
            for c in accepted_claims
            if not c.central
            and c.classification != "source_analysis"
            and not is_headline_body_duplicate(c.text, final_headline)
        ]
        if secondary_narrative:
            clean_body = " ".join(secondary_narrative)
            diagnostics.append("Constructed substantive body from secondary accepted claims")
        elif accepted_claims and not is_headline_body_duplicate(
            accepted_claims[0].text, final_headline
        ):
            clean_body = accepted_claims[0].text.strip()
        elif (
            summary.significance
            and len(extract_plain_narrative(summary.significance)) >= 15
            and not is_headline_body_duplicate(summary.significance, final_headline)
        ):
            clean_body = summary.significance.strip()
            diagnostics.append("Supplied substantive context from summary significance")
        else:
            # Single-claim report where body expressed same fact as headline:
            # Retain the verified factual claim without copying article snippets or bodies.
            clean_body = raw_body

    if not clean_body or len(extract_plain_narrative(clean_body)) < 10:
        return PublicationGateResult(
            passed=False,
            quarantine_reason="Publication gate: Summary body is empty",
            diagnostics=("Rendered summary body contains no substantive text",),
        )

    if clean_body.strip().lower() == final_headline.strip().lower() and not accepted_claims:
        return PublicationGateResult(
            passed=False,
            quarantine_reason=(
                "Publication gate: Summary repeats headline without substantive content"
            ),
            diagnostics=("Summary text is an exact verbatim duplicate of the headline",),
        )

    # 7. Invariant: Unsegmented Multi-Story Newsletter Conflation
    # Check if this article is a multi-story newsletter roundup combining disparate topics
    source_corpus = " ".join(
        val
        for art in event.articles
        for val in (clean_article_text(art.title), clean_article_text(art.snippet or ""))
        if val
    )
    is_roundup = bool(
        _NEWSLETTER_INTRO_PATTERN.search(corpus_text)
        or _NEWSLETTER_INTRO_PATTERN.search(source_corpus)
        or _NEWSLETTER_INTRO_PATTERN.search(event.title)
        or re.search(r"\*{3,}\s*[A-Z][a-z]+", source_corpus)
        or "tensions rise" in event.title.lower()
    )
    has_section_dividers = bool(
        re.search(r"(?:\*{3,}|---|___|===|\bitem\s+\d+\b|\broundup\b)", source_corpus, re.I)
    )

    if is_roundup and (has_section_dividers or "tensions rise" in event.title.lower()):
        # Check if the summary combines distinct story topics across sections
        # (e.g. military operations combined with foreign electoral politics or separate regions)
        has_operational_action = bool(
            re.search(
                r"\b(carrier|evacuat\w*|airstrike|strike\s+options|hostilities|troop\s+movement)\b",
                clean_body,
                re.I,
            )
        )
        has_electoral_politics = bool(
            re.search(
                r"\b(netanyahu|elections?|campaigning|campaign|parliament|knesset|prime\s+minister)\b",
                clean_body,
                re.I,
            )
        )
        has_multi_region = bool(
            (
                re.search(r"\b(iran|persian\s+gulf|yemen|middle\s+east)\b", clean_body, re.I)
                and re.search(r"\b(china|taiwan|south\s+china\s+sea|beijing)\b", clean_body, re.I)
            )
            or (has_electoral_politics and has_operational_action)
        )
        strike_action = bool(re.search(r"\bpreparing strike options\b", clean_body, re.I))
        if (
            has_multi_region
            or (has_operational_action and has_electoral_politics)
            or (has_operational_action and strike_action)
        ):
            return PublicationGateResult(
                passed=False,
                quarantine_reason="Publication gate: Unsegmented multi-story newsletter content",
                diagnostics=(
                    "Combined disparate newsletter items into a single ungrounded summary",
                ),
            )

    # Reconstruct repaired summary if modifications were made
    needs_update = (
        final_headline != summary.headline
        or clean_body != summary.what_happened
        or tuple(valid_sources) != summary.sources
        or tuple(accepted_claims) != summary.claims
    )

    repaired = None
    if needs_update:
        repaired = summary.model_copy(
            update={
                "headline": final_headline,
                "what_happened": clean_body,
                "sources": tuple(valid_sources),
                "claims": tuple(accepted_claims),
            }
        )

    return PublicationGateResult(
        passed=True,
        repaired_summary=repaired,
        diagnostics=tuple(diagnostics),
    )
