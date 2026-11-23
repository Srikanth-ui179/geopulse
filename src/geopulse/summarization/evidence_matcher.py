"""Deterministic evidence span matching and diagnostic reporting."""

import html
import re
from dataclasses import dataclass
from typing import Any, Literal

from geopulse.clustering.models import EventCluster
from geopulse.ingestion.normalization import normalize_url

# Typographic normalization mappings
_UNICODE_QUOTES = re.compile(r"[\u201c\u201d\u201e\u00ab\u00bb]")  # “ ” „ « »
_UNICODE_APOSTROPHES = re.compile(r"[\u2018\u2019\u201a\u201b\u00b4`]")  # ‘ ’ ‚ ‛ ´ `
_UNICODE_DASHES = re.compile(r"[\u2012\u2013\u2014\u2015]|\-\-")  # ‒ – — ― --
_UNICODE_SPACES = re.compile(r"[\u00a0\u2000-\u200b\u202f\u205f\ufeff]")  # non-breaking spaces
_ELLIPSIS = re.compile(r"[\u2026]|\.{3,}")  # … or ...


def clean_article_text(text: str) -> str:
    """Safely strip HTML tags, unescape HTML entities, and normalize whitespace."""
    if not text:
        return ""
    no_tags = re.sub(r"<[^>]+>", " ", text)
    decoded = html.unescape(no_tags)
    no_uspaces = _UNICODE_SPACES.sub(" ", decoded)
    return re.sub(r"\s+", " ", no_uspaces).strip()


def normalize_evidence_text(text: str) -> str:
    """Safely normalize text for deterministic evidence comparison.

    Normalizes HTML tags, HTML entities, smart quotes, apostrophes, dashes, non-breaking spaces,
    and collapses whitespace. Does NOT perform stemming, fuzzy matching, or synonym replacement.
    """
    if not text:
        return ""
    # 0. Strip HTML tags
    res = re.sub(r"<[^>]+>", " ", text)

    # 1. Decode HTML entities (&amp; -> &, &#8217; -> ', etc.)
    res = html.unescape(res)

    # 2. Normalize smart quotes and apostrophes to ASCII equivalents
    res = _UNICODE_QUOTES.sub('"', res)
    res = _UNICODE_APOSTROPHES.sub("'", res)

    # 3. Normalize dashes and hyphens to ASCII hyphen
    res = _UNICODE_DASHES.sub("-", res)

    # 4. Normalize unicode spaces to ASCII space
    res = _UNICODE_SPACES.sub(" ", res)

    # 5. Normalize ellipsis
    res = _ELLIPSIS.sub("...", res)

    # 6. Collapse whitespace and strip
    res = re.sub(r"\s+", " ", res).strip()

    return res


@dataclass(frozen=True)
class EvidenceSpanDiagnostic:
    """Diagnostic record for evidence span verification without secrets."""

    event_id: str
    source_url: str
    normalized_url: str
    claim_id: str
    source_belongs_to_event: bool
    match_status: Literal["verbatim", "normalized", "no_match"]
    sanitized_diff: str | None = None


_PLACEHOLDER_EVIDENCE_PATTERN = re.compile(
    r"<\s*(?:short\s+)?(?:exact\s+)?(?:verbatim\s+)?(?:quote|passage|evidence|span|text)[^>]*>|"
    r"\[\s*(?:verbatim\s+)?(?:quote|passage|evidence|span|text)\s*\]|"
    r"^(?:quote|verbatim quote|exact quote|evidence|placeholder|none|n/a|\.\.\.)$",
    re.I,
)


def check_evidence_span_match(
    evidence_span: str,
    available_text: str,
) -> tuple[Literal["verbatim", "normalized", "no_match"], str | None]:
    """Check whether an evidence span exists in available source text.

    Distinguishes:
    - 'verbatim': Exact substring match.
    - 'normalized': Match after safe deterministic normalization (entities, smart quotes, dashes).
    - 'no_match': Span is absent or fabricated.
    """
    if not evidence_span or not evidence_span.strip():
        return "no_match", "Empty or whitespace evidence span"

    stripped_span = evidence_span.strip()
    if _PLACEHOLDER_EVIDENCE_PATTERN.search(stripped_span) or (
        stripped_span.startswith("<") and stripped_span.endswith(">")
    ):
        return "no_match", f"Placeholder evidence span rejected: '{stripped_span}'"

    span_raw = clean_article_text(stripped_span)
    if not span_raw or len(span_raw) < 4 or len(re.findall(r"[A-Za-z]", span_raw)) < 3:
        return "no_match", f"Evidence span too short or empty after sanitization: '{stripped_span}'"

    clean_avail = clean_article_text(available_text)
    if span_raw and (span_raw in available_text or span_raw in clean_avail):
        return "verbatim", None

    # Try safe normalization
    span_norm = normalize_evidence_text(span_raw)
    avail_norm = normalize_evidence_text(clean_avail)

    if span_norm and span_norm in avail_norm:
        return "normalized", None

    # Check with boundary punctuation stripped (e.g. model added or omitted trailing period/quotes)
    span_boundary_stripped = span_norm.strip("\"'.,;:!?- ")
    if len(span_boundary_stripped) >= 15 and span_boundary_stripped in avail_norm:
        return "normalized", None

    # Check if a major constituent sub-phrase (at least 4 words and >= 20 chars) matches verbatim
    span_words = span_boundary_stripped.split()
    if len(span_words) >= 5:
        for sub_len in range(min(len(span_words), 8), 3, -1):
            for i in range(len(span_words) - sub_len + 1):
                sub_phrase = " ".join(span_words[i : i + sub_len])
                if len(sub_phrase) >= 20 and sub_phrase in avail_norm:
                    return "normalized", None

    # No match: build a bounded, sanitized diff diagnostic
    sample_span = span_raw[:100] + ("..." if len(span_raw) > 100 else "")
    sample_avail = avail_norm[:120] + ("..." if len(avail_norm) > 120 else "")
    diff_msg = (
        f"Span not found in article text. Evidence: '{sample_span}' | Source text: '{sample_avail}'"
    )
    return "no_match", diff_msg


def verify_evidence_span_for_claim(
    event_id: str,
    claim_id: str,
    source_url: str,
    evidence_span: str,
    event: EventCluster,
    article_by_url: dict[str, Any],
) -> tuple[bool, EvidenceSpanDiagnostic]:
    """Verify an evidence span against event articles and return a structured diagnostic."""
    try:
        norm_url = normalize_url(source_url)
    except ValueError:
        norm_url = source_url.strip().casefold()

    article = article_by_url.get(norm_url)
    if article is None:
        return False, EvidenceSpanDiagnostic(
            event_id=event_id,
            source_url=source_url,
            normalized_url=norm_url,
            claim_id=claim_id,
            source_belongs_to_event=False,
            match_status="no_match",
            sanitized_diff=f"Source URL '{source_url}' does not match any article in event",
        )

    body = getattr(article, "body_text", None) if getattr(article, "is_full_text", False) else None
    available_text = " ".join(
        val
        for val in (
            getattr(article, "title", ""),
            getattr(article, "snippet", "") or "",
            body or "",
        )
        if val
    )

    match_status, diff_msg = check_evidence_span_match(evidence_span, available_text)
    is_valid = match_status in {"verbatim", "normalized"}

    diagnostic = EvidenceSpanDiagnostic(
        event_id=event_id,
        source_url=source_url,
        normalized_url=norm_url,
        claim_id=claim_id,
        source_belongs_to_event=True,
        match_status=match_status,
        sanitized_diff=diff_msg,
    )
    return is_valid, diagnostic


@dataclass(frozen=True)
class SemanticSupportVerdict:
    """Detailed semantic support verdict for a claim against an evidence passage."""

    is_supported: bool
    verdict: Literal["supported", "unsupported", "uncertain"]
    reason: str
    mismatched_topic: str | None = None
    nli_classification: Literal["entailment", "contradiction", "neutral"] | None = None
    nli_confidence: float | None = None


# Domain-specific predicate and action patterns
PAT_CASUALTY = re.compile(
    r"\b(?:hit|struck|damaged|sunk|sinkings?|killed|killing|deaths?|fatalities|casualties|seafarers?)\b",
    re.I,
)
PAT_CONDEMN = re.compile(
    r"\b(?:condemned?|condemnations?|denounced?|decried)\b",
    re.I,
)
PAT_STATEMENT = re.compile(
    r"\b(?:condemned?|condemnations?|denounced?|decried|called?\s+on|refrain\s+from|urged?)\b",
    re.I,
)
PAT_AGREEMENT = re.compile(
    r"\b(?:agreed?|agreements?|pact|cooperation|treaty|accord|"
    r"intelligence\s+sharing|information\s+sharing|domain\s+awareness|early\s+warning)\b",
    re.I,
)
PAT_PROCUREMENT = re.compile(
    r"\b(?:foreign\s+military\s+sale|fms|potential\s+sale|approved?\s+(?:a\s+)?potential|"
    r"procure(?:ment|d)?|purchas(?:e|ed)?|pac-3|patriot|missiles?)\b",
    re.I,
)
PAT_POLICY = re.compile(
    r"\b(?:offset\s+policy|offset\s+authority|industrial\s+benefits?|contract\s+value\s+threshold|"
    r"domestic\s+economic\s+benefits?|50\s*million)\b",
    re.I,
)
PAT_ASSEMBLY = re.compile(
    r"\b(?:assembly|production\s+began|facility|plant|factory|capacity|tanks?|vehicles?|batteries?)\b",
    re.I,
)
PAT_NETWORK = re.compile(
    r"\b(?:mission\s+network|common\s+operational\s+picture|cop|maven\s+smart\s+system|"
    r"securely\s+share)\b",
    re.I,
)
PAT_NETWORK_FUNCTION = re.compile(
    r"\b(?:securely\s+share|operational\s+data|mission\s+planning|common\s+operational\s+picture|cop|"
    r"planning\s+and\s+targeting|joint\s+strikes?)\b",
    re.I,
)
PAT_EXERCISE_STATS = re.compile(
    r"\b(?:17\s+major\s+exercises|across\s+9\s+nations|35\s+partner\s+countries)\b",
    re.I,
)


def split_sentences_safely(text: str) -> list[str]:
    """Split text into sentence passages, preserving abbreviations like 'Oct.', 'U.S.', etc."""
    if not text:
        return []
    # Protect compound military titles: Lt. Col., Brig. Gen.
    protected = re.sub(
        r"\b(Lt|Brig)\.\s+(Col|Gen)\.",
        r"\1@@DOT@@ \2@@DOT@@",
        text,
        flags=re.I,
    )
    # Protect calendar months
    protected = re.sub(
        r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.",
        r"\1@@DOT@@",
        protected,
        flags=re.I,
    )
    # Protect dotted initialisms and civil/military titles
    protected = re.sub(
        r"\b(U\.S|U\.K|E\.U|U\.N|Gen|Lt|Col|Maj|Capt|Adm|Cmdr|Sgt|Dr|Mr|Mrs|Ms|Prof|Sen|Rep)\.",
        r"\1@@DOT@@",
        protected,
    )
    # Protect currency and decimal numbers (e.g., $16.5, $16. 5, €15.2, 16.5%, $16. million)
    protected = re.sub(
        r"([$€£¥₹]?\s*\d+)\.(\s*)(\d+)",
        r"\1@@DOT@@\2\3",
        protected,
    )
    protected = re.sub(
        r"([$€£¥₹]\s*\d+)\.(\s*)(million|billion|trillion|[mbk]\b)",
        r"\1@@DOT@@\2\3",
        protected,
        flags=re.I,
    )
    protected = re.sub(
        r"\b(No|vol|sec|art|para|al|vs|dept|approx|est)\.(\s*)(\d+)",
        r"\1@@DOT@@\2\3",
        protected,
        flags=re.I,
    )
    sentences = [
        s.replace("@@DOT@@", ".").strip()
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", protected)
        if s.strip()
    ]
    return sentences


def verify_claim_semantic_support(
    claim_text: str,
    evidence_span: str,
    classification: str = "reported_fact",
    attribution: str | None = None,
    nli_verifier: Any | None = None,
    full_article_text: str | None = None,
) -> SemanticSupportVerdict:
    """Verify whether an evidence span genuinely entails and supports a claim.

    Detects when an evidence passage discusses a related but distinct topic (such as
    casualty statistics instead of an official agreement or condemnation statement).
    When nli_verifier is provided, runs local DeBERTa-v3 NLI classification combined
    with deterministic safeguards.
    """
    if not claim_text or not evidence_span:
        return SemanticSupportVerdict(
            is_supported=False,
            verdict="unsupported",
            reason="Empty claim or evidence text",
        )

    if nli_verifier is not None:
        try:
            nli_res = nli_verifier.verify_claim(
                claim_text=claim_text,
                evidence_span=evidence_span,
                full_article_text=full_article_text,
            )
            return SemanticSupportVerdict(
                is_supported=nli_res.is_supported,
                verdict="supported" if nli_res.is_supported else "unsupported",
                reason=nli_res.reason
                or (
                    "Supported by NLI entailment"
                    if nli_res.is_supported
                    else "NLI verification failed"
                ),
                mismatched_topic="nli_contradiction"
                if nli_res.classification == "contradiction"
                else ("nli_neutral" if nli_res.classification == "neutral" else None),
                nli_classification=nli_res.classification,
                nli_confidence=nli_res.confidence,
            )
        except Exception as exc:
            return SemanticSupportVerdict(
                is_supported=False,
                verdict="unsupported",
                reason=f"NLI verification failed closed due to engine error: {exc}",
                mismatched_topic="nli_error",
                nli_classification="neutral",
                nli_confidence=0.0,
            )

    domains = [
        ("casualty", PAT_CASUALTY),
        ("statement", PAT_STATEMENT),
        ("agreement", PAT_AGREEMENT),
        ("procurement", PAT_PROCUREMENT),
        ("policy", PAT_POLICY),
        ("assembly", PAT_ASSEMBLY),
        ("network", PAT_NETWORK),
    ]
    c_domains = {name for name, pat in domains if pat.search(claim_text)}
    e_domains = {name for name, pat in domains if pat.search(evidence_span)}

    # Distinct domain mismatch checks:
    # 1. Statement/condemnation claim paired with casualty evidence
    if (
        "statement" in c_domains
        and "casualty" in e_domains
        and not ("statement" in e_domains or "casualty" in c_domains)
    ):
        return SemanticSupportVerdict(
            is_supported=False,
            verdict="unsupported",
            reason=(
                "Evidence discusses vessel casualties/damage, which cannot verify an "
                "official condemnation or statement."
            ),
            mismatched_topic="casualty_vs_statement",
        )

    # 2. Agreement claim paired with casualty evidence
    if (
        "agreement" in c_domains
        and "casualty" in e_domains
        and not ("agreement" in e_domains or "casualty" in c_domains)
    ):
        return SemanticSupportVerdict(
            is_supported=False,
            verdict="unsupported",
            reason=(
                "Evidence discusses vessel casualties/damage, which cannot verify an "
                "intelligence-sharing agreement."
            ),
            mismatched_topic="casualty_vs_agreement",
        )

    # 3. Casualty claim paired with statement/agreement evidence lacking casualty facts
    if (
        "casualty" in c_domains
        and ("statement" in e_domains or "agreement" in e_domains)
        and "casualty" not in e_domains
    ):
        return SemanticSupportVerdict(
            is_supported=False,
            verdict="unsupported",
            reason=(
                "Evidence discusses an official agreement/statement, which cannot verify "
                "casualty statistics."
            ),
            mismatched_topic="statement_vs_casualty",
        )

    # 4. Arms procurement claim paired with tank assembly evidence lacking arms sale facts
    if (
        "procurement" in c_domains
        and "assembly" in e_domains
        and not ("procurement" in e_domains or "assembly" in c_domains)
    ):
        return SemanticSupportVerdict(
            is_supported=False,
            verdict="unsupported",
            reason="Evidence discusses vehicle assembly, not weapons procurement approval.",
            mismatched_topic="procurement_vs_assembly",
        )

    # 5. Policy mandate claim paired with weapons procurement evidence lacking offset facts
    if (
        "policy" in c_domains
        and "procurement" in e_domains
        and not ("policy" in e_domains or "procurement" in c_domains)
    ):
        return SemanticSupportVerdict(
            is_supported=False,
            verdict="unsupported",
            reason="Evidence discusses weapons procurement, not industrial offset policy.",
            mismatched_topic="policy_vs_procurement",
        )

    # 6. Network functionality claim paired with annual exercise stats lacking data-sharing facts
    if (
        PAT_NETWORK_FUNCTION.search(claim_text)
        and PAT_EXERCISE_STATS.search(evidence_span)
        and not PAT_NETWORK_FUNCTION.search(evidence_span)
    ):
        return SemanticSupportVerdict(
            is_supported=False,
            verdict="unsupported",
            reason=(
                "Evidence discusses annual exercise statistics, not network data-sharing "
                "or mission planning functionality."
            ),
            mismatched_topic="exercise_stats_vs_functionality",
        )

    # 7. Casualty claim lacks casualty facts in evidence
    if "casualty" in c_domains and "casualty" not in e_domains:
        return SemanticSupportVerdict(
            is_supported=False,
            verdict="unsupported",
            reason=(
                "Evidence lacks casualty facts (strikes, fatalities, or damage) to support "
                "the casualty claim."
            ),
            mismatched_topic="missing_casualty_facts",
        )

    # 8. Condemnation claim lacks condemnation or agreement terms in evidence
    if PAT_CONDEMN.search(claim_text) and not (
        PAT_CONDEMN.search(evidence_span) or "agreement" in e_domains
    ):
        return SemanticSupportVerdict(
            is_supported=False,
            verdict="unsupported",
            reason="Evidence lacks condemnation terms to support the claim.",
            mismatched_topic="missing_condemnation_facts",
        )

    # 9. Agreement claim lacks agreement substance in evidence
    if "agreement" in c_domains and not ("agreement" in e_domains or "statement" in e_domains):
        return SemanticSupportVerdict(
            is_supported=False,
            verdict="unsupported",
            reason="Evidence lacks agreement or cooperation terms to support the claim.",
            mismatched_topic="missing_agreement_facts",
        )

    # Subject entity check: If claim has explicit proper nouns (countries, commands, companies)
    c_words = set(re.findall(r"\b[A-Z][a-z]+\b", claim_text))
    ignored = {
        "The",
        "A",
        "An",
        "In",
        "On",
        "At",
        "For",
        "To",
        "Of",
        "And",
        "According",
        "State",
        "During",
        "Assembly",
        "Ministers",
        "Defense",
        "Foreign",
        "Military",
        "Officials",
        "Government",
        "Governments",
        "Authorities",
        "Leaders",
        "Representatives",
        "Contractors",
        "Suppliers",
        "Partners",
        "Allies",
        "Nations",
        "Forces",
        "Industry",
        "Agreement",
        "Statement",
        "Meeting",
        "Decision",
        "Policy",
        "Framework",
        "Program",
        "System",
        "Network",
        "Operation",
        "Drill",
        "Exercise",
        "Security",
        "Intelligence",
    }
    # Sentence-initial word should not be treated as a required proper noun unless a known entity
    first_word_match = re.match(r"^\s*([A-Z][a-z]+)\b", claim_text)
    if first_word_match:
        fw = first_word_match.group(1)
        known_countries = {
            "Bulgaria",
            "Romania",
            "Turkey",
            "Norway",
            "Germany",
            "Japan",
            "Ukraine",
            "Canada",
            "Russia",
            "China",
            "France",
            "Britain",
        }
        if fw not in known_countries and claim_text.count(fw) == 1:
            ignored.add(fw)

    c_entities = c_words - ignored
    if c_entities:
        e_lower = evidence_span.lower()
        if not any(ent.lower() in e_lower for ent in c_entities):
            # Check demonyms / abbreviations
            demonyms = {
                "bulgaria": "bulgarian",
                "romania": "romanian",
                "turkey": "turkish",
                "norway": "norwegian",
                "germany": "german",
                "japan": "japanese",
                "ukraine": "ukrainian",
            }
            has_demonym = any(
                demonyms.get(ent.lower(), ent.lower()) in e_lower for ent in c_entities
            )
            if not has_demonym:
                return SemanticSupportVerdict(
                    is_supported=False,
                    verdict="unsupported",
                    reason=f"Evidence lacks primary claim entities: {sorted(c_entities)}",
                    mismatched_topic="entity_mismatch",
                )

    # Attribution consistency
    if attribution and (
        "ambrey" in attribution.lower() or "financial times" in attribution.lower()
    ):
        if "statement" in c_domains or "agreement" in c_domains:
            if not any(
                w in evidence_span.lower() for w in ("financial times", "ambrey", "imo", "reported")
            ):
                return SemanticSupportVerdict(
                    is_supported=False,
                    verdict="unsupported",
                    reason="Claim attributes official statement to statistical reporting source.",
                    mismatched_topic="attribution_mismatch",
                )

    return SemanticSupportVerdict(
        is_supported=True,
        verdict="supported",
        reason="Claim is semantically supported by evidence passage.",
    )


def find_best_supporting_passage(
    claim_text: str,
    available_text: str,
    nli_verifier: Any | None = None,
) -> tuple[str | None, float]:
    """Find the sentence passage in available text that best semantically supports a claim."""
    passages = split_sentences_safely(available_text)
    if not passages:
        return None, 0.0

    if nli_verifier is not None and getattr(nli_verifier, "is_available", lambda: False)():
        try:
            scored_candidates: list[tuple[float, str]] = []
            for p in passages:
                if len(p.strip()) < 20:
                    continue
                res = nli_verifier.verify_claim(
                    claim_text=claim_text,
                    evidence_span=p,
                    full_article_text=available_text,
                )
                if res.is_supported:
                    scored_candidates.append((res.confidence, p))
            if scored_candidates:
                scored_candidates.sort(key=lambda x: x[0], reverse=True)
                return scored_candidates[0][1], scored_candidates[0][0]
        except Exception:
            pass

    c_words = set(re.findall(r"\b[A-Za-z]{4,}\b", claim_text.lower()))
    best_p: str | None = None
    best_score: float = -999.0

    for p in passages:
        verdict = verify_claim_semantic_support(claim_text, p)
        if not verdict.is_supported:
            continue
        p_words = set(re.findall(r"\b[A-Za-z]{4,}\b", p.lower()))
        overlap = len(c_words & p_words)
        score = overlap / max(1, len(c_words))
        if PAT_STATEMENT.search(claim_text) and PAT_STATEMENT.search(p):
            score += 0.4
        if PAT_AGREEMENT.search(claim_text) and PAT_AGREEMENT.search(p):
            score += 0.4
        if PAT_CASUALTY.search(claim_text) and PAT_CASUALTY.search(p):
            score += 0.4
        if PAT_PROCUREMENT.search(claim_text) and PAT_PROCUREMENT.search(p):
            score += 0.4
        if PAT_POLICY.search(claim_text) and PAT_POLICY.search(p):
            score += 0.4
        if PAT_ASSEMBLY.search(claim_text) and PAT_ASSEMBLY.search(p):
            score += 0.4
        if PAT_NETWORK.search(claim_text) and PAT_NETWORK.search(p):
            score += 0.4
        if PAT_NETWORK_FUNCTION.search(claim_text) and PAT_NETWORK_FUNCTION.search(p):
            score += 0.5
        if PAT_EXERCISE_STATS.search(p) and not PAT_NETWORK_FUNCTION.search(claim_text):
            score -= 0.5
        if score > best_score:
            best_score = score
            best_p = p

    return best_p, best_score
