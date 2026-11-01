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

    span_raw = clean_article_text(evidence_span)
    clean_avail = clean_article_text(available_text)
    if span_raw in available_text or span_raw in clean_avail:
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

    available_text = " ".join(
        val for val in (getattr(article, "title", ""), getattr(article, "snippet", "") or "") if val
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
