"""Conservative URL normalization, HTML stripping, and deterministic content hashing."""

import hashlib
import html
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# Known marketing, social, and ad tracking query parameters.
# Preserves all meaningful application parameters (e.g., id, v, story_id, category, page).
TRACKING_QUERY_PARAMS: frozenset[str] = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "utm_id",
        "utm_name",
        "fbclid",
        "gclid",
        "gbraid",
        "wbraid",
        "msclkid",
        "dclid",
        "mc_cid",
        "mc_eid",
        "_hsenc",
        "_hsmi",
        "igshid",
        "ref",
        "ref_src",
        "ref_url",
        "ocid",
        "at_medium",
        "at_campaign",
    }
)

TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(raw_text: str | None) -> str | None:
    """Strip HTML tags, unescape HTML entities, and collapse whitespace."""
    if not raw_text:
        return None
    # Strip HTML tags
    no_html = TAG_RE.sub(" ", raw_text)
    # Unescape HTML entities (&amp;, &quot;, &#39;, etc.)
    unescaped = html.unescape(no_html)
    # Normalize whitespace
    collapsed = WHITESPACE_RE.sub(" ", unescaped).strip()
    return collapsed or None


def normalize_url(raw_url: str) -> str:
    """Conservatively normalize an article URL and strip tracking parameters.

    Preserves article identity, meaningful query parameters, and path structure.
    """
    cleaned = raw_url.strip()
    if not cleaned:
        raise ValueError("URL cannot be empty")

    split = urlsplit(cleaned)
    scheme = split.scheme.lower()
    if scheme not in {"http", "https"}:
        raise ValueError(f"Invalid URL scheme: {split.scheme}")

    # Normalize netloc: lowercase, remove standard default ports
    netloc = split.netloc.lower()
    if netloc.endswith(":80") and scheme == "http":
        netloc = netloc[:-3]
    elif netloc.endswith(":443") and scheme == "https":
        netloc = netloc[:-4]

    # Normalize path: collapse redundant slashes and strip trailing slash unless root
    path = re.sub(r"/{2,}", "/", split.path)
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    if not path:
        path = "/"

    # Filter out tracking query parameters, preserving all content query params
    filtered_params: list[tuple[str, str]] = []
    if split.query:
        for k, v in parse_qsl(split.query, keep_blank_values=True):
            if k.lower() not in TRACKING_QUERY_PARAMS:
                filtered_params.append((k, v))

    # Deterministic query string sorted by key
    filtered_params.sort(key=lambda item: (item[0], item[1]))
    query_str = urlencode(filtered_params) if filtered_params else ""

    # Fragments are stripped because they do not identify distinct news articles
    return urlunsplit((scheme, netloc, path, query_str, ""))


def compute_content_hash(
    canonical_url: str,
    title: str,
    snippet: str | None = None,
) -> str:
    """Compute deterministic SHA-256 content hash for duplicate detection."""
    norm_title = (clean_text(title) or "").strip().lower()
    norm_snippet = (clean_text(snippet) or "").strip().lower()
    payload = f"{canonical_url.strip()}\n{norm_title}\n{norm_snippet}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
