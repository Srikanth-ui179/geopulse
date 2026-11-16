"""Respectful, cached full-article text extraction layer with immutable revision history."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx
import trafilatura

from geopulse.ingestion.models import (
    ArticleRecord,
    ArticleRevisionRecord,
    RevisionClassification,
)
from geopulse.ingestion.normalization import normalize_url

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "Geopulse/0.1.0 (+https://github.com/geopulse-bot; contact@geopulse.local)"
DEFAULT_EXTRACTOR_TIMEOUT = 10.0
DEFAULT_CACHE_DIR = Path(".geopulse/article_cache")
DEFAULT_CACHE_TTL_DAYS = 7
DEFAULT_REVALIDATION_INTERVAL_MINUTES = 15
DEFAULT_MIN_DOMAIN_INTERVAL = 1.0  # Respectful 1-second interval between requests to same domain

ExtractionStatus = Literal[
    "success",
    "cached",
    "updated",
    "failed",
    "paywall_detected",
    "robots_denied",
    "empty",
    "snippet_fallback",
]

_PAYWALL_PATTERNS = re.compile(
    r"\b(subscribe to (?:continue|read)|sign in to read the full (?:article|story)|"
    r"this (?:article|content) is reserved for subscribers|subscribers only|"
    r"already a subscriber\? log in|start your free trial to continue reading|"
    r"purchase a subscription|become a member to continue reading)\b",
    re.I,
)

_EXPLICIT_RETRACTION_PATTERNS = re.compile(
    r"\b(?:retraction(?:\s+notice)?|has\s+retracted\s+(?:this\s+)?(?:article|story|report)|"
    r"this\s+(?:article|story|report)\s+has\s+been\s+retracted)\b",
    re.I,
)

_EXPLICIT_CORRECTION_PATTERNS = re.compile(
    r"\b(?:correction(?:\s+notice|\s+appended)?|"
    r"editor'?s\s+note(?:\s*:|\s+--|\s+—)?\s*(?:a\s+previous\s+version|an\s+earlier\s+version|"
    r"this\s+article\s+has\s+been\s+corrected|this\s+story\s+has\s+been\s+updated\s+to\s+correct|"
    r"clarification\s*:)|"
    r"an\s+earlier\s+version\s+of\s+this\s+article\s+(?:misstated|incorrectly\s+stated|erroneously\s+stated))\b",
    re.I,
)


def classify_revision(
    body_text: str | None,
    title: str | None,
    extraction_status: str,
    previous_revision: ArticleRevisionRecord | None,
    new_content_hash: str | None,
) -> RevisionClassification:
    """Classify the nature of an article revision objectively."""
    if extraction_status in {"failed", "empty", "paywall_detected", "robots_denied"}:
        return "unavailability"

    if previous_revision is None:
        return "initial_publication"

    combined_text = f"{title or ''}\n{body_text or ''}"

    # 1. Explicit Retraction notice published by outlet
    if _EXPLICIT_RETRACTION_PATTERNS.search(combined_text):
        return "explicit_retraction"

    # 2. Explicit Correction notice published by outlet
    if _EXPLICIT_CORRECTION_PATTERNS.search(combined_text):
        return "explicit_correction"

    # 3. Content Update (body text materially changed)
    if new_content_hash and previous_revision.content_hash:
        if new_content_hash != previous_revision.content_hash:
            return "content_update"

    # 4. Metadata Update (title or dates changed without substantive body change)
    if title and previous_revision.title and title.strip() != previous_revision.title.strip():
        return "metadata_update"

    return "content_update"


@dataclass(frozen=True)
class ArticlePassage:
    """A bounded passage extracted from an article body with document offsets."""

    section_label: str
    start_char: int
    end_char: int
    text: str


def select_bounded_article_passages(
    body_text: str,
    max_chars: int = 1800,
) -> tuple[str, list[ArticlePassage]]:
    """Select bounded, high-information passages spanning the full article.

    Ensures balanced coverage across opening, middle, and concluding sections,
    retaining character offsets for precise evidence tracing.
    """
    if not body_text or not body_text.strip():
        return "", []

    body_clean = body_text.strip()
    if len(body_clean) <= max_chars:
        passage = ArticlePassage(
            section_label="Full Body",
            start_char=0,
            end_char=len(body_clean),
            text=body_clean,
        )
        return body_clean, [passage]

    raw_paras: list[tuple[int, int, str]] = []
    for m in re.finditer(r"[^\n]+(?:\n(?![ \t]*\n)[^\n]+)*", body_clean):
        p_text = m.group(0).strip()
        if len(p_text) >= 40:
            raw_paras.append((m.start(), m.end(), p_text))

    if len(raw_paras) <= 2:
        truncated = body_clean[:max_chars].rsplit(" ", 1)[0]
        passage = ArticlePassage(
            section_label="Opening Section",
            start_char=0,
            end_char=len(truncated),
            text=truncated,
        )
        return truncated, [passage]

    overhead_chars = 160
    effective_budget = max(200, max_chars - overhead_chars)

    # 1. Opening section
    open_start, open_end, open_text = raw_paras[0]
    opening_label = "Lead Development"
    open_budget = max(120, int(effective_budget * 0.40))
    if len(open_text) > open_budget:
        open_text = open_text[:open_budget].rsplit(" ", 1)[0]
        open_end = open_start + len(open_text)

    # 2. Concluding section
    close_start, close_end, close_text = raw_paras[-1]
    concluding_label = "Concluding Context & Horizon"
    close_budget = max(100, int(effective_budget * 0.25))
    if len(close_text) > close_budget:
        close_text = close_text[:close_budget].rsplit(" ", 1)[0]
        close_end = close_start + len(close_text)

    # 3. Substantive middle paragraphs
    middle_candidates = raw_paras[1:-1]
    remaining_budget = effective_budget - len(open_text) - len(close_text)

    selected_middle: list[tuple[int, int, str, str]] = []
    if middle_candidates and remaining_budget >= 100:
        scored: list[tuple[float, int, int, str]] = []
        for s_idx, e_idx, t_str in middle_candidates:
            score = 1.0
            if re.search(
                r"\b(?:[$£€]?\d+(?:[.,]\d+)?(?:\s*(?:billion|million|percent|%|m|bn))?)\b",
                t_str,
                re.I,
            ):
                score += 3.0
            if re.search(
                r"\b(?:agreement|contract|procurement|subcontract|initiative|partnership|"
                r"facility|production|assembly|network|feeds?|tools?|translation|interoperability|"
                r"submarines?|missiles?|interceptors?|satellites?|spaceport|surveillance|"
                r"intelligence[- ]sharing|early[- ]warning|domain\s+awareness|threshold|policy|"
                r"capacity|battery|batteries|patrols?|joint\s+strikes?)\b",
                t_str,
                re.I,
            ):
                score += 3.0
            if re.search(
                r"\b(?:january|february|march|april|may|june|july|august|september|october|"
                r"november|december|202[4-9])\b",
                t_str,
                re.I,
            ):
                score += 1.5
            entities = re.findall(r"\b[A-Z][a-z]{3,}\b", t_str)
            score += min(len(entities) * 0.4, 2.0)
            scored.append((score, s_idx, e_idx, t_str))

        scored.sort(key=lambda x: x[0], reverse=True)

        budget_left = remaining_budget
        for _sc, s_idx, e_idx, t_str in scored:
            cand_text = t_str
            if len(cand_text) > budget_left:
                if budget_left >= 80:
                    cand_text = cand_text[:budget_left].rsplit(" ", 1)[0]
                    e_idx = s_idx + len(cand_text)
                else:
                    continue
            selected_middle.append((s_idx, e_idx, "Key Operational Passage", cand_text))
            budget_left -= len(cand_text)
            if budget_left < 80:
                break

    all_sections: list[tuple[int, int, str, str]] = [
        (open_start, open_end, opening_label, open_text),
        *selected_middle,
        (close_start, close_end, concluding_label, close_text),
    ]
    all_sections.sort(key=lambda x: x[0])

    blocks: list[str] = []
    passages: list[ArticlePassage] = []
    for idx, (start, end, label, text) in enumerate(all_sections, start=1):
        header = f"[Section {idx}: {label} | chars {start}-{end}]"
        blocks.append(f"{header}\n{text}")
        passages.append(
            ArticlePassage(
                section_label=label,
                start_char=start,
                end_char=end,
                text=text,
            )
        )

    return "\n\n".join(blocks), passages


@dataclass(frozen=True)
class ExtractedContent:
    """Structured extraction result with full provenance metadata and revision record."""

    url: str
    canonical_url: str
    body_text: str | None
    extraction_status: ExtractionStatus
    status_reason: str | None
    extracted_at: datetime
    content_hash: str | None
    word_count: int
    is_full_text: bool
    author: str | None = None
    title: str | None = None
    etag: str | None = None
    last_modified: str | None = None
    revalidated_at: datetime | None = None
    previous_content_hashes: tuple[str, ...] = ()
    revision: ArticleRevisionRecord | None = None


class ArticleExtractor:
    """Respectful, rate-limited, cached article body extraction with immutable revisions."""

    def __init__(
        self,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        cache_ttl_days: int = DEFAULT_CACHE_TTL_DAYS,
        revalidation_interval_minutes: int = DEFAULT_REVALIDATION_INTERVAL_MINUTES,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = DEFAULT_EXTRACTOR_TIMEOUT,
        min_domain_interval: float = DEFAULT_MIN_DOMAIN_INTERVAL,
        check_robots: bool = True,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_ttl = timedelta(days=cache_ttl_days)
        self.revalidation_interval = timedelta(minutes=revalidation_interval_minutes)
        self.user_agent = user_agent
        self.timeout = timeout
        self.min_domain_interval = min_domain_interval
        self.check_robots = check_robots
        self._external_client = http_client

        # Concurrency safety & rate limiting
        self._lock = threading.RLock()
        self._last_domain_request: dict[str, float] = {}
        self._robots_cache: dict[str, RobotFileParser] = {}

    def _get_cache_path(self, canonical_url: str) -> Path:
        """Derive deterministic flat cache path for backwards compatibility."""
        url_hash = hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{url_hash[:2]}" / f"{url_hash}.json"

    def _get_article_dir(self, canonical_url: str) -> Path:
        """Derive deterministic article revision directory path."""
        url_hash = hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{url_hash[:2]}" / url_hash

    def _get_manifest_path(self, canonical_url: str) -> Path:
        """Path to article revisions manifest."""
        return self._get_article_dir(canonical_url) / "manifest.json"

    def _get_revisions_dir(self, canonical_url: str) -> Path:
        """Path to article revisions directory."""
        return self._get_article_dir(canonical_url) / "revisions"

    def get_revision_history(self, canonical_url: str) -> list[ArticleRevisionRecord]:
        """Retrieve all recorded revisions for an article in sequence order."""
        with self._lock:
            manifest_path = self._get_manifest_path(canonical_url)
            if not manifest_path.exists():
                return []

            try:
                with open(manifest_path, encoding="utf-8") as f:
                    manifest = json.load(f)

                rev_ids = manifest.get("revisions", [])
                rev_dir = self._get_revisions_dir(canonical_url)

                revisions: list[ArticleRevisionRecord] = []
                for r_id in rev_ids:
                    r_file = rev_dir / f"{r_id}.json"
                    if r_file.exists():
                        try:
                            with open(r_file, encoding="utf-8") as rf:
                                r_data = json.load(rf)
                            revisions.append(ArticleRevisionRecord.model_validate(r_data))
                        except Exception as err:
                            logger.warning(
                                "Failed reading revision %s for %s: %s", r_id, canonical_url, err
                            )
                return revisions
            except Exception as exc:
                logger.debug("Failed reading manifest for %s: %s", canonical_url, exc)
                return []

    def get_revision(self, canonical_url: str, revision_id: str) -> ArticleRevisionRecord | None:
        """Retrieve a specific article revision by its unique revision ID."""
        with self._lock:
            r_file = self._get_revisions_dir(canonical_url) / f"{revision_id}.json"
            if not r_file.exists():
                return None
            try:
                with open(r_file, encoding="utf-8") as rf:
                    r_data = json.load(rf)
                return ArticleRevisionRecord.model_validate(r_data)
            except Exception as exc:
                logger.debug(
                    "Failed reading revision %s for %s: %s", revision_id, canonical_url, exc
                )
                return None

    def get_latest_revision(self, canonical_url: str) -> ArticleRevisionRecord | None:
        """Retrieve the current/latest revision record for an article."""
        history = self.get_revision_history(canonical_url)
        return history[-1] if history else None

    def _read_from_cache(self, canonical_url: str) -> ExtractedContent | None:
        """Read extraction record from local disk cache if valid and unexpired."""
        with self._lock:
            # 1. Try reading from immutable revision manifest
            manifest_path = self._get_manifest_path(canonical_url)
        if manifest_path.exists():
            try:
                with open(manifest_path, encoding="utf-8") as f:
                    manifest = json.load(f)

                current_rev_id = manifest.get("current_revision_id")
                latest_rev = (
                    self.get_revision(canonical_url, current_rev_id) if current_rev_id else None
                )

                if latest_rev is not None:
                    revalidated_at = None
                    if manifest.get("revalidated_at"):
                        revalidated_at = datetime.fromisoformat(manifest["revalidated_at"])
                        if revalidated_at.tzinfo is None:
                            revalidated_at = revalidated_at.replace(tzinfo=UTC)

                    history = self.get_revision_history(canonical_url)
                    prev_hashes = tuple(r.content_hash for r in history[:-1] if r.content_hash)

                    status: ExtractionStatus = "cached"
                    if latest_rev.extraction_status in {
                        "paywall_detected",
                        "robots_denied",
                        "empty",
                        "failed",
                    }:
                        status = latest_rev.extraction_status  # type: ignore

                    return ExtractedContent(
                        url=canonical_url,
                        canonical_url=canonical_url,
                        body_text=latest_rev.body_text,
                        extraction_status=status,
                        status_reason=latest_rev.relevant_metadata.get("status_reason"),
                        extracted_at=latest_rev.retrieved_at,
                        content_hash=latest_rev.content_hash,
                        word_count=len((latest_rev.body_text or "").split()),
                        is_full_text=latest_rev.relevant_metadata.get(
                            "is_full_text",
                            bool(latest_rev.body_text and len(latest_rev.body_text) >= 150),
                        ),
                        author=latest_rev.relevant_metadata.get("author"),
                        title=latest_rev.title,
                        etag=latest_rev.etag,
                        last_modified=latest_rev.last_modified,
                        revalidated_at=revalidated_at,
                        previous_content_hashes=prev_hashes,
                        revision=latest_rev,
                    )
            except Exception as exc:
                logger.debug("Failed reading manifest-based cache for %s: %s", canonical_url, exc)

        # 2. Fallback to reading legacy cache file
        cache_file = self._get_cache_path(canonical_url)
        if not cache_file.exists():
            return None

        try:
            with open(cache_file, encoding="utf-8") as f:
                data = json.load(f)

            extracted_at = datetime.fromisoformat(data["extracted_at"])
            if extracted_at.tzinfo is None:
                extracted_at = extracted_at.replace(tzinfo=UTC)

            # Check expiration
            if datetime.now(UTC) - extracted_at > self.cache_ttl:
                return None

            revalidated_at = None
            if data.get("revalidated_at"):
                revalidated_at = datetime.fromisoformat(data["revalidated_at"])
                if revalidated_at.tzinfo is None:
                    revalidated_at = revalidated_at.replace(tzinfo=UTC)

            prev_hashes = tuple(data.get("previous_content_hashes", ()))

            return ExtractedContent(
                url=data["url"],
                canonical_url=data["canonical_url"],
                body_text=data.get("body_text"),
                extraction_status="cached",
                status_reason=data.get("status_reason"),
                extracted_at=extracted_at,
                content_hash=data.get("content_hash"),
                word_count=data.get("word_count", 0),
                is_full_text=data.get("is_full_text", False),
                author=data.get("author"),
                title=data.get("title"),
                etag=data.get("etag"),
                last_modified=data.get("last_modified"),
                revalidated_at=revalidated_at,
                previous_content_hashes=prev_hashes,
                revision=None,
            )
        except Exception as exc:
            logger.debug("Failed to read cache for %s: %s", canonical_url, exc)
            return None

    def _write_to_cache(self, content: ExtractedContent) -> ExtractedContent:
        """Atomically persist extraction record and append-only revision to disk cache."""
        with self._lock:
            try:
                canonical_url = content.canonical_url
                article_id = str(
                    uuid.uuid5(uuid.NAMESPACE_URL, f"geopulse:article:{canonical_url}")
                )
                rev_dir = self._get_revisions_dir(canonical_url)
                manifest_path = self._get_manifest_path(canonical_url)

                rev_dir.mkdir(parents=True, exist_ok=True)

                # Read existing manifest if present
                manifest: dict[str, Any] = {
                    "article_id": article_id,
                    "canonical_url": canonical_url,
                    "current_revision_id": None,
                    "revisions": [],
                    "etag": content.etag,
                    "last_modified": content.last_modified,
                    "revalidated_at": (
                        content.revalidated_at.isoformat() if content.revalidated_at else None
                    ),
                }
                if manifest_path.exists():
                    try:
                        with open(manifest_path, encoding="utf-8") as mf:
                            loaded_mf = json.load(mf)
                            manifest.update(loaded_mf)
                    except Exception:
                        pass

                existing_revisions = self.get_revision_history(canonical_url)
                latest_rev = existing_revisions[-1] if existing_revisions else None

                # Determine whether a new revision should be appended
                effective_hash = content.content_hash or (
                    hashlib.sha256(content.body_text.encode("utf-8")).hexdigest()
                    if content.body_text
                    else f"empty-{canonical_url}"
                )
                effective_title = content.title or "Untitled Article"

                assigned_revision = content.revision

                # If 304 or content+title identical to latest revision, do not create new revision
                if latest_rev is not None and (
                    (
                        content.extraction_status == "cached"
                        and latest_rev.content_hash == effective_hash
                    )
                    or (
                        latest_rev.content_hash == effective_hash
                        and latest_rev.title == effective_title
                    )
                ):
                    assigned_revision = latest_rev
                    manifest["revalidated_at"] = (
                        content.revalidated_at.isoformat()
                        if content.revalidated_at
                        else datetime.now(UTC).isoformat()
                    )
                    if content.etag:
                        manifest["etag"] = content.etag
                    if content.last_modified:
                        manifest["last_modified"] = content.last_modified
                elif assigned_revision is None:
                    # New revision required
                    sequence = len(existing_revisions) + 1
                    rev_id = str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"geopulse:revision:{article_id}:{sequence}:{effective_hash}",
                        )
                    )
                    classification = classify_revision(
                        body_text=content.body_text,
                        title=content.title,
                        extraction_status=content.extraction_status,
                        previous_revision=latest_rev,
                        new_content_hash=effective_hash,
                    )
                    assigned_revision = ArticleRevisionRecord(
                        revision_id=rev_id,
                        article_id=article_id,
                        canonical_url=canonical_url,
                        content_hash=effective_hash,
                        title=effective_title,
                        publisher=urlparse(canonical_url).netloc.lower() or "Unknown Publisher",
                        published_at=None,
                        first_seen_at=latest_rev.first_seen_at
                        if latest_rev
                        else content.extracted_at,
                        retrieved_at=content.extracted_at,
                        last_modified=content.last_modified,
                        etag=content.etag,
                        body_text=content.body_text,
                        extraction_status=content.extraction_status,
                        extraction_method="trafilatura"
                        if content.is_full_text
                        else "snippet_fallback",
                        revision_sequence=sequence,
                        previous_revision_id=latest_rev.revision_id if latest_rev else None,
                        revision_classification=classification,
                        relevant_metadata={
                            "author": content.author,
                            "word_count": content.word_count,
                            "is_full_text": content.is_full_text,
                            "status_reason": content.status_reason,
                        },
                    )

                    # Persist revision file immutably
                    rev_file = rev_dir / f"{assigned_revision.revision_id}.json"
                    rev_tmp = rev_file.with_name(f"{rev_file.stem}.{uuid.uuid4().hex[:8]}.tmp")
                    with open(rev_tmp, "w", encoding="utf-8") as rf:
                        json.dump(assigned_revision.model_dump(mode="json"), rf, indent=2)
                    rev_tmp.replace(rev_file)

                    # Update manifest
                    manifest["current_revision_id"] = assigned_revision.revision_id
                    if assigned_revision.revision_id not in manifest["revisions"]:
                        manifest["revisions"].append(assigned_revision.revision_id)
                    manifest["etag"] = content.etag or manifest.get("etag")
                    manifest["last_modified"] = content.last_modified or manifest.get(
                        "last_modified"
                    )
                    manifest["revalidated_at"] = (
                        content.revalidated_at.isoformat()
                        if content.revalidated_at
                        else content.extracted_at.isoformat()
                    )

                # Persist manifest atomically
                manifest_tmp = manifest_path.with_name(
                    f"{manifest_path.stem}.{uuid.uuid4().hex[:8]}.tmp"
                )
                with open(manifest_tmp, "w", encoding="utf-8") as mf:
                    json.dump(manifest, mf, indent=2)
                manifest_tmp.replace(manifest_path)

                # Persist legacy single-file cache for backwards compatibility
                cache_file = self._get_cache_path(canonical_url)
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "url": content.url,
                    "canonical_url": content.canonical_url,
                    "body_text": content.body_text,
                    "extraction_status": content.extraction_status,
                    "status_reason": content.status_reason,
                    "extracted_at": content.extracted_at.isoformat(),
                    "content_hash": content.content_hash,
                    "word_count": content.word_count,
                    "is_full_text": content.is_full_text,
                    "author": content.author,
                    "title": content.title,
                    "etag": content.etag,
                    "last_modified": content.last_modified,
                    "revalidated_at": (
                        content.revalidated_at.isoformat() if content.revalidated_at else None
                    ),
                    "previous_content_hashes": list(content.previous_content_hashes),
                }
                legacy_tmp = cache_file.with_name(f"{cache_file.stem}.{uuid.uuid4().hex[:8]}.tmp")
                with open(legacy_tmp, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                legacy_tmp.replace(cache_file)

                # Return content enriched with revision record
                return ExtractedContent(
                    url=content.url,
                    canonical_url=content.canonical_url,
                    body_text=content.body_text,
                    extraction_status=content.extraction_status,
                    status_reason=content.status_reason,
                    extracted_at=content.extracted_at,
                    content_hash=content.content_hash,
                    word_count=content.word_count,
                    is_full_text=content.is_full_text,
                    author=content.author,
                    title=content.title,
                    etag=content.etag,
                    last_modified=content.last_modified,
                    revalidated_at=content.revalidated_at,
                    previous_content_hashes=content.previous_content_hashes,
                    revision=assigned_revision,
                )

            except Exception as exc:
                logger.debug(
                    "Failed to write extraction cache for %s: %s", content.canonical_url, exc
                )
                return content

    def apply_retention_policy(
        self,
        canonical_url: str,
        policy: Literal["full_text", "evidence_snippets_only", "metadata_only"] = "full_text",
    ) -> None:
        """Apply copyright retention rules: scrub full body text while preserving provenance."""
        if policy == "full_text":
            return

        with self._lock:
            revisions = self.get_revision_history(canonical_url)
            rev_dir = self._get_revisions_dir(canonical_url)

            for rev in revisions:
                if policy == "evidence_snippets_only":
                    # Retain bounded evidence excerpts, strip raw full article
                    pruned_text = (
                        (rev.body_text or "")[:400].rsplit(" ", 1)[0] if rev.body_text else None
                    )
                else:  # metadata_only
                    pruned_text = None

                updated_meta = dict(rev.relevant_metadata)
                updated_meta["retention_policy_applied"] = policy
                updated_meta["original_body_length"] = len(rev.body_text or "")

                pruned_rev = rev.model_copy(
                    update={
                        "body_text": pruned_text,
                        "relevant_metadata": updated_meta,
                    }
                )
                rev_file = rev_dir / f"{rev.revision_id}.json"
                with open(rev_file, "w", encoding="utf-8") as f:
                    json.dump(pruned_rev.model_dump(mode="json"), f, indent=2)

    def _respect_rate_limit(self, domain: str) -> None:
        """Enforce domain-level delay interval between successive network requests."""
        with self._lock:
            last_time = self._last_domain_request.get(domain, 0.0)
            elapsed = time.monotonic() - last_time
            if elapsed < self.min_domain_interval:
                time.sleep(self.min_domain_interval - elapsed)
            self._last_domain_request[domain] = time.monotonic()

    def _is_allowed_by_robots(self, url: str) -> bool:
        """Verify whether scraping the URL is permitted by robots.txt."""
        if not self.check_robots:
            return True

        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return True

        origin = f"{parsed.scheme}://{parsed.netloc}"
        with self._lock:
            rp = self._robots_cache.get(origin)

        if rp is None:
            rp = RobotFileParser()
            robots_url = f"{origin}/robots.txt"
            try:
                with httpx.Client(
                    timeout=httpx.Timeout(3.0),
                    headers={"User-Agent": self.user_agent},
                    follow_redirects=True,
                ) as client:
                    resp = client.get(robots_url)
                    if resp.status_code == 200:
                        rp.parse(resp.text.splitlines())
                    else:
                        rp.allow_all = True
            except Exception:
                rp.allow_all = True

            with self._lock:
                self._robots_cache[origin] = rp

        return rp.can_fetch(self.user_agent, url)

    def extract_url(
        self,
        url: str,
        canonical_url: str | None = None,
        original_snippet: str | None = None,
        force_revalidate: bool = False,
        title: str | None = None,
        publisher: str | None = None,
    ) -> ExtractedContent:
        """Fetch and cleanly extract substantive article body text with provenance tracking."""
        try:
            norm_canonical = canonical_url or normalize_url(url)
        except ValueError:
            norm_canonical = url

        # 1. Check local cache
        cached = self._read_from_cache(norm_canonical)
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()

        if cached is not None:
            last_checked = cached.revalidated_at or cached.extracted_at
            is_stale_for_revalidation = (
                datetime.now(UTC) - last_checked > self.revalidation_interval
            )
            # If fresh and not forced, return cached content immediately
            if not force_revalidate and not is_stale_for_revalidation:
                return cached

            # Conditional HTTP Revalidation if ETag or Last-Modified are present
            if cached.etag or cached.last_modified:
                cond_headers = {
                    "User-Agent": self.user_agent,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                }
                if cached.etag:
                    cond_headers["If-None-Match"] = cached.etag
                if cached.last_modified:
                    cond_headers["If-Modified-Since"] = cached.last_modified

                self._respect_rate_limit(domain)
                try:
                    if self._external_client is not None:
                        resp = self._external_client.get(url, headers=cond_headers)
                    else:
                        with httpx.Client(
                            timeout=self.timeout,
                            headers=cond_headers,
                            follow_redirects=True,
                        ) as client:
                            resp = client.get(url)

                    # 304 Not Modified: Cache is fresh!
                    if resp.status_code == 304:
                        updated_cached = ExtractedContent(
                            url=cached.url,
                            canonical_url=cached.canonical_url,
                            body_text=cached.body_text,
                            extraction_status="cached",
                            status_reason=None,
                            extracted_at=cached.extracted_at,
                            content_hash=cached.content_hash,
                            word_count=cached.word_count,
                            is_full_text=cached.is_full_text,
                            author=cached.author,
                            title=cached.title or title,
                            etag=resp.headers.get("etag") or cached.etag,
                            last_modified=resp.headers.get("last-modified") or cached.last_modified,
                            revalidated_at=datetime.now(UTC),
                            previous_content_hashes=cached.previous_content_hashes,
                            revision=cached.revision,
                        )
                        return self._write_to_cache(updated_cached)

                    # 200 OK: Content has changed or server does not support 304
                    if resp.status_code == 200:
                        return self._process_html_response(
                            html=resp.text,
                            url=url,
                            norm_canonical=norm_canonical,
                            original_snippet=original_snippet,
                            resp_headers=getattr(resp, "headers", {}) or {},
                            existing_cached=cached,
                            title=title,
                            publisher=publisher,
                        )

                except Exception as exc:
                    logger.warning("Revalidation network issue for %s: %s; using cached", url, exc)
                    return cached

        # 2. Check robots.txt
        if not self._is_allowed_by_robots(url):
            logger.info("Extraction disallowed by robots.txt for URL: %s", url)
            content = ExtractedContent(
                url=url,
                canonical_url=norm_canonical,
                body_text=original_snippet,
                extraction_status="robots_denied",
                status_reason="Disallowed by publisher robots.txt",
                extracted_at=datetime.now(UTC),
                content_hash=None,
                word_count=len((original_snippet or "").split()),
                is_full_text=False,
                title=title,
            )
            return self._write_to_cache(content)

        # 3. Rate-limited respectful network fetch
        self._respect_rate_limit(domain)

        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

        try:
            if self._external_client is not None:
                resp = self._external_client.get(url, headers=headers)
            else:
                with httpx.Client(
                    timeout=self.timeout,
                    headers=headers,
                    follow_redirects=True,
                ) as client:
                    resp = client.get(url)

            # Handle paywall or forbidden status codes (401, 403)
            if resp.status_code in {401, 403}:
                logger.info("Paywall or auth required (HTTP %d) for %s", resp.status_code, url)
                content = ExtractedContent(
                    url=url,
                    canonical_url=norm_canonical,
                    body_text=original_snippet,
                    extraction_status="paywall_detected",
                    status_reason=f"HTTP Error {resp.status_code} (Paywall/Auth Required)",
                    extracted_at=datetime.now(UTC),
                    content_hash=None,
                    word_count=len((original_snippet or "").split()),
                    is_full_text=False,
                    title=title,
                )
                return self._write_to_cache(content)

            # Handle 404 Not Found explicitly
            if resp.status_code == 404:
                logger.info("HTTP 404 Not Found for %s", url)
                content = ExtractedContent(
                    url=url,
                    canonical_url=norm_canonical,
                    body_text=original_snippet,
                    extraction_status="failed",
                    status_reason="HTTP Error 404 (Not Found)",
                    extracted_at=datetime.now(UTC),
                    content_hash=None,
                    word_count=len((original_snippet or "").split()),
                    is_full_text=False,
                    title=title,
                )
                return self._write_to_cache(content)

            resp.raise_for_status()
            return self._process_html_response(
                html=resp.text,
                url=url,
                norm_canonical=norm_canonical,
                original_snippet=original_snippet,
                resp_headers=getattr(resp, "headers", {}) or {},
                existing_cached=cached,
                title=title,
                publisher=publisher,
            )

        except Exception as exc:
            logger.warning("Network or HTTP failure extracting %s: %s", url, exc)
            content = ExtractedContent(
                url=url,
                canonical_url=norm_canonical,
                body_text=original_snippet,
                extraction_status="failed",
                status_reason=f"Fetch failed: {type(exc).__name__}: {exc}",
                extracted_at=datetime.now(UTC),
                content_hash=None,
                word_count=len((original_snippet or "").split()),
                is_full_text=False,
                title=title,
            )
            return self._write_to_cache(content)

    def _process_html_response(
        self,
        html: str,
        url: str,
        norm_canonical: str,
        original_snippet: str | None,
        resp_headers: httpx.Headers | dict[str, str] | None = None,
        existing_cached: ExtractedContent | None = None,
        title: str | None = None,
        publisher: str | None = None,
    ) -> ExtractedContent:
        """Parse HTML, detect paywalls, extract body text, and record revisions."""
        now = datetime.now(UTC)
        etag = resp_headers.get("etag") if resp_headers else None
        last_modified = resp_headers.get("last-modified") if resp_headers else None

        # Paywall pattern detection
        if _PAYWALL_PATTERNS.search(html):
            logger.info("Paywall pattern detected in article HTML for %s", url)
            content = ExtractedContent(
                url=url,
                canonical_url=norm_canonical,
                body_text=original_snippet,
                extraction_status="paywall_detected",
                status_reason="Publisher paywall or subscription prompt detected in page",
                extracted_at=now,
                content_hash=None,
                word_count=len((original_snippet or "").split()),
                is_full_text=False,
                title=title,
                etag=etag,
                last_modified=last_modified,
                revalidated_at=now,
            )
            return self._write_to_cache(content)

        # Extract substantive article text using trafilatura
        extracted = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
        )

        if not extracted or len(extracted.strip()) < 150:
            logger.info("Extracted content empty or too short (<150 chars) for %s", url)
            content = ExtractedContent(
                url=url,
                canonical_url=norm_canonical,
                body_text=original_snippet,
                extraction_status="empty",
                status_reason="Extracted text was empty or lacked substantive article paragraphs",
                extracted_at=now,
                content_hash=None,
                word_count=len((original_snippet or "").split()),
                is_full_text=False,
                title=title,
                etag=etag,
                last_modified=last_modified,
                revalidated_at=now,
            )
            return self._write_to_cache(content)

        clean_text = extracted.strip()
        word_count = len(clean_text.split())
        content_hash = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()

        # Revision tracking: check if content changed compared to previous cache
        status: ExtractionStatus = "success"
        prev_hashes: tuple[str, ...] = ()
        if existing_cached is not None and existing_cached.content_hash:
            if existing_cached.content_hash != content_hash:
                status = "updated"
                prev_hashes = (
                    *existing_cached.previous_content_hashes,
                    existing_cached.content_hash,
                )
            else:
                prev_hashes = existing_cached.previous_content_hashes

        content = ExtractedContent(
            url=url,
            canonical_url=norm_canonical,
            body_text=clean_text,
            extraction_status=status,
            status_reason="Article updated or corrected by publisher"
            if status == "updated"
            else None,
            extracted_at=now,
            content_hash=content_hash,
            word_count=word_count,
            is_full_text=True,
            title=title,
            etag=etag,
            last_modified=last_modified,
            revalidated_at=now,
            previous_content_hashes=prev_hashes,
        )
        return self._write_to_cache(content)

    def enrich_article_record(
        self,
        article: ArticleRecord,
        force_revalidate: bool = False,
    ) -> ArticleRecord:
        """Enrich an ArticleRecord with extracted full-article body and revision provenance."""
        if article.is_full_text and article.body_text and not force_revalidate:
            history = self.get_revision_history(article.canonical_url)
            current_rev = history[-1] if history else None
            if current_rev:
                return article.model_copy(
                    update={
                        "article_id": current_rev.article_id,
                        "current_revision_id": current_rev.revision_id,
                        "revision_sequence": current_rev.revision_sequence,
                        "revisions": tuple(history),
                    }
                )
            return article

        extracted = self.extract_url(
            url=article.url,
            canonical_url=article.canonical_url,
            original_snippet=article.snippet,
            force_revalidate=force_revalidate,
            title=article.title,
            publisher=article.source_name,
        )

        history = self.get_revision_history(article.canonical_url)
        current_rev = extracted.revision or (history[-1] if history else None)

        return article.model_copy(
            update={
                "body_text": extracted.body_text,
                "extraction_status": extracted.extraction_status,
                "extracted_at": extracted.extracted_at,
                "is_full_text": extracted.is_full_text,
                "article_id": current_rev.article_id if current_rev else article.article_id,
                "current_revision_id": current_rev.revision_id
                if current_rev
                else article.current_revision_id,
                "revision_sequence": current_rev.revision_sequence
                if current_rev
                else article.revision_sequence,
                "revisions": tuple(history),
            }
        )
