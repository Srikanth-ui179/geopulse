"""Idempotent persistence layer for sources and articles targeting Supabase PostgreSQL."""

import logging
import uuid
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any

from geopulse.config import Settings
from geopulse.ingestion.models import ArticleRecord, ArticleRevisionRecord, FeedSource

logger = logging.getLogger(__name__)


class ArticleRepository(ABC):
    """Abstract interface for source and article persistence."""

    @abstractmethod
    def get_or_create_source(self, source: FeedSource) -> str:
        """Fetch or create source by feed_url and return its UUID."""
        ...

    @abstractmethod
    def persist_articles(
        self,
        source_id: str,
        articles: list[ArticleRecord],
    ) -> tuple[int, int]:
        """Persist articles with deduplication. Returns (total, new_inserted)."""
        ...

    @abstractmethod
    def get_article_ids_by_urls(self, canonical_urls: list[str]) -> dict[str, str]:
        """Fetch database UUIDs mapped to canonical URLs."""
        ...

    @abstractmethod
    def get_articles_by_date_range(
        self,
        start_time: datetime,
        end_time: datetime,
        relevance_only: bool = True,
    ) -> list[ArticleRecord]:
        """Retrieve articles within a time window."""
        ...

    @abstractmethod
    def persist_article_revision(
        self,
        article_id: str,
        revision: ArticleRevisionRecord,
    ) -> bool:
        """Persist a single immutable revision for an article."""
        ...

    @abstractmethod
    def get_article_revisions(self, article_id: str) -> list[ArticleRevisionRecord]:
        """Fetch all historical revisions for an article, ordered by revision_sequence."""
        ...

    @abstractmethod
    def get_article_revision_by_id(self, revision_id: str) -> ArticleRevisionRecord | None:
        """Fetch a specific revision by its unique revision UUID."""
        ...


class InMemoryArticleRepository(ArticleRepository):
    """In-memory article repository for offline testing and dry-run execution."""

    def __init__(self) -> None:
        self._sources: dict[str, str] = {}
        self._articles: dict[str, ArticleRecord] = {}
        self._url_to_id: dict[str, str] = {}
        self._source_articles: dict[str, list[str]] = {}
        self._revisions: dict[str, list[ArticleRevisionRecord]] = {}
        self._revision_by_id: dict[str, ArticleRevisionRecord] = {}

    def get_or_create_source(self, source: FeedSource) -> str:
        if source.feed_url not in self._sources:
            src_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"geopulse:source:{source.feed_url}"))
            self._sources[source.feed_url] = src_id
        return self._sources[source.feed_url]

    def persist_articles(
        self,
        source_id: str,
        articles: list[ArticleRecord],
    ) -> tuple[int, int]:
        if not articles:
            return 0, 0
        new_count = 0
        for a in articles:
            if a.canonical_url not in self._articles:
                art_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"geopulse:article:{a.canonical_url}"))
                self._articles[a.canonical_url] = a
                self._url_to_id[a.canonical_url] = art_id
                self._source_articles.setdefault(source_id, []).append(art_id)
                new_count += 1
        return len(articles), new_count

    def get_article_ids_by_urls(self, canonical_urls: list[str]) -> dict[str, str]:
        return {url: self._url_to_id[url] for url in canonical_urls if url in self._url_to_id}

    def get_articles_by_date_range(
        self,
        start_time: datetime,
        end_time: datetime,
        relevance_only: bool = True,
    ) -> list[ArticleRecord]:
        results: list[ArticleRecord] = []
        for a in self._articles.values():
            if start_time <= a.fetched_at <= end_time:
                if relevance_only and a.relevance_status not in {"relevant", "pending"}:
                    continue
                results.append(a)
        return results

    def persist_article_revision(
        self,
        article_id: str,
        revision: ArticleRevisionRecord,
    ) -> bool:
        if revision.revision_id in self._revision_by_id:
            return False
        rev_list = self._revisions.setdefault(article_id, [])
        if any(r.revision_sequence == revision.revision_sequence for r in rev_list):
            return False
        rev_list.append(revision)
        self._revision_by_id[revision.revision_id] = revision
        return True

    def get_article_revisions(self, article_id: str) -> list[ArticleRevisionRecord]:
        revs = self._revisions.get(article_id, [])
        return sorted(revs, key=lambda r: r.revision_sequence)

    def get_article_revision_by_id(self, revision_id: str) -> ArticleRevisionRecord | None:
        return self._revision_by_id.get(revision_id)

    def clear(self) -> None:
        self._sources.clear()
        self._articles.clear()
        self._url_to_id.clear()
        self._source_articles.clear()
        self._revisions.clear()
        self._revision_by_id.clear()


class SupabaseArticleRepository(ArticleRepository):
    """Handles idempotent upserts of sources and articles into Supabase."""

    def __init__(self, settings: Settings | None = None, client: Any = None):
        if client is not None:
            self.client = client
        else:
            from geopulse.config import get_settings

            cfg = settings or get_settings()
            url, key = cfg.require_supabase()
            from supabase import Client, create_client

            self.client: Client = create_client(url, key)

    def get_or_create_source(self, source: FeedSource) -> str:
        """Fetch existing source id by feed_url or insert a new record idempotently."""
        query_res = (
            self.client.table("sources")
            .select("id")
            .eq("feed_url", source.feed_url)
            .limit(1)
            .execute()
        )

        if query_res.data:
            return str(query_res.data[0]["id"])

        insert_payload = {
            "name": source.name,
            "feed_url": source.feed_url,
            "enabled": source.enabled,
        }
        insert_res = self.client.table("sources").insert(insert_payload).execute()
        return str(insert_res.data[0]["id"])

    def persist_articles(
        self,
        source_id: str,
        articles: list[ArticleRecord],
    ) -> tuple[int, int]:
        """Persist articles with deduplication against existing canonical URLs.

        Returns:
            tuple[int, int]: (total_articles_processed, new_articles_inserted)
        """
        if not articles:
            return 0, 0

        canonical_urls = [a.canonical_url for a in articles]

        # Query existing canonical URLs to identify truly new articles
        existing_res = (
            self.client.table("articles")
            .select("canonical_url")
            .in_("canonical_url", canonical_urls)
            .execute()
        )
        existing_urls: set[str] = {
            row["canonical_url"] for row in (existing_res.data or []) if "canonical_url" in row
        }

        new_articles = [a for a in articles if a.canonical_url not in existing_urls]
        if not new_articles:
            return len(articles), 0

        payloads: list[dict[str, Any]] = []
        for a in new_articles:
            row: dict[str, Any] = {
                "source_id": source_id,
                "url": a.url,
                "canonical_url": a.canonical_url,
                "title": a.title,
                "author": a.author,
                "published_at": a.published_at.isoformat() if a.published_at else None,
                "fetched_at": a.fetched_at.isoformat(),
                "snippet": a.snippet,
                "content_hash": a.content_hash,
                "language": a.language,
                "relevance_status": a.relevance_status,
            }
            if getattr(a, "topic", None):
                row["topic"] = a.topic
            payloads.append(row)

        # Use ignore_duplicates=True to avoid race condition conflicts on unique(canonical_url)
        self.client.table("articles").upsert(
            payloads,
            on_conflict="canonical_url",
            ignore_duplicates=True,
        ).execute()

        return len(articles), len(new_articles)

    def get_article_ids_by_urls(self, canonical_urls: list[str]) -> dict[str, str]:
        """Fetch database UUIDs mapped to canonical URLs."""
        if not canonical_urls:
            return {}
        res = (
            self.client.table("articles")
            .select("id, canonical_url")
            .in_("canonical_url", canonical_urls)
            .execute()
        )
        return {
            str(row["canonical_url"]): str(row["id"])
            for row in (res.data or [])
            if "canonical_url" in row and "id" in row
        }

    def get_articles_by_date_range(
        self,
        start_time: datetime,
        end_time: datetime,
        relevance_only: bool = True,
    ) -> list[ArticleRecord]:
        """Retrieve articles within a time window, optionally filtering by relevance."""
        query = (
            self.client.table("articles")
            .select("*, sources(name)")
            .gte("fetched_at", start_time.isoformat())
            .lte("fetched_at", end_time.isoformat())
        )
        if relevance_only:
            query = query.in_("relevance_status", ["relevant", "pending"])
        res = query.execute()

        articles: list[ArticleRecord] = []
        for row in res.data or []:
            source_name = "Unknown Source"
            if isinstance(row.get("sources"), dict) and "name" in row["sources"]:
                source_name = row["sources"]["name"]
            pub_dt = (
                datetime.fromisoformat(row["published_at"]) if row.get("published_at") else None
            )
            fetch_dt = (
                datetime.fromisoformat(row["fetched_at"])
                if row.get("fetched_at")
                else datetime.now(UTC)
            )
            articles.append(
                ArticleRecord(
                    source_name=source_name,
                    title=row["title"],
                    url=row["url"],
                    canonical_url=row["canonical_url"],
                    published_at=pub_dt,
                    fetched_at=fetch_dt,
                    snippet=row.get("snippet"),
                    author=row.get("author"),
                    content_hash=row.get("content_hash") or f"hash-{row['canonical_url']}",
                    language=row.get("language") or "en",
                    relevance_status=row.get("relevance_status", "pending"),
                )
            )
        return articles

    def persist_article_revision(
        self,
        article_id: str,
        revision: ArticleRevisionRecord,
    ) -> bool:
        """Persist an article revision to Supabase."""
        payload = {
            "id": revision.revision_id,
            "article_id": article_id,
            "canonical_url": revision.canonical_url,
            "content_hash": revision.content_hash,
            "title": revision.title,
            "publisher": revision.publisher,
            "published_at": revision.published_at.isoformat() if revision.published_at else None,
            "first_seen_at": revision.first_seen_at.isoformat(),
            "retrieved_at": revision.retrieved_at.isoformat(),
            "last_modified": revision.last_modified,
            "etag": revision.etag,
            "body_text": revision.body_text,
            "extraction_status": revision.extraction_status,
            "extraction_method": revision.extraction_method,
            "revision_sequence": revision.revision_sequence,
            "previous_revision_id": revision.previous_revision_id,
            "revision_classification": revision.revision_classification,
            "relevant_metadata": revision.relevant_metadata,
        }
        try:
            self.client.table("article_revisions").insert(payload).execute()
            self.client.table("articles").update(
                {
                    "current_revision_id": revision.revision_id,
                    "revision_sequence": revision.revision_sequence,
                }
            ).eq("id", article_id).execute()
            return True
        except Exception as exc:
            logger.warning("Failed to persist revision %s: %s", revision.revision_id, exc)
            return False

    def get_article_revisions(self, article_id: str) -> list[ArticleRevisionRecord]:
        """Fetch all historical revisions for an article, ordered by sequence."""
        try:
            res = (
                self.client.table("article_revisions")
                .select("*")
                .eq("article_id", article_id)
                .order("revision_sequence")
                .execute()
            )
            revisions: list[ArticleRevisionRecord] = []
            for row in res.data or []:
                revisions.append(
                    ArticleRevisionRecord(
                        revision_id=str(row["id"]),
                        article_id=str(row["article_id"]),
                        canonical_url=row["canonical_url"],
                        content_hash=row["content_hash"],
                        title=row["title"],
                        publisher=row["publisher"],
                        published_at=(
                            datetime.fromisoformat(row["published_at"])
                            if row.get("published_at")
                            else None
                        ),
                        first_seen_at=(
                            datetime.fromisoformat(row["first_seen_at"])
                            if row.get("first_seen_at")
                            else datetime.now(UTC)
                        ),
                        retrieved_at=(
                            datetime.fromisoformat(row["retrieved_at"])
                            if row.get("retrieved_at")
                            else datetime.now(UTC)
                        ),
                        last_modified=row.get("last_modified"),
                        etag=row.get("etag"),
                        body_text=row.get("body_text"),
                        extraction_status=row.get("extraction_status", "success"),
                        extraction_method=row.get("extraction_method", "trafilatura"),
                        revision_sequence=row.get("revision_sequence", 1),
                        previous_revision_id=(
                            str(row["previous_revision_id"])
                            if row.get("previous_revision_id")
                            else None
                        ),
                        revision_classification=row.get(
                            "revision_classification", "initial_publication"
                        ),
                        relevant_metadata=row.get("relevant_metadata") or {},
                    )
                )
            return revisions
        except Exception as exc:
            logger.warning("Failed to fetch revisions for article %s: %s", article_id, exc)
            return []

    def get_article_revision_by_id(self, revision_id: str) -> ArticleRevisionRecord | None:
        """Fetch a specific revision by its unique UUID."""
        try:
            res = (
                self.client.table("article_revisions")
                .select("*")
                .eq("id", revision_id)
                .limit(1)
                .execute()
            )
            if not res.data:
                return None
            row = res.data[0]
            return ArticleRevisionRecord(
                revision_id=str(row["id"]),
                article_id=str(row["article_id"]),
                canonical_url=row["canonical_url"],
                content_hash=row["content_hash"],
                title=row["title"],
                publisher=row["publisher"],
                published_at=(
                    datetime.fromisoformat(row["published_at"]) if row.get("published_at") else None
                ),
                first_seen_at=(
                    datetime.fromisoformat(row["first_seen_at"])
                    if row.get("first_seen_at")
                    else datetime.now(UTC)
                ),
                retrieved_at=(
                    datetime.fromisoformat(row["retrieved_at"])
                    if row.get("retrieved_at")
                    else datetime.now(UTC)
                ),
                last_modified=row.get("last_modified"),
                etag=row.get("etag"),
                body_text=row.get("body_text"),
                extraction_status=row.get("extraction_status", "success"),
                extraction_method=row.get("extraction_method", "trafilatura"),
                revision_sequence=row.get("revision_sequence", 1),
                previous_revision_id=(
                    str(row["previous_revision_id"]) if row.get("previous_revision_id") else None
                ),
                revision_classification=row.get("revision_classification", "initial_publication"),
                relevant_metadata=row.get("relevant_metadata") or {},
            )
        except Exception as exc:
            logger.warning("Failed to fetch revision %s: %s", revision_id, exc)
            return None
