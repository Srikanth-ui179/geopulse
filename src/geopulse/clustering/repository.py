"""Repository layer for persisting event clusters and article associations."""

import logging
from abc import ABC, abstractmethod
from datetime import UTC, date, datetime
from typing import Any

from geopulse.classification.models import TopicCategory
from geopulse.clustering.models import EventCluster
from geopulse.config import Settings
from geopulse.ingestion.models import ArticleRecord
from geopulse.summarization.models import reconstruct_event_summary

logger = logging.getLogger(__name__)

__all__ = [
    "EventRepository",
    "InMemoryEventRepository",
    "SupabaseEventRepository",
    "reconstruct_event_summary",
]


class EventRepository(ABC):
    """Abstract interface for event cluster persistence and querying."""

    @abstractmethod
    def persist_events(
        self,
        events: list[EventCluster] | tuple[EventCluster, ...],
        article_url_to_id: dict[str, str] | None = None,
    ) -> tuple[int, int]:
        """Persist event clusters and their article relationships.

        Returns:
            tuple[int, int]: (total_events_processed, events_persisted)
        """
        ...

    @abstractmethod
    def get_events_by_date(self, target_date: date) -> list[EventCluster]:
        """Retrieve event clusters recorded on a specific date."""
        ...

    @abstractmethod
    def get_event_article_ids(self, event_id: str) -> list[str]:
        """Retrieve associated database article UUIDs for an event."""
        ...

    @abstractmethod
    def update_event_summaries(
        self,
        results: list[Any] | tuple[Any, ...],
    ) -> int:
        """Persist structured summary and significance to existing event records.

        Returns:
            int: Number of event summaries updated.
        """
        ...


class InMemoryEventRepository(EventRepository):
    """In-memory event repository for offline testing and dry-run execution."""

    def __init__(self) -> None:
        self._events: dict[str, EventCluster] = {}
        self._event_articles: dict[str, list[str]] = {}
        self._summaries: dict[str, Any] = {}

    def persist_events(
        self,
        events: list[EventCluster] | tuple[EventCluster, ...],
        article_url_to_id: dict[str, str] | None = None,
    ) -> tuple[int, int]:
        count = 0
        for ev in events:
            self._events[ev.id] = ev
            count += 1
            if article_url_to_id:
                art_ids = [
                    article_url_to_id[a.canonical_url]
                    for a in ev.articles
                    if a.canonical_url in article_url_to_id
                ]
                self._event_articles[ev.id] = art_ids
        return len(events), count

    def get_events_by_date(self, target_date: date) -> list[EventCluster]:
        return [
            ev
            for ev in self._events.values()
            if ev.first_seen_at.date() == target_date or ev.last_seen_at.date() == target_date
        ]

    def get_event_article_ids(self, event_id: str) -> list[str]:
        return list(self._event_articles.get(event_id, []))

    def update_event_summaries(
        self,
        results: list[Any] | tuple[Any, ...],
    ) -> int:
        updated = 0
        for r in results:
            ev_id = getattr(r, "event_id", None) or (
                getattr(r.event, "id", None) if hasattr(r, "event") else None
            )
            if ev_id and ev_id in self._events:
                self._summaries[ev_id] = getattr(r, "summary", None) or r
                updated += 1
        return updated

    def clear(self) -> None:
        self._events.clear()
        self._event_articles.clear()
        self._summaries.clear()


class SupabaseEventRepository(EventRepository):
    """Supabase PostgreSQL persistence for event clusters and article associations."""

    def __init__(self, settings: Settings | None = None, client: Any = None):
        if client is not None:
            self.client = client
        else:
            from geopulse.config import get_settings

            cfg = settings or get_settings()
            url, key = cfg.require_supabase()
            from supabase import Client, create_client

            self.client: Client = create_client(url, key)

    def persist_events(
        self,
        events: list[EventCluster] | tuple[EventCluster, ...],
        article_url_to_id: dict[str, str] | None = None,
    ) -> tuple[int, int]:
        """Idempotently persist events and map articles to public.event_articles."""
        if not events:
            return 0, 0

        # 1. Resolve canonical_url to article UUIDs if not already provided
        url_to_id: dict[str, str] = dict(article_url_to_id or {})
        if not url_to_id:
            all_urls = list({a.canonical_url for e in events for a in e.articles})
            if all_urls:
                res = (
                    self.client.table("articles")
                    .select("id, canonical_url")
                    .in_("canonical_url", all_urls)
                    .execute()
                )
                for row in res.data or []:
                    if "canonical_url" in row and "id" in row:
                        url_to_id[str(row["canonical_url"])] = str(row["id"])

        # 2. Prepare event records
        event_payloads: list[dict[str, Any]] = [
            {
                "id": e.id,
                "title": e.title,
                "category": e.category.value if hasattr(e.category, "value") else str(e.category),
                "region": e.region,
                "first_seen_at": e.first_seen_at.isoformat(),
                "last_seen_at": e.last_seen_at.isoformat(),
                "published_at": e.published_at.isoformat() if e.published_at else None,
                "is_analysis": e.is_analysis,
                "confidence_label": e.confidence_label,
            }
            for e in events
        ]

        self.client.table("events").upsert(
            event_payloads,
            on_conflict="id",
        ).execute()

        # 3. Prepare event-article join records
        assoc_payloads: list[dict[str, Any]] = []
        for e in events:
            for a in e.articles:
                art_id = url_to_id.get(a.canonical_url)
                if art_id:
                    assoc_payloads.append(
                        {
                            "event_id": e.id,
                            "article_id": art_id,
                            "relationship_type": "reports_on",
                        }
                    )

        if assoc_payloads:
            self.client.table("event_articles").upsert(
                assoc_payloads,
                on_conflict="event_id,article_id",
                ignore_duplicates=True,
            ).execute()

        return len(events), len(events)

    def get_events_by_date(self, target_date: date) -> list[EventCluster]:
        """Fetch event clusters whose temporal boundary includes target_date."""
        start_iso = datetime.combine(target_date, datetime.min.time(), tzinfo=UTC).isoformat()
        end_iso = datetime.combine(target_date, datetime.max.time(), tzinfo=UTC).isoformat()

        res = (
            self.client.table("events")
            .select("*")
            .lte("first_seen_at", end_iso)
            .gte("last_seen_at", start_iso)
            .execute()
        )

        clusters: list[EventCluster] = []
        for row in res.data or []:
            ev_id = str(row["id"])
            # Fetch associated articles via event_articles join
            art_res = (
                self.client.table("event_articles")
                .select("article_id, articles(*, sources(name))")
                .eq("event_id", ev_id)
                .execute()
            )

            articles: list[ArticleRecord] = []
            for arow in art_res.data or []:
                art_data = arow.get("articles")
                if isinstance(art_data, dict):
                    source_name = "Unknown Source"
                    if isinstance(art_data.get("sources"), dict):
                        source_name = art_data["sources"].get("name", "Unknown Source")

                    pub_dt = (
                        datetime.fromisoformat(art_data["published_at"])
                        if art_data.get("published_at")
                        else None
                    )
                    fetch_dt = (
                        datetime.fromisoformat(art_data["fetched_at"])
                        if art_data.get("fetched_at")
                        else datetime.now(UTC)
                    )
                    articles.append(
                        ArticleRecord(
                            source_name=source_name,
                            title=art_data.get("title", ""),
                            url=art_data.get("url", ""),
                            canonical_url=art_data.get("canonical_url", ""),
                            published_at=pub_dt,
                            fetched_at=fetch_dt,
                            snippet=art_data.get("snippet"),
                            author=art_data.get("author"),
                            content_hash=(
                                art_data.get("content_hash") or f"hash-{art_data.get('id')}"
                            ),
                            language=art_data.get("language") or "en",
                            relevance_status=art_data.get("relevance_status", "pending"),
                        )
                    )

            first_seen = (
                datetime.fromisoformat(row["first_seen_at"])
                if row.get("first_seen_at")
                else datetime.now(UTC)
            )
            last_seen = (
                datetime.fromisoformat(row["last_seen_at"])
                if row.get("last_seen_at")
                else datetime.now(UTC)
            )
            pub_date = (
                datetime.fromisoformat(row["published_at"])
                if row.get("published_at")
                else None
            )

            cat_str = row.get("category", "uncertain")
            try:
                category = TopicCategory(cat_str)
            except ValueError:
                category = TopicCategory.UNCERTAIN

            clusters.append(
                EventCluster(
                    id=ev_id,
                    title=row["title"],
                    category=category,
                    region=row.get("region"),
                    first_seen_at=first_seen,
                    last_seen_at=last_seen,
                    published_at=pub_date,
                    is_analysis=bool(row.get("is_analysis", False)),
                    confidence_label=row.get("confidence_label", "uncertain"),
                    articles=tuple(articles),
                )
            )

        return clusters

    def get_event_article_ids(self, event_id: str) -> list[str]:
        """Fetch article UUIDs attached to an event."""
        res = (
            self.client.table("event_articles")
            .select("article_id")
            .eq("event_id", event_id)
            .execute()
        )
        return [str(row["article_id"]) for row in (res.data or []) if "article_id" in row]

    def update_event_summaries(
        self,
        results: list[Any] | tuple[Any, ...],
    ) -> int:
        """Persist structured summary text and metadata to public.events."""
        updated = 0
        for r in results:
            ev_id = getattr(r, "event_id", None) or (
                getattr(r.event, "id", None) if hasattr(r, "event") else None
            )
            if not ev_id:
                continue

            summary_text: str | None = None
            significance_text: str | None = None
            headline_text: str | None = None
            key_details_data: dict[str, Any] | None = None
            conf_qual: str | None = None
            model_used_text: str | None = None

            if hasattr(r, "summary") and r.summary is not None:
                if isinstance(r.summary, str):
                    summary_text = r.summary
                    significance_text = getattr(r, "significance", None)
                else:
                    s = r.summary
                    summary_text = getattr(s, "what_happened", str(s))
                    significance_text = getattr(
                        s,
                        "significance",
                        getattr(r, "significance", None),
                    )
                    headline_text = getattr(s, "headline", None)
                    if hasattr(s, "key_details"):
                        if hasattr(s.key_details, "model_dump"):
                            key_details_data = s.key_details.model_dump()
                        elif isinstance(s.key_details, dict):
                            key_details_data = dict(s.key_details)
                    if key_details_data and "actors" in key_details_data:
                        key_details_data["actors"] = list(key_details_data["actors"])
                    conf_qual = getattr(s, "confidence_qualification", None)
                    model_used_text = getattr(s, "model_used", None)
            elif hasattr(r, "headline"):
                headline_text = getattr(r, "headline", None)

            payload: dict[str, Any] = {}
            if summary_text:
                payload["summary"] = summary_text
            if significance_text:
                payload["significance"] = significance_text
            if headline_text:
                payload["headline"] = headline_text
            if key_details_data is not None:
                payload["key_details"] = key_details_data
            if conf_qual:
                payload["confidence_qualification"] = conf_qual
            if model_used_text:
                payload["model_used"] = model_used_text

            if payload:
                self.client.table("events").update(payload).eq("id", ev_id).execute()
                updated += 1

        return updated
