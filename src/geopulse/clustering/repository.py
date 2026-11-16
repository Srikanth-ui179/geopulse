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
        url_to_id = dict(article_url_to_id or {})
        for ev in events:
            if url_to_id:
                art_ids = [
                    url_to_id[a.canonical_url] for a in ev.articles if a.canonical_url in url_to_id
                ]
                if not art_ids:
                    # Unmatched article IDs cause event to be skipped
                    continue
                self._events[ev.id] = ev
                self._event_articles[ev.id] = art_ids
            else:
                self._events[ev.id] = ev
            count += 1
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
        """Idempotently persist events and map articles to public.event_articles.

        Ensures every persisted event has at least one resolvable article ID.
        Events with zero resolvable articles are skipped before writing to prevent orphans.
        """
        if not events:
            return 0, 0

        # 1. Resolve canonical_url to article UUIDs if not already provided
        url_to_id: dict[str, str] = dict(article_url_to_id or {})
        missing_urls = [
            a.canonical_url for e in events for a in e.articles if a.canonical_url not in url_to_id
        ]
        if missing_urls:
            unique_missing = list(set(missing_urls))
            res = (
                self.client.table("articles")
                .select("id, canonical_url")
                .in_("canonical_url", unique_missing)
                .execute()
            )
            for row in res.data or []:
                if "canonical_url" in row and "id" in row:
                    url_to_id[str(row["canonical_url"])] = str(row["id"])

        # 2. Filter events to only those with at least one resolvable article ID
        valid_events: list[EventCluster] = []
        valid_event_articles: list[tuple[EventCluster, list[str]]] = []
        for e in events:
            art_ids = [
                url_to_id[a.canonical_url] for a in e.articles if a.canonical_url in url_to_id
            ]
            if not art_ids:
                logger.warning(
                    "Skipping event '%s' (%s): none of its %d articles have resolved database IDs.",
                    e.title,
                    e.id,
                    len(e.articles),
                )
                continue
            valid_events.append(e)
            valid_event_articles.append((e, art_ids))

        if not valid_events:
            logger.warning("No events had resolvable article IDs; skipping event persistence.")
            return len(events), 0

        # 3. Deduplicate events; merge identical event associations; reject conflicts
        deduped_events_map: dict[str, EventCluster] = {}
        event_art_ids_map: dict[str, list[str]] = {}

        for e, art_ids in valid_event_articles:
            if e.id in deduped_events_map:
                existing = deduped_events_map[e.id]
                # Explicitly reject duplicate event IDs with conflicting metadata in the same batch
                if existing.title != e.title or existing.category != e.category:
                    raise ValueError(
                        f"Conflicting duplicate event ID '{e.id}' in batch: "
                        f"title '{existing.title}' vs '{e.title}', "
                        f"category '{existing.category}' vs '{e.category}'."
                    )
                # Deterministically merge article associations for identical event records
                current_art_ids = event_art_ids_map[e.id]
                for aid in art_ids:
                    if aid not in current_art_ids:
                        current_art_ids.append(aid)
            else:
                deduped_events_map[e.id] = e
                event_art_ids_map[e.id] = list(art_ids)

        deduped_events: list[EventCluster] = list(deduped_events_map.values())
        deduped_event_articles: list[tuple[EventCluster, list[str]]] = [
            (e, event_art_ids_map[e.id]) for e in deduped_events
        ]

        # 4. Prepare event records for valid events
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
            for e in deduped_events
        ]

        # 5. Prepare event-article join records
        assoc_payloads: list[dict[str, Any]] = []
        seen_assocs: set[tuple[str, str]] = set()
        for e, art_ids in deduped_event_articles:
            for art_id in art_ids:
                pair = (e.id, art_id)
                if pair not in seen_assocs:
                    seen_assocs.add(pair)
                    assoc_payloads.append(
                        {
                            "event_id": e.id,
                            "article_id": art_id,
                            "relationship_type": "reports_on",
                        }
                    )

        # 6. Atomic persistence via PostgreSQL stored procedure
        # Automatic fallback to sequential table upserts is removed to prevent orphans.
        if not hasattr(self.client, "rpc"):
            raise RuntimeError(
                "Atomic event persistence is unavailable: Supabase client does not support RPC. "
                "Migration 006_atomic_event_persistence.sql must be reviewed and applied first."
            )

        try:
            rpc_res = self.client.rpc(
                "upsert_events_with_articles",
                {
                    "p_events": event_payloads,
                    "p_event_articles": assoc_payloads,
                },
            ).execute()
            logger.debug(
                "Atomically persisted %d event(s) and associations via RPC: %s",
                len(deduped_events),
                getattr(rpc_res, "data", rpc_res),
            )
            return len(events), len(deduped_events)
        except Exception as rpc_exc:
            err_str = str(rpc_exc)
            is_missing_rpc = (
                "PGRST202" in err_str
                or "Could not find the function" in err_str
                or (
                    "upsert_events_with_articles" in err_str
                    and ("could not find" in err_str.lower() or "does not exist" in err_str.lower())
                )
            )
            if is_missing_rpc:
                logger.error(
                    "Atomic event persistence procedure 'upsert_events_with_articles' is not "
                    "available in the database. Sequential fallback writes are disabled to prevent "
                    "data corruption or orphaned events. Migration "
                    "006_atomic_event_persistence.sql must be reviewed and applied first."
                )
                raise RuntimeError(
                    "Atomic event persistence is unavailable: stored procedure "
                    "'upsert_events_with_articles' not found in database "
                    "(migration 006 not applied). Review and apply "
                    "migrations/006_atomic_event_persistence.sql before persisting events. "
                    "Zero event or association records were written."
                ) from rpc_exc

            is_timeout_or_connection = any(
                term in err_str.lower()
                for term in (
                    "timeout",
                    "timed out",
                    "504",
                    "connection lost",
                    "connection refused",
                    "connection reset",
                )
            )
            if is_timeout_or_connection:
                logger.error(
                    "Atomic event persistence RPC timed out or lost connection for %d event(s): %s",
                    len(deduped_events),
                    rpc_exc,
                )
                raise RuntimeError(
                    f"Atomic event persistence RPC timed out or connection was lost: {rpc_exc}. "
                    "The database commit status is unknown (the transaction may have committed or "
                    "rolled back). Do NOT perform manual partial writes. Re-running the pipeline "
                    "command is safe because upsert_events_with_articles uses deterministic event "
                    "IDs and idempotent upsert semantics."
                ) from rpc_exc

            logger.error(
                "Atomic event persistence RPC failed for %d event(s): %s",
                len(deduped_events),
                rpc_exc,
            )
            raise RuntimeError(
                f"Failed to persist events atomically via RPC: {rpc_exc}"
            ) from rpc_exc

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
                datetime.fromisoformat(row["published_at"]) if row.get("published_at") else None
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
                    corroboration_status=row.get("corroboration_status"),
                    independent_chains_count=(
                        int(row["independent_chains_count"])
                        if row.get("independent_chains_count") is not None
                        else 0
                    ),
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
                    if getattr(s, "claims", None):
                        key_details_data = key_details_data or {}
                        key_details_data["_evidence"] = {
                            "claims": [claim.model_dump(mode="json") for claim in s.claims],
                            "sources": [source.model_dump(mode="json") for source in s.sources],
                        }
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
            if hasattr(r, "summary") and r.summary is not None and not isinstance(r.summary, str):
                if getattr(r.summary, "corroboration_status", None):
                    payload["corroboration_status"] = r.summary.corroboration_status
                if getattr(r.summary, "independent_chains_count", None) is not None:
                    payload["independent_chains_count"] = r.summary.independent_chains_count

            if payload:
                self.client.table("events").update(payload).eq("id", ev_id).execute()
                updated += 1

        return updated
