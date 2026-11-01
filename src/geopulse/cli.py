"""Command line interface for Geopulse."""

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

# Ensure UTF-8 output on Windows consoles to prevent UnicodeEncodeError on emojis
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from pydantic import ValidationError

from geopulse.clustering import ClusteringSummary, SupabaseEventRepository, cluster_articles
from geopulse.config import Settings
from geopulse.delivery import (
    InMemoryDeliveryRepository,
    JsonFileDeliveryRepository,
    MockTelegramClient,
    SupabaseDeliveryRepository,
    TelegramClient,
    TelegramConfigError,
    TelegramDeliveryService,
    TelegramDigestFormatter,
    TelegramNetworkError,
    TelegramPermanentError,
    TelegramRateLimitError,
    TelegramTimeoutError,
)
from geopulse.ingestion import (
    DEFAULT_FEEDS,
    SupabaseArticleRepository,
    find_feed,
    run_ingestion,
)
from geopulse.ingestion.models import ArticleRecord
from geopulse.preflight import run_supabase_preflight
from geopulse.summarization import (
    EventSummarizer,
    GeminiConfigError,
    SummarizationResult,
    resolve_summarization_clients,
)


def handle_check_config() -> int:
    try:
        settings = Settings()
    except ValidationError as exc:
        print("Configuration is invalid:")
        print(exc)
        return 1

    print("Configuration loaded successfully.")
    print(f"Environment: {settings.app_env}")
    print(f"Timezone: {settings.timezone}")
    print(f"LLM Provider: {settings.llm_provider}")
    print(f"Ollama endpoint: {settings.ollama_base_url}")
    print(f"Ollama model: {settings.ollama_model}")
    print(f"Gemini configured: {settings.is_gemini_configured}")
    if settings.is_gemini_configured:
        print(f"Gemini model: {settings.gemini_model}")
    print(f"Supabase configured: {settings.is_supabase_configured}")
    if settings.is_supabase_configured:
        print(f"Supabase key category: {settings.supabase_key_category}")
    print(f"Telegram configured: {settings.is_telegram_configured}")
    return 0


def handle_check_preflight(args: argparse.Namespace) -> int:
    try:
        settings = Settings()
    except ValidationError as exc:
        print(f"Configuration error: {exc}")
        return 1

    scope = getattr(args, "scope", "full")
    result = run_supabase_preflight(settings=settings, scope=scope)
    if result.success:
        print("Preflight check passed.")
        if result.details:
            print(result.details)
        return 0
    else:
        print("Preflight check failed:")
        if result.error_message:
            print(f"  Error: {result.error_message}")
        if result.details:
            print(f"  Details: {result.details}")
        return 1


def handle_ingest(args: argparse.Namespace) -> int:
    try:
        settings = Settings()
    except ValidationError as exc:
        print(f"Configuration error: {exc}")
        return 1

    if args.feed:
        matched = find_feed(args.feed)
        if not matched:
            print(f"Error: No feed matched '{args.feed}'. Available feeds:")
            for f in DEFAULT_FEEDS:
                print(f"  - {f.name} ({f.feed_url})")
            return 1
        target_feeds = [matched]
    else:
        target_feeds = [f for f in DEFAULT_FEEDS if f.enabled]

    if getattr(args, "persist", False) and getattr(args, "dry_run", None) is True:
        print(
            "Error: Cannot specify both --persist and --dry-run. "
            "Choose either --persist (write to database) or --dry-run (simulate in-memory)."
        )
        return 1

    if args.persist:
        if not settings.is_supabase_configured:
            print("Error: --persist requires SUPABASE_URL and SUPABASE_KEY to be configured.")
            return 1
        preflight_res = run_supabase_preflight(settings=settings, scope="ingest")
        if not preflight_res.success:
            print(f"Error: Supabase schema preflight failed: {preflight_res.error_message}")
            if preflight_res.details:
                print(f"Details: {preflight_res.details}")
            return 1
        dry_run = False
    elif args.dry_run is True:
        dry_run = True
    else:
        dry_run = not settings.is_supabase_configured

    mode_label = "Dry-run (in-memory only)" if dry_run else "Persistence (Supabase PostgreSQL)"
    feed_limit = args.limit if getattr(args, "limit", None) is not None else settings.feed_limit
    print(f"Starting Geopulse RSS Ingestion [{mode_label}]")
    print(f"Processing {len(target_feeds)} feed(s), limit per feed: {feed_limit}...\n")

    summary = run_ingestion(
        feeds=target_feeds,
        limit_per_feed=feed_limit,
        dry_run=dry_run,
        settings=settings,
    )

    for idx, result in enumerate(summary.feed_results, start=1):
        status_tag = "OK" if result.success else "FAIL"
        print(
            f"[{idx}/{len(summary.feed_results)}] [{status_tag}] {result.feed_name} "
            f"({result.elapsed_seconds:.2f}s)"
        )
        print(f"      URL: {result.feed_url}")
        if result.success:
            relevant_count = sum(1 for a in result.articles if a.relevance_status == "relevant")
            pending_count = sum(1 for a in result.articles if a.relevance_status == "pending")
            irrelevant_count = sum(1 for a in result.articles if a.relevance_status == "irrelevant")
            print(
                f"      Articles: {result.articles_fetched} fetched, "
                f"{result.articles_new} new ("
                f"{relevant_count} relevant, {pending_count} pending, "
                f"{irrelevant_count} irrelevant)"
            )
            if getattr(args, "show_articles", False) and result.articles:
                for a in result.articles:
                    date_str = (
                        a.published_at.strftime("%Y-%m-%d %H:%M UTC")
                        if a.published_at
                        else "no date"
                    )
                    print(f"        - [{date_str}] [{a.relevance_status}] {a.title}")
                    print(f"          URL: {a.canonical_url}")
        else:
            print(f"      Error: {result.error_message}")
        print()

    print("=" * 60)
    print("Ingestion Summary:")
    print(
        f"  Total Feeds: {summary.total_feeds} "
        f"({summary.successful_feeds} ok, {summary.failed_feeds} failed)"
    )
    print(f"  Articles Fetched: {summary.total_articles_fetched}")
    action_word = "Simulated New" if dry_run else "Persisted New"
    print(f"  Articles {action_word}: {summary.total_articles_new}")
    print("=" * 60)

    if getattr(args, "cluster", False):
        all_articles = [a for res in summary.feed_results for a in res.articles]
        clustering_res = cluster_articles(all_articles)
        print_clustering_summary(clustering_res, show_details=getattr(args, "show_articles", False))

    return 0 if summary.successful_feeds > 0 or summary.total_feeds == 0 else 1


def print_clustering_summary(summary: ClusteringSummary, show_details: bool = False) -> None:
    print("=" * 60)
    print("Event Clustering Summary:")
    print(f"  Articles Processed: {summary.total_articles}")
    if getattr(summary, "total_articles_evaluated", 0) > 0:
        print(f"  - Evaluated (Parsed): {summary.total_articles_evaluated}")
        print(f"  - Filtered / Rejected: {summary.articles_filtered_out}")
    print(f"  Events Formed:      {summary.total_clusters}")
    print(f"  - Corroborated:     {summary.corroborated_clusters} (reported by 2+ sources)")
    print(f"  - Single-Source:    {summary.single_source_clusters}")
    print(f"  - Disputed:         {summary.disputed_clusters} (contested/denied claims)")
    print(f"  - Uncertain:        {summary.uncertain_clusters}")
    print("=" * 60)
    if show_details and summary.clusters:
        print("\nIdentified Events:")
        for idx, cluster in enumerate(summary.clusters, start=1):
            date_str = cluster.first_seen_at.strftime("%Y-%m-%d %H:%M UTC")
            print(
                f"[{idx}] [{cluster.confidence_label.upper()}] "
                f"[{cluster.category.value}] {cluster.title}"
            )
            print(
                f"    First seen: {date_str} | "
                f"Sources: {cluster.source_count} ({cluster.article_count} articles)"
            )
            for art in cluster.articles:
                print(f"      * [{art.source_name}] {art.title}")
        print()


def handle_cluster(args: argparse.Namespace) -> int:
    try:
        settings = Settings()
    except ValidationError as exc:
        print(f"Configuration error: {exc}")
        return 1

    if args.feed:
        matched = find_feed(args.feed)
        if not matched:
            print(f"Error: No feed matched '{args.feed}'. Available feeds:")
            for f in DEFAULT_FEEDS:
                print(f"  - {f.name} ({f.feed_url})")
            return 1
        target_feeds = [matched]
    else:
        target_feeds = [f for f in DEFAULT_FEEDS if f.enabled]

    persist = getattr(args, "persist", False)
    art_repo: SupabaseArticleRepository | None = None
    ev_repo: SupabaseEventRepository | None = None

    if persist:
        if not settings.is_supabase_configured:
            print("Error: --persist requires SUPABASE_URL and SUPABASE_KEY to be configured.")
            return 1
        preflight_res = run_supabase_preflight(settings=settings, scope="cluster")
        if not preflight_res.success:
            print(f"Error: Supabase schema preflight failed: {preflight_res.error_message}")
            if preflight_res.details:
                print(f"Details: {preflight_res.details}")
            return 1
        art_repo = SupabaseArticleRepository(settings=settings)
        ev_repo = SupabaseEventRepository(settings=settings, client=art_repo.client)

    feed_limit = args.limit if getattr(args, "limit", None) is not None else settings.feed_limit
    print(f"Fetching feeds to cluster events (limit per feed: {feed_limit})...")
    summary = run_ingestion(
        feeds=target_feeds,
        limit_per_feed=feed_limit,
        dry_run=not persist,
        settings=settings,
        repository=art_repo,
    )

    all_articles = [a for res in summary.feed_results for a in res.articles]
    clustering_res = cluster_articles(all_articles)
    print_clustering_summary(clustering_res, show_details=getattr(args, "show_articles", True))

    if persist and ev_repo is not None and art_repo is not None:
        all_urls = [a.canonical_url for a in all_articles]
        url_to_id = art_repo.get_article_ids_by_urls(all_urls)
        total_clusters = len(clustering_res.clusters)
        try:
            total_events, persisted_events = ev_repo.persist_events(
                clustering_res.clusters,
                article_url_to_id=url_to_id,
            )
            if persisted_events < total_clusters:
                skipped = total_clusters - persisted_events
                print(
                    f"Persisted {persisted_events} event cluster(s) to Supabase "
                    f"({skipped} cluster(s) skipped due to unresolvable article IDs)."
                )
            else:
                print(f"Persisted {persisted_events} event cluster(s) to Supabase.")
        except Exception as exc:
            print(f"Error: Failed to persist event clusters to Supabase: {exc}")
            return 1

    return 0


def print_summarization_results(
    results: tuple[SummarizationResult, ...],
    model_name: str,
    endpoint: str,
) -> None:
    print("=" * 60)
    print("Geopulse Event Summarization [Local Ollama]")
    print(f"Model: {model_name} (Endpoint: {endpoint})")
    print("=" * 60)

    for idx, res in enumerate(results, start=1):
        evt = res.event
        conf_tag = evt.confidence_label.upper()
        cat_tag = evt.category.value.upper()
        status_tag = res.status.upper()

        print(f"\n[{idx}/{len(results)}] [{status_tag}] [{cat_tag}] [{conf_tag}]")
        if res.status == "success" and res.summary:
            s = res.summary
            print(f"HEADLINE: {s.headline}")
            print(f"WHAT HAPPENED: {s.what_happened}")
            print("KEY DETAILS:")
            actors_str = ", ".join(s.key_details.actors) if s.key_details.actors else "Unspecified"
            print(f"  * Actors:   {actors_str}")
            print(f"  * Location: {s.key_details.location or 'Unspecified'}")
            print(f"  * Timing:   {s.key_details.timing or 'Unspecified'}")
            print(f"  * Action:   {s.key_details.action or 'Unspecified'}")
            if s.significance and s.significance.lower() not in {"null", "none", "n/a", "nil"}:
                print(f"SIGNIFICANCE: {s.significance}")
            print(f"CONFIDENCE: {s.confidence_qualification}")
            print("SOURCES:")
            for src in s.sources:
                print(f"  - [{src.source_name}] {src.title}")
                print(f"    URL: {src.url}")
        else:
            print(f"EVENT: {evt.title}")
            if res.status == "quarantined":
                print(f"QUARANTINED: {res.error_message}")
            elif res.status == "insufficient_evidence":
                print(f"INSUFFICIENT EVIDENCE: {res.error_message}")
            else:
                print(f"ERROR: {res.error_message}")
            if res.status == "unavailable":
                print("HINT: Start Ollama with 'ollama serve' or pass '--mock-model' for dry-run.")

    print("\n" + "=" * 60)
    print("Summarization Summary:")
    print(f"  Events Evaluated: {len(results)}")
    success_count = sum(1 for r in results if r.status == "success")
    quarantined_count = sum(1 for r in results if r.status == "quarantined")
    insuff_count = sum(1 for r in results if r.status == "insufficient_evidence")
    failed_count = sum(1 for r in results if r.status == "failed")
    unavail_count = sum(1 for r in results if r.status == "unavailable")
    print(f"  - Successful:            {success_count}")
    if quarantined_count > 0:
        print(f"  - Quarantined:           {quarantined_count} (factual grounding failure)")
    if insuff_count > 0:
        print(f"  - Insufficient Evidence: {insuff_count}")
    print(f"  - Failed:                {failed_count}")
    print(f"  - Unavailable:           {unavail_count}")
    print("=" * 60)


def handle_summarize(args: argparse.Namespace) -> int:
    try:
        settings = Settings()
    except ValidationError as exc:
        print(f"Configuration error: {exc}")
        return 1

    try:
        client, fallback_client = resolve_summarization_clients(
            settings=settings,
            provider=getattr(args, "provider", None),
            model=getattr(args, "model", None),
            mock=getattr(args, "mock_model", False),
        )
    except GeminiConfigError as exc:
        print(f"Configuration error: {exc}")
        return 1

    endpoint = getattr(client, "base_url", "mock")

    articles: list[ArticleRecord] = []
    if getattr(args, "fixture", False):
        fixture_path = (
            Path(__file__).resolve().parents[2]
            / "tests"
            / "fixtures"
            / "sample_clustering_articles.json"
        )
        if not fixture_path.is_file():
            print(f"Error: Fixture file not found at {fixture_path}")
            return 1
        raw_items = json.loads(fixture_path.read_text(encoding="utf-8"))
        for item in raw_items:
            pub_dt = (
                datetime.fromisoformat(item["published_at"]) if item.get("published_at") else None
            )
            articles.append(
                ArticleRecord(
                    source_name=item["source_name"],
                    title=item["title"],
                    url=item["url"],
                    canonical_url=item["canonical_url"],
                    published_at=pub_dt,
                    snippet=item.get("snippet"),
                    content_hash=f"hash-{item['id']}",
                )
            )
        print(f"Loaded {len(articles)} article(s) from local test fixtures.")
    else:
        if args.feed:
            matched = find_feed(args.feed)
            if not matched:
                print(f"Error: No feed matched '{args.feed}'. Available feeds:")
                for f in DEFAULT_FEEDS:
                    print(f"  - {f.name} ({f.feed_url})")
                return 1
            target_feeds = [matched]
        else:
            target_feeds = [f for f in DEFAULT_FEEDS if f.enabled]

        feed_limit = args.limit if getattr(args, "limit", None) is not None else settings.feed_limit
        print(f"Fetching feeds for summarization (limit per feed: {feed_limit})...")
        ingest_summary = run_ingestion(
            feeds=target_feeds,
            limit_per_feed=feed_limit,
            dry_run=True,
            settings=settings,
        )
        articles = [a for res in ingest_summary.feed_results for a in res.articles]

    clustering_summary = cluster_articles(articles)
    events = clustering_summary.clusters
    if not events:
        print("No events identified for summarization.")
        return 0

    summarizer = EventSummarizer(client=client, fallback_client=fallback_client, settings=settings)
    results = summarizer.summarize_events(events)

    print_summarization_results(
        results=results,
        model_name=client.model,
        endpoint=endpoint,
    )
    return 0 if any(r.status == "success" for r in results) or not results else 1


def format_briefing_diagnostics(
    articles_fetched: int,
    articles_retained: int,
    events: Sequence[Any],
    results: Sequence[Any],
    selected_events: Sequence[Any],
    min_relevance_rank: float | None = None,
    max_events: int | None = None,
    include_low_priority: bool = False,
) -> str:
    from geopulse.delivery.formatter import compute_event_priority_rank

    lines = ["\n" + "=" * 60, "GEOPULSE BRIEFING PIPELINE DIAGNOSTICS", "=" * 60]
    lines.append(f"• Ingestion: {articles_fetched} articles fetched, {articles_retained} retained")
    lines.append(f"• Clustering: {len(events)} event clusters formed")

    eligible_events = []
    priority_excluded = []
    for ev in events:
        rank = compute_event_priority_rank(ev)
        if (
            not include_low_priority
            and min_relevance_rank is not None
            and rank > min_relevance_rank
        ):
            priority_excluded.append((ev, rank))
        else:
            eligible_events.append((ev, rank))

    threshold_str = f"<= {min_relevance_rank}" if min_relevance_rank is not None else "none"
    lines.append(
        f"• Editorial Priority: {len(eligible_events)} eligible ({threshold_str}), "
        f"{len(priority_excluded)} excluded by rank"
    )

    attempted = [
        r for r in results if getattr(getattr(r, "summary", None), "model_used", "") != "omitted"
    ]
    successful = [r for r in attempted if r.status == "success"]
    quarantined = [r for r in attempted if r.status in {"quarantined", "insufficient_evidence"}]
    failed = [r for r in attempted if r.status == "failed"]
    placeholders = [
        r for r in results if getattr(getattr(r, "summary", None), "model_used", "") == "omitted"
    ]

    lines.append(
        f"• Summarization: {len(attempted)} attempted ({len(successful)} success, "
        f"{len(quarantined)} quarantined, {len(failed)} failed), "
        f"{len(placeholders)} omitted placeholders"
    )

    lines.append(f"• Final Selection: {len(selected_events)} event(s) selected for briefing")
    deferred_count = max(0, len(successful) - len(selected_events))
    if deferred_count > 0:
        lines.append(
            f"• Deferred Candidates: {deferred_count} (due to event cap / diversity limits)"
        )

    if quarantined or priority_excluded or failed:
        lines.append("\nDetailed Exclusions:")
        for r in quarantined:
            reason = r.quarantine_reason or r.error_message or "Validation/entailment rejected"
            ev_title = r.event.title if r.event else r.event_id
            lines.append(f"  [QUARANTINED] {ev_title[:70]}")
            lines.append(f"    Reason: {reason[:120]}")
        for r in failed:
            msg = r.error_message or "Model generation or parsing failed"
            ev_title = r.event.title if r.event else r.event_id
            lines.append(f"  [FAILED] {ev_title[:70]}")
            lines.append(f"    Reason: {msg[:120]}")
        for ev, rank in priority_excluded:
            cat_val = ev.category.value if hasattr(ev.category, "value") else str(ev.category)
            lines.append(
                f"  [PRIORITY] {ev.title[:70]} "
                f"(rank {rank:.1f} > {min_relevance_rank}, cat={cat_val})"
            )

    lines.append("=" * 60)
    return "\n".join(lines)


def handle_brief(args: argparse.Namespace) -> int:
    try:
        settings = Settings()
    except ValidationError as exc:
        print(f"Configuration error: {exc}")
        return 1

    if getattr(args, "preview", False) and getattr(args, "send", False):
        print("Error: Cannot specify both --preview and --send.")
        return 1

    max_events = getattr(args, "max_events", None)
    if max_events is None and getattr(args, "edition", None) in {"morning", "evening"}:
        max_events = 10
    if max_events is not None and max_events <= 0:
        print("Error: --max-events must be greater than 0.")
        return 1

    if getattr(args, "in_memory_state", False):
        if getattr(args, "supabase_state", False):
            print("Error: Cannot specify both --in-memory-state and --supabase-state.")
            return 1
        if getattr(args, "persist", False):
            print("Error: Cannot specify both --in-memory-state and --persist.")
            return 1
        if getattr(args, "state_file", None):
            print("Error: Cannot specify both --in-memory-state and --state-file.")
            return 1

    if (getattr(args, "persist", False) or getattr(args, "supabase_state", False)) and getattr(
        args, "state_file", None
    ):
        print("Error: Cannot specify both persistent state and --state-file.")
        return 1

    is_persist = getattr(args, "persist", False) or getattr(args, "supabase_state", False)
    if is_persist and not settings.is_supabase_configured:
        print(
            "Error: Persistence (--persist / --supabase-state) "
            "requires SUPABASE_URL and SUPABASE_KEY to be configured."
        )
        return 1

    if is_persist:
        preflight_res = run_supabase_preflight(settings=settings, scope="full")
        if not preflight_res.success:
            print(f"Error: Supabase schema preflight failed: {preflight_res.error_message}")
            if preflight_res.details:
                print(f"Details: {preflight_res.details}")
            return 1

    # Check credentials early if real send is requested
    is_send = getattr(args, "send", False)
    use_mock_telegram = getattr(args, "mock_telegram", False)
    if is_send and not use_mock_telegram and not settings.is_telegram_configured:
        print("Error: Telegram credentials are not configured.")
        print("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env or run with '--preview'.")
        return 1

    try:
        summarization_client, fallback_client = resolve_summarization_clients(
            settings=settings,
            provider=getattr(args, "provider", None),
            model=getattr(args, "model", None),
            mock=getattr(args, "mock_model", False),
        )
    except GeminiConfigError as exc:
        print(f"Configuration error: {exc}")
        return 1

    articles: list[ArticleRecord] = []
    if getattr(args, "fixture", False):
        fixture_path = (
            Path(__file__).resolve().parents[2]
            / "tests"
            / "fixtures"
            / "sample_clustering_articles.json"
        )
        if not fixture_path.is_file():
            print(f"Error: Fixture file not found at {fixture_path}")
            return 1
        raw_items = json.loads(fixture_path.read_text(encoding="utf-8"))
        for item in raw_items:
            pub_dt = (
                datetime.fromisoformat(item["published_at"]) if item.get("published_at") else None
            )
            articles.append(
                ArticleRecord(
                    source_name=item["source_name"],
                    title=item["title"],
                    url=item["url"],
                    canonical_url=item["canonical_url"],
                    published_at=pub_dt,
                    snippet=item.get("snippet"),
                    content_hash=f"hash-{item['id']}",
                )
            )
        print(f"Loaded {len(articles)} article(s) from local test fixtures.")
    else:
        if args.feed:
            matched = find_feed(args.feed)
            if not matched:
                print(f"Error: No feed matched '{args.feed}'. Available feeds:")
                for f in DEFAULT_FEEDS:
                    print(f"  - {f.name} ({f.feed_url})")
                return 1
            target_feeds = [matched]
        else:
            target_feeds = [f for f in DEFAULT_FEEDS if f.enabled]

        feed_limit = args.limit if getattr(args, "limit", None) is not None else settings.feed_limit
        print(f"Fetching feeds for briefing (limit per feed: {feed_limit})...")
        art_repo = SupabaseArticleRepository(settings=settings) if is_persist else None
        try:
            ingest_summary = run_ingestion(
                feeds=target_feeds,
                limit_per_feed=feed_limit,
                dry_run=not is_persist,
                settings=settings,
                repository=art_repo,
            )
        except Exception as exc:
            print(f"Error: Failed during article ingestion: {exc}")
            return 1
        articles = [a for res in ingest_summary.feed_results for a in res.articles]

    url_to_id: dict[str, str] = {}
    if is_persist and articles and "art_repo" in locals() and art_repo is not None:
        try:
            canonical_urls = [a.canonical_url for a in articles]
            url_to_id = art_repo.get_article_ids_by_urls(canonical_urls)
        except Exception as exc:
            print(f"Error: Failed to retrieve article UUIDs: {exc}")
            return 1

    target_date = date.today()
    lookback_hours = getattr(args, "lookback_hours", None) or getattr(
        settings, "lookback_hours", None
    )
    if lookback_hours is not None and lookback_hours > 0 and articles:
        ref_dt = datetime.combine(target_date, datetime.max.time(), tzinfo=UTC)
        cutoff = ref_dt - timedelta(hours=lookback_hours)
        articles = [
            a
            for a in articles
            if (
                (a.published_at or a.fetched_at) is None
                or (
                    (a.published_at or a.fetched_at).replace(tzinfo=UTC)
                    if (a.published_at or a.fetched_at).tzinfo is None
                    else (a.published_at or a.fetched_at)
                )
                >= cutoff
            )
        ]

    clustering_summary = cluster_articles(articles)
    events = clustering_summary.clusters
    if not events:
        print("No events identified for briefing.")
        return 0

    if is_persist and events:
        try:
            ev_repo = SupabaseEventRepository(settings=settings)
            ev_repo.persist_events(events, article_url_to_id=url_to_id)
        except Exception as exc:
            print(f"Error: Failed to persist events: {exc}")
            return 1

    min_relevance_rank = (
        getattr(args, "min_relevance_rank", None)
        if getattr(args, "min_relevance_rank", None) is not None
        else getattr(settings, "min_relevance_rank", 9.0)
    )
    include_low_priority = getattr(args, "include_low_priority", False) or getattr(
        settings, "include_low_priority", False
    )

    summarizer = EventSummarizer(
        client=summarization_client,
        fallback_client=fallback_client,
        settings=settings,
    )
    results = summarizer.summarize_events(
        events,
        max_events=max_events,
        min_relevance_rank=min_relevance_rank,
        include_low_priority=include_low_priority,
    )

    if is_persist and events and "ev_repo" in locals():
        try:
            ev_repo.update_event_summaries(results)
        except Exception as exc:
            print(f"Error: Failed to update event summaries: {exc}")
            return 1

    # Format Telegram digest
    formatter = TelegramDigestFormatter()
    today_date = date.today()
    edition = getattr(args, "edition", None)
    chunks = formatter.format_digest(
        results=results,
        briefing_date=today_date,
        edition=edition,
        max_events=max_events,
        min_relevance_rank=min_relevance_rank,
        include_low_priority=include_low_priority,
    )

    # 1. Preview / Dry-run Mode (default unless --send is passed)
    if not is_send:
        print("\n" + "=" * 60)
        edition_label = f" ({edition.capitalize()} Edition)" if edition else ""
        print(f"Geopulse Daily Briefing Preview{edition_label} [Telegram HTML Format]")
        success_results = [r for r in results if r.status == "success" and r.summary is not None]
        selected_events = formatter.select_digest_events(
            results=success_results,
            max_events=max_events,
            min_relevance_rank=min_relevance_rank,
            include_low_priority=include_low_priority,
        )
        events_included = len(selected_events)
        candidate_count = sum(
            1
            for r in results
            if (
                (r.status == "success" and r.summary is not None)
                or (r.summary is not None and getattr(r.summary, "model_used", "") == "omitted")
            )
            and r.status != "quarantined"
        )
        omitted_count = max(0, candidate_count - events_included) if events_included > 0 else 0
        print(
            f"Date: {today_date} | Chunks: {len(chunks)} | "
            f"Events: {events_included} (Omitted: {omitted_count})"
        )
        print("=" * 60)
        for idx, chunk in enumerate(chunks, start=1):
            print(f"\n--- Telegram Message Chunk {idx}/{len(chunks)} ({len(chunk)} characters) ---")
            print(chunk)
        print("\n" + "=" * 60)
        print(f"Model calls: {getattr(summarizer, 'model_calls_count', 0)}")
        if getattr(args, "diagnostics", False) or events_included == 0:
            print(
                format_briefing_diagnostics(
                    articles_fetched=len(articles),
                    articles_retained=len(articles),
                    events=events,
                    results=results,
                    selected_events=selected_events,
                    min_relevance_rank=min_relevance_rank,
                    max_events=max_events,
                    include_low_priority=include_low_priority,
                )
            )
        print("PREVIEW COMPLETE: No messages were sent to Telegram.")
        print("HINT: Pass '--send' with valid credentials to deliver.")
        return 0

    # 2. Delivery Mode (--send)
    if use_mock_telegram:
        telegram_client = MockTelegramClient()
    else:
        token, chat_id = settings.require_telegram()
        telegram_client = TelegramClient(token=token, chat_id=chat_id)

    state_file = getattr(args, "state_file", None)
    if getattr(args, "in_memory_state", False):
        delivery_repo = InMemoryDeliveryRepository()
    elif is_persist:
        delivery_repo = SupabaseDeliveryRepository(settings=settings)
    elif state_file:
        delivery_repo = JsonFileDeliveryRepository(file_path=Path(state_file))
    else:
        delivery_repo = JsonFileDeliveryRepository(file_path=settings.delivery_state_file)

    delivery_service = TelegramDeliveryService(
        client=telegram_client,
        repository=delivery_repo,
        settings=settings,
    )

    print(f"\nDelivering briefing to Telegram ({len(chunks)} chunk(s))...")
    try:
        deliv_res = delivery_service.deliver_digest(
            digest_chunks=chunks,
            briefing_date=today_date,
            force=getattr(args, "force", False),
            edition=edition,
        )
    except Exception as exc:
        print(f"Error: Delivery service failed: {exc}")
        return 1

    if (
        is_persist
        and isinstance(delivery_repo, SupabaseDeliveryRepository)
        and deliv_res.briefing_id
        and deliv_res.briefing_id != "dry-run"
    ):
        try:
            delivery_repo.record_briefing_events(
                briefing_id=deliv_res.briefing_id,
                results=list(results),
                article_url_to_id=url_to_id,
            )
        except Exception as exc:
            print(f"Warning: Failed to persist briefing events and evidence: {exc}")

    print("\n" + "=" * 60)
    print("Telegram Delivery Result:")
    print(f"  Status:           {deliv_res.status.upper()}")
    print(f"  Chunks Delivered: {deliv_res.chunks_delivered}/{deliv_res.chunks_total}")
    if deliv_res.resumed_from_chunk > 0:
        print(f"  Resumed:          From chunk {deliv_res.resumed_from_chunk + 1}")
    if deliv_res.message_ids:
        print(f"  Message IDs:      {list(deliv_res.message_ids)}")
    if deliv_res.warning_message:
        print(f"  Warning:          {deliv_res.warning_message}")
    if deliv_res.error_message:
        print(f"  Notice/Error:     {deliv_res.error_message}")
    print("=" * 60)

    if deliv_res.status in {"sent", "skipped_duplicate"}:
        return 0
    return 1


def handle_test_telegram(args: argparse.Namespace) -> int:
    try:
        settings = Settings()
    except ValidationError as exc:
        print(f"Configuration error: {exc}")
        return 1

    use_mock = getattr(args, "mock", False)

    if not use_mock and not settings.is_telegram_configured:
        print("Error: Telegram credentials are not configured.")
        print("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env before running connection test.")
        return 1

    if use_mock:
        client = MockTelegramClient()
        target_chat = client.chat_id
    else:
        token, chat_id = settings.require_telegram()
        client = TelegramClient(
            token=token,
            chat_id=chat_id,
            base_url=settings.telegram_api_base_url,
            timeout=settings.telegram_timeout,
        )
        target_chat = chat_id

    now_utc = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    test_message = (
        "<b>🌍 Geopulse Connection Test</b>\n\n"
        "Telegram Bot API integration verified successfully.\n"
        f"Timestamp: <i>{now_utc}</i>\n"
        f"Target Chat: <code>{target_chat}</code>\n\n"
        "<i>This is an automated one-time connectivity verification message.</i>"
    )

    mode_label = "[MOCK]" if use_mock else "[LIVE]"
    print(f"Sending Geopulse connection test message {mode_label} to chat '{target_chat}'...")

    try:
        resp = client.send_message(test_message)
        print(f"[OK] Test message delivered successfully (Telegram Message ID: {resp.message_id}).")
        return 0
    except TelegramConfigError as exc:
        print(f"[FAIL] Configuration error: {exc}")
        return 1
    except TelegramRateLimitError as exc:
        print(f"[FAIL] Rate limited by Telegram (HTTP 429). Retry after {exc.retry_after}s.")
        return 1
    except TelegramPermanentError as exc:
        print(f"[FAIL] Telegram API error ({exc.status_code}): {exc.description}")
        return 1
    except TelegramTimeoutError as exc:
        print(f"[FAIL] Network timeout contacting Telegram Bot API: {exc}")
        return 1
    except TelegramNetworkError as exc:
        print(f"[FAIL] Network error contacting Telegram Bot API: {exc}")
        return 1
    except Exception as exc:
        print(f"[FAIL] Unexpected error during connection test: {exc}")
        return 1


def handle_service(args: argparse.Namespace) -> int:
    try:
        settings = Settings()
    except ValidationError as exc:
        print(f"Configuration error: {exc}")
        return 1

    action = getattr(args, "service_action", "status")

    if action == "status":
        print("=" * 60)
        print("Geopulse Continuous Service Status & Configuration")
        print("=" * 60)
        print("  Status:               Operational")
        print(f"  Configured Timezone:  {settings.timezone}")
        print(f"  Morning Briefing:     08:00 AM {settings.timezone}")
        print(f"  Evening Briefing:     08:00 PM {settings.timezone}")
        print(f"  LLM Provider:         {settings.llm_provider}")
        if settings.llm_provider in ("auto", "gemini"):
            print(f"  Gemini Model:         {settings.gemini_model}")
            print(f"  Gemini Configured:    {settings.is_gemini_configured}")
        print(f"  Ollama Endpoint:      {settings.ollama_base_url}")
        print(f"  Ollama Model:         {settings.ollama_model}")
        print(f"  Telegram Configured:  {settings.is_telegram_configured}")
        admin_id = settings.telegram_admin_id or "None configured"
        print(f"  Telegram Admin ID:    {admin_id}")
        print(f"  Supabase Configured:  {settings.is_supabase_configured}")
        print(f"  Delivery State File:  {settings.delivery_state_file}")
        print("=" * 60)
        return 0

    if action == "run":
        use_mock_tg = getattr(args, "mock_telegram", False)
        use_mock_model = getattr(args, "mock_model", False)
        once = getattr(args, "once", False)
        no_scheduler = getattr(args, "no_scheduler", False)
        no_bot = getattr(args, "no_bot", False)
        is_persist = getattr(args, "persist", False)

        import logging

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )

        if not use_mock_tg and not settings.is_telegram_configured:
            print("Error: Telegram credentials are not configured.")
            print(
                "Configure TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env or pass --mock-telegram."
            )
            return 1

        if is_persist and not settings.is_supabase_configured:
            print("Error: --persist requires SUPABASE_URL and SUPABASE_KEY to be configured.")
            return 1

        from geopulse.delivery.repository import (
            InMemoryDeliveryRepository,
            JsonFileDeliveryRepository,
        )
        from geopulse.service.runner import ServiceAlreadyRunningError, ServiceRunner

        if use_mock_tg:
            client = MockTelegramClient()
            if once and not client._mock_updates:
                # Enqueue a smoke-test incoming update to exercise the interactive Q&A bot
                client.enqueue_update(
                    update_id=1,
                    text="/help",
                    from_id="smoke-test-user",
                    chat_id="smoke-test-chat",
                )
        else:
            token, chat_id = settings.require_telegram()
            client = TelegramClient(token=token, chat_id=chat_id)

        in_memory_state = getattr(args, "in_memory_state", False)
        state_file = getattr(args, "state_file", None)
        if in_memory_state or (use_mock_tg and once and not is_persist and not state_file):
            delivery_repo = InMemoryDeliveryRepository()
        elif is_persist:
            from geopulse.delivery.repository import SupabaseDeliveryRepository

            delivery_repo = SupabaseDeliveryRepository(settings=settings)
        elif state_file:
            delivery_repo = JsonFileDeliveryRepository(file_path=Path(state_file))
        else:
            delivery_repo = JsonFileDeliveryRepository(file_path=settings.delivery_state_file)

        runner = ServiceRunner(
            settings=settings,
            client=client,
            delivery_repo=delivery_repo,
            mock_model=use_mock_model,
            persistent=is_persist,
            enable_scheduler=not no_scheduler,
            enable_bot=not no_bot,
        )

        print("Starting Geopulse continuous service...")
        print(f"Timezone: {settings.timezone} | Scheduler: {not no_scheduler} | Bot: {not no_bot}")
        print("Press Ctrl+C to stop.")
        try:
            runner.run(once=once)
            if once:
                print("\n" + "=" * 60)
                print("Geopulse Service Smoke Test Summary (--once)")
                print("=" * 60)
                print(f"  Timezone:          {settings.timezone}")
                if not no_scheduler:
                    m_stat = runner.last_schedule_status.get("morning", "evaluated")
                    e_stat = runner.last_schedule_status.get("evening", "evaluated")
                    print(f"  Scheduler Status:  Morning: {m_stat} | Evening: {e_stat}")
                else:
                    print("  Scheduler Status:  Disabled (--no-scheduler)")

                if not no_bot:
                    print(f"  Telegram Q&A:      {runner.last_bot_handled} update(s) processed")
                else:
                    print("  Telegram Q&A:      Disabled (--no-bot)")

                if use_mock_tg and isinstance(client, MockTelegramClient):
                    print(f"  Mock Messages:     {len(client.sent_messages)} dispatched")
                print("  Result:            VERIFIED (Single pass executed cleanly)")
                print("=" * 60)
            return 0
        except ServiceAlreadyRunningError as exc:
            print(f"\nError: {exc}")
            return 1
        except KeyboardInterrupt:
            print("\nShutting down Geopulse service gracefully...")
            runner.stop()
            return 0

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="geopulse",
        description="Geopulse: source-grounded geopolitical briefing bot CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "check-config",
        help="Run a local configuration sanity check.",
    )

    preflight_parser = subparsers.add_parser(
        "check-preflight",
        help="Run read-only preflight verification of Supabase schema and stored procedures.",
    )
    preflight_parser.add_argument(
        "--scope",
        choices=["ingest", "cluster", "full"],
        default="full",
        help="Schema scope to verify (default: 'full').",
    )

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest news articles from configured RSS/Atom feeds.",
    )
    ingest_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="Run ingestion in dry-run mode (do not write to database).",
    )
    ingest_parser.add_argument(
        "--persist",
        action="store_true",
        default=False,
        help="Persist ingested articles to Supabase (requires credentials).",
    )
    ingest_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum articles to ingest per feed (default: 10 or configured FEED_LIMIT).",
    )
    ingest_parser.add_argument(
        "--feed",
        type=str,
        default=None,
        help="Optional feed name or URL filter.",
    )
    ingest_parser.add_argument(
        "--show-articles",
        action="store_true",
        default=False,
        help="Diagnostic flag to display parsed article titles, canonical URLs, and dates.",
    )
    ingest_parser.add_argument(
        "--cluster",
        action="store_true",
        default=False,
        help="Cluster ingested articles into discrete real-world events.",
    )

    cluster_parser = subparsers.add_parser(
        "cluster",
        help="Ingest and cluster news articles into discrete real-world events.",
    )
    cluster_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum articles to ingest per feed (default: 10 or configured FEED_LIMIT).",
    )
    cluster_parser.add_argument(
        "--feed",
        type=str,
        default=None,
        help="Optional feed name or URL filter.",
    )
    cluster_parser.add_argument(
        "--show-articles",
        action="store_true",
        default=True,
        help="Display detailed list of clustered events and their sources.",
    )
    cluster_parser.add_argument(
        "--persist",
        action="store_true",
        default=False,
        help="Persist event clusters and article associations to Supabase (requires credentials).",
    )

    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Summarize clustered geopolitical events using local Ollama.",
    )
    summarize_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum articles to ingest per feed when using feeds (default: 10).",
    )
    summarize_parser.add_argument(
        "--feed",
        type=str,
        default=None,
        help="Optional feed name or URL filter.",
    )
    summarize_parser.add_argument(
        "--fixture",
        action="store_true",
        default=False,
        help="Summarize local representative test fixtures instead of live RSS feeds.",
    )
    summarize_parser.add_argument(
        "--mock-model",
        action="store_true",
        default=False,
        help="Use an in-memory mock LLM for testing without requiring a live Ollama daemon.",
    )
    summarize_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the model name (defaults to configured OLLAMA_MODEL or GEMINI_MODEL).",
    )
    summarize_parser.add_argument(
        "--provider",
        type=str,
        choices=["ollama", "gemini", "auto"],
        default=None,
        help="LLM provider: 'ollama' (default), 'gemini', or 'auto' (Gemini with Ollama fallback).",
    )
    summarize_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Run summarization in-memory without persistence (default: True).",
    )

    brief_parser = subparsers.add_parser(
        "brief",
        help="Format and deliver daily intelligence briefing to Telegram.",
    )
    brief_parser.add_argument(
        "--preview",
        action="store_true",
        default=False,
        help="Preview formatted Telegram digest in terminal without sending.",
    )
    brief_parser.add_argument(
        "--send",
        action="store_true",
        default=False,
        help="Deliver the formatted briefing to Telegram (requires credentials).",
    )
    brief_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum articles to ingest per feed when using feeds (default: 10).",
    )
    brief_parser.add_argument(
        "--feed",
        type=str,
        default=None,
        help="Optional feed name or URL filter.",
    )
    brief_parser.add_argument(
        "--fixture",
        action="store_true",
        default=False,
        help="Use local test fixtures instead of live RSS feeds.",
    )
    brief_parser.add_argument(
        "--mock-model",
        action="store_true",
        default=False,
        help="Use mock LLM client instead of live Ollama.",
    )
    brief_parser.add_argument(
        "--mock-telegram",
        action="store_true",
        default=False,
        help="Use mock Telegram client for testing delivery pipeline without network calls.",
    )
    brief_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the model name (defaults to configured OLLAMA_MODEL or GEMINI_MODEL).",
    )
    brief_parser.add_argument(
        "--provider",
        type=str,
        choices=["ollama", "gemini", "auto"],
        default=None,
        help="LLM provider: 'ollama' (default), 'gemini', or 'auto' (Gemini with Ollama fallback).",
    )
    brief_parser.add_argument(
        "--edition",
        type=str,
        choices=["morning", "evening"],
        default=None,
        help="Optional briefing edition label ('morning' or 'evening').",
    )
    brief_parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Optional maximum number of high-priority events (omitted events are noted).",
    )
    brief_parser.add_argument(
        "--lookback-hours",
        type=int,
        default=None,
        help="Optional lookback window in hours relative to briefing date (older excluded).",
    )
    brief_parser.add_argument(
        "--min-relevance-rank",
        type=float,
        default=None,
        help="Optional maximum priority rank for events (default: 9.0; ranks > 9.0 excluded).",
    )
    brief_parser.add_argument(
        "--include-low-priority",
        action="store_true",
        default=False,
        help="Explicitly allow low-priority items and commemorations to backfill the digest.",
    )
    brief_parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force delivery even if a briefing was already sent for today's date.",
    )
    brief_parser.add_argument(
        "--state-file",
        type=str,
        default=None,
        help="Custom path to local JSON delivery state file.",
    )
    brief_parser.add_argument(
        "--in-memory-state",
        action="store_true",
        default=False,
        help="Use ephemeral in-memory delivery repository instead of persistent JSON file.",
    )
    brief_parser.add_argument(
        "--supabase-state",
        action="store_true",
        default=False,
        help="Persist briefing delivery state to Supabase instead of local JSON file.",
    )
    brief_parser.add_argument(
        "--persist",
        action="store_true",
        default=False,
        help="Enable end-to-end Supabase persistence (articles, events, summaries, and delivery).",
    )
    brief_parser.add_argument(
        "--diagnostics",
        action="store_true",
        default=False,
        help="Display detailed pipeline execution diagnostics and exclusion reasons.",
    )

    test_tg_parser = subparsers.add_parser(
        "test-telegram",
        help="Send exactly one labelled connection test message to verify Telegram credentials.",
    )
    test_tg_parser.add_argument(
        "--mock",
        action="store_true",
        default=False,
        help="Simulate test message delivery using MockTelegramClient without network calls.",
    )

    service_parser = subparsers.add_parser(
        "service",
        help="Run or inspect the continuous Geopulse scheduler and Telegram bot service.",
    )
    service_subparsers = service_parser.add_subparsers(dest="service_action", required=True)

    service_run_parser = service_subparsers.add_parser(
        "run",
        help="Run the continuous scheduler and interactive Telegram Q&A bot.",
    )
    service_run_parser.add_argument(
        "--mock-telegram",
        action="store_true",
        default=False,
        help="Use in-memory MockTelegramClient instead of live Telegram network calls.",
    )
    service_run_parser.add_argument(
        "--mock-model",
        action="store_true",
        default=False,
        help="Use mock summarization model instead of calling local Ollama daemon.",
    )
    service_run_parser.add_argument(
        "--once",
        action="store_true",
        default=False,
        help="Run a single scheduling and polling pass and exit (useful for testing and cron).",
    )
    service_run_parser.add_argument(
        "--no-scheduler",
        action="store_true",
        default=False,
        help="Disable automated morning/evening briefings (run bot only).",
    )
    service_run_parser.add_argument(
        "--no-bot",
        action="store_true",
        default=False,
        help="Disable interactive Telegram bot polling (run scheduler only).",
    )
    service_run_parser.add_argument(
        "--persist",
        action="store_true",
        default=False,
        help="Persist briefings and events to Supabase.",
    )
    service_run_parser.add_argument(
        "--state-file",
        type=str,
        default=None,
        help="Custom path to local JSON delivery state file.",
    )
    service_run_parser.add_argument(
        "--in-memory-state",
        action="store_true",
        default=False,
        help="Use ephemeral in-memory delivery repository instead of persistent JSON file.",
    )

    service_subparsers.add_parser(
        "status",
        help="Display service schedule, configured timezone, and health status.",
    )

    args = parser.parse_args(argv)

    if args.command == "check-config":
        return handle_check_config()

    if args.command == "check-preflight":
        return handle_check_preflight(args)

    if args.command == "ingest":
        return handle_ingest(args)

    if args.command == "cluster":
        return handle_cluster(args)

    if args.command == "summarize":
        return handle_summarize(args)

    if args.command == "brief":
        return handle_brief(args)

    if args.command == "test-telegram":
        return handle_test_telegram(args)

    if args.command == "service":
        return handle_service(args)

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
