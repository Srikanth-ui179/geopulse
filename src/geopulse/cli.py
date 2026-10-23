"""Command line interface for Geopulse."""

import argparse
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path

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
    MockOllamaClient,
    OllamaClient,
    SummarizationResult,
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
    print(f"Ollama endpoint: {settings.ollama_base_url}")
    print(f"Ollama model: {settings.ollama_model}")
    print(f"Supabase configured: {settings.is_supabase_configured}")
    print(f"Telegram configured: {settings.is_telegram_configured}")
    return 0


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
    print(f"Starting Geopulse RSS Ingestion [{mode_label}]")
    print(f"Processing {len(target_feeds)} feed(s), limit per feed: {args.limit}...\n")

    summary = run_ingestion(
        feeds=target_feeds,
        limit_per_feed=args.limit,
        dry_run=dry_run,
        settings=settings,
    )

    for idx, result in enumerate(summary.feed_results, start=1):
        status_tag = "OK" if result.success else "FAIL"
        print(f"[{idx}/{len(summary.feed_results)}] [{status_tag}] {result.feed_name}")
        print(f"      URL: {result.feed_url}")
        if result.success:
            relevant_count = sum(1 for a in result.articles if a.relevance_status == "relevant")
            pending_count = sum(1 for a in result.articles if a.relevance_status == "pending")
            irrelevant_count = sum(
                1 for a in result.articles if a.relevance_status == "irrelevant"
            )
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

    print(f"Fetching feeds to cluster events (limit per feed: {args.limit})...")
    summary = run_ingestion(
        feeds=target_feeds,
        limit_per_feed=args.limit,
        dry_run=True,
        settings=settings,
    )

    all_articles = [a for res in summary.feed_results for a in res.articles]
    clustering_res = cluster_articles(all_articles)
    print_clustering_summary(clustering_res, show_details=getattr(args, "show_articles", True))

    if getattr(args, "persist", False):
        if not settings.is_supabase_configured:
            print("Error: --persist requires SUPABASE_URL and SUPABASE_KEY to be configured.")
            return 1
        preflight_res = run_supabase_preflight(settings=settings, scope="cluster")
        if not preflight_res.success:
            print(f"Error: Supabase schema preflight failed: {preflight_res.error_message}")
            if preflight_res.details:
                print(f"Details: {preflight_res.details}")
            return 1
        ev_repo = SupabaseEventRepository(settings=settings)
        ev_repo.persist_events(clustering_res.clusters)
        print(f"Persisted {len(clustering_res.clusters)} event cluster(s) to Supabase.")

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
            print(f"ERROR: {res.error_message}")
            if res.status == "unavailable":
                print("HINT: Start Ollama with 'ollama serve' or pass '--mock-model' for dry-run.")

    print("\n" + "=" * 60)
    print("Summarization Summary:")
    print(f"  Events Evaluated: {len(results)}")
    success_count = sum(1 for r in results if r.status == "success")
    failed_count = sum(1 for r in results if r.status == "failed")
    unavail_count = sum(1 for r in results if r.status == "unavailable")
    print(f"  - Successful:     {success_count}")
    print(f"  - Failed:         {failed_count}")
    print(f"  - Unavailable:    {unavail_count}")
    print("=" * 60)


def handle_summarize(args: argparse.Namespace) -> int:
    try:
        settings = Settings()
    except ValidationError as exc:
        print(f"Configuration error: {exc}")
        return 1

    model_name = getattr(args, "model", None) or settings.ollama_model
    endpoint = settings.ollama_base_url

    if getattr(args, "mock_model", False):
        client = MockOllamaClient(model=f"mock-{model_name}")
    else:
        client = OllamaClient(base_url=endpoint, model=model_name)

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
                datetime.fromisoformat(item["published_at"])
                if item.get("published_at")
                else None
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

        print(f"Fetching feeds for summarization (limit per feed: {args.limit})...")
        ingest_summary = run_ingestion(
            feeds=target_feeds,
            limit_per_feed=args.limit,
            dry_run=True,
            settings=settings,
        )
        articles = [a for res in ingest_summary.feed_results for a in res.articles]

    clustering_summary = cluster_articles(articles)
    events = clustering_summary.clusters
    if not events:
        print("No events identified for summarization.")
        return 0

    summarizer = EventSummarizer(client=client, settings=settings)
    results = summarizer.summarize_events(events)

    print_summarization_results(
        results=results,
        model_name=client.model,
        endpoint=endpoint,
    )
    return 0 if any(r.status == "success" for r in results) or not results else 1


def handle_brief(args: argparse.Namespace) -> int:
    try:
        settings = Settings()
    except ValidationError as exc:
        print(f"Configuration error: {exc}")
        return 1

    if getattr(args, "preview", False) and getattr(args, "send", False):
        print("Error: Cannot specify both --preview and --send.")
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

    model_name = getattr(args, "model", None) or settings.ollama_model
    endpoint = settings.ollama_base_url

    if getattr(args, "mock_model", False):
        summarization_client = MockOllamaClient(model=f"mock-{model_name}")
    else:
        summarization_client = OllamaClient(base_url=endpoint, model=model_name)

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
                datetime.fromisoformat(item["published_at"])
                if item.get("published_at")
                else None
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

        print(f"Fetching feeds for briefing (limit per feed: {args.limit})...")
        art_repo = SupabaseArticleRepository(settings=settings) if is_persist else None
        try:
            ingest_summary = run_ingestion(
                feeds=target_feeds,
                limit_per_feed=args.limit,
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

    summarizer = EventSummarizer(client=summarization_client, settings=settings)
    results = summarizer.summarize_events(events)

    if is_persist and events and "ev_repo" in locals():
        try:
            ev_repo.update_event_summaries(results)
        except Exception as exc:
            print(f"Error: Failed to update event summaries: {exc}")
            return 1

    # Format Telegram digest
    formatter = TelegramDigestFormatter()
    today_date = date.today()
    chunks = formatter.format_digest(results=results, briefing_date=today_date)

    # 1. Preview / Dry-run Mode (default unless --send is passed)
    if not is_send:
        print("\n" + "=" * 60)
        print("Geopulse Daily Briefing Preview [Telegram HTML Format]")
        print(f"Date: {today_date} | Chunks: {len(chunks)} | Events: {len(events)}")
        print("=" * 60)
        for idx, chunk in enumerate(chunks, start=1):
            print(f"\n--- Telegram Message Chunk {idx}/{len(chunks)} ({len(chunk)} characters) ---")
            print(chunk)
        print("\n" + "=" * 60)
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
        default=10,
        help="Maximum articles to ingest per feed (default: 10).",
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
        default=5,
        help="Maximum articles to ingest per feed (default: 5).",
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
        default=3,
        help="Maximum articles to ingest per feed when using feeds (default: 3).",
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
        help="Override the Ollama model name (defaults to configured OLLAMA_MODEL).",
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
        default=3,
        help="Maximum articles to ingest per feed when using feeds (default: 3).",
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
        help="Override the Ollama model name (defaults to configured OLLAMA_MODEL).",
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

    args = parser.parse_args(argv)

    if args.command == "check-config":
        return handle_check_config()

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

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
