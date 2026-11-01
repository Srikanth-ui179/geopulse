"""Preflight schema and connectivity verification for Supabase persistence.

Executes read-only, non-destructive probes before any pipeline writes occur.
Detects missing tables or missing columns resulting from unapplied migrations
(001 through 005) and fails closed without leaking credentials.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from geopulse.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Required tables and columns expected across migrations 001-005
FULL_REQUIRED_SCHEMA: dict[str, tuple[str, ...]] = {
    "sources": ("id", "name", "feed_url", "enabled"),
    "articles": (
        "id",
        "source_id",
        "url",
        "canonical_url",
        "title",
        "relevance_status",
        "topic",
    ),
    "events": (
        "id",
        "title",
        "category",
        "confidence_label",
        "published_at",
        "is_analysis",
        "headline",
        "key_details",
        "confidence_qualification",
        "model_used",
    ),
    "event_articles": ("event_id", "article_id"),
    "briefings": (
        "id",
        "briefing_date",
        "status",
        "telegram_chat_id",
        "telegram_message_ids",
        "digest_hash",
        "chunks_delivered",
    ),
    "briefing_events": (
        "briefing_id",
        "event_id",
        "rank",
        "summary",
        "headline",
        "key_details",
        "confidence_qualification",
        "model_used",
    ),
    "briefing_event_evidence": ("briefing_id", "event_id", "article_id"),
}

INGEST_REQUIRED_SCHEMA: dict[str, tuple[str, ...]] = {
    "sources": ("id", "name", "feed_url", "enabled"),
    "articles": (
        "id",
        "source_id",
        "url",
        "canonical_url",
        "title",
        "relevance_status",
        "topic",
    ),
}

CLUSTER_REQUIRED_SCHEMA: dict[str, tuple[str, ...]] = {
    "sources": ("id", "name", "feed_url", "enabled"),
    "articles": (
        "id",
        "source_id",
        "url",
        "canonical_url",
        "title",
        "relevance_status",
        "topic",
    ),
    "events": (
        "id",
        "title",
        "category",
        "confidence_label",
        "published_at",
        "is_analysis",
    ),
    "event_articles": ("event_id", "article_id"),
}


@dataclass
class PreflightResult:
    """Outcome of Supabase connectivity and schema preflight validation."""

    success: bool
    checked_tables: tuple[str, ...] = ()
    missing_tables: tuple[str, ...] = ()
    missing_columns: dict[str, list[str]] = field(default_factory=dict)
    missing_functions: tuple[str, ...] = ()
    error_message: str | None = None
    details: str | None = None


def run_supabase_preflight(
    client: Any = None,
    settings: Settings | None = None,
    scope: str = "full",
) -> PreflightResult:
    """Perform read-only preflight connectivity and schema inspection before writes.

    Args:
        client: Optional Supabase client instance. If None, created from settings.
        settings: Application settings. If None, loaded from environment.
        scope: Required schema scope ("ingest", "cluster", or "full").

    Returns:
        PreflightResult: Detailed preflight outcome.
    """
    cfg = settings or get_settings()

    if client is None:
        if not cfg.is_supabase_configured:
            return PreflightResult(
                success=False,
                error_message=(
                    "Supabase credentials are not configured. "
                    "Set SUPABASE_URL and SUPABASE_KEY in .env."
                ),
            )
        if cfg.supabase_key_category == "publishable/anon":
            return PreflightResult(
                success=False,
                error_message=(
                    "SUPABASE_KEY is configured with a publishable/anon key. "
                    "Persistent operations require a service_role secret key to bypass "
                    "PostgreSQL Row-Level Security (RLS)."
                ),
                details=(
                    "In the Supabase Dashboard, go to Project Settings > API > Project API keys "
                    "and copy the 'service_role' (secret) key into SUPABASE_KEY in .env."
                ),
            )
        try:
            url, key = cfg.require_supabase()
            from supabase import create_client

            client = create_client(url, key)
        except Exception as exc:
            return PreflightResult(
                success=False,
                error_message=f"Failed to connect to Supabase: {exc}",
            )

    if scope == "ingest":
        target_schema = INGEST_REQUIRED_SCHEMA
    elif scope == "cluster":
        target_schema = CLUSTER_REQUIRED_SCHEMA
    else:
        target_schema = FULL_REQUIRED_SCHEMA

    checked_tables: list[str] = []
    missing_tables: list[str] = []
    missing_columns: dict[str, list[str]] = {}

    for table_name, columns in target_schema.items():
        # First attempt: Probe all expected columns via read-only limit(0)
        try:
            cols_str = ",".join(columns)
            client.table(table_name).select(cols_str).limit(0).execute()
            checked_tables.append(table_name)
        except Exception as exc:
            # Check if failure is due to missing table
            table_missing = False
            try:
                # Probe table presence with a simple wildcard probe
                client.table(table_name).select("*").limit(0).execute()
            except Exception as tbl_exc:
                t_err = str(tbl_exc).lower()
                if (
                    "does not exist" in t_err
                    or "not found" in t_err
                    or "schema cache" in t_err
                    or "404" in t_err
                    or "relation" in t_err
                ):
                    table_missing = True
                    missing_tables.append(table_name)
                else:
                    return PreflightResult(
                        success=False,
                        checked_tables=tuple(checked_tables),
                        error_message=f"Preflight error querying table '{table_name}': {tbl_exc}",
                    )

            if not table_missing:
                # Table exists, so one or more columns are missing. Isolate missing columns.
                failed_cols: list[str] = []
                for col in columns:
                    try:
                        client.table(table_name).select(col).limit(0).execute()
                    except Exception as col_exc:
                        c_err = str(col_exc).lower()
                        if (
                            "column" in c_err
                            or "schema cache" in c_err
                            or "does not exist" in c_err
                            or "400" in c_err
                        ):
                            failed_cols.append(col)
                        else:
                            failed_cols.append(col)
                if failed_cols:
                    missing_columns[table_name] = failed_cols
                else:
                    return PreflightResult(
                        success=False,
                        checked_tables=tuple(checked_tables),
                        error_message=f"Preflight error querying columns in '{table_name}': {exc}",
                    )

    # 2. Probe stored procedures for atomic clustering when in scope and client supports RPC
    missing_functions: list[str] = []
    if scope in ("cluster", "full") and hasattr(client, "rpc"):
        try:
            # Read-only probe: passing empty arrays returns immediately with 0 writes
            client.rpc(
                "upsert_events_with_articles",
                {"p_events": [], "p_event_articles": []},
            ).execute()
        except Exception as rpc_exc:
            r_err = str(rpc_exc)
            is_missing_func = (
                "PGRST202" in r_err
                or "Could not find the function" in r_err
                or (
                    "upsert_events_with_articles" in r_err
                    and ("could not find" in r_err.lower() or "does not exist" in r_err.lower())
                )
            )
            if is_missing_func:
                missing_functions.append("upsert_events_with_articles")
            elif any(
                t in r_err.lower() for t in ("401", "403", "permission denied", "unauthorized")
            ):
                return PreflightResult(
                    success=False,
                    checked_tables=tuple(checked_tables),
                    missing_tables=tuple(missing_tables),
                    missing_columns=missing_columns,
                    missing_functions=tuple(missing_functions),
                    error_message=(
                        f"Permission denied executing RPC 'upsert_events_with_articles': {rpc_exc}"
                    ),
                    details=(
                        "Ensure the configured Supabase API key has EXECUTE permission on "
                        "function public.upsert_events_with_articles."
                    ),
                )
            else:
                return PreflightResult(
                    success=False,
                    checked_tables=tuple(checked_tables),
                    missing_tables=tuple(missing_tables),
                    missing_columns=missing_columns,
                    missing_functions=tuple(missing_functions),
                    error_message=f"Error probing RPC 'upsert_events_with_articles': {rpc_exc}",
                )

    if missing_tables or missing_columns or missing_functions:
        reasons: list[str] = []
        if missing_tables:
            reasons.append(f"Missing required table(s): {', '.join(missing_tables)}")
        if missing_columns:
            col_desc = "; ".join(
                f"{tbl} ({', '.join(cols)})" for tbl, cols in missing_columns.items()
            )
            reasons.append(f"Missing required column(s): {col_desc}")
        if missing_functions:
            reasons.append(f"Missing required stored procedure(s): {', '.join(missing_functions)}")

        if missing_functions and "upsert_events_with_articles" in missing_functions:
            migration_details = (
                "Pending database migration 006_atomic_event_persistence.sql must be reviewed "
                "and applied to Supabase before persistent clustering can proceed."
            )
        else:
            migration_details = (
                "Pending database migrations (001-005) must be applied to Supabase "
                "before persistent execution can proceed."
            )

        return PreflightResult(
            success=False,
            checked_tables=tuple(checked_tables),
            missing_tables=tuple(missing_tables),
            missing_columns=missing_columns,
            missing_functions=tuple(missing_functions),
            error_message="; ".join(reasons),
            details=migration_details,
        )

    return PreflightResult(
        success=True,
        checked_tables=tuple(checked_tables),
        details=(
            f"Preflight passed: verified accessibility of {len(checked_tables)} "
            "required table(s) and critical columns in PostgREST schema cache."
        ),
    )
