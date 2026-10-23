-- Geopulse Phase 6B: Structured Summary Metadata Enhancements
-- DO NOT APPLY TO LIVE DATABASE AUTOMATICALLY. Draft migration for offline review.
--
-- This migration enhances public.events and public.briefing_events with columns
-- to persist structured LLM summary outputs (key details, headline, confidence qualification, model).

alter table if exists public.events
    add column if not exists headline text,
    add column if not exists key_details jsonb,
    add column if not exists confidence_qualification text,
    add column if not exists model_used text;

alter table if exists public.briefing_events
    add column if not exists headline text,
    add column if not exists key_details jsonb,
    add column if not exists confidence_qualification text,
    add column if not exists model_used text;
