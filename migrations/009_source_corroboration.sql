-- Geopulse Phase 3 MVP: Independent Source Corroboration Metadata
-- DO NOT APPLY TO LIVE DATABASE AUTOMATICALLY. Draft migration for offline review.
--
-- This additive migration enhances events, briefing_events, and briefing_event_evidence
-- with source lineage and claim-level corroboration tracking columns.
--
-- Explicitly represents historical legacy records as unassessed (0 chains) to avoid
-- fabricating historical corroboration for unanalyzed events.

-- 1. Add corroboration fields to public.events
alter table if exists public.events
    add column if not exists corroboration_status text default 'unassessed',
    add column if not exists independent_chains_count integer not null default 0,
    add column if not exists corroboration_detail jsonb not null default '{}'::jsonb;

-- 2. Add corroboration fields to public.briefing_events
alter table if exists public.briefing_events
    add column if not exists corroboration_status text default 'unassessed',
    add column if not exists independent_chains_count integer not null default 0,
    add column if not exists corroboration_detail jsonb not null default '{}'::jsonb;

-- 3. Add lineage fields to public.briefing_event_evidence
alter table if exists public.briefing_event_evidence
    add column if not exists lineage_type text,
    add column if not exists original_source text;

-- 4. Fast lookup indexes
create index if not exists events_corroboration_status_idx
    on public.events (corroboration_status);

create index if not exists briefing_events_corroboration_status_idx
    on public.briefing_events (corroboration_status);

create index if not exists briefing_evidence_lineage_idx
    on public.briefing_event_evidence (lineage_type);
