-- Geopulse Phase 3 MVP Rollback: Independent Source Corroboration Metadata
-- DO NOT APPLY TO LIVE DATABASE AUTOMATICALLY.

-- 1. Drop fast lookup indexes
drop index if exists public.briefing_evidence_lineage_idx;
drop index if exists public.briefing_events_corroboration_status_idx;
drop index if exists public.events_corroboration_status_idx;

-- 2. Drop columns from briefing_event_evidence
alter table if exists public.briefing_event_evidence
    drop column if exists lineage_type,
    drop column if exists original_source;

-- 3. Drop columns from briefing_events
alter table if exists public.briefing_events
    drop column if exists corroboration_status,
    drop column if exists independent_chains_count,
    drop column if exists corroboration_detail;

-- 4. Drop columns from events
alter table if exists public.events
    drop column if exists corroboration_status,
    drop column if exists independent_chains_count,
    drop column if exists corroboration_detail;

