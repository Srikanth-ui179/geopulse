-- Geopulse Phase 6: Event Metadata & Delivery Status Constraint Alignment
-- DO NOT APPLY TO LIVE DATABASE AUTOMATICALLY. Review and approve before applying.
--
-- 1. Add published_at and is_analysis to public.events to align with Phase 4-5 clustering models
alter table if exists public.events
    add column if not exists published_at timestamptz,
    add column if not exists is_analysis boolean not null default false;

-- 2. Update status check constraint on public.briefings to permit 'ambiguous' delivery state
alter table if exists public.briefings
    drop constraint if exists briefings_status_check;

alter table if exists public.briefings
    add constraint briefings_status_check
        check (status in ('pending', 'processing', 'sent', 'partial', 'failed', 'ambiguous'));

-- 3. Composite index for querying events by analysis status and chronology
create index if not exists events_is_analysis_idx
    on public.events (is_analysis, first_seen_at desc);
