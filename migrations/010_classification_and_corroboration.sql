-- ==============================================================================
-- STATUS: UNAPPLIED / DRAFT ONLY
-- DO NOT EXECUTE AUTOMATICALLY AGAINST PRODUCTION SUPABASE INSTANCES.
-- ==============================================================================
-- Geopulse Quality Improvements: Category Expansion & Cross-Day Deduplication
--
-- This migration:
-- 1. Updates topic category check constraints to include:
--    'intelligence_espionage' and 'military_technology'.
-- 2. Adds cross-day persistent event identity, semantic signatures, and update flags.
-- 3. Adds transparent editorial evidence labels ('Single-source', 'Partially corroborated', 'Independently corroborated').
-- 4. Adds origin type tracking to briefing evidence records.
-- ==============================================================================

-- 1. Add persistent event identity, update status, and evidence labels to public.events
alter table if exists public.events
    add column if not exists is_update boolean not null default false,
    add column if not exists persistent_event_id text,
    add column if not exists semantic_signature text,
    add column if not exists evidence_label text default 'Single-source';

-- 2. Add same columns to public.briefing_events for delivery provenance
alter table if exists public.briefing_events
    add column if not exists is_update boolean not null default false,
    add column if not exists persistent_event_id text,
    add column if not exists semantic_signature text,
    add column if not exists evidence_label text default 'Single-source';

-- 3. Add origin type to public.briefing_event_evidence
alter table if exists public.briefing_event_evidence
    add column if not exists origin_type text default 'unknown';

-- 4. Update topic_category check constraint on public.articles if present
alter table if exists public.articles
    drop constraint if exists articles_topic_check;

alter table if exists public.articles
    add constraint articles_topic_check check (
        topic in (
            'armed_conflict',
            'military_operations',
            'defence_policy',
            'weapons_procurement',
            'diplomacy',
            'sanctions',
            'terrorism_insurgency',
            'cyber_hybrid_warfare',
            'security_incident',
            'intelligence_espionage',
            'military_technology',
            'other_geopolitical',
            'uncertain'
        )
    );

-- 5. Update category check constraint on public.events if present
alter table if exists public.events
    drop constraint if exists events_category_check;

alter table if exists public.events
    add constraint events_category_check check (
        category in (
            'armed_conflict',
            'military_operations',
            'defence_policy',
            'weapons_procurement',
            'diplomacy',
            'sanctions',
            'terrorism_insurgency',
            'cyber_hybrid_warfare',
            'security_incident',
            'intelligence_espionage',
            'military_technology',
            'other_geopolitical',
            'uncertain'
        )
    );

-- 6. Indexes for persistent identity and semantic signature lookup
create index if not exists events_semantic_signature_idx
    on public.events (semantic_signature);

create index if not exists events_persistent_id_idx
    on public.events (persistent_event_id);

create index if not exists briefing_events_semantic_sig_idx
    on public.briefing_events (semantic_signature);

