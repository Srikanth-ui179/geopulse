-- ==============================================================================
-- STATUS: UNAPPLIED / DRAFT ONLY
-- DO NOT EXECUTE AUTOMATICALLY AGAINST PRODUCTION SUPABASE INSTANCES.
-- ==============================================================================
-- Rollback for Migration 010: Category Expansion & Cross-Day Deduplication
-- ==============================================================================

drop index if exists events_semantic_signature_idx;
drop index if exists events_persistent_id_idx;
drop index if exists briefing_events_semantic_sig_idx;

alter table if exists public.briefing_event_evidence
    drop column if exists origin_type;

alter table if exists public.briefing_events
    drop column if exists is_update,
    drop column if exists persistent_event_id,
    drop column if exists semantic_signature,
    drop column if exists evidence_label;

alter table if exists public.events
    drop column if exists is_update,
    drop column if exists persistent_event_id,
    drop column if exists semantic_signature,
    drop column if exists evidence_label;

